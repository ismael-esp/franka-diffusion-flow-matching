#!/usr/bin/env bash
# Minimal path from a clean clone to the headline results (stack, drawer, lift: diffusion vs flow matching, stack DAgger).
# Every step skips work whose output exists, so it can be interrupted and re-run. Syntax-checked with `bash -n`; NOT executed end to end.
# Needs: the isaaclab python, a free GPU (one job at a time), about 150 GB of disk, and for lift a trained PPO teacher:
#   TEACHER=/path/to/logs/rsl_rl/franka_lift/<run>/model_1499.pt bash scripts/run_repro_minimal.sh
# Skipped on purpose: the pour part (needs the external TRLDistill_v5 environment; run_flow.sh tries it last and reports FAILED if unavailable),
# the older lift experiments (run_pipeline*.sh, run_dagger.sh, run_recovery.sh, run_resolution.sh train chains, run_final.sh, run_v3.sh).
# Not used for the reported numbers: run_night.sh (its stack half trains on clean + DART 0.03 and would take over runs/stack/dp), run_stack.sh
# (DART 0.10 variant), run_drawer.sh (PPO-teacher drawer, the 8.7 % negative result; needs logs/rsl_rl/franka_drawer).
set -euo pipefail
cd "$(dirname "$0")/.."
P=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}
TEACHER=${TEACHER:?set TEACHER to the lift PPO checkpoint (model_1499.pt), see README section 3}
S=runs/stack; D=runs/drawer_s; Z=runs
mkdir -p $S $D
log() { echo "=== $(date +%m-%d\ %H:%M:%S) $*"; }
run() { local out=$1; shift; "$@" > "$out" 2>&1 || { echo "FAILED $out"; tail -15 "$out"; exit 1; }; }

log "1. scripted-teacher demos: stack (clean, DART 0.01) and drawer (clean, DART 0.03)"
[ -f $S/demos_clean/meta.json ]  || run $S/demos_clean.log  $P collect_task.py --task stack  --teacher scripted --output $S/demos_clean  --target 400 --num-envs 24 --seed 71
[ -f $S/demos_dart01/meta.json ] || run $S/demos_dart01.log $P collect_task.py --task stack  --teacher scripted --output $S/demos_dart01 --target 400 --num-envs 24 --noise-std 0.01 --seed 72
[ -f $D/demos_clean/meta.json ]  || run $D/demos_clean.log  $P collect_task.py --task drawer --teacher scripted --output $D/demos_clean  --target 400 --num-envs 32 --seed 71
[ -f $D/demos_dart03/meta.json ] || run $D/demos_dart03.log $P collect_task.py --task drawer --teacher scripted --output $D/demos_dart03 --target 400 --num-envs 32 --noise-std 0.03 --seed 72

log "2. diffusion baselines for stack and drawer (30k steps) and their held-out evaluation"
[ -f $S/dp/bc_final.pt ] || run $S/train_dp.log $P train_dp.py --dataset $S/demos_clean $S/demos_dart01 --output $S/dp --steps 30000 --bf16
[ -f $D/dp/bc_final.pt ] || run $D/train_dp.log $P train_dp.py --dataset $D/demos_clean $D/demos_dart03 --output $D/dp --steps 30000 --bf16
for s in 1001 1002 1003; do
  [ -f $S/test_dp_s$s.json ] || run $S/test_dp_s$s.log $P eval_task.py --task stack  --checkpoint $S/dp/bc_best.pt --episodes 200 --num-envs 50 --seed $s --output $S/test_dp_s$s.json
  [ -f $D/test_dp_s$s.json ] || run $D/test_dp_s$s.log $P eval_task.py --task drawer --checkpoint $D/dp/bc_best.pt --episodes 200 --num-envs 50 --seed $s --output $D/test_dp_s$s.json
done

log "3. stack DAgger rounds 1-3 (needs $S/dp, demos_clean, demos_dart01)"
bash run_stack_dagger.sh

log "4. lift: the five zoom demo sets (1,900 episodes) from the PPO teacher"
T=$TEACHER
[ -d $Z/zoom_clean ]    || run $Z/zoom_clean.log    $P collect_demos.py --teacher $T --wrist --zoom --target 300 --seed 11 --output $Z/zoom_clean
[ -d $Z/zoom_dart ]     || run $Z/zoom_dart.log     $P collect_demos.py --teacher $T --wrist --zoom --target 200 --noise-std 0.05 --seed 12 --output $Z/zoom_dart
[ -d $Z/zoom_clean2 ]   || run $Z/zoom_clean2.log   $P collect_demos.py --teacher $T --wrist --zoom --target 300 --seed 31 --output $Z/zoom_clean2
[ -d $Z/zoom_dart10 ]   || run $Z/zoom_dart10.log   $P collect_demos.py --teacher $T --wrist --zoom --target 500 --noise-std 0.10 --seed 32 --max-batches 60 --output $Z/zoom_dart10
[ -d $Z/zoom_recovery ] || run $Z/zoom_recovery.log $P collect_demos.py --teacher $T --wrist --zoom --perturb --target 600 --num-envs 64 --max-batches 30 --seed 51 --output $Z/zoom_recovery

log "5. flow matching vs diffusion: stack, drawer, lift (both objectives from scratch on the zoom demos), sampling-step sweep, pour if available"
bash run_flow.sh

log "6. stack: clean held-out re-evaluation, from-scratch retraining on all DAgger data, round 4"
bash run_improve.sh
log "done"
