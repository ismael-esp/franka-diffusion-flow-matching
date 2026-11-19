#!/usr/bin/env bash
# DAgger on top of the wrist-camera Diffusion Policy: the student drives, the privileged PPO teacher labels every
# visited state, the student is fine-tuned on demos + all DAgger data, then evaluated (held-out seeds 201/301).
set -euo pipefail
cd "$(dirname "$0")"
P=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}
TEACHER=${TEACHER:?set TEACHER to the rsl_rl checkpoint}
ROUNDS=${ROUNDS:-2}; STEPS=${STEPS:-15000}
R=runs
step() { echo "=== $(date +%H:%M:%S) $*"; }
prev=$R/dp_wrist/bc_best.pt
data="$R/demos_wrist_clean $R/demos_wrist_dart"
for k in $(seq 1 $ROUNDS); do
  step "DAgger round $k: student rollouts labelled by the teacher (300 episodes)"
  [ -d $R/dagger$k ] || $P collect_demos.py --teacher "$TEACHER" --wrist --student $prev --beta 0.0 --target 300 --seed $((20 + k)) \
      --output $R/dagger$k > $R/dagger$k.log 2>&1
  data="$data $R/dagger$k"
  step "DAgger round $k: fine-tune on $data"
  [ -d $R/dp_dagger$k ] || $P train_dp.py --dataset $data --init $prev --steps $STEPS --warmup-steps 200 --output $R/dp_dagger$k --bf16 > $R/dp_dagger$k.log 2>&1
  prev=$R/dp_dagger$k/bc_best.pt
  for s in 201 301; do
    [ -f $R/eval_dp_dagger${k}_s$s.json ] || $P eval_policy.py --checkpoint $prev --output $R/eval_dp_dagger${k}_s$s.json --episodes 200 --seed $s > $R/eval_dp_dagger${k}_s$s.log 2>&1
  done
  step "DAgger round $k: $(python3 -c "import json;print([json.load(open('$R/eval_dp_dagger${k}_s'+s+'.json'))['success_rate'] for s in ('201','301')])")"
done
[ -f $R/video_dp_dagger.json ] || $P eval_policy.py --checkpoint $prev --output $R/video_dp_dagger.json --episodes 8 --num-envs 8 --video 8 --seed 7 > $R/video_dp_dagger.log 2>&1
step "done"
