#!/usr/bin/env bash
# Fallback after DAgger hurt (the PPO teacher labels off-distribution states poorly): more teacher-driven data,
# including DART noise (the teacher recovers from its own perturbed states, so labels stay clean), and longer training.
set -euo pipefail
cd "$(dirname "$0")"
P=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}
TEACHER=${TEACHER:?set TEACHER to the rsl_rl checkpoint}
R=runs
step() { echo "=== $(date +%H:%M:%S) $*"; }
step "collect 300 more clean + 500 DART(0.10) demos, workspace + wrist camera"
[ -d $R/demos_wrist_clean2 ] || $P collect_demos.py --teacher "$TEACHER" --wrist --output $R/demos_wrist_clean2 --target 300 --seed 31 > $R/demos_wrist_clean2.log 2>&1
[ -d $R/demos_wrist_dart10 ] || $P collect_demos.py --teacher "$TEACHER" --wrist --output $R/demos_wrist_dart10 --target 500 --noise-std 0.10 --seed 32 --max-batches 60 > $R/demos_wrist_dart10.log 2>&1
step "continue training the 56% model on all 1300 teacher-driven episodes (30k steps)"
[ -d $R/dp_wrist_more ] || $P train_dp.py --dataset $R/demos_wrist_clean $R/demos_wrist_dart $R/demos_wrist_clean2 $R/demos_wrist_dart10 \
    --init $R/dp_wrist/bc_best.pt --steps 30000 --warmup-steps 200 --output $R/dp_wrist_more --bf16 > $R/dp_wrist_more.log 2>&1
for s in 201 301; do
  [ -f $R/eval_dp_wrist_more_s$s.json ] || $P eval_policy.py --checkpoint $R/dp_wrist_more/bc_best.pt --output $R/eval_dp_wrist_more_s$s.json --episodes 200 --seed $s > $R/eval_dp_wrist_more_s$s.log 2>&1
done
step "result: $(python3 -c "import json;print([json.load(open('$R/eval_dp_wrist_more_s'+s+'.json'))['success_rate'] for s in ('201','301')])")"
[ -f $R/video_dp_wrist_more.json ] || $P eval_policy.py --checkpoint $R/dp_wrist_more/bc_best.pt --output $R/video_dp_wrist_more.json --episodes 8 --num-envs 8 --video 8 --seed 7 > $R/video_dp_wrist_more.log 2>&1
step "done"
