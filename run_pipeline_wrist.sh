#!/usr/bin/env bash
# Same pipeline as run_pipeline.sh with the eye-in-hand camera added (student image = [workspace | wrist], 128x256).
set -euo pipefail
cd "$(dirname "$0")"
P=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}
TEACHER=${TEACHER:?set TEACHER to the rsl_rl checkpoint}
R=runs
step() { echo "=== $(date +%H:%M:%S) $*"; }
step "collect 300 clean + 200 DART demos (workspace + wrist camera)"
[ -d $R/demos_wrist_clean ] || $P collect_demos.py --teacher "$TEACHER" --wrist --output $R/demos_wrist_clean --target 300 --seed 11 > $R/demos_wrist_clean.log 2>&1
[ -d $R/demos_wrist_dart ]  || $P collect_demos.py --teacher "$TEACHER" --wrist --output $R/demos_wrist_dart --target 200 --noise-std 0.05 --seed 12 > $R/demos_wrist_dart.log 2>&1
step "train the Diffusion Policy"
[ -d $R/dp_wrist ] || $P train_dp.py --dataset $R/demos_wrist_clean $R/demos_wrist_dart --output $R/dp_wrist --bf16 > $R/dp_wrist.log 2>&1
step "frozen eval, held-out seeds 201 / 301, 200 episodes each"
for s in 201 301; do
  [ -f $R/eval_dp_wrist_s$s.json ] || $P eval_policy.py --checkpoint $R/dp_wrist/bc_best.pt --output $R/eval_dp_wrist_s$s.json --episodes 200 --seed $s > $R/eval_dp_wrist_s$s.log 2>&1
done
[ -f $R/video_dp_wrist.json ] || $P eval_policy.py --checkpoint $R/dp_wrist/bc_best.pt --output $R/video_dp_wrist.json --episodes 8 --num-envs 8 --video 8 --seed 7 > $R/video_dp_wrist.log 2>&1
step "done"
