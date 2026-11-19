#!/usr/bin/env bash
# Franka cube-lift: privileged PPO teacher -> camera demos -> Diffusion Policy (pouring_diffusion dp_core) -> frozen eval.
# Usage: TEACHER=logs/rsl_rl/franka_lift/<run>/model_1499.pt bash run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"
P=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}
TEACHER=${TEACHER:?set TEACHER to the rsl_rl checkpoint}
R=runs
step() { echo "=== $(date +%H:%M:%S) $*"; }

step "teacher eval (privileged state, 200 episodes)"
[ -f $R/eval_teacher.json ] || $P eval_policy.py --teacher "$TEACHER" --output $R/eval_teacher.json --episodes 200 > $R/eval_teacher.log 2>&1
[ -f $R/video_teacher.json ] || $P eval_policy.py --teacher "$TEACHER" --output $R/video_teacher.json --episodes 8 --num-envs 8 --video 8 --seed 7 > $R/video_teacher.log 2>&1

step "collect 300 clean + 200 DART-noise teacher demos with the workspace camera"
[ -d $R/demos_clean ] || $P collect_demos.py --teacher "$TEACHER" --output $R/demos_clean --target 300 --seed 11 > $R/demos_clean.log 2>&1
[ -d $R/demos_dart ]  || $P collect_demos.py --teacher "$TEACHER" --output $R/demos_dart --target 200 --noise-std 0.05 --seed 12 > $R/demos_dart.log 2>&1

step "train the Diffusion Policy (RGB + proprio, no privileged state)"
[ -d $R/dp ] || $P train_dp.py --dataset $R/demos_clean $R/demos_dart --output $R/dp --bf16 > $R/dp.log 2>&1

step "frozen eval of the Diffusion Policy on held-out seeds (200 episodes each)"
for s in 201 301; do
  [ -f $R/eval_dp_s$s.json ] || $P eval_policy.py --checkpoint $R/dp/bc_best.pt --output $R/eval_dp_s$s.json --episodes 200 --seed $s > $R/eval_dp_s$s.log 2>&1
done
[ -f $R/video_dp.json ] || $P eval_policy.py --checkpoint $R/dp/bc_best.pt --output $R/video_dp.json --episodes 8 --num-envs 8 --video 8 --seed 7 > $R/video_dp.log 2>&1
step "done"
