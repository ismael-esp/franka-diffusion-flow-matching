#!/usr/bin/env bash
# Robust (distillable) PPO teacher: train, then the same perturbation suite as the original teacher (seed 401).
set -euo pipefail
cd "$(dirname "$0")"
P=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}
while pgrep -f "run_recovery.sh" > /dev/null; do sleep 60; done          # one GPU job at a time
echo "=== $(date '+%H:%M:%S') train robust teacher (2000 iterations)"
$P train_teacher_robust.py --iterations 2000 > runs/robust_teacher_train.log 2>&1
RT=$(ls -d logs/rsl_rl/franka_lift_robust/*/ | sort | tail -1)model_1999.pt
echo "=== $(date '+%H:%M:%S') teacher: $RT"
$P eval_policy.py --teacher $PWD/$RT --robust-teacher --output runs/eval_teacher_robust.json --episodes 200 > runs/eval_teacher_robust.log 2>&1
echo "unperturbed: $(python3 -c "import json;print(json.load(open('runs/eval_teacher_robust.json'))['success_rate'])")"
$P diagnose.py --teacher $PWD/$RT --robust-teacher --output runs/diag/diag_teacher_robust.json > runs/diag_teacher_robust.log 2>&1
grep "^\[diag\]" runs/diag_teacher_robust.log
echo "=== $(date '+%H:%M:%S') done"
