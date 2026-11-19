#!/usr/bin/env bash
# Recovery demos: teacher rollouts with injected failure states (empty close / knock / drop), successful recoveries kept,
# then fine-tune the best Diffusion Policy on all teacher data + recovery demos; frozen eval + perturbation suite.
set -euo pipefail
cd "$(dirname "$0")"
P=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}
TEACHER=${TEACHER:?set TEACHER}
R=runs
step() { echo "=== $(date '+%H:%M:%S') $*"; }
step "collect 600 recovery demos"
[ -d $R/demos_wrist_recovery ] || $P collect_demos.py --teacher "$TEACHER" --wrist --perturb --target 600 --num-envs 64 --max-batches 30 --seed 51 --output $R/demos_wrist_recovery > $R/demos_wrist_recovery.log 2>&1
step "fine-tune the 59.8% model on 1,300 teacher demos + 600 recovery demos (30k steps)"
[ -d $R/dp_recovery ] || $P train_dp.py --dataset $R/demos_wrist_clean $R/demos_wrist_dart $R/demos_wrist_clean2 $R/demos_wrist_dart10 $R/demos_wrist_recovery \
    --init $R/dp_wrist_more/bc_best.pt --steps 30000 --warmup-steps 200 --output $R/dp_recovery --bf16 > $R/dp_recovery.log 2>&1
for s in 201 301; do
  [ -f $R/eval_dp_recovery_s$s.json ] || $P eval_policy.py --checkpoint $R/dp_recovery/bc_best.pt --output $R/eval_dp_recovery_s$s.json --episodes 200 --seed $s > $R/eval_dp_recovery_s$s.log 2>&1
done
step "result: $(python3 -c "import json;print([json.load(open('$R/eval_dp_recovery_s'+s+'.json'))['success_rate'] for s in ('201','301')])")"
step "perturbation suite (same seed as the diagnosis)"
[ -f $R/diag/diag_recovery.json ] || $P diagnose.py --teacher "$TEACHER" --checkpoint $R/dp_recovery/bc_best.pt --output $R/diag/diag_recovery.json > $R/diag_recovery.log 2>&1
grep "^\[diag\] dp" $R/diag_recovery.log || true
[ -f $R/video_dp_recovery.json ] || $P eval_policy.py --checkpoint $R/dp_recovery/bc_best.pt --output $R/video_dp_recovery.json --episodes 8 --num-envs 8 --video 8 --seed 7 > $R/video_dp_recovery.log 2>&1
step "done"
