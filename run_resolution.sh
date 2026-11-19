#!/usr/bin/env bash
# Resolution tests (all fine-tuned 30k steps from runs/dp_recovery/bc_best.pt, 72.5%), then the robust teacher.
#   control  : same data, same architecture        -> effect of the extra training alone
#   zoom     : workspace camera focal 25 mm (1.4x larger cube), all 1,900 episodes re-collected with it
#   nomaxpool: ResNet stem max-pool removed -> 8x8 instead of 4x4 encoder grid, original data
set -euo pipefail
cd "$(dirname "$0")"
P=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}
T=${TEACHER:?set TEACHER}
R=runs; INIT=$R/dp_recovery/bc_best.pt
step() { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }
DATA="$R/demos_wrist_clean $R/demos_wrist_dart $R/demos_wrist_clean2 $R/demos_wrist_dart10 $R/demos_wrist_recovery"
step "re-collect the 1,900 episodes with the zoomed camera (same seeds/settings)"
[ -d $R/zoom_clean ]    || $P collect_demos.py --teacher $T --wrist --zoom --target 300 --seed 11 --output $R/zoom_clean > $R/zoom_clean.log 2>&1
[ -d $R/zoom_dart ]     || $P collect_demos.py --teacher $T --wrist --zoom --target 200 --noise-std 0.05 --seed 12 --output $R/zoom_dart > $R/zoom_dart.log 2>&1
[ -d $R/zoom_clean2 ]   || $P collect_demos.py --teacher $T --wrist --zoom --target 300 --seed 31 --output $R/zoom_clean2 > $R/zoom_clean2.log 2>&1
[ -d $R/zoom_dart10 ]   || $P collect_demos.py --teacher $T --wrist --zoom --target 500 --noise-std 0.10 --seed 32 --max-batches 60 --output $R/zoom_dart10 > $R/zoom_dart10.log 2>&1
[ -d $R/zoom_recovery ] || $P collect_demos.py --teacher $T --wrist --zoom --perturb --target 600 --num-envs 64 --max-batches 30 --seed 51 --output $R/zoom_recovery > $R/zoom_recovery.log 2>&1
ZDATA="$R/zoom_clean $R/zoom_dart $R/zoom_clean2 $R/zoom_dart10 $R/zoom_recovery"
step "train control";   [ -d $R/res_control ]   || $P train_dp.py --dataset $DATA  --init $INIT --steps 30000 --warmup-steps 200 --output $R/res_control --bf16 > $R/res_control.log 2>&1
step "train zoom";      [ -d $R/res_zoom ]      || $P train_dp.py --dataset $ZDATA --init $INIT --steps 30000 --warmup-steps 200 --output $R/res_zoom --bf16 > $R/res_zoom.log 2>&1
step "train nomaxpool"; [ -d $R/res_nomaxpool ] || $P train_dp.py --dataset $DATA  --init $INIT --steps 30000 --warmup-steps 200 --no-maxpool --accum 2 --output $R/res_nomaxpool --bf16 > $R/res_nomaxpool.log 2>&1
for v in control zoom nomaxpool; do
  for s in 201 301; do
    [ -f $R/eval_res_${v}_s$s.json ] || $P eval_policy.py --checkpoint $R/res_$v/bc_best.pt --output $R/eval_res_${v}_s$s.json --episodes 200 --seed $s > $R/eval_res_${v}_s$s.log 2>&1
  done
  Z=""; [ $v = zoom ] && Z="--zoom"
  [ -f $R/diag/diag_res_$v.json ] || $P diagnose.py --teacher $T --checkpoint $R/res_$v/bc_best.pt $Z --output $R/diag/diag_res_$v.json > $R/diag_res_$v.log 2>&1
  step "$v: $(python3 -c "import json;print([json.load(open('$R/eval_res_${v}_s'+s+'.json'))['success_rate'] for s in ('201','301')])")"
  grep "^\[diag\] dp" $R/diag_res_$v.log || true
done
step "robust teacher (2000 iterations)"
$P train_teacher_robust.py --iterations 2000 > $R/robust_teacher_train.log 2>&1
RT=$(ls -d logs/rsl_rl/franka_lift_robust/*/ | sort | tail -1)model_1999.pt
$P eval_policy.py --teacher $PWD/$RT --robust-teacher --output $R/eval_teacher_robust.json --episodes 200 > $R/eval_teacher_robust.log 2>&1
step "robust teacher $RT unperturbed: $(python3 -c "import json;print(json.load(open('$R/eval_teacher_robust.json'))['success_rate'])")"
$P diagnose.py --teacher $PWD/$RT --robust-teacher --output $R/diag/diag_teacher_robust.json > $R/diag_teacher_robust.log 2>&1
grep "^\[diag\]" $R/diag_teacher_robust.log
step "done"
