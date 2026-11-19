#!/usr/bin/env bash
# Essential final run: robust-teacher demos + 1 DAgger round -> fine-tune the zoom model -> untouched test set + recovery
# test with held-out perturbations. Test seeds 1001-1005 are used here for the first and only time.
set -euo pipefail
cd "$(dirname "$0")"
P=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}
RT=${RT:?set RT to the robust-teacher checkpoint, e.g. $PWD/logs/rsl_rl/franka_lift_robust/<run>/model_1999.pt}
R=runs; BASE=$R/res_zoom/bc_best.pt
step() { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }
C="--teacher $RT --robust-teacher --zoom --wrist"
step "collect robust-teacher data (zoom + wrist cameras)"
[ -d $R/rob_clean ]    || $P collect_demos.py $C --target 300 --seed 61 --output $R/rob_clean > $R/rob_clean.log 2>&1
[ -d $R/rob_dart10 ]   || $P collect_demos.py $C --target 400 --noise-std 0.10 --seed 62 --max-batches 60 --output $R/rob_dart10 > $R/rob_dart10.log 2>&1
[ -d $R/rob_recovery ] || $P collect_demos.py $C --perturb --target 600 --num-envs 64 --max-batches 30 --seed 63 --output $R/rob_recovery > $R/rob_recovery.log 2>&1
[ -d $R/rob_dagger ]   || $P collect_demos.py $C --student $BASE --beta 0.0 --target 300 --seed 64 --output $R/rob_dagger > $R/rob_dagger.log 2>&1
step "fine-tune the zoom model (78.3 %) on robust-teacher demos + DAgger (30k steps)"
[ -d $R/res_robust ] || $P train_dp.py --dataset $R/rob_clean $R/rob_dart10 $R/rob_recovery $R/rob_dagger --init $BASE --steps 30000 --warmup-steps 200 --output $R/res_robust --bf16 > $R/res_robust.log 2>&1
step "untouched test set: seeds 1001-1005 x 200 episodes"
for m in res_zoom res_robust; do
  for s in 1001 1002 1003 1004 1005; do
    [ -f $R/test_${m}_s$s.json ] || $P eval_policy.py --checkpoint $R/$m/bc_best.pt --output $R/test_${m}_s$s.json --episodes 200 --seed $s > $R/test_${m}_s$s.log 2>&1
  done
  python3 -c "
import json, math
d=[json.load(open('$R/test_${m}_s%d.json'%s)) for s in range(1001,1006)]; k=sum(x['successes'] for x in d); n=sum(x['episodes'] for x in d); p=k/n
print('TEST $m: %d/%d = %.1f%% +- %.1f (SE)  per seed:' % (k, n, 100*p, 100*math.sqrt(p*(1-p)/n)), [x['success_rate'] for x in d])"
done
step "recovery test: training perturbations + held-out ones (push_held, arm_shove, late_knock)"
[ -f $R/diag/diag_final.json ] || $P diagnose.py --teacher $RT --robust-teacher --zoom --checkpoint $BASE --checkpoint $R/res_robust/bc_best.pt \
    --kinds none,empty_close,drop,knock,push_held,arm_shove,late_knock --output $R/diag/diag_final.json > $R/diag_final.log 2>&1
grep "^\[diag\]" $R/diag_final.log
step "done"
