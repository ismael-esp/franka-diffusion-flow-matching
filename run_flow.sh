#!/usr/bin/env bash
# Flow matching vs diffusion (DDIM) on the 4 tasks, same data / recipe / test seeds as each diffusion baseline.
# Waits for run_stack_dagger.sh. Each step is skipped when its output exists, so a re-run resumes.
#   stack, drawer : FM from scratch on the baseline's demos (30k steps) vs runs/stack/dp, runs/drawer_s/dp
#   lift          : diffusion AND FM from scratch on the zoom demos (30k) (the lift baselines were fine-tune chains)
#   pour          : FM from scratch on demos_lag400 + demos_lag800b (80k) vs pouring_diffusion/runs/bc_lag1200
# Eval: 10 flow steps on the baseline's test seeds; 1 and 5 steps on one seed (speed / quality trade-off).
set -uo pipefail
cd "$(dirname "$0")"
P=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}
R=runs/flow; mkdir -p $R
POUR=${POURING_DIFFUSION:-$PWD/pouring_diffusion}   # needs runs/demos_lag400, runs/demos_lag800b (TRLDistill_v5 pour env, see README)
log() { echo "=== $(date +%m-%d\ %H:%M:%S) $*"; }
run() { local name=$1; shift; "$@" > $R/$name.log 2>&1 || { echo "FAILED $name"; tail -15 $R/$name.log; return 1; }; grep -hE "^EVAL|done:|success_rate" $R/$name.log | tail -2 || true; }
train() { local out=$1; shift; [ -f $R/$out/bc_final.pt ] || { rm -rf $R/$out; run train_$out $P train_dp.py --output $R/$out --steps 30000 --bf16 "$@"; }; }
task_eval() { local task=$1 m=$2 s=$3 k=$4; local o=$R/test_${m}_k${k}_s$s.json
  [ -f $o ] || run test_${m}_k${k}_s$s $P eval_task.py --task $task --checkpoint $R/$m/bc_best.pt --episodes 200 --num-envs 50 --seed $s --inference-steps $k --output $o; }
lift_eval() { local m=$1 s=$2 k=$3; local o=$R/test_${m}_k${k}_s$s.json
  [ -f $o ] || run test_${m}_k${k}_s$s $P eval_policy.py --checkpoint $R/$m/bc_best.pt --episodes 200 --seed $s --ddim-steps $k --output $o; }

log "waiting for run_stack_dagger.sh"
while pgrep -f "bash run_stack_dagger.sh" > /dev/null; do sleep 60; done

log "stack: flow matching"
train fm_stack --objective flow --dataset runs/stack/demos_clean runs/stack/demos_dart01
for s in 1001 1002 1003; do task_eval stack fm_stack $s 10; done
for k in 1 5; do task_eval stack fm_stack 1001 $k; done

log "drawer: flow matching"
train fm_drawer --objective flow --dataset runs/drawer_s/demos_clean runs/drawer_s/demos_dart03
for s in 1001 1002 1003; do task_eval drawer fm_drawer $s 10; done
for k in 1 5; do task_eval drawer fm_drawer 1001 $k; done

log "lift: diffusion and flow matching from scratch on the zoom demos"
Z="runs/zoom_clean runs/zoom_dart runs/zoom_clean2 runs/zoom_dart10 runs/zoom_recovery"
train dp_lift --objective diffusion --dataset $Z
train fm_lift --objective flow --dataset $Z
for m in dp_lift fm_lift; do for s in 1001 1002 1003; do lift_eval $m $s 10; done; done
for k in 1 5; do lift_eval fm_lift 1001 $k; lift_eval dp_lift 1001 $k; done

log "pour: flow matching (bc_lag1200 recipe)"
PR=$POUR/runs/fm_lag1200
[ -f $PR/bc_final.pt ] || { rm -rf $PR; (cd $POUR && $P train_fm.py --dataset runs/demos_lag400 runs/demos_lag800b --steps 80000 --bf16 --output runs/fm_lag1200) > $R/train_fm_pour.log 2>&1 || { echo "FAILED train_fm_pour"; tail -15 $R/train_fm_pour.log; }; }
for k in 5 10 50; do
  o=$POUR/runs/sweep/fm_lag1200_k$k.json
  [ -f $o ] || (cd $POUR && $P -B eval_fm.py --snap-grip --num-envs 32 --episodes 200 --headless --seed 201 --checkpoint runs/fm_lag1200/bc_best.pt --ddim-steps $k --output $o) > $R/test_fm_pour_k$k.log 2>&1
  echo "pour fm k=$k $(grep -o '"success_rate": [0-9.]*' $o 2>/dev/null)"
done
log "done"
