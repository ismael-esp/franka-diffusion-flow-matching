#!/usr/bin/env bash
# Stacking improvement, diffusion AND flow matching, after run_flow.sh has finished (idle GPU -> no contention).
#  A. clean re-evaluation of the existing stack models on held-out seeds 1002-1005 (seed 1001 = model selection only)
#  B. from scratch (no --init chain) on all data so far (demos + dagger1-3): diffusion and flow matching
#  C. DAgger round 4 driven by the better of the two (picked on seed 1001), then both retrained from scratch
# Steps are skipped when their output exists, so a re-run resumes.
set -uo pipefail
cd "$(dirname "$0")"
P=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}
R=runs/stack; I=runs/improve; mkdir -p $I
log() { echo "=== $(date +%m-%d\ %H:%M:%S) $*"; }
run() { local name=$1; shift; "$@" > $I/$name.log 2>&1 || { echo "FAILED $name"; tail -15 $I/$name.log; return 1; }; grep -hE "^EVAL|\[collect\] \{|done:" $I/$name.log | tail -2 || true; }
ev() { local m=$1 ck=$2 s=$3; local o=$I/eval_${m}_s$s.json
  [ -f $o ] || run eval_${m}_s$s $P eval_task.py --task stack --checkpoint $ck --episodes 200 --num-envs 50 --seed $s --output $o; }
ev4() { for s in 1002 1003 1004 1005; do ev $1 $2 $s; done; }
succ() { $P -c "import json,sys;print(json.load(open(sys.argv[1]))['success'])" $1; }

log "waiting for run_flow.sh"
while pgrep -f "bash run_flow.sh" > /dev/null; do sleep 60; done

log "A: clean held-out evaluation (idle GPU)"
ev4 dp_demos $R/dp/bc_best.pt
ev4 fm_demos runs/flow/fm_stack/bc_best.pt
ev4 dagger2 $R/dp_dagger2/bc_best.pt
ev4 dagger3 $R/dp_dagger3/bc_best.pt

log "B: from scratch on all data (demos + dagger1-3)"
D="$R/demos_clean $R/demos_dart01 $R/dagger1 $R/dagger2 $R/dagger3"
for obj in diffusion flow; do
  m=all_$obj
  [ -f $I/$m/bc_final.pt ] || { rm -rf $I/$m; run train_$m $P train_dp.py --objective $obj --dataset $D --steps 30000 --bf16 --output $I/$m; }
  ev $m $I/$m/bc_best.pt 1001
  ev4 $m $I/$m/bc_best.pt
done

log "C: DAgger round 4 with the better student (selected on seed 1001)"
best=all_diffusion
$P -c "import sys;sys.exit(0 if float(sys.argv[1])>float(sys.argv[2]) else 1)" $(succ $I/eval_all_flow_s1001.json) $(succ $I/eval_all_diffusion_s1001.json) && best=all_flow
echo "round-4 student: $best" | tee $I/round4_student.txt
[ -f $R/dagger4/meta.json ] || { rm -rf $R/dagger4; run collect_dagger4 $P collect_task.py --task stack --teacher scripted --student $I/$best/bc_best.pt \
    --beta 0.0 --output $R/dagger4 --target 400 --num-envs 24 --seed 84; }
D4="$D $R/dagger4"
for obj in diffusion flow; do
  m=all4_$obj
  [ -f $I/$m/bc_final.pt ] || { rm -rf $I/$m; run train_$m $P train_dp.py --objective $obj --dataset $D4 --steps 30000 --bf16 --output $I/$m; }
  ev $m $I/$m/bc_best.pt 1001
  ev4 $m $I/$m/bc_best.pt
done
log "done"
