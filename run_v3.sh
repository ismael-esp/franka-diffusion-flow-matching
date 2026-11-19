#!/usr/bin/env bash
# v3 recipe from the literature review (see train_dp_v3.py / fd_core/model_v3.py):
# fresh teacher-driven demos with cube positions -> 150k-step training with checkpoints every 25k ->
# checkpoint chosen by simulator success on validation seed 101 -> frozen eval on held-out seeds 201/301 + video.
set -euo pipefail
cd "$(dirname "$0")"
P=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}
TEACHER=${TEACHER:?set TEACHER to the rsl_rl checkpoint}
STEPS=${STEPS:-150000}
R=runs
step() { echo "=== $(date '+%m-%d %H:%M:%S') $*"; }
while pgrep -f run_more_data.sh > /dev/null; do sleep 60; done   # one GPU job at a time
step "collect 400 clean + 400 DART(0.05) + 400 DART(0.10) demos with wrist camera and cube positions"
[ -d $R/v3_clean ]  || $P collect_demos.py --teacher "$TEACHER" --wrist --output $R/v3_clean  --target 400 --seed 41 > $R/v3_clean.log 2>&1
[ -d $R/v3_dart05 ] || $P collect_demos.py --teacher "$TEACHER" --wrist --output $R/v3_dart05 --target 400 --noise-std 0.05 --seed 42 > $R/v3_dart05.log 2>&1
[ -d $R/v3_dart10 ] || $P collect_demos.py --teacher "$TEACHER" --wrist --output $R/v3_dart10 --target 400 --noise-std 0.10 --seed 43 --max-batches 60 > $R/v3_dart10.log 2>&1
step "train v3 for $STEPS steps"
[ -d $R/dp_v3 ] || $P train_dp_v3.py --dataset $R/v3_clean $R/v3_dart05 $R/v3_dart10 --output $R/dp_v3 --steps $STEPS --bf16 > $R/dp_v3.log 2>&1
step "checkpoint selection by success on validation seed 101 (100 episodes each)"
for ck in $(ls $R/dp_v3/ckpt_*.pt | sort -V); do
  n=$(basename $ck .pt)
  [ -f $R/dp_v3/sel_$n.json ] || $P eval_policy.py --checkpoint $ck --output $R/dp_v3/sel_$n.json --episodes 100 --seed 101 > $R/dp_v3/sel_$n.log 2>&1
done
BEST=$(python3 -c "
import json,glob
r=sorted((json.load(open(f))['success_rate'],int(f.split('_')[-1][:-5]),f) for f in glob.glob('$R/dp_v3/sel_ckpt_*.json'))
print(r[-1][2].replace('sel_','').replace('.json','.pt'))")
step "selected $BEST"
for s in 201 301; do
  [ -f $R/eval_dp_v3_s$s.json ] || $P eval_policy.py --checkpoint $BEST --output $R/eval_dp_v3_s$s.json --episodes 200 --seed $s > $R/eval_dp_v3_s$s.log 2>&1
done
step "result: $(python3 -c "import json;print([json.load(open('$R/eval_dp_v3_s'+s+'.json'))['success_rate'] for s in ('201','301')])")"
[ -f $R/video_dp_v3.json ] || $P eval_policy.py --checkpoint $BEST --output $R/video_dp_v3.json --episodes 8 --num-envs 8 --video 8 --seed 7 > $R/video_dp_v3.log 2>&1
step "done"
