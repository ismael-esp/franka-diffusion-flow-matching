#!/usr/bin/env bash
# Stack DP after run_night.sh's drawer part: waits for a running night pipeline, collects the clean + DART 0.01 demos if they are missing, then
# trains on clean + DART 0.01 demos, tests on fresh seeds, records videos.
set -uo pipefail
cd "$(dirname "$0")"
P=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}
R=runs/stack; mkdir -p $R
log() { echo "=== $(date +%m-%d\ %H:%M:%S) $*"; }
run() { local name=$1; shift; "$@" > $R/$name.log 2>&1 || { echo "FAILED $name"; tail -20 $R/$name.log; return 1; }; grep -hE "^EVAL|\[collect\] \{|done:" $R/$name.log || true; }
log "waiting for run_night.sh and the DART collection"
while pgrep -f "bash run_night.sh" > /dev/null || pgrep -f "demos_dart01" > /dev/null; do sleep 60; done
[ -f $R/demos_clean/meta.json ] || run demos_clean $P collect_task.py --task stack --teacher scripted --output $R/demos_clean --target 400 --num-envs 24 --seed 71 || exit 1
[ -f $R/demos_dart01/meta.json ] || run demos_dart01 $P collect_task.py --task stack --teacher scripted --output $R/demos_dart01 --target 400 --num-envs 24 --noise-std 0.01 --seed 72 || exit 1
grep -h "\[collect\] {" $R/demos_dart01.log
log "train DP"
[ -f $R/dp/bc_final.pt ] || run train_dp $P train_dp.py --dataset $R/demos_clean $R/demos_dart01 --output $R/dp --steps 30000 --bf16 || exit 1
log "test on fresh seeds"
for s in 1001 1002 1003; do
  [ -f $R/test_dp_s$s.json ] || run test_dp_s$s $P eval_task.py --task stack --checkpoint $R/dp/bc_best.pt --episodes 200 --num-envs 50 --seed $s --output $R/test_dp_s$s.json
done
log "videos"
[ -f $R/vid_teacher.json ] || run vid_teacher $P eval_task.py --task stack --teacher scripted --num-envs 6 --seed 31 --video-dir $R/videos_teacher --output $R/vid_teacher.json
[ -f $R/vid_dp.json ]      || run vid_dp $P eval_task.py --task stack --checkpoint $R/dp/bc_best.pt --num-envs 8 --seed 32 --video-dir $R/videos_dp --output $R/vid_dp.json
log "done"
