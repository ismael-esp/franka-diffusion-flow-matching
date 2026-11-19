#!/usr/bin/env bash
# Drawer opening: teacher check -> demos (clean + DART) -> Diffusion Policy from scratch -> eval on fresh seeds + videos.
# Every step logs to runs/drawer/<step>.log; finished steps are skipped on a re-run.
set -euo pipefail
cd "$(dirname "$0")"
P=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}
T=$(ls -d $PWD/logs/rsl_rl/franka_drawer/*/ | tail -1); T=$(ls $T/model_*.pt | sort -V | tail -1)
R=runs/drawer; mkdir -p $R
log() { echo "=== $(date +%m-%d\ %H:%M:%S) $*"; }
run() { local name=$1; shift; "$@" > $R/$name.log 2>&1 || { echo "FAILED $name"; tail -20 $R/$name.log; exit 1; }; grep -hE "^EVAL|\[collect\] \{|done:" $R/$name.log || true; }
log "teacher $T"
[ -f $R/eval_teacher.json ] || run eval_teacher $P eval_task.py --task drawer --teacher $T --episodes 200 --num-envs 50 --seed 201 --output $R/eval_teacher.json
log "collect demos"
[ -f $R/demos_clean/meta.json ]  || run demos_clean  $P collect_task.py --task drawer --teacher $T --output $R/demos_clean --target 400 --seed 71
[ -f $R/demos_dart10/meta.json ] || run demos_dart10 $P collect_task.py --task drawer --teacher $T --output $R/demos_dart10 --target 400 --noise-std 0.10 --seed 72
log "train DP (30k steps)"
[ -f $R/dp/bc_final.pt ] || run train_dp $P train_dp.py --dataset $R/demos_clean $R/demos_dart10 --output $R/dp --steps 30000 --bf16
log "eval DP on fresh seeds"
for s in 1001 1002 1003; do
  [ -f $R/test_dp_s$s.json ] || run test_dp_s$s $P eval_task.py --task drawer --checkpoint $R/dp/bc_best.pt --episodes 200 --num-envs 50 --seed $s --output $R/test_dp_s$s.json
done
log "videos"
[ -f $R/vid_teacher.json ] || run vid_teacher $P eval_task.py --task drawer --teacher $T --num-envs 6 --seed 31 --video-dir $R/videos_teacher --output $R/vid_teacher.json
[ -f $R/vid_dp.json ]      || run vid_dp $P eval_task.py --task drawer --checkpoint $R/dp/bc_best.pt --num-envs 8 --seed 32 --video-dir $R/videos_dp --output $R/vid_dp.json
log "done"
