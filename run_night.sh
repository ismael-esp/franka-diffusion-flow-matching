#!/usr/bin/env bash
# Overnight: scripted teachers for both tasks -> demos (clean + small DART noise) -> DP -> fresh-seed test + videos.
#   stack:    runs/stack   (scripted IK teacher; clean demos already collected)
#   drawer_s: runs/drawer_s (scripted IK drawer teacher; the PPO-teacher attempt is runs/drawer)
# Skips finished steps on a re-run.
set -uo pipefail
cd "$(dirname "$0")"
P=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}
log() { echo "=== $(date +%m-%d\ %H:%M:%S) $*"; }
run() { local R=$1 name=$2; shift 2; "$@" > $R/$name.log 2>&1 || { echo "FAILED $R/$name"; tail -20 $R/$name.log; return 1; }; grep -hE "^EVAL|\[collect\] \{|done:" $R/$name.log || true; }
S=runs/stack; D=runs/drawer_s; mkdir -p $S $D
log "collect (parallel): stack DART 0.03 | drawer scripted clean + DART 0.03"
( [ -f $S/demos_dart03/meta.json ] || run $S demos_dart03 $P collect_task.py --task stack --teacher scripted --output $S/demos_dart03 --target 400 --num-envs 24 --noise-std 0.03 --seed 72 ) &
( [ -f $D/demos_clean/meta.json ]  || run $D demos_clean  $P collect_task.py --task drawer --teacher scripted --output $D/demos_clean --target 400 --num-envs 32 --seed 71
  [ -f $D/demos_dart03/meta.json ] || run $D demos_dart03 $P collect_task.py --task drawer --teacher scripted --output $D/demos_dart03 --target 400 --num-envs 32 --noise-std 0.03 --seed 72 ) &
wait
pipeline() {  # task dir datasets...
  local task=$1 R=$2; shift 2
  [ -f $R/eval_teacher.json ] || run $R eval_teacher $P eval_task.py --task $task --teacher scripted --episodes 200 --num-envs 50 --seed 201 --output $R/eval_teacher.json
  log "$task: train DP"
  [ -f $R/dp/bc_final.pt ] || run $R train_dp $P train_dp.py --dataset "$@" --output $R/dp --steps 30000 --bf16 || return 1
  log "$task: test on fresh seeds"
  for s in 1001 1002 1003; do
    [ -f $R/test_dp_s$s.json ] || run $R test_dp_s$s $P eval_task.py --task $task --checkpoint $R/dp/bc_best.pt --episodes 200 --num-envs 50 --seed $s --output $R/test_dp_s$s.json
  done
  log "$task: videos"
  [ -f $R/vid_teacher.json ] || run $R vid_teacher $P eval_task.py --task $task --teacher scripted --num-envs 6 --seed 31 --video-dir $R/videos_teacher --output $R/vid_teacher.json
  [ -f $R/vid_dp.json ]      || run $R vid_dp $P eval_task.py --task $task --checkpoint $R/dp/bc_best.pt --num-envs 8 --seed 32 --video-dir $R/videos_dp --output $R/vid_dp.json
}
pipeline drawer $D $D/demos_clean $D/demos_dart03
pipeline stack  $S $S/demos_clean $S/demos_dart03
log "done"
