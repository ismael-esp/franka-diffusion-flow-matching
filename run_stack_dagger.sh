#!/usr/bin/env bash
# Stacking DAgger: the DP student drives, the scripted teacher (resync mode: phase re-derived from the scene) labels
# every visited state; fine-tune on all data so far; quick check on seed 1001 each round. Final model: 3 fresh seeds.
# Steps are skipped when their output exists, so a re-run resumes.
set -uo pipefail
cd "$(dirname "$0")"
P=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}
R=runs/stack; ROUNDS=${ROUNDS:-3}; STEPS=${STEPS:-15000}
log() { echo "=== $(date +%m-%d\ %H:%M:%S) $*"; }
run() { local name=$1; shift; "$@" > $R/$name.log 2>&1 || { echo "FAILED $name"; tail -20 $R/$name.log; exit 1; }; grep -hE "^EVAL|\[collect\] \{|done:" $R/$name.log || true; }
prev=$R/dp/bc_best.pt
data="$R/demos_clean $R/demos_dart01"
for k in $(seq 1 $ROUNDS); do
  log "round $k: student rollouts labelled by the scripted teacher"
  [ -f $R/dagger$k/meta.json ] || { rm -rf $R/dagger$k; run dagger$k $P collect_task.py --task stack --teacher scripted --student $prev --beta 0.0 \
      --output $R/dagger$k --target 400 --num-envs 24 --seed $((80 + k)); }
  data="$data $R/dagger$k"
  log "round $k: fine-tune on $data"
  [ -f $R/dp_dagger$k/bc_final.pt ] || { rm -rf $R/dp_dagger$k; run train_dagger$k $P train_dp.py --dataset $data --init $prev --steps $STEPS --warmup-steps 200 --output $R/dp_dagger$k --bf16; }
  prev=$R/dp_dagger$k/bc_best.pt
  [ -f $R/check_dagger${k}.json ] || run check_dagger$k $P eval_task.py --task stack --checkpoint $prev --episodes 200 --num-envs 50 --seed 1001 --output $R/check_dagger${k}.json
done
log "final test on fresh seeds"
for s in 1002 1003; do
  [ -f $R/test_dagger_s$s.json ] || run test_dagger_s$s $P eval_task.py --task stack --checkpoint $prev --episodes 200 --num-envs 50 --seed $s --output $R/test_dagger_s$s.json
done
[ -f $R/vid_dagger.json ] || run vid_dagger $P eval_task.py --task stack --checkpoint $prev --num-envs 8 --seed 32 --video-dir $R/videos_dagger --output $R/vid_dagger.json
log "done"
