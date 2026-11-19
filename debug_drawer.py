#!/usr/bin/env python3
"""Measure the PPO drawer teacher's grasp: hand pose in the handle frame at the first gripper close, and handle data."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))
p = argparse.ArgumentParser(); p.add_argument("--envs", type=int, default=8); p.add_argument("--teacher", required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.headless = True; args.task = "drawer"
app = AppLauncher(args).app

import torch  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

from collect_task import make_teacher  # noqa: E402
from fd_core.tasks import Arm, drawer_pos, make_task_env  # noqa: E402

env, raw = make_task_env("drawer", args.envs, args.device, 5)
arm = Arm(raw)
actor = make_teacher(args, arm, raw)
obs, _ = env.reset(); obs = obs["policy"]
if hasattr(actor, "reset"): actor.reset(args.envs)
robot = raw.scene["robot"]; hand = robot.find_bodies("panda_hand")[0][0]
cf = raw.scene["cabinet_frame"]
print("cabinet_frame targets", cf.data.target_frame_names, flush=True)
done = torch.zeros(args.envs, dtype=torch.bool, device=args.device)
for t in range(120):
    a = actor(t, obs, None, None)
    if t in (0, 1):
        hp = cf.data.target_pos_w[:, 0]; hq = cf.data.target_quat_w[:, 0]
        print("handle pos(local root)", (hp - robot.data.root_pos_w)[0].tolist(), "quat", hq[0].tolist(), flush=True)
    close = a[:, 7] < 0
    new = close & ~done
    for i in torch.nonzero(new).flatten().tolist():
        hp = cf.data.target_pos_w[i:i+1, 0]; hq = cf.data.target_quat_w[i:i+1, 0]
        rp, rq = subtract_frame_transforms(hp, hq, robot.data.body_pos_w[i:i+1, hand], robot.data.body_quat_w[i:i+1, hand])
        print(f"env{i} t{t} q {[round(x,3) for x in robot.data.joint_pos[i,:7].tolist()]} close: hand in handle frame pos {[round(x,4) for x in rp[0].tolist()]} quat {[round(x,3) for x in rq[0].tolist()]} handle_q {[round(x,3) for x in hq[0].tolist()]}", flush=True)
    done |= close
    obs, *_ = env.step(arm.norm_to_env(a)); obs = obs["policy"]
    if t % 20 == 19:
        print(f"t{t} drawer {[round(x,3) for x in drawer_pos(raw).tolist()]}", flush=True)
sys.stdout.flush(); os._exit(0)
