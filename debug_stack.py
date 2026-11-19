#!/usr/bin/env python3
"""Debug the scripted stack teacher (no cameras): log each env's phase transitions and the final stack result."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))
p = argparse.ArgumentParser(); p.add_argument("--envs", type=int, default=8); p.add_argument("--seed", type=int, default=3)
p.add_argument("--vmax", type=float, default=0.04); p.add_argument("--resync", action="store_true")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402

from fd_core.scripted import StackScript  # noqa: E402
from fd_core.tasks import TASKS, Arm, make_task_env, stack_success  # noqa: E402

env, raw = make_task_env("stack", args.envs, args.device, args.seed)
arm = Arm(raw)
obs, _ = env.reset()
sc = StackScript(raw, arm, vmax=args.vmax, resync=args.resync)
names = ["ABOVE", "DESC", "CLOSE", "LIFT", "OVER", "LOWER", "OPEN", "RETR", "DONE"]
prev = (sc.sub * 10 + sc.phase).clone()
log = [[] for _ in range(args.envs)]
for t in range(TASKS["stack"].steps - 1):   # the last step auto-resets
    a = sc(t, obs, None, None)
    obs, *_ = env.step(arm.norm_to_env(a))
    cur = sc.sub * 10 + sc.phase
    for i in torch.nonzero(cur != prev).flatten().tolist():
        log[i].append(f"{t}:{int(sc.sub[i])}{names[int(sc.phase[i])]}")
    prev = cur.clone()
ok12, ok23 = stack_success(raw)
from fd_core.tasks import stack_heights
c1,c2,c3=stack_heights(raw)
print("CUBE_SIZE?", raw.scene["cube_1"].cfg.spawn.scale if hasattr(raw.scene["cube_1"].cfg.spawn,"scale") else None, flush=True)
for i in range(args.envs):
    print("d12", [round(x,4) for x in (c2[i]-c1[i]).tolist()], "d23", [round(x,4) for x in (c3[i]-c2[i]).tolist()], "fing", [round(x,3) for x in raw.scene["robot"].data.joint_pos[i,7:9].tolist()], flush=True)
    print(f"env{i} ok12={bool(ok12[i])} ok23={bool(ok23[i])} :: " + " ".join(log[i]), flush=True)
sys.stdout.flush(); os._exit(0)
