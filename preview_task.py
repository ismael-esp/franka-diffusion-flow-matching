#!/usr/bin/env python3
"""Save the student camera images (and the video camera) of a few envs right after reset, to set up camera poses."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))
p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--num-envs", type=int, default=4)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.enable_cameras = True; args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from fd_core.tasks import Arm, make_task_env  # noqa: E402

env, raw = make_task_env(args.task, args.num_envs, args.device, 5, cameras=True, image_size=256, video_size=(480, 360))
arm = Arm(raw)
env.reset()
zero = torch.zeros(args.num_envs, raw.action_manager.total_action_dim, device=args.device); zero[:, -1] = 1.0
for _ in range(5):
    env.step(zero)
rows = []
for i in range(args.num_envs):
    pol = arm.image()[i].cpu().numpy()
    vid = np.asarray(Image.fromarray(arm.image("video_camera")[i].cpu().numpy()).resize((341, 256)))
    rows.append(np.concatenate([vid, pol], 1))
args.out.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(np.concatenate(rows, 0)).save(args.out)
print("saved", args.out)
os._exit(0)
