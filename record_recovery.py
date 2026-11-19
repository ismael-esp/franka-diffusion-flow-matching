#!/usr/bin/env python3
"""Record Diffusion Policy rollouts with an injected failure (fd_core.rollout.PerturbActor) and save one mp4 per env,
named with the outcome, so successful recoveries can be picked for the report.
Uses the standard (non-tiled) video camera, which does not disturb the policy's tiled cameras."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--kind", choices=["empty_close", "drop", "knock"], required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=21)
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def main(args):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from diagnose import TIMING
    from fd_core.env import Franka, make_env
    from fd_core.policy import camera_setup, load_dp
    from fd_core.rollout import DiffusionActor, PerturbActor, run_batch

    torch.manual_seed(args.seed)
    wrist, zoom = camera_setup(args.checkpoint)
    env, raw = make_env(args.num_envs, args.device, args.seed, 128, video_size=(640, 480), wrist=wrist, zoom=zoom)
    fr = Franka(raw)
    model, cfg = load_dp(args.checkpoint, args.device)
    n = args.num_envs; k = PerturbActor.KINDS.index(args.kind); t0, dur = TIMING[args.kind]
    actor = PerturbActor(DiffusionActor(model, cfg.exec_horizon), fr, torch.full((n,), k, device=args.device),
                         torch.full((n,), t0, device=args.device), torch.full((n,), dur, device=args.device),
                         generator=torch.Generator(device=args.device).manual_seed(args.seed))
    r = run_batch(env, raw, fr, actor, obs_horizon=cfg.obs_horizon, video_cam="video_camera")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = ["success", "never_lifted", "dropped", "held_off_goal", "fell_off_table"]
    summary = []
    for i in range(n):
        out = names[int(r["outcome"][i])]
        path = args.out_dir / f"{args.kind}_s{args.seed}_e{i}_{out}.mp4"
        w = imageio.get_writer(path, fps=50, codec="libx264", quality=8)
        for f in r["frames"][i].numpy():
            w.append_data(f)
        w.close()
        summary.append({"env": i, "outcome": out, "file": path.name})
    print("[rec] " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    cli = parse_args()
    cli.enable_cameras = True; cli.headless = True
    app = AppLauncher(cli).app
    code = 0
    try:
        main(cli)
    except BaseException:
        traceback.print_exc(); sys.stdout.flush(); code = 1
    os._exit(code)
