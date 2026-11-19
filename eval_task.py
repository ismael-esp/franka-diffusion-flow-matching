#!/usr/bin/env python3
"""Evaluate a Diffusion Policy (or the teacher) on a fd_core.tasks task; optional videos (third-person | policy input).
Prints 'EVAL <name>: k/n = p% +- SE' and the outcome counts; writes a JSON."""
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
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", default=None, help="DP checkpoint; omit to evaluate --teacher")
    p.add_argument("--teacher", default=None)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--num-envs", type=int, default=50)
    p.add_argument("--seed", type=int, default=201)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--video-dir", type=Path, default=None, help="Record one batch of videos (use a small --num-envs).")
    p.add_argument("--inference-steps", type=int, default=0, help="DDIM / flow steps; 0 keeps the checkpoint's.")
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def main(args):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from fd_core.tasks import TASKS, Arm, make_task_env, run_episodes

    torch.manual_seed(args.seed)
    spec = TASKS[args.task]
    env, raw = make_task_env(args.task, args.num_envs, args.device, args.seed, cameras=True,
                             video_size=(480, 360) if args.video_dir else None)
    arm = Arm(raw)
    if args.checkpoint:
        from fd_core.policy import load_dp
        from fd_core.rollout import DiffusionActor
        model, cfg = load_dp(args.checkpoint, args.device)
        if args.inference_steps:
            from dp_core.model import ddim_timesteps
            model.ddim_steps = ddim_timesteps(cfg.num_train_timesteps, args.inference_steps).to(model.ddim_steps.device)
        actor, obs_h = DiffusionActor(model, cfg.exec_horizon), cfg.obs_horizon
    else:
        from collect_task import make_teacher
        actor, obs_h = make_teacher(args, arm, raw), 2
    succ, outc, prog = [], [], []
    batches = max(1, args.episodes // args.num_envs) if not args.video_dir else 1
    for b in range(batches):
        r = run_episodes(env, raw, arm, actor, args.task, obs_horizon=obs_h, video_cam="video_camera" if args.video_dir else None)
        succ += r["success"].tolist(); outc += r["outcome"].tolist(); prog += r["progress"].tolist()
        print(f"[eval] batch {b}: {r['success'].float().mean():.3f}", flush=True)
        if args.video_dir:
            args.video_dir.mkdir(parents=True, exist_ok=True)
            ims = r["frames"]; pol = None
            for i in range(args.num_envs):
                name = spec.outcome_names[int(r["outcome"][i])]
                w = imageio.get_writer(args.video_dir / f"{args.task}_s{args.seed}_e{i}_{name}.mp4", fps=30, codec="libx264", quality=8)
                for f in ims[i].numpy():
                    w.append_data(f)
                w.close()
    n, k = len(succ), int(sum(succ)); p = k / n; se = (p * (1 - p) / n) ** 0.5
    counts = {nm: outc.count(j) for j, nm in enumerate(spec.outcome_names)}
    res = {"task": args.task, "checkpoint": args.checkpoint, "teacher": args.teacher, "seed": args.seed, "episodes": n,
           "inference_steps": args.inference_steps or None, "success": p, "se": se, "outcomes": counts, "mean_progress": float(np.mean(prog))}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(res, indent=1))
    name = Path(args.checkpoint).parent.name if args.checkpoint else "teacher"
    print(f"EVAL {name}: {k}/{n} = {100 * p:.1f}% +- {100 * se:.1f}  {counts}", flush=True)


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
