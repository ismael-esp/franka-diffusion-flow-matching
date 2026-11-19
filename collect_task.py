#!/usr/bin/env python3
"""Collect teacher demos (or DAgger data) for a fd_core.tasks task, in the pouring_diffusion dataset format.
Keeps successful episodes (all episodes with --student). Scripted teachers: --teacher scripted (stack)."""
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
    p.add_argument("--teacher", required=True, help="rsl_rl checkpoint, or 'scripted'")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target", type=int, default=300)
    p.add_argument("--num-envs", type=int, default=48)
    p.add_argument("--max-batches", type=int, default=60)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--student", type=Path, default=None)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=11)
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def make_teacher(args, arm, raw):
    from fd_core.env import load_teacher
    from fd_core.rollout import TeacherActor
    if args.teacher == "scripted":
        from fd_core.scripted import DrawerScript, StackScript
        if args.task == "stack":   # under DAgger the student drives: keep the teacher's phase in sync with the scene
            return StackScript(raw, arm, resync=bool(getattr(args, "student", None)))
        return DrawerScript(raw, arm)
    return TeacherActor(load_teacher(args.teacher, args.device), arm)


def main(args):
    import numpy as np
    import torch
    from fd_core.tasks import TASKS, Arm, make_task_env, run_episodes

    spec = TASKS[args.task]; T = spec.steps
    env, raw = make_task_env(args.task, args.num_envs, args.device, args.seed, cameras=True)
    arm = Arm(raw)
    actor = make_teacher(args, arm, raw); obs_h = 2
    if args.student:
        from fd_core.policy import load_dp
        from fd_core.rollout import DaggerActor, DiffusionActor
        model, cfg = load_dp(args.student, args.device)
        actor, obs_h = DaggerActor(DiffusionActor(model, cfg.exec_horizon), actor, args.beta), cfg.obs_horizon
    out = args.output; out.mkdir(parents=True)
    images = np.lib.format.open_memmap(out / "images.npy", "w+", np.uint8, (args.target, T, 128, 256, 3))
    proprio = np.zeros((args.target, T, 1, 18), np.float32); action = np.zeros((args.target, T, 8), np.float32)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    kept, tried, succ = 0, 0, 0
    for b in range(args.max_batches):
        r = run_episodes(env, raw, arm, actor, args.task, obs_horizon=obs_h, record=True, noise_std=args.noise_std, generator=gen)
        tried += args.num_envs; succ += int(r["success"].sum())
        ids = range(args.num_envs) if args.student else r["success"].nonzero().flatten().tolist()
        for i in ids:
            if kept == args.target:
                break
            images[kept] = r["image"][i].numpy(); proprio[kept] = r["proprio"][i].numpy(); action[kept] = r["action"][i].numpy()
            kept += 1
        print(f"[collect] batch {b} success={r['success'].float().mean():.3f} kept={kept}/{args.target}", flush=True)
        if kept == args.target:
            break
    images.flush()
    np.save(out / "proprio.npy", proprio[:kept]); np.save(out / "action.npy", action[:kept])
    np.save(out / "share.npy", np.zeros((kept, T, 1, 1), np.float32))
    m = np.zeros(args.target, bool); m[:kept] = True; np.save(out / "kept.npy", m)
    meta = {"task": args.task, "teacher": args.teacher, "kept": kept, "tried": tried, "rollout_success_rate": succ / tried,
            "noise_std": args.noise_std, "student": str(args.student) if args.student else None, "beta": args.beta,
            "seed": args.seed, "image_size": 128, "wrist": True, "zoom": False}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("[collect] " + json.dumps(meta), flush=True)


if __name__ == "__main__":
    cli = parse_args()
    if cli.output.exists():
        sys.exit(f"{cli.output} exists")
    cli.enable_cameras = True; cli.headless = True
    app = AppLauncher(cli).app
    code = 0
    try:
        main(cli)
    except BaseException:
        traceback.print_exc(); sys.stdout.flush(); code = 1
    os._exit(code)
