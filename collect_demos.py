#!/usr/bin/env python3
"""Roll out the privileged PPO teacher with the camera(s) and keep the successful episodes.

With --student (DAgger): the Diffusion Policy drives (the teacher for a --beta fraction of envs), every visited
state is labelled with the teacher's action and every episode is kept, successful or not.

Writes the pouring_diffusion dataset format: images.npy [E,T,H,W,3] uint8, proprio.npy and share.npy
[E,T,1,D], action.npy [E,T,8] (normalized absolute commands) and kept.npy.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New dataset folder; an existing one is refused.")
    p.add_argument("--target", type=int, default=300, help="Successful episodes to keep.")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=40)
    p.add_argument("--noise-std", type=float, default=0.0, help="DART noise on the executed arm command.")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--wrist", action="store_true", help="Add the eye-in-hand camera (image = [workspace | wrist]).")
    p.add_argument("--student", type=Path, default=None, help="DAgger: Diffusion Policy checkpoint that drives.")
    p.add_argument("--beta", type=float, default=0.0, help="DAgger: fraction of envs driven by the teacher.")
    p.add_argument("--zoom", action="store_true", help="Zoomed workspace camera (fd_core.env.ZOOM_FOCAL).")
    p.add_argument("--robust-teacher", action="store_true", help="Teacher from train_teacher_robust.py (no last-action obs).")
    p.add_argument("--perturb", action="store_true",
                   help="Recovery demos: inject random failure states (empty close / drop / knock) into teacher rollouts; "
                        "keep the successful recoveries, labelled with the teacher's own actions.")
    p.add_argument("--seed", type=int, default=11)
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def main(args):
    import numpy as np
    import torch

    from fd_core.env import EPISODE_STEPS, PROPRIO_DIM, Franka, load_teacher, make_env
    from fd_core.rollout import TeacherActor, run_batch

    device = args.device
    env, raw = make_env(args.num_envs, device, args.seed, args.image_size, wrist=args.wrist, robust_teacher=args.robust_teacher, zoom=args.zoom)
    franka = Franka(raw)
    actor = TeacherActor(load_teacher(args.teacher, device), franka)
    obs_h = 2
    if args.student:
        from fd_core.policy import load_dp
        from fd_core.rollout import DaggerActor, DiffusionActor
        model, cfg = load_dp(args.student, device)
        actor = TeacherActor(actor.teacher, franka, own_last_action=not args.robust_teacher)   # robust teacher has no last-action obs
        actor, obs_h = DaggerActor(DiffusionActor(model, cfg.exec_horizon), actor, args.beta), cfg.obs_horizon
    out = args.output
    out.mkdir(parents=True)
    s = args.image_size
    images = np.lib.format.open_memmap(out / "images.npy", "w+", np.uint8, (args.target, EPISODE_STEPS, s, s * (2 if args.wrist else 1), 3))
    proprio = np.zeros((args.target, EPISODE_STEPS, 1, PROPRIO_DIM), np.float32)
    action = np.zeros((args.target, EPISODE_STEPS, 8), np.float32)
    cube = np.zeros((args.target, EPISODE_STEPS, 3), np.float32)        # privileged, auxiliary target only
    gen = torch.Generator(device=device).manual_seed(args.seed)
    kept, tried, succ_total, dists = 0, 0, 0, []
    for b in range(args.max_batches):
        act_b = actor
        if args.perturb:
            from fd_core.rollout import PerturbActor
            n_ = args.num_envs; u = torch.rand(n_, device=device, generator=gen)
            kinds = torch.where(u < 0.45, 1, torch.where(u < 0.80, 3, 2))            # 45% empty close, 35% knock, 20% drop
            t0 = torch.where(kinds == 1, torch.randint(10, 46, (n_,), device=device, generator=gen),
                 torch.where(kinds == 3, torch.randint(15, 46, (n_,), device=device, generator=gen),
                             torch.randint(80, 141, (n_,), device=device, generator=gen)))
            dur = torch.where(kinds == 3, 1, torch.randint(8, 21, (n_,), device=device, generator=gen))
            act_b = PerturbActor(actor, franka, kinds, t0, dur, generator=gen)
        r = run_batch(env, raw, franka, act_b, obs_horizon=obs_h, record=True, noise_std=args.noise_std, generator=gen)
        tried += args.num_envs
        succ_total += int(r["success"].sum())
        dists += r["dist"].tolist()
        keep_ids = range(args.num_envs) if args.student else r["success"].nonzero().flatten().tolist()
        for i in keep_ids:
            if kept == args.target:
                break
            images[kept] = r["image"][i].numpy()
            proprio[kept] = r["proprio"][i].numpy()
            cube[kept] = r["cube"][i].numpy()
            action[kept] = r["action"][i].numpy()
            kept += 1
        print(f"[collect] batch {b} batch_success={r['success'].float().mean():.3f} kept={kept}/{args.target}", flush=True)
        if kept == args.target:
            break
    images.flush()
    np.save(out / "proprio.npy", proprio[:kept])
    np.save(out / "share.npy", np.zeros((kept, EPISODE_STEPS, 1, 1), np.float32))
    np.save(out / "action.npy", action[:kept])
    np.save(out / "cube.npy", cube[:kept])
    kept_mask = np.zeros(args.target, bool); kept_mask[:kept] = True
    np.save(out / "kept.npy", kept_mask)
    meta = {"teacher": str(args.teacher), "kept": kept, "tried": tried, "rollout_success_rate": succ_total / tried,
            "noise_std": args.noise_std, "student": str(args.student) if args.student else None, "perturb": args.perturb, "beta": args.beta, "seed": args.seed, "image_size": s, "wrist": args.wrist, "zoom": args.zoom, "mean_final_dist": float(np.mean(dists))}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("[collect] " + json.dumps(meta), flush=True)


if __name__ == "__main__":
    cli = parse_args()
    if cli.output.exists():
        sys.exit(f"{cli.output} already exists; choose a new --output")
    cli.enable_cameras = True
    cli.headless = True
    app = AppLauncher(cli).app
    code = 0
    try:
        main(cli)
    except BaseException:
        traceback.print_exc(); sys.stdout.flush(); code = 1
    import os
    os._exit(code)
