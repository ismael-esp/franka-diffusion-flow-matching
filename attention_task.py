#!/usr/bin/env python3
"""record_attention.py for fd_core.tasks tasks (drawer, stack): the SAME attention method (spatial-softmax heatmap
weighted by how strongly the policy reads each keypoint channel, plus the 12 most-read keypoints) on the images the
student sees. Flow-matching checkpoints work unchanged (same encoder). Saves one npz per env (image, third-person
frames, heat, kp, kw, outcome); stills are composed offline by make_attention_stills.py."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=32)
    p.add_argument("--inference-steps", type=int, default=0)
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def main(args):
    import numpy as np
    import torch
    from fd_core.policy import load_dp
    from fd_core.rollout import DiffusionActor
    from fd_core.tasks import TASKS, Arm, make_task_env, run_episodes
    from record_attention import attention

    torch.manual_seed(args.seed)
    spec = TASKS[args.task]
    env, raw = make_task_env(args.task, args.num_envs, args.device, args.seed, cameras=True, video_size=(480, 360))
    arm = Arm(raw)
    model, cfg = load_dp(args.checkpoint, args.device)
    if args.inference_steps:
        from dp_core.model import ddim_timesteps
        model.ddim_steps = ddim_timesteps(cfg.num_train_timesteps, args.inference_steps).to(model.ddim_steps.device)
    r = run_episodes(env, raw, arm, DiffusionActor(model, cfg.exec_horizon), args.task, obs_horizon=cfg.obs_horizon,
                     record=True, video_cam="video_camera")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i in range(args.num_envs):
            o = int(r["outcome"][i]); imgs = r["image"][i]
            heat, kp, kw = attention(model, imgs)
            np.savez_compressed(args.out_dir / f"att_{args.task}_s{args.seed}_e{i}_{spec.outcome_names[o]}.npz",
                                image=imgs.numpy(), frames=r["frames"][i].numpy(), heat=heat.numpy(), kp=kp.numpy(),
                                kw=kw.numpy(), outcome=o)
    print("[att] " + str([(i, spec.outcome_names[int(r["outcome"][i])]) for i in range(args.num_envs)]),
          "image", tuple(r["image"].shape), flush=True)


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
