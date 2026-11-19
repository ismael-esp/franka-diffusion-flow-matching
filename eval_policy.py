#!/usr/bin/env python3
"""Evaluate the Diffusion Policy (or the privileged teacher) on Franka lift: success over N episodes.

Success = the cube is within 5 cm of the commanded goal at the last of 250 steps and never fell off.
"""
from __future__ import annotations

import argparse
import os
import json
import math
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(os.environ.get("DP_ROOT", str(Path(__file__).resolve().parent / "pouring_diffusion")))   # dp_core


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=Path, help="Diffusion Policy checkpoint (train_dp.py).")
    g.add_argument("--teacher", type=Path, help="Evaluate the privileged PPO teacher instead.")
    p.add_argument("--output", type=Path, required=True, help="JSON result file; an existing one is refused.")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--num-envs", type=int, default=50)
    p.add_argument("--seed", type=int, default=201)
    p.add_argument("--ddim-steps", type=int, default=0, help="0 keeps the checkpoint's.")
    p.add_argument("--exec-horizon", type=int, default=0, help="0 keeps the checkpoint's.")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--robust-teacher", action="store_true", help="--teacher is a train_teacher_robust.py checkpoint.")
    p.add_argument("--wrist", action="store_true", help="Eye-in-hand camera too (automatic for a checkpoint trained on wrist demos).")
    p.add_argument("--video", type=int, default=0, help="Also save an mp4 of the first N envs of the first batch (use a small --num-envs: every env gets a 640x480 camera).")
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def wilson(s, n, z=1.96):
    p = s / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def save_video(frames, success, path, fps=50):
    import imageio.v2 as imageio
    import numpy as np
    n = frames.shape[0]
    cols = min(n, 4); rows = math.ceil(n / cols)
    t, h, w = frames.shape[1:4]
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for k in range(t):
        grid = np.zeros((rows * h, cols * w, 3), np.uint8)
        for i in range(n):
            f = frames[i, k].numpy().copy()
            if k > t - 40:                                     # green/red border at the end = success/failure
                c = (40, 200, 60) if success[i] else (220, 40, 40)
                f[:6], f[-6:], f[:, :6], f[:, -6:] = c, c, c, c
            grid[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = f
        writer.append_data(grid)
    writer.close()


def main(args):
    import torch
    from fd_core.env import Franka, load_teacher, make_env
    from fd_core.rollout import DiffusionActor, TeacherActor, run_batch

    torch.manual_seed(args.seed)
    device = args.device
    wrist, zoom = args.wrist, False
    if args.checkpoint:   # use the cameras the checkpoint was trained with
        ds = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("dataset", [])
        from fd_core.policy import camera_setup
        w_, zoom = camera_setup(args.checkpoint)
        wrist = wrist or w_
    env, raw = make_env(args.num_envs, device, args.seed, args.image_size, video_size=(640, 480) if args.video else None,
                        wrist=wrist, robust_teacher=args.robust_teacher, zoom=zoom)
    franka = Franka(raw)
    if args.teacher:
        actor, obs_h, info = TeacherActor(load_teacher(args.teacher, device), franka), 1, {"kind": "teacher"}
    else:
        from dp_core.model import ddim_timesteps
        from fd_core.policy import load_dp
        model, cfg = load_dp(args.checkpoint, device)
        if args.ddim_steps:
            model.ddim_steps = ddim_timesteps(cfg.num_train_timesteps, args.ddim_steps).to(device)
        exec_h = args.exec_horizon or cfg.exec_horizon
        actor, obs_h = DiffusionActor(model, exec_h), cfg.obs_horizon
        info = {"kind": "diffusion_policy", "wrist": wrist, "zoom": zoom, "ddim_steps": args.ddim_steps or cfg.num_inference_steps, "exec_horizon": exec_h}
    succ, done, dists = 0, 0, []
    while done < args.episodes:
        first = done == 0 and args.video > 0
        probe = {}
        def on_step(t, ih, ph):   # perception check: v3's cube head vs the true cube position, at steps 1 and 120
            if os.environ.get("FD_SNAP") and t in (0, 40, 80, 120, 200):
                probe.setdefault("snap", []).append(ih[:4, -1].permute(0, 2, 3, 1).cpu().numpy())
            if t in (1, 120) and hasattr(actor, "model") and hasattr(actor.model, "predict_cube"):
                probe[t] = float((actor.model.predict_cube(ih.float() / 255.0, ph) - franka.cube_pos()).norm(dim=-1).mean())
        r = run_batch(env, raw, franka, actor, obs_horizon=obs_h, video_cam="video_camera" if first else None, on_step=on_step)
        if getattr(actor, "grip_stats", None):
            import numpy as _np
            gs = _np.array(actor.grip_stats); actor.grip_stats.clear()
            print(f"[eval] grip |g|={gs[:, 0].mean():.3f} closed={gs[:, 1].mean():.3f} near0={gs[:, 2].mean():.3f}", flush=True)
        if "snap" in probe:
            import numpy as _np
            from PIL import Image as _Im
            _Im.fromarray(_np.concatenate([_np.concatenate(list(s), 0) for s in probe.pop("snap")], 1)).save(args.output.with_suffix(f".b{done}.png"))
        if probe:
            print(f"[eval] cube_err_from_images t1={probe.get(1, -1):.4f} t120={probe.get(120, -1):.4f} m", flush=True)
        take = min(args.num_envs, args.episodes - done)
        succ += int(r["success"][:take].sum()); done += take; dists += r["dist"][:take].tolist()
        print(f"[eval] {done}/{args.episodes} success={succ / done:.3f}", flush=True)
        if first:
            save_video(r["frames"][: args.video], r["success"][: args.video], args.output.with_suffix(".mp4"))
    lo, hi = wilson(succ, done)
    res = {**info, "source": str(args.checkpoint or args.teacher), "episodes": done, "successes": succ,
           "success_rate": succ / done, "se": math.sqrt(succ / done * (1 - succ / done) / done), "ci95": [lo, hi],
           "mean_final_dist": sum(dists) / len(dists), "seed": args.seed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=1))
    print("[eval] " + json.dumps(res), flush=True)


if __name__ == "__main__":
    cli = parse_args()
    if cli.output.exists():
        sys.exit(f"{cli.output} already exists; choose a new --output")
    cli.output = cli.output.resolve()
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
