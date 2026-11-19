#!/usr/bin/env python3
"""Record one episode of several envs as an H.264 video: third-person grid and the policy's own camera."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import subprocess  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

CELL_W, CELL_H, COLUMNS = 424, 240, 3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New .mp4 file; the policy view is saved next to it.")
    p.add_argument("--stage", default="eval_nolag", help="Env rules from dp_core.env.STAGES.")
    p.add_argument("--num-envs", type=int, default=6)
    p.add_argument("--seed", type=int, default=301)
    p.add_argument("--fps", type=int, default=30, help="30 plays the 60 Hz sim at the hardware's half speed.")
    p.add_argument("--snap-grip", action="store_true", help="Round grip commands to fully open/closed, like the demos.")
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def to_uint8(frames) -> "np.ndarray":  # noqa: F821
    """[N,H,W,3] uint8 from an RGB tensor in either layout and range."""
    if frames.shape[-1] != 3:
        frames = frames.permute(0, 2, 3, 1)
    frames = frames[..., :3].float()
    if float(frames.max()) <= 1.01:
        frames = frames * 255.0
    return frames.clamp(0.0, 255.0).byte().cpu().numpy()


def open_writer(path: Path, width: int, height: int, fps: int) -> subprocess.Popen:
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def grid(cells, status) -> "np.ndarray":  # noqa: F821
    import cv2
    import numpy as np

    colors = {"running": (255, 255, 255), "SUCCESS": (60, 220, 60), "STOPPED": (240, 70, 70), "FAILED": (240, 70, 70)}
    tiles = []
    for i, cell in enumerate(cells):
        tile = cv2.resize(cell, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA).copy()
        cv2.putText(tile, f"env {i}: {status[i]}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[status[i]], 2, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % COLUMNS:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + COLUMNS]) for i in range(0, len(tiles), COLUMNS)])


def main(args):
    import torch

    from dp_core.env import build_pouring_env, right_image
    from dp_core.stepper import ChunkStepper
    from eval_policy import load_policy

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, cfg, kind = load_policy(args.checkpoint, device)
    env, raw_env = build_pouring_env(args.num_envs, device, args.seed, args.output.with_suffix("").with_name(args.output.stem + "_env"),
                                     stage=args.stage, record=True)
    stepper = ChunkStepper(env, cfg.obs_horizon, cfg.exec_horizon, device, right_image)
    horizon = int(raw_env.max_episode_length)
    rows = (args.num_envs + COLUMNS - 1) // COLUMNS
    size = (CELL_W * COLUMNS, CELL_H * rows)
    third_person = open_writer(args.output, *size, args.fps)
    policy_view = open_writer(args.output.with_name(args.output.stem + "_policy_view.mp4"), *size, args.fps)

    status = ["running"] * args.num_envs
    frozen_scene, frozen_view = [None] * args.num_envs, [None] * args.num_envs
    succeeded = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    with torch.no_grad():
        for t in range(horizon):
            need = stepper.needs_plan()
            if need.any():
                plan = model.sample(stepper.image_history[need], stepper.obs_history[need])
                if args.snap_grip:
                    plan[..., [6, 13]] = torch.where(plan[..., [6, 13]] >= 0, 1.0, -1.0)
                stepper.set_plan(need, plan)
            scene = to_uint8(raw_env.get_camera_debug_images()["record_rgb"])
            view = to_uint8(right_image({"img_right": stepper.image_history[:, -1]}))
            succeeded |= raw_env.success_buf.bool().to(device)
            for i in range(args.num_envs):
                if status[i] == "running" and bool(succeeded[i]):
                    status[i] = "SUCCESS"
            third_person.stdin.write(grid([f if f is not None else s for f, s in zip(frozen_scene, scene)], status).tobytes())
            policy_view.stdin.write(grid([f if f is not None else v for f, v in zip(frozen_view, view)], status).tobytes())
            if t == horizon - 1:
                break
            _, done, _, _ = stepper.step()
            for i in done.nonzero().flatten().tolist():  # a guard stop resets the env, so hold its last frame
                if frozen_scene[i] is None:
                    frozen_scene[i], frozen_view[i] = scene[i], view[i]
                    if status[i] == "running":
                        status[i] = "STOPPED"
    status = [s if s != "running" else "FAILED" for s in status]
    for writer in (third_person, policy_view):
        writer.stdin.close()
        writer.wait()
    print(f"[video] {kind} stage={args.stage} successes={status.count('SUCCESS')}/{args.num_envs} status={status} -> {args.output}", flush=True)
    raw_env.close()


if __name__ == "__main__":
    cli = parse_args()
    if cli.output.exists():
        sys.exit(f"{cli.output} already exists; choose a new --output")
    cli.output = cli.output.resolve()
    cli.checkpoint = cli.checkpoint.resolve()
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.enable_cameras = True
    simulation_app = AppLauncher(cli).app
    code = 0
    try:
        main(cli)
    except BaseException:
        # Closing Isaac Sim can kill the process, so report the failure first.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        code = 1
    simulation_app.close()
    sys.exit(code)
