#!/usr/bin/env python3
"""Teacher-labelled rollouts from the states the Diffusion Policy reaches, in the render_demos.py format.

--mode takeover (default, recovery demos): the policy drives until a random switch step, then the
    privileged teacher drives to the end. Only episodes the teacher finishes successfully without a
    guard stop are kept, and start.npy marks the switch step: training uses the teacher-driven steps
    only, so every action chunk is one the teacher really executed under the lag.
--mode dagger: the policy drives (the teacher executes a chunk with probability --beta) and every
    state gets the teacher's one-step label. Round 1 of this dropped the policy from 45% to 5%:
    stitched one-step labels are not coherent 32-step chunks.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True, help="Diffusion Policy that drives the rollouts.")
    p.add_argument("--output", type=Path, required=True, help="New dataset folder; an existing one is refused.")
    p.add_argument("--mode", choices=["takeover", "dagger"], default="takeover")
    p.add_argument("--target", type=int, default=300, help="Episodes to keep.")
    p.add_argument("--max-batches", type=int, default=16)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--switch-min", type=int, default=32, help="takeover: earliest switch step.")
    p.add_argument("--switch-max", type=int, default=480, help="takeover: latest switch step (exclusive).")
    p.add_argument("--beta", type=float, default=0.2, help="dagger: probability that the teacher executes a chunk.")
    p.add_argument("--no-snap-grip", action="store_true", help="Execute the policy's raw grip commands.")
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--stage", default="collect_lag")
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def main(args):
    import numpy as np
    import torch

    from collect_teacher_demos import load_teacher, teacher_mean
    from dp_core.env import IMAGE_HEIGHT, IMAGE_WIDTH, RELEASE, build_pouring_env, right_image
    from eval_policy import load_policy

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True)
    env, raw_env = build_pouring_env(args.num_envs, device, args.seed, args.output / "env", stage=args.stage)
    teacher = load_teacher(RELEASE / "teacher" / "transformer_534.pt", device)
    model, cfg, _ = load_policy(args.checkpoint, device)
    horizon = int(raw_env.max_episode_length)
    n = args.num_envs
    images = np.lib.format.open_memmap(args.output / "images.npy", mode="w+", dtype=np.uint8,
                                       shape=(args.target, horizon, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    proprio, share, action = None, None, np.zeros((args.target, horizon, 14), dtype=np.float32)
    start = np.zeros(args.target, dtype=np.int64)
    success = np.zeros(args.target, dtype=bool)
    count, attempts, successes, stopped, teacher_steps = 0, 0, 0, 0, 0
    for batch in range(args.max_batches):
        obs, state, obs_dict = env.reset()
        image = right_image(obs_dict).to(device)
        image_history = image[:, None].repeat(1, cfg.obs_horizon, 1, 1, 1)
        obs_history = obs.to(device)[:, None].repeat(1, cfg.obs_horizon, 1, 1)
        if args.mode == "takeover":
            switch = torch.randint(args.switch_min, args.switch_max, (n,), device=device)
        else:
            switch = torch.full((n,), horizon, device=device)
        frames = np.zeros((horizon, n, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        obs_seq, state_seq, act_seq = [], [], []
        succeeded = torch.zeros(n, dtype=torch.bool, device=device)
        terminated = torch.zeros(n, dtype=torch.bool, device=device)
        for t in range(horizon):
            cursor = t % cfg.exec_horizon
            with torch.no_grad():
                if cursor == 0:
                    plan = model.sample(image_history, obs_history)
                    if not args.no_snap_grip:
                        plan[..., [6, 13]] = torch.where(plan[..., [6, 13]] >= 0, 1.0, -1.0)
                    chunk_teacher = torch.rand(n, device=device) < (args.beta if args.mode == "dagger" else 0.0)
                clean = teacher_mean(teacher, obs_dict, n).to(device).reshape(n, 14)
            use_teacher = chunk_teacher | (t >= switch)
            executed = torch.where(use_teacher[:, None], clean, plan[:, cursor])
            teacher_steps += int(use_teacher.sum())
            frames[t] = (image * 255.0).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            obs_seq.append(obs.cpu().numpy())
            state_seq.append(state.cpu().numpy())
            act_seq.append(clean.cpu().numpy())
            obs, _, state, _, done, _, obs_dict = env.step(executed.reshape(n, 2, 7).permute(1, 0, 2), t)
            if t < horizon - 1:
                terminated |= done[:, 0].bool().to(device)
            succeeded |= raw_env.success_buf.bool().to(device)
            image = right_image(obs_dict).to(device)
            image_history = torch.cat([image_history[:, 1:], image[:, None]], dim=1)
            obs_history = torch.cat([obs_history[:, 1:], obs.to(device)[:, None]], dim=1)
        attempts += n
        successes += int(succeeded.sum())
        stopped += int(terminated.sum())
        obs_seq, state_seq, act_seq = np.stack(obs_seq, 1), np.stack(state_seq, 1), np.stack(act_seq, 1)
        if proprio is None:
            proprio = np.zeros((args.target, *obs_seq.shape[1:]), dtype=np.float32)
            share = np.zeros((args.target, *state_seq.shape[1:]), dtype=np.float32)
        keep = (succeeded & ~terminated) if args.mode == "takeover" else torch.ones_like(succeeded)
        for env_id in np.flatnonzero(keep.cpu().numpy())[: args.target - count]:
            images[count] = frames[:, env_id]
            proprio[count], share[count], action[count] = obs_seq[env_id], state_seq[env_id], act_seq[env_id]
            start[count] = min(int(switch[env_id]), horizon - 1) if args.mode == "takeover" else 0
            success[count] = bool(succeeded[env_id])
            count += 1
        print(f"[collect:{args.mode}] batch={batch} kept={count}/{args.target} attempts={attempts} "
              f"success={successes / attempts:.3f} guard_stopped={stopped / attempts:.3f} "
              f"teacher_steps={teacher_steps / (attempts * horizon):.3f}", flush=True)
        if count >= args.target:
            break
    images.flush()
    kept = np.zeros(args.target, dtype=bool)
    kept[:count] = True
    np.save(args.output / "proprio.npy", proprio)
    np.save(args.output / "share.npy", share)
    np.save(args.output / "action.npy", action)
    np.save(args.output / "kept.npy", kept)
    np.save(args.output / "start.npy", start)
    np.save(args.output / "success.npy", success)
    meta = {"mode": args.mode, "episodes": args.target, "kept": count, "horizon": horizon, "attempts": attempts,
            "successes": successes, "guard_stopped": stopped, "beta": args.beta,
            "switch_range": [args.switch_min, args.switch_max], "snap_grip": not args.no_snap_grip,
            "stage": args.stage, "seed": args.seed, "policy": str(args.checkpoint),
            "teacher": "release/pouring/teacher/transformer_534.pt"}
    (args.output / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[collect:{args.mode}] done " + json.dumps(meta), flush=True)
    raw_env.close()


if __name__ == "__main__":
    cli = parse_args()
    if cli.output.exists():
        sys.exit(f"{cli.output} already exists; choose a new --output")
    cli.output = cli.output.resolve()
    cli.checkpoint = cli.checkpoint.resolve()
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
