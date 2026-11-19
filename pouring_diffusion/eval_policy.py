#!/usr/bin/env python3
"""Evaluate a BC or DPPO checkpoint in Isaac Lab under the frozen-eval rules (full horizon, strict contact)."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="JSON result file; an existing one is refused.")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=201)
    p.add_argument("--stage", default="eval", help="Env rules from dp_core.env.STAGES.")
    p.add_argument("--snap-grip", action="store_true", help="Round grip commands to fully open/closed, like the demos.")
    p.add_argument("--ddim-steps", type=int, default=0, help="DDIM sampling steps; 0 keeps the checkpoint's.")
    p.add_argument("--exec-horizon", type=int, default=0, help="Actions executed per plan; 0 keeps the checkpoint's.")
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def load_policy(path: Path, device):
    import torch

    from dp_core.model import DiffusionPolicy, PolicyConfig

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["kind"] == "dsrl":
        from dp_core.dsrl import build_dsrl
        policy = build_dsrl(checkpoint, device)
        return policy, policy.cfg, "dsrl"
    cfg = PolicyConfig(**checkpoint["config"])
    model = DiffusionPolicy(cfg, torch.zeros(cfg.proprio_dim), torch.ones(cfg.proprio_dim))
    if checkpoint.get("finetune_steps", 0) > 0:
        model.freeze_early_steps(checkpoint["finetune_steps"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), cfg, checkpoint["kind"]


def main(args):
    import torch

    from dp_core.env import build_pouring_env, right_image
    from dp_core.stepper import ChunkStepper, info_success

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, cfg, kind = load_policy(args.checkpoint, device)
    if args.ddim_steps and kind == "dsrl":
        sys.exit("--ddim-steps does not apply to a DSRL checkpoint: its actor was trained for its own chain")
    if args.ddim_steps:
        from dp_core.model import ddim_timesteps
        model.ddim_steps = ddim_timesteps(cfg.num_train_timesteps, args.ddim_steps).to(device)
    env, raw_env = build_pouring_env(args.num_envs, device, args.seed, args.output.parent / f"{args.output.stem}_env", stage=args.stage)
    exec_horizon = args.exec_horizon or cfg.exec_horizon
    stepper = ChunkStepper(env, cfg.obs_horizon, exec_horizon, device, right_image)

    successes, finished = 0, 0
    with torch.no_grad():
        while finished < args.episodes:
            need = stepper.needs_plan()
            if need.any():
                plan = model.sample(stepper.image_history[need], stepper.obs_history[need])
                if args.snap_grip:
                    plan[..., [6, 13]] = torch.where(plan[..., [6, 13]] >= 0, 1.0, -1.0)
                stepper.set_plan(need, plan)
            _, done, _, infos = stepper.step()
            for i in done.nonzero().flatten().tolist():
                successes += int(info_success(infos[i]))
                finished += 1
            if done.any():
                print(f"[eval] finished={finished}/{args.episodes} success={successes / finished:.3f}", flush=True)

    low, high = wilson_interval(successes, finished)
    result = {"checkpoint": str(args.checkpoint), "kind": kind, "episodes": finished, "successes": successes,
              "success_rate": successes / finished, "ci95": [low, high], "seed": args.seed, "stage": args.stage,
              "snap_grip": args.snap_grip, "exec_horizon": exec_horizon, "ddim_steps": args.ddim_steps or cfg.num_inference_steps}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1))
    print("[eval] " + json.dumps(result), flush=True)
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
