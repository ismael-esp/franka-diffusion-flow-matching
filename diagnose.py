#!/usr/bin/env python3
"""Failure taxonomy + recovery test for the teacher and a Diffusion Policy on Franka lift.

For each policy and perturbation kind (none / empty_close / drop / knock, see fd_core.rollout.PerturbActor) run N
episodes and report success and outcomes: never lifted, dropped, held but off goal, fell off the table.
A policy 'recovers' if its success under a perturbation stays close to its unperturbed success."""
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
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, action="append", default=[], help="Diffusion Policy checkpoint(s) to test.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--num-envs", type=int, default=50)
    p.add_argument("--seed", type=int, default=401)
    p.add_argument("--kinds", default="none,empty_close,drop,knock")
    p.add_argument("--robust-teacher", action="store_true")
    p.add_argument("--zoom", action="store_true", help="Zoomed workspace camera (needed for zoom-trained checkpoints).")
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


# perturbation timing (50 Hz): empty close during the approach, drop after the lift, knock before the grasp
TIMING = {"none": (0, 0), "empty_close": (15, 15), "drop": (100, 20), "knock": (30, 1),
          "push_held": (110, 1), "arm_shove": (75, 12), "late_knock": (55, 1)}   # held-out kinds: never in training data
OUTCOMES = ["success", "never_lifted", "dropped", "held_off_goal", "fell_off_table"]


def main(args):
    import torch
    from fd_core.env import Franka, load_teacher, make_env
    from fd_core.policy import load_dp
    from fd_core.rollout import DiffusionActor, PerturbActor, TeacherActor, run_batch

    torch.manual_seed(args.seed)
    env, raw = make_env(args.num_envs, args.device, args.seed, 128, wrist=True, robust_teacher=args.robust_teacher, zoom=args.zoom)
    fr = Franka(raw)
    g = torch.Generator(device=args.device).manual_seed(args.seed)
    policies = [("teacher", TeacherActor(load_teacher(args.teacher, args.device), fr), 1)]
    for ck in args.checkpoint:
        m, cfg = load_dp(ck, args.device)
        policies.append((f"dp:{ck.parent.name}/{ck.stem}", DiffusionActor(m, cfg.exec_horizon), cfg.obs_horizon))
    results = {}
    n = args.num_envs
    for name, actor, oh in policies:
        for kind in args.kinds.split(","):
            k = PerturbActor.KINDS.index(kind); t0, dur = TIMING[kind]
            pa = PerturbActor(actor, fr, torch.full((n,), k, device=args.device), torch.full((n,), t0, device=args.device),
                              torch.full((n,), dur, device=args.device), generator=g)
            counts = torch.zeros(5, dtype=torch.long); done = 0
            while done < args.episodes:
                r = run_batch(env, raw, fr, pa, obs_horizon=oh)
                take = min(n, args.episodes - done)
                counts += torch.bincount(r["outcome"][:take], minlength=5); done += take
            row = {o: int(c) for o, c in zip(OUTCOMES, counts)}; row["episodes"] = done; row["success_rate"] = row["success"] / done
            results[f"{name} | {kind}"] = row
            print(f"[diag] {name:40s} {kind:12s} " + " ".join(f"{o}={row[o]}" for o in OUTCOMES), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"timing": TIMING, "seed": args.seed, "results": results}, indent=1))


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
