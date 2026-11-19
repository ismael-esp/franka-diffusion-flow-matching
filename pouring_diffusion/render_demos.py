#!/usr/bin/env python3
"""Add right-camera images to the oracle demos by replaying their commands in Isaac Lab.

Each episode restarts from its recorded object spawn and replays its absolute commands.
Physics can drift from the original run, so only replays that still succeed are kept.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="New dataset folder; an existing one is refused.")
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--max-episodes", type=int, default=0, help="Replay only the first N demos (0 = all).")
    p.add_argument("--seed", type=int, default=9)
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def install_spawn_hook(raw_env):
    """After every reset, move the objects to the poses queued in raw_env.replay_spawn."""
    import torch

    original = raw_env._reset_idx
    names = ("cup", "bottle")

    def reset_idx(env_ids):
        original(env_ids)
        spawn = getattr(raw_env, "replay_spawn", None)
        if spawn is None:
            return
        ids = torch.as_tensor(env_ids, device=raw_env.device, dtype=torch.long)
        for slot, name in enumerate(names):
            obj = raw_env._rigid_objects[name]
            rot = raw_env.env_origins.new_tensor(raw_env.cfg.rigid_objects_cfg[name].init_state.rot).repeat(len(ids), 1)
            world = spawn[ids, slot] + raw_env.env_origins[ids]
            obj.write_root_pose_to_sim(torch.cat([world, rot], dim=-1), ids)
            getattr(raw_env, f"initial_{name}_pos")[ids] = world

    raw_env._reset_idx = reset_idx


def main(args):
    import numpy as np
    import torch

    from dp_core.env import DEMO_PATH, IMAGE_HEIGHT, IMAGE_WIDTH, TRAJ_PATH, build_pouring_env, right_image

    device = torch.device(args.device)
    args.output.mkdir(parents=True)
    with open(DEMO_PATH, "rb") as handle:
        demo = pickle.load(handle)[0]
    with open(TRAJ_PATH, "rb") as handle:
        spawn_all = pickle.load(handle)[0]["init_cube_pos"]
    actions = np.stack(demo["actions"], axis=1)  # [E,T,2,7]
    episodes = actions.shape[0] if args.max_episodes <= 0 else min(args.max_episodes, actions.shape[0])
    horizon = actions.shape[1]

    env, raw_env = build_pouring_env(args.num_envs, device, args.seed, args.output / "env", stage="replay")
    install_spawn_hook(raw_env)
    cup_x, bottle_x = (raw_env.cfg.rigid_objects_cfg[n].init_state.pos[0] for n in ("cup", "bottle"))
    # Demo_Traj stores the two spawns without names, so match them to the nominal positions.
    first_is_cup = abs(float(spawn_all[:, 0, 0].mean()) - cup_x) < abs(float(spawn_all[:, 0, 0].mean()) - bottle_x)
    spawn_ordered = spawn_all if first_is_cup else spawn_all[:, ::-1]

    images = np.lib.format.open_memmap(
        args.output / "images.npy", mode="w+", dtype=np.uint8, shape=(episodes, horizon, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    )
    proprio, share, kept = None, None, np.zeros(episodes, dtype=bool)
    for start in range(0, episodes, args.num_envs):
        batch = np.arange(start, min(start + args.num_envs, episodes))
        # Spare envs repeat the last demo and are ignored, keeping every env in step.
        index = np.resize(batch, args.num_envs)
        raw_env.replay_spawn = torch.as_tensor(spawn_ordered[index].copy(), device=device, dtype=torch.float32)
        obs, state, obs_dict = env.reset()
        succeeded = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
        terminated = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
        for t in range(horizon):
            if proprio is None:
                proprio = np.zeros((episodes, horizon, *obs.shape[1:]), dtype=np.float32)
                share = np.zeros((episodes, horizon, *state.shape[1:]), dtype=np.float32)
            frame = right_image(obs_dict)
            if float(frame.max()) > 1.01:
                raise RuntimeError(f"expected images in [0,1], got max {float(frame.max())}")
            pixels = (frame * 255.0).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            images[batch, t] = pixels[: len(batch)]
            proprio[batch, t] = obs[: len(batch)].cpu().numpy()
            share[batch, t] = state[: len(batch)].cpu().numpy()
            act = torch.as_tensor(actions[index, t], device=device, dtype=torch.float32).permute(1, 0, 2)
            obs, _, state, _, done, infos, obs_dict = env.step(act, t)
            if t < horizon - 1:
                terminated |= done[:, 0].bool()
            succeeded |= raw_env.success_buf.bool()
        # A guard stop before the horizon means the frames after it show a different episode.
        kept[batch] = (succeeded & ~terminated)[: len(batch)].cpu().numpy()
        print(f"[render] episodes {batch[0]}..{batch[-1]} kept so far {int(kept.sum())}/{batch[-1] + 1}", flush=True)
    images.flush()

    np.save(args.output / "proprio.npy", proprio)
    np.save(args.output / "share.npy", share)
    np.save(args.output / "action.npy", actions[: episodes].reshape(episodes, horizon, -1))
    np.save(args.output / "kept.npy", kept)
    meta = {"episodes": int(episodes), "kept": int(kept.sum()), "horizon": int(horizon), "demo": str(DEMO_PATH)}
    (args.output / "meta.json").write_text(json.dumps(meta, indent=1))
    print("[render] done " + json.dumps(meta), flush=True)
    raw_env.close()


if __name__ == "__main__":
    cli = parse_args()
    if cli.output.exists():
        sys.exit(f"{cli.output} already exists; choose a new --output")
    cli.output = cli.output.resolve()
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
