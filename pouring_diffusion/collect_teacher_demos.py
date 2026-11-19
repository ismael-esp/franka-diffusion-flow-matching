#!/usr/bin/env python3
"""Roll out the released privileged teacher under the lagged arm and record right-camera demos.

Writes the render_demos.py dataset format. Only episodes that succeed and are not stopped by
the strict contact guard are kept. With --noise-std the arms execute a noisy action while the
clean teacher action is recorded (DART), so the demos also show how to recover.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="New dataset folder; an existing one is refused.")
    p.add_argument("--target", type=int, default=300, help="Successful episodes to keep.")
    p.add_argument("--max-batches", type=int, default=60)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--noise-std", type=float, default=0.0, help="Executed arm noise in normalized action units.")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--stage", default="collect_lag")
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def teacher_args() -> SimpleNamespace:
    """Architecture recorded in release/pouring/students/*/command.json (MAT, no temporal context)."""
    return SimpleNamespace(
        algorithm_name="mat", lr=1e-4, opti_eps=1e-5, weight_decay=0.0, use_policy_active_masks=False,
        n_block=2, n_embd=128, n_head=4, encode_state=True, dec_actor=False, share_actor=False,
        critic_n_block_separate=1, state_dependent_std=False, use_vision=False, vision_obs_dim=0, vision_feat_dim=64,
        vision_fusion="concat", vision_dropout=0.0, vision_critic_use_global=False, vision_mode="spatial_softmax",
        vision_resize_h=8, vision_resize_w=8, vision_channels=3, vision_frame_stack=1, vision_spatial_num_kp=32,
        vision_split_stem=True, vision_token_patch_size=4, vision_token_cross_layers=1, vision_resnet_pretrained=True,
        use_recurrent_policy=False, use_naive_recurrent_policy=False, recurrent_N=1, hidden_size=64,
        mat_temporal_mode="none", mat_temporal_context_len=8, mat_temporal_n_layer=1, mat_temporal_dropout=0.0,
        mat_entity_token_mode="none", mat_entity_n_layer=2, mat_entity_dropout=0.0, min_std=0.0, action_std_scale=0.05,
        max_std=2.0, action_std_floor=0.0, action_chunk_len=1, action_chunk_exec_mode="ensemble",
    )


def load_teacher(path: Path, device):
    import gymnasium as gym
    import numpy as np

    from mat.policy.transformer_policy import TransformerPolicy

    state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(49,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    policy = TransformerPolicy(args=teacher_args(), obs_space=state_space, cent_obs_space=state_space,
                               act_space=action_space, num_agents=2, device=device)
    policy.restore(str(path))
    policy.eval()
    return policy


def teacher_mean(policy, obs_dict, num_envs: int):
    """Deterministic teacher action [N,2,7] from the privileged per-agent state."""
    import numpy as np
    import torch

    state = obs_dict["teacher_policy"].reshape(num_envs * 2, -1).detach().cpu().numpy()
    rnn = np.zeros((num_envs * 2, 1, 64), dtype=np.float32)
    masks = np.ones((num_envs * 2, 1), dtype=np.float32)
    _, mean, _, _, _ = policy.get_actions(cent_obs=state, obs=state, rnn_states_actor=rnn, rnn_states_critic=rnn.copy(),
                                          masks=masks, available_actions=None, deterministic=True)
    return torch.as_tensor(mean).float().reshape(num_envs, 2, 7).clamp(-1.0, 1.0)


def main(args):
    import numpy as np
    import torch

    from dp_core.env import IMAGE_HEIGHT, IMAGE_WIDTH, RELEASE, build_pouring_env, right_image

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True)
    env, raw_env = build_pouring_env(args.num_envs, device, args.seed, args.output / "env", stage=args.stage)
    teacher = load_teacher(RELEASE / "teacher" / "transformer_534.pt", device)
    horizon = int(raw_env.max_episode_length)
    images = np.lib.format.open_memmap(args.output / "images.npy", mode="w+", dtype=np.uint8,
                                       shape=(args.target, horizon, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    proprio, share, action = None, None, np.zeros((args.target, horizon, 14), dtype=np.float32)
    count, attempts, successes, stopped = 0, 0, 0, 0
    spawns = []
    for batch in range(args.max_batches):
        obs, state, obs_dict = env.reset()
        spawns.append(raw_env.initial_cup_pos.sub(raw_env.env_origins).cpu().numpy())
        frames = np.zeros((horizon, args.num_envs, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        obs_seq, state_seq, act_seq = [], [], []
        succeeded = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
        terminated = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
        for t in range(horizon):
            with torch.no_grad():
                clean = teacher_mean(teacher, obs_dict, args.num_envs).to(device)
            noisy = clean.clone()
            if args.noise_std > 0:
                noisy[..., :6] = (noisy[..., :6] + args.noise_std * torch.randn_like(noisy[..., :6])).clamp(-1.0, 1.0)
            frames[t] = (right_image(obs_dict) * 255.0).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            obs_seq.append(obs.cpu().numpy())
            state_seq.append(state.cpu().numpy())
            act_seq.append(clean.reshape(args.num_envs, 14).cpu().numpy())
            obs, _, state, _, done, _, obs_dict = env.step(noisy.permute(1, 0, 2), t)
            if t < horizon - 1:
                terminated |= done[:, 0].bool()
            succeeded |= raw_env.success_buf.bool()
        good = (succeeded & ~terminated).cpu().numpy()
        attempts += args.num_envs
        successes += int(succeeded.sum())
        stopped += int(terminated.sum())
        obs_seq, state_seq, act_seq = np.stack(obs_seq, 1), np.stack(state_seq, 1), np.stack(act_seq, 1)
        if proprio is None:
            proprio = np.zeros((args.target, *obs_seq.shape[1:]), dtype=np.float32)
            share = np.zeros((args.target, *state_seq.shape[1:]), dtype=np.float32)
        for env_id in np.flatnonzero(good)[: args.target - count]:
            images[count] = frames[:, env_id]
            proprio[count], share[count], action[count] = obs_seq[env_id], state_seq[env_id], act_seq[env_id]
            count += 1
        print(f"[collect] batch={batch} kept={count}/{args.target} attempts={attempts} "
              f"success={successes / attempts:.3f} guard_stopped={stopped / attempts:.3f}", flush=True)
        if count >= args.target:
            break
    images.flush()
    kept = np.zeros(args.target, dtype=bool)
    kept[: count] = True
    np.save(args.output / "proprio.npy", proprio)
    np.save(args.output / "share.npy", share)
    np.save(args.output / "action.npy", action)
    np.save(args.output / "kept.npy", kept)
    spawn_std = np.concatenate(spawns, 0).std(0).round(4).tolist()
    meta = {"episodes": args.target, "kept": count, "horizon": horizon, "attempts": attempts, "successes": successes,
            "guard_stopped": stopped, "noise_std": args.noise_std, "stage": args.stage, "cup_spawn_std": spawn_std,
            "teacher": "release/pouring/teacher/transformer_534.pt"}
    (args.output / "meta.json").write_text(json.dumps(meta, indent=1))
    print("[collect] done " + json.dumps(meta), flush=True)
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
