#!/usr/bin/env python3
"""Train a *distillable* PPO teacher for Franka lift (fix for DAgger: the original teacher labels off-distribution states badly).

Changes vs Isaac-Lift-Cube-Franka-v0 (DextrAH-style robust teacher):
- no last-action observation (the teacher must not depend on its own previous command, which the student overrides),
- observation noise on joint pos/vel and the cube position,
- random arm start poses (+-0.15 rad per joint),
- random cube pushes every 1-3 s (also while held), so the teacher learns to re-approach and re-grasp after knocks/drops.
Same PPO hyper-parameters as the original teacher."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--num-envs", type=int, default=4096)
p.add_argument("--iterations", type=int, default=2000)
p.add_argument("--seed", type=int, default=43)
p.add_argument("--log-root", default="logs/rsl_rl/franka_lift_robust")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab.envs import mdp as base_mdp  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise  # noqa: E402
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg import LiftCubePPORunnerCfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fd_core.env import robust_teacher_cfg  # noqa: E402

cfg = robust_teacher_cfg(args.num_envs, args.device, args.seed, train=True)
env = RslRlVecEnvWrapper(gym.make("Isaac-Lift-Cube-Franka-v0", cfg=cfg), clip_actions=None)
agent = LiftCubePPORunnerCfg()
agent.max_iterations = args.iterations; agent.seed = args.seed; agent.device = args.device
agent.experiment_name = "franka_lift_robust"
log_dir = os.path.join(args.log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=log_dir, device=agent.device)
runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=True)
print("[robust-teacher] done ->", log_dir, flush=True)
os._exit(0)
