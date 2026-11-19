#!/usr/bin/env python3
"""Train a distillable privileged PPO teacher for a fd_core.tasks task (no last-action obs, obs noise, randomized
object poses and arm starts, 30 Hz). Stock Isaac Lab PPO hyper-parameters for that task."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--task", default="drawer")
p.add_argument("--num-envs", type=int, default=4096)
p.add_argument("--iterations", type=int, default=1500)
p.add_argument("--seed", type=int, default=43)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fd_core.tasks import TASKS, task_cfg  # noqa: E402

cfg = task_cfg(args.task, args.num_envs, args.device, args.seed, teacher_train=True)
env = RslRlVecEnvWrapper(gym.make(TASKS[args.task].gym_id, cfg=cfg), clip_actions=None)
if args.task == "drawer":
    from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.agents.rsl_rl_ppo_cfg import CabinetPPORunnerCfg as Runner
else:
    raise SystemExit(f"no PPO config for {args.task}")
agent = Runner()
agent.max_iterations = args.iterations; agent.seed = args.seed; agent.device = args.device
agent.experiment_name = f"franka_{args.task}"
log_dir = os.path.join("logs/rsl_rl", agent.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=log_dir, device=agent.device)
runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=True)
print("[teacher] done ->", log_dir, flush=True)
os._exit(0)
