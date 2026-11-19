"""Run chunked policies on a vectorized env whose episodes end at different times."""
from __future__ import annotations

import torch


def info_success(info) -> bool:
    return bool(info) and bool(info.get("success", False) or info.get("done_success", False))


class ChunkStepper:
    """Per-env action chunk and observation history.

    An env that resets mid-chunk replans at once: actions are absolute joint targets,
    so the rest of an old chunk would yank the arms after the reset.
    """

    def __init__(self, env, obs_horizon: int, exec_horizon: int, device, get_image):
        self.env = env
        self.exec_horizon = exec_horizon
        self.device = device
        self.get_image = get_image
        obs, share, obs_dict = env.reset()
        self.obs_history = obs.to(device)[:, None].repeat(1, obs_horizon, 1, 1)
        self.image_history = get_image(obs_dict).to(device)[:, None].repeat(1, obs_horizon, 1, 1, 1)
        self.share = share.to(device)
        n = obs.shape[0]
        self.num_envs = n
        self.cursor = torch.zeros(n, dtype=torch.long, device=device)
        self.plan = None
        self.step_count = 0

    def needs_plan(self) -> torch.Tensor:
        return self.cursor == 0

    def set_plan(self, mask: torch.Tensor, actions: torch.Tensor) -> None:
        if self.plan is None:
            self.plan = torch.zeros(self.num_envs, *actions.shape[1:], device=self.device)
        self.plan[mask] = actions

    def step(self):
        """Apply one action; returns reward [N], done [N], chunk_closed [N] and infos."""
        rows = torch.arange(self.num_envs, device=self.device)
        action = self.plan[rows, self.cursor]
        half = action.shape[-1] // 2
        obs, _, share, reward, done, infos, obs_dict = self.env.step(
            torch.stack([action[:, :half], action[:, half:]], dim=0), self.step_count
        )
        self.step_count += 1
        obs, share = obs.to(self.device), share.to(self.device)
        done = done[:, 0].bool().to(self.device)
        self.obs_history = torch.cat([self.obs_history[:, 1:], obs[:, None]], dim=1)
        self.obs_history[done] = obs[done][:, None]
        image = self.get_image(obs_dict).to(self.device)
        self.image_history = torch.cat([self.image_history[:, 1:], image[:, None]], dim=1)
        self.image_history[done] = image[done][:, None]
        self.share = share
        self.cursor += 1
        closed = done | (self.cursor >= self.exec_horizon)
        self.cursor[closed] = 0
        return reward[:, 0, 0].to(self.device), done, closed, infos
