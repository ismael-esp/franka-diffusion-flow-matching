"""Noise-space RL (DSRL, Wagenmaker et al. 2025) on top of the frozen BC Diffusion Policy.

Per-step Gaussian noise on the executed joint targets ruins this task (X2 with a 10-step chain:
57% at std 0.01, 21% at 0.05, 1% at 0.1), so DPPO cannot explore without breaking the policy.
Here the diffusion policy stays frozen and deterministic (DDIM, eta=0); a small Gaussian actor
chooses the initial noise x_T instead. Every explored chunk is still a decoded, demo-like plan,
and an actor with mean 0 and std 1 is exactly the BC policy, so RL starts from its success rate.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from .dppo import TransformerCritic
from .model import DiffusionPolicy, PolicyConfig, ddim_timesteps

LOG_STD_MIN, LOG_STD_MAX = math.log(0.05), math.log(1.5)


class NoiseActor(nn.Module):
    """Gaussian over x_T [Tp,Da] from the frozen condition tokens; starts as N(0, I)."""

    def __init__(self, cfg: PolicyConfig, hidden: int = 512, init_std: float = 1.0):
        super().__init__()
        self.shape = (cfg.pred_horizon, cfg.action_dim)
        size = cfg.num_cond_tokens * cfg.d_model
        self.net = nn.Sequential(
            nn.LayerNorm(size), nn.Linear(size, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, self.shape[0] * self.shape[1]),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.log_std = nn.Parameter(torch.full(self.shape, math.log(init_std)))

    def forward(self, cond: torch.Tensor) -> Normal:
        mean = self.net(cond.flatten(1)).reshape(-1, *self.shape)
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mean)
        return Normal(mean, std)


class DSRLPolicy(nn.Module):
    def __init__(self, base: DiffusionPolicy, share_mean: torch.Tensor, share_std: torch.Tensor,
                 ddim_steps: int = 0, init_std: float = 1.0):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        if ddim_steps:
            base.cfg.num_inference_steps = ddim_steps
            base.ddim_steps = ddim_timesteps(base.cfg.num_train_timesteps, ddim_steps)
        self.actor = NoiseActor(base.cfg, init_std=init_std)
        self.critic = TransformerCritic(base.cfg.share_obs_dim, base.cfg.num_agents)
        self.register_buffer("share_mean", share_mean.clone())
        self.register_buffer("share_std", share_std.clone())

    @property
    def cfg(self) -> PolicyConfig:
        return self.base.cfg

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()  # the frozen policy never runs in training mode
        return self

    def value(self, share_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(((share_obs - self.share_mean) / self.share_std).clamp(-5.0, 5.0))

    @torch.no_grad()
    def decode(self, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Deterministic DDIM from x_T = noise; the same chain as DiffusionPolicy.sample."""
        x, b = noise, noise.shape[0]
        steps = self.base.ddim_steps
        for i, t in enumerate(steps):
            prev = steps[i + 1] if i + 1 < len(steps) else torch.tensor(-1, device=noise.device)
            t_batch = t.expand(b)
            x, _ = self.base.ddim_mean(x, self.base.denoiser_for(i)(x, t_batch, cond), t_batch, prev.expand(b), eta=0.0)
        return x.clamp(-1.0, 1.0)

    def log_prob(self, cond: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.actor(cond).log_prob(noise).sum(dim=(-1, -2))

    @torch.no_grad()
    def act(self, cond: torch.Tensor):
        """Rollout step: sampled noise, its log-probability and the decoded chunk."""
        dist = self.actor(cond)
        noise = dist.sample()
        return noise, dist.log_prob(noise).sum(dim=(-1, -2)), self.decode(noise, cond)

    @torch.no_grad()
    def sample(self, image: torch.Tensor, proprio: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Same interface as DiffusionPolicy.sample, for eval_policy and the hardware adapter."""
        cond = self.base.encode(image, proprio)
        dist = self.actor(cond)
        return self.decode(dist.mean if deterministic else dist.sample(), cond)

    def ppo_losses(self, batch: dict, clip_coef: float = 0.2):
        new_lp = self.log_prob(batch["cond"], batch["noise"])
        log_ratio = new_lp - batch["old_logprob"]
        ratio = log_ratio.exp()
        advantage = batch["advantage"]
        policy_loss = torch.max(-advantage * ratio, -advantage * ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef)).mean()
        value_loss = 0.5 * (self.value(batch["share"]) - batch["return"]).square().mean()
        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()
        return policy_loss, value_loss, {"approx_kl": approx_kl, "clip_fraction": clip_fraction}


def build_dsrl(payload: dict, device="cpu") -> DSRLPolicy:
    """DSRLPolicy from a train_dsrl.py checkpoint (or its BC base and fresh actor if it has none)."""
    cfg = PolicyConfig(**payload["config"])
    base = DiffusionPolicy(cfg, torch.zeros(cfg.proprio_dim), torch.ones(cfg.proprio_dim))
    base.load_state_dict(payload["model"])
    policy = DSRLPolicy(base, payload["share_mean"], payload["share_std"])
    if "actor" in payload:
        policy.actor.load_state_dict(payload["actor"])
        policy.critic.load_state_dict(payload["critic"])
    return policy.to(device).eval()
