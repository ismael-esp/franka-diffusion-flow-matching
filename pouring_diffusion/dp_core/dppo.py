"""Diffusion PPO (Ren et al. 2024) on top of the BC transformer denoiser.

Each DDIM transition is a Gaussian policy step, so a chunk's log-probability is a sum
over denoising steps and PPO can fine-tune the denoiser directly. Objective constants
follow the official irom-lab/dppo configs. Only the last denoising steps are fine-tuned
and earlier ones use a frozen copy of the BC denoiser, because a policy this precise
degrades when small mean shifts compound through the whole chain. The image and joint
encoder stays frozen, so rollouts only need to store its condition tokens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Normal

from .model import DiffusionPolicy


LOGPROB_MIN, LOGPROB_MAX = -5.0, 2.0


@dataclass
class ChainSample:
    actions: torch.Tensor  # [B,Tp,Da]
    chain: torch.Tensor  # [B,K+1,Tp,Da]
    logprobs: torch.Tensor  # [B,K,Tp,Da]


class TransformerCritic(nn.Module):
    """Centralized value from the privileged per-agent observation tokens [B,2,Ds]."""

    def __init__(self, share_obs_dim: int, num_agents: int = 2, d_model: int = 128, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(share_obs_dim, d_model)
        self.agent_emb = nn.Parameter(torch.zeros(1, num_agents, d_model))
        nn.init.normal_(self.agent_emb, std=0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=4 * d_model, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True,
            ),
            num_layers,
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, share_obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.proj(share_obs) + self.agent_emb).mean(dim=1)).squeeze(-1)


class DPPOPolicy(nn.Module):
    def __init__(
        self,
        policy: DiffusionPolicy,
        share_mean: torch.Tensor,
        share_std: torch.Tensor,
        *,
        eta: float = 1.0,
        min_sampling_std: float = 0.05,
        min_logprob_std: float = 0.05,
        noise_clip: float = 3.0,
        finetune_steps: int = 0,
    ):
        super().__init__()
        self.policy = policy
        for module in (policy.image_encoder, policy.proprio_proj):
            module.requires_grad_(False)
        policy.obs_time_emb.requires_grad_(False)
        policy.obs_kind_emb.requires_grad_(False)
        self.critic = TransformerCritic(policy.cfg.share_obs_dim, policy.cfg.num_agents)
        self.register_buffer("share_mean", share_mean.clone())
        self.register_buffer("share_std", share_std.clone())
        self.eta = eta
        self.min_sampling_std = min_sampling_std
        self.min_logprob_std = min_logprob_std
        self.noise_clip = noise_clip
        if 0 < finetune_steps < self.num_denoise:
            policy.freeze_early_steps(finetune_steps)
        self.num_finetune = self.num_denoise - policy.first_finetune

    @property
    def num_denoise(self) -> int:
        return len(self.policy.ddim_steps)

    def value(self, share_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(((share_obs - self.share_mean) / self.share_std).clamp(-5.0, 5.0))

    def _prev_timesteps(self, index: torch.Tensor) -> torch.Tensor:
        steps = self.policy.ddim_steps
        extended = torch.cat([steps, torch.tensor([-1], device=steps.device, dtype=steps.dtype)])
        return extended[index + 1]

    def _transition(self, x: torch.Tensor, index: torch.Tensor, cond: torch.Tensor):
        t = self.policy.ddim_steps[index]
        eps = self.policy.denoiser_for(int(index.min()))(x, t, cond)
        mean, std = self.policy.ddim_mean(x, eps, t, self._prev_timesteps(index), eta=self.eta)
        return mean, std

    @torch.no_grad()
    def sample(self, cond: torch.Tensor) -> ChainSample:
        """Stochastic chain whose last log-probability refers to the clamped action."""
        cfg = self.policy.cfg
        b = cond.shape[0]
        x = torch.randn(b, cfg.pred_horizon, cfg.action_dim, device=cond.device)
        chain, logprobs = [x], []
        for i in range(self.num_denoise):
            index = torch.full((b,), i, device=cond.device, dtype=torch.long)
            mean, std = self._transition(x, index, cond)
            noise = torch.randn_like(x).clamp(-self.noise_clip, self.noise_clip)
            x = mean + std.clamp(min=self.min_sampling_std) * noise
            if i == self.num_denoise - 1:
                x = x.clamp(-1.0, 1.0)
            logprobs.append(Normal(mean, std.clamp(min=self.min_logprob_std)).log_prob(x))
            chain.append(x)
        return ChainSample(x, torch.stack(chain, dim=1), torch.stack(logprobs, dim=1))

    def logprobs(self, cond, prev: torch.Tensor, nxt: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        mean, std = self._transition(prev, index, cond)
        return Normal(mean, std.clamp(min=self.min_logprob_std)).log_prob(nxt)

    def _reduce_logprob(self, logprob: torch.Tensor) -> torch.Tensor:
        """Mean over the executed part of the chunk, as in the official objective."""
        return logprob[:, : self.policy.cfg.exec_horizon].clamp(LOGPROB_MIN, LOGPROB_MAX).mean(dim=(-1, -2))

    def clip_coef(self, index: torch.Tensor, coef: float, base: float, rate: float) -> torch.Tensor:
        """Clipping tightens toward the early, more noise-like denoising steps."""
        if self.num_denoise == 1:
            return torch.full_like(index, coef, dtype=torch.float32)
        frac = index.float() / (self.num_denoise - 1)
        return base + (coef - base) * (torch.exp(rate * frac) - 1.0) / (math.exp(rate) - 1.0)

    def ppo_losses(self, batch: dict, *, gamma_denoising=0.99, clip_coef=0.01, clip_base=0.001, clip_rate=3.0):
        """Clipped objective; advantages must already be normalized over the whole rollout."""
        index = batch["index"]
        rows = torch.arange(index.shape[0], device=index.device)
        chain = batch["chain"]
        new_lp = self._reduce_logprob(self.logprobs(batch["cond"], chain[rows, index], chain[rows, index + 1], index))
        old_lp = self._reduce_logprob(batch["old_logprobs"][rows, index])
        log_ratio = new_lp - old_lp
        ratio = log_ratio.exp()

        advantage = batch["advantage"] * gamma_denoising ** (self.num_denoise - index.float() - 1.0)
        clip = self.clip_coef(index, clip_coef, clip_base, clip_rate)
        policy_loss = torch.max(-advantage * ratio, -advantage * ratio.clamp(1.0 - clip, 1.0 + clip)).mean()
        value_loss = 0.5 * (self.value(batch["share"]) - batch["return"]).square().mean()

        with torch.no_grad():
            element_kl = ((ratio - 1.0) - log_ratio).mean()
            # log_ratio is averaged over chunk elements, but a decision's log-ratio is their
            # sum, and KL is quadratic in it: rescale before squaring, then add the steps.
            cfg = self.policy.cfg
            joint_kl = 0.5 * (log_ratio * cfg.exec_horizon * cfg.action_dim).square().mean() * self.num_finetune
        return policy_loss, value_loss, {"element_kl": element_kl, "joint_kl": joint_kl,
                                         "clip_fraction": ((ratio - 1.0).abs() > clip).float().mean()}


def gae(rewards, values, dones, env_ids, last_values, gamma: float, lam: float):
    """GAE over decisions ordered in time inside each env.

    `last_values[env]` bootstraps the decision still open when the rollout stopped.
    Done decisions are terminal: the task horizon is a fixed end of the episode.
    """
    advantages = torch.zeros_like(rewards)
    for env in env_ids.unique():
        indices = (env_ids == env).nonzero().flatten()
        next_value, next_advantage = last_values[env], 0.0
        for i in reversed(indices.tolist()):
            nonterminal = 1.0 - float(dones[i])
            delta = rewards[i] + gamma * next_value * nonterminal - values[i]
            advantages[i] = next_advantage = delta + gamma * lam * nonterminal * next_advantage
            next_value = values[i]
    return advantages, advantages + values
