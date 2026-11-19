"""Transformer diffusion policy on right-camera RGB and per-arm joint tokens.

Observations become one image token plus one token per arm at each history step; the
noisy chunk gives one token per (action step, arm). Dropout is deliberately absent
because PPO needs log-probabilities to be repeatable.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


@dataclass
class PolicyConfig:
    proprio_dim: int = 7
    share_obs_dim: int = 0
    agent_action_dim: int = 7
    num_agents: int = 2
    obs_horizon: int = 2
    pred_horizon: int = 32
    exec_horizon: int = 16
    d_model: int = 256
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 4
    num_keypoints: int = 32
    num_train_timesteps: int = 100
    num_inference_steps: int = 5

    @property
    def action_dim(self) -> int:
        return self.agent_action_dim * self.num_agents

    @property
    def num_cond_tokens(self) -> int:
        return self.obs_horizon * (1 + self.num_agents)

    def to_dict(self) -> dict:
        return asdict(self)


def cosine_alphas_cumprod(num_steps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """The squared-cosine schedule used by Diffusion Policy."""

    def alpha_bar(t: float) -> float:
        return math.cos((t + s) / (1 + s) * math.pi / 2) ** 2

    betas = [min(1 - alpha_bar((i + 1) / num_steps) / alpha_bar(i / num_steps), max_beta) for i in range(num_steps)]
    return torch.cumprod(1.0 - torch.tensor(betas, dtype=torch.float32), dim=0)


def ddim_timesteps(num_train: int, num_inference: int) -> torch.Tensor:
    """Descending DDIM steps ending at 0, e.g. 100/5 -> [80, 60, 40, 20, 0]."""
    return torch.arange(num_inference - 1, -1, -1) * (num_train // num_inference)


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1))
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ImageEncoder(nn.Module):
    """ResNet18 with GroupNorm and spatial softmax, the standard Diffusion Policy encoder."""

    def __init__(self, output_dim: int, num_keypoints: int):
        super().__init__()
        net = torchvision.models.resnet18(weights=None, norm_layer=lambda c: nn.GroupNorm(max(1, c // 16), c))
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        self.keypoints = nn.Conv2d(512, num_keypoints, 1)
        self.proj = nn.Linear(2 * num_keypoints, output_dim)
        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406])[:, None, None])
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225])[:, None, None])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        heat = self.keypoints(self.backbone((image - self.pixel_mean) / self.pixel_std))
        b, k, h, w = heat.shape
        prob = F.softmax(heat.reshape(b, k, -1), dim=-1).reshape(b, k, h, w)
        ys = torch.linspace(-1.0, 1.0, h, device=heat.device)
        xs = torch.linspace(-1.0, 1.0, w, device=heat.device)
        x = (prob.sum(2) * xs).sum(-1)
        y = (prob.sum(3) * ys).sum(-1)
        return self.proj(torch.cat([x, y], dim=-1))


class TransformerDenoiser(nn.Module):
    """Predicts the noise of a chunk [B,Tp,14] from condition tokens [B,Nc,d]."""

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.action_proj = nn.Linear(cfg.agent_action_dim, d)
        self.agent_emb = nn.Parameter(torch.zeros(1, cfg.num_agents, d))
        self.action_time_emb = nn.Parameter(torch.zeros(1, cfg.pred_horizon, d))
        self.step_mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.Mish(), nn.Linear(4 * d, d))
        common = dict(
            d_model=d, nhead=cfg.num_heads, dim_feedforward=4 * d, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**common), cfg.num_encoder_layers, enable_nested_tensor=False
        )
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(**common), cfg.num_decoder_layers)
        self.final_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.agent_action_dim)
        nn.init.normal_(self.agent_emb, std=0.02)
        nn.init.normal_(self.action_time_emb, std=0.02)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        b, tp = sample.shape[:2]
        step_token = self.step_mlp(sinusoidal_embedding(timestep, cfg.d_model))[:, None]
        memory = self.encoder(torch.cat([step_token, cond], dim=1))
        per_agent = sample.reshape(b, tp, cfg.num_agents, cfg.agent_action_dim)
        tokens = self.action_proj(per_agent) + self.action_time_emb[:, :, None] + self.agent_emb[:, None]
        output = self.head(self.final_norm(self.decoder(tokens.reshape(b, tp * cfg.num_agents, cfg.d_model), memory)))
        return output.reshape(b, tp, cfg.action_dim)


class DiffusionPolicy(nn.Module):
    """Observation encoding, DDPM training loss and DDIM sampling around a denoiser."""

    def __init__(self, cfg: PolicyConfig, proprio_mean: torch.Tensor, proprio_std: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.image_encoder = ImageEncoder(d, cfg.num_keypoints)
        self.proprio_proj = nn.Linear(cfg.proprio_dim, d)
        self.obs_time_emb = nn.Parameter(torch.zeros(1, cfg.obs_horizon, d))
        self.obs_kind_emb = nn.Parameter(torch.zeros(1, 1 + cfg.num_agents, d))  # image, then each arm
        nn.init.normal_(self.obs_time_emb, std=0.02)
        nn.init.normal_(self.obs_kind_emb, std=0.02)
        self.denoiser = TransformerDenoiser(cfg)
        self.early_denoiser = None  # frozen copy for the steps before fine-tuning starts
        self.first_finetune = 0
        self.register_buffer("proprio_mean", proprio_mean.clone())
        self.register_buffer("proprio_std", proprio_std.clone())
        self.register_buffer("alphas_cumprod", cosine_alphas_cumprod(cfg.num_train_timesteps))
        self.register_buffer("ddim_steps", ddim_timesteps(cfg.num_train_timesteps, cfg.num_inference_steps))

    def freeze_early_steps(self, finetune_steps: int) -> None:
        """Keep a frozen copy of the denoiser for all but the last finetune_steps DDIM steps."""
        self.first_finetune = len(self.ddim_steps) - finetune_steps
        self.early_denoiser = copy.deepcopy(self.denoiser).requires_grad_(False)

    def denoiser_for(self, step: int) -> TransformerDenoiser:
        return self.early_denoiser if step < self.first_finetune else self.denoiser

    def encode(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Condition tokens [B,To*3,d] from images [B,To,3,H,W] and joints [B,To,2,7]."""
        b, t = image.shape[:2]
        image_token = self.image_encoder(image.flatten(0, 1)).reshape(b, t, 1, -1)
        normalized = ((proprio - self.proprio_mean) / self.proprio_std).clamp(-5.0, 5.0)
        tokens = torch.cat([image_token, self.proprio_proj(normalized)], dim=2)
        tokens = tokens + self.obs_time_emb[:, :, None] + self.obs_kind_emb[:, None]
        return tokens.flatten(1, 2)

    def bc_loss(self, image, proprio, action, generator=None) -> torch.Tensor:
        noise = torch.randn(action.shape, device=action.device, generator=generator)
        t = torch.randint(self.cfg.num_train_timesteps, (action.shape[0],), device=action.device, generator=generator)
        alpha = self.alphas_cumprod[t][:, None, None]
        noisy = alpha.sqrt() * action + (1.0 - alpha).sqrt() * noise
        return F.mse_loss(self.denoiser(noisy, t, self.encode(image, proprio)), noise)

    def alpha_at(self, t: torch.Tensor) -> torch.Tensor:
        """Cumulative alpha [B,1,1]; DDIM's step past t=0 (t=-1) has alpha 1."""
        alpha = torch.where(t < 0, torch.ones_like(t, dtype=torch.float32),
                            self.alphas_cumprod[t.clamp(min=0)].to(torch.float32))
        return alpha[:, None, None]

    def ddim_mean(self, sample, eps, t, t_prev, eta: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and schedule std of one DDIM transition, with x0 clipped to the action range."""
        alpha, alpha_prev = self.alpha_at(t), self.alpha_at(t_prev)
        x0 = ((sample - (1.0 - alpha).sqrt() * eps) / alpha.sqrt()).clamp(-1.0, 1.0)
        variance = (1.0 - alpha_prev) / (1.0 - alpha) * (1.0 - alpha / alpha_prev)
        std = eta * variance.clamp(min=0.0).sqrt()
        direction = (1.0 - alpha_prev - std.square()).clamp(min=0.0).sqrt()
        return alpha_prev.sqrt() * x0 + direction * eps, std

    @torch.no_grad()
    def sample(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Deterministic DDIM (eta=0) chunk [B,Tp,14] in the env's action range."""
        b = image.shape[0]
        cond = self.encode(image, proprio)
        x = torch.randn(b, self.cfg.pred_horizon, self.cfg.action_dim, device=image.device)
        for i, t in enumerate(self.ddim_steps):
            prev = self.ddim_steps[i + 1] if i + 1 < len(self.ddim_steps) else torch.tensor(-1, device=image.device)
            t_batch = t.expand(b)
            x, _ = self.ddim_mean(x, self.denoiser_for(i)(x, t_batch, cond), t_batch, prev.expand(b), eta=0.0)
        return x.clamp(-1.0, 1.0)
