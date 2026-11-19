"""Flow-matching variant of DiffusionPolicy: same encoder, same transformer, same data and checkpoint format.

Training (rectified flow / conditional flow matching): x_t = (1 - t) * noise + t * action with t ~ U(0, 1); the
network predicts the velocity v = action - noise (the timestep input is t * num_train_timesteps, so the sinusoidal
embedding sees the same range as the diffusion model). Sampling: Euler integration from noise (t = 0) to the action
(t = 1) with len(ddim_steps) steps, so every existing "--ddim-steps" flag sets the number of flow steps.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import DiffusionPolicy


class FlowPolicy(DiffusionPolicy):
    objective = "flow"

    def bc_loss(self, image, proprio, action, generator=None) -> torch.Tensor:
        noise = torch.randn(action.shape, device=action.device, generator=generator)
        t = torch.rand((action.shape[0],), device=action.device, generator=generator)
        x_t = (1.0 - t)[:, None, None] * noise + t[:, None, None] * action
        v = self.denoiser(x_t, t * self.cfg.num_train_timesteps, self.encode(image, proprio))
        return F.mse_loss(v, action - noise)

    @torch.no_grad()
    def sample(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        b = image.shape[0]
        cond = self.encode(image, proprio)
        x = torch.randn(b, self.cfg.pred_horizon, self.cfg.action_dim, device=image.device)
        k = len(self.ddim_steps)
        for i in range(k):
            t = torch.full((b,), i / k, device=image.device)
            x = x + self.denoiser_for(i)(x, t * self.cfg.num_train_timesteps, cond) / k
        return x.clamp(-1.0, 1.0)
