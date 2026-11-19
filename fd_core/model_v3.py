"""v3 student: pouring_diffusion's transformer Diffusion Policy (dp_core, unmodified) with three literature-backed changes.

1. One ResNet18 (GroupNorm + spatial softmax) per camera instead of one encoder on the side-by-side image
   (Diffusion Policy, Chi et al. 2023, Sec. 3.2: separate encoders per view).
2. Joint velocities dropped from the student's proprio; joint positions + goal kept
   (robomimic, Mandlekar et al. 2021: adding velocity features hurt image agents, Square 82% -> 58%).
3. Auxiliary head predicting the cube position from the two image features of the last observation step,
   loss weight 0.1 (DextrAH-G/-RGB: L = L_action + 0.1 L_pos), aimed at the few-cm grasp misses.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

DP_ROOT = os.environ.get("DP_ROOT", str(Path(__file__).resolve().parents[1] / "pouring_diffusion"))
if DP_ROOT not in sys.path:
    sys.path.append(DP_ROOT)
from dp_core.model import DiffusionPolicy, ImageEncoder, PolicyConfig  # noqa: E402

PROPRIO_KEEP = list(range(9)) + [18, 19, 20]     # 21-d env proprio -> joint pos (9) + goal (3)


class DiffusionPolicyV3(DiffusionPolicy):
    def __init__(self, cfg: PolicyConfig, proprio_mean, proprio_std, cube_mean=None, cube_std=None, aux_weight=0.1):
        assert cfg.proprio_dim == len(PROPRIO_KEEP)
        super().__init__(cfg, proprio_mean, proprio_std)          # self.image_encoder = workspace camera
        d = cfg.d_model
        self.wrist_encoder = ImageEncoder(d, cfg.num_keypoints)
        self.obs_kind_emb = nn.Parameter(torch.zeros(1, 3, d))    # workspace, wrist, proprio
        nn.init.normal_(self.obs_kind_emb, std=0.02)
        self.aux_head = nn.Sequential(nn.Linear(2 * d, d), nn.Mish(), nn.Linear(d, 3))
        self.aux_weight = aux_weight
        self.register_buffer("keep", torch.tensor(PROPRIO_KEEP))
        self.register_buffer("cube_mean", torch.zeros(3) if cube_mean is None else cube_mean.clone())
        self.register_buffer("cube_std", torch.ones(3) if cube_std is None else cube_std.clone())
        self._image_feat = None

    def encode(self, image: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """image [B,To,3,H,2W] = [workspace | wrist], proprio [B,To,1,21] -> tokens [B,To*3,d]."""
        b, t = image.shape[:2]
        w = image.shape[-1] // 2
        ws = self.image_encoder(image[..., :w].flatten(0, 1)).reshape(b, t, 1, -1)
        wr = self.wrist_encoder(image[..., w:].flatten(0, 1)).reshape(b, t, 1, -1)
        p = proprio[..., self.keep]
        p = ((p - self.proprio_mean) / self.proprio_std).clamp(-5.0, 5.0)
        tokens = torch.cat([ws, wr, self.proprio_proj(p)], dim=2)
        tokens = tokens + self.obs_time_emb[:, :, None] + self.obs_kind_emb[:, None]
        self._image_feat = torch.cat([ws[:, -1, 0], wr[:, -1, 0]], dim=-1)
        return tokens.flatten(1, 2)

    def loss(self, image, proprio, action, cube, generator=None):
        """(total, diffusion, aux); cube [B,3] = cube position at the last observation step."""
        bc = self.bc_loss(image, proprio, action, generator)
        aux = F.mse_loss(self.aux_head(self._image_feat).float(), (cube - self.cube_mean) / self.cube_std)
        return bc + self.aux_weight * aux, bc, aux

    @torch.no_grad()
    def predict_cube(self, image, proprio):
        self.encode(image, proprio)
        return self.aux_head(self._image_feat) * self.cube_std + self.cube_mean
