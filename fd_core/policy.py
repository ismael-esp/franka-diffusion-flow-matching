"""Load a Diffusion Policy checkpoint written by train_dp.py (model code: pouring_diffusion/dp_core, unmodified)."""
from __future__ import annotations

import os
from pathlib import Path
import sys

DP_ROOT = os.environ.get("DP_ROOT", str(Path(__file__).resolve().parents[1] / "pouring_diffusion"))


def load_dp(path, device):
    import torch
    if DP_ROOT not in sys.path:
        sys.path.append(DP_ROOT)          # appended, so this project's own modules always win name clashes
    from dp_core.model import DiffusionPolicy, PolicyConfig
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = PolicyConfig(**ck["config"])
    no_maxpool = bool((ck.get("args") or {}).get("no_maxpool", False))
    if ck.get("kind") == "v3":
        from fd_core.model_v3 import DiffusionPolicyV3
        model = DiffusionPolicyV3(cfg, torch.zeros(cfg.proprio_dim), torch.ones(cfg.proprio_dim))
        model.load_state_dict(ck["model"])
        return model.to(device).eval(), cfg
    if ck.get("objective") == "flow":
        from dp_core.flow import FlowPolicy
        model = FlowPolicy(cfg, torch.zeros(cfg.proprio_dim), torch.ones(cfg.proprio_dim))
    else:
        model = DiffusionPolicy(cfg, torch.zeros(cfg.proprio_dim), torch.ones(cfg.proprio_dim))
    if no_maxpool:
        remove_maxpool(model)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg


def remove_maxpool(model):
    """Option 2: drop the ResNet18 stem max-pool -> stride 16, 8x8 spatial-softmax grid per 128 px camera (was 4x4).
    The slot becomes Identity, so state-dict keys (and pretrained weights) stay compatible."""
    import torch.nn as nn
    bb = model.image_encoder.backbone
    assert isinstance(bb[3], nn.MaxPool2d), bb[3]
    bb[3] = nn.Identity()
    return model


def camera_setup(path):
    """(wrist, zoom) the checkpoint was trained with, read from its datasets' meta.json."""
    import json
    from pathlib import Path
    import torch
    root = Path(__file__).resolve().parents[1]
    ds = torch.load(path, map_location="cpu", weights_only=False).get("dataset", [])
    metas = [json.loads((root / d / "meta.json").read_text()) for d in ds if (root / d / "meta.json").exists()]
    if not metas:
        raise FileNotFoundError(f"training datasets of {path} not found: {ds}")
    return any(m.get("wrist", False) for m in metas), any(m.get("zoom", False) for m in metas)
