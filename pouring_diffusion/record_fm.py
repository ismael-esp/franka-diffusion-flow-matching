#!/usr/bin/env python3
"""record_video.py for flow-matching checkpoints; extra flag --ddim-steps K sets the number of Euler steps."""
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
steps = 0
if "--ddim-steps" in sys.argv:
    i = sys.argv.index("--ddim-steps"); steps = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
import dp_core.model  # noqa: E402
from dp_core.flow import FlowPolicy  # noqa: E402

dp_core.model.DiffusionPolicy = FlowPolicy
import eval_policy  # noqa: E402

_load = eval_policy.load_policy


def load_policy(path, device):
    model, cfg, kind = _load(path, device)
    if steps:
        model.ddim_steps = dp_core.model.ddim_timesteps(cfg.num_train_timesteps, steps).to(device)
    return model, cfg, kind


eval_policy.load_policy = load_policy
sys.argv[0] = str(root / "record_video.py")
runpy.run_path(sys.argv[0], run_name="__main__")
