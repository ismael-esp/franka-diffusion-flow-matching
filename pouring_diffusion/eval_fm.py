#!/usr/bin/env python3
"""eval_policy.py for flow-matching checkpoints (train_fm.py); same flags. --ddim-steps sets the Euler steps."""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dp_core.model  # noqa: E402
from dp_core.flow import FlowPolicy  # noqa: E402

dp_core.model.DiffusionPolicy = FlowPolicy
sys.argv[0] = str(Path(__file__).resolve().parent / "eval_policy.py")
runpy.run_path(sys.argv[0], run_name="__main__")
