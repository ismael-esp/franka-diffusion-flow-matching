#!/usr/bin/env python3
"""train_bc.py with the flow-matching objective (dp_core.flow.FlowPolicy); same flags, same checkpoint format plus
"objective": "flow". Evaluate with eval_fm.py."""
import sys

import dp_core.model
from dp_core.flow import FlowPolicy

dp_core.model.DiffusionPolicy = FlowPolicy          # train_bc imports the name from dp_core.model
import train_bc  # noqa: E402

_save = train_bc.save


def save(path, model, cfg, share_mean, share_std, step, extra):
    _save(path, model, cfg, share_mean, share_std, step, {**extra, "objective": "flow"})


train_bc.save = save
train_bc.DiffusionPolicy = FlowPolicy
if __name__ == "__main__":
    sys.argv[0] = "train_fm.py"
    train_bc.main()
