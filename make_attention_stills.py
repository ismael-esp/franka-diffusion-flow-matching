#!/usr/bin/env python3
"""Compose attention stills from the npz written by attention_task.py / record_attention.py (same compose()).
usage: make_attention_stills.py out.png P hz npz:frame:label [npz:frame:label ...]   -> one column of stills"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_attention import compose  # noqa: E402

out, P, hz = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
rows = []
for spec in sys.argv[4:]:
    f, t, label = spec.split(":", 2)
    z = np.load(f); t = int(t); ok = int(z["outcome"]) == 0
    k = min(t, z["heat"].shape[0] - 1)
    rows.append(compose(z["frames"][k], z["image"][k], __import__("torch").from_numpy(z["heat"][k:k + 1])[0],
                        __import__("torch").from_numpy(z["kp"][k]), __import__("torch").from_numpy(z["kw"]), P, label, ok,
                        int(k / hz * 50)))
Image.fromarray(np.vstack(rows)).save(out)
print(out, np.vstack(rows).shape)
