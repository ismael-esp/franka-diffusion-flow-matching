#!/usr/bin/env python3
"""Re-render attention videos from the .npz files written by record_attention.py, concatenating episodes into one clip.
Usage: render_attention.py OUT.mp4 EP.npz[:note] [EP.npz[:note] ...]   (run with the Isaac Lab python, no simulator needed)"""
from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_attention import compose  # noqa: E402

LABELS = ["SUCCESS", "FAIL · never lifted", "FAIL · dropped", "FAIL · off goal", "FAIL · fell off"]


def main():
    out, eps = Path(sys.argv[1]), sys.argv[2:]
    wr = imageio.get_writer(out, fps=25, codec="libx264", quality=7, macro_block_size=8)
    for spec in eps:
        path, _, note = spec.partition(":")
        d = np.load(path)
        o = int(d["outcome"]); heat, kp, kw = (torch.from_numpy(d[k]) for k in ("heat", "kp", "kw"))
        for t in range(d["image"].shape[0]):
            wr.append_data(compose(d["frames"][t], d["image"][t], heat[t], kp[t], kw, 320, LABELS[o], o == 0, t, note or None))
    wr.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
