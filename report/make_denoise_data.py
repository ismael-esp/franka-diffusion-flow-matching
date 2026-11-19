import os
#!/usr/bin/env python3
"""runs/denoise/*_iterates.npz (record_denoise.py) -> report/data/denoise_lift_e{N}.json
(imported by report/video/src/converge.tsx) for the Remotion convergence scenes. Only real recorded values; 4 of the 8 action dims are kept (J1, J2, J7, gripper)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.append(os.environ.get("DP_ROOT", str(Path(__file__).resolve().parents[1] / "pouring_diffusion")))
from dp_core.model import cosine_alphas_cumprod

root = Path(__file__).resolve().parents[1]
env = int(sys.argv[1]) if len(sys.argv) > 1 else 1
DIMS = [0, 1, 6, 7]
meta = json.loads((root / "runs/denoise/meta.json").read_text())
r = lambda a: np.round(np.asarray(a, dtype=np.float64), 2).tolist()
out = {"task": "lift", "seed": meta["seed"], "env": env, "dims": DIMS, "dim_names": ["joint 1", "joint 2", "joint 7", "gripper"],
       "alphas_cumprod": r(cosine_alphas_cumprod(100).numpy()), "models": {}}
for name in ("diffusion", "flow"):
    d = np.load(root / f"runs/denoise/{name}_iterates.npz")
    out["models"][name] = {
        "checkpoint": meta["models"][name]["checkpoint"], "success": bool(d["success"][env]), "final_dist": float(d["dist"][env]),
        "times": d["times"].tolist(), "query_step": d["query_step"].tolist(),
        "iterates": r(d["iterates"][:, :, env][..., DIMS]),   # [Q, K+1, 16, 4]  noise, then after each step
        "preds": r(d["preds"][:, :, env][..., DIMS]),         # [Q, K, 16, 4]    diffusion: clipped x0 estimate, flow: velocity
        "executed": r(d["executed"][env][:, DIMS]),           # [250, 4]
    }
(root / f"report/data/denoise_lift_e{env}.json").write_text(json.dumps(out, separators=(",", ":")))
print("wrote", root / f"report/data/denoise_lift_e{env}.json")
