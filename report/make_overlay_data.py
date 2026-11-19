#!/usr/bin/env python3
"""runs/denoise_v3/*_iterates.npz (record_denoise.py) -> report/data/overlay_lift_e{ENV}.json for the Remotion overlay scenes.
Joint chunks -> FK (geom.py, verified against the simulator's panda_hand to 1e-6 m) -> projection into the 640x480 video camera.
Everything is computed from the recorded model outputs: nothing is generated. Joint targets are clipped to the joint limits before FK."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from geom import fk, norm_to_q, project
from grav import rollout_path, settle
ALPHA = 0.02          # per-step tracking rate of the PD controller (fitted, see validate_overlay.py)

root = Path(__file__).resolve().parents[1]
env = int(sys.argv[1]) if len(sys.argv) > 1 else 2
PAUSE_Q = [0, 8, 15]
DT = 0.1
GX, GY, X0, X1, Y0, Y1, SIG = 18, 13, 90, 570, 40, 400, 38.0
al = json.load(open(root / "report_v2/data/denoise_lift_e1.json"))["alphas_cumprod"]
out = {"env": env, "alphas_cumprod": al, "pause_q": PAUSE_Q, "grid": {"gx": GX, "gy": GY, "x0": X0, "x1": X1, "y0": Y0, "y1": Y1}, "models": {}}
pix = lambda a, lo, hi, q0: project(fk(rollout_path(norm_to_q(a[..., :7], lo, hi), q0, ALPHA)))[0]   # [...,16,8] -> [...,16,2]: predicted gripper path (gravity sag + tracking lag)
rd = lambda a, n=1: np.round(a, n).tolist()
report = {}
for name in ("diffusion", "flow"):
    d = np.load(root / f"runs/denoise_v3/{name}_iterates.npz")
    ke = list(d["keep_envs"]).index(env)
    lo, hi = d["joint_low"].astype(np.float64), d["joint_high"].astype(np.float64)
    it = d["many_iter"][:, :, ke].astype(np.float64)        # [Q,K+1,S,16,8]
    pr = d["many_pred"][:, :, ke].astype(np.float64)        # [Q,K,S,16,8]
    Q, K1, S = it.shape[:3]
    q0s = d['qpos'][env].astype(np.float64)[d['query_step']]      # measured joints at each policy call [Q,7]
    P = pix(it, lo, hi, q0s[:, None, None, :])                                      # [Q,K+1,S,16,2]
    ee3 = fk(settle(norm_to_q(it[:, -1, :, :, :7], lo, hi)))   # spread of the commanded (sag-corrected) targets, not of the lag-filtered path          # final chunks, [Q,S,16,3]
    spread = ee3[:, :, 15].std(axis=1).mean(axis=-1)         # metres, std of the chunk's last EE point over the S samples
    # simple two-mode test per call: split along the first principal axis, compare the gap with the within-group spread
    sep = []
    for q in range(Q):
        x = ee3[q, :, 15]; x = x - x.mean(0)
        u, s_, vt = np.linalg.svd(x, full_matrices=False); pc = x @ vt[0]
        g = pc > np.median(pc) if False else pc > pc.mean()
        a_, b_ = pc[g], pc[~g]
        sep.append(float(abs(a_.mean() - b_.mean()) / (np.sqrt((a_.var() + b_.var()) / 2) + 1e-9)) if len(a_) > 2 and len(b_) > 2 else 0.0)
    m = {"final_all": rd(P[:, -1]),                          # every call: the S clean samples [Q,S,16,2]
         "executed_px": rd(project(d["hand_pos"][env] * 0 + fk(norm_to_q(d["executed"][env][:, :7].astype(np.float64), lo, hi)))[0]),  # commanded TCP per step [250,2]
         "actual_px": rd(project(fk(d["qpos"][env].astype(np.float64)))[0]),                                                           # real TCP per step
         "pause": {}}
    for q in PAUSE_Q:
        e = {"iter": rd(P[q])}                               # [K+1,S,16,2]
        if name == "flow":                                   # projected end-effector velocity field: FK(x + dt v) - FK(x), in pixels per dt
            a1 = it[q, :-1] + DT * pr[q]                     # [K,S,16,8]
            a1[..., 7] = it[q, :-1][..., 7]
            disp = pix(a1, lo, hi, q0s[q]) - P[q, :-1]               # [K,S,16,2]
            e["vel"] = rd(disp[:, :, ::3], 1)                # arrows of chunk steps 0,3,6,... [K,S,6,2]
            gx, gy = np.meshgrid(np.linspace(X0, X1, GX), np.linspace(Y0, Y1, GY))
            fld = np.zeros((10, GY, GX, 3))
            for i in range(10):
                pos = P[q, i].reshape(-1, 2); dv = disp[i].reshape(-1, 2)
                ok = np.isfinite(pos).all(1) & (np.abs(pos - [320, 240]).max(1) < 2000)
                pos, dv = pos[ok], dv[ok]
                w = np.exp(-(((gx[..., None] - pos[:, 0]) ** 2 + (gy[..., None] - pos[:, 1]) ** 2) / (2 * SIG ** 2)))   # [GY,GX,n]
                ws = w.sum(-1)
                fld[i, ..., 0] = (w * dv[:, 0]).sum(-1) / (ws + 1e-9); fld[i, ..., 1] = (w * dv[:, 1]).sum(-1) / (ws + 1e-9); fld[i, ..., 2] = ws
            e["field"] = rd(fld, 1)                          # [10,GY,GX,(dx,dy,weight)]
        m["pause"][str(q)] = e
    out["models"][name] = m
    report[name] = {"ee_endpoint_std_m_median": float(np.median(spread)), "two_mode_separation_median": float(np.median(sep)), "two_mode_separation_max": float(np.max(sep))}
out["multimodality"] = report
(root / f"report_v2/data/overlay_lift_e{env}.json").write_text(json.dumps(out, separators=(",", ":")))
print(json.dumps(report, indent=1))
