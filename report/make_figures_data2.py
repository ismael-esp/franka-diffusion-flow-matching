#!/usr/bin/env python3
"""Second pass of report figure data, following the visualization review:
- occlusion sensitivity (causal: grey out a 16x16 patch, measure the change of the executed 8-step action chunk, fixed noise)
  for the trained model AND a randomly initialised copy (sanity baseline, Adebayo et al. 2018);
- teacher-forced chunk RMSE over 20 held-out episodes, in radians;
- joint targets converted from normalized [-1,1] to radians with the Franka limits used by the env.
Held-out data: runs/v3_clean (never in the training set of runs/dp_wrist_more)."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.append(os.environ.get("DP_ROOT", str(Path(__file__).resolve().parents[1] / "pouring_diffusion")))
from fd_core.policy import load_dp  # noqa: E402
from dp_core.model import ddim_timesteps  # noqa: E402

OUT = ROOT / "report" / "data"; MEDIA = ROOT / "report" / "media"
dev = "cuda:0"
LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
to_rad = lambda n: LOW + (np.asarray(n)[..., :7] + 1) * 0.5 * (HIGH - LOW)

model, cfg = load_dp(ROOT / "runs/dp_wrist_more/bc_best.pt", dev)
rand = copy.deepcopy(model)
torch.manual_seed(123)
for m in rand.modules():
    if hasattr(m, "reset_parameters") and m is not rand:
        m.reset_parameters()
imgs = np.load(ROOT / "runs/v3_clean/images.npy", mmap_mode="r"); prop = np.load(ROOT / "runs/v3_clean/proprio.npy")
act = np.load(ROOT / "runs/v3_clean/action.npy")
EP = 3


@torch.no_grad()
def chunk(mdl, im, pr, seed=0):
    ddim = ddim_timesteps(cfg.num_train_timesteps, 10).to(dev)
    b = im.shape[0]
    cond = mdl.encode(im, pr)
    x = torch.randn(1, cfg.pred_horizon, cfg.action_dim, device=dev, generator=torch.Generator(device=dev).manual_seed(seed)).expand(b, -1, -1)
    for i, t in enumerate(ddim):
        prev = ddim[i + 1] if i + 1 < len(ddim) else torch.tensor(-1, device=dev)
        x, _ = mdl.ddim_mean(x, mdl.denoiser(x, t.expand(b), cond), t.expand(b), prev.expand(b), eta=0.0)
    return x


def obs(ep, t):
    ts = [max(0, t - 1), t]
    im = torch.from_numpy(np.stack([imgs[ep, s] for s in ts])).permute(0, 3, 1, 2)[None].float().to(dev) / 255
    return im, torch.from_numpy(prop[ep, ts])[None].to(dev)


def occlusion(mdl, ep, t, patch=16, stride=8):
    im, pr = obs(ep, t)
    base = chunk(mdl, im, pr)[:, :8]
    H, W = im.shape[-2:]
    ys, xs = list(range(0, H - patch + 1, stride)), list(range(0, W - patch + 1, stride))
    heat = np.zeros((H, W)); cnt = np.zeros((H, W))
    grey = im.mean()
    for y in ys:
        batch = []
        for x in xs:
            o = im.clone(); o[:, :, :, y:y + patch, x:x + patch] = grey      # occlude both history frames
            batch.append(o)
        bim = torch.cat(batch); bpr = pr.expand(len(batch), -1, -1, -1)
        d = (chunk(mdl, bim, bpr)[:, :8] - base).square().mean(dim=(1, 2)).sqrt().cpu().numpy()
        for x, v in zip(xs, d):
            heat[y:y + patch, x:x + patch] += v; cnt[y:y + patch, x:x + patch] += 1
    return heat / np.maximum(cnt, 1)


def render(rgb, heat, vmax):
    from matplotlib import cm
    s = np.clip(heat / vmax, 0, 1)
    col = cm.inferno(s)[..., :3]
    base = rgb.astype(np.float32) / 255
    return (np.clip(0.35 * base + 0.65 * (col * (0.25 + 0.75 * s[..., None]) + base * 0.25 * (1 - s[..., None])), 0, 1) * 255).astype(np.uint8)


import matplotlib
matplotlib.use("Agg")
phases = {"approach": 25, "grasp": 70, "lift": 140}
meta = {}
for name, t in phases.items():
    rgb = np.asarray(imgs[EP, t])
    h_tr, h_rd = occlusion(model, EP, t), occlusion(rand, EP, t)
    vmax = max(np.percentile(h_tr, 99.5), 1e-6)
    row = np.concatenate([rgb, np.full((128, 4, 3), 255, np.uint8), render(rgb, h_tr, vmax), np.full((128, 4, 3), 255, np.uint8),
                          render(rgb, h_rd, max(np.percentile(h_rd, 99.5), 1e-6))], 1)
    Image.fromarray(row).resize((row.shape[1] * 2, 256), Image.LANCZOS).save(MEDIA / f"occlusion_{name}.png", optimize=True)
    Image.fromarray(render(rgb, h_tr, vmax)).resize((512, 256), Image.LANCZOS).save(MEDIA / f"occl_{name}.png", optimize=True)
    top = h_tr > np.percentile(h_tr, 90)
    meta[name] = {"t": t, "time_s": t / 50, "wrist_share_top10pct": float(top[:, 128:].sum() / top.sum()),
                  "trained_peak_rad_rmse": float(h_tr.max()), "random_peak": float(h_rd.max())}
    print(name, meta[name], flush=True)
json.dump(meta, open(OUT / "occlusion.json", "w"), indent=1)

# teacher-forced chunk RMSE over 20 held-out episodes (radians, executed 8 steps)
errs = []
for ep in range(20):
    for t in range(0, 240, 16):
        im, pr = obs(ep, t)
        a = chunk(model, im, pr, seed=t)[0, :8].cpu().numpy()
        errs.append(np.sqrt(((to_rad(a) - to_rad(act[ep, t:t + 8])) ** 2).mean()))
summary = {"teacher_forced_chunk_rmse_rad_mean": float(np.mean(errs)), "p90": float(np.percentile(errs, 90)), "n_chunks": len(errs), "episodes": 20}
print(summary, flush=True)
json.dump(summary, open(OUT / "chunk_rmse.json", "w"), indent=1)

# radians versions of the denoise / receding data
d = json.load(open(OUT / "denoise.json"))
d["gt_rad"] = to_rad(d["gt"]).tolist(); d["trace_rad"] = [to_rad(x).tolist() for x in d["trace"]]; d["samples_rad"] = [to_rad(x).tolist() for x in d["samples"]]
d["note"] = "trace = actual DDIM samples x_k after each of 10 steps (x_10 = noise ... x_0); radians; teacher-forced observation"
json.dump(d, open(OUT / "denoise.json", "w"))
r = json.load(open(OUT / "receding.json"))
r["episode_gt_rad"] = to_rad(r["episode_gt"]).tolist(); r["chunks_rad"] = [{"t": c["t"], "a": to_rad(c["a"]).tolist()} for c in r["chunks"]]
r["note"] = "teacher-forced: each chunk predicted from the teacher's observation at its start step; held-out episode"
json.dump(r, open(OUT / "receding.json", "w"))
print("done", flush=True)
