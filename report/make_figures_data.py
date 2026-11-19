#!/usr/bin/env python3
"""Offline figure data for the report (no simulator): saliency maps, DDIM denoising, receding-horizon chunks,
learning curves and the v3 cube-position head. Uses the best Franka model (runs/dp_wrist_more/bc_best.pt) on held-out
demos (runs/v3_clean: collected after that model was trained, never in its training set).
Writes report/data/*.json and report/media/saliency_*.png."""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(os.environ.get("DP_ROOT", str(Path(__file__).resolve().parents[1] / "pouring_diffusion")))
from fd_core.policy import load_dp  # noqa: E402

OUT = ROOT / "report" / "data"; OUT.mkdir(exist_ok=True)
MEDIA = ROOT / "report" / "media"
dev = "cuda:0"
torch.manual_seed(0)
model, cfg = load_dp(ROOT / "runs/dp_wrist_more/bc_best.pt", dev)
for p in model.parameters():
    p.requires_grad_(False)

imgs = np.load(ROOT / "runs/v3_clean/images.npy", mmap_mode="r")        # [E,250,128,256,3]
prop = np.load(ROOT / "runs/v3_clean/proprio.npy")                       # [E,250,1,21]
act = np.load(ROOT / "runs/v3_clean/action.npy")                         # [E,250,8]
cube = np.load(ROOT / "runs/v3_clean/cube.npy")                          # [E,250,3]
EP = 3


def obs_at(ep, t):
    ts = [max(0, t - 1), t]
    im = torch.from_numpy(np.stack([imgs[ep, s] for s in ts])).permute(0, 3, 1, 2)[None].float().to(dev) / 255
    pr = torch.from_numpy(prop[ep, ts])[None].to(dev)
    return im, pr


def sample_trace(im, pr, steps=10, gen_seed=0, grad=False):
    """DDIM (eta=0) from noise; returns the list of x_t (x_T ... x_0)."""
    from dp_core.model import ddim_timesteps
    ddim = ddim_timesteps(cfg.num_train_timesteps, steps).to(dev)
    g = torch.Generator(device=dev).manual_seed(gen_seed)
    b = im.shape[0]
    cond = model.encode(im, pr)
    x = torch.randn(b, cfg.pred_horizon, cfg.action_dim, device=dev, generator=g)
    trace = [x]
    for i, t in enumerate(ddim):
        prev = ddim[i + 1] if i + 1 < len(ddim) else torch.tensor(-1, device=dev)
        tb = t.expand(b)
        x, _ = model.ddim_mean(x, model.denoiser(x, tb, cond), tb, prev.expand(b), eta=0.0)
        trace.append(x)
    return trace


# ---------------------------------------------------------------- 1. saliency (Jacobian-norm estimate)
def saliency(ep, t, k=12):
    im, pr = obs_at(ep, t)
    acc = torch.zeros_like(im[0, -1])
    for j in range(k):
        x = im.clone().requires_grad_(True)
        with torch.enable_grad():
            a = sample_trace(x, pr, steps=10, gen_seed=j)[-1][:, :8]           # executed part of the chunk
            w = torch.randn(a.shape, device=dev, generator=torch.Generator(device=dev).manual_seed(100 + j))
            (a * w).sum().backward()
        acc += x.grad[0, -1].abs()
    s = acc.sum(0)                                                            # [H, W]
    s = F.avg_pool2d(s[None, None], 7, 1, 3)[0, 0]
    return s.cpu().numpy()


def overlay(rgb, sal):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import cm
    out = []
    for half in (slice(0, 128), slice(128, 256)):                           # normalise each camera separately
        s = sal[:, half]; s = (s - s.min()) / (np.percentile(s, 99.5) - s.min() + 1e-8); s = np.clip(s, 0, 1)
        heat = cm.inferno(s)[..., :3]
        base = rgb[:, half].astype(np.float32) / 255
        out.append(np.clip(0.45 * base + 0.55 * heat * (0.35 + 0.65 * s[..., None]), 0, 1))
    return (np.concatenate(out, 1) * 255).astype(np.uint8)


phases = {"approach": 25, "grasp": 70, "lift": 140}
sal_meta = {}
for name, t in phases.items():
    sal = saliency(EP, t)
    rgb = np.asarray(imgs[EP, t])
    pair = np.concatenate([rgb, np.full((128, 4, 3), 255, np.uint8), overlay(rgb, sal)], 1)
    Image.fromarray(pair).resize((pair.shape[1] * 2, 256), Image.LANCZOS).save(MEDIA / f"saliency_{name}.png", optimize=True)
    ws, wr = sal[:, :128].sum(), sal[:, 128:].sum()
    sal_meta[name] = {"t": t, "time_s": t / 50, "wrist_share": float(wr / (ws + wr))}
print("saliency", sal_meta, flush=True)

# ---------------------------------------------------------------- 2. DDIM denoising of one chunk + sample spread
T0 = 55
im, pr = obs_at(EP, T0)
with torch.no_grad():
    trace = [x[0].cpu().numpy() for x in sample_trace(im, pr, steps=10, gen_seed=1)]
    samples = [sample_trace(im, pr, steps=10, gen_seed=s)[-1][0].cpu().numpy() for s in range(24)]
gt = act[EP, T0:T0 + 16]
var = gt[:, :7].std(0)
joints = [int(j) for j in np.argsort(-var)[:3]]
denoise = {"t": T0, "joints": joints, "gt": gt.tolist(), "trace": [x.tolist() for x in trace], "samples": [s.tolist() for s in samples],
           "final_mse": float(((trace[-1][:8] - gt[:8]) ** 2).mean()),
           "sample_spread": float(np.std(np.stack(samples)[:, :8, :7], axis=0).mean())}
json.dump(denoise, open(OUT / "denoise.json", "w"))
print("denoise joints", joints, "mse", denoise["final_mse"], "spread", denoise["sample_spread"], flush=True)

# ---------------------------------------------------------------- 3. receding-horizon chunks over one episode
chunks = []
with torch.no_grad():
    for t in range(0, 248, 8):
        im, pr = obs_at(EP, t)
        chunks.append({"t": t, "a": sample_trace(im, pr, steps=10, gen_seed=t)[-1][0].cpu().numpy().tolist()})
rh = {"episode_gt": act[EP].tolist(), "chunks": chunks, "joints": joints,
      "chunk_mse": float(np.mean([((np.array(c["a"])[:8] - act[EP, c["t"]:c["t"] + 8]) ** 2).mean() for c in chunks]))}
json.dump(rh, open(OUT / "receding.json", "w"))
print("receding chunk mse", rh["chunk_mse"], flush=True)

# ---------------------------------------------------------------- 4. learning curves + v3 checkpoint selection
def curve(run, offset=0):
    rows = [json.loads(l) for l in open(ROOT / "runs" / run / "metrics.jsonl")]
    return [[r["step"] + offset, r["val_action_mse"]] for r in rows]
curves = {"workspace cam (500 demos)": curve("dp"), "+ wrist cam (500 demos)": curve("dp_wrist"),
          "+ more data (1,300 demos)": curve("dp_wrist_more", offset=30000), "v3 recipe (1,200 demos)": curve("dp_v3")}
sel = sorted((int(Path(f).stem.split("_")[-1]), json.load(open(f))["success_rate"]) for f in glob.glob(str(ROOT / "runs/dp_v3/sel_ckpt_*.json")))
json.dump({"val_action_mse": curves, "v3_selection_success": sel}, open(OUT / "curves.json", "w"))

# ---------------------------------------------------------------- 5. v3 cube-position head on its held-out val episodes
v3, _ = load_dp(ROOT / "runs/dp_v3/ckpt_50000.pt", dev)
from dp_core.data import split_episodes  # noqa: E402
n_total = sum(int(np.load(ROOT / f"runs/{d}/kept.npy").sum()) for d in ("v3_clean", "v3_dart05", "v3_dart10"))
_, val_ids = split_episodes(n_total, 0.05, 7)
vid = int(val_ids[0])
src = ["v3_clean", "v3_dart05", "v3_dart10"][vid // 400]; e = vid % 400
I2 = np.load(ROOT / f"runs/{src}/images.npy", mmap_mode="r"); P2 = np.load(ROOT / f"runs/{src}/proprio.npy"); C2 = np.load(ROOT / f"runs/{src}/cube.npy")
pred = []
with torch.no_grad():
    for t in range(0, 250, 5):
        ts = [max(0, t - 1), t]
        im = torch.from_numpy(np.stack([I2[e, s] for s in ts])).permute(0, 3, 1, 2)[None].float().to(dev) / 255
        pr = torch.from_numpy(P2[e, ts])[None].to(dev)
        pred.append([t] + v3.predict_cube(im, pr)[0].cpu().numpy().tolist() + C2[e, t].tolist())
pred = np.array(pred)
err = np.linalg.norm(pred[:, 1:4] - pred[:, 4:7], axis=1)
json.dump({"episode": f"{src}#{e} (v3 validation split)", "rows": pred.tolist(), "mean_err_m": float(err.mean())}, open(OUT / "cube_head.json", "w"))
print("cube head mean err (m)", float(err.mean()), flush=True)
