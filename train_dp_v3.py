#!/usr/bin/env python3
"""Train the v3 Diffusion Policy (fd_core/model_v3.py) on Franka lift demos recorded with --wrist and cube.npy.

Differences from train_dp.py, each from the literature review:
- long training (default 150k steps; Diffusion Policy trains image policies for ~3000 epochs),
- checkpoints every --ckpt-every steps, chosen afterwards by simulator success on a separate validation seed
  (select_v3.sh), not by validation loss (robomimic: val loss and success diverge),
- per-camera encoders, no joint velocities, auxiliary cube-position loss (see model_v3.py).
Augmentation is dp_core's random shift (+-4 px, a crop-style augmentation) plus brightness/contrast.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(os.environ.get("DP_ROOT", str(Path(__file__).resolve().parent / "pouring_diffusion")))

from dp_core.data import WindowSampler, augment_images, load_dataset, normalizer_stats, split_episodes  # noqa: E402
from dp_core.model import PolicyConfig  # noqa: E402
from fd_core.model_v3 import PROPRIO_KEEP, DiffusionPolicyV3  # noqa: E402
from train_dp import Prefetcher  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--ckpt-every", type=int, default=25_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--ema", type=float, default=0.9995)
    p.add_argument("--aux-weight", type=float, default=0.1)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--val-every", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ddim-steps", type=int, default=10)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


class CubeSampler(WindowSampler):
    """dp_core windows plus the cube position at the last observation step."""

    def __init__(self, data, cube, *a):
        super().__init__(data, *a)
        self.cube = cube

    def sample(self, batch_size, generator=None):
        episodes = torch.randint(self.num_episodes, (batch_size,), generator=generator)
        steps = (torch.rand(batch_size, generator=generator) * self.length).long()
        image, proprio, action = self.windows(episodes, steps)
        return image, proprio, action, self.cube[episodes, steps].to(self.device)


def load_cube(folders):
    out = []
    for f in folders:
        kept = np.flatnonzero(np.load(Path(f) / "kept.npy"))
        if not (Path(f) / "cube.npy").exists():
            sys.exit(f"{f} has no cube.npy: re-collect it with the current collect_demos.py")
        out.append(torch.from_numpy(np.load(Path(f) / "cube.npy")[kept]).float())
    return torch.cat(out)


@torch.no_grad()
def validate(model, sampler, cfg, seed, batches=8, bs=64):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    ng = torch.Generator(device=sampler.device).manual_seed(seed)
    loss = aux = 0.0
    for _ in range(batches):
        _, b, a = model.loss(*sampler.sample(bs, g), generator=ng)
        loss += float(b) / batches; aux += float(a) / batches
    torch.manual_seed(seed)
    image, proprio, action, cube = sampler.sample(bs, g)
    mse = float((model.sample(image, proprio) - action)[:, : cfg.exec_horizon].square().mean())
    cube_err = float((model.predict_cube(image, proprio) - cube).norm(dim=-1).mean())
    model.train()
    return loss, aux, mse, cube_err


def save(path, model, cfg, step, extra):
    tmp = path.with_suffix(".tmp")
    torch.save({"kind": "v3", "config": cfg.to_dict(), "model": model.state_dict(), "aux_weight": model.aux_weight,
                "step": step, **extra}, tmp)
    tmp.replace(path)


def main():
    args = parse_args()
    if args.smoke:
        args.steps, args.ckpt_every, args.val_every, args.log_every, args.batch_size = 6, 3, 3, 2, 4
    if args.output.exists():
        sys.exit(f"{args.output} already exists; choose a new --output")
    args.output.mkdir(parents=True)
    (args.output / "command.json").write_text(json.dumps(sys.argv, indent=1))
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    folders = [str(d) for d in args.dataset]
    demos = load_dataset(folders[0] if len(folders) == 1 else folders)
    cube = load_cube(folders)
    assert cube.shape[0] == demos["proprio"].shape[0]
    train_ids, val_ids = split_episodes(demos["proprio"].shape[0], args.val_ratio, args.seed)
    pm, ps = normalizer_stats(demos["proprio"][train_ids][..., PROPRIO_KEEP])
    cm, cs = cube[train_ids].reshape(-1, 3).mean(0), cube[train_ids].reshape(-1, 3).std(0).clamp(min=0.01)
    cfg = PolicyConfig(proprio_dim=len(PROPRIO_KEEP), share_obs_dim=1, obs_horizon=2, pred_horizon=16, exec_horizon=8,
                       d_model=256, num_agents=1, agent_action_dim=demos["action"].shape[-1], num_inference_steps=args.ddim_steps)
    sub = lambda ids: {k: v[ids] for k, v in demos.items()}
    train = CubeSampler(sub(train_ids), cube[train_ids], cfg.obs_horizon, cfg.pred_horizon, device)
    val = CubeSampler(sub(val_ids), cube[val_ids], cfg.obs_horizon, cfg.pred_horizon, device)
    model = DiffusionPolicyV3(cfg, pm, ps, cm, cs, args.aux_weight).to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[v3] episodes train={len(train_ids)} val={len(val_ids)} params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    extra = {"dataset": folders, "args": {k: str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v
                                          for k, v in vars(args).items()}}
    batches = Prefetcher(lambda: train.sample(args.batch_size))
    start = time.time()
    with open(args.output / "metrics.jsonl", "a", buffering=1) as log:
        for step in range(1, args.steps + 1):
            lr = args.lr * min(1.0, step / args.warmup_steps) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
            for g in opt.param_groups:
                g["lr"] = lr
            image, proprio, action, cb = batches.get()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                loss, bc, aux = model.loss(augment_images(image), proprio, action, cb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for t_, s_ in zip(ema.parameters(), model.parameters()):
                    t_.lerp_(s_, 1.0 - args.ema)
            if step % args.log_every == 0:
                el = time.time() - start
                print(f"[v3] step={step} diff={float(bc):.5f} aux={float(aux):.4f} steps/s={step / el:.1f} "
                      f"eta={(args.steps - step) / (step / el) / 3600:.1f}h", flush=True)
            if step % args.val_every == 0 or step == args.steps:
                vl, va, mse, ce = validate(ema, val, cfg, args.seed)
                row = {"step": step, "val_loss": vl, "val_aux": va, "val_action_mse": mse, "val_cube_err_m": ce}
                log.write(json.dumps(row) + "\n")
                print(f"[v3] val {json.dumps(row)}", flush=True)
            if step % args.ckpt_every == 0 or step == args.steps:
                save(args.output / f"ckpt_{step}.pt", ema, cfg, step, extra)
    print(f"[v3] done in {(time.time() - start) / 3600:.2f} h -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
