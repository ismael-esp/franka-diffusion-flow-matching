#!/usr/bin/env python3
"""Stage 1: train the RGB transformer Diffusion Policy on the rendered demos (no simulator)."""
from __future__ import annotations

import argparse
import copy
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dp_core.data import WindowSampler, augment_images, load_dataset, normalizer_stats, split_episodes  # noqa: E402
from dp_core.model import DiffusionPolicy, PolicyConfig  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, nargs="+", required=True,
                   help="Folders written by render_demos.py / collect_*.py, trained on as one dataset.")
    p.add_argument("--init", type=Path, default=None,
                   help="Start from this checkpoint (its config and normalizer); the architecture flags are ignored.")
    p.add_argument("--output", type=Path, required=True, help="New run folder; an existing one is refused.")
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-episodes", type=int, default=0, help="Use only the first N demos (0 = all).")
    p.add_argument("--obs-horizon", type=int, default=2)
    p.add_argument("--pred-horizon", type=int, default=32)
    p.add_argument("--exec-horizon", type=int, default=16)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--bf16", action="store_true", help="bf16 autocast for the training forward pass (~1.3x faster).")
    p.add_argument("--smoke", action="store_true", help="Very short run for checking the pipeline.")
    return p.parse_args()


class Prefetcher:
    """Reads the next batches from disk in a background thread while the GPU trains on the current one."""

    def __init__(self, make_batch, depth: int = 3):
        self.make_batch = make_batch
        self.queue = queue.Queue(maxsize=depth)
        self.error = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            while True:
                self.queue.put(self.make_batch())
        except BaseException as error:  # surfaced on the next get()
            self.error = error
            self.queue.put(None)

    def get(self):
        batch = self.queue.get()
        if batch is None:
            raise RuntimeError("batch loading failed") from self.error
        return batch


@torch.no_grad()
def validate(model, sampler: WindowSampler, cfg: PolicyConfig, seed: int, batches: int = 10, batch_size: int = 64):
    """Denoising loss and sampled-action error on held-out episodes, with fixed randomness."""
    model.eval()
    window_generator = torch.Generator().manual_seed(seed)
    noise_generator = torch.Generator(device=sampler.device).manual_seed(seed)
    loss = sum(
        float(model.bc_loss(*sampler.sample(batch_size, window_generator), noise_generator)) for _ in range(batches)
    ) / batches
    torch.manual_seed(seed)
    image, proprio, action = sampler.sample(batch_size, window_generator)
    sampled = model.sample(image, proprio)
    action_mse = float((sampled - action)[:, : cfg.exec_horizon].square().mean())
    model.train()
    return loss, action_mse


def save(path: Path, model, cfg: PolicyConfig, share_mean, share_std, step: int, extra: dict) -> None:
    payload = {
        "kind": "bc", "config": cfg.to_dict(), "model": model.state_dict(), "share_mean": share_mean,
        "share_std": share_std, "step": step, **extra,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main():
    args = parse_args()
    if args.smoke:
        args.steps, args.val_every, args.log_every, args.batch_size = 6, 3, 2, 4
    if args.output.exists():
        sys.exit(f"{args.output} already exists; choose a new --output")
    args.output.mkdir(parents=True)
    (args.output / "command.json").write_text(json.dumps(sys.argv, indent=1))

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    demos = load_dataset(args.dataset[0] if len(args.dataset) == 1 else args.dataset, args.max_episodes)
    train_ids, val_ids = split_episodes(demos["proprio"].shape[0], args.val_ratio, args.seed)
    proprio_mean, proprio_std = normalizer_stats(demos["proprio"][train_ids])
    share_mean, share_std = normalizer_stats(demos["share"][train_ids])
    init = torch.load(args.init, map_location="cpu", weights_only=False) if args.init else None
    if init is not None:
        cfg = PolicyConfig(**init["config"])
        share_mean, share_std = init["share_mean"], init["share_std"]
    else:
        cfg = PolicyConfig(
            proprio_dim=demos["proprio"].shape[-1], share_obs_dim=demos["share"].shape[-1], obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon, exec_horizon=args.exec_horizon, d_model=args.d_model,
        )
    train = WindowSampler({k: v[train_ids] for k, v in demos.items()}, cfg.obs_horizon, cfg.pred_horizon, device)
    val = WindowSampler({k: v[val_ids] for k, v in demos.items()}, cfg.obs_horizon, cfg.pred_horizon, device)

    model = DiffusionPolicy(cfg, proprio_mean, proprio_std)
    if init is not None:
        model.load_state_dict(init["model"])  # keeps the checkpoint's proprio normalizer
    model = model.to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[bc] train episodes={len(train_ids)} val episodes={len(val_ids)} "
          f"params={sum(p.numel() for p in model.parameters()):,} device={device}", flush=True)

    extra = {"dataset": [str(d) for d in args.dataset], "init": str(args.init) if args.init else None,"args": {k: str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v
                                        for k, v in vars(args).items()}}
    best = math.inf
    batches = Prefetcher(lambda: train.sample(args.batch_size))
    start = time.time()
    with open(args.output / "metrics.jsonl", "a", buffering=1) as log:
        for step in range(1, args.steps + 1):
            warmup = min(1.0, step / max(1, args.warmup_steps))
            decay = 0.5 * (1.0 + math.cos(math.pi * step / args.steps))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * warmup * decay

            image, proprio, action = batches.get()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                loss = model.bc_loss(augment_images(image), proprio, action)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 1.0 - args.ema)

            if step % args.log_every == 0:
                print(f"[bc] step={step} loss={float(loss):.5f} steps/s={step / (time.time() - start):.1f}", flush=True)
            if step % args.val_every == 0 or step == args.steps:
                val_loss, action_mse = validate(ema, val, cfg, args.seed)
                row = {"step": step, "train_loss": float(loss), "val_loss": val_loss, "val_action_mse": action_mse}
                log.write(json.dumps(row) + "\n")
                print(f"[bc] val step={step} val_loss={val_loss:.5f} val_action_mse={action_mse:.6f}", flush=True)
                # Selection uses held-out demos only, never simulator results.
                if action_mse < best:
                    best = action_mse
                    save(args.output / "bc_best.pt", ema, cfg, share_mean, share_std, step, extra)
    save(args.output / "bc_final.pt", ema, cfg, share_mean, share_std, args.steps, extra)
    print(f"[bc] done: best val_action_mse={best:.6f} in {args.output}", flush=True)


if __name__ == "__main__":
    main()
