"""Rendered pouring demos as fixed-size diffusion training windows.

A dataset folder (render_demos.py) holds images.npy [E,T,H,W,3] uint8, proprio.npy and
share.npy [E,T,2,D], action.npy [E,T,14] and kept.npy, the replays that still succeeded.
Actions are already absolute commands in [-1, 1], so only state inputs are normalized.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

OBS_STD_FLOOR = 0.05


class DiskImages:
    """Episodes of images.npy files read per batch: 400 demos are 21 GB, too much to copy into RAM.

    Episode i is episode episodes[i] of file files[i], so several datasets can be indexed as one.
    """

    def __init__(self, path, episodes: np.ndarray, files: np.ndarray | None = None):
        self.paths = [Path(p) for p in path] if isinstance(path, (list, tuple)) else [Path(path)]
        self.arrays = [np.load(p, mmap_mode="r") for p in self.paths]
        self.episodes = np.asarray(episodes)
        self.files = np.zeros(len(self.episodes), dtype=np.int64) if files is None else np.asarray(files)

    @staticmethod
    def concat(parts: list["DiskImages"]) -> "DiskImages":
        paths, episodes, files = [], [], []
        for part in parts:
            files.append(part.files + len(paths))
            paths.extend(part.paths)
            episodes.append(part.episodes)
        return DiskImages(paths, np.concatenate(episodes), np.concatenate(files))

    @property
    def shape(self) -> tuple[int, ...]:
        t, h, w, c = self.arrays[0].shape[1:]
        return len(self.episodes), t, c, h, w

    def __getitem__(self, key):
        """Tensor index -> a subset of episodes; (rows, steps) -> uint8 frames [...,3,H,W]."""
        if not isinstance(key, tuple):
            index = key.numpy()
            return DiskImages(self.paths, self.episodes[index], self.files[index])
        rows, steps = key
        rows, steps = rows.numpy(), np.broadcast_to(steps.numpy(), np.broadcast_shapes(rows.shape, steps.shape))
        rows = np.broadcast_to(rows, steps.shape)
        t, h, w, c = self.arrays[0].shape[1:]
        frames = np.empty((*steps.shape, h, w, c), dtype=np.uint8)
        file_of = self.files[rows]
        for f in np.unique(file_of):
            mask = file_of == f
            frames[mask] = self.arrays[f][self.episodes[rows[mask]], steps[mask]]
        return torch.from_numpy(frames).moveaxis(-1, -3)


def load_dataset(folder: str | Path | list, max_episodes: int = 0) -> dict:
    """Load the kept episodes: images stay on disk as uint8 [E,T,3,H,W], the rest in CPU memory.

    A list of folders is loaded as one dataset; all of them must share the episode length.
    """
    if isinstance(folder, (list, tuple)):
        parts = [load_dataset(f, max_episodes) for f in folder]
        lengths = {p["proprio"].shape[1] for p in parts}
        if len(lengths) != 1:
            raise RuntimeError(f"datasets have different episode lengths: {sorted(lengths)}")
        merged = {k: torch.cat([p[k] for p in parts]) for k in ("proprio", "share", "action", "start")}
        merged["image"] = DiskImages.concat([p["image"] for p in parts])
        return merged
    folder = Path(folder)
    kept = np.flatnonzero(np.load(folder / "kept.npy"))
    if max_episodes > 0:
        kept = kept[:max_episodes]
    if len(kept) == 0:
        raise RuntimeError(f"no usable episodes in {folder}")
    # start.npy (recovery demos): the first step whose action the teacher executed; earlier ones are not trained on.
    start = np.load(folder / "start.npy")[kept] if (folder / "start.npy").exists() else np.zeros(len(kept), dtype=np.int64)
    return {
        "image": DiskImages(folder / "images.npy", kept),
        "proprio": torch.from_numpy(np.load(folder / "proprio.npy")[kept]).float(),
        "share": torch.from_numpy(np.load(folder / "share.npy")[kept]).float(),
        "action": torch.from_numpy(np.load(folder / "action.npy")[kept]).float(),
        "start": torch.from_numpy(start).long(),
    }


def normalizer_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension mean/std over every step and agent, with a floor for constant dims."""
    flat = x.reshape(-1, x.shape[-1])
    return flat.mean(0), flat.std(0).clamp(min=OBS_STD_FLOOR)


def split_episodes(num_episodes: int, val_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split by whole episode so validation windows never overlap training windows."""
    order = torch.randperm(num_episodes, generator=torch.Generator().manual_seed(seed))
    num_val = max(1, round(num_episodes * val_ratio))
    return order[num_val:].sort().values, order[:num_val].sort().values


class WindowSampler:
    """Random (image history, proprio history, action chunk) windows.

    A window at step t holds observations t-To+1 .. t and actions t .. t+Tp-1. History
    before the first step repeats the first frame; actions past the end repeat the last
    action. Images stay on the CPU and only each batch moves to the device.
    """

    def __init__(self, data: dict, obs_horizon: int, pred_horizon: int, device):
        self.image, self.proprio, self.action = data["image"], data["proprio"], data["action"]
        self.start = data.get("start", torch.zeros(self.proprio.shape[0], dtype=torch.long))
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.device = device
        self.num_episodes, self.length = self.image.shape[:2]

    def windows(self, episodes: torch.Tensor, steps: torch.Tensor):
        obs_steps = (steps[:, None] + torch.arange(1 - self.obs_horizon, 1)).clamp(min=0)
        action_steps = (steps[:, None] + torch.arange(self.pred_horizon)).clamp(max=self.length - 1)
        rows = episodes[:, None]
        image = self.image[rows, obs_steps].to(self.device, non_blocking=True).float() / 255.0
        return image, self.proprio[rows, obs_steps].to(self.device), self.action[rows, action_steps].to(self.device)

    def sample(self, batch_size: int, generator=None):
        episodes = torch.randint(self.num_episodes, (batch_size,), generator=generator)
        start = self.start[episodes]
        steps = start + (torch.rand(batch_size, generator=generator) * (self.length - start)).long()
        return self.windows(episodes, steps)


def augment_images(image: torch.Tensor, max_shift: int = 4) -> torch.Tensor:
    """Random shift plus brightness and contrast per sequence, for sim-to-real robustness."""
    b, t, c, h, w = image.shape
    device = image.device
    padded = torch.nn.functional.pad(image.reshape(b * t, c, h, w), [max_shift] * 4, mode="replicate")
    shift = torch.randint(0, 2 * max_shift + 1, (b, 2), device=device).repeat_interleave(t, dim=0)
    columns = torch.arange(w, device=device)[None] + shift[:, 0, None]
    rows = torch.arange(h, device=device)[None] + shift[:, 1, None]
    index = torch.arange(b * t, device=device)[:, None, None]
    shifted = padded[index, :, rows[:, :, None], columns[:, None, :]].permute(0, 3, 1, 2).reshape(b, t, c, h, w)
    brightness = 1.0 + 0.2 * (torch.rand(b, 1, 1, 1, 1, device=device) * 2.0 - 1.0)
    contrast = 1.0 + 0.2 * (torch.rand(b, 1, 1, 1, 1, device=device) * 2.0 - 1.0)
    mean = shifted.mean(dim=(2, 3, 4), keepdim=True)
    return ((shifted * brightness - mean) * contrast + mean).clamp(0.0, 1.0)
