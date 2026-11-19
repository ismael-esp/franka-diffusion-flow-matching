#!/usr/bin/env python3
"""Rollouts of a Diffusion Policy with its visual attention drawn on the camera images it sees.

The encoder's spatial softmax is the policy's only path from pixels to actions: each of its 32 keypoint channels is a
softmax over the 4x8 feature grid of the [workspace | wrist] image, and the policy receives only each channel's expected
(x, y). Per frame we draw (1) the attention heatmap = the softmax maps averaged over channels, weighted by how strongly
the policy reads each channel (norm of its weights in the projection layer), and (2) the 12 most-read keypoints as dots.
Writes one mp4 per env (third-person view | workspace view + attention | wrist view + attention), named by outcome."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=12)
    p.add_argument("--seed", type=int, default=601)
    p.add_argument("--panel", type=int, default=320, help="Height in px of each panel of the output video.")
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def attention(model, images, top=12):
    """images uint8 [T,H,W,3] -> heat [T,h,w] (normalised per frame), keypoints [T,top,2] in [-1,1], weights [top]."""
    import torch
    import torch.nn.functional as F
    enc = model.image_encoder
    w = enc.proj.weight                                     # [d, 2K]: inputs are (x_1..x_K, y_1..y_K)
    k = enc.keypoints.out_channels
    imp = (w[:, :k].square().sum(0) + w[:, k:].square().sum(0)).sqrt()
    imp = imp / imp.sum()
    order = imp.argsort(descending=True)[:top]
    heats, kps = [], []
    for i in range(0, len(images), 64):
        x = images[i:i + 64].permute(0, 3, 1, 2).float().div(255.0).to(w.device)
        heat = enc.keypoints(enc.backbone((x - enc.pixel_mean) / enc.pixel_std))
        b, _, h, wd = heat.shape
        prob = F.softmax(heat.reshape(b, k, -1), dim=-1).reshape(b, k, h, wd)
        ys, xs = torch.linspace(-1, 1, h, device=w.device), torch.linspace(-1, 1, wd, device=w.device)
        kx, ky = (prob.sum(2) * xs).sum(-1), (prob.sum(3) * ys).sum(-1)
        hm = (prob * imp[None, :, None, None]).sum(1)
        heats.append((hm / hm.amax(dim=(1, 2), keepdim=True)).cpu())
        kps.append(torch.stack([kx[:, order], ky[:, order]], -1).cpu())
    return torch.cat(heats), torch.cat(kps), imp[order].cpu()


def compose(frame3p, img, heat, kp, kw, P, label, ok, t, note=None):
    """One output frame: [third-person | workspace + attention | wrist + attention], each P px tall."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from matplotlib import colormaps
    from PIL import Image, ImageDraw, ImageFont
    H, W = img.shape[:2]                                   # 128 x 256 policy image
    hw = heat.shape[1] // 2                                # upsample each camera's half separately: no blur across the seam
    up = torch.cat([F.interpolate(heat[None, None, :, s:s + hw], size=(P, P), mode="bicubic", align_corners=False)[0, 0]
                    for s in (0, hw)], dim=1).clamp(0, 1).numpy()
    base = np.asarray(Image.fromarray(img).resize((2 * P, P), Image.BICUBIC)).astype(np.float32) / 255.0
    col = colormaps["inferno"](up)[..., :3]
    a = (0.15 + 0.6 * up ** 1.5)[..., None]
    over = Image.fromarray((255 * ((1 - a) * base * 0.85 + a * col)).astype(np.uint8))
    d = ImageDraw.Draw(over)
    r0 = max(3, P // 70)
    for (x, y), wv in zip(kp.tolist(), (kw / kw.max()).tolist()):
        px, py = (x + 1) / 2 * (2 * P - 1), (y + 1) / 2 * (P - 1)
        r = r0 * (0.8 + 0.8 * wv)
        d.ellipse([px - r, py - r, px + r, py + r], fill=(80, 230, 255), outline=(255, 255, 255), width=1)
    d.line([(P, 0), (P, P)], fill=(255, 255, 255), width=2)
    th = Image.fromarray(frame3p).resize((int(P * frame3p.shape[1] / frame3p.shape[0]), P), Image.BICUBIC)
    out = Image.new("RGB", (th.width + 2 * P, P), (255, 255, 255))
    out.paste(th, (0, 0)); out.paste(over, (th.width, 0))
    d = ImageDraw.Draw(out)
    fs = max(12, P // 12)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
        fn = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(fs * 0.8))
    except OSError:
        f = fn = ImageFont.load_default()
    tag = (36, 140, 80) if ok else (196, 60, 50)
    tw = d.textlength(label, font=f)
    d.rectangle([8, 8, 8 + tw + 16, 8 + fs + 12], fill=tag); d.text((16, 13), label, fill="white", font=f)
    ts = f"{t / 50:.1f} s"
    d.rectangle([8, P - fs - 20, 8 + d.textlength(ts, font=f) + 16, P - 8], fill=(18, 32, 43)); d.text((16, P - fs - 15), ts, fill="white", font=f)
    if note:
        nw = d.textlength(note, font=fn)
        d.rectangle([8, 8 + fs + 18, 8 + nw + 16, 8 + fs + 18 + int(fs * 0.8) + 12], fill=(255, 255, 255))
        d.text((16, 8 + fs + 23), note, fill=(18, 32, 43), font=fn)
    for x0, txt in ((th.width + 8, "WORKSPACE"), (th.width + P + 8, "WRIST")):
        d.rectangle([x0, P - fs - 20, x0 + d.textlength(txt, font=fn) + 14, P - 8], fill=(18, 32, 43))
        d.text((x0 + 7, P - fs - 14), txt, fill="white", font=fn)
    return np.asarray(out)


def main(args):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from fd_core.env import Franka, make_env
    from fd_core.policy import camera_setup, load_dp
    from fd_core.rollout import DiffusionActor, run_batch

    torch.manual_seed(args.seed)
    wrist, zoom = camera_setup(args.checkpoint)
    env, raw = make_env(args.num_envs, args.device, args.seed, 128, video_size=(480, 360), wrist=wrist, zoom=zoom)
    fr = Franka(raw)
    model, cfg = load_dp(args.checkpoint, args.device)
    r = run_batch(env, raw, fr, DiffusionActor(model, cfg.exec_horizon), obs_horizon=cfg.obs_horizon, record=True,
                  video_cam="video_camera")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = ["success", "never_lifted", "dropped", "held_off_goal", "fell_off_table"]
    labels = ["SUCCESS", "FAIL · never lifted", "FAIL · dropped", "FAIL · off goal", "FAIL · fell off"]
    summary = []
    with torch.no_grad():
        for i in range(args.num_envs):
            o = int(r["outcome"][i]); imgs = r["image"][i]          # [T,128,256,3] uint8
            heat, kp, kw = attention(model, imgs)
            np.savez_compressed(args.out_dir / f"att_s{args.seed}_e{i}.npz", image=imgs.numpy(), frames=r["frames"][i].numpy(),
                                heat=heat.numpy(), kp=kp.numpy(), kw=kw.numpy(), outcome=o)
            path = args.out_dir / f"att_s{args.seed}_e{i}_{names[o]}.mp4"
            wr = imageio.get_writer(path, fps=25, codec="libx264", quality=8, macro_block_size=8)
            for t in range(imgs.shape[0]):
                wr.append_data(compose(r["frames"][i, t].numpy(), imgs[t].numpy(), heat[t], kp[t], kw, args.panel,
                                       labels[o], o == 0, t))
            wr.close()
            # share of attention mass on the wrist half, per frame (for the report)
            wshare = heat[:, :, heat.shape[2] // 2:].sum((1, 2)) / heat.sum((1, 2))
            summary.append({"env": i, "outcome": names[o], "file": path.name, "wrist_share": [round(float(v), 3) for v in wshare]})
    (args.out_dir / f"att_s{args.seed}.json").write_text(json.dumps(summary))
    print("[att] " + json.dumps([(s["env"], s["outcome"]) for s in summary]), flush=True)


if __name__ == "__main__":
    cli = parse_args()
    cli.enable_cameras = True; cli.headless = True
    app = AppLauncher(cli).app
    code = 0
    try:
        main(cli)
    except BaseException:
        traceback.print_exc(); sys.stdout.flush(); code = 1
    os._exit(code)
