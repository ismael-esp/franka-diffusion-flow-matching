#!/usr/bin/env python3
"""Page 4 diagrams: the shared policy network and how diffusion / flow matching learn.

Writes report/diagrams/{architecture,learning_diffusion,learning_flow,learning}.svg (explicit colours, no CSS, so the video
can reuse them) and report/page4.html (page4.template.html with the SVGs inlined).
Real vs schematic: every size, count and hyper-parameter is read from the code / checkpoints (see report/RESULTS.md, page 4);
the five small multiples per method are a REAL demo chunk (runs/drawer_s/demos_clean) corrupted with the REAL cosine schedule
(dp_core.model.cosine_alphas_cumprod) for diffusion and the straight blend for flow; the sampling-path plots are SCHEMATIC
(one action coordinate, a perfect network), using the real DDIM step grid and schedule.
Run: PYTHONPATH=pouring_diffusion python tools/make_diagrams.py   (python from the isaaclab env: numpy + torch)"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
PROJ = R.parent
sys.path.insert(0, os.environ.get("DP_ROOT", str(Path(__file__).resolve().parents[2] / "pouring_diffusion")))
from dp_core.model import cosine_alphas_cumprod, ddim_timesteps  # noqa: E402

INK, MUTED, LINE, REF = "#12202b", "#55646f", "#d3dbe2", "#8b98a4"
SLATE, SLATE_T = "#3d5568", "#e6ecf1"
DP, DP_T = "#df6327", "#fbe6da"
FM, FM_T = "#149484", "#d9f0ed"
FONT = "Liberation Sans, Arial, Helvetica, sans-serif"
DISPLAY = "Liberation Sans Narrow, Arial Narrow, Liberation Sans, Arial, sans-serif"


# ----------------------------------------------------------------------------------------------------------- primitives
def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12.5, weight=400, fill=INK, anchor="start", font=FONT, extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{extra}>{s}</text>'


def sub(a, b, c=""):
    """a, subscript b, then c (tspans with dy; cairosvg and browsers both honour it)."""
    return f'{esc(a)}<tspan dy="3" font-size="9.5">{esc(b)}</tspan><tspan dy="-3">{esc(c)}</tspan>'


def box(x, y, w, h, lines, fill=SLATE_T, stroke=SLATE, size=12.5, tcolor=INK, bold_first=True, rx=6, sw=1.5, lh=None, extra=""):
    lh = lh or size * 1.28
    out = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra}/>']
    y0 = y + h / 2 - (len(lines) - 1) * lh / 2 + size * 0.34
    for i, ln in enumerate(lines):
        col = tcolor
        wt = 700 if (bold_first and i == 0) else 400
        if isinstance(ln, tuple):
            ln, col = ln
        out.append(text(x + w / 2, y0 + i * lh, ln, size, wt, col, "middle"))
    return "\n".join(out)


def arrow(pts, color=MUTED, w=1.7, head=7, extra=""):
    """Polyline with an arrow head at the end (explicit polygon: no markers, so every renderer agrees)."""
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    d = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * math.cos(d), y2 - head * math.sin(d)
    px, py = -math.sin(d) * head * 0.55, math.cos(d) * head * 0.55
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts[:-1] + [(bx, by)])
    return (f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="{w}" stroke-linejoin="round"{extra}/>'
            f'<polygon points="{x2:.1f},{y2:.1f} {bx+px:.1f},{by+py:.1f} {bx-px:.1f},{by-py:.1f}" fill="{color}"/>')


def svg(w, h, body, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{esc(label)}">\n'
            + body + "\n</svg>")


# ----------------------------------------------------------------------------------------------------------- architecture
def architecture():
    o = []
    for x, s in [(0, "Observation encoder"), (790, "Transformer encoder"), (1070, "Transformer decoder"), (1370, "Output")]:
        o.append(text(x, 20, s, 14, 700, INK))
    # row A: image chain
    o += [box(0, 40, 150, 62, ["Image, 2 steps", "128 × 256 pixels", "workspace | wrist"], size=12),
          box(178, 40, 120, 62, ["ResNet18", "GroupNorm", "4 × 8 feature grid"], size=12),
          box(326, 40, 168, 62, ["Spatial softmax", "32 keypoints", "→ 64 coordinates"], size=12),
          box(522, 40, 110, 62, ["Linear", "64 → 256"], size=12)]
    o += [arrow([(150, 71), (178, 71)]), arrow([(298, 71), (326, 71)]), arrow([(494, 71), (522, 71)]), arrow([(632, 71), (660, 71)])]
    # row B: joint chain
    o += [box(0, 122, 150, 62, ["Joint state, 2 steps", "18 numbers (drawer)", "21 numbers (lift)"], size=12),
          box(178, 122, 120, 62, ["Normalise,", "clip to ±5,", "Linear → 256"], size=12)]
    o += [arrow([(150, 153), (178, 153)]), arrow([(298, 153), (660, 153)])]
    o.append(text(480, 143, "a learned step and kind embedding is added to each token", 11.5, 400, MUTED, "middle"))
    # row C: timestep chain
    o += [box(0, 204, 150, 62, ["Timestep input", ("diffusion: t = 0 … 99", DP), ("flow: t × 100", FM)], size=12),
          box(178, 204, 316, 62, ["Sinusoidal embedding, 256 numbers", "MLP 256 → 1024 → 256 (Mish)"], size=12)]
    o += [arrow([(150, 235), (178, 235)]), arrow([(494, 235), (660, 235)])]
    # tokens
    o += [box(660, 40, 92, 62, ["image", "token × 2"], fill="#ffffff", size=12),
          box(660, 122, 92, 62, ["joint", "token × 2"], fill="#ffffff", size=12),
          box(660, 204, 92, 62, ["timestep", "token × 1"], fill="#ffffff", size=12)]
    o += [arrow([(752, 71), (770, 71), (770, 120), (790, 120)]), arrow([(752, 153), (790, 153)]), arrow([(752, 235), (770, 235), (770, 186), (790, 186)])]
    # encoder / decoder
    o.append(box(790, 40, 200, 226, ["Transformer", "encoder", "", "2 layers, 4 heads", "width 256, feed-forward 1024", "pre-norm, no dropout", "", "5 tokens in,", "5 memory tokens out"], size=12.5))
    o.append(arrow([(990, 153), (1070, 153)]))
    o.append(text(1030, 143, "memory", 11.5, 400, MUTED, "middle"))
    o.append(box(1070, 40, 260, 226, ["Transformer decoder", "", "4 layers, 4 heads, width 256", "pre-norm, GELU, no dropout", "", "self-attention over the", "16 action tokens,", "cross-attention to the", "5 memory tokens"], size=12.5))
    # action chunk input
    o.append(text(790, 296, "Action chunk going in", 14, 700, INK))
    o += [box(790, 306, 200, 62, ["16 × 8 action chunk", ("diffusion: noisy chunk", DP), ("flow: noise-to-action blend", FM)], size=12),
          box(1070, 306, 260, 62, ["Linear 8 → 256", "+ position embedding (16 steps)"], size=12)]
    o += [arrow([(990, 337), (1070, 337)]), arrow([(1200, 306), (1200, 266)])]
    # output
    o += [box(1370, 96, 166, 84, ["LayerNorm,", "Linear 256 → 8"], size=12.5),
          box(1370, 214, 166, 92, ["16 × 8 output", ("diffusion: the noise ε", DP), ("flow: the velocity v", FM)], size=12.5),
          box(1370, 330, 166, 62, ["Execute the first 8", "of 16 actions, then", "plan again"], fill="#ffffff", size=12, bold_first=False)]
    o += [arrow([(1330, 138), (1370, 138)]), arrow([(1453, 180), (1453, 214)]), arrow([(1453, 306), (1453, 330)])]
    return svg(1536, 404, "\n".join(o), "Policy network shared by diffusion and flow matching: image and joint tokens, a timestep token, a 2-layer transformer encoder, a 4-layer transformer decoder over a 16-step action chunk")



# ----------------------------------------------------------------------------------------------------------- one diagram per model
def arch_model(kind):
    """Training step and sampling loop of ONE model. Slate = literally the same code in both models, colour = what differs.
    Facts: dp_core/model.py:70-125 (encoder, denoiser), 157-164 (encode), 166-171 (diffusion loss), 179-198 (DDIM),
    dp_core/flow.py:19-35 (flow loss and Euler sampler)."""
    dp = kind == "dp"
    col, tint = (DP, DP_T) if dp else (FM, FM_T)
    o = [text(0, 14, "Training: one gradient step", 14, 700, INK)]
    Y, H = 26, 122
    mid = Y + H / 2
    o.append(box(0, Y, 112, H, ["Demo chunk a", "16 × 8 actions", "", "+ 2 past frames", "and joint state"], fill="#ffffff", stroke=SLATE, size=12))
    if dp:
        l2 = ["Add noise (schedule)", "t: integer 0 … 99", "ε ~ N(0, I)", "x_t = √α_t · a + √(1 − α_t) · ε", "", "cosine schedule, t = 99 is noise"]
        l4 = ["Network output", "the noise ε̂", "", "loss = MSE(ε̂, ε)", "model.py:171"]
        tin = "time input: t (integer 0 … 99)"
    else:
        l2 = ["Blend (straight line)", "t ~ U(0, 1), ε ~ N(0, I)", "", "x_t = (1 − t) · ε + t · a", "", "no schedule; t = 0 is noise,", "t = 1 is the action"]
        l4 = ["Network output", "the velocity v̂", "", "loss = MSE(v̂, a − ε)", "flow.py:24"]
        tin = "time input: t × 100 (float)"
    o.append(box(144, Y, 206, H, l2, fill=tint, stroke=col, size=12, lh=15.5))
    o.append(box(382, Y, 200, H, ["Network (same code)", "ResNet18 + spatial softmax", "→ 4 observation tokens", "+ 1 time token, encoder 2 layers", "decoder 4 layers over 16 tokens", "Linear 256 → 8", (tin, col)], fill=SLATE_T, stroke=SLATE, size=11.5, lh=15.5))
    o.append(box(614, Y, 146, H, l4, fill=tint, stroke=col, size=12, lh=15.5))
    for x1, x2 in ((112, 144), (350, 382), (582, 614)):
        o.append(arrow([(x1, mid), (x2, mid)], head=6))
    # sampling
    o.append(text(0, 176, "Sampling: every 8 environment steps, gives the 16 × 8 chunk", 14, 700, INK))
    SY, SH = 190, 160
    label = ("repeat 10 times,  t = 90, 80, … 0" if dp else "repeat k times (k = 10 here),  t = i / k")
    o.append(box(0, SY + 20, 112, 90, ["Start", "pure noise", "x ~ N(0, I)", "16 × 8"], fill="#ffffff", stroke=SLATE, size=12))
    o.append(f'<rect x="136" y="{SY}" width="496" height="{SH}" rx="10" fill="none" stroke="{col}" stroke-width="1.6" stroke-dasharray="6 4"/>')
    o.append(text(148, SY + 18, label, 12.5, 700, col))
    o.append(box(150, SY + 34, 132, 96, ["Network call", "same code", "in: x, time, tokens", ("out: " + ("ε̂" if dp else "v̂"), col)], fill=SLATE_T, stroke=SLATE, size=12, lh=16))
    if dp:
        up = ["DDIM update (deterministic)",
              "x0 = (x − √(1 − α_t) · ε̂) / √α_t",
              "x0 is clipped to ±1 at every step",
              "x ← √α_prev · x0 + √(1 − α_prev) · ε̂",
              "last call: α_prev = 1, so x = x0"]
    else:
        up = ["Euler step", "x ← x + v̂ / k", "", "a straight step of size 1 / k,", "no clipping inside the loop"]
    o.append(box(310, SY + 34, 308, 96, up, fill=tint, stroke=col, size=12, lh=16))
    o.append(arrow([(282, SY + 82), (310, SY + 82)], head=6))
    o.append(arrow([(464, SY + 130), (464, SY + 148), (216, SY + 148), (216, SY + 130)], color=col, head=6))
    o.append(text(340, SY + 143, "new x goes back in", 11.5, 400, MUTED, "middle"))
    o.append(box(662, SY + 20, 98, 90, ["Chunk", "clamp to ±1,", "gripper → ±1,", "run first 8"], fill="#ffffff", stroke=SLATE, size=12, bold_first=True))
    o.append(arrow([(112, SY + 65), (136, SY + 65)], head=6))
    o.append(arrow([(632, SY + 65), (662, SY + 65)], head=6))
    # legend + number
    ly = SY + SH + 28
    o.append(f'<rect x="0" y="{ly - 11}" width="14" height="14" rx="3" fill="{SLATE_T}" stroke="{SLATE}" stroke-width="1.5"/>')
    o.append(text(22, ly, "same code in both models", 12, 400, MUTED))
    o.append(f'<rect x="208" y="{ly - 11}" width="14" height="14" rx="3" fill="{tint}" stroke="{col}" stroke-width="1.5"/>')
    o.append(text(230, ly, "differs between diffusion and flow matching", 12, 400, MUTED))
    o.append(text(760, ly + 2, "10 network calls per chunk" if dp else "1, 5 or 10 network calls per chunk", 18, 800, col, "end", font=DISPLAY))
    name = "Diffusion policy" if dp else "Flow-matching policy"
    return svg(760, 396, "\n".join(o), name + ": training step (noise or velocity target) and sampling loop (DDIM or Euler)")

# ----------------------------------------------------------------------------------------------------------- real data
def demo_chunk():
    """A real 16-step window of normalised drawer-demo actions: the gripper command (dim 7, +1 open to -1 closed) switching in the middle of the window
    (an arm-joint window moves by at most 0.4 and would vanish next to unit noise)."""
    src = PROJ / "runs/drawer_s/demos_clean/action.npy"
    if not src.exists():   # clean checkout: use the chunk saved by an earlier run (report/data/arch_chunk.json)
        return json.loads((R / "data/arch_chunk.json").read_text())["chunk"]
    a = np.load(src, mmap_mode="r")
    for ep in range(0, 40):
        for st in range(0, 220):
            c = np.asarray(a[ep, st:st + 16, 7]).copy()
            if c[0] >= 0.95 and c[-1] <= -0.95 and 6 <= int(np.argmax(c < 0)) <= 9:
                return {"episode": int(ep), "start": int(st), "dim": 7, "values": [float(v) for v in c]}
    raise RuntimeError("no gripper step window")


def pick_noise(a8):
    """A seeded noise draw (16 values) whose step-8 value is on the other side of the demo value, so the paths are readable."""
    for seed in range(1, 200):
        n = np.random.default_rng(seed).standard_normal(16)
        if 0.7 < abs(n[8]) < 1.25 and abs(n).max() < 2.0 and (a8 >= 0) == (n[8] < 0):
            return seed, n
    raise RuntimeError("no seed")


# ----------------------------------------------------------------------------------------------------------- learning panels
def multiples(chunk, noise, kind, ab):
    """Five small multiples (demo dashed, corrupted solid) for diffusion t = 0, 25, 50, 75, 99 or flow t = 1, .75, .5, .25, 0."""
    col = DP if kind == "dp" else FM
    o = []
    pw, gap, y0, ph = 137, 15, 36, 122
    lo, hi = -2.0, 2.0
    sy = lambda v: y0 + ph - (v - lo) / (hi - lo) * ph
    sx = lambda i, x0: x0 + 8 + i * (pw - 16) / 15
    if kind == "dp":
        steps = [0, 25, 50, 75, 99]
        caps = [(f"t = {t}", f"α = {float(ab[t]):.2f}") for t in steps]
        xs = [math.sqrt(float(ab[t])) * np.array(chunk) + math.sqrt(1 - float(ab[t])) * noise for t in steps]
    else:
        steps = [1.0, 0.75, 0.5, 0.25, 0.0]
        caps = [(f"t = {t:.2f}", ("demo" if t == 1 else "noise" if t == 0 else "blend")) for t in steps]
        xs = [(1 - t) * noise + t * np.array(chunk) for t in steps]
    for k, (x, cap) in enumerate(zip(xs, caps)):
        x0 = k * (pw + gap)
        o.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#ffffff" stroke="{LINE}" stroke-width="1"/>')
        o.append(f'<line x1="{x0}" y1="{sy(0):.1f}" x2="{x0+pw}" y2="{sy(0):.1f}" stroke="{LINE}" stroke-width="1"/>')
        demo = " ".join(f"{sx(i, x0):.1f},{sy(v):.1f}" for i, v in enumerate(chunk))
        o.append(f'<polyline points="{demo}" fill="none" stroke="{REF}" stroke-width="1.6" stroke-dasharray="4 3"/>')
        pts = " ".join(f"{sx(i, x0):.1f},{sy(float(np.clip(v, lo + .05, hi - .05))):.1f}" for i, v in enumerate(x))
        o.append(f'<polyline class="draw" style="--d:{0.15 + 0.15 * k:.2f}s" points="{pts}" fill="none" stroke="{col}" stroke-width="2.2" stroke-linejoin="round"/>')
        o.append(text(x0 + pw / 2, y0 + ph + 17, cap[0], 12.5, 700, INK, "middle"))
        o.append(text(x0 + pw / 2, y0 + ph + 33, cap[1], 12, 400, MUTED, "middle"))
    return "\n".join(o)


def path_plot(kind, a, eps, ab, ddim):
    """Schematic sampling path of one action coordinate with a perfect network; dots are network calls."""
    col = DP if kind == "dp" else FM
    X0, X1, Y0, Y1 = 52, 452, 8, 150                  # plot area
    lo, hi = -1.6, 1.6
    sy = lambda v: Y1 - (v - lo) / (hi - lo) * (Y1 - Y0)
    sx = lambda p: X0 + p * (X1 - X0)
    o = [f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="{MUTED}" stroke-width="1.2"/>',
         f'<line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}" stroke="{MUTED}" stroke-width="1.2"/>',
         f'<line x1="{X0}" y1="{sy(a):.1f}" x2="{X1}" y2="{sy(a):.1f}" stroke="{REF}" stroke-width="1.2" stroke-dasharray="4 3"/>']
    for p in (0, .2, .4, .6, .8, 1):
        o.append(f'<line x1="{sx(p):.1f}" y1="{Y1}" x2="{sx(p):.1f}" y2="{Y1+4}" stroke="{MUTED}" stroke-width="1.2"/>')
        o.append(text(sx(p), Y1 + 17, f"{p:g}", 11.5, 400, MUTED, "middle"))
    o.append(text((X0 + X1) / 2, Y1 + 35, "share of the way from noise to action", 12, 400, MUTED, "middle"))
    o.append(text(X0 - 8, sy(eps) + 4, "noise", 11.5, 400, MUTED, "end"))
    o.append(text(X0 - 8, sy(a) + 4, "demo", 11.5, 400, MUTED, "end"))
    if kind == "dp":
        T = [int(t) for t in ddim]                    # 90, 80, ..., 0
        abs_ = [float(ab[t]) for t in T[1:]] + [1.0]  # alpha after each of the 10 calls
        ys = [eps] + [math.sqrt(al) * a + math.sqrt(1 - al) * eps for al in abs_]
        ps = [i / 10 for i in range(11)]
        pts = " ".join(f"{sx(p):.1f},{sy(v):.1f}" for p, v in zip(ps, ys))
        o.append(f'<polyline class="draw" points="{pts}" fill="none" stroke="{col}" stroke-width="2.4" stroke-linejoin="round"/>')
        for i, (p, v) in enumerate(zip(ps, ys)):
            o.append(f'<circle cx="{sx(p):.1f}" cy="{sy(v):.1f}" r="{4.6 if i in (0, 10) else 3.8}" fill="{col if i else "#ffffff"}" stroke="{col}" stroke-width="2"/>')
    else:
        o.append(f'<line class="draw" x1="{sx(0):.1f}" y1="{sy(eps):.1f}" x2="{sx(1):.1f}" y2="{sy(a):.1f}" stroke="{col}" stroke-width="2.4"/>')
        for i in range(11):
            p = i / 10
            o.append(f'<circle cx="{sx(p):.1f}" cy="{sy(eps + p * (a - eps)):.1f}" r="3.2" fill="#ffffff" stroke="{col}" stroke-width="1.6"/>')
        for i in range(6):
            p = i / 5
            o.append(f'<circle cx="{sx(p):.1f}" cy="{sy(eps + p * (a - eps)):.1f}" r="4.8" fill="{col}" stroke="{col}" stroke-width="2"/>')
        o.append(arrow([(sx(0.0) + 6, sy(eps) - 16), (sx(1.0) - 6, sy(a) - 16)], color=SLATE, w=1.5, extra=' stroke-dasharray="5 4"'))
    return "\n".join(o)


def loop_strip(y, kind):
    col, tint = (DP, DP_T) if kind == "dp" else (FM, FM_T)
    pred = "noise ε" if kind == "dp" else "velocity v"
    items = [["Demo chunk", "16 × 8 actions"], ["Corrupt it", "random t each step"], ["Network predicts", pred], ["Squared error", "against the target"], ["AdamW step", "EMA of the weights"]]
    o = []
    for i, ln in enumerate(items):
        x = i * 155
        o.append(box(x, y, 125, 50, ln, fill=tint if i in (1, 2) else "#ffffff", stroke=col if i in (1, 2) else SLATE, size=12, lh=15))
        if i < 4:
            o.append(arrow([(x + 125, y + 25), (x + 155, y + 25)], color=MUTED, head=6))
    return "\n".join(o)


def learning(kind, chunk, noise, ab, ddim, a_val, eps_val):
    col = DP if kind == "dp" else FM
    o = [text(0, 16, "Training: corrupt a real demo chunk (dashed: a drawer demo's gripper command), then undo it", 13, 700, INK)]
    o.append(multiples(chunk, noise, kind, ab))
    if kind == "dp":
        o.append(text(0, 226, sub("input:   √α", "t", " · demo + √(1 − α") + sub("", "t", ") · noise,   t picked at random from 0 … 99"), 13, 400, INK))
        o.append(text(0, 246, "target: the noise that was added  (predict ε, loss = mean squared error)", 13, 400, INK))
    else:
        o.append(text(0, 226, "input:   (1 − t) · noise + t · demo,   t picked at random between 0 and 1", 13, 400, INK))
        o.append(text(0, 246, "target: the velocity  demo − noise  (predict v, loss = mean squared error)", 13, 400, INK))
    o.append(f'<g transform="translate(0,276)">')
    o.append(text(0, 0, "Sampling: from pure noise to an action (schematic, one coordinate, a perfect network)", 13, 700, INK))
    o.append(f'<g transform="translate(0,14)">{path_plot(kind, a_val, eps_val, ab, ddim)}</g>')
    tx = 492
    if kind == "dp":
        o += [text(tx, 38, "10 network calls", 24, 800, DP, font=DISPLAY),
              text(tx, 62, "DDIM, deterministic (η = 0)", 13, 400, INK),
              text(tx, 80, "100 training steps → 10 sampling steps", 13, 400, INK),
              text(tx, 104, "the path curves: the noise level follows", 13, 400, MUTED),
              text(tx, 122, "a cosine schedule", 13, 400, MUTED),
              text(tx, 150, "lift, seed 1001: 63.0 % at 10 calls,", 13, 700, INK),
              text(tx, 168, "60.5 % at 5 (1 call: n/c, not comparable)", 13, 700, INK)]
    else:
        o += [text(tx, 38, "1, 5 or 10 calls", 24, 800, FM, font=DISPLAY),
              text(tx, 62, "Euler steps along a straight line,", 13, 400, INK),
              text(tx, 80, "the result is clamped to ±1", 13, 400, INK),
              text(tx, 104, "filled dots: 5 calls, rings: 10 calls,", 13, 400, MUTED),
              text(tx, 122, "dashed arrow: 1 call", 13, 400, MUTED),
              text(tx, 150, "lift, seed 1001: 77.0 % at 10 calls,", 13, 700, INK),
              text(tx, 168, "81.0 % at 5, 66.0 % at 1", 13, 700, INK)]
    o.append("</g>")
    return svg(745, 500, "\n".join(o), ("Diffusion" if kind == "dp" else "Flow matching") + " learning: corrupted demo chunks, the sampling path and the training loop")


# ----------------------------------------------------------------------------------------------------------- main
def main():
    (R / "diagrams").mkdir(exist_ok=True)
    ab = cosine_alphas_cumprod(100)
    ddim = ddim_timesteps(100, 10)
    ch = demo_chunk()
    seed, noise = pick_noise(ch["values"][8])
    a_val, eps_val = ch["values"][8], float(noise[8])
    json.dump({"chunk": ch, "noise_seed": seed, "alphas_cumprod_at_0_25_50_75_99": [float(ab[t]) for t in (0, 25, 50, 75, 99)],
               "ddim_steps": [int(t) for t in ddim]}, open(R / "data/arch_chunk.json", "w"), indent=1)
    arch = architecture()
    ld, lf = learning("dp", ch["values"], noise, ab, ddim, a_val, eps_val), learning("fm", ch["values"], noise, ab, ddim, a_val, eps_val)
    (R / "diagrams/architecture.svg").write_text(arch)
    (R / "diagrams/learning_diffusion.svg").write_text(ld)
    (R / "diagrams/learning_flow.svg").write_text(lf)
    inner = lambda s: s[s.index(">") + 1: s.rindex("</svg>")]
    both = svg(1530, 500, f'<g>{inner(ld)}</g><g transform="translate(785,0)">{inner(lf)}</g>', "Diffusion (left) and flow matching (right): how each one learns")
    (R / "diagrams/learning.svg").write_text(both)
    ad, af = arch_model("dp"), arch_model("fm")
    (R / "diagrams/architecture_diffusion.svg").write_text(ad)
    (R / "diagrams/architecture_flow.svg").write_text(af)
    tpl = (R / "page4.template.html").read_text()
    html = tpl.replace("<!--SVG:architecture-->", arch).replace("<!--SVG:arch_dp-->", ad).replace("<!--SVG:arch_fm-->", af).replace("<!--SVG:diffusion-->", ld).replace("<!--SVG:flow-->", lf)
    (R / "page4.html").write_text(html)
    print("chunk", ch["episode"], ch["start"], ch["dim"], "noise seed", seed, "a8=%.2f eps8=%.2f" % (a_val, eps_val))


if __name__ == "__main__":
    main()
