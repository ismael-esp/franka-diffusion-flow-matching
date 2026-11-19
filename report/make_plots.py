#!/usr/bin/env python3
"""Standalone plots (PNG 220 dpi + SVG) and CSV tables of every number on the report slide.
Reads report/data/*.json; bar values are the frozen-eval results quoted on the slide."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent
D = R / "data"
BASE = Path(os.environ.get("FD_PLOTS_OUT", R / "export_build"))   # default: overwrite the committed plots/ and data/ in place
OUT = BASE / "plots"; OUT.mkdir(parents=True, exist_ok=True)
TAB = BASE / "data"; TAB.mkdir(parents=True, exist_ok=True)
INK, MUTED, LINE = "#12202b", "#5a6a77", "#d3dbe2"
DP, DPS, TEACH, REF, NEG = "#df6327", "#f3c3a8", "#3d5568", "#8b98a4", "#b8473f"
JC = ["#df6327", "#2f7f9e", "#7a5aa6"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": LINE, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bars(name, title, rows, note):
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows) + 0.9))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, se, kind) in zip(y, rows):
        col = {"teacher": TEACH, "dp": DP, "dp-soft": DPS, "ref": REF, "neg": "white"}[kind]
        ax.barh(yi, v, color=col, height=0.6, edgecolor=NEG if kind == "neg" else "none", hatch="////" if kind == "neg" else None, linewidth=0.8)
        if se:
            ax.errorbar(v, yi, xerr=se, color=INK, capsize=3, lw=1)
        ax.text(v + (se or 0) + 1.2, yi, f"{v:.1f}%", va="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_yticks(y, [r[0] for r in rows], color=INK); ax.set_xlim(0, 105); ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.xaxis.grid(True, color=LINE, lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left"); fig.text(0.01, -0.02, note, fontsize=8, color=MUTED)
    save(fig, name)
    with open(TAB / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["configuration", "success_pct", "se_pct", "kind"]); w.writerows(rows)


bars("pouring_success", "Dual-arm pouring · success rate", [
    ("Teacher · privileged (MAT)", 76.0, 0, "teacher"), ("DP · bc_x2 (fine-tuned, 1,200 demos)", 63.0, 2.41, "dp"), ("DP · no-lag model", 62.5, 2.42, "dp"),
    ("TMAD-RGB (non-diffusion)", 89.2, 0, "ref")],
    "50 DDIM steps · 200 episodes per held-out seed (201, 301) · ±1 SE")
bars("franka_configurations", "Franka cube lift · configurations", [
    ("Teacher · privileged (PPO)", 96.0, 1.39, "teacher"), ("DP · workspace cam", 42.0, 2.47, "dp"), ("DP · + wrist cam", 56.3, 2.48, "dp"),
    ("DP · + recovery demos", 72.5, 2.23, "dp"), ("DP · + zoom cam · test", 75.2, 1.37, "dp"), ("DP · + robust DAgger · test", 76.4, 1.34, "dp"),
    ("DAgger · PPO teacher", 48.0, 2.50, "neg")],
    "400 episodes each (teacher 200) · seeds 201 + 301; 'test' = 1,000 episodes on fresh seeds 1001-1005 · ±1 SE")
bars("franka_inference_settings", "Franka · inference settings", [
    ("Wrist · execute 8 / 16", 57.0, 3.50, "dp"), ("Wrist · execute 4 / 16", 56.5, 3.51, "dp-soft"), ("Wrist · execute 2 / 16", 52.5, 3.53, "dp-soft"),
    ("Workspace · DDIM 10", 42.0, 3.49, "dp"), ("Workspace · DDIM 50", 43.0, 4.95, "dp-soft")],
    "seed 201 · 200 episodes (DDIM 50: 100) · ±1 SE")

occ = json.load(open(D / "occlusion.json"))
fig, ax = plt.subplots(figsize=(4.2, 2.2))
names = ["approach", "grasp", "lift"]; vals = [100 * occ[n]["wrist_share_top10pct"] for n in names]
ax.bar([f"{n}\n{occ[n]['time_s']:.1f} s" for n in names], vals, color=[DPS, DP, DPS], width=0.55)
for i, v in enumerate(vals):
    ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold", color=INK)
ax.set_ylim(0, 100); ax.set_ylabel("wrist-camera share"); ax.yaxis.grid(True, color=LINE, lw=0.6); ax.set_axisbelow(True)
ax.set_title("Where the policy looks · wrist share of top-10% patches", loc="left", fontsize=10)
save(fig, "occlusion_wrist_share")
with open(TAB / "occlusion_wrist_share.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["phase", "time_s", "wrist_share_top10pct"]); [w.writerow([n, occ[n]["time_s"], occ[n]["wrist_share_top10pct"]]) for n in names]

# ---- denoising small multiples
d = json.load(open(D / "denoise.json")); J = d["joints"]; gt = np.array(d["gt_rad"]); tr = np.array(d["trace_rad"])
fig, axs = plt.subplots(1, 5, figsize=(11, 2.6), sharey=True)
for ax, idx, k in zip(axs, [0, 3, 6, 9, 10], [10, 7, 4, 1, 0]):
    ax.axvspan(-0.5, 7.5, color="#eef1f4", zorder=0)
    for c, j in zip(JC, J):
        ax.plot(gt[:, j], color=INK, lw=2.6, zorder=2)
        ax.plot(tr[idx][:, j], color=c, lw=1.6, zorder=3, marker="o", ms=2)
    ax.set_title(f"k = {k}" + (" (noise)" if k == 10 else " (final)" if k == 0 else ""), fontsize=10)
    ax.set_xlabel("chunk step")
axs[0].set_ylabel("joint target (rad)")
fig.suptitle("Denoising one 16-step action chunk (DDIM samples x_k) vs teacher (dark) · shaded = executed steps", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
fig.text(0.01, -0.04, f"Joints J{J[0]+1}, J{J[1]+1}, J{J[2]+1} · held-out state · final MSE {d['final_mse']:.4f} (normalized units)", fontsize=8, color=MUTED)
fig.tight_layout(); save(fig, "denoising_sequence")

# ---- sampling consistency
sm = np.array(d["samples_rad"])
fig, axs = plt.subplots(1, 3, figsize=(9, 2.5))
for ax, c, j in zip(axs, JC, J):
    for s in sm:
        ax.plot(s[:, j], color=c, alpha=0.25, lw=1)
    ax.plot(gt[:, j], color=INK, lw=2.4)
    ax.set_title(f"J{j+1}", fontsize=10); ax.set_xlabel("chunk step")
axs[0].set_ylabel("rad")
fig.suptitle("Sampling consistency · 24 noise seeds (colour) vs teacher (dark)", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
fig.text(0.01, -0.05, f"Deterministic teacher → tight spread expected · spread {d['sample_spread']:.3f} (normalized units)", fontsize=8, color=MUTED)
fig.tight_layout(); save(fig, "sampling_consistency")

# ---- receding horizon
r = json.load(open(D / "receding.json")); eg = np.array(r["episode_gt_rad"]); rm = json.load(open(D / "chunk_rmse.json"))
fig, axs = plt.subplots(2, 1, figsize=(9, 3.8), sharex=True)
for ax, c, j in zip(axs, JC, J[:2]):
    ax.plot(np.arange(250) / 50, eg[:, j], color=INK, lw=2.4, label="teacher")
    for ch in r["chunks_rad"]:
        t0 = ch["t"]; a = np.array(ch["a"])[:, j]; ts = (t0 + np.arange(16)) / 50
        ax.plot(ts[:8], a[:8], color=c, lw=1.6); ax.plot(ts[7:], a[7:], color=c, lw=1, alpha=0.3)
    ax.set_ylabel(f"J{j+1} (rad)")
axs[1].set_xlabel("time (s)")
fig.suptitle("Receding-horizon chunks over one held-out episode (teacher-forced, open-loop) · solid = executed 8, faded = discarded 8",
             x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
fig.text(0.01, -0.03, f"Chunk RMSE {rm['teacher_forced_chunk_rmse_rad_mean']:.3f} rad mean over {rm['n_chunks']} chunks / {rm['episodes']} held-out episodes (p90 {rm['p90']:.3f})", fontsize=8, color=MUTED)
fig.tight_layout(); save(fig, "receding_horizon")

# ---- cube head
c = json.load(open(D / "cube_head.json")); rows = np.array(c["rows"])
fig, ax = plt.subplots(figsize=(5.2, 2.6))
ax.plot(rows[:, 0] / 50, rows[:, 6], color=INK, lw=2.4, label="true cube height")
ax.plot(rows[:, 0] / 50, rows[:, 3], "o", color=DP, ms=3.5, label="predicted from pixels")
ax.set_xlabel("time (s)"); ax.set_ylabel("height in robot frame (m)"); ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("Cube position decoded from the two camera images", loc="left", fontsize=10)
fig.text(0.01, -0.05, f"v3 auxiliary head · held-out validation episode · mean 3-D error {1000 * c['mean_err_m']:.1f} mm", fontsize=8, color=MUTED)
save(fig, "cube_from_pixels")
with open(TAB / "cube_from_pixels.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["step", "pred_x", "pred_y", "pred_z", "true_x", "true_y", "true_z"]); w.writerows(rows.round(5).tolist())

# ---- recovery under injected failures
groups = ["No perturbation", "Missed grasp", "Cube dropped", "Cube knocked", "Pushed in hand (held-out)", "Arm shoved (held-out)", "Late knock (held-out)"]
series = [("Robust teacher", [100, 98, 90, 84, 100, 100, 90], TEACH), ("DP · zoom", [76, 51, 12, 49, 81, 76, 37], DPS), ("DP + robust-teacher DAgger", [74, 54, 38, 56, 75, 75, 43], DP)]
fig, ax = plt.subplots(figsize=(6.4, 4.6)); yb = np.arange(len(groups))[::-1]
for k, (name, vals, c) in enumerate(series):
    ax.barh(yb + (1 - k) * 0.26, vals, height=0.24, color=c, label=name)
    for yv, v in zip(yb + (1 - k) * 0.26, vals):
        ax.text(v + 1, yv, f"{v}%", va="center", fontsize=8, color=INK, fontweight="bold" if k == 2 else "normal")
ax.set_yticks(yb, groups, color=INK); ax.set_xlim(0, 105); ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
ax.xaxis.grid(True, color=LINE, lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.set_title("Recovery under injected failures · success (100 episodes each)", loc="left")
fig.text(0.01, -0.03, "Fixed injection times · seed 401 · held-out kinds never appear in training data", fontsize=8, color=MUTED)
save(fig, "recovery_test")
with open(TAB / "recovery_test.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["perturbation", "robust_teacher", "dp_zoom", "dp_robust_dagger"])
    for g, a, b, c2 in zip(groups, *[s_[1] for s_ in series]): w.writerow([g, a, b, c2])

# ---- summary table
with open(TAB / "summary_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "policy", "success_pct", "se_pct", "episodes", "notes"])
    w.writerows([
        ["franka_lift", "teacher PPO privileged", 96.0, 1.39, 200, "seed 201"],
        ["franka_lift", "DP workspace cam", 42.0, 2.47, 400, "500 demos"],
        ["franka_lift", "DP + wrist cam", 56.3, 2.48, 400, "500 demos"],
        ["franka_lift", "DP + more data", 59.8, 2.45, 400, "1300 demos, +30k steps"],
        ["franka_lift", "DP + recovery demos (1,900 demos)", 72.5, 2.23, 400, "68.5 / 76.5 (best)"],
        ["franka_lift", "DP + DAgger (teacher sees student action)", 48.0, 2.50, 400, "negative result"],
        ["franka_lift", "DP + DAgger (teacher own last action)", 23.3, 2.11, 400, "negative result"],
        ["franka_lift", "DP + zoom cam (fresh test)", 75.2, 1.37, 1000, "seeds 1001-1005; 78.3 on selection seeds"],
        ["franka_lift", "DP + robust-teacher demos + DAgger (fresh test)", 76.4, 1.34, 1000, "seeds 1001-1005, final model"],
        ["franka_lift", "DP v3 literature recipe", "~45", "", "11 runs x 50 (seed 204)", "renderer-sensitive; not in charts"],
        ["pouring", "teacher MAT privileged", 76.0, "", "", "with arm lag"],
        ["pouring", "DP bc_x2_ft1200 (fine-tuned from bc_lag400 on 1200 demos), 50 DDIM", 63.0, 2.41, 400, "seeds 201 / 301: 132/202 and 123/201; NOT bc_lag1200 (51.5 %, seed 201)"],
        ["pouring", "DP no-lag model, 50 DDIM", 62.5, 2.42, 400, "59 / 66"],
        ["pouring", "DP 400 demos, 50 DDIM", 56.0, 3.51, 200, "one seed"],
        ["pouring", "DP bc_x2_ft1200, 5 DDIM", 46.5, 2.49, 400, "seeds 201 / 301: 39.5 / 51.5; NOT bc_lag1200"],
        ["pouring", "TMAD-RGB (thesis, non-diffusion)", 89.2, "", "", "reference; hardware 22/25"],
    ])
print("plots:", sorted(p.name for p in OUT.iterdir()))
