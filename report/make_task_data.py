#!/usr/bin/env python3
"""Task video data (drawer, stack, lift) -> data/task_<task>.json for the Remotion scene.
python make_task_data.py drawer --data-diffusion runs/task_drawer/b0 --data-flow runs/task_drawer/b0 --envs-diffusion 0 4 --envs-flow 0 4 --sim runs/sim_drawer.npz"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from geom import fk, project, project_cam
from task_select import ep_info, marker_pos3, RATE
root = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("task"); ap.add_argument("--sim"); ap.add_argument("--out")
ap.add_argument("--base")
for m in ("diffusion", "flow"): ap.add_argument(f"--envs-{m}", nargs="+")
args = ap.parse_args()
task = args.task
EYE_T = {"drawer": ((-0.35, -1.25, 1.35), (0.55, 0.0, 0.5)), "stack": ((1.45, 0.75, 0.75), (0.42, 0.0, 0.18))}
GX, GY, X0, X1, Y0, Y1, SIG = 18, 13, 90, 570, 40, 400, 38.0
rd = lambda a, n=1: np.round(a, n).tolist()
eye, tgt = EYE_T[task]
P3 = lambda q: project_cam(q, eye, tgt)[0]
pxs = lambda q: P3(fk(np.asarray(q, float)))
R = np.load(root / args.sim)
al = json.load(open(root / "report_v2/data/denoise_lift_e1.json"))["alphas_cumprod"]
out = {"task": task, "hz": RATE[task], "alphas_cumprod": al, "grid": {"gx": GX, "gy": GY, "x0": X0, "x1": X1, "y0": Y0, "y1": Y1}, "models": {}}
table, checks = [], []
for name in ("diffusion", "flow"):
    atts = []
    for tok in getattr(args, "envs_" + name):
        b_, e = tok.split(":"); e = int(e); d = np.load(root / args.base / b_ / f"{name}_iterates.npz"); qs = d["query_step"]; keep = list(d["keep_envs"]); it = d["many_iter"]; cid = f"{b_}e{e}"
        info = ep_info(d, e, task); cut = info["cut"]; ke = keep.index(e)
        q9 = d["qpos9"][e].astype(float); tcp = fk(q9[:, :7]); apx = pxs(q9[:cut + 1, :7] if False else q9[:cut + 1, :7])
        mk3 = marker_pos3(d, e, task).astype(float)
        markers = []
        if task == "drawer":
            h = d["handle"][e].astype(float)
            if info["success"]:
                dirv = h[info["t_succ"]] - h[0]; dirv = dirv / np.linalg.norm(dirv)
            else:
                yaw = 2 * np.arctan2(d["cab_root"][e, 0, 6], d["cab_root"][e, 0, 3]); dirv = np.array([-np.cos(yaw), -np.sin(yaw), 0])
            tgt3 = h[0] + 0.30 * dirv
            markers = [{"label": "handle", "kind": "square", "t0": 0, "t1": cut, "px": rd(P3(h[:cut + 1]))},
                       {"label": "open target", "kind": "cross", "t0": 0, "t1": cut, "px": rd(P3(tgt3))}]
            goal3 = tgt3
        else:
            c = d["cubes"][e].astype(float); ok12 = d["ok12"][e][:, 0] > 0.5
            t12 = int(np.argmax(ok12)) if ok12.any() else cut + 1
            t12 = min(t12, cut + 1)
            tgt1 = c[0, 0, :3] + [0, 0, 0.0468]
            markers = [{"label": "cube 2", "kind": "square", "t0": 0, "t1": min(cut, t12), "px": rd(P3(c[:cut + 1, 1, :3]))},
                       {"label": "target: on cube 1", "kind": "cross", "t0": 0, "t1": min(cut, t12), "px": rd(P3(tgt1))}]
            if t12 <= cut:
                markers += [{"label": "cube 3", "kind": "square", "t0": t12, "t1": cut, "px": rd(P3(c[:cut + 1, 2, :3]))},
                            {"label": "target: on cube 2", "kind": "cross", "t0": t12, "t1": cut, "px": rd(P3(c[t12, 1, :3] + [0, 0, 0.0468]))}]
            goal3 = tgt1
        fin = R[f"{name}_{cid}_final"]; c0 = int(R[f"{name}_{cid}_single"]); s0 = int(qs[c0])
        # validation: simulator re-simulation of the executed chunk vs the recorded fingertip path
        verr = []
        for ci in range(fin.shape[0]):
            s = int(qs[ci]); n = min(8, len(q9) - s - 1)
            if n > 0: verr.append(float(np.linalg.norm(pxs(fin[ci, 0, :n]) - pxs(q9[s + 1:s + 1 + n, :7]), axis=-1).mean()))
        P = pxs(R[f"{name}_{cid}_{c0}_iter"]); rp = {"call": c0, "step": s0, "iter": rd(P)}
        if name == "flow":
            P1 = pxs(R[f"{name}_{cid}_{c0}_a1"]); disp = P1 - P[:10]; rp["vel"] = rd(disp[:, :, 9], 1)
            gx, gy = np.meshgrid(np.linspace(X0, X1, GX), np.linspace(Y0, Y1, GY)); fld = np.zeros((10, GY, GX, 3))
            for i in range(10):
                pos = P[i].reshape(-1, 2); dv = disp[i].reshape(-1, 2); ok = np.abs(pos - [320, 240]).max(1) < 2000
                w = np.exp(-(((gx[..., None] - pos[ok, 0]) ** 2 + (gy[..., None] - pos[ok, 1]) ** 2) / (2 * SIG ** 2))); ws = w.sum(-1)
                fld[i, ..., 0] = (w * dv[ok, 0]).sum(-1) / (ws + 1e-9); fld[i, ..., 1] = (w * dv[ok, 1]).sum(-1) / (ws + 1e-9); fld[i, ..., 2] = ws
            rp["field"] = rd(fld, 1)
        T = fk(fin[c0].astype(float)); chunk = it[c0, -1, ke, :, :, 7]; end = T[:, 15]
        m = {"model": name, "env": e, "call": c0 + 1, "step": s0, "now_to_marker_m": float(np.linalg.norm(tcp[s0] - mk3[s0])),
             "end16_to_marker_m": float(np.linalg.norm(end[0] - mk3[s0])), "end16_to_marker_px": float(np.linalg.norm(P3(end[0]) - P3(mk3[s0]))),
             "end16_to_target_m": float(np.linalg.norm(end[0] - goal3)), "end16_to_target_px": float(np.linalg.norm(P3(end[0]) - P3(goal3))),
             "spread_mm": float(1000 * end.std(0).mean()), "val_px_mean": float(np.mean(verr)), "val_px_max": float(np.max(verr))}
        kc = int(np.argmax(chunk[0] < 0)) if (chunk[0] < 0).any() else -1
        m["close_index"] = kc; m["close_to_marker_m"] = float(np.linalg.norm(T[0, kc] - mk3[s0])) if kc >= 0 else None
        rp["metrics"] = m; table.append(m)
        if info["t_close"] is not None:   # marker placement: fingertip at the close command vs the marker
            tcl = info["t_close"]; checks.append({"model": name, "env": e, "t_close": tcl, "tcp_to_marker_m": float(np.linalg.norm(tcp[tcl] - mk3[tcl])),
                                                  "tcp_to_marker_px": float(np.linalg.norm(P3(tcp[tcl]) - P3(mk3[tcl])))})
        # hand FK vs simulator hand (env frame)
        hand_err = float(np.abs(fk(q9[:, :7], tcp=False) - d["hand"][e]).max())
        atts.append({"env": e, "clip": cid, "success": info["success"], "t_succ": info["t_succ"], "t_close": info["t_close"], "cut": cut, "last_call": int(fin.shape[0] - 1),
                     "rate": 1 if cut <= 150 else 2, "actual_px": rd(apx), "finals": rd(pxs(fin)), "markers": markers, "replay": rp, "hand_fk_err_m": hand_err})
    out["models"][name] = {"attempts": atts}
out["metrics"] = table; out["marker_checks"] = checks
Path(root / (args.out or f"report_v2/data/task_{task}.json")).write_text(json.dumps(out, separators=(",", ":")))
for m in table:
    print("%-9s env%-2d call %2d step %3d | gripper->marker now %.3f m | end16->marker %.3f m (%.0f px) ->target %.3f m (%.0f px) | close idx %s -> marker %s m | spread %.1f mm | sim-vs-actual px mean %.2f max %.2f" % (
        m["model"], m["env"], m["call"], m["step"], m["now_to_marker_m"], m["end16_to_marker_m"], m["end16_to_marker_px"], m["end16_to_target_m"], m["end16_to_target_px"], m["close_index"],
        "-" if m["close_to_marker_m"] is None else "%.3f" % m["close_to_marker_m"], m["spread_mm"], m["val_px_mean"], m["val_px_max"]))
for c in checks: print("marker check", c)
