#!/usr/bin/env python3
"""runs/denoise_v4 (record_denoise.py) + runs/sim_paths.npz (sim_chunks.py) -> data/overlay2.json for the two-attempt overlay scenes.
Every drawn path is the SIMULATOR's gripper path (FK of the joints the env reaches when the sampled chunk is applied from the recorded state), projected
into the video camera. Cube / goal are the recorded poses. Metrics per replayed call are written next to the data and printed."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from geom import fk, project
from select_calls import ENVS, episode
root = Path(__file__).resolve().parents[1]
SINGLE = len(sys.argv) > 1 and sys.argv[1] == "single"
R = np.load(root / ("runs/sim_paths_single.npz" if SINGLE else "runs/sim_paths.npz"))
GX, GY, X0, X1, Y0, Y1, SIG = 18, 13, 90, 570, 40, 400, 38.0
rd = lambda a, n=1: np.round(a, n).tolist()
pxs = lambda q: project(fk(np.asarray(q, float)))[0]
al = json.load(open(root / "report_v2/data/denoise_lift_e1.json"))["alphas_cumprod"]
out = {"alphas_cumprod": al, "grid": {"gx": GX, "gy": GY, "x0": X0, "x1": X1, "y0": Y0, "y1": Y1}, "envs": list(ENVS), "models": {}}
table = []
for name in ("diffusion", "flow"):
    d = np.load(root / f"runs/denoise_v4/{name}_iterates.npz"); qs = d["query_step"]; it = d["many_iter"]
    atts = []
    for ai, e in enumerate(ENVS):
        ep = episode(d, e); cut = ep["cut"]
        q = d["qpos"][e].astype(float); tcp = fk(q); cube = d["cube_state"][e][:, :3].astype(float); goal = d["goal"][e].astype(float)
        a = {"env": e, "t_succ": ep["t_succ"], "t_close": ep["t_close"], "cut": cut, "last_call": ep["last_call"],
             "actual_px": rd(pxs(q[:cut + 1])), "cube_px": rd(project(cube[:cut + 1])[0]), "goal_px": rd(project(goal[0])[0]), "cube0": rd(cube[0], 3), "goal": rd(goal[0], 3),
             "finals": rd(pxs(R[f"{name}_{e}_final"])), "replays": []}
        fin = R[f"{name}_{e}_final"]
        if SINGLE: ep["replays"] = [int(R[f"{name}_{e}_single"])]; ep["kinds"] = {ep["replays"][0]: "closest"}
        for c in ep["replays"]:
            s = int(qs[c]); k = ep["kinds"][c]
            P = pxs(R[f"{name}_{e}_{c}_iter"])                       # [11,S,16,2]
            rp = {"call": c, "step": s, "kind": k, "iter": rd(P)}
            if name == "flow":
                P1 = pxs(R[f"{name}_{e}_{c}_a1"]); disp = P1 - P[:10]
                rp["vel"] = rd(disp[:, :, 9], 1)
                gx, gy = np.meshgrid(np.linspace(X0, X1, GX), np.linspace(Y0, Y1, GY)); fld = np.zeros((10, GY, GX, 3))
                for i in range(10):
                    pos = P[i].reshape(-1, 2); dv = disp[i].reshape(-1, 2); ok = np.abs(pos - [320, 240]).max(1) < 2000
                    w = np.exp(-(((gx[..., None] - pos[ok, 0]) ** 2 + (gy[..., None] - pos[ok, 1]) ** 2) / (2 * SIG ** 2))); ws = w.sum(-1)
                    fld[i, ..., 0] = (w * dv[ok, 0]).sum(-1) / (ws + 1e-9); fld[i, ..., 1] = (w * dv[ok, 1]).sum(-1) / (ws + 1e-9); fld[i, ..., 2] = ws
                rp["field"] = rd(fld, 1)
            # metrics of the executed sample (index 0) and of all 32 samples, from the simulator-predicted joint paths
            J = fin[c]; T = fk(J.astype(float))                       # [S,16,3] TCP after actions 0..15
            chunk = it[c, -1, e, :, :, 7]                             # [S,16] raw gripper values
            end = T[:, 15]; cb = cube[s]; gl = goal[s]
            dcube = np.linalg.norm(end - cb, axis=-1); dgoal = np.linalg.norm(end - gl, axis=-1)
            kc = [int(np.argmax(chunk[i] < 0)) if (chunk[i] < 0).any() else -1 for i in range(len(chunk))]
            dgr = [float(np.linalg.norm(T[i, kc[i]] - cb)) if kc[i] >= 0 else None for i in range(len(chunk))]
            px_end = project(end)[0]; px_cube = project(cb)[0]; px_goal = project(gl)[0]
            m = {"model": name, "env": e, "call": c + 1, "step": s, "kind": k,
                 "tcp_to_cube_now_m": float(np.linalg.norm(tcp[s] - cb)), "end16_to_cube_m": float(dcube[0]), "end16_to_cube_px": float(np.linalg.norm(px_end[0] - px_cube)),
                 "end16_to_goal_m": float(dgoal[0]), "end16_to_goal_px": float(np.linalg.norm(px_end[0] - px_goal)), "goal_to_cube_now_m": float(np.linalg.norm(cb - gl)),
                 "close_index": kc[0], "close_to_cube_m": dgr[0], "samples_end16_std_mm": float(1000 * end.std(0).mean()),
                 "samples_end16_to_cube_m_median": float(np.median(dcube))}
            rp["metrics"] = m; table.append(m); a["replays"].append(rp)
        atts.append(a)
    out["models"][name] = {"attempts": atts}
out["metrics"] = table
(root / ("report_v2/data/overlay2_single.json" if SINGLE else "report_v2/data/overlay2.json")).write_text(json.dumps(out, separators=(",", ":")))
for m in table:
    print("%-9s env%d call %2d step %3d %-8s now %.3f m | end16->cube %.3f m (%.0f px) ->goal %.3f m (%.0f px) | close idx %s -> cube %s m | spread %.1f mm" % (
        m["model"], m["env"], m["call"], m["step"], m["kind"], m["tcp_to_cube_now_m"], m["end16_to_cube_m"], m["end16_to_cube_px"], m["end16_to_goal_m"], m["end16_to_goal_px"], m["close_index"],
        "-" if m["close_to_cube_m"] is None else "%.3f" % m["close_to_cube_m"], m["samples_end16_std_mm"]))
