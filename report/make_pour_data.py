#!/usr/bin/env python3
"""Pour video data -> data/task_pour.json (+ prints the verification numbers).
python make_pour_data.py --sim runs/sim_pour.npz --item diffusion:runs/pour/dp_t1:0 --item flow:runs/pour/fm_t1:0   (name:rec_dir:env, attempts in play order)
Drawn path = the bottle mouth ("pour point", bottle root + bottle_pour_offset rotated by the bottle quaternion, the point the env's pour stage compares with the cup opening),
carried by the right hand; the simulator gives it for every sampled chunk. Markers: cup opening (square), the pour region of the env (half cylinder r 3.75 cm, z -2..+10 cm
around the cup opening on the bottle side) and the right-hand pour pose P_R_POUR (cross)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pour_common as pc
from pour_select import ep_info, HZ
root = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--item", action="append", required=True); ap.add_argument("--out", default="report_v2/data/task_pour.json")
ap.add_argument("--rate-pre", type=int, default=3); ap.add_argument("--rate-post", type=int, default=2)
args = ap.parse_args()
R = np.load(root / args.sim); rd = lambda a, n=1: np.round(a, n).tolist()
P = lambda x: pc.project(np.asarray(x, float))
RAD, ZMIN, ZMAX = 0.0375, -0.02, 0.10          # PouringEnv.pour_radius / pour_z_min / pour_z_max
out = {"task": "pour", "hz": HZ, "vw": pc.REC_W, "vh": pc.REC_H, "models": {}}
table = []
for tok in args.item:
    name, rec, e = tok.split(":"); e = int(e); rec = root / rec
    D = np.load(rec / "iterates.npz"); E = np.load(rec / "episode.npz"); meta = json.load(open(rec / "meta.json")); cs = D["call_step"]; sel = D["sel"].tolist(); K = meta["ddim_steps"]
    info = ep_info(E, e); cut = info["cut"] - (info["cut"] % 2); cid = f"e{e}"
    Pg = E["P_R_POUR"]; rh = E["rh"][e].astype(float); bo = E["bottle_pour_offset"]; co = E["cup_opening_offset"]
    mouth = E["bottle"][e, :, :3] + pc.quat_apply_np(E["bottle"][e, :, 3:7], bo); opening = E["cup"][e, :, :3] + pc.quat_apply_np(E["cup"][e, :, 3:7], co)
    stage = E["subgoal"][e].min(1).astype(int)
    mth = lambda b: b[..., :3] + pc.quat_apply_np(b[..., 3:7], bo)          # bottle root state [...,7] -> mouth
    fin_m = mth(R[f"{name}_{cid}_final_bottle"].astype(float)); fin_rh = R[f"{name}_{cid}_final_rh"].astype(float)
    calls = [c for c in range(len(cs)) if cs[c] <= info["cut"]]
    c0 = int(R[f"{name}_{cid}_single"]); s0 = int(cs[c0]); ic = calls.index(c0)
    path = lambda start, arr: np.concatenate([np.broadcast_to(start, arr.shape[:-2] + (1, 3)), arr], -2)
    finals_m = path(mouth[cs[calls]][:, None, None, :], fin_m); finals = P(finals_m)
    it_m = path(mouth[s0], mth(R[f"{name}_{cid}_{c0}_iter_bottle"].astype(float)))
    # the last sampler state is the same set of chunks as finals[call]; simulated twice in different batch slots, the in-hand bottle can end up differently (contact chaos, see notes): use the one
    # simulation for both so that the replay ends exactly on the lines that are drawn afterwards
    nslip = int((np.linalg.norm(it_m[-1, :, 16] - finals_m[ic, :, 16], axis=-1) > 0.05).sum()); itm_last_diff_mm = float(np.median(1000 * np.linalg.norm(it_m[-1, :, 16] - finals_m[ic, :, 16], axis=-1)))
    it_m[-1] = finals_m[ic]
    # sampler state labels: diffusion = DDIM timestep t of the state (98 = noise for 50 steps over 100), flow = flow time t
    if name == "flow": labels = [i / K for i in sel]
    else: ts = [int(x) for x in (np.arange(K - 1, -1, -1) * (100 // K))]; labels = [(ts[i] if i < K else 0) for i in sel]
    # validation: simulator re-simulation of the EXECUTED chunk (sample 0) of every call vs the recorded run, mouth and right hand
    ep, em, ehm = [], [], []
    for i, c in enumerate(calls):
        t = int(cs[c]); n = min(16, info["cut"] - t)
        if n > 0:
            ep.append(np.linalg.norm(P(fin_m[i, 0, :n]) - P(mouth[t + 1:t + 1 + n]), axis=-1).mean()); em.append(1000 * np.linalg.norm(fin_m[i, 0, :n] - mouth[t + 1:t + 1 + n], axis=-1).mean())
            ehm.append(1000 * np.linalg.norm(fin_rh[i, 0, :n] - rh[t + 1:t + 1 + n], axis=-1).mean())
    end = fin_m[ic, :, 15]; it_end = it_m[:, :, 16]
    # pour region polygons (cup frame, bottle side), per step
    cq = E["cup"][e, :cut + 1, 3:7]; cp = opening[:cut + 1]
    qinv = np.concatenate([cq[:, :1], -cq[:, 1:]], -1); v = pc.quat_apply_np(qinv, E["bottle"][e, :cut + 1, :3] - cp); psi = np.arctan2(v[:, 1], v[:, 0])
    phi = np.linspace(-np.pi / 2, np.pi / 2, 13)
    loc = lambda z: np.stack([RAD * np.cos(phi[None] + psi[:, None]), RAD * np.sin(phi[None] + psi[:, None]), np.full((cut + 1, len(phi)), z)], -1)
    poly = np.concatenate([loc(ZMIN), loc(ZMAX)[:, ::-1]], 1)
    poly_w = cp[:, None] + pc.quat_apply_np(cq[:, None], poly)
    t5, t4 = info["t_pour_done"], info["t_stage4"]
    # check of the region/marker geometry against the env's own stage-4 test at the step the pour stage completes (mouth inside the half cylinder, hand at the pour pose)
    def inside(t):
        d = pc.quat_apply_np(np.concatenate([E["cup"][e, t, 3:4], -E["cup"][e, t, 4:7]]), mouth[t] - opening[t]); vv = pc.quat_apply_np(np.concatenate([E["cup"][e, t, 3:4], -E["cup"][e, t, 4:7]]), E["bottle"][e, t, :3] - opening[t])
        fd = vv[:2] / max(np.linalg.norm(vv[:2]), 1e-6)
        return bool(np.linalg.norm(d[:2]) <= RAD and ZMIN <= d[2] <= ZMAX and d[:2] @ fd >= 0)
    chk = {"model": name, "env": e, "t_stage4": t4, "t_pour_done": t5, "rh_to_pour_pose_at_pour_done_m": None if t5 is None else float(np.linalg.norm(rh[t5] - Pg)),
           "rh_to_pour_pose_at_pour_done_px": None if t5 is None else float(np.linalg.norm(P(rh[t5]) - P(Pg))), "mouth_in_region_at_pour_done": None if t5 is None else inside(t5),
           "mouth_to_opening_at_pour_done_m": None if t5 is None else float(np.linalg.norm(mouth[t5] - opening[t5]))}
    m = {"model": name, "env": e, "call": c0 + 1, "step": s0, "t_tilt_call_step": s0, "mouth_now_to_opening_m": float(np.linalg.norm(mouth[s0] - opening[s0])), "hand_now_to_pose_m": float(np.linalg.norm(rh[s0] - Pg)),
         "end16_to_opening_m": float(np.linalg.norm(end[0] - opening[min(s0 + 16, len(opening) - 1)])), "end16_to_opening_px": float(np.linalg.norm(P(end[0]) - P(opening[min(s0 + 16, len(opening) - 1)]))),
         "end16_move_m": float(np.linalg.norm(end[0] - mouth[s0])),
         "spread_mm": float(1000 * end.std(0).mean()), "spread_noise_mm": float(1000 * it_end[0].std(0).mean()), "spread_iter_mm": [round(float(1000 * x.std(0).mean()), 1) for x in it_end],
         "val_px_mean": float(np.mean(ep)), "val_px_max": float(np.max(ep)), "val_mm_mean": float(np.mean(em)), "val_hand_mm_mean": float(np.mean(ehm)), "t_succ": info["t_succ"], "check": chk, "resim_same_actions_median_mm": itm_last_diff_mm, "resim_same_actions_n_gt5cm": nslip}
    table.append(m)
    mk = [{"label": "cup opening", "kind": "square", "t0": 0, "t1": cut, "px": rd(P(opening[:cut + 1]))}]
    att = {"env": e, "clip": cid, "success": info["success"], "t_succ": info["t_succ"], "t_stage4": t4, "t_pour_done": t5, "cut": cut, "last_call": int(len(calls) - 1), "rate_pre": args.rate_pre, "rate_post": args.rate_post,
           "actual_px": rd(P(mouth[:cut + 1])), "region_px": rd(P(poly_w), 1), "pose_px": rd(P(Pg)), "stage": stage[:cut + 1].tolist(), "call_steps": [int(x) for x in cs[calls]],
           "finals": rd(finals), "markers": mk, "replay": {"call": c0, "step": s0, "iter": rd(P(it_m)), "labels": labels, "metrics": m}}
    out["models"].setdefault(name, {"ddim_steps": K, "checkpoint": "pouring_diffusion/runs/" + str(meta["checkpoint"]).split("/runs/", 1)[-1], "attempts": []})["attempts"].append(att)
out["metrics"] = table
(root / args.out).write_text(json.dumps(out, separators=(",", ":")))
for m in table:
    print("%-9s env%-2d call %2d step %3d | mouth->opening now %.3f m, end16 (sample 0) %.3f m (%.0f px), moves %.3f m | hand->pose now %.3f m | spread end %.1f mm (noise %.1f) iter %s | sim-vs-actual mouth %.2f px mean %.2f max (%.2f mm; hand %.2f mm) | t_succ %s\n    check: %s" % (
        m["model"], m["env"], m["call"], m["step"], m["mouth_now_to_opening_m"], m["end16_to_opening_m"], m["end16_to_opening_px"], m["end16_move_m"], m["hand_now_to_pose_m"], m["spread_mm"], m["spread_noise_mm"], m["spread_iter_mm"],
        m["val_px_mean"], m["val_px_max"], m["val_mm_mean"], m["val_hand_mm_mean"], m["t_succ"], m["check"]))
