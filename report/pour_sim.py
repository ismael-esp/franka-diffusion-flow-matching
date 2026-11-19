#!/usr/bin/env python3
"""Restore the recorded pour state (joints, cup, bottle, real-match lag filters) in a no-camera replica env and apply sampled chunks for 16 steps.
For every (model, env): finals of the 32 samples of all calls up to the cut, a validation of sample 0 of every call against the recorded run, the replay call (rule in
pour_select.py) with the stored iterates. Contact guard is 'monitor' here (the recording env terminates an episode on a guard violation; a replay must not reset).
python pour_sim.py --out runs/sim_pour.npz --item diffusion:runs/pour/diffusion:3 --item flow:runs/pour/flow:3"""
import sys
sys.dont_write_bytecode = True   # TRLDistill_v5 is checksum-sealed: never write .pyc into it
import argparse, os, sys, traceback
from pathlib import Path
from isaaclab.app import AppLauncher
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pour_common as pc
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--item", action="append", required=True, help="name:rec_dir:env[:call]")
p.add_argument("--seed", type=int, default=201); p.add_argument("--batch", type=int, default=64); p.add_argument("--cameras", action="store_true")
AppLauncher.add_app_launcher_args(p)

def main(a):
    import numpy as np, torch
    import pour_select as ps
    ROOT = Path(__file__).resolve().parents[1]; a.out = str((ROOT / a.out).resolve()); dev = torch.device(a.device); B = a.batch; ids = torch.arange(B, device=dev)
    env, raw = pc.build(B, dev, a.seed, Path(a.out).resolve().parent / "pour_sim_env", stage="eval", record=False, cameras=a.cameras, guard={"mode": "monitor", "table_cmd": True})
    env.reset(); n_step = [0]
    def simulate(states, acts):
        """states: dict of arrays [N,...] (one state per row), acts [N,16,14] normalized -> dict of post-step arrays [N,16,...]"""
        N = len(acts); R = {k: np.zeros((N,) + s, np.float32) for k, s in (("rh", (16, 3)), ("lh", (16, 3)), ("bottle", (16, 7)), ("cup", (16, 7)))}
        for b0 in range(0, N, B):
            n = min(B, N - b0); sl = np.arange(b0, b0 + n); pad = np.concatenate([sl, np.full(B - n, sl[-1])])
            env.reset(); n_step[0] = 0
            st = {k: torch.as_tensor(v[pad], device=dev) for k, v in states.items()}; pc.set_state(raw, st, ids)
            ac = torch.as_tensor(acts[pad], dtype=torch.float32, device=dev).clamp(-1, 1); ac[..., [6, 13]] = torch.where(ac[..., [6, 13]] >= 0, 1.0, -1.0)
            org = raw.scene.env_origins
            for k in range(16):
                env.step(torch.stack([ac[:, k, :7], ac[:, k, 7:]], 0), n_step[0]); n_step[0] += 1
                R["rh"][b0:b0 + n, k] = (raw.right_robot.data.body_pos_w[:n, 6] - org[:n]).cpu().numpy(); R["lh"][b0:b0 + n, k] = (raw.left_robot.data.body_pos_w[:n, 6] - org[:n]).cpu().numpy()
                for nm, o in (("bottle", raw.bottle), ("cup", raw.cup)):
                    x = o.data.root_state_w[:n, :7].clone(); x[:, :3] -= org[:n]; R[nm][b0:b0 + n, k] = x.cpu().numpy()
        return R
    res = {}
    for tok in a.item:
        parts = tok.split(":"); name, rec, e = parts[0], ROOT / parts[1], int(parts[2]); forced = int(parts[3]) if len(parts) > 3 else None
        D = np.load(rec / "iterates.npz"); E = np.load(rec / "episode.npz"); it = np.moveaxis(D["iter"].astype(np.float32), 1, 0); cs = D["call_step"]; S = it.shape[3]
        info = ps.ep_info(E, e); calls = [c for c in range(len(cs)) if cs[c] <= info["cut"]]
        def rep(c, n): return {k[6:]: np.repeat(D[k][c, e][None], n, 0) for k in D.files if k.startswith("state_")}
        def cat(lst): return {k: np.concatenate([x[k] for x in lst]) for k in lst[0]}
        st = cat([rep(c, S) for c in calls]); ac = np.concatenate([it[-1, c, e] for c in calls])
        fin = simulate(st, ac); fin = {k: v.reshape(len(calls), S, 16, -1) for k, v in fin.items()}
        for k, v in fin.items(): res[f"{name}_e{e}_final_{k}"] = v
        # validation of sample 0 vs the recorded run (state after k actions = logged state at step t + k)
        err = []
        for i, c in enumerate(calls):
            t = int(cs[c]); n = min(16, info["cut"] - t)
            if n > 0: err.append(np.linalg.norm(fin["rh"][i, 0, :n] - E["rh"][e, t + 1:t + 1 + n], axis=-1))
        res[f"{name}_e{e}_val_rh_m"] = np.array([x.mean() for x in err]); res[f"{name}_e{e}_val_rh_max_m"] = np.array([x.max() for x in err])
        # call choice: the call that contains the start of the pour tilt (rule in pour_select.py)
        c0 = forced if forced is not None else ps.tilt_call(E, e); res[f"{name}_e{e}_single"] = np.array(c0)
        sel_n = it.shape[0]
        sim = simulate(rep(c0, sel_n * S), it[:, c0, e].reshape(-1, 16, 14))
        for k, v in sim.items(): res[f"{name}_e{e}_{c0}_iter_{k}"] = v.reshape(sel_n, S, 16, -1)
        print("done", name, e, "call", c0, info, "val rh mean mm", 1000 * float(np.mean([x.mean() for x in err])), flush=True)
    np.savez_compressed(a.out, **res)

if __name__ == "__main__":
    c = p.parse_args(); c.headless = True; c.enable_cameras = bool(c.cameras); app = AppLauncher(c).app; code = 0
    try: main(c)
    except BaseException: traceback.print_exc(); sys.stdout.flush(); code = 1
    sys.stdout.flush(); os._exit(code)
