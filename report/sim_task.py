#!/usr/bin/env python3
"""sim_chunks.py for the drawer and stack tasks: restore robot + task objects from the recorded state and apply chunks in the simulator.
For every (model, env): finals of all calls up to the cut, the single replay call (rule in task_select.py) with its 10 iterates (+ x + dt v for flow),
and a validation re-simulation of the executed chunk of every call."""
import argparse, os, sys, traceback
from pathlib import Path
from isaaclab.app import AppLauncher
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.append(os.environ.get("DP_ROOT", str(Path(__file__).resolve().parents[1] / "pouring_diffusion")))
p = argparse.ArgumentParser(); p.add_argument("--task", required=True); p.add_argument("--out", required=True); p.add_argument("--base", required=True, help="runs/task_<task>")
for m in ("diffusion", "flow"):
    p.add_argument(f"--envs-{m}", nargs="+", required=True, help="batch:env tokens, e.g. b0:10")
AppLauncher.add_app_launcher_args(p)

def main(a):
    import numpy as np, torch
    from fd_core.tasks import Arm, make_task_env
    from geom import fk
    from task_select import ep_info, marker_pos3
    dev = a.device; B = 64
    env, raw = make_task_env(a.task, B, dev, 7); arm = Arm(raw); env.reset()
    robot, org = arm.robot, raw.scene.env_origins; ids = torch.arange(B, device=dev)
    T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)
    def simulate(q9, qd9, objs, act):
        N = len(q9); out = np.zeros((N, 16, 7), np.float32)
        for b0 in range(0, N, B):
            n = min(B, N - b0); sl = slice(b0, b0 + n)
            def pad(x): x = T(x[sl]); return torch.cat([x, x[-1:].expand(B - n, *x.shape[1:])]) if n < B else x
            raw.episode_length_buf[:] = 0
            q, qd, ac = pad(q9), pad(qd9), pad(act)
            robot.write_joint_state_to_sim(q, qd, env_ids=ids); robot.set_joint_position_target(q, env_ids=ids)
            if a.task == "drawer":
                cab = raw.scene["cabinet"]; r = pad(objs["cab_root"]).clone(); r[:, :3] += org
                cab.write_root_state_to_sim(r, env_ids=ids); cab.write_joint_state_to_sim(pad(objs["cab_q"]), pad(objs["cab_qd"]), env_ids=ids)
            else:
                cs = pad(objs["cubes"])
                for i in range(3):
                    r = cs[:, i].clone(); r[:, :3] += org; raw.scene[f"cube_{i + 1}"].write_root_state_to_sim(r, env_ids=ids)
            ac = ac.clone(); ac[..., 7] = torch.where(ac[..., 7] >= 0, 1.0, -1.0)
            for k in range(16):
                env.step(arm.norm_to_env(ac[:, k])); out[b0:b0 + n, k] = robot.data.joint_pos[:n, :7].cpu().numpy()
        return out
    res = {}
    for name in ("diffusion", "flow"):
        cache = {}
        for tok in getattr(a, "envs_" + name):
            b_, e = tok.split(":"); e = int(e); d = np.load(f"{a.base}/{b_}/{name}_iterates.npz"); it = d["many_iter"].astype(np.float32); pr = d["many_pred"].astype(np.float32); qs = d["query_step"]; keep = list(d["keep_envs"]); S = it.shape[3]
            def rep(e, s, n, d=d):
                o = {k: d[k][e, s] for k in (("cab_root", "cab_q", "cab_qd") if a.task == "drawer" else ("cubes",))}; return d["qpos9"][e, s][None].repeat(n, 0), d["qvel9"][e, s][None].repeat(n, 0), {k: v[None].repeat(n, 0) for k, v in o.items()}
            ke = keep.index(e); info = ep_info(d, e, a.task); calls = [c for c in range(len(qs)) if qs[c] <= info["cut"]]
            q9 = np.concatenate([rep(e, qs[c], S)[0] for c in calls]); qd9 = np.concatenate([rep(e, qs[c], S)[1] for c in calls])
            ob = {k: np.concatenate([rep(e, qs[c], S)[2][k] for c in calls]) for k in rep(e, 0, 1)[2]}
            fin = simulate(q9, qd9, ob, it[calls, -1, ke].reshape(-1, 16, 8)).reshape(len(calls), S, 16, 7)
            res[f"{name}_{b_}e{e}_final"] = fin
            # validation: executed chunk (sample 0) of every call, from the recorded state
            res[f"{name}_{b_}e{e}_val"] = fin[:, 0]
            mk = marker_pos3(d, e, a.task)
            tc = info["t_close"] if info["t_close"] is not None else qs[calls[-1]]
            cand = []
            for c in calls:
                if qs[c] > tc: break
                cand.append((float(np.linalg.norm(fk(fin[c, 0].astype(float))[15] - mk[qs[c]])), c))
            c = min(cand)[1]; s0 = qs[c]; res[f"{name}_{b_}e{e}_single"] = np.array(c)
            qq, qqd, oo = rep(e, s0, 11 * S)
            res[f"{name}_{b_}e{e}_{c}_iter"] = simulate(qq, qqd, oo, it[c, :, ke].reshape(-1, 16, 8)).reshape(11, S, 16, 7)
            if name == "flow":
                a1 = it[c, :-1, ke] + 0.1 * pr[c, :, ke]; a1[..., 7] = it[c, :-1, ke][..., 7]
                qq, qqd, oo = rep(e, s0, 10 * S)
                res[f"{name}_{b_}e{e}_{c}_a1"] = simulate(qq, qqd, oo, a1.reshape(-1, 16, 8)).reshape(10, S, 16, 7)
            print("done", name, e, "call", c + 1, info, flush=True)
    np.savez_compressed(a.out, **res)

if __name__ == "__main__":
    c = p.parse_args(); c.enable_cameras = True; c.headless = True; app = AppLauncher(c).app; code = 0
    try: main(c)
    except BaseException: traceback.print_exc(); sys.stdout.flush(); code = 1
    sys.stdout.flush(); os._exit(code)
