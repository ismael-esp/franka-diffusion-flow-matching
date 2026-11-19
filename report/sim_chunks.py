#!/usr/bin/env python3
"""Predicted gripper paths from the SIMULATOR itself: restore the recorded robot (joint pos + vel) and cube state of a policy call in a batch of envs and
apply a chunk of normalised joint targets for 16 steps (the env's own PD controller and gravity). Used for the 32 final samples of every call that is shown, all
10 iterates of the replayed calls, and (flow) x + dt v of every Euler step. Validation: the executed chunk of every call of every env, re-simulated from the recorded state.
Writes runs/sim_paths.npz."""
import argparse, os, sys, traceback
from pathlib import Path
from isaaclab.app import AppLauncher
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.append(os.environ.get("DP_ROOT", str(Path(__file__).resolve().parents[1] / "pouring_diffusion")))
p = argparse.ArgumentParser(); p.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "runs/denoise_v4")); p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs/sim_paths.npz")); p.add_argument("--single", action="store_true", help="add the iterates of the ONE replay call (chunk end closest to the cube up to the grasp call) to a copy of runs/sim_paths.npz")
AppLauncher.add_app_launcher_args(p)

def main(args):
    import numpy as np, torch
    from fd_core.env import Franka, make_env
    from select_calls import ENVS, episode
    dev = args.device; B = 64
    env, raw = make_env(B, dev, 7, 64); franka = Franka(raw); env.reset()
    robot, obj, org = franka.robot, raw.scene["object"], raw.scene.env_origins
    ids = torch.arange(B, device=dev)
    def simulate(q9, qd9, cube, act):
        """q9,qd9 [N,9], cube [N,13] env frame, act [N,16,8] normalised -> joint pos after each of the 16 actions [N,16,7]"""
        N = len(q9); out = np.zeros((N, 16, 7), np.float32)
        for b0 in range(0, N, B):
            n = min(B, N - b0); sl = slice(b0, b0 + n)
            def pad(a): a = torch.as_tensor(np.asarray(a[sl]), dtype=torch.float32, device=dev); return torch.cat([a, a[-1:].expand(B - n, *a.shape[1:])]) if n < B else a
            raw.episode_length_buf[:] = 0
            q, qd, c, a = pad(q9), pad(qd9), pad(cube), pad(act)
            robot.write_joint_state_to_sim(q, qd, env_ids=ids); robot.set_joint_position_target(q, env_ids=ids)
            c = c.clone(); c[:, :3] += org; obj.write_root_state_to_sim(c, env_ids=ids)
            a = a.clone(); a[..., 7] = torch.where(a[..., 7] >= 0, 1.0, -1.0)
            for k in range(16):
                env.step(franka.norm_to_env(a[:, k])); out[b0:b0 + n, k] = robot.data.joint_pos[:n, :7].cpu().numpy()
        return out
    if args.single:
        from geom import fk
        res = dict(np.load(str(Path(__file__).resolve().parents[1] / "runs/sim_paths.npz"))); old = dict(res)
        for name in ("diffusion", "flow"):
            d = np.load(f"{args.data}/{name}_iterates.npz"); it = d["many_iter"].astype(np.float32); pr = d["many_pred"].astype(np.float32); qs = d["query_step"]
            q9, qd9, cs = d["qpos9"], d["qvel9"], d["cube_state"]; S = it.shape[3]
            for e in ENVS:
                ep = episode(d, e); cube = cs[e][:, :3].astype(float)
                cand = [(float(np.linalg.norm(fk(old[f"{name}_{e}_final"][c, 0].astype(float))[15] - cube[qs[c]])), c) for c in range(ep["last_call"] + 1) if 8 * c <= ep["t_close"]]
                c = min(cand)[1]; s0 = qs[c]; res[f"{name}_{e}_single"] = np.array(c)
                rep = lambda a: np.repeat(a[e, s0][None], 11 * S, 0)
                res[f"{name}_{e}_{c}_iter"] = simulate(rep(q9), rep(qd9), rep(cs), it[c, :, e].reshape(-1, 16, 8)).reshape(11, S, 16, 7)
                if name == "flow":
                    a1 = it[c, :-1, e] + 0.1 * pr[c, :, e]; a1[..., 7] = it[c, :-1, e][..., 7]
                    rep10 = lambda a: np.repeat(a[e, s0][None], 10 * S, 0)
                    res[f"{name}_{e}_{c}_a1"] = simulate(rep10(q9), rep10(qd9), rep10(cs), a1.reshape(-1, 16, 8)).reshape(10, S, 16, 7)
                print("single", name, e, "call", c + 1, flush=True)
        np.savez_compressed(args.out, **res); return
    res = {}
    for name in ("diffusion", "flow"):
        d = np.load(f"{args.data}/{name}_iterates.npz")
        it = d["many_iter"].astype(np.float32); pr = d["many_pred"].astype(np.float32)     # [Q,K+1,E,S,16,8], [Q,K,E,S,16,8]
        qs = d["query_step"]; q9, qd9, cs = d["qpos9"], d["qvel9"], d["cube_state"]
        # validation: executed chunk of every call, every env (index 0 of the samples)
        v = [(e, c) for e in range(8) for c in range(len(qs)) if qs[c] < 249]
        st = np.array([qs[c] for e, c in v]); E_ = np.array([e for e, c in v]); C_ = np.array([c for e, c in v])
        res[f"{name}_val"] = simulate(q9[E_, st], qd9[E_, st], cs[E_, st], it[C_, -1, E_, 0])
        res[f"{name}_val_idx"] = np.stack([E_, C_], 1)
        for e in ENVS:
            ep = episode(d, e); calls = list(range(ep["last_call"] + 1)); s = [qs[c] for c in calls]
            S = it.shape[3]
            fin = it[calls, -1, e]                                                         # [C,S,16,8]
            res[f"{name}_{e}_final"] = simulate(np.repeat(q9[e, s], S, 0), np.repeat(qd9[e, s], S, 0), np.repeat(cs[e, s], S, 0), fin.reshape(-1, 16, 8)).reshape(len(calls), S, 16, 7)
            for c in ep["replays"]:
                s0 = qs[c]; rep = lambda a: np.repeat(a[e, s0][None], 11 * S, 0)
                res[f"{name}_{e}_{c}_iter"] = simulate(rep(q9), rep(qd9), rep(cs), it[c, :, e].reshape(-1, 16, 8)).reshape(11, S, 16, 7)
                if name == "flow":
                    a1 = it[c, :-1, e] + 0.1 * pr[c, :, e]; a1[..., 7] = it[c, :-1, e][..., 7]
                    rep10 = lambda a: np.repeat(a[e, s0][None], 10 * S, 0)
                    res[f"{name}_{e}_{c}_a1"] = simulate(rep10(q9), rep10(qd9), rep10(cs), a1.reshape(-1, 16, 8)).reshape(10, S, 16, 7)
            print("done", name, e, flush=True)
    np.savez_compressed(args.out, **res)

if __name__ == "__main__":
    cli = p.parse_args(); cli.enable_cameras = True; cli.headless = True
    app = AppLauncher(cli).app; code = 0
    try: main(cli)
    except BaseException: traceback.print_exc(); sys.stdout.flush(); code = 1
    os._exit(code)
