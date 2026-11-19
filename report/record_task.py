#!/usr/bin/env python3
"""record_denoise.py for the drawer and stack tasks (fd_core.tasks): both models drive the same batch (same seed => same reset), 32 noise seeds per policy call,
all iterates stored, simulator state logged every step (robot, task objects) for the sim-restore step (sim_task.py), videos from the task's video camera.
python record_task.py --task drawer --dp runs/drawer_s/dp/bc_best.pt --fm runs/flow/fm_drawer/bc_best.pt --out-dir runs/task_drawer"""
from __future__ import annotations
import argparse, json, os, sys, traceback
from pathlib import Path
from isaaclab.app import AppLauncher
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.append(os.environ.get("DP_ROOT", str(Path(__file__).resolve().parents[1] / "pouring_diffusion")))

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=["drawer", "stack"]); p.add_argument("--dp", type=Path, required=True); p.add_argument("--fm", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True); p.add_argument("--seed", type=int, default=41); p.add_argument("--num-envs", type=int, default=12)
    p.add_argument("--batches", type=int, default=1, help="batches with seeds seed, seed+1, ...; each is saved in out-dir/b<k>")
    p.add_argument("--samples", type=int, default=32); p.add_argument("--keep-envs", type=int, nargs="+", default=None)
    p.add_argument("--video-size", type=int, nargs=2, default=[960, 720]); p.add_argument("--steps", type=int, default=10)
    AppLauncher.add_app_launcher_args(p); return p.parse_args()

def main(args):
    import imageio.v2 as imageio, numpy as np, torch
    from fd_core.policy import load_dp
    from fd_core.tasks import TASKS, Arm, drawer_pos, make_task_env, run_episodes, stack_success
    from dp_core.model import ddim_timesteps
    from record_denoise import TracingActor
    dev = args.device; spec = TASKS[args.task]
    env, raw = make_task_env(args.task, args.num_envs, dev, args.seed, cameras=True, video_size=tuple(args.video_size)); arm = Arm(raw)
    org = raw.scene.env_origins; robot = arm.robot; hand_id = robot.find_bodies("panda_hand")[0][0]
    keep = args.keep_envs if args.keep_envs is not None else list(range(args.num_envs))
    class Logged:                       # logs the simulator state before every action, then delegates
        def __init__(self, inner): self.inner = inner; self.log = {}
        def reset(self, n): self.inner.reset(n); self.log = {}
        def put(self, k, v): self.log.setdefault(k, []).append(v.cpu())
        def __call__(self, t, obs, ih, ph):
            d = robot.data; self.put("qpos9", d.joint_pos[:, :9]); self.put("qvel9", d.joint_vel[:, :9]); self.put("hand", d.body_pos_w[:, hand_id] - org)
            if args.task == "drawer":
                cab = raw.scene["cabinet"]; st = cab.data.root_state_w[:, :13].clone(); st[:, :3] -= org
                self.put("cab_root", st); self.put("cab_q", cab.data.joint_pos); self.put("cab_qd", cab.data.joint_vel)
                self.put("handle", raw.scene["cabinet_frame"].data.target_pos_w[:, 0] - org); self.put("drawer_pos", drawer_pos(raw)[:, None])
                self.put("succ", (drawer_pos(raw) >= 0.30)[:, None].float())
            else:
                cs = []
                for i in (1, 2, 3):
                    s = raw.scene[f"cube_{i}"].data.root_state_w[:, :13].clone(); s[:, :3] -= org; cs.append(s)
                self.put("cubes", torch.stack(cs, 1)); ok12, ok23 = stack_success(raw)
                rel = robot.data.joint_pos[:, 7:9].mean(-1) > 0.03
                self.put("ok12", ok12[:, None].float()); self.put("succ", (ok23 & rel)[:, None].float())
            return self.inner(t, obs, ih, ph)
    for b in range(args.batches):
        od = args.out_dir / f"b{b}"; od.mkdir(parents=True, exist_ok=True); seed = args.seed + b; meta = {"task": args.task, "seed": seed, "num_envs": args.num_envs, "models": {}}
        res = {}
        for name, path, objective in (("diffusion", args.dp, "diffusion"), ("flow", args.fm, "flow")):
            model, cfg = load_dp(path, dev); model.ddim_steps = ddim_timesteps(cfg.num_train_timesteps, args.steps).to(dev)
            gen = torch.Generator(device=dev).manual_seed(seed + 1000)
            tr = TracingActor(model, cfg.exec_horizon, objective, args.samples, keep, gen); actor = Logged(tr)
            torch.manual_seed(seed)
            r = run_episodes(env, raw, arm, actor, args.task, obs_horizon=cfg.obs_horizon, video_cam="video_camera")
            succ = r["success"].tolist(); print(f"[run] batch {b} {name}: success {sum(succ)}/{len(succ)} {succ}", flush=True)
            out = {"many_iter": torch.stack([m[0] for m in tr.many]).numpy(), "many_pred": torch.stack([m[1] for m in tr.many]).numpy(), "query_step": np.array([q["t"] for q in tr.queries]),
                   "executed": torch.stack(tr.executed, 1).numpy(), "keep_envs": np.array(keep), "joint_low": arm.low.cpu().numpy(), "joint_high": arm.high.cpu().numpy(),
                   "success": np.array(succ), "outcome": r["outcome"].numpy(), "progress": r["progress"].numpy()}
            for k, v in actor.log.items(): out[k] = torch.stack(v, 1).numpy()
            np.savez_compressed(od / f"{name}_iterates.npz", **out)
            res[name] = succ
            for e in range(args.num_envs):
                if succ[e] or e < 2:
                    w = imageio.get_writer(od / f"{name}_e{e}_{'ok' if succ[e] else 'fail'}.mp4", fps=30, codec="libx264", quality=9)
                    for f in r["frames"][e].numpy(): w.append_data(f)
                    w.close()
            meta["models"][name] = {"checkpoint": str(path), "objective": objective, "success": succ, "outcome": r["outcome"].tolist(), "exec_horizon": cfg.exec_horizon, "steps": args.steps}
            del r
        both = [e for e in range(args.num_envs) if res["diffusion"][e] and res["flow"][e]]
        meta["both_success_envs"] = both; print(f"[batch {b}] both succeed in envs {both}", flush=True)
        (od / "meta.json").write_text(json.dumps(meta, indent=1))

if __name__ == "__main__":
    cli = parse_args(); cli.enable_cameras = True; cli.headless = True; app = AppLauncher(cli).app; code = 0
    try: main(cli)
    except BaseException: traceback.print_exc(); sys.stdout.flush(); code = 1
    os._exit(code)
