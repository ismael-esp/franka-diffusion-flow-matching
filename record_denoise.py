#!/usr/bin/env python3
"""Record the real intermediate iterates of the action chunk of the diffusion (DDIM) and the flow-matching (Euler)
policy while they drive the SAME lift episodes (same seed, same initial states, one rollout per model).

For every policy call (every 8 env steps) it stores all sampler iterates [K+1, 16, 8] (the noise sample, then the
state after each of the K steps; the last one is the clamped chunk that is executed) plus, per step, the model's own
prediction (diffusion: clipped x0 estimate; flow: velocity), the executed 8 actions and the env step index.
Videos: one 640x480 third-person mp4 per env and model (video frame i = env step i).

  python record_denoise.py --out-dir runs/denoise --seed 41 --num-envs 8
The sampler loops below are copies of DiffusionPolicy.sample / FlowPolicy.sample that also keep the iterates; the script
asserts that, with the same random noise, they reproduce model.sample() exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(os.environ.get("DP_ROOT", str(Path(__file__).resolve().parent / "pouring_diffusion")))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dp", type=Path, default=Path("runs/flow/dp_lift/bc_best.pt"))
    p.add_argument("--fm", type=Path, default=Path("runs/flow/fm_lift/bc_best.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/denoise"))
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--samples", type=int, default=32, help="Chunks sampled per policy call and env (index 0 = the one that is executed).")
    p.add_argument("--video-size", type=int, nargs=2, default=[1280, 960])
    p.add_argument("--keep-envs", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7], help="Envs whose many-sample iterates are stored.")
    p.add_argument("--steps", type=int, default=10, help="DDIM steps / Euler steps (both 10 in the reported lift numbers).")
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def trace_diffusion(model, cond, noise):
    """DDIM (eta=0), identical to DiffusionPolicy.sample, keeping x_i and the clipped x0 estimate of every step."""
    import torch
    b = noise.shape[0]
    x, xs, preds, ts = noise, [noise], [], []
    for i, t in enumerate(model.ddim_steps):
        prev = model.ddim_steps[i + 1] if i + 1 < len(model.ddim_steps) else torch.tensor(-1, device=noise.device)
        tb, pb = t.expand(b), prev.expand(b)
        eps = model.denoiser_for(i)(x, tb, cond)
        alpha = model.alpha_at(tb)
        x0 = ((x - (1.0 - alpha).sqrt() * eps) / alpha.sqrt()).clamp(-1.0, 1.0)
        x, _ = model.ddim_mean(x, eps, tb, pb, eta=0.0)
        xs.append(x); preds.append(x0); ts.append(int(t))
    xs[-1] = x.clamp(-1.0, 1.0)
    return xs, preds, ts


def trace_flow(model, cond, noise):
    """Euler integration, identical to FlowPolicy.sample, keeping x_i and the velocity of every step."""
    import torch
    b = noise.shape[0]
    k = len(model.ddim_steps)
    x, xs, preds, ts = noise, [noise], [], []
    for i in range(k):
        t = torch.full((b,), i / k, device=noise.device)
        v = model.denoiser_for(i)(x, t * model.cfg.num_train_timesteps, cond)
        x = x + v / k
        xs.append(x); preds.append(v); ts.append(i / k)
    xs[-1] = x.clamp(-1.0, 1.0)
    return xs, preds, ts


class TracingActor:
    """Same behaviour as fd_core.rollout.DiffusionActor (replan every exec steps, snapped gripper) + logging."""

    def __init__(self, model, exec_horizon, objective, samples=1, keep=(), gen=None):
        self.model, self.exec, self.objective = model, exec_horizon, objective
        self.S, self.keep, self.gen = samples, list(keep), gen
        self.plan = None
        self.queries = []       # dicts: env step, iterates [K+1,N,16,8], preds [K,N,16,8], times, snapped plan
        self.executed = []      # per step [N,8] the action actually sent (normalized)
        self.many = []

    def reset(self, n):
        self.plan = None

    def __call__(self, t, obs, image_hist, proprio_hist):
        import torch
        k = t % self.exec
        if k == 0:
            m = self.model
            image, proprio = image_hist.float() / 255.0, proprio_hist
            cond = m.encode(image, proprio)
            noise = torch.randn(image.shape[0], m.cfg.pred_horizon, m.cfg.action_dim, device=image.device)
            tr = trace_flow if self.objective == "flow" else trace_diffusion
            xs, preds, ts = tr(m, cond, noise)
            if self.S > 1:                         # extra noise seeds for the same observation (separate RNG: the rollout is unchanged)
                n = image.shape[0]
                noise2 = torch.randn(n * (self.S - 1), m.cfg.pred_horizon, m.cfg.action_dim, device=image.device, generator=self.gen)
                xs2, pr2, _ = tr(m, cond.repeat_interleave(self.S - 1, 0), noise2)
                sh = lambda a: a.reshape(n, self.S - 1, *a.shape[1:])[self.keep]
                it = torch.stack([torch.cat([a[self.keep][:, None], sh(b)], 1) for a, b in zip(xs, xs2)])      # [K+1,E,S,16,8]
                pd = torch.stack([torch.cat([a[self.keep][:, None], sh(b)], 1) for a, b in zip(preds, pr2)])
                self.many.append((it.half().cpu(), pd.half().cpu()))
            plan = xs[-1].clone()
            raw_plan = plan.clone()
            plan[..., 7] = torch.where(plan[..., 7] >= 0, 1.0, -1.0)      # gripper snap, as in DiffusionActor
            self.plan = plan
            self.queries.append({"t": t, "xs": torch.stack(xs).cpu(), "preds": torch.stack(preds).cpu(), "times": ts,
                                 "chunk": raw_plan.cpu()})
        a = self.plan[:, k]
        self.executed.append(a.cpu())
        return a


def main(args):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from fd_core.env import Franka, make_env
    from fd_core.policy import camera_setup, load_dp
    from fd_core.rollout import run_batch
    from dp_core.model import ddim_timesteps

    dev = args.device
    wrist, zoom = camera_setup(args.dp)
    assert (wrist, zoom) == camera_setup(args.fm), "both models must see the same cameras"
    env, raw = make_env(args.num_envs, dev, args.seed, 128, video_size=tuple(args.video_size), wrist=wrist, zoom=zoom)
    franka = Franka(raw)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"task": "lift", "seed": args.seed, "num_envs": args.num_envs, "steps": args.steps, "models": {}}
    first_frames = {}
    for name, path, objective in (("diffusion", args.dp, "diffusion"), ("flow", args.fm, "flow")):
        model, cfg = load_dp(path, dev)
        model.ddim_steps = ddim_timesteps(cfg.num_train_timesteps, args.steps).to(dev)
        assert getattr(model, "objective", "diffusion") == objective, (name, type(model))
        # exactness check: traced sampler == model.sample() for the same noise
        img = torch.randint(0, 255, (2, cfg.obs_horizon, 3, 128, 256 if wrist else 128), device=dev, dtype=torch.uint8)
        prop = torch.randn(2, cfg.obs_horizon, 1, cfg.proprio_dim, device=dev)
        with torch.no_grad():
            torch.manual_seed(5); ref = model.sample(img.float() / 255.0, prop[:, :, 0] if prop.dim() == 5 else prop)
            torch.manual_seed(5)
            c = model.encode(img.float() / 255.0, prop[:, :, 0] if prop.dim() == 5 else prop)
            xs, _, _ = (trace_flow if objective == "flow" else trace_diffusion)(
                model, c, torch.randn(2, cfg.pred_horizon, cfg.action_dim, device=dev))
        err = float((xs[-1] - ref).abs().max())
        print(f"[trace] {name}: max |traced - model.sample| = {err:.2e}", flush=True)
        assert err < 1e-5

        gen = torch.Generator(device=dev).manual_seed(args.seed + 1000)
        actor = TracingActor(model, cfg.exec_horizon, objective, args.samples, args.keep_envs, gen)
        hand_id = franka.robot.find_bodies("panda_hand")[0][0]
        hand, qpos, qvel9, qpos9, cube, goal, gdist = [], [], [], [], [], [], []
        def on_step(t, ih, ph):                                        # simulator truth for verifying the FK / camera projection, and the task state
            hand.append((franka.robot.data.body_pos_w[:, hand_id] - raw.scene.env_origins).cpu())
            qpos.append(franka.robot.data.joint_pos[:, :7].cpu())
            qpos9.append(franka.robot.data.joint_pos[:, :9].cpu()); qvel9.append(franka.robot.data.joint_vel[:, :9].cpu())
            st = raw.scene["object"].data.root_state_w[:, :13].clone(); st[:, :3] -= raw.scene.env_origins
            cube.append(st.cpu()); goal.append(raw.command_manager.get_command("object_pose")[:, :3].cpu()); gdist.append(franka.goal_distance().cpu())
        torch.manual_seed(args.seed)                                   # same reset randomness for both models
        r = run_batch(env, raw, franka, actor, obs_horizon=cfg.obs_horizon, video_cam="video_camera", on_step=on_step)
        first_frames[name] = r["frames"][:, 0].float()
        succ = r["success"].tolist()
        print(f"[run] {name}: success {sum(succ)}/{len(succ)} {succ}", flush=True)
        xs = torch.stack([q["xs"] for q in actor.queries])             # [Q,K+1,N,16,8]
        pr = torch.stack([q["preds"] for q in actor.queries])          # [Q,K,N,16,8]
        np.savez_compressed(args.out_dir / f"{name}_iterates.npz", iterates=xs.numpy().astype(np.float32),
                            preds=pr.numpy().astype(np.float32), query_step=np.array([q["t"] for q in actor.queries]),
                            times=np.array(actor.queries[0]["times"], dtype=np.float32),
                            executed=torch.stack(actor.executed, 1).numpy().astype(np.float32),   # [N,250,8]
                            many_iter=torch.stack([m_[0] for m_ in actor.many]).numpy(), many_pred=torch.stack([m_[1] for m_ in actor.many]).numpy(),
                            keep_envs=np.array(args.keep_envs), qpos9=torch.stack(qpos9, 1).numpy(), qvel9=torch.stack(qvel9, 1).numpy(),
                            cube_state=torch.stack(cube, 1).numpy(), goal=torch.stack(goal, 1).numpy(), goal_dist=torch.stack(gdist, 1).numpy(), hand_pos=torch.stack(hand, 1).numpy(), qpos=torch.stack(qpos, 1).numpy(),
                            joint_low=franka.low.cpu().numpy(), joint_high=franka.high.cpu().numpy(),
                            success=np.array(succ), dist=r["dist"].numpy(), outcome=r["outcome"].numpy())
        for i in range(args.num_envs):
            w = imageio.get_writer(args.out_dir / f"{name}_e{i}_{'ok' if succ[i] else 'fail'}.mp4", fps=30, codec="libx264", quality=9)
            for f in r["frames"][i].numpy():
                w.append_data(f)
            w.close()
        meta["models"][name] = {"checkpoint": str(path), "objective": objective, "success": succ, "outcome": r["outcome"].tolist(),
                                "final_dist": r["dist"].tolist(), "exec_horizon": cfg.exec_horizon, "pred_horizon": cfg.pred_horizon,
                                "num_train_timesteps": cfg.num_train_timesteps}
        del model
    d = float((first_frames["diffusion"] - first_frames["flow"]).abs().max())
    meta["first_frame_max_abs_diff"] = d
    print(f"[check] first video frame diffusion vs flow, max abs pixel diff = {d}", flush=True)
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    cli = parse_args()
    cli.enable_cameras = True
    cli.headless = True
    app = AppLauncher(cli).app
    code = 0
    try:
        main(cli)
    except BaseException:
        traceback.print_exc(); sys.stdout.flush(); code = 1
    os._exit(code)
