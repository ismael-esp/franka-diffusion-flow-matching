#!/usr/bin/env python3
"""Pouring: roll out one model (diffusion or flow matching) in the TRLDistill_v5 pour env (stage eval, snapped grip, exec horizon 16, same rules as pouring_diffusion/eval_policy.py)
and store for every policy call: the full restorable simulator state, 32 sampled chunks with the sampler iterates (sample 0 is the executed one), and per step the task facts + the
recording-camera frame (one mp4 per env, frame i = env step i).
python pour_record.py --kind diffusion --checkpoint .../runs/bc_lag1200/bc_best.pt --ddim-steps 50 --out-dir runs/pour/diffusion --seed 201 --num-envs 12"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True   # TRLDistill_v5 is checksum-sealed: never write .pyc into it
import argparse, json, os, subprocess, sys, traceback
from pathlib import Path
from isaaclab.app import AppLauncher
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pour_common as pc

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["diffusion", "flow"], required=True); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ddim-steps", type=int, required=True); p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=201); p.add_argument("--num-envs", type=int, default=12); p.add_argument("--samples", type=int, default=32)
    p.add_argument("--store-iters", type=int, default=10, help="diffusion: keep this many evenly spaced sampler states besides the noise (flow keeps all)")
    p.add_argument("--after-success", type=int, default=60, help="env steps recorded after success (60 Hz => 1 s)")
    AppLauncher.add_app_launcher_args(p); return p.parse_args()

def trace_diffusion(model, cond, noise):
    """DDIM (eta=0), copy of DiffusionPolicy.sample keeping every x_i."""
    import torch
    b = noise.shape[0]; x, xs = noise, [noise]
    for i, t in enumerate(model.ddim_steps):
        prev = model.ddim_steps[i + 1] if i + 1 < len(model.ddim_steps) else torch.tensor(-1, device=noise.device)
        x, _ = model.ddim_mean(x, model.denoiser_for(i)(x, t.expand(b), cond), t.expand(b), prev.expand(b), eta=0.0); xs.append(x)
    xs[-1] = x.clamp(-1.0, 1.0); return xs

def trace_flow(model, cond, noise):
    """Euler integration, copy of FlowPolicy.sample keeping every x_i."""
    import torch
    b = noise.shape[0]; k = len(model.ddim_steps); x, xs = noise, [noise]
    for i in range(k):
        t = torch.full((b,), i / k, device=noise.device); x = x + model.denoiser_for(i)(x, t * model.cfg.num_train_timesteps, cond) / k; xs.append(x)
    xs[-1] = x.clamp(-1.0, 1.0); return xs

def main(a):
    import numpy as np, torch
    import dp_core.model
    from dp_core.flow import FlowPolicy
    if a.kind == "flow": dp_core.model.DiffusionPolicy = FlowPolicy
    import eval_policy
    from dp_core.model import ddim_timesteps
    from dp_core.env import right_image
    from dp_core.stepper import ChunkStepper
    dev = torch.device(a.device); N = a.num_envs; S = a.samples
    torch.manual_seed(a.seed)
    model, cfg, kind = eval_policy.load_policy(a.checkpoint, dev)
    model.ddim_steps = ddim_timesteps(cfg.num_train_timesteps, a.ddim_steps).to(dev)
    assert cfg.exec_horizon == 16 and cfg.pred_horizon == 32
    out = a.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    env, raw = pc.build(N, dev, a.seed, out / "env", stage="eval", record=True)
    from utils import oracle_tune as ot
    P_R_POUR = np.array([0.08, 0.28, 1.34]) + np.asarray(ot.pouring_hand_offsets()[1]) + np.asarray(ot.get("pouring_pour_shift"))
    step = ChunkStepper(env, cfg.obs_horizon, cfg.exec_horizon, dev, right_image)
    horizon = int(raw.max_episode_length); org = raw.scene.env_origins
    tr = trace_flow if a.kind == "flow" else trace_diffusion
    gen = torch.Generator(device=dev).manual_seed(a.seed + 1000)
    K = a.ddim_steps
    sel = list(range(K + 1)) if a.kind == "flow" else sorted(set(round(i * K / a.store_iters) for i in range(a.store_iters + 1)))
    def writer(i):
        return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{pc.REC_W}x{pc.REC_H}", "-r", "30", "-i", "-", "-c:v", "libx264",
                                 "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "fast", str(out / f"e{i}.mp4")], stdin=subprocess.PIPE)
    wr = [writer(i) for i in range(N)]
    alive = np.ones(N, bool); t_succ = np.full(N, -1); t_end = np.full(N, -1)
    L = {k: [] for k in ("rh", "lh", "bottle", "cup", "subgoal", "succ", "act", "rq", "lq")}
    calls, ITER, CALLQ, ST = [], [], [], []
    init = None; check = None
    with torch.no_grad():
        for t in range(horizon):
            if t == 0:
                init = {"cup": (raw.cup.data.root_pos_w - org).cpu().numpy(), "bottle": (raw.bottle.data.root_pos_w - org).cpu().numpy()}
            need = step.needs_plan()
            if t % 16 == 0 and need.any():
                idx = need.nonzero().flatten(); n = len(idx)
                image, prop = step.image_history[idx], step.obs_history[idx]
                cond = model.encode(image, prop)
                rng = torch.cuda.get_rng_state(dev)
                noise = torch.randn(n, cfg.pred_horizon, cfg.action_dim, device=dev)
                xs = tr(model, cond, noise)
                if check is None:       # the traced loop must reproduce model.sample exactly for the executed sample
                    torch.cuda.set_rng_state(rng, dev); ref = model.sample(image, prop); check = float((ref - xs[-1]).abs().max()); print("[check] trace vs model.sample max diff", check, flush=True)
                noise2 = torch.randn(n * (S - 1), cfg.pred_horizon, cfg.action_dim, device=dev, generator=gen)
                xs2 = tr(model, cond.repeat_interleave(S - 1, 0), noise2)
                it = torch.stack([torch.cat([x1[:, None], x2.reshape(n, S - 1, *x2.shape[1:])], 1) for x1, x2 in zip(xs, xs2)])      # [K+1, n, S, 32, 14]
                full = torch.zeros(len(sel), N, S, 16, 14, dtype=torch.float16)
                full[:, idx.cpu()] = it[sel][:, :, :, :16].half().cpu()
                ITER.append(full.numpy()); CALLQ.append(t)
                plan = xs[-1].clone(); plan[..., [6, 13]] = torch.where(plan[..., [6, 13]] >= 0, 1.0, -1.0); step.set_plan(need, plan)
                ST.append({k: v.cpu().numpy() for k, v in pc.get_state(raw).items()})
            # per step facts (state before the action)
            rows = torch.arange(N, device=dev); act = step.plan[rows, step.cursor]
            L["rh"].append((raw.right_robot.data.body_pos_w[:, 6] - org).cpu().numpy()); L["lh"].append((raw.left_robot.data.body_pos_w[:, 6] - org).cpu().numpy())
            for nm, o in (("bottle", raw.bottle), ("cup", raw.cup)):
                x = o.data.root_state_w[:, :7].clone(); x[:, :3] -= org; L[nm].append(x.cpu().numpy())
            L["subgoal"].append(raw.subgoal.cpu().numpy()); L["succ"].append(raw.success_buf.cpu().numpy()); L["act"].append(act.cpu().numpy())
            L["rq"].append(raw.right_robot.data.joint_pos[:, :6].cpu().numpy()); L["lq"].append(raw.left_robot.data.joint_pos[:, :6].cpu().numpy())
            frames = raw.get_camera_debug_images()["record_rgb"]
            if frames.shape[-1] != 3: frames = frames.permute(0, 2, 3, 1)
            frames = frames[..., :3].float(); frames = (frames * 255.0 if float(frames.max()) <= 1.01 else frames).clamp(0, 255).byte().cpu().numpy()
            sc = raw.success_buf.cpu().numpy()
            for i in range(N):
                if alive[i]:
                    wr[i].stdin.write(frames[i].tobytes())
                    if sc[i] and t_succ[i] < 0: t_succ[i] = t
                    if t_succ[i] >= 0 and t - t_succ[i] >= a.after_success: alive[i] = False; t_end[i] = t
            _, done, _, infos = step.step()
            for i in done.nonzero().flatten().tolist():
                if alive[i]: alive[i] = False; t_end[i] = t
            if not alive.any(): break
    for w in wr: w.stdin.close(); w.wait()
    T = len(L["act"])
    rate = int((t_succ >= 0).sum()); print(f"[run] {a.kind}: success {rate}/{N} envs, t_succ {t_succ.tolist()}, t_end {t_end.tolist()}", flush=True)
    sdict = {k: np.stack([s[k] for s in ST]) for k in ST[0]}
    np.savez_compressed(out / "iterates.npz", iter=np.stack(ITER).astype(np.float16), sel=np.array(sel), call_step=np.array(CALLQ), **{f"state_{k}": v for k, v in sdict.items()})
    np.savez_compressed(out / "episode.npz", t_succ=t_succ, t_end=t_end, P_R_POUR=P_R_POUR, init_cup=init["cup"], init_bottle=init["bottle"],
                        bottle_pour_offset=raw.bottle_pour_offset[0].cpu().numpy(), cup_opening_offset=raw.cup_opening_offset[0].cpu().numpy(),
                        **{k: np.stack(v, 1) for k, v in L.items()})
    json.dump({"kind": a.kind, "checkpoint": str(a.checkpoint), "ddim_steps": K, "seed": a.seed, "num_envs": N, "samples": S, "stored_iterate_indices": sel, "success": (t_succ >= 0).tolist(),
               "t_succ": t_succ.tolist(), "t_end": t_end.tolist(), "trace_vs_sample_maxdiff": check, "horizon": horizon, "P_R_POUR": P_R_POUR.tolist()}, open(out / "meta.json", "w"), indent=1)

if __name__ == "__main__":
    cli = parse_args(); cli.enable_cameras = True; cli.headless = True; app = AppLauncher(cli).app; code = 0
    try: main(cli)
    except BaseException: traceback.print_exc(); sys.stdout.flush(); code = 1
    sys.stdout.flush(); os._exit(code)
