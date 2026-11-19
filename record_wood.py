#!/usr/bin/env python3
"""Cosmetic drawer videos with a wood cabinet. The policy runs on the normal (white) cabinet exactly as in eval_task.py;
the robot / cabinet states of every step are recorded, then the cabinet gets an oak material and the states are
replayed (no physics) while only the third-person video camera renders. The policy never sees the wood.
Writes <video-dir>/drawer_s<seed>_e<i>_<outcome>.mp4."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", default=None); p.add_argument("--teacher", default=None)
p.add_argument("--num-envs", type=int, default=6); p.add_argument("--seed", type=int, default=31)
p.add_argument("--video-dir", type=Path, required=True)
p.add_argument("--soft-arm", action="store_true", help="default-gain arm, as used by the PPO drawer runs (runs/drawer)")
p.add_argument("--material", default="Materials/Base/Wood/Oak.mdl")
p.add_argument("--probe", action="store_true", help="only render one wood frame per env to check the material")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.enable_cameras = True; args.headless = True; args.task = "drawer"
app = AppLauncher(args).app


def main():
    import imageio.v2 as imageio
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR
    from fd_core.tasks import TASKS, Arm, make_task_env, run_episodes

    torch.manual_seed(args.seed)
    spec = TASKS["drawer"]
    env, raw = make_task_env("drawer", args.num_envs, args.device, args.seed, cameras=True, video_size=(480, 360),
                             stiff_arm=not args.soft_arm)
    arm = Arm(raw)
    robot, cab, cam = raw.scene["robot"], raw.scene["cabinet"], raw.scene["video_camera"]
    n = raw.num_envs

    def wood():
        mat = sim_utils.MdlFileCfg(mdl_path=f"{NVIDIA_NUCLEUS_DIR}/{args.material}", project_uvw=True, texture_scale=(0.6, 0.6))
        mat.func("/World/Looks/WoodCabinet", mat)
        for i in range(n):
            sim_utils.bind_visual_material(f"/World/envs/env_{i}/Cabinet", "/World/Looks/WoodCabinet", stronger_than_descendants=True)

    def render():
        for _ in range(2):                       # let the renderer settle on the written state
            raw.sim.render()
        cam.update(raw.step_dt, force_recompute=True)
        return cam.data.output["rgb"][..., :3].to(torch.uint8).cpu()

    if args.probe:
        env.reset(); wood()
        for _ in range(20):
            raw.sim.render()
        f = render()
        args.video_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            imageio.imwrite(args.video_dir / f"probe_e{i}.png", f[i].numpy())
        print("probe written", flush=True); return

    if args.checkpoint:
        from fd_core.policy import load_dp
        from fd_core.rollout import DiffusionActor
        model, cfg = load_dp(args.checkpoint, args.device)
        actor, obs_h = DiffusionActor(model, cfg.exec_horizon), cfg.obs_horizon
    else:
        from collect_task import make_teacher
        actor, obs_h = make_teacher(args, arm, raw), 2

    # 1) policy rollout on the white cabinet, logging the full state after each step (the step's own video frame
    #    in eval_task.py is taken before the action, i.e. state t; log the same)
    log = []
    orig_step = env.step

    def step_and_log(a):
        log.append((robot.data.joint_pos.clone(), cab.data.joint_pos.clone(), cab.data.root_state_w.clone(),
                    robot.data.root_state_w.clone()))
        return orig_step(a)
    env.step = step_and_log
    r = run_episodes(env, raw, arm, actor, "drawer", obs_horizon=obs_h)
    env.step = orig_step
    names = [spec.outcome_names[int(o)] for o in r["outcome"]]
    print(f"[wood] rollout success {r['success'].float().mean():.3f} outcomes {names}", flush=True)

    # 2) wood material, replay the states without physics
    wood()
    for _ in range(20):
        raw.sim.render()
    ids = torch.arange(n, device=raw.device)
    frames = []
    for (rq, cq, croot, rroot) in log:
        robot.write_root_state_to_sim(rroot, ids); cab.write_root_state_to_sim(croot, ids)
        robot.write_joint_state_to_sim(rq, torch.zeros_like(rq), env_ids=ids)
        cab.write_joint_state_to_sim(cq, torch.zeros_like(cq), env_ids=ids)
        raw.sim.forward()
        raw.scene.update(0.0)
        frames.append(render())
    frames = torch.stack(frames, 1)
    args.video_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        w = imageio.get_writer(args.video_dir / f"drawer_s{args.seed}_e{i}_{names[i]}.mp4", fps=30, codec="libx264", quality=8)
        for f in frames[i].numpy():
            w.append_data(f)
        w.close()
    print(f"[wood] wrote {n} videos to {args.video_dir}", flush=True)


code = 0
try:
    main()
except BaseException:
    traceback.print_exc(); code = 1
sys.stdout.flush(); os._exit(code)
