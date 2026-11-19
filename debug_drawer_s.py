#!/usr/bin/env python3
"""Debug the scripted drawer teacher (no cameras): phase log and drawer opening per env."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from isaaclab.app import AppLauncher
sys.path.insert(0, str(Path(__file__).resolve().parent))
p = argparse.ArgumentParser(); p.add_argument("--envs", type=int, default=8); p.add_argument("--seed", type=int, default=5); p.add_argument("--roll", type=float, default=0.0); p.add_argument("--posture", type=float, default=0.0); p.add_argument("--home-after", type=int, default=0); p.add_argument("--trace", type=int, default=-1); p.add_argument("--tf", type=float, default=0.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.headless = True
app = AppLauncher(args).app
import torch  # noqa: E402
from fd_core.scripted import DrawerScript  # noqa: E402
from fd_core.tasks import TASKS, Arm, drawer_pos, make_task_env  # noqa: E402
env, raw = make_task_env("drawer", args.envs, args.device, args.seed)
arm = Arm(raw)
obs, _ = env.reset()
sc = DrawerScript(raw, arm, roll=args.roll, posture=args.posture, home_after=args.home_after, translate_first=args.tf)
names = ["PRE", "REACH", "GRASP", "PULL", "HOLD"]
prev = sc.phase.clone(); log = [[] for _ in range(args.envs)]
for t in range(TASKS["drawer"].steps - 1):
    a = sc(t, obs, None, None)
    obs, *_ = env.step(arm.norm_to_env(a))
    for i in torch.nonzero(sc.phase != prev).flatten().tolist():
        log[i].append(f"{t}:{names[int(sc.phase[i])]}")
    prev = sc.phase.clone()
    if int(sc.phase[0]) == 3 and t % 8 == 0:
        from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat as mq
        d = raw.scene["robot"].data; cf = raw.scene["cabinet_frame"].data
        hp, hq = sc._local(d.body_pos_w[:, sc.hand], d.body_quat_w[:, sc.hand])
        tcpw = d.body_pos_w[:, sc.hand] + (mq(d.body_quat_w[:, sc.hand]) @ torch.tensor([0, 0, 0.1034], device=args.device))
        lf = d.body_pos_w[:, robot_lf] if False else None
        rp, _ = subtract_frame_transforms(cf.target_pos_w[:, 0], cf.target_quat_w[:, 0], tcpw)
        print(f"R t{t} ph{int(sc.phase[0])} tcp_in_handle {[round(x,4) for x in rp[0].tolist()]} fingers {[round(x,3) for x in d.joint_pos[0,7:].tolist()]} drawer {drawer_pos(raw)[0]:.3f} lag {(sc.cmd[0]-sc._local(tcpw)[0][0]).norm():.3f} q {[round(x,2) for x in d.joint_pos[0,:7].tolist()]}", flush=True)
    if t % 20 == 0 and False:
        from isaaclab.utils.math import matrix_from_quat
        d = raw.scene["robot"].data
        hp, hq = sc._local(d.body_pos_w[:, sc.hand], d.body_quat_w[:, sc.hand])
        cf = raw.scene["cabinet_frame"].data
        h_p, h_q = sc._local(cf.target_pos_w[:, 0], cf.target_quat_w[:, 0])
        Rh = matrix_from_quat(hq[0]); Rt = matrix_from_quat(h_q[0])
        print(f"t{t} hand {[round(x,3) for x in hp[0].tolist()]} handle {[round(x,3) for x in h_p[0].tolist()]} hand_z {[round(x,2) for x in Rh[:,2].tolist()]} handle_z {[round(x,2) for x in Rt[:,2].tolist()]} q {[round(x,2) for x in d.joint_pos[0,:7].tolist()]}", flush=True)
    if args.trace >= 0 and t % 6 == 0 and t < 120:
        from isaaclab.utils.math import matrix_from_quat as _m
        i = args.trace; d = raw.scene["robot"].data; cf = raw.scene["cabinet_frame"].data
        hp, hq = sc._local(d.body_pos_w[:, sc.hand], d.body_quat_w[:, sc.hand]); tc = hp + _m(hq) @ torch.tensor([0, 0, 0.1034], device=args.device)
        h_p, h_q = sc._local(cf.target_pos_w[:, 0], cf.target_quat_w[:, 0]); hz = _m(h_q)[..., 2]
        print(f"T t{t} ph{int(sc.phase[i])} err {(h_p[i]-0.08*hz[i]-tc[i]).norm():.3f} tcp {[round(x,3) for x in tc[i].tolist()]} rot {(_m(hq[i])[:,2]*hz[i]).sum():.2f} q {[round(x,2) for x in d.joint_pos[i,:7].tolist()]}", flush=True)
    if t % 40 == 0:
        print(f"t{t} drawer {[round(x,3) for x in drawer_pos(raw).tolist()]}", flush=True)
dp = drawer_pos(raw)
from isaaclab.utils.math import matrix_from_quat as _mq
d = raw.scene["robot"].data; cf = raw.scene["cabinet_frame"].data
hp, hq = sc._local(d.body_pos_w[:, sc.hand], d.body_quat_w[:, sc.hand])
tcp = hp + _mq(hq) @ torch.tensor([0, 0, 0.1034], device=args.device)
h_p, h_q = sc._local(cf.target_pos_w[:, 0], cf.target_quat_w[:, 0]); h_z = _mq(h_q)[..., 2]
tgt = h_p - 0.08 * h_z; lim = d.soft_joint_pos_limits[:, :7]
cab = raw.scene["cabinet"].data.root_state_w[:, :7] - torch.cat([raw.scene.env_origins, torch.zeros(args.envs, 4, device=args.device)], -1)
for i in torch.nonzero(sc.phase == 0).flatten().tolist():
    q = d.joint_pos[i, :7]; near = torch.minimum(q - lim[i, :, 0], lim[i, :, 1] - q)
    print(f"STUCK env{i} err {(tgt[i]-tcp[i]).norm():.3f} vec {[round(x,3) for x in (tgt[i]-tcp[i]).tolist()]} rot {(_mq(hq[i])[:,2]*h_z[i]).sum():.3f} q {[round(x,2) for x in q.tolist()]} lim_margin {[round(x,2) for x in near.tolist()]} cab {[round(x,3) for x in cab[i].tolist()]} hz {[round(x,2) for x in h_z[i].tolist()]}", flush=True)
for i in range(args.envs):
    print(f"env{i} open={dp[i]:.3f} :: " + " ".join(log[i]), flush=True)
sys.stdout.flush(); os._exit(0)
