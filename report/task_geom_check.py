import os, sys, traceback
from pathlib import Path
from isaaclab.app import AppLauncher
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
p = argparse.ArgumentParser(); p.add_argument("--task"); AppLauncher.add_app_launcher_args(p)
def main(a):
    import numpy as np, torch
    from fd_core.tasks import make_task_env, Arm
    from geom import fk
    for task in (a.task,):
        env, raw = make_task_env(task, 2, a.device, 3, cameras=False); arm = Arm(raw); env.reset()
        r = raw.scene["robot"]; org = raw.scene.env_origins
        hid = r.find_bodies("panda_hand")[0][0]
        print(task, "root pos", (r.data.root_pos_w - org)[0].cpu().numpy(), "quat", r.data.root_quat_w[0].cpu().numpy(), "dt", raw.cfg.sim.dt, "decim", raw.cfg.decimation)
        hand = (r.data.body_pos_w[:, hid] - org).cpu().numpy(); q = r.data.joint_pos[:, :7].cpu().numpy()
        print(" hand", hand[0], "fk", fk(q[0], tcp=False), "diff", hand[0] - fk(q[0], tcp=False))
        print(" scene keys", list(raw.scene.keys())); 
        if task == "drawer":
            print(" handle", (raw.scene["cabinet_frame"].data.target_pos_w[0, 0] - org[0]).cpu().numpy(), "cabinet root", (raw.scene["cabinet"].data.root_pos_w[0] - org[0]).cpu().numpy(), raw.scene["cabinet"].data.root_quat_w[0].cpu().numpy(), raw.scene["cabinet"].joint_names)
        else:
            for i in (1, 2, 3): print(" cube", i, (raw.scene[f"cube_{i}"].data.root_pos_w[0] - org[0]).cpu().numpy())
if __name__ == "__main__":
    c = p.parse_args(); c.enable_cameras = True; c.headless = True; app = AppLauncher(c).app; code = 0
    try: main(c)
    except BaseException: traceback.print_exc(); code = 1
    sys.stdout.flush(); os._exit(code)
