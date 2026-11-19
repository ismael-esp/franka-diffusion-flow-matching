"""Shared helpers for the pouring convergence video (pour_record.py, pour_sim.py, make_pour_data.py).
The pour task is the bimanual TRLDistill_v5 env (two UR3 + Allegro hands, rigid cup and bottle; there are NO water particles in this env:
pouring is judged from the bottle mouth, cup opening, hand pose and tilt). Everything is restorable: joints, objects and the real-match lag filters."""
import json, sys
from pathlib import Path
import numpy as np

import os
POUR = Path(os.environ.get("DP_ROOT", Path(__file__).resolve().parents[1] / "pouring_diffusion"))   # holds dp_core/ (and, for the pour runs, runs/)
sys.path.insert(0, str(POUR))
REC_W, REC_H = 848, 480
# recording camera (PouringEnvCfg): focal 24 mm, horizontal aperture 33.2367757485 mm, look-at (eye -> target) in the env frame
CAM_EYE, CAM_TARGET, FOCAL, APERT = (0.0, 1.85, 1.65), (0.0, 0.40, 1.12), 24.0, 33.2367757485


def project(p, W=REC_W, H=REC_H, eye=CAM_EYE, target=CAM_TARGET):
    """p [...,3] env frame -> pixels [...,2] of a W x H frame of the recording camera (pinhole, same model as geom.project_cam)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye; f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, f)
    d = np.asarray(p, float) - eye; z = d @ f; fx = FOCAL / APERT * W
    return np.stack([W / 2 + fx * (d @ right) / z, H / 2 - fx * (d @ up) / z], -1)


def quat_apply_np(q, v):
    """q [...,4] wxyz, v [...,3] -> rotated v"""
    q = np.asarray(q, float); v = np.asarray(v, float); w, xyz = q[..., :1], q[..., 1:]
    t = 2 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def build(num_envs, device, seed, run_dir, stage, record, cameras=True, guard=None, rec_size=(REC_W, REC_H)):
    """dp_core.env.build_pouring_env with a larger recording camera, optional no-camera env, optional contact-guard override (stage copy)."""
    import dp_core.env as E
    so = E.student_overrides
    def student(seed_):
        o = so(seed_)
        if record: o.update({"recording_camera_width": rec_size[0], "recording_camera_height": rec_size[1]})
        if not cameras:
            o.update({"use_vision": False, "vision_use_front_camera": False, "vision_use_front_right_camera": False})
        return o
    E.student_overrides = student
    if guard is not None:
        E.STAGES = dict(E.STAGES); E.STAGES["replay_" + stage] = dict(E.STAGES[stage], guard=guard); stage = "replay_" + stage
    if not cameras:                      # no cameras => nothing to verify
        E.check_camera_poses = lambda raw_env: None
    return E.build_pouring_env(num_envs, device, seed, Path(run_dir), stage=stage, record=record)


def get_state(raw):
    """full restorable state of all envs: joints of both robots, cup and bottle root states (env frame) and the real-match lag filters (ring stored relative to ptr)."""
    import torch
    org = raw.scene.env_origins; st = {}
    for s, rob in (("r", raw.right_robot), ("l", raw.left_robot)):
        st[f"{s}_q"] = rob.data.joint_pos.clone(); st[f"{s}_qd"] = rob.data.joint_vel.clone()
    for nm, obj in (("cup", raw.cup), ("bottle", raw.bottle)):
        x = obj.data.root_state_w[:, :13].clone(); x[:, :3] -= org; st[nm] = x
    for s in ("right", "left"):
        st[f"slew_{s}"] = raw._rm_slew[s].last.clone()
        for kind, d in (("arm", raw._rm_arm), ("hand", raw._rm_hand)):
            L = d[s]; K = L.d + 1
            st[f"{kind}_{s}_ring"] = torch.stack([L.ring[:, (L.ptr - k) % K] for k in range(K)], 1)
            st[f"{kind}_{s}_q"] = L.q.clone()
    return st


def set_state(raw, st, ids):
    """write a batch of states st (tensors [B,...], B = len(ids)) into envs ids of a freshly reset env."""
    org = raw.scene.env_origins[ids]
    for s, rob in (("r", raw.right_robot), ("l", raw.left_robot)):
        rob.write_joint_state_to_sim(st[f"{s}_q"], st[f"{s}_qd"], env_ids=ids); rob.set_joint_position_target(st[f"{s}_q"], env_ids=ids)
    for nm, obj in (("cup", raw.cup), ("bottle", raw.bottle)):
        x = st[nm].clone(); x[:, :3] += org; obj.write_root_state_to_sim(x, env_ids=ids)
    for s in ("right", "left"):
        raw._rm_slew[s].last[ids] = st[f"slew_{s}"]
        for kind, d in (("arm", raw._rm_arm), ("hand", raw._rm_hand)):
            L = d[s]; K = L.d + 1; rel = st[f"{kind}_{s}_ring"]
            for k in range(K): L.ring[ids, (L.ptr - k) % K] = rel[:, k]
            L.q[ids] = st[f"{kind}_{s}_q"]
    raw.episode_length_buf[ids] = 0
