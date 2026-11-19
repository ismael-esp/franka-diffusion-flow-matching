"""Build the TRLDistill_v5 pouring env with the observations the real robot can provide.

The policy sees the student contract of hw/hardware_env.py: 14D arm and grip joints plus
the front-right RGB image. No teacher is used; only the demo collection settings stored in
release/pouring are read. TRLDistill_v5 is imported read-only because it is sealed by
checksums, so entry scripts must set sys.dont_write_bytecode before importing this module.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# TRLDistill_v5 is NOT part of this repository: set TRLDISTILL_V5 to its checkout (default: a sibling folder of this repo).
V5_ROOT = Path(os.environ.get("TRLDISTILL_V5", Path(__file__).resolve().parents[2].parent / "TRLDistill_v5"))
RELEASE = V5_ROOT / "release" / "pouring"
DEMO_TAG = "e2e_ab3ad819849f_final"
DEMO_PATH = RELEASE / "demos" / f"Demo_success_pouring_{DEMO_TAG}.pickle"
TRAJ_PATH = RELEASE / "demos" / f"Demo_Traj_pouring_{DEMO_TAG}.pickle"

IMAGE_HEIGHT, IMAGE_WIDTH = 120, 212  # half of the real 424x240; the aperture keeps the same field of view

# Measured camera poses in the Isaac Lab world (new_calib_data.txt).
CAMERA_OVERRIDES = {
    "vision_front_camera_eye_local": (0.08236, 1.6051, 1.50215),
    "vision_front_camera_target_local": (0.10607, 0.64064, 1.23898),
    "vision_front_camera_quat_wxyz": (0.709406, 0.092715, 0.095022, -0.692183),
    "vision_front_right_camera_eye_local": (-0.1626, 1.57834, 1.53127),
    "vision_front_right_camera_target_local": (-0.12263, 0.6238, 1.23587),
    "vision_front_right_camera_quat_wxyz": (0.713648, 0.103393, 0.107813, -0.684392),
}

# Rules per stage: "replay" reproduces demo collection, "train" is the real-matched
# training env with lag and randomization, "eval" is the strict nominal-look check.
STAGES = {
    "replay": dict(context="collect", real_match={"mode": "collect", "lag": False},
                   guard={"mode": "monitor", "table_cmd": True}, vision_dr=True, terminate=False),
    "train": dict(context="teacher", real_match={"mode": "teacher"},
                  guard={"mode": "terminate", "table_cmd": False, "arm_lim": -0.01}, vision_dr=True, terminate=True),
    "eval": dict(context="eval", real_match={"mode": "eval"},
                 guard={"mode": "terminate", "table_cmd": True}, vision_dr=False, terminate=False),
    "eval_nolag": dict(context="eval", real_match={"mode": "eval", "lag": False},
                       guard={"mode": "terminate", "table_cmd": True}, vision_dr=False, terminate=False),
    "collect_lag": dict(context="eval", real_match={"mode": "eval"},
                        guard={"mode": "terminate", "table_cmd": True}, vision_dr=True, terminate=False),
}


def student_overrides(seed: int) -> dict:
    """The env fields the RGB student runner sets, without its teacher or depth paths."""
    return {
        "seed": int(seed),
        "randomize_object_pos": True,
        "use_vision": True,
        "vision_student_raw_mode": True,
        "distill_dual_obs": True,
        "distill_small_obs": True,
        "distill_arm_grip_obs": True,
        "strict_vision_only_policy": True,
        "vision_use_depth": False,
        "vision_use_front_camera": True,
        "vision_use_front_right_camera": True,
        "vision_camera_width": IMAGE_WIDTH,
        "vision_camera_height": IMAGE_HEIGHT,
        "vision_resize_h": IMAGE_HEIGHT,
        "vision_resize_w": IMAGE_WIDTH,
        "vision_update_stride": 1,
        "num_rerenders_on_reset": 2,  # otherwise the first frame of an episode is the previous episode's
        "dextrah_right_img_flip": False,
        "dextrah_use_hardware_camera_pose": True,
        "dextrah_hardware_camera_pose_mode": "view_target",
        "dextrah_camera_pose_rand_enable": False,
        **CAMERA_OVERRIDES,
    }


def calibrated_camera_cfgs() -> dict:
    """Camera configs that spawn at the calibrated pose.

    Isaac Lab 2.3.2 writes set_world_poses only to Fabric and the pose reverts to the spawn
    offset before the first frame, so the env's per-reset placement alone leaves the
    cameras at the env origin.
    """
    from isaaclab.sensors import TiledCameraCfg
    from tasks.task_pouring import PouringEnvCfg

    defaults = PouringEnvCfg()
    configs = {}
    for name in ("front", "front_right"):
        offset = TiledCameraCfg.OffsetCfg(
            pos=CAMERA_OVERRIDES[f"vision_{name}_camera_eye_local"],
            rot=CAMERA_OVERRIDES[f"vision_{name}_camera_quat_wxyz"],
            convention="world",
        )
        configs[f"{name}_camera_cfg"] = getattr(defaults, f"{name}_camera_cfg").replace(offset=offset)
    return configs


def recording_camera_overrides() -> dict:
    """The third-person video camera, spawned at its pose for the reason in calibrated_camera_cfgs."""
    import torch
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab.utils.math import quat_from_matrix
    from tasks.task_pouring import PouringEnvCfg

    defaults = PouringEnvCfg()
    eye = torch.tensor(defaults.recording_camera_eye_local)
    forward = torch.nn.functional.normalize(torch.tensor(defaults.recording_camera_target_local) - eye, dim=0)
    left = torch.nn.functional.normalize(torch.linalg.cross(torch.tensor([0.0, 0.0, 1.0]), forward), dim=0)
    rot = quat_from_matrix(torch.stack([forward, left, torch.linalg.cross(forward, left)], dim=1))
    offset = TiledCameraCfg.OffsetCfg(pos=tuple(eye.tolist()), rot=tuple(rot.tolist()), convention="world")
    return {"recording_camera_enable": True, "recording_camera_cfg": defaults.recording_camera_cfg.replace(offset=offset)}


def check_camera_poses(raw_env) -> None:
    """Refuse to run if a camera is not where the calibration puts it."""
    import torch

    for name, attr in (("front", "vision_camera_front"), ("front_right", "vision_camera_front_right")):
        eye = raw_env.env_origins.new_tensor(CAMERA_OVERRIDES[f"vision_{name}_camera_eye_local"])
        actual = getattr(raw_env, attr)._view.get_world_poses()[0]
        error = float(torch.norm(actual - raw_env.env_origins - eye, dim=-1).max())
        if error > 1e-3:
            raise RuntimeError(f"{name} camera is {error:.3f} m from its calibrated pose")


def build_pouring_env(num_envs: int, device, seed: int, run_dir: Path, *, stage: str, record: bool = False):
    """Return (vec_env, raw_env) for one of the STAGES; record adds the third-person camera."""
    rules = STAGES[stage]
    os.chdir(V5_ROOT)
    sys.path.insert(0, str(V5_ROOT))
    run_dir.mkdir(parents=True, exist_ok=True)

    context = json.loads((RELEASE / "demos" / "context.json").read_text())
    context["stage"] = rules["context"]
    context["env"]["seed"] = int(seed)
    context["raw_limits"] = json.loads((RELEASE / "teacher" / "context.json").read_text())["raw_limits"]
    context_path = run_dir / "env_context.json"
    context_path.write_text(json.dumps(context, indent=1))
    os.environ["TRL_E2E_CONTEXT"] = str(context_path)
    os.environ["TRL_REAL_MATCH"] = json.dumps(rules["real_match"])
    os.environ["TRL_CONTACT_GUARD"] = json.dumps(rules["guard"])
    os.environ["TRL_ORACLE_TUNE"] = (RELEASE / "demos" / "oracle_tune.json").read_text()
    if rules["vision_dr"]:
        os.environ["TRL_VISION_DR"] = json.dumps({"mode": "on"})
    else:
        os.environ.pop("TRL_VISION_DR", None)

    from task_registry import make_env

    overrides = student_overrides(seed)
    (run_dir / "env_overrides.json").write_text(json.dumps(overrides, indent=1, sort_keys=True))
    overrides.update(calibrated_camera_cfgs())
    if record:
        overrides.update(recording_camera_overrides())
    env, raw_env, _ = make_env("pouring", num_envs, device=str(device), env_cfg_overrides=overrides)
    check_camera_poses(raw_env)
    raw_env.cfg.terminate_on_success = rules["terminate"]
    return env, raw_env


def right_image(obs_dict: dict) -> "torch.Tensor":  # noqa: F821
    """Front-right RGB as [N,3,H,W] floats in [0,1], whatever layout the env used."""
    image = obs_dict["img_right"]
    if image.shape[-1] == 3:
        image = image.permute(0, 3, 1, 2)
    return image.float()
