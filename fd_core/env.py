"""Isaac Lab Franka cube-lift (Isaac-Lift-Cube-Franka-v0) with a workspace RGB camera, for the Diffusion Policy.

The privileged PPO teacher sees the env's own 36-d policy observation (it includes the cube position).
The student sees only the camera image and proprio = joint pos (9) + joint vel (9) + goal position (3).
Student actions are absolute commands in [-1, 1]: 7 arm joint targets normalized by the joint limits, then
the gripper (+1 open, -1 close), the same convention as pouring_diffusion.
"""
from __future__ import annotations

import math

import torch

TASK = "Isaac-Lift-Cube-Franka-v0"
EPISODE_STEPS = 250            # 5 s at 50 Hz
SUCCESS_DIST = 0.05
import os as _os
RENDER_V2 = _os.environ.get("FD_RENDER", "v1") == "v2"   # default v1 = the renderer every reported model was trained/evaluated with; FD_RENDER=v2 = no AI denoiser + no sampled lighting (experimental)            # cube centre within 5 cm of the goal at the last step
PROPRIO_DIM = 21
ACTION_DIM = 8
CAM_EYE = (1.05, 0.35, 0.62)   # env frame; robot base at the origin facing +x, table centre at x=0.5
CAM_TARGET = (0.45, 0.0, 0.20)
ZOOM_FOCAL, ZOOM_TARGET = 25.0, (0.5, 0.0, 0.13)   # option 1: 1.4x larger cube; all start positions + first 27 cm of lift in view


def look_at_quat(eye, target) -> tuple[float, float, float, float]:
    """wxyz quaternion for Isaac Lab's "world" camera convention (forward +x, up +z)."""
    f = [t - e for t, e in zip(target, eye)]
    n = math.sqrt(sum(v * v for v in f))
    f = [v / n for v in f]
    yaw, pitch = math.atan2(f[1], f[0]), -math.asin(f[2])
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return (cy * cp, -sy * sp, cy * sp, sy * cp)   # qz(yaw) * qy(pitch)


def _robustify_obs(cfg, noise: bool):
    """Teacher observation of the robust teacher: no last-action term; noisy joints/cube while training."""
    from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
    from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
    pol = cfg.observations.policy
    pol.actions = None
    if noise:
        pol.joint_pos.noise = Unoise(n_min=-0.01, n_max=0.01)
        pol.joint_vel.noise = Unoise(n_min=-0.05, n_max=0.05)
        pol.object_position.noise = Gnoise(std=0.005)
    pol.enable_corruption = noise


def robust_teacher_cfg(num_envs: int, device: str, seed: int, train: bool = True):
    """Training env of the robust (distillable) teacher: see train_teacher_robust.py."""
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg
    cfg = FrankaCubeLiftEnvCfg()
    cfg.scene.num_envs = num_envs; cfg.sim.device = device; cfg.seed = seed
    _robustify_obs(cfg, noise=train)
    if train:
        cfg.events.reset_arm = EventTerm(func=mdp.reset_joints_by_offset, mode="reset", params={
            "position_range": (-0.15, 0.15), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])})
        cfg.events.push_cube = EventTerm(func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(1.0, 3.0), params={
            "velocity_range": {"x": (-0.6, 0.6), "y": (-0.6, 0.6), "z": (0.0, 0.3)}, "asset_cfg": SceneEntityCfg("object")})
    return cfg


def make_env(num_envs: int, device: str, seed: int, image_size: int = 128, video_size: tuple[int, int] | None = None,
             wrist: bool = False, robust_teacher: bool = False, zoom: bool = False):
    """The registered lift env with a tiled RGB camera per env (and optionally a larger one for videos)."""
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg
    import isaaclab_tasks  # noqa: F401  (registers the tasks)

    cfg = FrankaCubeLiftEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = 8.0                          # keep neighbouring robots out of the camera view
    cfg.sim.device = device
    cfg.seed = seed
    cfg.commands.object_pose.debug_vis = False          # no goal markers in the images; the goal is in proprio
    cfg.observations.policy.enable_corruption = False
    if robust_teacher:
        _robustify_obs(cfg, noise=False)                 # clean teacher obs without the last-action term
    cfg.rerender_on_reset = True                        # first frame after a reset shows the new scene
    cfg.sim.render.antialiasing_mode = "FXAA"           # no temporal AA: TAA/DLSS ghost the cube's old pose after a reset
    if RENDER_V2:   # deterministic images: in ~1 of 8 processes the stochastic sampled lighting + AI denoiser corrupted
        cfg.sim.render.enable_dl_denoiser = False        # the wrist frames with blotches and blocky patches (runs/v3_debug)
    rot = look_at_quat(CAM_EYE, ZOOM_TARGET if zoom else CAM_TARGET)
    cfg.scene.camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/WorkspaceCam", update_period=0.0, height=image_size, width=image_size, data_types=["rgb"],
        offset=TiledCameraCfg.OffsetCfg(pos=CAM_EYE, rot=rot, convention="world"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=ZOOM_FOCAL if zoom else 18.0, horizontal_aperture=20.955, clipping_range=(0.05, 10.0)),
    )
    if video_size is not None:   # defined BEFORE the wrist camera: defined after it, the wrist camera rendered from a wrong pose
        eye, target = (1.45, 0.75, 0.75), (0.42, 0.0, 0.18)
        from isaaclab.sensors import CameraCfg   # standard (non-tiled) camera: a second-resolution TiledCamera made the tiled
        cfg.scene.video_camera = CameraCfg(       # renderer draw the wrist camera with the wrong projection (runs/vidcam_compare.png)
            prim_path="{ENV_REGEX_NS}/VideoCam", update_period=0.0, height=video_size[1], width=video_size[0],
            data_types=["rgb"], offset=CameraCfg.OffsetCfg(pos=eye, rot=look_at_quat(eye, target), convention="world"),
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.05, 10.0)),
        )
    if wrist:   # eye-in-hand camera on the back of panda_hand, tilted 35 deg to look along the fingers at the fingertips
        cfg.scene.wrist_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam", update_period=0.0, height=image_size, width=image_size,
            data_types=["rgb"],
            offset=TiledCameraCfg.OffsetCfg(pos=(0.07, 0.0, 0.02), rot=(0.9537, 0.0, -0.3007, 0.0), convention="ros"),
            spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=20.955, clipping_range=(0.01, 2.0)),
        )
    env = gym.make(TASK, cfg=cfg)
    if RENDER_V2:   # set after the env exists: camera/replicator setup resets this setting to the preset value
        import carb
        carb.settings.get_settings().set_bool("/rtx/directLighting/sampledLighting/enabled", False)
    return env, env.unwrapped


class Franka:
    """Observation, action and success helpers around the raw ManagerBasedRLEnv."""

    def __init__(self, raw):
        self.raw = raw
        robot = raw.scene["robot"]
        self.robot = robot
        self.arm_ids = list(range(7))
        lim = robot.data.soft_joint_pos_limits[0, :7]
        self.low, self.high = lim[:, 0].clone(), lim[:, 1].clone()
        self.default = robot.data.default_joint_pos[0, :7].clone()
        self.scale = 0.5                                     # JointPositionActionCfg(scale=0.5, use_default_offset=True)

    # ---- actions
    def env_to_norm(self, env_action: torch.Tensor) -> torch.Tensor:
        """Env action [N,8] (arm offsets/0.5, gripper sign) -> normalized absolute command [N,8]."""
        q = self.default + self.scale * env_action[:, :7]
        arm = (2 * (q - self.low) / (self.high - self.low) - 1).clamp(-1, 1)
        grip = torch.where(env_action[:, 7:8] < 0, -1.0, 1.0)
        return torch.cat([arm, grip], dim=-1)

    def norm_to_env(self, norm: torch.Tensor) -> torch.Tensor:
        q = self.low + (norm[:, :7].clamp(-1, 1) + 1) * 0.5 * (self.high - self.low)
        return torch.cat([(q - self.default) / self.scale, torch.where(norm[:, 7:8] < 0, -1.0, 1.0)], dim=-1)

    # ---- observations
    def teacher_obs(self) -> torch.Tensor:
        return self.raw.observation_manager.compute_group("policy")

    def proprio(self) -> torch.Tensor:
        """[N,1,21] joint pos rel, joint vel rel, goal position in the robot base frame."""
        d = self.robot.data
        cmd = self.raw.command_manager.get_command("object_pose")[:, :3]
        return torch.cat([d.joint_pos - d.default_joint_pos, d.joint_vel - d.default_joint_vel, cmd], dim=-1)[:, None]

    def image(self, name: str = "camera") -> torch.Tensor:
        """uint8 [N,H,W,3]; with a wrist camera the student image is [workspace | wrist], [N,H,2W,3]."""
        img = self.raw.scene[name].data.output["rgb"][..., :3].to(torch.uint8)
        if name == "camera" and "wrist_cam" in self.raw.scene.keys():
            img = torch.cat([img, self.raw.scene["wrist_cam"].data.output["rgb"][..., :3].to(torch.uint8)], dim=2)
        return img

    def cube_pos(self) -> torch.Tensor:
        """[N,3] cube centre in the robot base frame (privileged; only an auxiliary training target for the student)."""
        from isaaclab.utils.math import subtract_frame_transforms
        obj = self.raw.scene["object"].data.root_pos_w
        return subtract_frame_transforms(self.robot.data.root_pos_w, self.robot.data.root_quat_w, obj)[0]

    def goal_distance(self) -> torch.Tensor:
        from isaaclab.utils.math import subtract_frame_transforms
        obj = self.raw.scene["object"].data.root_pos_w
        obj_b, _ = subtract_frame_transforms(self.robot.data.root_pos_w, self.robot.data.root_quat_w, obj)
        return (obj_b - self.raw.command_manager.get_command("object_pose")[:, :3]).norm(dim=-1)


def load_teacher(path, device):
    """rsl_rl actor MLP (ELU) from a training checkpoint; deterministic mean action."""
    import torch.nn as nn
    sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
    layers = []
    for j, i in enumerate(idx):
        w, b = sd[f"actor.{i}.weight"], sd[f"actor.{i}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0]); lin.weight.data.copy_(w); lin.bias.data.copy_(b)
        layers.append(lin)
        if j < len(idx) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers).to(device).eval()
