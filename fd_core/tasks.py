"""Task-generic Franka environments for the teacher -> Diffusion Policy pipeline (drawer opening, cube stacking).

The lift code (fd_core.env / rollout) stays untouched so its reported results remain reproducible; this module adds
the same pieces for other Isaac Lab Franka tasks:
- `task_cfg(name, ...)`: the registered env config, made distillable (no last-action teacher obs, randomized
  object poses, 30 Hz control) plus the student cameras (workspace + wrist, same convention as lift),
- `Arm`: action normalization / proprio / images (the arm action scale is read from the env config),
- `TASKS[name].success(raw)` and `.outcome(...)`: task success and a coarse failure taxonomy,
- `run_episodes(...)`: one synchronized batch, same contract as fd_core.rollout.run_batch.
Student actions keep the lift convention: 7 normalized absolute joint targets + binary gripper (8-D)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from fd_core.env import look_at_quat


@dataclass
class TaskSpec:
    gym_id: str
    steps: int                      # control steps per episode
    cam_eye: tuple
    cam_target: tuple
    focal: float = 18.0
    video_eye: tuple = (1.45, 0.75, 0.75)
    video_target: tuple = (0.42, 0.0, 0.18)
    outcome_names: tuple = field(default_factory=lambda: ("success", "failure"))


TASKS = {
    # Cabinet at x=0.8 facing the robot; top drawer handle ~0.69 m high. 8 s at 30 Hz.
    "drawer": TaskSpec("Isaac-Open-Drawer-Franka-v0", 240, cam_eye=(0.05, -0.75, 1.05), cam_target=(0.62, 0.0, 0.55), focal=16.0,
                       video_eye=(-0.35, -1.25, 1.35), video_target=(0.55, 0.0, 0.5),
                       outcome_names=("success", "never_grasped", "partly_open", "slipped")),
    # Three cubes on the table, stack cube_2 on cube_1 and cube_3 on cube_2. 15 s at 20 Hz (the task's own rate).
    "stack": TaskSpec("Isaac-Stack-Cube-Franka-v0", 300, cam_eye=(1.05, 0.35, 0.62), cam_target=(0.5, 0.0, 0.05), focal=26.0,
                      outcome_names=("success", "none_stacked", "one_stacked", "toppled")),
}
DRAWER_OPEN = 0.30          # m of the top drawer's 0.39 m travel


def task_cfg(name: str, num_envs: int, device: str, seed: int, *, teacher_train: bool = False, cameras: bool = False,
             image_size: int = 128, wrist: bool = True, video_size=None, stiff_arm: bool = True):
    """stiff_arm (drawer): high-gain gravity-compensated Franka. The PPO drawer teacher (runs/drawer) used False."""
    import isaaclab.sim as sim_utils
    from isaaclab.envs import mdp
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
    spec = TASKS[name]
    if name == "drawer":
        from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.joint_pos_env_cfg import FrankaCabinetEnvCfg
        cfg = FrankaCabinetEnvCfg()
        cfg.decimation = 2                                   # 30 Hz control (sim stays at 60 Hz)
        cfg.sim.render_interval = cfg.decimation
        cfg.episode_length_s = spec.steps / 30.0
        pol = cfg.observations.policy
        pol.actions = None                                   # distillable teacher: no own-last-action input
        pol.joint_pos.noise = Unoise(n_min=-0.01, n_max=0.01)
        pol.joint_vel.noise = Unoise(n_min=-0.05, n_max=0.05)
        pol.rel_ee_drawer_distance.noise = Unoise(n_min=-0.005, n_max=0.005)
        pol.enable_corruption = teacher_train
        cfg.scene.cabinet_frame.debug_vis = False
        # Cabinet pose randomized every reset (the stock task always puts it at the same spot, which a camera
        # policy could memorize): +-5 cm depth, +-12 cm sideways, +-0.2 rad yaw.
        cfg.events.reset_cabinet = EventTerm(func=mdp.reset_root_state_uniform, mode="reset", params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.12, 0.12), "yaw": (-0.2, 0.2)}, "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cabinet")})
        if teacher_train:
            cfg.events.reset_robot_joints.params["position_range"] = (-0.15, 0.15)
        if stiff_arm:   # stiff gravity-compensated arm, needed by the scripted IK teacher (as in 'stack')
            from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
            init = cfg.scene.robot.init_state
            cfg.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", init_state=init)
    elif name == "stack":
        from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_joint_pos_env_cfg import FrankaCubeStackEnvCfg
        cfg = FrankaCubeStackEnvCfg()
        cfg.episode_length_s = spec.steps * cfg.sim.dt * cfg.decimation       # 15 s at 20 Hz
        # fixed-length synchronized episodes: no early resets on success or a dropped cube (scored at the end)
        cfg.terminations.success = None
        cfg.terminations.cube_1_dropping = None; cfg.terminations.cube_2_dropping = None; cfg.terminations.cube_3_dropping = None
        # stiff, gravity-compensated arm (Isaac Lab's FRANKA_PANDA_HIGH_PD_CFG): the default gains sag several cm under
        # gravity, so joint targets would not mean poses (the scripted IK teacher could not track)
        from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
        init = cfg.scene.robot.init_state
        cfg.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", init_state=init)
    else:
        raise KeyError(name)
    cfg.scene.num_envs = num_envs; cfg.sim.device = device; cfg.seed = seed
    if cameras:
        from isaaclab.sensors import CameraCfg, TiledCameraCfg
        cfg.scene.env_spacing = 8.0
        cfg.rerender_on_reset = True
        cfg.sim.render.antialiasing_mode = "FXAA"
        cfg.scene.camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/WorkspaceCam", update_period=0.0, height=image_size, width=image_size, data_types=["rgb"],
            offset=TiledCameraCfg.OffsetCfg(pos=spec.cam_eye, rot=look_at_quat(spec.cam_eye, spec.cam_target), convention="world"),
            spawn=sim_utils.PinholeCameraCfg(focal_length=spec.focal, horizontal_aperture=20.955, clipping_range=(0.05, 10.0)))
        if video_size is not None:   # standard camera, defined before the wrist camera (see fd_core.env.make_env)
            cfg.scene.video_camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/VideoCam", update_period=0.0, height=video_size[1], width=video_size[0], data_types=["rgb"],
                offset=CameraCfg.OffsetCfg(pos=spec.video_eye, rot=look_at_quat(spec.video_eye, spec.video_target), convention="world"),
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.05, 10.0)))
        if wrist:
            cfg.scene.wrist_cam = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam", update_period=0.0, height=image_size, width=image_size,
                data_types=["rgb"],
                offset=TiledCameraCfg.OffsetCfg(pos=(0.07, 0.0, 0.02), rot=(0.9537, 0.0, -0.3007, 0.0), convention="ros"),
                spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=20.955, clipping_range=(0.01, 2.0)))
    return cfg


def make_task_env(name, num_envs, device, seed, **kw):
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    cfg = task_cfg(name, num_envs, device, seed, **kw)
    env = gym.make(TASKS[name].gym_id, cfg=cfg)
    return env, env.unwrapped


class Arm:
    """Franka helpers that work for any of the task configs (arm action scale read from the env)."""

    def __init__(self, raw):
        self.raw = raw
        self.robot = raw.scene["robot"]
        lim = self.robot.data.soft_joint_pos_limits[0, :7]
        self.low, self.high = lim[:, 0].clone(), lim[:, 1].clone()
        self.default = self.robot.data.default_joint_pos[0, :7].clone()
        self.scale = float(raw.cfg.actions.arm_action.scale)

    def env_to_norm(self, a):
        q = self.default + self.scale * a[:, :7]
        arm = (2 * (q - self.low) / (self.high - self.low) - 1).clamp(-1, 1)
        return torch.cat([arm, torch.where(a[:, 7:8] < 0, -1.0, 1.0)], dim=-1)

    def norm_to_env(self, n):
        q = self.low + (n[:, :7].clamp(-1, 1) + 1) * 0.5 * (self.high - self.low)
        return torch.cat([(q - self.default) / self.scale, torch.where(n[:, 7:8] < 0, -1.0, 1.0)], dim=-1)

    def proprio(self):
        """[N,1,18] joint pos and vel relative to the defaults (no privileged object state)."""
        d = self.robot.data
        return torch.cat([d.joint_pos - d.default_joint_pos, d.joint_vel - d.default_joint_vel], dim=-1)[:, None]

    def image(self, name="camera"):
        img = self.raw.scene[name].data.output["rgb"][..., :3].to(torch.uint8)
        if name == "camera" and "wrist_cam" in self.raw.scene.keys():
            img = torch.cat([img, self.raw.scene["wrist_cam"].data.output["rgb"][..., :3].to(torch.uint8)], dim=2)
        return img


# ---------------------------------------------------------------- task progress / success
def drawer_pos(raw):
    cab = raw.scene["cabinet"]
    j = cab.joint_names.index("drawer_top_joint")
    return cab.data.joint_pos[:, j]


def handle_dist(raw):
    ee = raw.scene["ee_frame"].data.target_pos_w[:, 0]
    h = raw.scene["cabinet_frame"].data.target_pos_w[:, 0]
    return (ee - h).norm(dim=-1)


def stack_heights(raw):
    """z of cube_2 over cube_1 and cube_3 over cube_2, and xy offsets."""
    c = [raw.scene[f"cube_{i}"].data.root_pos_w for i in (1, 2, 3)]
    return c


def stack_success(raw):
    """Isaac Lab's cubes_stacked criterion (xy < 4 cm, height 4.68 cm +- 5 mm) per pair; success also needs the
    gripper released (checked by the caller through the final state, as the env's own termination does)."""
    c1, c2, c3 = stack_heights(raw)
    ok12 = ((c2[:, :2] - c1[:, :2]).norm(dim=-1) < 0.04) & ((c2[:, 2] - c1[:, 2] - 0.0468).abs() < 0.005)
    ok23 = ((c3[:, :2] - c2[:, :2]).norm(dim=-1) < 0.04) & ((c3[:, 2] - c2[:, 2] - 0.0468).abs() < 0.005) & ok12
    return ok12, ok23


@torch.no_grad()
def run_episodes(env, raw, arm, actor, name, obs_horizon=2, record=False, noise_std=0.0, video_cam=None, generator=None):
    """Same contract as fd_core.rollout.run_batch: dict(success, outcome, progress[, image, proprio, action, frames])."""
    spec = TASKS[name]
    obs, _ = env.reset()
    obs = obs["policy"]
    n = raw.num_envs
    actor.reset(n)
    img = arm.image().permute(0, 3, 1, 2); prop = arm.proprio()
    image_hist = img[:, None].repeat(1, obs_horizon, 1, 1, 1); proprio_hist = prop[:, None].repeat(1, obs_horizon, 1, 1)
    rec = {"image": [], "proprio": [], "action": []} if record else None
    frames = [] if video_cam else None
    best = torch.zeros(n, device=raw.device); near = torch.zeros(n, dtype=torch.bool, device=raw.device)
    for t in range(spec.steps):
        action = actor(t, obs, image_hist, proprio_hist)
        action, label = action if isinstance(action, tuple) else (action, action)
        if record:
            rec["image"].append(img.permute(0, 2, 3, 1).cpu()); rec["proprio"].append(prop.cpu()); rec["action"].append(label.cpu())
        if video_cam:
            frames.append(arm.image(video_cam).cpu())
        ex = action.clone()
        if noise_std > 0:
            ex[:, :7] = (ex[:, :7] + noise_std * torch.randn(n, 7, device=raw.device, generator=generator)).clamp(-1, 1)
        if t == spec.steps - 1:                              # state after the last action, before the time-out reset
            final = _final(raw, name)
        if name == "drawer":
            best = torch.maximum(best, drawer_pos(raw)); near |= handle_dist(raw) < 0.03
        obs, _, _, _, _ = env.step(arm.norm_to_env(ex))
        obs = obs["policy"]
        img = arm.image().permute(0, 3, 1, 2); prop = arm.proprio()
        image_hist = torch.cat([image_hist[:, 1:], img[:, None]], 1); proprio_hist = torch.cat([proprio_hist[:, 1:], prop[:, None]], 1)
    success, outcome, progress = final
    if name == "drawer":                                     # 1 never at the handle / 2 opened < 30 cm / 3 was open, closed again
        outcome[~success & ~near] = 1
        outcome[~success & near & (best < DRAWER_OPEN)] = 2
        outcome[~success & (best >= DRAWER_OPEN)] = 3
    out = {"success": success.cpu(), "outcome": outcome.cpu(), "progress": progress.cpu()}
    if record:
        out.update({k: torch.stack(v, 1) for k, v in rec.items()})
    if video_cam:
        out["frames"] = torch.stack(frames, 1)
    return out


def _final(raw, name):
    n = raw.num_envs
    outcome = torch.zeros(n, dtype=torch.long, device=raw.device)
    if name == "drawer":
        p = drawer_pos(raw).clone()
        return p >= DRAWER_OPEN, outcome, p
    ok12, ok23 = stack_success(raw)
    released = raw.scene["robot"].data.joint_pos[:, 7:9].mean(-1) > 0.03
    s = ok23 & released
    outcome[~s] = torch.where(ok12[~s], 2, 1)
    return s, outcome, ok12.float() + ok23.float()
