"""One synchronized batch of lift episodes (all envs reset together, fixed 250-step horizon)."""
from __future__ import annotations

import torch

from fd_core.env import EPISODE_STEPS, SUCCESS_DIST


class TeacherActor:
    """Privileged PPO teacher; returns normalized absolute commands [N,8].

    own_last_action: the teacher's observation ends with the last action (8 dims). When another policy drives
    (DAgger), that slot would hold the student's action, which makes the teacher chatter (13 gripper flips per
    episode instead of 1); with this flag the slot gets the teacher's own previous action instead.
    """

    def __init__(self, teacher, franka, own_last_action: bool = False):
        self.teacher, self.franka, self.own = teacher, franka, own_last_action
        self.prev = None

    def reset(self, n):
        self.prev = None

    def __call__(self, t, obs, image_hist, proprio_hist):
        if self.own:
            obs = obs.clone()
            obs[:, -8:] = 0.0 if self.prev is None else self.prev
        raw = self.teacher(obs)
        self.prev = raw
        return self.franka.env_to_norm(raw)


class DaggerActor:
    """Executes the student's chunked action (or the teacher's, for a beta fraction of envs) and labels every
    visited state with the privileged teacher's action: (executed, label)."""

    def __init__(self, student, teacher, beta: float = 0.0):
        self.student, self.teacher, self.beta = student, teacher, beta
        self.use_teacher = None

    def reset(self, n):
        self.student.reset(n)
        self.teacher.reset(n)
        self.use_teacher = None

    def __call__(self, t, obs, image_hist, proprio_hist):
        s = self.student(t, obs, image_hist, proprio_hist)
        label = self.teacher(t, obs, image_hist, proprio_hist)
        if self.use_teacher is None:
            self.use_teacher = torch.rand(s.shape[0], device=s.device) < self.beta
        return torch.where(self.use_teacher[:, None], label, s), label


class PerturbActor:
    """Injects failure states into any actor's rollout, per env:
    kind 1 = empty close: gripper forced closed for `dur` steps during the approach (before the grasp),
    kind 2 = drop: gripper forced open for `dur` steps after the lift (the cube falls),
    kind 3 = knock: the cube is moved 4-7 cm on the table at t0 (before the grasp).
    Returns (executed, label): executed has the override, label is the wrapped actor's own action,
    so recorded data teaches what the actor does from the perturbed state."""

    KINDS = ("none", "empty_close", "drop", "knock",
             # held-out kinds (never injected into training data; only for testing recovery generalisation):
             "push_held",   # 4: cube hit with a 0.8-1.2 m/s sideways velocity while held/being lifted
             "arm_shove",   # 5: arm joint targets offset by 0.25 (normalized) for `dur` steps mid-lift
             "late_knock")  # 6: cube moved 4-7 cm at t0 later than any training knock (during the grasp)

    def __init__(self, actor, franka, kinds, t0, dur, generator=None):
        self.actor, self.franka, self.g = actor, franka, generator
        self.kinds, self.t0, self.dur = kinds, t0, dur            # per-env tensors [N]

    def reset(self, n):
        self.actor.reset(n)

    def __call__(self, t, obs, image_hist, proprio_hist):
        out = self.actor(t, obs, image_hist, proprio_hist)
        a, label = out if isinstance(out, tuple) else (out, out)
        knock = ((self.kinds == 3) | (self.kinds == 6)) & (self.t0 == t)
        push = (self.kinds == 4) & (self.t0 == t)
        if push.any():
            obj = self.franka.raw.scene["object"]
            ids = push.nonzero().flatten()
            vel = obj.data.root_vel_w[ids].clone()
            ang = torch.rand(len(ids), device=vel.device, generator=self.g) * 6.2832
            sp = 0.8 + 0.4 * torch.rand(len(ids), device=vel.device, generator=self.g)
            vel[:, 0] += sp * torch.cos(ang); vel[:, 1] += sp * torch.sin(ang)
            obj.write_root_velocity_to_sim(vel, env_ids=ids)
        if knock.any():
            obj = self.franka.raw.scene["object"]
            st = obj.data.root_state_w.clone()
            ids = knock.nonzero().flatten()
            ang = torch.rand(len(ids), device=st.device, generator=self.g) * 6.2832
            r = 0.04 + 0.03 * torch.rand(len(ids), device=st.device, generator=self.g)
            st[ids, 0] += r * torch.cos(ang); st[ids, 1] += r * torch.sin(ang); st[ids, 7:] = 0.0
            obj.write_root_state_to_sim(st[ids], env_ids=ids)
        win = (t >= self.t0) & (t < self.t0 + self.dur)
        ex = a.clone()
        ex[win & (self.kinds == 1), 7] = -1.0                      # force close
        ex[win & (self.kinds == 2), 7] = 1.0                       # force open
        shove = win & (self.kinds == 5)
        if shove.any():
            if not hasattr(self, "_dir") or self._dir.shape[0] != ex.shape[0]:
                self._dir = torch.sign(torch.rand(ex.shape[0], 7, device=ex.device, generator=self.g) - 0.5)
            ex[shove, :7] = (ex[shove, :7] + 0.25 * self._dir[shove]).clamp(-1, 1)
        return ex, label


class DiffusionActor:
    """Chunked Diffusion Policy: replans every exec_horizon steps from the last obs_horizon frames."""

    def __init__(self, model, exec_horizon: int, snap_grip: bool = True):
        self.model, self.exec, self.snap = model, exec_horizon, snap_grip
        self.plan = None
        self.grip_stats = []        # per replan: mean |grip|, fraction closed, fraction near the 0 threshold

    def reset(self, n):
        self.plan = None

    def __call__(self, t, obs, image_hist, proprio_hist):
        k = t % self.exec
        if k == 0:
            self.plan = self.model.sample(image_hist.float() / 255.0, proprio_hist)
            g = self.plan[..., 7]
            self.grip_stats.append((float(g.abs().mean()), float((g < 0).float().mean()), float((g.abs() < 0.2).float().mean())))
            if self.snap:
                self.plan[..., 7] = torch.where(self.plan[..., 7] >= 0, 1.0, -1.0)
        return self.plan[:, k]


@torch.no_grad()
def run_batch(env, raw, franka, actor, obs_horizon: int = 2, record: bool = False, noise_std: float = 0.0,
              video_cam: str | None = None, generator=None, on_step=None):
    """Returns success [N] bool, final goal distance [N] and, if record, the clean per-step data.

    noise_std (DART): the executed arm command gets Gaussian noise in normalized units while the
    recorded command stays the actor's clean one.
    """
    obs, _ = env.reset()
    obs = obs["policy"]
    n = raw.num_envs
    actor.reset(n)
    img = franka.image().permute(0, 3, 1, 2)
    prop = franka.proprio()
    image_hist = img[:, None].repeat(1, obs_horizon, 1, 1, 1)
    proprio_hist = prop[:, None].repeat(1, obs_horizon, 1, 1)
    alive = torch.ones(n, dtype=torch.bool, device=raw.device)
    rec = {"image": [], "proprio": [], "action": [], "cube": []} if record else None
    frames = [] if video_cam else None
    dist = None
    z0 = franka.cube_pos()[:, 2].clone(); zmax = z0.clone()
    for t in range(EPISODE_STEPS):
        if on_step is not None:
            on_step(t, image_hist, proprio_hist)
        action = actor(t, obs, image_hist, proprio_hist)
        action, label = action if isinstance(action, tuple) else (action, action)
        if record:
            rec["image"].append(img.permute(0, 2, 3, 1).cpu())
            rec["proprio"].append(prop.cpu())
            rec["cube"].append(franka.cube_pos().cpu())
            rec["action"].append(label.cpu())          # the teacher label under DAgger, else what was executed
        if video_cam:
            frames.append(franka.image(video_cam).cpu())
        executed = action.clone()
        if noise_std > 0:
            executed[:, :7] = (executed[:, :7] + noise_std * torch.randn(n, 7, device=raw.device, generator=generator)).clamp(-1, 1)
        if t == EPISODE_STEPS - 1:
            dist = franka.goal_distance()          # state after 249 actions, before the time-out reset
            zfin = franka.cube_pos()[:, 2].clone()
        else:
            zmax = torch.maximum(zmax, franka.cube_pos()[:, 2])
        obs, _, terminated, _, _ = env.step(franka.norm_to_env(executed))
        obs = obs["policy"]
        alive &= ~terminated.bool()                # cube dropped off the table -> failure
        img = franka.image().permute(0, 3, 1, 2)
        prop = franka.proprio()
        image_hist = torch.cat([image_hist[:, 1:], img[:, None]], dim=1)
        proprio_hist = torch.cat([proprio_hist[:, 1:], prop[:, None]], dim=1)
    success = alive & (dist < SUCCESS_DIST)
    # failure taxonomy: never lifted / lifted then dropped / held in the air but off the goal / fell off the table
    lifted = zmax > z0 + 0.04
    outcome = torch.full((n,), 3, dtype=torch.long, device=raw.device)          # 3 = held but off goal
    outcome[lifted & (zfin < z0 + 0.02)] = 2                                     # 2 = dropped
    outcome[~lifted] = 1                                                         # 1 = never lifted
    outcome[~alive] = 4                                                          # 4 = fell off the table
    outcome[success] = 0
    out = {"success": success.cpu(), "dist": dist.cpu(), "outcome": outcome.cpu()}
    if record:
        out.update({k: torch.stack(v, dim=1) for k, v in rec.items()})   # [N,T,...]
    if video_cam:
        out["frames"] = torch.stack(frames, dim=1)
    return out
