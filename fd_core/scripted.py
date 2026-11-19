"""Scripted privileged teacher for cube stacking (fd_core.tasks 'stack').

Per env, a state machine runs two pick-and-place subtasks (cube_2 onto cube_1, then cube_3 onto cube_2):
above-pick -> descend -> close -> lift -> above-place -> lower -> open -> retreat. Targets are recomputed every step
from the true cube poses (so a slipped or knocked cube is re-targeted), the fingertip point moves at most `vmax`
per step toward its target, the hand stays pointing down with its yaw aligned to the cube (nearest of the 4
symmetric yaws), and a damped-least-squares differential IK turns the target into joint targets. Output: the
student's action format (normalized absolute joint targets + binary gripper)."""
from __future__ import annotations

import math

import torch

TCP = 0.1034            # panda_hand -> fingertip centre
CUBE_H = 0.0468         # centre-to-centre height of two stacked cubes (Isaac Lab cubes_stacked)
ABOVE, DESCEND, CLOSE, LIFT, OVER, LOWER, OPEN, RETREAT, DONE = range(9)


def _yaw(q):            # wxyz -> yaw of the body x-axis
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _down_quat(yaw):    # hand pointing down (z_hand = -z_world) with x_hand at angle `yaw`: qz(yaw) * qx(pi)
    h = yaw / 2
    return torch.stack([torch.zeros_like(h), torch.cos(h), torch.sin(h), torch.zeros_like(h)], -1)


class StackScript:
    def __init__(self, raw, arm, vmax: float = 0.04, close_steps: int = 7, resync: bool = False):
        """resync: re-derive the subtask/phase from the true state every step (for DAgger labels, where the student
        drives and the state machine's own bookkeeping can drift from what actually happened)."""
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        self.raw, self.arm, self.vmax, self.close_steps, self.resync = raw, arm, vmax, close_steps, resync
        self.robot = raw.scene["robot"]
        self.hand = self.robot.find_bodies("panda_hand")[0][0]
        self.jac_idx = self.hand - 1                     # fixed base: the root has no Jacobian row
        n, dev = raw.num_envs, raw.device
        self.ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), n, dev)
        self.pairs = [("cube_2", "cube_1"), ("cube_3", "cube_2")]
        self.reset(n)

    def reset(self, n):
        dev = self.raw.device
        self.sub = torch.zeros(n, dtype=torch.long, device=dev)       # which pick/place pair
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.timer = torch.zeros(n, dtype=torch.long, device=dev)
        self.grip_yaw = None

    # ------------------------------------------------------------------ helpers (all in the robot base frame)
    def _local(self, pos_w, quat_w=None):
        from isaaclab.utils.math import subtract_frame_transforms
        d = self.robot.data
        return subtract_frame_transforms(d.root_pos_w, d.root_quat_w, pos_w, quat_w)

    def _cube(self, name):
        o = self.raw.scene[name].data
        return self._local(o.root_pos_w, o.root_quat_w)

    def __call__(self, t, obs, image_hist, proprio_hist):
        from isaaclab.utils.math import matrix_from_quat
        n, dev = self.raw.num_envs, self.raw.device
        d = self.robot.data
        hand_p, hand_q = self._local(d.body_pos_w[:, self.hand], d.body_quat_w[:, self.hand])
        tcp = hand_p + (matrix_from_quat(hand_q) @ torch.tensor([0.0, 0.0, TCP], device=dev))
        picks, places = zip(*[(self._cube(a), self._cube(b)) for a, b in self.pairs])
        sel = lambda lst, i: torch.stack([lst[k][i] for k in range(len(lst))], 0)[self.sub.clamp(max=len(self.pairs) - 1), torch.arange(n, device=dev)]
        if self.resync:
            self._resync(tcp, picks, places)
        pick_p, pick_q = sel(picks, 0), sel(picks, 1)
        place_p, place_q = sel(places, 0), sel(places, 1)

        ph = self.phase
        target = tcp.clone(); grip = torch.ones(n, device=dev)
        up = torch.tensor([0.0, 0.0, 1.0], device=dev)
        m = ph == ABOVE;   target[m] = pick_p[m] + 0.10 * up
        m = ph == DESCEND; target[m] = pick_p[m] + 0.005 * up
        m = ph == CLOSE;   target[m] = pick_p[m] + 0.005 * up; grip[m] = -1
        m = ph == LIFT;    target[m] = pick_p[m] * torch.tensor([1, 1, 0.0], device=dev) + 0.14 * up; grip[m] = -1
        m = ph == OVER;    target[m] = place_p[m] + (CUBE_H + 0.07) * up; grip[m] = -1
        m = ph == LOWER;   target[m] = place_p[m] + (CUBE_H + 0.006) * up; grip[m] = -1
        m = ph == OPEN;    target[m] = place_p[m] + (CUBE_H + 0.006) * up
        m = ph == RETREAT; target[m] = place_p[m] + (CUBE_H + 0.07) * up
        m = ph == DONE;    target[m] = tcp[m]

        # hand yaw: the cube's yaw (pick phases) or the support cube's yaw (place phases), nearest symmetric copy
        cur = _yaw(hand_q)
        wrap = lambda a: a - (math.pi / 2) * torch.round(a / (math.pi / 2))      # to [-45, 45] deg (cube symmetry)
        # pick phases: align the fingers with the cube to pick; carrying: turn by the held cube's yaw offset to the
        # support cube (the held cube turns with the hand); after release: keep the current yaw
        want = torch.where(ph <= CLOSE, cur + wrap(_yaw(pick_q) - cur),
               torch.where(ph <= LOWER, cur + wrap(_yaw(place_q) - _yaw(pick_q)), cur))
        want = cur + (want - cur).clamp(-0.12, 0.12)                              # limited yaw rate per step

        # limited fingertip speed, then IK on the hand pose
        step = target - tcp
        nrm = step.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        des_tcp = tcp + step * torch.clamp(self.vmax / nrm, max=1.0)
        des_q = _down_quat(want)
        des_hand = des_tcp - matrix_from_quat(des_q) @ torch.tensor([0.0, 0.0, TCP], device=dev)
        self.ik.set_command(torch.cat([des_hand, des_q], -1))
        jac = self.robot.root_physx_view.get_jacobians()[:, self.jac_idx, :, :7]
        q_des = self.ik.compute(hand_p, hand_q, jac, d.joint_pos[:, :7])

        # phase transitions
        err = (target - tcp).norm(dim=-1)
        self.timer += 1
        nxt = torch.zeros_like(ph, dtype=torch.bool)
        xy = (target - tcp)[:, :2].norm(dim=-1)
        nxt |= (ph == ABOVE) & (xy < 0.008) & (err < 0.03)
        nxt |= (ph == DESCEND) & ((err < 0.008) | (self.timer > 25))
        nxt |= (ph == CLOSE) & (self.timer >= self.close_steps)
        nxt |= (ph == LIFT) & (err < 0.03)
        nxt |= (ph == OVER) & (xy < 0.008) & (err < 0.03)
        nxt |= (ph == LOWER) & ((err < 0.008) | (self.timer > 25))
        nxt |= (ph == OPEN) & (self.timer >= self.close_steps)
        nxt |= (ph == RETREAT) & (err < 0.03)
        # lost the cube while carrying it: go back and pick it again
        lost = (ph >= LIFT) & (ph <= LOWER) & ((pick_p - tcp).norm(dim=-1) > 0.05)
        self.phase = torch.where(nxt, ph + 1, ph)
        self.phase = torch.where(lost, torch.full_like(ph, ABOVE), self.phase)
        fin = (self.phase == DONE) & (self.sub < len(self.pairs) - 1)
        self.sub = torch.where(fin, self.sub + 1, self.sub)
        self.phase = torch.where(fin, torch.full_like(ph, ABOVE), self.phase)
        self.timer = torch.where(nxt | lost | fin, torch.zeros_like(self.timer), self.timer)

        env_a = torch.cat([(q_des - self.arm.default) / self.arm.scale, grip[:, None]], -1)
        return self.arm.env_to_norm(env_a)

    def _resync(self, tcp, picks, places):
        """Make (sub, phase) consistent with the scene: which pair is next follows the actual stack; a cube already
        held skips to LIFT; descending/closing far from the cube goes back above it; a toppled stack restarts."""
        from fd_core.tasks import stack_success
        ok12, ok23 = stack_success(self.raw)
        fing = self.robot.data.joint_pos[:, 7:9].mean(-1)
        ph, sub = self.phase.clone(), self.sub.clone()
        held_by = lambda p: ((p - tcp).norm(dim=-1) < 0.025) & (fing > 0.012) & (fing < 0.032)
        # loose "cube_2 rests on cube_1" (a just-placed cube settles for a few steps and briefly fails the strict test)
        c1, c2 = places[0][0], picks[0][0]
        dz = c2[:, 2] - c1[:, 2]; dxy = (c2[:, :2] - c1[:, :2]).norm(dim=-1)
        on = (dz > 0.03) & (dxy < 0.04)
        # subtask from the real stack (cube_2 placed -> work on cube_3; cube_2 back on the table -> back to cube_2)
        to1 = on & (sub == 0) & ~held_by(c2) & ((ph < LOWER) | (ph == DONE))
        to0 = (dz < 0.02) & (sub == 1) & ~held_by(picks[1][0])
        sub = torch.where(to1, 1, torch.where(to0, 0, sub))
        ph = torch.where(to1 | to0, ABOVE, ph)
        ph = torch.where(ok23 & (ph < OPEN), torch.full_like(ph, RETREAT), ph)          # finished: release and leave
        pick = torch.where((sub == 0)[:, None], picks[0][0], picks[1][0])
        held = held_by(pick)
        ph = torch.where(held & (ph <= CLOSE), torch.full_like(ph, LIFT), ph)           # student already grasped
        far = ((pick - tcp)[:, :2].norm(dim=-1) > 0.03) & ((ph == DESCEND) | (ph == CLOSE))
        ph = torch.where(far, torch.full_like(ph, ABOVE), ph)
        self.timer = torch.where((ph != self.phase) | (sub != self.sub), torch.zeros_like(self.timer), self.timer)
        self.phase, self.sub = ph, sub


PRE, REACH, GRASP, PULL, HOLD = range(5)


class DrawerScript(StackScript):
    """Scripted privileged teacher for drawer opening (fd_core.tasks 'drawer').

    The hand takes the handle frame's orientation (hand z = handle z = into the cabinet, fingers closing across the
    bar), goes to a pre-grasp point 8 cm in front of the handle, moves straight in, closes, then pulls along the
    handle's -z (which moves with the drawer) until the drawer is ~0.37 m open, and holds. Losing the handle while
    pulling restarts from the pre-grasp point. Smooth, rate-limited motion (unlike the PPO teacher's bang-bang)."""

    def __init__(self, raw, arm, vmax: float = 0.03, close_steps: int = 6, open_to: float = 0.36, roll: float = 0.0,
                 posture: float = 0.0, home_after: int = 0, translate_first: float = 0.0):
        """posture > 0: null-space gain toward the home joint posture. home_after > 0: if the pre-grasp point is not
        reached within that many steps, drive the joints back to the home posture and retry (v2 teacher; both 0 =
        the teacher behind runs/drawer_s)."""
        super().__init__(raw, arm, vmax=vmax, close_steps=close_steps)
        self.open_to, self.roll, self.posture, self.home_after = open_to, roll, posture, home_after
        self.translate_first = translate_first
        self.homing = None

    def reset(self, n):
        dev = self.raw.device
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.timer = torch.zeros(n, dtype=torch.long, device=dev)
        self.cmd = None

    def __call__(self, t, obs, image_hist, proprio_hist):
        from isaaclab.utils.math import matrix_from_quat
        from fd_core.tasks import drawer_pos
        n, dev = self.raw.num_envs, self.raw.device
        d = self.robot.data
        cf = self.raw.scene["cabinet_frame"].data
        hand_p, hand_q = self._local(d.body_pos_w[:, self.hand], d.body_quat_w[:, self.hand])
        z = torch.tensor([0.0, 0.0, 1.0], device=dev)
        tcp = hand_p + matrix_from_quat(hand_q) @ (TCP * z)
        h_p, h_q = self._local(cf.target_pos_w[:, 0], cf.target_quat_w[:, 0])
        h_z = matrix_from_quat(h_q) @ z                          # into the cabinet
        if self.roll:                                            # grasp orientation: handle frame rolled about its z
            from isaaclab.utils.math import quat_from_angle_axis as _qa, quat_mul as _qm
            h_q = _qm(h_q, _qa(torch.full((n,), self.roll, device=dev), z.expand(n, 3)))
        ph = self.phase
        target = tcp.clone(); grip = torch.ones(n, device=dev); vmax = torch.full((n,), self.vmax, device=dev)
        m = ph == PRE;   target[m] = h_p[m] - 0.08 * h_z[m]
        m = ph == REACH; target[m] = h_p[m] + 0.004 * h_z[m]; vmax[m] = 0.012
        m = ph == GRASP; target[m] = h_p[m] + 0.004 * h_z[m]; grip[m] = -1
        dpos = drawer_pos(self.raw)
        m = ph == PULL;  target[m] = h_p[m] - (self.open_to + 0.02 - dpos[m])[:, None] * h_z[m]; grip[m] = -1; vmax[m] = 0.012   # the fully open handle position
        m = ph == HOLD;  target[m] = h_p[m]; grip[m] = -1

        # carrot: a commanded point moving toward the target at vmax, kept within 4 cm of the real fingertips (the
        # stiff PD arm covers only ~15% of a one-step command at 30 Hz, so stepping from the real pose crawls)
        if self.cmd is None or t == 0:
            self.cmd = tcp.clone()
        step = target - self.cmd
        nrm = step.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self.cmd = self.cmd + step * torch.clamp(vmax[:, None] / nrm, max=1.0)
        lag = self.cmd - tcp
        cap = torch.where(ph == PULL, 0.12, 0.04)[:, None]                     # pulling needs a bigger spring offset
        self.cmd = tcp + lag * torch.clamp(cap / lag.norm(dim=-1, keepdim=True).clamp(min=1e-6), max=1.0)
        des_tcp = self.cmd
        # orientation: rotate toward the handle frame by at most 0.15 rad per step
        from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_from_angle_axis, quat_mul
        rv = axis_angle_from_quat(quat_mul(h_q, quat_conjugate(hand_q)))
        ang = rv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        rv = rv * torch.clamp(0.15 / ang, max=1.0)
        if self.translate_first:   # far from the pre-grasp point: hold the orientation, rotate only once close
            far = (ph == PRE) & ((target - tcp).norm(dim=-1) > self.translate_first)
            rv = torch.where(far[:, None], torch.zeros_like(rv), rv)
        des_q = quat_mul(quat_from_angle_axis(rv.norm(dim=-1), rv / rv.norm(dim=-1, keepdim=True).clamp(min=1e-9)), hand_q)
        des_hand = des_tcp - matrix_from_quat(des_q) @ (TCP * z)
        self.ik.set_command(torch.cat([des_hand, des_q], -1))
        jac = self.robot.root_physx_view.get_jacobians()[:, self.jac_idx, :, :7]
        q = d.joint_pos[:, :7]
        dq = self.ik.compute(hand_p, hand_q, jac, q) - q
        if self.posture:   # null-space pull toward the home posture: keeps the arm out of folded joint-limit configurations
            Jp = torch.linalg.pinv(jac)
            N = torch.eye(7, device=dev) - Jp @ jac
            dq = dq + (N @ (self.posture * (self.arm.default - q))[..., None]).squeeze(-1)
        jcap = torch.where(ph == PULL, 0.3, 0.12)[:, None]                     # joint offset per step (PD damping limits speed)
        q_des = q + dq * torch.clamp(jcap / dq.abs().amax(-1, keepdim=True).clamp(min=1e-6), max=1.0)
        if self.home_after:   # stuck short of the pre-grasp point (folded arm at joint limits): go home, then retry
            if self.homing is None or t == 0:
                self.homing = torch.zeros(n, dtype=torch.bool, device=dev)
            err_pre = (target - tcp).norm(dim=-1)
            self.homing |= (ph == PRE) & (self.timer > self.home_after) & (err_pre > 0.05)
            hq = q + (self.arm.default - q).clamp(-0.12, 0.12)
            q_des = torch.where(self.homing[:, None], hq, q_des)
            back = self.homing & ((self.arm.default - q).abs().amax(-1) < 0.1)
            self.homing &= ~back
            self.timer = torch.where(back, torch.zeros_like(self.timer), self.timer)
            grip = torch.where(self.homing, torch.ones_like(grip), grip)

        err = (target - tcp).norm(dim=-1)
        lateral = (target - tcp) - ((target - tcp) * h_z).sum(-1, keepdim=True) * h_z   # off the approach axis
        rot_ok = (matrix_from_quat(hand_q)[..., 2] * h_z).sum(-1) > 0.97
        self.timer += 1
        nxt = torch.zeros_like(ph, dtype=torch.bool)
        nxt |= (ph == PRE) & (err < 0.015) & rot_ok
        nxt |= (ph == REACH) & (((err < 0.006) & (lateral.norm(dim=-1) < 0.006)) | (self.timer > 30))
        nxt |= (ph == GRASP) & (self.timer >= self.close_steps)
        nxt |= (ph == PULL) & (dpos >= self.open_to)
        lost = (ph == PULL) & ((h_p - tcp).norm(dim=-1) > 0.04)
        self.phase = torch.where(nxt, (ph + 1).clamp(max=HOLD), ph)
        self.phase = torch.where(lost, torch.full_like(ph, PRE), self.phase)
        self.timer = torch.where(nxt | lost, torch.zeros_like(self.timer), self.timer)

        env_a = torch.cat([(q_des - self.arm.default) / self.arm.scale, grip[:, None]], -1)
        return self.arm.env_to_norm(env_a)
