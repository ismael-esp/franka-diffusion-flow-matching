"""Franka Panda forward kinematics (modified DH) and the pinhole projection of the video camera (fd_core/env.py make_env video_camera)."""
import numpy as np

DH_A = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]
DH_D = [0.333, 0, 0.316, 0, 0.384, 0, 0]
DH_AL = [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2]
FLANGE_D, TCP_D = 0.107, 0.1034          # flange = panda_hand frame origin; TCP = between the fingertips


def fk(q, tcp=True):
    """q [...,7] radians -> position [...,3] of the panda_hand frame (tcp=False) or the fingertip centre (tcp=True), robot base frame."""
    q = np.asarray(q, dtype=np.float64)
    T = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
    def mul(T, A): return T @ A
    for i in range(7):
        ca, sa, ct, st = np.cos(DH_AL[i]), np.sin(DH_AL[i]), np.cos(q[..., i]), np.sin(q[..., i])
        A = np.zeros(q.shape[:-1] + (4, 4))
        A[..., 0, 0] = ct; A[..., 0, 1] = -st; A[..., 0, 3] = DH_A[i]
        A[..., 1, 0] = st * ca; A[..., 1, 1] = ct * ca; A[..., 1, 2] = -sa; A[..., 1, 3] = -sa * DH_D[i]
        A[..., 2, 0] = st * sa; A[..., 2, 1] = ct * sa; A[..., 2, 2] = ca; A[..., 2, 3] = ca * DH_D[i]
        A[..., 3, 3] = 1
        T = mul(T, A)
    d = FLANGE_D + (TCP_D if tcp else 0.0)
    return T[..., :3, 3] + T[..., :3, 2] * d


CAM_EYE, CAM_TARGET = np.array([1.45, 0.75, 0.75]), np.array([0.42, 0.0, 0.18])
W, H, FOCAL, APERT = 640, 480, 18.0, 20.955
FX = FOCAL / APERT * W


def project(p, W_=W, H_=H):
    """p [...,3] in the env frame (robot base at the origin) -> pixel [...,2] of the video camera (and depth)."""
    f = CAM_TARGET - CAM_EYE; f = f / np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, f)
    d = np.asarray(p, dtype=np.float64) - CAM_EYE
    z = d @ f
    return np.stack([W_ / 2 + FX * (d @ right) / z, H_ / 2 - FX * (d @ up) / z], -1), z


def norm_to_q(a, low, high):
    return low + (np.clip(a, -1, 1) + 1) * 0.5 * (high - low)


def project_cam(p, eye, target, W_=W, H_=H):
    """Same pinhole model (focal 18 mm, aperture 20.955 mm) for a task's video camera, pixel coordinates of a 640 x 480 frame."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye; f = f / np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, f)
    d = np.asarray(p, dtype=np.float64) - eye
    z = d @ f
    return np.stack([W_ / 2 + FX * (d @ right) / z, H_ / 2 - FX * (d @ up) / z], -1), z
