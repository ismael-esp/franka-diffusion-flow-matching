"""Gravity sag of the env's joint-position PD (stiffness 80 on all 7 arm joints in IsaacLab's FRANKA_PANDA_CFG): the arm settles at
q = target - (dU/dq) / K, U = sum m_i g z_i(COM_i). Link masses / COMs: franka_description (panda_arm.urdf.xacro inertials)."""
import numpy as np
from geom import DH_A, DH_D, DH_AL, FLANGE_D
M = [4.970684, 0.646926, 3.228604, 3.587895, 1.225946, 1.666555, 0.735522, 0.76]
COM = np.array([[0.003875, 0.002081, -0.04762], [-0.003141, -0.02872, 0.003495], [0.027518, 0.039252, -0.066502], [-0.05317, 0.104419, 0.027454],
                [-0.011953, 0.041065, -0.038437], [0.060149, -0.014117, -0.010517], [0.010517, -0.004252, 0.061597], [-0.01, 0.0, 0.03]])
K = 80.0

def com_height(q):
    """q [...,7] -> U/g = sum m_i z_i [...]"""
    q = np.asarray(q, float); T = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy(); U = 0
    for i in range(8):
        if i < 7:
            ca, sa, ct, st = np.cos(DH_AL[i]), np.sin(DH_AL[i]), np.cos(q[..., i]), np.sin(q[..., i])
            A = np.zeros(q.shape[:-1] + (4, 4))
            A[..., 0, 0] = ct; A[..., 0, 1] = -st; A[..., 0, 3] = DH_A[i]
            A[..., 1, 0] = st * ca; A[..., 1, 1] = ct * ca; A[..., 1, 2] = -sa; A[..., 1, 3] = -sa * DH_D[i]
            A[..., 2, 0] = st * sa; A[..., 2, 1] = ct * sa; A[..., 2, 2] = ca; A[..., 2, 3] = ca * DH_D[i]
            A[..., 3, 3] = 1
            T = T @ A
        else:
            F = np.broadcast_to(np.eye(4), T.shape).copy(); F[..., 2, 3] = FLANGE_D; T = T @ F
        U = U + M[i] * (T[..., 2, :3] @ COM[i] + T[..., 2, 3])
    return U

def grad(q, eps=1e-4):
    q = np.asarray(q, float); g = np.zeros_like(q)
    for j in range(7):
        d = np.zeros(7); d[j] = eps
        g[..., j] = (com_height(q + d) - com_height(q - d)) / (2 * eps)
    return g * 9.81

def settle(target, iters=6, k=K):
    """steady-state joint positions reached for joint targets [...,7]"""
    q = np.asarray(target, float).copy()
    for _ in range(iters):
        q = target - grad(q) / k
    return q

def rollout_path(target, q0, alpha, iters=4):
    """Predicted joint path of the PD controller for a chunk of joint targets [...,16,7] starting at the measured q0 [...,7]:
    first-order tracking of the gravity-sagged set point. Returns the joint positions after each action [...,16,7]."""
    sp = settle(target, iters)
    q = np.broadcast_to(q0, sp[..., 0, :].shape).copy(); out = []
    for k in range(sp.shape[-2]):
        q = q + alpha * (sp[..., k, :] - q); out.append(q)
    return np.stack(out, -2)
