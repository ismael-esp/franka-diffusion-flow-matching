"""Episode facts and the single-replay call rule for the task videos (drawer, stack). Rule: among the policy calls that start at or before the first
open -> closed gripper command, take the one whose simulator-predicted 16-step chunk end is closest to the relevant marker (drawer: handle, stack: cube 2)."""
import numpy as np
RATE = {"drawer": 30, "stack": 20}      # control Hz: the success hold is 1 s

def ep_info(d, e, task):
    succ = d["succ"][e][:, 0] > 0.5; n = len(succ)
    ts = None
    for t in range(n):
        if succ[t] and succ[t:min(n, t + 10)].all(): ts = t; break
    g = d["executed"][e][:, 7]; closes = np.where((g[:-1] > 0) & (g[1:] < 0))[0]
    tc = int(closes[0] + 1) if len(closes) else None
    ended = ts is not None
    cut = min(n - 1, (ts if ended else n - 1 - RATE[task]) + RATE[task])
    return {"success": bool(ended), "t_succ": ts, "t_close": tc, "cut": int(cut), "last_call": int(cut // 8)}

def marker_pos3(d, e, task):
    """per-step 3D position (env frame) of the marker the chunk should approach before grasping"""
    return d["handle"][e] if task == "drawer" else d["cubes"][e][:, 1, :3]
