"""Which episodes and policy calls the overlay scenes show (rules, not hand-picking): attempt 1 = env 4, attempt 2 = env 6 (both models succeed on both,
cube starts differ). Episode is cut HOLD steps after the first step the lift success condition (cube within 5 cm of the goal) holds."""
import numpy as np
ENVS = (4, 6); HOLD = 50

def episode(d, e):
    gd = d["goal_dist"][e]; ts = int(np.where(gd < 0.05)[0][0]); g = d["executed"][e][:, 7]
    tc = int(np.where(g < 0)[0][-0 if False else 0]) if False else int(np.where((g[:-1] > 0) & (g[1:] < 0))[0][0] + 1)   # first open -> closed command
    cut = min(249, ts + HOLD)
    last = (ts - 1) // 8                         # last call that starts before success
    B = tc // 8 + 0                              # call containing the closing command
    calls = [0, B, min(last, B + 3)]
    if last - calls[-1] >= 2: calls.append(last)
    calls = sorted(set(c for c in calls if c <= last))
    kinds = {}
    for c in calls:
        kinds[c] = "approach" if 8 * c + 16 <= tc - 4 or c == 0 and tc >= 16 else ("grasp" if 8 * c <= tc < 8 * c + 8 else "lift")
    return {"t_succ": ts, "t_close": tc, "cut": cut, "last_call": cut // 8, "replays": calls, "kinds": kinds}
