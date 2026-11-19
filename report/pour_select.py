"""Episode facts and the single-replay call rule of the pour video.
Rule (same for both models, all attempts): the replayed policy call is the one that contains the START OF THE POUR TILT: the first env step at which the bottle is tilted
by more than 45 degrees from upright (z-component of the bottle's up axis < cos 45 deg), taken from the recorded episode; calls start every 16 steps, so call = step // 16.
It is the call in which the arm swings the bottle mouth over the cup and starts to tilt it; it is a rule on the recorded episode (not on any sampled chunk)."""
import numpy as np
HZ = 60          # control steps per second (decimation 2 at 120 Hz physics)
AFTER = 60       # 1 s after success

def ep_info(E, e):
    sg = E["subgoal"][e]; succ = E["succ"][e]
    ts = int(E["t_succ"][e]); te = int(E["t_end"][e])
    s5 = np.where((sg >= 5).all(1))[0]; s4 = np.where((sg >= 4).all(1))[0]
    t5 = int(s5[0]) if len(s5) else None; t4 = int(s4[0]) if len(s4) else None
    n = len(sg)
    cut = min(n - 1, (ts + AFTER) if ts >= 0 else n - 1)
    return {"success": ts >= 0, "t_succ": ts if ts >= 0 else None, "t_stage4": t4, "t_pour_done": t5, "cut": int(cut), "last_call": int(cut // 16)}


def tilt_call(E, e):
    q = E["bottle"][e, :, 3:7]; z = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    ts = np.where(z < np.cos(np.pi / 4))[0]
    return None if len(ts) == 0 else int(ts[0]) // 16
