#!/usr/bin/env python3
"""data/overlay2_single.json (lift, built by make_overlay_data2.py single) -> data/task_lift.json in the common task-video schema; copies the lift clips."""
import json, shutil
from pathlib import Path
R = Path(__file__).resolve().parent
J = json.load(open(R / "data/overlay2_single.json"))
out = {"task": "lift", "hz": 50, "alphas_cumprod": J["alphas_cumprod"], "grid": J["grid"], "models": {}, "metrics": [], "marker_checks": []}
for name, mv in J["models"].items():
    atts = []
    for a in mv["attempts"]:
        rp = a["replays"][0]; m = rp["metrics"]
        cut = a["cut"]
        atts.append({"env": a["env"], "clip": f"e{a['env']}", "success": True, "t_succ": a["t_succ"], "t_close": a["t_close"], "cut": cut, "last_call": a["last_call"], "rate": 1, "actual_px": a["actual_px"], "finals": a["finals"],
                     "markers": [{"label": "cube", "kind": "square", "t0": 0, "t1": cut, "px": a["cube_px"]}, {"label": "goal", "kind": "cross", "t0": 0, "t1": cut, "px": a["goal_px"]}],
                     "replay": {"call": rp["call"], "step": rp["step"], "iter": rp["iter"], **({"vel": rp["vel"], "field": rp["field"]} if "vel" in rp else {}),
                                "metrics": {"end16_to_marker_m": m["end16_to_cube_m"], "end16_to_target_m": m["end16_to_goal_m"], "now_to_marker_m": m["tcp_to_cube_now_m"],
                                            "close_index": m["close_index"], "close_to_marker_m": m["close_to_cube_m"], "spread_mm": m["samples_end16_std_mm"]}}})
        shutil.copy(R / f"video/public/denoise_{name}_e{a['env']}.mp4", R / f"video/public/task_lift_{name}_e{a['env']}.mp4")
    out["models"][name] = {"attempts": atts}
(R / "data/task_lift.json").write_text(json.dumps(out, separators=(",", ":")))
print("ok")
