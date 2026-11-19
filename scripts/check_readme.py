#!/usr/bin/env python3
"""Consistency check for README.md (no Isaac Sim needed; run from anywhere with any python 3).

1. Every repo-relative path written in `code spans` or markdown links must exist, unless it is a generated output
   (runs/..., logs/..., or listed in GENERATED below).
2. Every `--flag` used in a README command that calls one of the repository's scripts must be defined by that script
   (found by reading its add_argument calls) or be an Isaac Lab AppLauncher flag.
4. README and other published text must not contain an absolute home path, 'placeholder', 'TODO', 'owner' or 'to be decided'.
3. If the Isaac Lab python is the one running this script (`import isaaclab` works), every script command is also fed to
   that script's real `parse_args()` (no simulator is launched), so argparse itself validates flags, types and required arguments.
Exit code 1 if anything is wrong.  Usage: python scripts/check_readme.py [README.md]"""
from __future__ import annotations

import importlib
import os
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "README.md"
GENERATED = ("runs/", "logs/", "data/", "outputs/", "demos/", "$")     # created by the scripts / by the reader
SKIP_NAMES = {"README.md", "LICENSE", "meta.json", "export_build/", "media/", "video/", "release/pouring/", "command.json", ".mp4"}
SEARCH = ["", "report", "report/tools", "pouring_diffusion", "pouring_diffusion/dp_core", "fd_core", "scripts"]   # bare names in tables
APPLAUNCHER = {"--headless", "--device", "--enable_cameras", "--livestream", "--verbose", "--info", "--experience",
               "--rendering_mode", "--kit_args", "--cpu", "--xr", "--anim_recording_enabled", "--anim_recording_start_time",
               "--anim_recording_stop_time", "--visualizer", "--help"}
EXTERNAL = ("scripts/tutorials/",)           # scripts of other repositories (Isaac Lab) used in commands
text = README.read_text()
problems: list[str] = []

# ---- 1. paths
cands = set(re.findall(r"`([^`\s]+)`", text)) | set(re.findall(r"\]\(([^)#\s]+)\)", text))
checked = 0
for c in sorted(cands):
    c = c.rstrip(",.;:")
    if c.startswith(("http", "--", "-")) or c.startswith(GENERATED) or "*" in c or "<" in c or "{" in c or c in SKIP_NAMES:
        continue
    if not re.search(r"\.(py|sh|md|json|png|svg|jpg|mp4|txt|html|css|csv|tsx|ts|pptx)$|/$", c):
        continue
    checked += 1
    if not any((ROOT / d / c).exists() for d in SEARCH):
        problems.append(f"missing path: {c}")

# ---- 2. flags per script
def script_flags(py: Path) -> set[str]:
    return set(re.findall(r"""add_argument\(\s*["'](--[\w-]+)""", py.read_text())) | \
           set(re.findall(r"""add_argument\([^)]*?["'](--[\w-]+)["']""", py.read_text()))

try:
    import isaaclab  # noqa: F401
    REAL_PARSE = True
except Exception:
    REAL_PARSE = False


def real_parse(script: Path, rest: str) -> str | None:
    """Run the script's parse_args() on the command's arguments; returns an error text or None."""
    src = script.read_text()
    if "def parse_args" not in src or "__main__" not in src:      # module-level scripts would launch the simulator on import
        return None
    args = [("x" if a.startswith("$") else a) for a in shlex.split(rest.split("#")[0])]
    old_argv, old_cwd, old_path = sys.argv, os.getcwd(), list(sys.path)
    sys.argv = [script.name] + args
    sys.path.insert(0, str(script.parent)); os.chdir(script.parent)
    try:
        mod = importlib.import_module(script.stem)
        if not hasattr(mod, "parse_args"):
            return None
        mod.parse_args()
    except SystemExit as e:
        return f"argparse exit {e.code}"
    except Exception as e:                       # import problem: report, do not hide
        return f"{type(e).__name__}: {e}"
    finally:
        sys.argv = old_argv; os.chdir(old_cwd); sys.path[:] = old_path; sys.modules.pop(script.stem, None)
    return None


blocks = re.findall(r"```(?:bash|sh|console)?\n(.*?)```", text, re.S)
n_cmd = 0
n_real = 0
for blk in blocks:
    blk = re.sub(r"\\\n\s*", " ", blk)                              # join continuation lines
    cwd = ""
    for line in blk.splitlines():
        cd = re.match(r"\s*cd\s+([\w./-]+)\s*$", line)
        if cd and (ROOT / cd.group(1)).is_dir():
            cwd = cd.group(1); continue
        if cd:                                               # cd into a directory outside the repo (e.g. IsaacLab): not ours
            cwd = None; continue
        m = re.search(r"(?:python3?|\$P|\$PY|\./isaaclab\.sh -p)\s+(?:-B\s+)?(?:-u\s+)?([\w./-]+\.py)\b(.*)", line)
        if not m:
            continue
        script, rest = m.group(1), m.group(2)
        if cwd is None or script == "scripts/check_readme.py":
            continue
        script = (cwd + "/" + script) if cwd else script
        path = ROOT / script
        if not path.exists():
            if not script.startswith(("scripts/reinforcement_learning", "$") + EXTERNAL):
                problems.append(f"command uses missing script: {script}")
            continue
        n_cmd += 1
        if REAL_PARSE and "<" not in rest and "..." not in rest and "|" not in rest:
            err = real_parse(path, rest)
            if err:
                problems.append(f"{script}: real parse_args failed ({err}) for: {line.strip()[:110]}")
            else:
                n_real += 1
        WRAP = {"eval_fm.py": "eval_policy.py", "train_fm.py": "train_bc.py"}     # thin wrappers around another script's argparse
        known = script_flags(path.with_name(WRAP.get(path.name, path.name))) | APPLAUNCHER
        for fl in re.findall(r"(?<![\w-])(--[\w-]+)", rest.split("#")[0]):
            if fl not in known:
                problems.append(f"{script}: unknown flag {fl}   in: {line.strip()[:100]}")
# ---- 4. wording rules for published text
FORBIDDEN = ["/home/" + "isma", "placeholder", "TODO", "owner", "to be decided"]
PUBLISHED = [README, ROOT / "THIRD_PARTY_NOTICES.md", ROOT / "report/SCRIPTS.md", ROOT / "report/RESULTS.md", ROOT / "report/DESIGN.md", ROOT / "scripts/run_repro_minimal.sh"]
for f in PUBLISHED:
    if f.exists():
        t = f.read_text()
        for w in FORBIDDEN:
            if re.search(re.escape(w), t, 0 if w == "TODO" else re.I):
                problems.append(f"{f.relative_to(ROOT)}: contains forbidden text '{w}'")

print(f"{README.name}: {checked} paths checked, {n_cmd} script commands checked by flag name, "
      f"{n_real} by the script's real parse_args ({'enabled' if REAL_PARSE else 'skipped: run with the isaaclab python'})")
for p in problems:
    print("PROBLEM", p)
sys.exit(1 if problems else 0)
