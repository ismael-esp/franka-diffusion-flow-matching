#!/usr/bin/env bash
# Regenerate every chart / diagram of the report that can be rebuilt from the committed files (no Isaac Sim, no GPU).
#   report/make_plots.py       -> PNG + SVG plots and CSV tables from report/data/*.json
#   report/tools/make_diagrams.py -> report/diagrams/*.svg and report/page4.html
# By default this overwrites the committed outputs in place (the SVGs only change in their embedded date).
# FD_PLOTS_OUT=/some/dir redirects make_plots.py (it writes plots/ and data/ below it); the diagrams always go to report/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
$PY report/make_plots.py
$PY report/tools/make_diagrams.py
