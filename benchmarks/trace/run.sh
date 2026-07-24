#!/usr/bin/env bash
# TRACE (Experiment 1) local entry point.
#
# NOTE ON THE POD: the GPU dispatcher's remote wrapper execs `python3 -u run.py`
# DIRECTLY (it prefers run.py over run.sh). So on the pod this script is NOT the
# entry point -- run.py is. This script exists to (a) satisfy the dispatcher's
# push_code() existence check for benchmarks/<bench>/run.sh, and (b) provide a
# one-command LOCAL smoke/run entry that mirrors the pod environment (venv +
# requirements + data staging happen the same way).
#
# Data staging (fetch_data.stage_trace) is invoked from INSIDE run.py's trace
# path, so it runs identically whether launched here or by the pod wrapper.
set -euo pipefail
cd "$(dirname "$0")"

VENV="${VENV:-.venv}"
if [ ! -d "$VENV" ]; then
    echo "[run.sh] creating venv at $VENV"
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"

python3 -m pip install --upgrade pip >/dev/null
# torch is intentionally NOT pinned in requirements.txt; install a CPU build
# locally for smoke (the pod uses its CUDA-matched image torch).
python3 -c "import torch" 2>/dev/null || python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r requirements.txt

exec python3 -u run.py "$@"
