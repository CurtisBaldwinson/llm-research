#!/usr/bin/env bash
# trace_preflight local entry point.
#
# NOTE ON THE POD: the GPU dispatcher's remote wrapper execs `python3 -u run.py`
# DIRECTLY (it prefers run.py over run.sh). On the pod this script is NOT the
# entry point -- run.py is. It exists to (a) satisfy push_code()'s existence
# check for benchmarks/<bench>/run.sh, and (b) provide a one-command LOCAL
# smoke that mirrors the pod environment.
#
# trace_preflight runs the REAL Batch 2 equivalence proof (from the sibling
# trace benchmark) and writes _batch2_preflight.json into the results dir; the
# dispatcher pulls it home automatically. --smoke is a plumbing check only
# (real model on CPU, smaller probes) and is NEVER scored-run evidence.
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
