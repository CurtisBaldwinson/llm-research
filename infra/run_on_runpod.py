#!/usr/bin/env python3
"""Compatibility shim: legacy RunPod CLI surface, delegates to ``run_on_cloud.py``.

Before the multi-cloud refactor this file held the full RunPod dispatcher (~1,200
lines of REST/GraphQL, SSH/tar transport, staging-dir orchestration, etc.).
The dispatcher is now provider-agnostic and lives in
:mod:`scripts.run_on_cloud`; the RunPod control plane lives in
:mod:`scripts.cloud_providers.runpod`.

This file exists to preserve the historical CLI surface so the project's
Makefile targets and the operator's muscle memory continue to work:

  python3 scripts/run_on_runpod.py --benchmark <name> --gpu-type "NVIDIA A40,..."

The shim does three things:

  1. Re-exec into ``scripts/run_on_cloud.py`` with ``--provider runpod``
     prepended.
  2. Translate any RunPod-specific SKU strings in ``--gpu-type`` (e.g.,
     ``"NVIDIA A40"``) to the canonical short names the orchestrator
     expects (``"A40"``).  Already-canonical names pass through.
  3. Pass everything else through unchanged.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

# Make ``cloud_providers`` importable so we can read its SKU map.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from cloud_providers.runpod import _CANONICAL_TO_SKU as _CANONICAL_TO_RUNPOD


# Inverse of cloud_providers.runpod._CANONICAL_TO_SKU.  Built once at import.
_RUNPOD_SKU_TO_CANONICAL: Dict[str, str] = {
    sku: canonical for canonical, sku in _CANONICAL_TO_RUNPOD.items()
}


def _translate_gpu_arg(value: str) -> str:
    """Translate a comma-separated GPU list from legacy RunPod SKUs to
    canonical names.  Any token already in canonical form passes through.

    Examples:
        "NVIDIA A40"                   -> "A40"
        "NVIDIA A40,NVIDIA RTX A6000"  -> "A40,A6000"
        "A40,A6000,L40S"               -> "A40,A6000,L40S"  (unchanged)
        "NVIDIA Mystery Card"          -> "NVIDIA Mystery Card"  (unrecognised -> pass through,
                                                                   orchestrator will UnknownGpuError later)
    """
    tokens = [t.strip() for t in value.split(",") if t.strip()]
    out: List[str] = []
    for tok in tokens:
        if tok in _RUNPOD_SKU_TO_CANONICAL:
            out.append(_RUNPOD_SKU_TO_CANONICAL[tok])
        else:
            out.append(tok)
    return ",".join(out)


def main() -> None:
    argv = list(sys.argv[1:])
    new_argv: List[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        # Translate --gpu-type / -g values when given as the next token.
        if a in ("--gpu-type", "-g") and i + 1 < len(argv):
            new_argv.append(a)
            new_argv.append(_translate_gpu_arg(argv[i + 1]))
            i += 2
            continue
        # Translate --gpu-type=VALUE in equals form.
        if a.startswith("--gpu-type="):
            new_argv.append("--gpu-type=" + _translate_gpu_arg(a.split("=", 1)[1]))
            i += 1
            continue
        new_argv.append(a)
        i += 1

    # Always force --provider runpod, regardless of whether the legacy caller
    # passed --provider (which it wouldn't have, but defending anyway).
    if "--provider" not in new_argv and "-p" not in new_argv:
        new_argv = ["--provider", "runpod"] + new_argv

    run_on_cloud = _SCRIPT_DIR / "run_on_cloud.py"
    os.execv(sys.executable, [sys.executable, str(run_on_cloud)] + new_argv)


if __name__ == "__main__":
    main()
