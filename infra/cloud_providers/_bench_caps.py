"""Per-benchmark resource declarations and matching against canonical GPUs.

A benchmark directory may contain ``bench_caps.json`` declaring the hardware
envelope it needs.  The orchestrator reads this before dispatching and filters
GPU candidates that don't meet the requirements, regardless of cloud provider.

Schema (all fields optional; absent → no constraint):

    {
      "min_vram_gb":               <int>,
      "required_compute_capability": "sm_80" | "sm_86" | ...,
      "preferred_gpus":            ["A40", "A6000", ...],
      "notes":                     "free-form prose"
    }

The orchestrator interprets `required_compute_capability` as a minimum:
``sm_80`` allows sm_80, sm_86, sm_89, sm_90, ... (any later generation).  Use
this for "the benchmark uses features that only Ampere+ supports", not for
pinning to a single generation.

`preferred_gpus` is informational only — if set, the orchestrator emits a
short note when an operator's `--gpu-type` candidate list doesn't overlap.

The schema is intentionally minimal.  Resist the urge to add CPU / RAM / disk
fields here; they belong on the orchestrator's CLI as `--container-disk-gb`
etc., not on the per-benchmark declaration.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class BenchCaps:
    """Parsed contents of a ``bench_caps.json``."""
    min_vram_gb: Optional[int] = None
    required_compute_capability: Optional[str] = None
    preferred_gpus: Optional[List[str]] = None
    notes: Optional[str] = None
    # Opt-in flag for the pod-side encrypt-upload-self-terminate stack
    # (see docs/2026-05-25_remote-finalize-architecture.md).  When True,
    # the dispatcher pushes the finalize payload, wires the wrapper to
    # invoke pod_finalize.py after a successful benchmark exit, and
    # accepts that the pod will self-destruct without waiting for the
    # dispatcher's destroy call.  Defaults to False (legacy SCP-pull
    # path; pod waits for dispatcher).
    remote_finalize: bool = False


_EMPTY = BenchCaps()


def load_bench_caps(benchmark_dir: Path) -> BenchCaps:
    """Read ``<benchmark_dir>/bench_caps.json`` if present.  Returns an empty
    :class:`BenchCaps` if the file does not exist."""
    path = benchmark_dir / "bench_caps.json"
    if not path.exists():
        return _EMPTY
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}: invalid JSON ({exc})"
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level must be a JSON object")
    return BenchCaps(
        min_vram_gb=raw.get("min_vram_gb"),
        required_compute_capability=raw.get("required_compute_capability"),
        preferred_gpus=raw.get("preferred_gpus"),
        notes=raw.get("notes"),
        remote_finalize=bool(raw.get("remote_finalize", False)),
    )


# Compute-capability ordering for ">=" comparisons.  Larger numbers are
# later generations; sm_120 (Blackwell) > sm_90 (Hopper) > sm_89 (Ada) > etc.
def _sm_rank(sm: str) -> int:
    """Return an integer rank for ``sm_XY`` strings such that newer generations
    sort higher.  Raises ``ValueError`` on unrecognised strings."""
    m = re.fullmatch(r"sm_(\d+)", sm)
    if not m:
        raise ValueError(f"not a valid compute capability: {sm!r}")
    return int(m.group(1))


def gpu_meets_caps(gpu_spec, caps: BenchCaps) -> bool:
    """Return True iff a GPU (from :data:`cloud_providers.GPU_CANONICAL`)
    meets the benchmark's declared requirements.

    A ``None`` field in ``caps`` is treated as "no constraint".
    """
    if caps.min_vram_gb is not None and gpu_spec.vram_gb < caps.min_vram_gb:
        return False
    if caps.required_compute_capability is not None:
        try:
            need = _sm_rank(caps.required_compute_capability)
            have = _sm_rank(gpu_spec.compute_capability)
        except ValueError:
            # Unknown SM string somewhere; fail open so we don't accidentally
            # filter out everything because of a typo in bench_caps.json.
            return True
        if have < need:
            return False
    return True
