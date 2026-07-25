#!/usr/bin/env python3
"""Thin wrapper around run.calibrate_signal (Design Note 02).

Do not add a second model-loading path here. All calibration logic lives in
``run.py`` so cloud dispatch and local runs share one implementation.

USAGE
    python calibrate_signal.py --out results/calib.json \\
        --taus 0.1 0.2 0.3 --ema-windows 1 15 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, ROOT / "sm_src", ROOT):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run as trace_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--taus", type=float, nargs="+", default=None)
    ap.add_argument("--ema-windows", type=int, nargs="+", default=None)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--calib-tasks", type=int, nargs="+", default=None)
    ap.add_argument("--model_id", type=str,
                    default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--data_root", type=Path, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    rec = trace_run.calibrate_signal(
        taus=args.taus,
        ema_windows=args.ema_windows,
        alpha=args.alpha,
        steps=args.steps,
        calib_tasks=args.calib_tasks,
        model_id=args.model_id,
        data_root=args.data_root,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2) + "\n")
    summary = trace_run._calib_summary(rec)
    (args.out.parent / "results_summary.txt").write_text(summary + "\n")
    print(summary)
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
