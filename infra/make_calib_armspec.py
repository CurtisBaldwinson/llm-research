#!/usr/bin/env python3
"""Generate arm-spec JSON for signal calibration (Design Note 02).

Fan-out by TRACE TASK: one pod per task. Each pod runs the full 2-D grid
(tau x ema_window) on its shard via run.py --calibrate.

USAGE
    python infra/make_calib_armspec.py \\
        --n-tasks 8 \\
        --taus 0.1 0.2 0.3 0.4 0.5 0.6 0.7 \\
        --ema-windows 1 5 15 40 80 \\
        --out lab_reports/_staging/_armspecs/calib_signal.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_spec(n_tasks, taus, ema_windows, alpha, steps, extra_flags):
    spec = []
    taus_str = " ".join(str(t) for t in taus)
    ema_str = " ".join(str(w) for w in ema_windows)
    for ti in range(n_tasks):
        parts = [
            "--calibrate",
            f"--calib-tasks {ti}",
            f"--taus {taus_str}",
            f"--ema-windows {ema_str}",
            f"--calib-alpha {alpha}",
            f"--calib-steps {steps}",
        ]
        if extra_flags:
            parts.append(extra_flags.strip())
        spec.append({
            "slug": f"task{ti}",
            "value": f"task{ti}",
            "extra_args": " ".join(parts),
        })
    return spec


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--n-tasks", type=int, default=8)
    ap.add_argument(
        "--taus", type=float, nargs="+",
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
    )
    ap.add_argument(
        "--ema-windows", type=int, nargs="+",
        default=[1, 5, 15, 40, 80],
        help="EMA windows swept inside every pod (default 1 5 15 40 80)",
    )
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument(
        "--steps", type=int, default=150,
        help="base steps; actual per point = max(steps, 10*ema_window)",
    )
    ap.add_argument("--extra-flags", default="")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    spec = build_spec(
        args.n_tasks, args.taus, args.ema_windows,
        args.alpha, args.steps, args.extra_flags,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, indent=2) + "\n")

    n_points = args.n_tasks * len(args.taus) * len(args.ema_windows)
    print(f"wrote {args.out}")
    print(f"  pods (task shards)     : {args.n_tasks}")
    print(f"  taus                   : {args.taus}")
    print(f"  ema_windows            : {args.ema_windows}")
    print(f"  base steps             : {args.steps}  (actual max(base,10w))")
    print(f"  points per pod         : {len(args.taus) * len(args.ema_windows)}")
    print(f"  total points (all pods): {n_points}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
