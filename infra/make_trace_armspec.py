#!/usr/bin/env python3
"""Generate a launch_parallel_sweep.py --arm-spec JSON for the TRACE Exp1 matrix.

One parallel arm (== one GPU pod) per continual-learning method. Each pod runs
that arm's full seed x lambda grid *internally* via run.py's --arms/--seeds/
--lambdas flags, so the whole 7 x 5 x |lambda| matrix is covered by 7 pods.

This is the arm-axis fan-out (matches v1's proven per-arm parallelism). It is
NOT a flat 175-pod fan-out; each pod does 5 seeds x |lambdas| sequentially.

Usage:
    python infra/make_trace_armspec.py \
        --seeds 0 1 2 3 4 \
        --lambdas 0.001 0.01 0.03 0.1 0.3 \
        --out lab_reports/_staging/_armspecs/trace_exp1.json
    # then feed --out to launch_parallel_sweep.py --arm-spec

Keep --arms in sync with EXP1_ARMS in benchmarks/trace/run.py.
"""
import argparse
import json
import sys
from pathlib import Path

# Mirror of EXP1_ARMS in benchmarks/trace/run.py (kept explicit so the launcher
# is auditable; a mismatch is caught by the --check flag below).
EXP1_ARMS = [
    "gradient_aligned_m",   # M + lock  (mechanism under test)
    "raw_fisher_signal",    # same signal, no lock (the isolating control)
    "ewc",                  # classical Fisher floor
    "elastic_reset",        # weight-space
    "model_averaging",      # weight-space, cheap
    "replay",               # rehearsal
    "vanilla",              # sanity floor
]


def build_spec(arms, seeds, lambdas, extra_flags):
    """One arm-spec entry per method. extra_args is a single shell-ready string
    consumed by run_on_cloud.py --extra-args and forwarded to run.py."""
    spec = []
    seeds_str = " ".join(str(s) for s in seeds)
    for arm in arms:
        parts = [f"--arms {arm}", f"--seeds {seeds_str}"]
        if lambdas:
            parts.append("--lambdas " + " ".join(str(l) for l in lambdas))
        if extra_flags:
            parts.append(extra_flags.strip())
        spec.append({
            "slug": arm,
            "value": arm,
            "extra_args": " ".join(parts),
        })
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=EXP1_ARMS,
                    help="arms to fan out (default: all 7 EXP1_ARMS -> 7 pods)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="seeds run *inside* each pod (default 0..4, n=5 per README)")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.001, 0.01, 0.03, 0.1, 0.3],
                    help="lambda sweep run inside each pod")
    ap.add_argument("--extra-flags", default="",
                    help="extra run.py flags appended verbatim to every arm "
                         "(e.g. '--n_tasks 8 --batch_size 8')")
    ap.add_argument("--out", type=Path, required=True,
                    help="output arm-spec JSON path")
    ap.add_argument("--check", action="store_true",
                    help="assert --arms is a subset of the known EXP1_ARMS")
    args = ap.parse_args()

    if args.check:
        unknown = [a for a in args.arms if a not in EXP1_ARMS]
        if unknown:
            print(f"ERROR: unknown arms not in EXP1_ARMS: {unknown}", file=sys.stderr)
            return 2

    spec = build_spec(args.arms, args.seeds, args.lambdas, args.extra_flags)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, indent=2) + "\n")

    n_cells = len(args.arms) * len(args.seeds) * (len(args.lambdas) or 1)
    print(f"wrote {args.out}")
    print(f"  pods (parallel arms) : {len(args.arms)}")
    print(f"  seeds per pod        : {len(args.seeds)}  ({args.seeds})")
    print(f"  lambdas per pod      : {len(args.lambdas)}  ({args.lambdas})")
    print(f"  total matrix cells   : {n_cells}  (arms x seeds x lambdas)")
    print(f"  cells per pod        : {n_cells // len(args.arms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
