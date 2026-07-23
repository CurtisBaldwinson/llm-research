#!/usr/bin/env python3
"""Launch N cloud-dispatcher arms in parallel, one per sweep value.

Each arm gets:

  - its own detached `python3 scripts/run_on_cloud.py ...` process
    (start_new_session=True; survives terminal close)
  - its own GPU pod (independent provisioning, independent SSH)
  - its own staging dir under lab_reports/_staging/<bench>_results_<ts>/
  - its own outer-nohup log under
    lab_reports/_staging/_logs/<sweep_name>_<ts>/arm_<value>.nohup.log

A manifest.json is written under the sweep's _logs subdir so all arms
can be tracked together (PIDs, log paths, dispatched commands, sweep
metadata).

Usage (a Makefile target should be the typical caller):

  python3 scripts/launch_parallel_sweep.py \
      --name imp05_v20_pp9_lambda \
      --benchmark 05_m_coupled_kl_rlhf \
      --provider runpod \
      --cloud SECURE \
      --values 0.001 0.01 0.03 0.1 \
      --extra-args-template \
        '--v2_mode --alpha_preset uniform --lambda_reg {value} --seeds 42 --conditions M_coupled_DPO vanilla_DPO'

The --extra-args-template string is `str.format(value=...)`-substituted
once per arm.  Use `{value}` exactly once to mark the swept slot.

Arms are spaced by --launch-spacing-s (default 2.0s) to avoid bursty
provider API rate limiting.

Exit code is 0 on successful spawn of all arms; non-zero if any arm's
subprocess.Popen raised.  Per-arm dispatcher exit codes are observed
asynchronously by the dispatcher itself; this launcher does NOT wait
for arm completion (intentional: it's a fire-and-forget so the
operator can close the laptop).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _slugify_value(value: str) -> str:
    """Make a value safe for a filename: '0.03' -> '0p03', 'pathway_diff' -> 'pathway_diff'."""
    return str(value).replace(".", "p").replace("/", "_").replace(" ", "_")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name",
        required=True,
        help="sweep name slug; used in log dir + manifest filenames",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="benchmark dir under benchmarks/ (e.g. 05_m_coupled_kl_rlhf)",
    )
    parser.add_argument(
        "--provider",
        default="runpod",
        help="cloud provider (passed to run_on_cloud.py; default runpod)",
    )
    parser.add_argument(
        "--gpu-type",
        default=None,
        help="comma-separated canonical GPU list (e.g. 'A100-80GB,A6000'); "
        "leave unset to use the benchmark's bench_caps.json preferred_gpus",
    )
    parser.add_argument(
        "--cloud",
        default=None,
        help="SECURE or COMMUNITY (runpod-only; ignored by other providers)",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        default=None,
        help="sweep values; one parallel arm is launched per value. "
        "Mutually exclusive with --arm-spec.",
    )
    parser.add_argument(
        "--extra-args-template",
        default=None,
        help="extra-args template with {value} placeholder; "
        "str.format(value=...)-substituted per arm. Required with --values.",
    )
    parser.add_argument(
        "--arm-spec",
        default=None,
        help="path to a JSON file describing arms for multi-axis sweeps. "
        "File must contain a JSON array of objects, each with 'slug' (str) "
        "and 'extra_args' (str) keys; optional 'value' (str) for human display. "
        "Mutually exclusive with --values.",
    )
    parser.add_argument(
        "--launch-spacing-s",
        type=float,
        default=2.0,
        help="seconds between consecutive arm launches (default 2.0)",
    )
    args = parser.parse_args()

    # Validate the two dispatch modes are mutually exclusive and one is supplied.
    if args.arm_spec and args.values:
        print("ERROR: --arm-spec and --values are mutually exclusive.", file=sys.stderr)
        return 2
    if not args.arm_spec and not args.values:
        print("ERROR: one of --arm-spec or --values is required.", file=sys.stderr)
        return 2
    if args.values and not args.extra_args_template:
        print("ERROR: --extra-args-template is required when using --values.", file=sys.stderr)
        return 2
    if args.values and "{value}" not in args.extra_args_template:
        print(
            "ERROR: --extra-args-template must contain '{value}' placeholder.",
            file=sys.stderr,
        )
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    cloud_script = repo_root / "scripts" / "run_on_cloud.py"
    if not cloud_script.exists():
        print(f"ERROR: dispatcher not found at {cloud_script}", file=sys.stderr)
        return 2

    # Build the unified arm-list regardless of dispatch mode.
    arm_inputs: list[dict] = []
    if args.arm_spec:
        spec_path = Path(args.arm_spec)
        if not spec_path.is_absolute():
            spec_path = repo_root / spec_path
        if not spec_path.exists():
            print(f"ERROR: arm-spec file not found: {spec_path}", file=sys.stderr)
            return 2
        arm_inputs = json.loads(spec_path.read_text())
        if not isinstance(arm_inputs, list):
            print(f"ERROR: arm-spec file must contain a JSON array; got {type(arm_inputs).__name__}", file=sys.stderr)
            return 2
        for i, a in enumerate(arm_inputs):
            if not isinstance(a, dict) or "slug" not in a or "extra_args" not in a:
                print(
                    f"ERROR: arm-spec[{i}] must be an object with 'slug' and 'extra_args' keys; got {a!r}",
                    file=sys.stderr,
                )
                return 2
    else:
        for value in args.values:
            arm_inputs.append({
                "value": value,
                "slug": _slugify_value(value),
                "extra_args": args.extra_args_template.format(value=value),
            })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_dir = repo_root / "lab_reports" / "_staging" / "_logs" / f"{args.name}_{stamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"Launching parallel sweep '{args.name}' ({len(arm_inputs)} arms):")
    print(f"  benchmark = {args.benchmark}")
    print(f"  provider  = {args.provider}")
    print(f"  gpu_type  = {args.gpu_type or '(bench_caps default)'}")
    print(f"  cloud     = {args.cloud or '(provider default)'}")
    print(f"  mode      = {'arm-spec' if args.arm_spec else 'values+template'}")
    print(f"  log dir   = {sweep_dir.relative_to(repo_root)}")
    print()

    arms: list[dict] = []
    for idx, arm_in in enumerate(arm_inputs):
        extra_args = arm_in["extra_args"]
        slug = arm_in["slug"]
        value = arm_in.get("value", slug)
        log_path = sweep_dir / f"arm_{slug}.nohup.log"
        pid_path = sweep_dir / f"arm_{slug}.pid"

        cmd: list[str] = [
            sys.executable,
            str(cloud_script),
            "--provider",
            args.provider,
            "--benchmark",
            args.benchmark,
        ]
        if args.gpu_type:
            cmd.extend(["--gpu-type", args.gpu_type])
        if args.cloud and args.provider == "runpod":
            cmd.extend(["--cloud", args.cloud])
        cmd.extend(["--extra-args", extra_args])

        with open(log_path, "wb") as logfile:
            proc = subprocess.Popen(
                cmd,
                stdout=logfile,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                cwd=str(repo_root),
            )

        pid_path.write_text(f"{proc.pid}\n")
        arm_record = {
            "value": value,
            "slug": slug,
            "pid": proc.pid,
            "extra_args": extra_args,
            "cmd": " ".join(shlex.quote(c) for c in cmd),
            "log_path": str(log_path.relative_to(repo_root)),
            "pid_path": str(pid_path.relative_to(repo_root)),
            "launched_utc": datetime.now(timezone.utc).isoformat(),
        }
        arms.append(arm_record)
        print(f"  [{idx + 1}/{len(arm_inputs)}] value={str(value):<24}  pid={proc.pid}  log={log_path.relative_to(repo_root)}")

        if idx < len(arm_inputs) - 1 and args.launch_spacing_s > 0:
            time.sleep(args.launch_spacing_s)

    manifest = {
        "name": args.name,
        "stamp": stamp,
        "benchmark": args.benchmark,
        "provider": args.provider,
        "gpu_type": args.gpu_type,
        "cloud": args.cloud,
        "dispatch_mode": "arm-spec" if args.arm_spec else "values+template",
        "extra_args_template": args.extra_args_template,
        "arm_spec_path": args.arm_spec,
        "arm_count": len(arms),
        "arms": arms,
    }
    manifest_path = sweep_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    rel_sweep = sweep_dir.relative_to(repo_root)
    print()
    print(f"Launched {len(arms)} parallel arms.")
    print(f"  manifest      : {manifest_path.relative_to(repo_root)}")
    print(f"  follow all    : tail -f {rel_sweep}/arm_*.nohup.log")
    print(
        f"  liveness check: for f in {rel_sweep}/arm_*.pid; do "
        'pid=$(cat $f); kill -0 $pid 2>/dev/null '
        '&& echo "$f alive" || echo "$f done"; done'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
