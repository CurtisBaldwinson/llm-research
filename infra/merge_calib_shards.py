#!/usr/bin/env python3
"""Merge per-task calibration shards into one lookup + signal_tuning.json.

Design Note 02. Each parallel calib pod writes
``results/calib_signal_shard<i>.json``. This tool unions all ``points``,
rebuilds the 2-D (tau x ema_window) lookup at full JSON precision, recomputes
the nearest-50% default-arm recommendation, emits frontier endpoints for the
pre-registered protected-fraction targets, and optionally writes
``signal_tuning.json`` for mechanism-arm autowire.

USAGE
    python infra/merge_calib_shards.py \\
        --glob 'lab_reports/_staging/**/calib_signal_shard*.json' \\
        --out lab_reports/_staging/calib_merged.json \\
        --write-signal-tuning lab_reports/_staging/signal_tuning.json
"""
from __future__ import annotations

import argparse
import json
import sys
from glob import glob
from pathlib import Path

# Allow importing run.py helpers without executing main.
_REPO = Path(__file__).resolve().parents[1]
_TRACE = _REPO / "benchmarks" / "trace"
if str(_TRACE) not in sys.path:
    sys.path.insert(0, str(_TRACE))
if str(_REPO / "sm_src") not in sys.path:
    sys.path.insert(0, str(_REPO / "sm_src"))


def _load_shards(patterns: list[str]) -> list[dict]:
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(Path(p) for p in glob(pat, recursive=True))
    # de-dupe
    uniq = sorted({p.resolve() for p in paths if p.is_file()})
    if not uniq:
        raise FileNotFoundError(f"no shard files matched: {patterns}")
    shards = []
    for p in uniq:
        data = json.loads(p.read_text())
        if not isinstance(data, dict) or "points" not in data:
            raise ValueError(f"not a calib shard: {p}")
        data["_source_path"] = str(p)
        shards.append(data)
    return shards


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--glob", action="append", dest="globs", default=None,
        help="glob for calib_signal_shard*.json (repeatable). "
             "Default: lab_reports/**/calib_signal_shard*.json",
    )
    ap.add_argument("--out", type=Path, required=True,
                    help="merged calib JSON path")
    ap.add_argument(
        "--write-signal-tuning", type=Path, default=None,
        help="if set, write signal_tuning.json from recommended row",
    )
    ap.add_argument(
        "--allow-unconfirmed", action="store_true",
        help="write signal_tuning even though confirmation_required stays true",
    )
    args = ap.parse_args()

    patterns = args.globs or [
        "lab_reports/**/calib_signal_shard*.json",
        "lab_reports/_staging/**/calib_signal_shard*.json",
        "lab_reports/_staging/_smokes/**/calib_signal_shard*.json",
    ]
    shards = _load_shards(patterns)
    print(f"loaded {len(shards)} shards:")
    for s in shards:
        print(f"  {s['_source_path']}  tasks={s.get('task_shard')} "
              f"n_points={len(s.get('points') or [])}")

    # Union points (full precision).
    points = []
    for s in shards:
        points.extend(s.get("points") or [])
    if not points:
        print("ERROR: no points in shards", file=sys.stderr)
        return 2

    alphas = {round(float(s.get("alpha", 0.1)), 9) for s in shards}
    norms = {s.get("normalization") for s in shards}
    if len(alphas) > 1 or len(norms) > 1:
        print(f"ERROR: incompatible shards: alphas={sorted(alphas)} "
              f"normalizations={sorted(n or '?' for n in norms)}",
              file=sys.stderr)
        return 2
    alpha = alphas.pop()
    taus = sorted({float(p["tau"]) for p in points})
    ema_windows = sorted({int(p["ema_window"]) for p in points})

    # Import rebuild helpers from run.py
    import run as trace_run  # noqa: E402

    table, recommended, pmap, frontier = trace_run._build_lookup_and_recommend(
        points, taus, ema_windows, alpha,
    )

    # EMA-on verdict (DN02): selectable rows only (adoptable w, n_tasks>=min,
    # real std). Under-populated std=None and high-w over-smooth rows stay
    # mapped but cannot win. Std term requires BOTH sides have real std.
    max_w = int(getattr(trace_run, "MAX_ADOPTABLE_EMA_WINDOW", 15))
    min_n = int(getattr(trace_run, "MIN_TASKS_FOR_VERDICT", 2))
    std_delta = float(getattr(trace_run, "EMA_ON_MATERIAL_STD_DELTA", 0.02))
    prot_band = float(getattr(trace_run, "EMA_ON_COMPARABLE_PROT_BAND", 0.10))
    row_ok = getattr(
        trace_run, "_row_selectable_for_verdict",
        lambda t: t.get("selectable_for_verdict", False),
    )
    # One comparator everywhere (per-window + cross-window + run.py recommend).
    row_key = getattr(trace_run, "_verdict_row_key", None)
    if row_key is None:
        def row_key(t):
            return (
                int(t["n_mass_split_fail_tasks"]),
                abs(float(t["protected_frac_mean"]) - 0.5)
                + 2.0 * float(t["protected_frac_std"]),
            )

    def _best_for_w(w: int):
        if int(w) > max_w:
            return None  # measured but non-adoptable window
        rows = [
            t for t in table
            if int(t["ema_window"]) == int(w) and row_ok(t)
        ]
        if not rows:
            return None
        return min(rows, key=row_key)

    w1 = _best_for_w(1)
    w_on = None
    for w in ema_windows:
        if w <= 1 or int(w) > max_w:
            continue
        cand = _best_for_w(w)
        if cand is None:
            continue
        # Same key as _best_for_w / run._verdict_row_key — never std-first.
        if w_on is None or row_key(cand) < row_key(w_on):
            w_on = cand

    ema_verdict = {
        "rule": (
            f"Adopt EMA-on only for selectable rows "
            f"(w<=MAX_ADOPTABLE_EMA_WINDOW({max_w}), "
            f"n_tasks>=MIN_TASKS_FOR_VERDICT({min_n}), real std). "
            "Require mass-split improvement, OR a material std reduction "
            f"(>={std_delta}) at comparable protected mass "
            f"(within {prot_band}) when BOTH sides have real std. "
            "If either std is unknown, decide on mass-split alone. "
            "Else record 'rank alone suffices on TRACE'."
        ),
        "max_adoptable_ema_window": max_w,
        "min_tasks_for_verdict": min_n,
        "best_w1": w1,
        "best_ema_on": w_on,
        "adopt_ema_on": None,
        "rationale": None,
    }
    if w1 and w_on:
        better_mass = (
            w_on["n_mass_split_fail_tasks"] < w1["n_mass_split_fail_tasks"]
        )
        std_w1 = w1.get("protected_frac_std")
        std_on = w_on.get("protected_frac_std")
        both_real_std = std_w1 is not None and std_on is not None
        prot_w1 = float(w1["protected_frac_mean"])
        prot_on = float(w_on["protected_frac_mean"])
        comparable_prot = abs(prot_on - prot_w1) <= prot_band
        material_std = False
        if both_real_std and comparable_prot:
            material_std = (
                float(std_on) <= float(std_w1) - std_delta
            )
        # Std term only when both sides have real std; else mass-split alone.
        adopt = bool(better_mass or (both_real_std and material_std))
        ema_verdict["adopt_ema_on"] = adopt
        ema_verdict["comparable_protected"] = comparable_prot
        ema_verdict["both_real_std"] = both_real_std
        ema_verdict["material_std_improvement"] = material_std
        ema_verdict["mass_split_improvement"] = better_mass
        if adopt and better_mass and material_std:
            ema_verdict["rationale"] = "ema_on_mass_and_std"
        elif adopt and better_mass:
            ema_verdict["rationale"] = "ema_on_mass_split"
        elif adopt and material_std:
            ema_verdict["rationale"] = "ema_on_material_std_comparable_prot"
        elif not both_real_std and not better_mass:
            ema_verdict["rationale"] = "rank_alone_suffices_std_unknown"
        else:
            ema_verdict["rationale"] = "rank_alone_suffices"
    elif w1 and not w_on:
        ema_verdict["adopt_ema_on"] = False
        ema_verdict["rationale"] = "no_healthy_adoptable_ema_on_row"
    elif w_on and not w1:
        adopt = w_on["n_mass_split_fail_tasks"] == 0
        ema_verdict["adopt_ema_on"] = adopt
        ema_verdict["rationale"] = (
            "only_ema_on_healthy" if adopt else "ema_on_mass_split_fail"
        )

    # If EMA-on wins, prefer that row as recommended default.
    if ema_verdict.get("adopt_ema_on") and w_on is not None:
        recommended = w_on
    elif w1 is not None:
        recommended = w1

    merged = {
        "harness": "calibrate_signal_merged",
        "design_note": "02_signal_normalization_pivot",
        "calibration_scope": "init_signal",
        "calibration_scope_note": shards[0].get("calibration_scope_note"),
        "normalization": "rank",
        "alpha": alpha,
        "taus": taus,
        "ema_windows": ema_windows,
        "n_shards": len(shards),
        "shard_sources": [s["_source_path"] for s in shards],
        "points": points,
        "lookup_table": table,
        "protected_frac_map": pmap,
        "frontier_endpoints": frontier,
        "recommended": recommended,
        "ema_verdict": ema_verdict,
        "protected_frac_targets": getattr(
            trace_run, "PROTECTED_FRAC_TARGETS", [0.05, 0.10, 0.20, 0.40]
        ),
        "note": (
            "Merged init-signal calibration. recommended feeds "
            "signal_tuning.json default arm beta/ema_window. "
            "frontier_endpoints map pre-registered protected fractions. "
            "confirmation_required until post-1-epoch-task0 agrees."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"wrote merged: {args.out}")
    if recommended:
        print(
            f"RECOMMENDED tau={recommended['tau']} w={recommended['ema_window']} "
            f"prot%={100*(recommended['protected_frac_mean'] or 0):.1f} "
            f"+/-{100*(recommended['protected_frac_std'] or 0):.1f} "
            f"beta={trace_run._bt_beta_for_breakeven(alpha, recommended['tau']):.4f}"
        )
    print(f"ema_verdict: {ema_verdict.get('rationale')}")

    if args.write_signal_tuning:
        if not recommended:
            print("ERROR: no recommended row; cannot write signal_tuning",
                  file=sys.stderr)
            return 2
        out_st = trace_run.write_signal_tuning_json(
            recommended, alpha, args.write_signal_tuning,
        )
        print(f"wrote signal_tuning: {out_st}")
        if not args.allow_unconfirmed:
            print(
                "NOTE: confirmation_required=true — run post-1-epoch-task0 "
                "before fleet; pass --allow-unconfirmed only if intentional."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
