#!/usr/bin/env python3
"""
Deterministic lab-report finalizer.

Given a local results directory produced by ``scripts/run_on_runpod.py`` (or
any equivalent local-disk layout), this tool:

1. Extracts every numeric and structural field from the run's artifacts
   (``manifest.json``, ``eval_results.csv``, ``summary_table.csv``,
   ``results_summary.txt``, ``events.csv``, ``run.log``) into a
   machine-readable ``report_data.json``.
2. Creates the lab-report folder under ``lab_reports/<date>_<hhmm>_<slug>/``
   (Pacific Time, per ``LAB_REPORT_STANDARD.md`` §1) and copies the run's
   artifacts into ``<report>/results/``.
3. Renders ``lab_reports/_template.md`` with all numeric / structural
   placeholders populated; prose-only sections (§4 analysis, §5 decision
   narrative, §1 framing, Verdict line) are left as
   ``<!-- AGENT: ... -->`` placeholders for the operator to fill in.
4. Optionally updates ``COMPUTE_LEDGER.md`` with the new row, computing
   cumulative and remaining hours from the previous row plus the new
   wallclock.  Flags loudly if cumulative would exceed the documented
   180.0 h Phase 1+2 budget.

The tool never writes numbers that aren't sourced from a CSV cell, the
manifest, or a git command.  It refuses to silently re-derive an estimate.

Usage::

    # Inspect what would be written, without touching disk:
    python3 scripts/finalize_run.py \\
        --results-dir ~/05_m_coupled_kl_rlhf_results_20260512-174451 \\
        --benchmark 05_m_coupled_kl_rlhf \\
        --slug imp05-v10-headline-refuted \\
        --operator agent:oz-warp-2026-05-13 \\
        --gpu-rate-usd-per-hour 1.50 \\
        --dispatcher-log logs/imp05_headline_3B_20260513-004421.log \\
        --dry-run

    # Actually create the lab-report folder, render report.md, update the
    # ledger (does NOT git-commit; that's an explicit operator step):
    python3 scripts/finalize_run.py \\
        --results-dir ~/05_m_coupled_kl_rlhf_results_20260512-174451 \\
        --benchmark 05_m_coupled_kl_rlhf \\
        --slug imp05-v10-headline-refuted \\
        --operator agent:oz-warp-2026-05-13 \\
        --gpu-rate-usd-per-hour 1.50 \\
        --dispatcher-log logs/imp05_headline_3B_20260513-004421.log

Then the operator fills in the ``<!-- AGENT: ... -->`` prose blocks and
runs::

    python3 scripts/verify_report.py \\
        lab_reports/2026-05-13_0430_imp05-v10-headline-refuted/

to confirm no numeric drift between the prose and the source data.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo  # py>=3.9
except ImportError:  # pragma: no cover — the project's python is 3.11+ but be safe
    ZoneInfo = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAB_REPORTS = PROJECT_ROOT / "lab_reports"
TEMPLATE_PATH = LAB_REPORTS / "_template.md"
LEDGER_PATH = PROJECT_ROOT / "COMPUTE_LEDGER.md"

# Phase 1+2 budget per `COMPUTE_LEDGER.md` (revised 2026-05-10 from 60.0 h).
# Hard-coded here only as a sanity check; the actual remaining-hours value
# is always derived from the ledger's most recent row, not from this constant.
BUDGET_HOURS_PHASE_1_2 = 180.0

# Names of sentinel files written by the dispatcher's remote wrapper.  See
# `scripts/run_on_runpod.py::_build_remote_wrapper` for the producer side.
# Filesystem markers are preferred over regex-scraping `run.log` because the
# filenames are stable across any future log-format churn.  The regex paths
# remain as a backward-compat fallback for runs from before these sentinels
# existed (e.g. the v1.0 imp05 headline ~/05_m_coupled_kl_rlhf_results_*).
STARTED_SENTINEL_PREFIX = "_STARTED."
FINISHED_SENTINEL_PREFIX = "_FINISHED."
DONE_SENTINEL_PREFIX = "_DONE."


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

@dataclass
class RunData:
    """Everything we extract from a completed run's artifacts."""
    # Filesystem
    results_dir: Path
    # From manifest.json
    manifest: Dict[str, Any]
    experiment: str
    config: Dict[str, Any]
    environment: Dict[str, Any]
    file_hashes: Dict[str, str]
    # From run.log (wrapper start/end)
    wrapper_start_utc: datetime
    wrapper_end_utc: datetime
    wrapper_exit_code: int
    wallclock_seconds: float
    # From run.log (base eval line)
    base_evals: Dict[str, float]
    # From run.log (CUDA OK line)
    hardware: str
    # From results_summary.txt (already parsed by the runner)
    hypothesis_status: List[Tuple[str, str, str]]  # (id, status, evidence)
    decision: str
    sweep_mode: bool  # True if any condition row has multiple lambda values
    # From eval_results.csv
    eval_rows: List[Dict[str, Any]]
    # From summary_table.csv  (richer per-(cond, seed) view incl. lambda_reg)
    summary_rows: List[Dict[str, Any]]
    # From events.csv (Phase A / B / C boundaries per seed)
    events: List[Dict[str, str]]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_utc(s: str) -> datetime:
    """Parse a UTC timestamp from the runner's emitted formats.

    Two shapes appear:
      - ``[wrapper] starting at 2026-05-13T00:44:51Z`` (Z suffix, second-precision)
      - ``2026-05-13T11:16:12.868848+00:00`` (manifest environment.utc_time)
    """
    if s.endswith("Z"):
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(s)


def _sentinel_utc(artifacts_dir: Path, prefix: str) -> Optional[datetime]:
    """Return the UTC datetime encoded in the first `<prefix><utc>` file, or None.

    The wrapper writes one sentinel per phase (_STARTED, _FINISHED) with the
    UTC time baked into the filename (`_STARTED.2026-05-13T00:44:51Z`).  This
    helper decodes that filename without consulting any log content, so the
    timestamp is stable even if `run.log` is later truncated or the wrapper's
    log-line format changes.
    """
    matches = sorted(artifacts_dir.glob(prefix + "*"))
    if not matches:
        return None
    suffix = matches[0].name[len(prefix):]
    try:
        return _parse_utc(suffix)
    except (ValueError, TypeError):
        return None


def _sentinel_exit_code(artifacts_dir: Path) -> Optional[int]:
    """Return the int exit code encoded in `_DONE.<rc>`, or None."""
    matches = sorted(artifacts_dir.glob(DONE_SENTINEL_PREFIX + "*"))
    if not matches:
        return None
    suffix = matches[0].name[len(DONE_SENTINEL_PREFIX):]
    # Some legacy wrappers wrote `_DONE.<rc>` with rc as a plain int; newer
    # wrappers also emit `_FINISHED.<utc>` separately, so the rc parse is
    # straightforward here.
    try:
        return int(suffix)
    except ValueError:
        return None


def _parse_wrapper_times(
    run_log: str, artifacts_dir: Optional[Path] = None,
) -> Tuple[datetime, datetime, int]:
    """Return (start_utc, end_utc, exit_code) preferring sentinels, regex fallback.

    Source-of-truth precedence:
      1. `_STARTED.<utc>` / `_FINISHED.<utc>` / `_DONE.<rc>` files in
         `artifacts_dir`.  Stable across any future churn in the wrapper's
         log-line format — the filename IS the data.
      2. Regex against `run.log` for the `[wrapper] starting at ...` and
         `[wrapper] python exit=<rc> at ...` lines.  Kept as a backward-compat
         path so finalize_run can still process older runs (e.g. the v1.0
         imp05 headline) that pre-date the sentinel files.
    """
    started_dt = end_dt = None
    rc: Optional[int] = None
    if artifacts_dir is not None:
        started_dt = _sentinel_utc(artifacts_dir, STARTED_SENTINEL_PREFIX)
        end_dt = _sentinel_utc(artifacts_dir, FINISHED_SENTINEL_PREFIX)
        rc = _sentinel_exit_code(artifacts_dir)

    if started_dt is None:
        m_start = re.search(
            r"\[wrapper\] starting at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)",
            run_log,
        )
        if not m_start:
            raise RuntimeError(
                "could not determine wrapper start time: no _STARTED.<utc> "
                "sentinel and no '[wrapper] starting at <UTC>' marker in run.log"
            )
        started_dt = _parse_utc(m_start.group(1))

    if end_dt is None or rc is None:
        m_end = re.search(
            r"\[wrapper\] python exit=(-?\d+) at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)",
            run_log,
        )
        if not m_end:
            raise RuntimeError(
                "could not determine wrapper end time / exit code: no "
                "_FINISHED.<utc>+_DONE.<rc> sentinels and no "
                "'[wrapper] python exit=<rc> at <UTC>' marker in run.log"
            )
        if end_dt is None:
            end_dt = _parse_utc(m_end.group(2))
        if rc is None:
            rc = int(m_end.group(1))

    return started_dt, end_dt, rc


def _parse_base_evals(run_log: str) -> Dict[str, float]:
    """Pull '[main] base evals: MMLU=X  ARC=Y  TQA=Z' from run.log."""
    m = re.search(
        r"\[main\] base evals: MMLU=(\d+\.\d+)\s+ARC=(\d+\.\d+)\s+TQA=(\d+\.\d+)",
        run_log,
    )
    if not m:
        raise RuntimeError(
            "run.log missing '[main] base evals: MMLU=... ARC=... TQA=...' marker"
        )
    return {
        "mmlu": float(m.group(1)),
        "arc": float(m.group(2)),
        "tqa": float(m.group(3)),
    }


def _parse_hardware(run_log: str) -> str:
    """Pull 'CUDA OK: <hardware>' from run.log."""
    m = re.search(r"CUDA OK:\s*(.+)", run_log)
    if not m:
        return "unknown"
    return m.group(1).strip()


def _parse_hypothesis_lines(results_summary: str) -> List[Tuple[str, str, str]]:
    """Pull each '[Hn] STATUS evidence' line from results_summary.txt.

    Returns a list of (id, status, evidence) tuples in declared order.
    The evidence string is the full free-text after the status label.
    """
    out: List[Tuple[str, str, str]] = []
    for m in re.finditer(
        r"\[(H\d)\]\s+(\S+)\s+(.+)",
        results_summary,
    ):
        out.append((m.group(1), m.group(2), m.group(3).strip()))
    return out


def _parse_decision(results_summary: str) -> str:
    """Pull the '** Decision: ... **' line from results_summary.txt."""
    m = re.search(r"\*\*\s*Decision:\s*(.+?)\s*\*\*", results_summary)
    if not m:
        return "UNKNOWN"
    return m.group(1).strip()


def _detect_sweep_mode(summary_rows: List[Dict[str, Any]]) -> bool:
    """True if any condition has multiple distinct lambda_reg values."""
    by_cond: Dict[str, set] = {}
    for r in summary_rows:
        by_cond.setdefault(r["condition"], set()).add(r["lambda_reg"])
    return any(len(s) > 1 for s in by_cond.values())


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open()))


def extract(results_dir: Path) -> RunData:
    """Pull every numeric/structural field from a results directory.

    `results_dir` may be either the dispatcher's outer timestamp-named
    directory (``~/<bench>_results_<timestamp>/`` or
    ``lab_reports/_staging/<bench>_results_<timestamp>/``) or the inner
    ``results/`` directory it contains.  Both layouts are accepted because
    the dispatcher produces the outer form and the lab-report folder
    mirrors the inner.
    """
    if not results_dir.is_dir():
        raise FileNotFoundError(f"--results-dir does not exist: {results_dir}")

    # Auto-detect whether we were pointed at the outer or the inner dir.
    inner = results_dir / "results"
    if (inner / "manifest.json").exists():
        artifacts_dir = inner
    elif (results_dir / "manifest.json").exists():
        artifacts_dir = results_dir
    else:
        raise FileNotFoundError(
            f"could not find manifest.json under {results_dir} or {inner}"
        )

    manifest_path = artifacts_dir / "manifest.json"
    run_log_path = artifacts_dir / "run.log"
    summary_path = artifacts_dir / "results_summary.txt"
    eval_csv = artifacts_dir / "eval_results.csv"
    summary_csv = artifacts_dir / "summary_table.csv"
    events_csv = artifacts_dir / "events.csv"

    for required in (manifest_path, run_log_path, summary_path, eval_csv, summary_csv):
        if not required.exists():
            raise FileNotFoundError(
                f"required artifact missing: {required}"
            )

    manifest = json.loads(_read_text(manifest_path))
    run_log = _read_text(run_log_path)
    results_summary = _read_text(summary_path)

    file_hashes = {f["name"]: f["sha256"] for f in manifest.get("files", [])}
    config = manifest.get("config", {})
    environment = manifest.get("environment", {})

    wrapper_start, wrapper_end, wrapper_rc = _parse_wrapper_times(
        run_log, artifacts_dir=artifacts_dir,
    )
    wallclock_s = (wrapper_end - wrapper_start).total_seconds()

    base_evals = _parse_base_evals(run_log)
    hardware = _parse_hardware(run_log)

    # Prefer the structured `results_summary.json` sidecar over regex-scraping
    # the .txt.  Both are written by the v1.1+ runner; older runs (e.g. the
    # v1.0 imp05 headline) ship only .txt so we keep the regex fallback alive.
    summary_json_path = artifacts_dir / "results_summary.json"
    if summary_json_path.exists():
        try:
            summary_data = json.loads(_read_text(summary_json_path))
            hypothesis_status = [
                (h["id"], h["status"], h["evidence"])
                for h in summary_data.get("hypotheses", [])
            ]
            decision = summary_data.get("decision") or _parse_decision(results_summary)
            if not hypothesis_status:
                hypothesis_status = _parse_hypothesis_lines(results_summary)
        except (json.JSONDecodeError, KeyError, TypeError):
            hypothesis_status = _parse_hypothesis_lines(results_summary)
            decision = _parse_decision(results_summary)
    else:
        hypothesis_status = _parse_hypothesis_lines(results_summary)
        decision = _parse_decision(results_summary)

    # Cast CSV cells to native types where useful; preserve strings for IDs.
    raw_summary_rows = _load_csv(summary_csv)
    summary_rows: List[Dict[str, Any]] = []
    for r in raw_summary_rows:
        new = dict(r)
        for k in ("seed",):
            if k in new and new[k] != "":
                new[k] = int(new[k])
        for k in (
            "lambda_reg", "beta_dpo",
            "m_mean_after_a", "m_std_after_a", "bimodality_coef_after_a",
            "fisher_mean", "fisher_std",
            "final_dpo_loss", "final_dpo_train_acc", "final_aux_penalty",
            "mmlu_acc", "arc_easy_acc", "truthful_qa_mc1",
            "implicit_reward_acc", "mean_reward_gap",
            "mmlu_drop_pct", "arc_drop_pct", "truthful_qa_drop_pct",
            "wall_seconds",
        ):
            if k in new and new[k] not in ("", None):
                try:
                    new[k] = float(new[k])
                except ValueError:
                    pass
        summary_rows.append(new)

    eval_rows = _load_csv(eval_csv)
    for r in eval_rows:
        for k in ("seed",):
            if k in r and r[k] != "":
                r[k] = int(r[k])
        for k in (
            "mmlu_acc", "arc_easy_acc", "truthful_qa_mc1",
            "implicit_reward_acc", "mean_reward_gap",
            "mmlu_drop_pct", "arc_drop_pct", "truthful_qa_drop_pct",
            "wallclock_s",
        ):
            if k in r and r[k] not in ("", None):
                try:
                    r[k] = float(r[k])
                except ValueError:
                    pass

    events = _load_csv(events_csv)

    sweep_mode = _detect_sweep_mode(summary_rows)

    return RunData(
        results_dir=artifacts_dir,
        manifest=manifest,
        experiment=manifest.get("experiment", ""),
        config=config,
        environment=environment,
        file_hashes=file_hashes,
        wrapper_start_utc=wrapper_start,
        wrapper_end_utc=wrapper_end,
        wrapper_exit_code=wrapper_rc,
        wallclock_seconds=wallclock_s,
        base_evals=base_evals,
        hardware=hardware,
        hypothesis_status=hypothesis_status,
        decision=decision,
        sweep_mode=sweep_mode,
        eval_rows=eval_rows,
        summary_rows=summary_rows,
        events=events,
    )


# ---------------------------------------------------------------------------
# Aggregation for the report tables
# ---------------------------------------------------------------------------

def _stats_mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if isinstance(x, (int, float))]
    if not xs:
        return None
    return sum(xs) / len(xs)


def build_aggregate_rows(summary_rows: List[Dict[str, Any]], sweep_mode: bool) -> List[Dict[str, Any]]:
    """Group rows by condition (or (cond, lambda) in sweep mode) and emit one summary row per group.

    Each emitted row carries the same shape as the markdown aggregate table.
    """
    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for r in summary_rows:
        key = (r["condition"], r["lambda_reg"]) if sweep_mode else (r["condition"],)
        groups.setdefault(key, []).append(r)
    out: List[Dict[str, Any]] = []
    for key, rows in groups.items():
        label = (
            f"{key[0]} \u03bb={key[1]}" if sweep_mode else key[0]
        )
        out.append({
            "label": label,
            "n_seeds": len({r["seed"] for r in rows}),
            "lambda_reg": rows[0]["lambda_reg"],
            "mmlu_mean":          _stats_mean([r["mmlu_acc"]              for r in rows]),
            "mmlu_drop_mean":     _stats_mean([r["mmlu_drop_pct"]         for r in rows]),
            "arc_mean":           _stats_mean([r["arc_easy_acc"]          for r in rows]),
            "tqa_mean":           _stats_mean([r["truthful_qa_mc1"]       for r in rows]),
            "ira_mean":           _stats_mean([r["implicit_reward_acc"]   for r in rows]),
            "dpo_loss_mean":      _stats_mean([r["final_dpo_loss"]        for r in rows]),
            "aux_pen_mean":       _stats_mean([r["final_aux_penalty"]     for r in rows]),
        })
    return out


def format_aggregate_table(rows: List[Dict[str, Any]]) -> str:
    """Render the §3.1 markdown aggregate table from grouped rows."""
    header = (
        "| condition | n | MMLU | MMLU \u0394% | ARC | TQA | IRA | DPO loss | aux pen |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    body_lines = []
    for r in rows:
        body_lines.append(
            f"| {r['label']} | {r['n_seeds']} | "
            f"{r['mmlu_mean']:.3f} | {r['mmlu_drop_mean']:.2f} | "
            f"{r['arc_mean']:.3f} | {r['tqa_mean']:.3f} | "
            f"{r['ira_mean']:.3f} | {r['dpo_loss_mean']:.3f} | "
            f"{r['aux_pen_mean']:.4f} |"
        )
    return header + "\n" + "\n".join(body_lines)


def format_per_seed_mmlu_drop_table(eval_rows: List[Dict[str, Any]]) -> str:
    """Render the §3.2 per-seed MMLU drop table from eval_results.csv rows."""
    conditions = sorted({r["condition"] for r in eval_rows})
    seeds = sorted({r["seed"] for r in eval_rows})
    header = "| seed | " + " | ".join(conditions) + " |"
    sep = "|" + "|".join(["---"] * (len(conditions) + 1)) + "|"
    body_lines = []
    for seed in seeds:
        cells = [str(seed)]
        for cond in conditions:
            match = [r for r in eval_rows if r["seed"] == seed and r["condition"] == cond]
            if match:
                cells.append(f"{match[0]['mmlu_drop_pct']:.2f}")
            else:
                cells.append("—")
        body_lines.append("| " + " | ".join(cells) + " |")
    # Mean row
    mean_cells = ["**mean**"]
    for cond in conditions:
        rows = [r for r in eval_rows if r["condition"] == cond]
        mean = _stats_mean([r["mmlu_drop_pct"] for r in rows])
        mean_cells.append(f"**{mean:.2f}**" if mean is not None else "—")
    body_lines.append("| " + " | ".join(mean_cells) + " |")
    return header + "\n" + sep + "\n" + "\n".join(body_lines)


def format_hypothesis_status_block(hyp: List[Tuple[str, str, str]]) -> str:
    """Render §3.3: one bullet per hypothesis with status + evidence."""
    return "\n".join(
        f"- **{h_id} {status}.** {evidence}"
        for h_id, status, evidence in hyp
    )


def format_provenance_outputs(file_hashes: Dict[str, str]) -> str:
    """Render the Provenance §6 'Outputs (with manifest sha256)' block."""
    return "\n".join(
        f"  - `results/{name}` \u2014 `{sha}`"
        for name, sha in sorted(file_hashes.items())
    )


# ---------------------------------------------------------------------------
# Pacific Time naming (per LAB_REPORT_STANDARD.md §1)
# ---------------------------------------------------------------------------

def utc_to_pacific(dt_utc: datetime) -> datetime:
    """Convert a UTC datetime to Pacific Time.

    Uses `zoneinfo.ZoneInfo("America/Los_Angeles")` so DST transitions are
    handled by the IANA tz database, not the previous month-boundary
    heuristic which got within a few days of correct on the March/November
    boundaries but was structurally fragile (and a v1.1 readiness item).
    Falls back to the heuristic only if `zoneinfo` is unavailable, which
    shouldn't happen on the supported Python 3.11+ baseline.
    """
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    if ZoneInfo is not None:
        return dt_utc.astimezone(ZoneInfo("America/Los_Angeles"))
    # Last-resort fallback (legacy behaviour).
    month = dt_utc.month
    offset = -7 if 3 <= month <= 10 else -8
    return dt_utc + timedelta(hours=offset)


# ---------------------------------------------------------------------------
# Sidecar loading: pod_provisioning.json + dispatcher.log discovery
# ---------------------------------------------------------------------------

def _discover_sidecar(artifacts_dir: Path, filename: str) -> Optional[Path]:
    """Find `filename` alongside the results dir, walking up one level.

    The dispatcher's staging layout puts `pod_provisioning.json` and
    `dispatcher.log` at the per-launch staging directory:

        <staging>/<launch>/
          ├── pod_provisioning.json   <- here
          ├── dispatcher.log          <- here
          └── results/
              ├── manifest.json
              └── ...

    `artifacts_dir` is the inner `results/` after auto-detection in
    extract().  We check that dir first (covers the case where a lab
    report has already absorbed the sidecar) and then walk one level up.
    """
    candidates = [artifacts_dir / filename, artifacts_dir.parent / filename]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_pod_provisioning(artifacts_dir: Path) -> Dict[str, Any]:
    """Return the `pod_provisioning.json` payload, or empty dict if absent.

    Empty dict is the backward-compat signal: runs from before the dispatcher
    wrote this sidecar do not produce it, so the caller falls back to whatever
    was passed via `--gpu-rate-usd-per-hour` and the hardware string parsed
    from run.log.
    """
    path = _discover_sidecar(artifacts_dir, "pod_provisioning.json")
    if path is None:
        return {}
    try:
        return json.loads(_read_text(path))
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Compute ledger
# ---------------------------------------------------------------------------

def read_last_ledger_row(ledger_text: str) -> Tuple[int, float, float, float]:
    """Return (row_number, wallclock_h, cumulative_h, remaining_h) for the most recent row.

    Raises if no rows are present (cold-start case handled by caller).
    """
    rows = re.findall(
        r"^\|\s*(\d+)\s*\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|",
        ledger_text,
        re.MULTILINE,
    )
    if not rows:
        raise RuntimeError("COMPUTE_LEDGER.md contains no ledger rows")
    last = rows[-1]
    return int(last[0]), float(last[1]), float(last[2]), float(last[3])


def render_ledger_row(
    *,
    row_n: int,
    date_utc: str,
    hardware_short: str,
    backend: str,
    description: str,
    wallclock_h: float,
    cumulative_h: float,
    remaining_h: float,
) -> str:
    return (
        f"| {row_n} | {date_utc} | {hardware_short} | {backend} | "
        f"{description} | {wallclock_h:.2f} | {cumulative_h:.2f} | {remaining_h:.2f} |"
    )


def update_ledger(
    ledger_path: Path,
    new_row_description: str,
    hardware: str,
    backend: str,
    wallclock_h: float,
    run_date_utc: str,
    dry_run: bool = False,
) -> Tuple[int, float, float]:
    """Append a row and refresh the 'Last updated' line.

    Returns (new_row_number, new_cumulative_h, new_remaining_h).
    Raises if cumulative would exceed the documented budget.
    """
    text = ledger_path.read_text()
    last_n, _, last_cum, _ = read_last_ledger_row(text)
    new_n = last_n + 1
    new_cum = round(last_cum + wallclock_h, 2)
    new_remaining = round(BUDGET_HOURS_PHASE_1_2 - new_cum, 2)

    if new_cum > BUDGET_HOURS_PHASE_1_2:
        print(
            f"!! WARNING: cumulative {new_cum} h exceeds Phase 1+2 budget "
            f"{BUDGET_HOURS_PHASE_1_2} h (per program \u00a77.3 this is a hard stop). "
            f"Escalate before committing.",
            file=sys.stderr,
        )

    new_row = render_ledger_row(
        row_n=new_n,
        date_utc=run_date_utc,
        hardware_short=hardware,
        backend=backend,
        description=new_row_description,
        wallclock_h=wallclock_h,
        cumulative_h=new_cum,
        remaining_h=new_remaining,
    )

    # Find the last existing row and append after it.
    lines = text.splitlines(keepends=True)
    last_row_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\|\s*\d+\s*\|.*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|", line):
            last_row_idx = i
    if last_row_idx is None:
        raise RuntimeError("could not locate last ledger row for append")
    lines.insert(last_row_idx + 1, new_row + "\n")

    # Update 'Last updated:' line.
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    new_text = "".join(lines)
    new_text = re.sub(
        r"^Last updated:.*$",
        f"Last updated: {now_utc} ({new_row_description}; {new_cum:.2f} h cumulative; "
        f"{new_remaining:.2f} h remaining of {BUDGET_HOURS_PHASE_1_2} h Phase 1+2 budget)",
        new_text,
        flags=re.MULTILINE,
    )

    if not dry_run:
        ledger_path.write_text(new_text)

    return new_n, new_cum, new_remaining


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_commit_for_pattern(pattern: str, cwd: Path = PROJECT_ROOT) -> Optional[str]:
    """Find the short hash of the most recent commit whose subject matches `pattern`."""
    try:
        out = subprocess.check_output(
            ["git", "log", "--grep", pattern, "-n", "1", "--pretty=format:%h"],
            cwd=cwd,
            text=True,
        ).strip()
        return out or None
    except subprocess.CalledProcessError:
        return None


# ---------------------------------------------------------------------------
# Reproducibility invocation
# ---------------------------------------------------------------------------

def build_reproducibility_invocation(
    benchmark: str,
    seeds: List[int],
    sweep_mode: bool,
    lambda_override: Optional[float],
) -> Tuple[str, str]:
    """Return (invocation_string, args_summary) for the Provenance section."""
    if sweep_mode:
        args = "--extra-args=--lambda_sweep"
    elif lambda_override is not None:
        args = f"--extra-args=\"--lambda_reg {lambda_override}\""
    else:
        args = ""
    seeds_str = " ".join(str(s) for s in seeds)
    invocation = (
        f"python3 scripts/run_on_runpod.py --benchmark {benchmark} \\\n"
        f"    --gpu-type \"NVIDIA A100-SXM4-80GB,NVIDIA A100 80GB PCIe,NVIDIA RTX 6000 Ada Generation\" \\\n"
        f"    --cloud SECURE {args}".strip()
    )
    return invocation, args


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_report(run: RunData, *, args) -> Tuple[str, Dict[str, Any]]:
    """Render the markdown report and the matching report_data.json payload."""
    config = run.config
    env = run.environment

    # Hypotheses / winner criteria are loaded from a per-benchmark meta file
    # if present; otherwise the report leaves those sections as TODO blocks
    # for the operator to fill in.  Format: ``benchmarks/<bench>/lab_report_meta.json``
    # with keys "predeclared_hypotheses" and "winner_criteria" (lists of strings).
    meta_path = PROJECT_ROOT / "benchmarks" / args.benchmark / "lab_report_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}
    hypotheses_block = "\n".join(
        f"- {h}" for h in meta.get("predeclared_hypotheses", ["<!-- AGENT: list H1..Hn with thresholds -->"])
    )
    winner_block = "\n".join(
        f"- {w}" for w in meta.get("winner_criteria", ["<!-- AGENT: list winner criteria from SPEC \u00a75 -->"])
    )

    # Aggregate + per-seed tables
    agg_rows = build_aggregate_rows(run.summary_rows, run.sweep_mode)
    agg_table = format_aggregate_table(agg_rows)
    per_seed_table = format_per_seed_mmlu_drop_table(run.eval_rows)

    # Hypothesis block
    hyp_block = format_hypothesis_status_block(run.hypothesis_status)

    # Provenance outputs
    prov_outputs = format_provenance_outputs(run.file_hashes)

    # Source-code commits via git log lookup
    runner_commit = git_commit_for_pattern(args.benchmark)
    eval_commit = git_commit_for_pattern("eval_capabilities|eval:")
    dispatcher_commit = git_commit_for_pattern("dispatcher")
    source_code_lines = []
    source_code_lines.append(f"  - `benchmarks/{args.benchmark}/run.py`"
                             + (f" (commit `{runner_commit}`)" if runner_commit else ""))
    source_code_lines.append(f"  - `benchmarks/{args.benchmark}/eval_capabilities.py`"
                             + (f" (commit `{eval_commit}`)" if eval_commit else ""))
    source_code_lines.append(f"  - `sm_src/synthetic_myelination/`")
    source_code_lines.append(f"  - `scripts/run_on_runpod.py`"
                             + (f" (commit `{dispatcher_commit}`)" if dispatcher_commit else ""))
    source_code_block = "\n".join(source_code_lines)

    # Wallclock and cost
    wallclock_h = round(run.wallclock_seconds / 3600.0, 2)
    cost_usd = round(wallclock_h * args.gpu_rate_usd_per_hour, 2)

    # Ledger update (or dry-run preview)
    if args.no_ledger:
        ledger_row = -1
        cum_h = -1.0
        remaining_h = -1.0
    else:
        ledger_row, cum_h, remaining_h = update_ledger(
            LEDGER_PATH,
            new_row_description=args.ledger_description or f"`{args.benchmark}` run",
            hardware=run.hardware + " (RunPod)",
            backend="CUDA",
            wallclock_h=wallclock_h,
            run_date_utc=run.wrapper_end_utc.strftime("%Y-%m-%d"),
            dry_run=args.dry_run,
        )

    # Reproducibility
    sweep_pairs_in_summary = {r["lambda_reg"] for r in run.summary_rows
                              if r["condition"] == "M_coupled_DPO"}
    lambda_override = None
    if len(sweep_pairs_in_summary) == 1:
        only_lam = next(iter(sweep_pairs_in_summary))
        # Distinguish "default 0.1 from CONDITIONS" from "override".  Without
        # extra info we can't tell, so we mark whichever value is non-default
        # as the override.  For the present runner the headline always passes
        # --lambda_reg explicitly when not in sweep mode.
        lambda_override = only_lam
    invocation, args_string = build_reproducibility_invocation(
        benchmark=args.benchmark,
        seeds=list(config.get("seeds", [])),
        sweep_mode=run.sweep_mode,
        lambda_override=None if run.sweep_mode else lambda_override,
    )

    # Title
    title = args.title or f"{args.benchmark} ({args.slug})"

    # Supersedes
    supersedes_line = ""
    if args.supersedes:
        supersedes_line = f"**Supersedes:** {args.supersedes}"

    # Patent claim path
    patent_path = f"benchmarks/{args.benchmark}/CONTINUATION_CLAIMS.md"

    # Substitutions
    subs: Dict[str, Any] = {
        "title": title,
        "experiment_id": f"benchmarks/{args.benchmark}/run.py {args_string}".strip(),
        "run_completion_utc": run.wrapper_end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "operator": args.operator,
        "status": args.status or run.decision,
        "supersedes_line": supersedes_line,
        "hypotheses_predeclared_block": hypotheses_block,
        "winner_criteria_block": winner_block,
        "model_id": config.get("model_id", ""),
        "phase_a_dataset": config.get("phase_a", {}).get("dataset_id", ""),
        "phase_a_n_examples": config.get("phase_a", {}).get("n_examples", ""),
        "phase_a_max_length": config.get("phase_a", {}).get("max_length", ""),
        "phase_b_dataset": config.get("phase_b", {}).get("dataset_id", ""),
        "phase_b_subset": config.get("phase_b", {}).get("dataset_subset", ""),
        "phase_b_n_train": config.get("phase_b", {}).get("n_train", ""),
        "phase_b_max_length": config.get("phase_b", {}).get("max_length", ""),
        "phase_b_batch_size": config.get("phase_b", {}).get("batch_size", ""),
        "phase_b_grad_accum": config.get("phase_b", {}).get("grad_accum_steps", ""),
        "phase_b_n_steps": config.get("phase_b", {}).get("n_steps", ""),
        "phase_b_lr": config.get("phase_b", {}).get("lr", ""),
        "lora_r": config.get("lora_config", {}).get("r", ""),
        "lora_alpha": config.get("lora_config", {}).get("lora_alpha", ""),
        "lora_dropout": config.get("lora_config", {}).get("lora_dropout", ""),
        "lora_targets": ", ".join(config.get("lora_config", {}).get("target_modules", [])),
        "lambda_override": lambda_override if lambda_override is not None else "n/a",
        "conditions_list": ", ".join(config.get("conditions", [])),
        "seeds": json.dumps(config.get("seeds", [])),
        "backend": "RunPod SECURE cloud",
        "hardware": run.hardware,
        "cloud_type": "SECURE",
        "torch_version": env.get("torch_version", ""),
        "max_mmlu": config.get("eval_caps", {}).get("max_mmlu", ""),
        "max_arc": config.get("eval_caps", {}).get("max_arc", ""),
        "max_truthful_qa": config.get("eval_caps", {}).get("max_truthful_qa", ""),
        "n_unique_seeds": len({r["seed"] for r in run.summary_rows}),
        "base_mmlu": f"{run.base_evals['mmlu']:.3f}",
        "base_arc": f"{run.base_evals['arc']:.3f}",
        "base_tqa": f"{run.base_evals['tqa']:.3f}",
        "aggregate_table": agg_table,
        "per_seed_mmlu_drop_table": per_seed_table,
        "hypothesis_status_block": hyp_block,
        "decision": run.decision,
        "source_code_block": source_code_block,
        "provenance_outputs_block": prov_outputs,
        "wallclock_hours": f"{wallclock_h:.2f}",
        "cost_usd": f"{cost_usd:.2f}",
        "gpu_rate_usd_per_hour": f"{args.gpu_rate_usd_per_hour:.2f}",
        "cumulative_hours": f"{cum_h:.2f}",
        "budget_hours": f"{BUDGET_HOURS_PHASE_1_2:.1f}",
        "remaining_hours": f"{remaining_h:.2f}",
        "ledger_row": str(ledger_row),
        "reproducibility_args": args_string or "(no extra args)",
        "reproducibility_invocation": invocation,
        "patent_claims_path": patent_path,
    }

    # Render template
    template = _read_text(TEMPLATE_PATH)
    rendered = template
    for key, val in subs.items():
        rendered = rendered.replace("{{" + key + "}}", str(val))

    # Detect un-substituted placeholders (these are bugs in the template / extractor).
    unfilled = re.findall(r"\{\{[a-zA-Z_]+\}\}", rendered)
    if unfilled:
        raise RuntimeError(
            f"Template has unfilled placeholders: {sorted(set(unfilled))}. "
            f"Add them to `subs` in render_report() or update the template."
        )

    # Build the matching report_data.json (everything the verifier needs).
    report_data = {
        "experiment": run.experiment,
        "benchmark": args.benchmark,
        "slug": args.slug,
        "operator": args.operator,
        "status": subs["status"],
        "decision": run.decision,
        "run_completion_utc": subs["run_completion_utc"],
        "wallclock_seconds": run.wallclock_seconds,
        "wallclock_hours": wallclock_h,
        "cost_usd": cost_usd,
        "gpu_rate_usd_per_hour": args.gpu_rate_usd_per_hour,
        "model_id": config.get("model_id", ""),
        "hardware": run.hardware,
        "seeds": list(config.get("seeds", [])),
        "n_unique_seeds": subs["n_unique_seeds"],
        "base_evals": run.base_evals,
        "aggregate_rows": agg_rows,
        "eval_rows": run.eval_rows,
        "hypothesis_status": [
            {"id": h, "status": s, "evidence": e} for h, s, e in run.hypothesis_status
        ],
        "manifest_sha256": run.file_hashes,
        "source_commits": {
            "runner": runner_commit,
            "eval_capabilities": eval_commit,
            "dispatcher": dispatcher_commit,
        },
        "ledger": {
            "row": ledger_row,
            "cumulative_hours": cum_h,
            "remaining_hours": remaining_h,
            "budget_hours": BUDGET_HOURS_PHASE_1_2,
        },
        "sweep_mode": run.sweep_mode,
        "lambda_override": lambda_override,
    }

    return rendered, report_data


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

def lab_report_paths(run: RunData, slug: str) -> Tuple[Path, str]:
    """Compute the lab-report directory and filename per LAB_REPORT_STANDARD \u00a71."""
    pt = utc_to_pacific(run.wrapper_end_utc)
    date_str = pt.strftime("%Y-%m-%d")
    hhmm = pt.strftime("%H%M")
    folder = f"{date_str}_{hhmm}_{slug}"
    return LAB_REPORTS / folder, folder + ".md"


def setup_report_dir(
    report_dir: Path,
    run: RunData,
    dispatcher_log: Optional[Path],
    dry_run: bool,
    source_action: str = "copy",
) -> None:
    """Stage the run artifacts into ``<report_dir>/results/``.

    `source_action` controls how the artifacts are transferred:

    * ``"copy"``  — (legacy) shutil.copy2 each known artifact into the
      report folder.  Leaves the source dir intact; useful for retries
      and dry-runs.
    * ``"move"``  — (default for new runs) rename the entire `results/`
      directory into ``<report_dir>/results/`` and move any sibling
      sidecars (``dispatcher.log``, ``pod_provisioning.json``) alongside
      it.  Atomic on the same filesystem; eliminates the
      copy-into-home-then-copy-into-lab-report two-step that the
      ``lab_reports/_staging/`` directory was introduced to remove.
    * ``"leave"`` — do not touch the source.  The report references
      results by `run.results_dir` (manual workflow / debugging).
    """
    if dry_run:
        print(f"  [dry-run] ({source_action}) would create {report_dir}")
        return

    if source_action == "leave":
        report_dir.mkdir(parents=True, exist_ok=True)
        return

    if source_action == "move":
        # rename the whole results/ folder into report_dir/results/.  This
        # is atomic when the staging dir and lab_reports/ live on the same
        # filesystem (the common case under PROJECT_ROOT/lab_reports/_staging/).
        # Falls back to copy+remove if the rename crosses devices, which is
        # also what `shutil.move` does internally.
        report_dir.mkdir(parents=True, exist_ok=True)
        dest_results = report_dir / "results"
        if dest_results.exists():
            raise RuntimeError(
                f"refusing to move into existing {dest_results}; either pass "
                f"--source-action copy or remove the destination first"
            )
        shutil.move(str(run.results_dir), str(dest_results))
        # Move sidecars (dispatcher.log, pod_provisioning.json) sitting at
        # the per-launch staging level.  After the move above, run.results_dir
        # no longer exists — but its parent (the staging launch dir) still
        # holds the sidecars.
        staging_parent = run.results_dir.parent
        for sidecar in ("dispatcher.log", "pod_provisioning.json"):
            src = staging_parent / sidecar
            if src.exists() and not (dest_results / sidecar).exists():
                shutil.move(str(src), str(dest_results / sidecar))
        # If the dispatcher staged this launch under its own dir, try to
        # remove the now-empty parent so the staging area doesn't accumulate
        # cruft.  We swallow errors because a non-empty dir is fine to leave.
        try:
            staging_parent.rmdir()
        except OSError:
            pass
        # If the operator passed a --dispatcher-log override, copy it in too.
        if dispatcher_log and dispatcher_log.exists() and not (dest_results / "dispatcher.log").exists():
            shutil.copy2(dispatcher_log, dest_results / "dispatcher.log")
        return

    # source_action == "copy" (legacy behaviour, preserved verbatim)
    (report_dir / "results").mkdir(parents=True, exist_ok=True)
    for name in (
        "dpo_curves.csv", "eval_results.csv", "events.csv",
        "summary_table.csv", "results_summary.txt", "results_summary.json",
        "manifest.json", "receipt.txt", "run.log",
        # New-format sidecars from the v1.1+ dispatcher's staging layout.
        # Silently skipped on older runs that didn't produce them.
        "pod_provisioning.json",
    ):
        src = run.results_dir / name
        if src.exists():
            shutil.copy2(src, report_dir / "results" / name)
    # Sentinel files: copy any _STARTED.* / _FINISHED.* / _DONE.* found.
    for sentinel in run.results_dir.glob("_STARTED.*"):
        shutil.copy2(sentinel, report_dir / "results" / sentinel.name)
    for sentinel in run.results_dir.glob("_FINISHED.*"):
        shutil.copy2(sentinel, report_dir / "results" / sentinel.name)
    for sentinel in run.results_dir.glob("_DONE.*"):
        shutil.copy2(sentinel, report_dir / "results" / sentinel.name)
    # Dispatcher log: prefer explicit override, otherwise auto-detect one
    # level above results_dir (the dispatcher's staging layout writes
    # dispatcher.log at the per-launch level, not inside results/).
    auto_disp = _discover_sidecar(run.results_dir, "dispatcher.log")
    chosen_disp = dispatcher_log if (dispatcher_log and dispatcher_log.exists()) else auto_disp
    if chosen_disp and chosen_disp.exists() and not (report_dir / "results" / "dispatcher.log").exists():
        shutil.copy2(chosen_disp, report_dir / "results" / "dispatcher.log")
    # Sibling pod_provisioning.json at the staging launch level.
    auto_prov = _discover_sidecar(run.results_dir, "pod_provisioning.json")
    if auto_prov and auto_prov.exists() and not (report_dir / "results" / "pod_provisioning.json").exists():
        shutil.copy2(auto_prov, report_dir / "results" / "pod_provisioning.json")
    # Optional sweep_winner.txt (sweep runs only)
    sweep_winner = run.results_dir / "sweep_winner.txt"
    if sweep_winner.exists():
        shutil.copy2(sweep_winner, report_dir / "results" / "sweep_winner.txt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Local results directory from the dispatcher (contains manifest.json, run.log, etc.)")
    parser.add_argument("--benchmark", required=True,
                        help="Benchmark dir name under benchmarks/ (e.g. 05_m_coupled_kl_rlhf)")
    parser.add_argument("--slug", required=True,
                        help="kebab-case slug for the report filename and folder (per LAB_REPORT_STANDARD \u00a71)")
    parser.add_argument("--operator", required=True,
                        help="Operator string per LAB_REPORT_STANDARD \u00a73.7; e.g. agent:oz-warp-2026-05-13")
    parser.add_argument("--gpu-rate-usd-per-hour", type=float, default=None,
                        help="Per-hour cost of the GPU used.  If unset, the rate is "
                             "read from the dispatcher's pod_provisioning.json sidecar "
                             "(cost_per_hr_usd, recorded by run_on_runpod.py at pod "
                             "creation time).  Pass this flag to override the recorded "
                             "rate, or to supply a rate for older runs that pre-date "
                             "the sidecar.  Last-resort default is 1.50 USD/h.")
    parser.add_argument("--dispatcher-log", type=Path, default=None,
                        help="Optional path to the laptop-side dispatcher log to bundle into the report. "
                             "If omitted, finalize_run looks for `dispatcher.log` alongside the results/ "
                             "directory (the dispatcher's staging layout) and bundles it automatically.")
    parser.add_argument("--source-action", choices=["copy", "move", "leave"], default="move",
                        help="How to transfer the source results into the lab report folder. "
                             "`move` (default) renames the staged results/ directory into "
                             "<lab_report>/results/ and is atomic on the same filesystem. "
                             "`copy` leaves the source dir intact (legacy, useful for retries "
                             "against shared results, and for processing older runs whose "
                             "results live outside the staging tree).  `leave` does not touch "
                             "the source at all (manual workflows / debugging).")
    parser.add_argument("--force-smoke-report", action="store_true",
                        help="Allow finalizing a smoke-config run into a real lab report. "
                             "Smoke runs (manifest.config.smoke == true, or pod_provisioning.smoke == true, "
                             "or results-dir under `_smokes/`) are otherwise refused because their "
                             "hypothesis statuses come from tiny configs and don't reflect the "
                             "full SPEC-declared evaluation.")
    parser.add_argument("--title", default=None,
                        help="Override the report's H1 title (default: '<benchmark> (<slug>)').")
    parser.add_argument("--status", default=None,
                        help="Override the status enum (default: parsed from results_summary.txt Decision).")
    parser.add_argument("--supersedes", default=None,
                        help="If this report supersedes a prior one, give its filename.")
    parser.add_argument("--ledger-description", default=None,
                        help="Description string for the new COMPUTE_LEDGER.md row.")
    parser.add_argument("--no-ledger", action="store_true",
                        help="Skip COMPUTE_LEDGER.md update (use for sweep runs that share a phase's ledger row, or for retries).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print intended actions but don't touch disk.")
    args = parser.parse_args()

    # 1. Extract
    print(f"==> Extracting run data from {args.results_dir}", flush=True)
    run = extract(args.results_dir)
    print(f"    experiment   : {run.experiment}")
    print(f"    decision     : {run.decision}")
    print(f"    wallclock    : {run.wallclock_seconds/3600:.2f} h")
    print(f"    hardware     : {run.hardware}")
    print(f"    sweep_mode   : {run.sweep_mode}")
    print(f"    seeds        : {run.config.get('seeds', [])}")
    print(f"    H statuses   : {[(h,s) for h,s,_ in run.hypothesis_status]}")

    # 1a. Smoke guard.  Reject smoke configs unless explicitly forced; a
    # smoke result is by definition statistically meaningless against
    # SPEC-declared hypothesis thresholds and writing a real lab report
    # from one would taint the headline-tracking pipeline.
    is_smoke = bool(run.config.get("smoke", False))
    provisioning = load_pod_provisioning(run.results_dir)
    if provisioning.get("smoke") is True:
        is_smoke = True
    # Treat anything staged under .../_smokes/... as smoke even if the
    # config flag is missing (defensive: hand-staged runs sometimes lack
    # the manifest's smoke=true if they were re-tarred from an older run).
    if any(part == "_smokes" for part in args.results_dir.resolve().parts):
        is_smoke = True
    if is_smoke and not args.force_smoke_report:
        print(
            "!! Refusing to finalize a smoke-config run into a lab report.\n"
            "   Reason: hypothesis statuses from a smoke result do not reflect the \n"
            "   SPEC's evaluation caps; promoting one into the lab-report tree would \n"
            "   contaminate the headline-tracking provenance.  Pass --force-smoke-report\n"
            "   if you intend to record a smoke result for archival / debugging.",
            file=sys.stderr,
        )
        sys.exit(2)

    # 1b. Resolve the GPU hourly rate.  Source-of-truth precedence:
    #   1. --gpu-rate-usd-per-hour CLI flag (explicit override)
    #   2. pod_provisioning.json:cost_per_hr_usd (recorded by the dispatcher)
    #   3. 1.50 USD/h fallback with a loud warning
    if args.gpu_rate_usd_per_hour is not None:
        rate_source = "--gpu-rate-usd-per-hour CLI override"
        resolved_rate = args.gpu_rate_usd_per_hour
    elif provisioning.get("cost_per_hr_usd") is not None:
        rate_source = "pod_provisioning.json:cost_per_hr_usd"
        resolved_rate = float(provisioning["cost_per_hr_usd"])
    else:
        rate_source = "DEFAULT fallback (no provisioning sidecar, no CLI override)"
        resolved_rate = 1.50
        print(
            "!! No GPU rate available from pod_provisioning.json AND no "
            "--gpu-rate-usd-per-hour override; falling back to 1.50 USD/h. "
            "This is almost certainly wrong; pass --gpu-rate-usd-per-hour to fix.",
            file=sys.stderr,
        )
    args.gpu_rate_usd_per_hour = resolved_rate
    print(f"    gpu rate     : {resolved_rate} USD/h  ({rate_source})")

    # 2. Layout
    report_dir, report_md_name = lab_report_paths(run, args.slug)
    report_md_path = report_dir / report_md_name
    report_json_path = report_dir / "report_data.json"
    print(f"==> Lab report folder: {report_dir}")
    print(f"    report file       : {report_md_name}")

    if not args.dry_run and report_dir.exists():
        # Don't blindly overwrite a prior hand-written report.
        if report_md_path.exists():
            print(
                f"!! {report_md_path} already exists.  Move it aside or use --dry-run to preview.",
                file=sys.stderr,
            )
            sys.exit(1)

    setup_report_dir(
        report_dir, run, args.dispatcher_log, args.dry_run,
        source_action=args.source_action,
    )

    # 3. Render
    print(f"==> Rendering report and report_data.json")
    rendered_md, report_data = render_report(run, args=args)

    if args.dry_run:
        print("--- begin rendered markdown (dry-run preview) ---")
        print(rendered_md)
        print("--- end rendered markdown ---")
        print("\n--- report_data.json (dry-run preview) ---")
        print(json.dumps(report_data, indent=2, default=str)[:2000] + "  ...")
        return

    report_md_path.write_text(rendered_md)
    report_json_path.write_text(json.dumps(report_data, indent=2, default=str))
    print(f"    wrote {report_md_path}")
    print(f"    wrote {report_json_path}")

    print()
    print("=" * 70)
    print("Next steps:")
    print(f"  1. Edit {report_md_name} and fill in every <!-- AGENT: ... --> block.")
    print(f"  2. Run: python3 scripts/verify_report.py {report_dir}")
    print("  3. git add lab_reports/<...>/ COMPUTE_LEDGER.md && git commit")
    print("=" * 70)


if __name__ == "__main__":
    main()
