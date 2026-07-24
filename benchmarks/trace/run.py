#!/usr/bin/env python3
"""Experiment 1 harness -- does the asymptotic soft lock do anything?

Compares, on a continual-learning task sequence, the Phase-1 isolation-screen
mechanism arms (gradient_aligned_m WITH asymptotic soft lock vs
raw_fisher_signal WITHOUT lock, identical importance signal, task-boundary
M build) against the full mandatory-baseline set (vanilla, ewc,
elastic_reset, model_averaging, replay).

Phase-1 dependent variable: **BWT under a uniform next-token-accuracy (NTA)
proxy** (pad-masked). Not task-faithful TRACE metrics — those are Phase 2.
See eval/metrics.py METRIC_SCOPE and docs/design_notes/01_temporal_fidelity.md.

Also registered (not in the default Exp-1 matrix): gradient_aligned_m_online,
the Phase-2 continuous on_step consolidation arm.

Two run modes:

  --smoke   Tiny synthetic task sequence on a tiny GPT-2-ish model, CPU-safe,
            single seed by default. PREFLIGHT ONLY -- proves the harness
            logic, the arm interface, and the metric wiring. Never evidence.

  (default) Real TRACE task sequence via HuggingFace + a real base model on
            GPU. n>=5 seeds, full lambda sweep. This is the run of record and
            is dispatched through ../../infra on cloud GPUs.

  --tune-baselines  Short 1-seed / 2-task pre-study sweeping weight-space
            knobs (rho, alpha). Writes chosen values; does not run the matrix.

The harness NEVER special-cases an arm: every arm is driven through the
baselines.base.Arm interface, which is what enforces "only the lock varies."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Local layout:  benchmarks/trace/            -> two levels up to Synthetic2/, sm_src sibling there.
# Remote layout: /workspace/synth/trace/      -> one level up to /workspace/synth/, sm_src sibling there
#                (the GPU dispatcher lays sm_src + the benchmark dir side-by-side under REMOTE_BASE).
# eval/ and baselines/ are VENDORED into this benchmark dir so they travel with
# it when the GPU dispatcher tar-pushes benchmarks/<bench>/ to the pod (the
# dispatcher pushes only sm_src/ + the benchmark dir). Adding this dir to the
# path makes the vendored packages import as top-level modules, identically
# whether run locally (benchmarks/trace/) or on the pod (cd REMOTE_BASE/trace).
HERE = Path(__file__).resolve().parent            # benchmarks/trace  (local) or REMOTE_BASE/trace (remote)
ROOT = Path(__file__).resolve().parents[2]        # Synthetic2/       (local only)
BENCH_PARENT = Path(__file__).resolve().parents[1]  # benchmarks/ (local) or REMOTE_BASE/ (remote)
_candidates = [
    HERE,                           # vendored eval/ + baselines/ + trace_data + arms_mechanism
    ROOT / "sm_src",                # local:  Synthetic2/sm_src
    BENCH_PARENT / "sm_src",        # remote: REMOTE_BASE/sm_src
    ROOT,                           # local:  project root (harmless if absent remotely)
]
for p in _candidates:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval.metrics import (                                         # noqa: E402
    summarize,
    mean_std,
    exceeds_2sigma,
    matched_plasticity,
    best_lambda_run,
    METRIC_SCOPE,
    METRIC_SCOPE_NOTE,
)
from baselines import (                                            # noqa: E402
    ARM_REGISTRY as BASELINE_ARMS,
    DegenerateSignalError,
    DEFAULT_ELASTIC_RESET_RHO,
    DEFAULT_MODEL_AVERAGING_ALPHA,
)
from arms_mechanism import ARM_REGISTRY as MECH_ARMS               # noqa: E402

ALL_ARMS = {**BASELINE_ARMS, **MECH_ARMS}

#: Arms reported in Experiment 1 Phase-1 isolation screen + mandatory baselines.
#: gradient_aligned_m_online is registered but not in the default matrix
#: (Phase-2 continuous consolidation; enable via --arms).
EXP1_ARMS = [
    "gradient_aligned_m",   # M + asymptotic soft lock (task-boundary isolation screen)
    "raw_fisher_signal",    # same signal, no lock (the isolating control)
    "ewc",                  # classical cumulative Fisher floor
    "elastic_reset",        # weight-space (rho fixed after --tune-baselines)
    "model_averaging",      # weight-space (alpha fixed after --tune-baselines)
    "replay",               # rehearsal
    "vanilla",              # sanity floor
]

DEFAULT_SEEDS = [0, 1, 2, 3, 4]   # n>=5 per README

# Weight-space knobs fixed after pre-study; not swept in the main matrix.
_BASELINE_TUNING = {
    "elastic_reset_rho": DEFAULT_ELASTIC_RESET_RHO,
    "model_averaging_alpha": DEFAULT_MODEL_AVERAGING_ALPHA,
}


# ---------------------------------------------------------------------------
# Task sequence abstraction
# ---------------------------------------------------------------------------

def build_task_sequence(mode: str, tokenizer=None, n_tasks: int = 5, seed: int = 0,
                        data_root=None, batch_size: int = 8, max_len: int = 512):
    """Return a list of tasks; each task = (train_loader, eval_loader).

    smoke: synthetic permuted-token sequences (self-contained, CPU-safe).
    trace: the real 8-task TRACE sequence in canonical order.
    """
    if mode == "smoke":
        return _synthetic_sequence(n_tasks=n_tasks, seed=seed)
    elif mode == "trace":
        return _trace_sequence(tokenizer=tokenizer, data_root=data_root,
                               batch_size=batch_size, max_len=max_len)
    raise ValueError(f"unknown mode {mode!r}")


def _synthetic_sequence(n_tasks: int, seed: int, vocab: int = 64,
                        seq_len: int = 16, n_train: int = 64, n_eval: int = 32,
                        batch: int = 8):
    """Distinct next-token tasks: each task uses a disjoint token sub-alphabet
    with a fixed deterministic successor rule, so a model CAN learn each and
    CAN forget it. Enough structure to exercise BWT without any downloads.
    """
    import torch

    g = torch.Generator().manual_seed(seed)
    band = vocab // n_tasks

    def make_task(t):
        lo = t * band
        hi = lo + band
        def gen(n):
            # sequences over [lo, hi); label = shift-by-1 within band
            xs = torch.randint(lo, hi, (n, seq_len), generator=g)
            return xs
        train = gen(n_train)
        ev = gen(n_eval)
        def loader(data):
            idx = 0
            batches = []
            while idx < data.shape[0]:
                batches.append(data[idx:idx + batch])
                idx += batch
            return batches
        return loader(train), loader(ev)

    return [make_task(t) for t in range(n_tasks)]


def _trace_sequence(tokenizer, data_root, batch_size: int = 8, max_len: int = 512):
    """Real 8-task TRACE sequence via the official JSON task folders.

    Stages the official TRACE archive on first use (idempotent). On the pod the
    data isn't shipped in the code tar -- fetch_data.stage_trace() pulls it from
    the official Google Drive release and normalizes it under <data_root>.
    """
    from pathlib import Path
    from trace_data import build_trace_sequence
    from fetch_data import stage_trace
    if tokenizer is None:
        raise ValueError("trace mode requires a tokenizer")
    if data_root is None:
        # Default to a data/ dir NEXT TO this benchmark (HERE/data/TRACE), which
        # is writable both locally (benchmarks/trace/data) and on the pod
        # (REMOTE_BASE/trace/data). ROOT/parents[2] is unreliable on the pod.
        data_root = HERE / "data" / "TRACE"
    data_root = Path(data_root)
    # Idempotent fetch-on-pod (and locally if absent); fails loud on network error.
    # stage_trace() appends its own TRACE/ segment and returns the real staged
    # dir (…/TRACE). build_trace_sequence() reads <root>/<task> directly, so it
    # must receive that returned path, NOT the pre-stage data_root (otherwise the
    # TRACE segment is doubled: data/TRACE/TRACE vs the expected data/TRACE).
    staged_root = stage_trace(data_root)
    return build_trace_sequence(
        tokenizer=tokenizer,
        data_root=staged_root,
        batch_size=batch_size,
        max_len=max_len,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(mode: str, vocab: int = 64):
    """Tiny causal LM for smoke; real base model for trace (GPU path)."""
    import torch
    import torch.nn as nn

    if mode == "smoke":
        class TinyLM(nn.Module):
            """Minimal GPT-ish LM: embed -> 1 linear block -> tie head.

            Exposes the HF-style forward(input_ids, labels=...) -> .loss/.logits
            interface the sm_src hooks and Fisher estimator expect.
            """
            def __init__(self, vocab, d=32):
                super().__init__()
                self.emb = nn.Embedding(vocab, d)
                self.block = nn.Linear(d, d)
                self.head = nn.Linear(d, vocab)

            def forward(self, input_ids, labels=None, attention_mask=None):
                h = self.emb(input_ids)
                h = torch.tanh(self.block(h))
                logits = self.head(h)
                out = type("O", (), {})()
                out.logits = logits
                if labels is not None:
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss = nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                    out.loss = loss
                return out

        return TinyLM(vocab)

    raise NotImplementedError(
        "For trace mode call build_real_model() in trace_data.py directly; "
        "the orchestrator wires model + tokenizer together (see run())."
    )


# ---------------------------------------------------------------------------
# Core: train one arm over the sequence, return the T x T accuracy matrix
# ---------------------------------------------------------------------------

def evaluate(model, eval_loader, pad_token_id=None) -> float:
    """Pad-masked next-token accuracy (Phase-1 NTA proxy).

    Pad positions never count toward correct or total. This is intentionally
    NOT task-faithful TRACE scoring (see METRIC_SCOPE).
    """
    import torch
    from synthetic_myelination import unpack_lm_batch

    model.eval()
    correct = 0
    total = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in eval_loader:
            ids, labels, attention_mask = unpack_lm_batch(
                batch, device, pad_token_id=pad_token_id,
            )
            try:
                logits = model(ids, attention_mask=attention_mask).logits
            except TypeError:
                logits = model(ids).logits
            pred = logits[:, :-1, :].argmax(-1)
            gold = labels[:, 1:]
            # Valid where gold is not ignore_index (-100)
            valid = gold != -100
            correct += ((pred == gold) & valid).sum().item()
            total += valid.sum().item()
    return correct / total if total else 0.0


def run_arm_over_sequence(
    arm, model, sequence, lr: float, epochs: int, pad_token_id=None,
) -> Dict:
    """Train `arm` sequentially over all tasks; fill NTA-proxy matrix R.

    R[i][j] = pad-masked next-token accuracy on task j after finishing task i.
    """
    import torch
    from synthetic_myelination import unpack_lm_batch

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    T = len(sequence)
    R = [[0.0] * T for _ in range(T)]
    step_i = 0

    arm.on_train_start(model)
    prev_loader = None
    for i, (train_loader, _eval_loader) in enumerate(sequence):
        arm.on_task_start(model, i, prev_loader)
        model.train()
        for _ep in range(epochs):
            for batch in train_loader:
                opt.zero_grad(set_to_none=True)
                device = next(model.parameters()).device
                ids, labels, attention_mask = unpack_lm_batch(
                    batch, device, pad_token_id=pad_token_id,
                )
                try:
                    base = model(
                        ids, labels=labels, attention_mask=attention_mask,
                    ).loss
                except TypeError:
                    base = model(ids, labels=labels).loss
                aux = arm.aux_loss(model)
                loss = base if aux is None else (base + aux)
                loss.backward()
                opt.step()
                arm.on_step(model, step_i, batch)
                step_i += 1
        arm.on_task_end(model, i, train_loader)
        prev_loader = train_loader
        # evaluate on all tasks seen so far (and future = 0 by convention)
        for j, (_tl, ev) in enumerate(sequence):
            R[i][j] = evaluate(model, ev, pad_token_id=pad_token_id)

    arm.on_train_end(model)
    tele = arm.telemetry()
    # Top-level signal-health fields ride in the cell checkpoint so go/kill
    # analysis survives interruption (not buried only inside telemetry).
    return {
        "R": R,
        "metrics": summarize(R),
        "telemetry": tele,
        "metric_scope": METRIC_SCOPE,
        "signal_health": tele.get("signal_health", "ok"),
        "m_distribution_warning": tele.get("m_distribution_warning"),
        "m_distribution_warnings": tele.get("m_distribution_warnings", []),
        "fisher_health_log": tele.get("fisher_health_log", []),
        "importance_kind": tele.get("importance_kind"),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(mode: str, arms: List[str], seeds: List[int], lambdas: List[float],
        n_tasks: int, lr: float, epochs: int,
        model_id: str = None, data_root=None,
        batch_size: int = 8, max_len: int = 512, ckpt_dir=None,
        baseline_tuning: Dict = None) -> Dict:
    """Run the arms x seeds x lambdas matrix.

    Robustness (added 2026-07-24):
      1. Model + data built ONCE; pristine weights snapshotted to CPU and
         restored before each cell (was: full 3B reload per cell, 175x).
      2. Each completed cell checkpointed immediately to
         ckpt_dir/cell_<lam>_<arm>_<seed>.json + progress.jsonl, so an
         interrupted run loses at most the in-flight cell (was: lost all).
      3. On restart, checkpointed cells are loaded and skipped (resume).
         Flushed per-cell progress prints give live timing.
      4. Degenerate M/Fisher aborts only the cell (checkpoint + continue).
    """
    import copy
    import time
    import torch

    tuning = dict(_BASELINE_TUNING)
    if baseline_tuning:
        tuning.update(baseline_tuning)

    results = {a: {"lambda_runs": {}} for a in arms}

    ckpt_path = Path(ckpt_dir) if ckpt_dir else None
    if ckpt_path:
        ckpt_path.mkdir(parents=True, exist_ok=True)
        progress_log = ckpt_path / "progress.jsonl"
    else:
        progress_log = None

    def _cell_file(lam, arm_name, seed):
        return ckpt_path / f"cell_{lam}_{arm_name}_{seed}.json" if ckpt_path else None

    def _load_cell(lam, arm_name, seed):
        f = _cell_file(lam, arm_name, seed)
        if f and f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:
                return None
        return None

    # ---- FIX 2: build model + data ONCE -----------------------------------
    pad_token_id = None
    if mode == "trace":
        from trace_data import build_real_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[run] building base model ONCE (model_id={model_id}, device={device})", flush=True)
        model, tokenizer = build_real_model(model_id, device)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        sequence = build_task_sequence(
            mode, tokenizer=tokenizer, data_root=data_root,
            batch_size=batch_size, max_len=max_len)
    else:
        model = build_model(mode)
        tokenizer = None
        sequence = build_task_sequence(mode, n_tasks=n_tasks, seed=0)

    # Pristine weight snapshot (CPU) restored before each cell so every
    # (arm, seed) trains from identical init without reloading from disk.
    pristine = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    print(f"[run] snapshotted {len(pristine)} tensors as pristine init", flush=True)

    total_cells = len(lambdas) * len(arms) * len(seeds)
    done = 0
    t_matrix = time.time()

    for lam in lambdas:
        for arm_name in arms:
            per_seed_metrics = []
            for seed in seeds:
                done += 1
                cached = _load_cell(lam, arm_name, seed)
                if cached is not None:
                    print("[cell %d/%d] SKIP (checkpointed) lam=%s arm=%s seed=%s"
                          % (done, total_cells, lam, arm_name, seed), flush=True)
                    per_seed_metrics.append(cached)
                    continue

                model.load_state_dict(pristine, strict=True)
                torch.manual_seed(seed)

                t0 = time.time()
                print("[cell %d/%d] START lam=%s arm=%s seed=%s"
                      % (done, total_cells, lam, arm_name, seed), flush=True)

                ArmCls = ALL_ARMS[arm_name]
                arm = _construct_arm(
                    ArmCls, lam, seed,
                    pad_token_id=pad_token_id,
                    baseline_tuning=tuning,
                )
                try:
                    out = run_arm_over_sequence(
                        arm, model, sequence, lr, epochs,
                        pad_token_id=pad_token_id,
                    )
                except DegenerateSignalError as exc:
                    dt = time.time() - t0
                    details = dict(getattr(exc, "details", {}) or {})
                    # Best-effort telemetry from the arm (warnings accumulated
                    # before the hard fail) so the cell JSON keeps go/kill
                    # diagnostics even on abort.
                    try:
                        tele = arm.telemetry()
                    except Exception:
                        tele = {}
                    m_warns = (
                        tele.get("m_distribution_warnings")
                        or details.get("m_distribution_warnings")
                        or []
                    )
                    last_m_warn = tele.get("m_distribution_warning")
                    if last_m_warn is None and m_warns:
                        tail = m_warns[-1]
                        last_m_warn = (
                            tail.get("warning") if isinstance(tail, dict) else tail
                        )
                    out = {
                        "degenerate": True,
                        "signal_health": "degenerate",
                        "degenerate_message": str(exc),
                        "degenerate_details": details,
                        "m_distribution_warning": last_m_warn,
                        "m_distribution_warnings": m_warns,
                        "fisher_health_log": (
                            tele.get("fisher_health_log")
                            or details.get("fisher_health_log")
                            or []
                        ),
                        "importance_kind": tele.get("importance_kind") or details.get("kind"),
                        "R": None,
                        "metrics": {
                            "bwt": float("nan"),
                            "acc": float("nan"),
                            "target_task_acc": float("nan"),
                            "learning_acc": float("nan"),
                            "n_tasks": float(len(sequence)),
                            "metric_scope": METRIC_SCOPE,
                        },
                        "telemetry": tele,
                        "metric_scope": METRIC_SCOPE,
                        "_meta": {
                            "lam": lam, "arm": arm_name, "seed": seed,
                            "seconds": round(dt, 1), "degenerate": True,
                        },
                    }
                    f = _cell_file(lam, arm_name, seed)
                    if f:
                        f.write_text(json.dumps(out, indent=2))
                        if progress_log:
                            with open(progress_log, "a") as pl:
                                pl.write(json.dumps({
                                    "lam": lam, "arm": arm_name, "seed": seed,
                                    "seconds": round(dt, 1),
                                    "degenerate": True,
                                    "signal_health": "degenerate",
                                    "message": str(exc),
                                    "m_distribution_warning": out.get(
                                        "m_distribution_warning"
                                    ),
                                }) + "\n")
                    print("[cell %d/%d] DEGENERATE lam=%s arm=%s seed=%s  %s  %.1fs"
                          % (done, total_cells, lam, arm_name, seed, exc, dt),
                          flush=True)
                    per_seed_metrics.append(out)
                    continue

                dt = time.time() - t0
                out["_meta"] = {"lam": lam, "arm": arm_name, "seed": seed,
                                "seconds": round(dt, 1)}
                # ---- FIX 1: checkpoint this cell immediately --------------
                f = _cell_file(lam, arm_name, seed)
                if f:
                    f.write_text(json.dumps(out, indent=2))
                    with open(progress_log, "a") as pl:
                        pl.write(json.dumps({
                            "lam": lam, "arm": arm_name, "seed": seed,
                            "seconds": round(dt, 1),
                            "bwt": out["metrics"]["bwt"],
                            "signal_health": out.get("signal_health", "ok"),
                            "m_distribution_warning": out.get(
                                "m_distribution_warning"
                            ),
                        }) + "\n")
                print("[cell %d/%d] DONE  lam=%s arm=%s seed=%s  bwt=%+.4f  %.1fs"
                      % (done, total_cells, lam, arm_name, seed,
                         out["metrics"]["bwt"], dt), flush=True)
                per_seed_metrics.append(out)

            live = [m for m in per_seed_metrics if not m.get("degenerate")]
            bwts = [m["metrics"]["bwt"] for m in live]
            targ = [m["metrics"]["target_task_acc"] for m in live]
            results[arm_name]["lambda_runs"][str(lam)] = {
                "bwt": mean_std(bwts),
                "target_task_acc": mean_std(targ),
                "per_seed": per_seed_metrics,
                "n_degenerate": sum(1 for m in per_seed_metrics if m.get("degenerate")),
                "metric_scope": METRIC_SCOPE,
            }
    print("[run] matrix complete: %d/%d cells in %.1f min"
          % (done, total_cells, (time.time() - t_matrix) / 60), flush=True)
    results["_baseline_tuning"] = tuning
    results["_metric_scope"] = METRIC_SCOPE
    return results


def _construct_arm(ArmCls, lam: float, seed: int, pad_token_id=None,
                   baseline_tuning: Dict = None):
    """Construct an arm with λ, seed, pad id, and fixed weight-space knobs.

    Weight-space ``alpha`` (model averaging) and M-dynamics ``alpha`` share a
    name — only pass the weight-space value to ModelAveragingArm so we do not
    silently retune MyelinState.
    """
    tuning = baseline_tuning or _BASELINE_TUNING
    name = getattr(ArmCls, "name", "")
    kw = {
        "lambda_reg": lam,
        "seed": seed,
        "pad_token_id": pad_token_id,
    }
    if name == "elastic_reset":
        kw["rho"] = tuning.get("elastic_reset_rho", DEFAULT_ELASTIC_RESET_RHO)
    if name == "model_averaging":
        kw["alpha"] = tuning.get(
            "model_averaging_alpha", DEFAULT_MODEL_AVERAGING_ALPHA
        )
    try:
        return ArmCls(**kw)
    except TypeError:
        kw.pop("seed", None)
        try:
            return ArmCls(**kw)
        except TypeError:
            return ArmCls(lambda_reg=lam)


def _live_seed_metrics(run_dict: dict, key: str) -> List[float]:
    """Collect per-seed metric values, skipping degenerate cells."""
    out = []
    for p in run_dict.get("per_seed") or []:
        if p.get("degenerate"):
            continue
        m = p.get("metrics") or {}
        if key in m and m[key] == m[key]:  # reject NaN
            out.append(float(m[key]))
    return out


def decide(results: Dict) -> Dict:
    """Evaluate all pre-registered README relevance criteria for Exp 1.

    Reports (pass/fail each, no invented verdict keywords beyond the flags):
      1. lock_vs_control_2sigma — gradient_aligned_m vs raw_fisher_signal
         on BWT (NTA proxy), >2σ at n>=5, per shared λ and at best λ.
      2. beat_ewc_floor — mechanism best-λ BWT vs EWC best-λ BWT, >2σ,
         with matched target-task NTA plasticity.
      3. beat_weight_space — mechanism vs best of elastic_reset /
         model_averaging on the same axes.
      4. target_task_retention — plasticity not sacrificed vs control/EWC.

    Cost/Pareto multi-λ frontier dominance is reported as a λ-grid summary
    (contiguous region count) but wall-clock/$ requires infra provisioning
    fields and is left as a structured placeholder when absent.
    """
    m_runs = results.get("gradient_aligned_m", {}).get("lambda_runs", {})
    f_runs = results.get("raw_fisher_signal", {}).get("lambda_runs", {})
    ewc_runs = results.get("ewc", {}).get("lambda_runs", {})

    lock_vs_control = {}
    for lam in sorted(set(m_runs) & set(f_runs), key=float):
        mech_bwt = _live_seed_metrics(m_runs[lam], "bwt")
        ctrl_bwt = _live_seed_metrics(f_runs[lam], "bwt")
        mech_t = _live_seed_metrics(m_runs[lam], "target_task_acc")
        ctrl_t = _live_seed_metrics(f_runs[lam], "target_task_acc")
        gate = exceeds_2sigma(mech_bwt, ctrl_bwt)
        plast = matched_plasticity(mech_t, ctrl_t)
        lock_vs_control[lam] = {
            **gate,
            **plast,
            "passes_matched_plasticity_2sigma": float(
                gate["passes_2sigma_n5"] > 0 and plast["matched"] > 0
            ),
        }

    def _vs_baseline(base_name: str) -> Dict:
        b_runs = results.get(base_name, {}).get("lambda_runs", {})
        mb = best_lambda_run(results.get("gradient_aligned_m", {}))
        bb = best_lambda_run(results.get(base_name, {}))
        if mb is None or bb is None:
            return {"available": 0.0, "reason": "missing arm or empty runs"}
        m_lam, m_run = mb
        b_lam, b_run = bb
        mech_bwt = _live_seed_metrics(m_run, "bwt")
        base_bwt = _live_seed_metrics(b_run, "bwt")
        mech_t = _live_seed_metrics(m_run, "target_task_acc")
        base_t = _live_seed_metrics(b_run, "target_task_acc")
        gate = exceeds_2sigma(mech_bwt, base_bwt)
        plast = matched_plasticity(mech_t, base_t)
        return {
            "available": 1.0,
            "mechanism_best_lambda": m_lam,
            "baseline_best_lambda": b_lam,
            **gate,
            **plast,
            "passes_beat_baseline": float(
                gate["passes_2sigma_n5"] > 0 and plast["matched"] > 0
            ),
        }

    beat_ewc = _vs_baseline("ewc")
    beat_er = _vs_baseline("elastic_reset")
    beat_ma = _vs_baseline("model_averaging")
    # Weight-space criterion: beat at least one of ER / MA
    ws_pass = max(
        beat_er.get("passes_beat_baseline", 0.0),
        beat_ma.get("passes_beat_baseline", 0.0),
    )

    # Contiguous λ region where lock beats control at 2σ (Pareto-ish summary)
    lams_sorted = sorted(lock_vs_control.keys(), key=float)
    flags = [
        lock_vs_control[l].get("passes_matched_plasticity_2sigma", 0.0) > 0
        for l in lams_sorted
    ]
    max_contig = 0
    cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        max_contig = max(max_contig, cur)

    return {
        "metric_scope": METRIC_SCOPE,
        "metric_scope_note": METRIC_SCOPE_NOTE,
        "lock_vs_control_2sigma_by_lambda": lock_vs_control,
        "beat_ewc_floor": beat_ewc,
        "beat_elastic_reset": beat_er,
        "beat_model_averaging": beat_ma,
        "beat_weight_space_any": {
            "passes": ws_pass,
            "note": "passes if mechanism beats ER or MA at matched plasticity >2σ",
        },
        "pareto_contiguous_lambda_region": {
            "max_contiguous_pass_lambdas": max_contig,
            "n_lambda_grid": len(lams_sorted),
            "passes_nontrivial_region": float(max_contig >= 2),
        },
        "cost_adjustment": {
            "available": 0.0,
            "note": (
                "Wall-clock/$ per arm comes from infra pod_provisioning.json; "
                "not scored inside the TRACE harness. Fill at report time."
            ),
        },
        "baseline_tuning": results.get("_baseline_tuning", dict(_BASELINE_TUNING)),
    }


def tune_baselines(
    mode: str,
    *,
    n_tasks: int = 2,
    seed: int = 0,
    lr: float = 1e-2,
    epochs: int = 1,
    model_id: str = None,
    data_root=None,
    batch_size: int = 8,
    max_len: int = 512,
    rhos: List[float] = None,
    alphas: List[float] = None,
) -> Dict:
    """Short 1-seed / 2-task pre-study for weight-space knobs.

    Sweeps elastic_reset.rho and model_averaging.alpha independently,
    picks the value with best BWT (NTA proxy), and returns the chosen
    dict for the main matrix. Does NOT add these axes to the 175-cell run.
    """
    import copy
    import torch

    rhos = rhos or [0.1, 0.3, 0.5, 0.7, 0.9]
    alphas = alphas or [0.1, 0.3, 0.5, 0.7, 0.9]

    pad_token_id = None
    if mode == "trace":
        from trace_data import build_real_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer = build_real_model(model_id, device)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        sequence = build_task_sequence(
            mode, tokenizer=tokenizer, data_root=data_root,
            batch_size=batch_size, max_len=max_len, n_tasks=n_tasks,
        )
    else:
        model = build_model(mode)
        sequence = build_task_sequence(mode, n_tasks=n_tasks, seed=seed)

    # Truncate sequence if build ignored n_tasks (trace always loads 8).
    sequence = sequence[:n_tasks]
    pristine = copy.deepcopy(
        {k: v.detach().cpu() for k, v in model.state_dict().items()}
    )

    def _eval_arm(ArmCls, **kw):
        model.load_state_dict(pristine, strict=True)
        torch.manual_seed(seed)
        arm = ArmCls(lambda_reg=0.0, seed=seed, pad_token_id=pad_token_id, **kw)
        out = run_arm_over_sequence(
            arm, model, sequence, lr, epochs, pad_token_id=pad_token_id,
        )
        return out["metrics"]["bwt"]

    rho_scores = {}
    for rho in rhos:
        rho_scores[str(rho)] = _eval_arm(ALL_ARMS["elastic_reset"], rho=rho)
        print(f"[tune] elastic_reset rho={rho}: BWT={rho_scores[str(rho)]:+.4f}",
              flush=True)
    best_rho = max(rho_scores.items(), key=lambda kv: kv[1])[0]

    alpha_scores = {}
    for alpha in alphas:
        alpha_scores[str(alpha)] = _eval_arm(
            ALL_ARMS["model_averaging"], alpha=alpha,
        )
        print(f"[tune] model_averaging alpha={alpha}: "
              f"BWT={alpha_scores[str(alpha)]:+.4f}", flush=True)
    best_alpha = max(alpha_scores.items(), key=lambda kv: kv[1])[0]

    chosen = {
        "elastic_reset_rho": float(best_rho),
        "model_averaging_alpha": float(best_alpha),
        "rho_scores": rho_scores,
        "alpha_scores": alpha_scores,
        "n_tasks": n_tasks,
        "seed": seed,
        "metric_scope": METRIC_SCOPE,
        "note": (
            "Pre-study only. Fix these values in the main matrix; "
            "do not sweep rho/alpha in the 175-cell run of record."
        ),
    }
    print(f"[tune] CHOSEN rho={best_rho} alpha={best_alpha}", flush=True)
    return chosen


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU preflight; single seed by default")
    ap.add_argument("--tune-baselines", action="store_true",
                    help="1-seed/2-task pre-study for elastic_reset.rho and "
                         "model_averaging.alpha; write chosen values and exit")
    ap.add_argument("--baseline-tuning-json", type=Path, default=None,
                    help="load previously chosen rho/alpha from --tune-baselines")
    ap.add_argument("--arms", nargs="+", default=EXP1_ARMS)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--lambdas", type=float, nargs="+", default=None)
    ap.add_argument("--n_tasks", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model_id", type=str,
                    default="meta-llama/Llama-3.2-3B-Instruct",
                    help="HF base model for the real TRACE run")
    ap.add_argument("--data_root", type=Path, default=None,
                    help="dir holding the official TRACE task folders")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    mode = "smoke" if args.smoke else "trace"
    seeds = args.seeds or ([0] if args.smoke else DEFAULT_SEEDS)
    lambdas = args.lambdas or ([0.1] if args.smoke else [0.0, 0.01, 0.1, 1.0, 10.0])
    n_tasks = args.n_tasks or (3 if args.smoke else 8)

    started = datetime.now(timezone.utc).isoformat()
    stamp = started[:19].replace(":", "")
    # Archive contract: anything that must survive cloud dispatch lives under
    # the results/ directory the wrapper SCP/B2-pulls. Prefer (in order):
    #   1. explicit --out
    #   2. $SYNTH_RESULTS_DIR (set by the cloud wrapper)
    #   3. ./results next to cwd (pod layout: REMOTE_BASE/<bench>/results)
    #   4. local lab_reports/ (laptop-direct runs only)
    if args.out is not None:
        out = Path(args.out)
    else:
        env_results = os.environ.get("SYNTH_RESULTS_DIR")
        cwd_results = Path.cwd() / "results"
        if env_results:
            out_dir = Path(env_results)
        elif cwd_results.is_dir():
            out_dir = cwd_results
        else:
            out_dir = ROOT / "lab_reports"
        out = out_dir / f"exp1_{'smoke' if args.smoke else 'trace'}_{stamp}.json"
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.tune_baselines:
        chosen = tune_baselines(
            mode, n_tasks=min(2, n_tasks), seed=(seeds[0] if seeds else 0),
            lr=args.lr, epochs=args.epochs, model_id=args.model_id,
            data_root=args.data_root, batch_size=args.batch_size,
            max_len=args.max_len,
        )
        tune_path = out.parent / "baseline_tuning.json"
        tune_path.write_text(json.dumps(chosen, indent=2))
        print(f"[tune] written: {tune_path}")
        return

    if not args.smoke and len(seeds) < 5:
        print("[run] REFUSING: run of record requires n>=5 seeds (README). "
              "Use --smoke for preflight.", file=sys.stderr)
        sys.exit(2)

    tuning = dict(_BASELINE_TUNING)
    if args.baseline_tuning_json and args.baseline_tuning_json.exists():
        loaded = json.loads(args.baseline_tuning_json.read_text())
        for k in ("elastic_reset_rho", "model_averaging_alpha"):
            if k in loaded:
                tuning[k] = float(loaded[k])
        print(f"[run] loaded baseline tuning: {tuning}", flush=True)

    # Cell checkpoints MUST sit next to the final JSON so B2/SCP capture them.
    ckpt_dir = out.parent / "cells"

    results = run(mode, args.arms, seeds, lambdas, n_tasks, args.lr, args.epochs,
                  model_id=args.model_id, data_root=args.data_root,
                  batch_size=args.batch_size, max_len=args.max_len,
                  ckpt_dir=ckpt_dir, baseline_tuning=tuning)
    relevance = decide(results)

    record = {
        "experiment": "exp1_does_the_lock_do_anything",
        "mode": mode,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "metric_scope": METRIC_SCOPE,
        "metric_scope_note": METRIC_SCOPE_NOTE,
        "lock_kind": "asymptotic_soft_penalty_based_consolidation",
        "config": {
            "arms": args.arms, "seeds": seeds, "lambdas": lambdas,
            "n_tasks": n_tasks, "lr": args.lr, "epochs": args.epochs,
            "baseline_tuning": tuning,
            "phase1_isolation_screen": (
                "gradient_aligned_m vs raw_fisher_signal; task-boundary M; "
                "only use_asymptotic_lock differs"
            ),
        },
        "results": results,
        # Back-compat key: lock-vs-control by λ only
        "lock_vs_no_lock_2sigma_gate": relevance.get(
            "lock_vs_control_2sigma_by_lambda", {}
        ),
        "relevance_criteria": relevance,
        "note": (
            "smoke mode is PREFLIGHT ONLY, never evidence. n>=5 real TRACE "
            "run is the run of record. Phase-1 DV = BWT under NTA proxy; "
            "faithful TRACE metrics are Phase 2. Lock = asymptotic soft "
            "penalty-based consolidation (not hard freeze)."
        ),
    }

    out.write_text(json.dumps(record, indent=2))

    # Human + dispatcher summary sibling (dispatcher looks for results_summary.txt).
    summary_lines = [
        f"=== Experiment 1 ({mode})  metric_scope={METRIC_SCOPE} ===",
        f"record: {out.name}",
    ]
    print(f"\n=== Experiment 1 ({mode})  metric_scope={METRIC_SCOPE} ===")
    for arm in args.arms:
        lr_runs = results[arm]["lambda_runs"]
        best = max(
            ((k, v) for k, v in lr_runs.items() if v["bwt"]["n"] > 0),
            key=lambda kv: kv[1]["bwt"]["mean"],
            default=None,
        )
        if best:
            lam, r = best
            line = (f"  {arm:22s} best@λ={lam:>6}: "
                    f"BWT={r['bwt']['mean']:+.4f}±{r['bwt']['std']:.4f}  "
                    f"targetAcc={r['target_task_acc']['mean']:.3f}"
                    f"  deg={r.get('n_degenerate', 0)}")
            print(line)
            summary_lines.append(line)
    print("\n  lock-vs-control 2σ (mechanism − isolation control), NTA proxy:")
    summary_lines.append("lock-vs-control 2σ (NTA proxy):")
    for lam, v in relevance.get("lock_vs_control_2sigma_by_lambda", {}).items():
        flag = "PASS" if v.get("passes_matched_plasticity_2sigma") else "no"
        line = (f"    λ={lam:>6}: ΔBWT={v['delta_bwt']:+.4f}  "
                f"Δ/σ={v['delta_over_sigma']:+.2f}  matched_plast="
                f"{int(v.get('matched', 0))}  [{flag}]")
        print(line)
        summary_lines.append(line)
    ewc = relevance.get("beat_ewc_floor", {})
    ewc_line = (f"  beat EWC floor: "
                f"{'PASS' if ewc.get('passes_beat_baseline') else 'no'}  "
                f"ΔBWT={ewc.get('delta_bwt', float('nan')):+.4f}")
    ws_line = (f"  beat weight-space (any): "
               f"{'PASS' if relevance.get('beat_weight_space_any', {}).get('passes') else 'no'}")
    print(f"\n{ewc_line}")
    print(ws_line)
    summary_lines.extend([ewc_line, ws_line, f"written: {out}"])
    (out.parent / "results_summary.txt").write_text("\n".join(summary_lines) + "\n")
    print(f"\n  written: {out}")


if __name__ == "__main__":
    main()
