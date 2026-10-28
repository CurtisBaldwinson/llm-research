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
from typing import Dict, List, Optional

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

#: Arms whose behaviour does NOT depend on lambda_reg, verified against their
#: source: sweeping lambda for these produces bit-identical cells, so the matrix
#: runs them at a SINGLE lambda (the first in the grid) instead of the full
#: sweep. This is not a scientific trim -- it removes duplicate recomputation
#: and stops the logs from implying a lambda sensitivity these arms do not have.
#: Evidence (grep the classes):
#:   vanilla        -> VanillaArm.aux_loss returns None (no penalty term)
#:   elastic_reset  -> lambda_reg declared "unused; kept for uniform construction"
#:                     (driven by rho, fixed after --tune-baselines)
#:   model_averaging-> self.lambda_reg referenced 0x in body (driven by alpha)
#: The lambda-CONSUMING arms (gradient_aligned_m, raw_fisher_signal, ewc,
#: replay) are deliberately excluded and keep their full sweep.
LAMBDA_INVARIANT_ARMS = frozenset({"vanilla", "elastic_reset", "model_averaging"})

DEFAULT_SEEDS = [0, 1, 2, 3, 4]   # n>=5 per README

# Weight-space knobs fixed after pre-study; not swept in the main matrix.
_BASELINE_TUNING = {
    "elastic_reset_rho": DEFAULT_ELASTIC_RESET_RHO,
    "model_averaging_alpha": DEFAULT_MODEL_AVERAGING_ALPHA,
}

#: Fix #2 (matched-plasticity frontier, 2026-07-25 review): the weight-space
#: baselines are lambda-invariant, so at a single tuned operating point they
#: are a DOT while the mechanism is a CURVE -- there is nothing to match M's
#: plasticity against, and "M dominates" becomes an unmatched-plasticity
#: artifact. To give each baseline its own plasticity frontier we sweep its
#: native knob (rho for elastic_reset, alpha for model_averaging). Because
#: these arms ignore lambda_reg, the swept knob value is carried in the
#: lambda-key slot of results[arm]["lambda_runs"] (each cell tags sweep_knob/
#: sweep_value for honest logs); _frontier_dominance then reads it exactly
#: like the mechanism's lambda curve. The single tuned value from
#: --tune-baselines is retained as the marked operating point on this curve,
#: not discarded. Sweeping a knob these arms DO consume is a scientific
#: comparison (a real baseline frontier), not the duplicate-recompute that
#: LAMBDA_INVARIANT_ARMS trims away.
BASELINE_SWEEPS = {
    "elastic_reset": ("rho", [0.1, 0.3, 0.5, 0.7, 0.9]),
    "model_averaging": ("alpha", [0.1, 0.3, 0.5, 0.7, 0.9]),
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
    the official Google Drive release and normalizes it under <parent>/TRACE.

    ``data_root`` may be either:
      * the stage parent (e.g. HERE/data) -- stage_trace creates TRACE/ under it; or
      * an already-staged task root (…/TRACE with ScienceQA/train.json present).
    """
    from pathlib import Path
    from trace_data import build_trace_sequence
    from fetch_data import stage_trace
    if tokenizer is None:
        raise ValueError("trace mode requires a tokenizer")
    if data_root is None:
        # Parent dir under which stage_trace creates TRACE/. Writable both
        # locally (benchmarks/trace/data) and on the pod (REMOTE_BASE/trace/data).
        data_root = HERE / "data"
    data_root = Path(data_root)
    # Accept an already-staged task root so --data_root …/TRACE keeps working.
    if (data_root / "ScienceQA" / "train.json").exists():
        staged_root = data_root
    else:
        # stage_trace(parent) creates <parent>/TRACE and returns that path.
        # build_trace_sequence reads <root>/<task> directly.
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
    device = next(model.parameters()).device
    # Output-preserving perf (2026-07-26 efficiency review): accumulate the
    # correct/total counts as int64 tensors ON GPU and synchronize to the host
    # exactly ONCE per loader, instead of two .item() device->host syncs per
    # batch. The arithmetic is identical; only the number of stalls changes.
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = torch.zeros((), dtype=torch.long, device=device)
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
            correct += ((pred == gold) & valid).sum()
            total += valid.sum()
    total_i = int(total.item())  # single sync
    correct_i = int(correct.item())  # single sync
    return correct_i / total_i if total_i else 0.0


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
        # Evaluate ONLY tasks seen so far (j <= i). R was zero-initialised, so
        # the strict upper triangle (future tasks, j > i) stays 0.0. Future-task
        # cells are intentionally left at 0.0. This changes the raw upper triangle,
        # but is lossless for the documented metrics because every metric in
        # eval/metrics.py reads only the lower triangle:
        #   backward_transfer -> final row R[T-1][j] + diagonal R[j][j]
        #   average_accuracy  -> final row R[T-1][j]
        #   learning_accuracy -> diagonal R[j][j]
        #   target_task_acc   -> R[T-1][T-1]
        # None read R[i][j] for j > i, so not computing them changes nothing.
        # Removes ~44% of eval passes (28 of 64 task-evals over 8 tasks).
        for j in range(i + 1):
            ev = sequence[j][1]
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

def _seeded_sequence(sequence, seed, mode, batch_size, fixed_train_order=False,
                     crop_batches=False, sort_eval_batches=False):
    """Return a per-seed view of ``sequence`` with train example order permuted.

    Turns the ``seeds`` axis into genuine replicates: each seed visits the
    within-task training examples in a different (seeded) order, producing
    independent SGD trajectories and non-degenerate across-seed variance. The
    *task* sequence order and the eval loaders are left untouched -- only the
    within-task training order varies, so it is a clean replicate mechanism,
    not a change to the experimental design.

    For ``smoke`` mode the synthetic sequence is already seeded at build time,
    so we return it unchanged. If ``fixed_train_order`` is set (debug/escape
    hatch), the original order is preserved and seeds collapse to identical
    runs -- only use this to reproduce the old deterministic behavior.

    Batch 2 efficiency transforms (both OFF by default, gated by a passing
    real-model preflight before any scored run -- see main()):
      * ``sort_eval_batches`` -- regroup each eval loader by real length so the
        subsequent crop trims more. Output-preserving for AGGREGATE eval metrics
        only (see trace_data.sort_eval_loader_by_length). Applied to eval
        loaders only; train loaders keep their seeded order (sorting train would
        destroy the replicate mechanism above).
      * ``crop_batches`` -- drop shared padding columns from every train AND
        eval batch. Forward-equivalent under a correct attention mask (see
        trace_data.crop_batch_padding). Applied LAST so it also crops the
        length-sorted eval batches.
    Both are no-ops for ``smoke``/``fixed_train_order`` paths only where noted;
    cropping is safe to apply in all real modes. They are kept here so both
    run() call sites get identical treatment from one place.
    """
    from trace_data import (
        reshuffle_train_loader, crop_loader_padding, sort_eval_loader_by_length,
    )

    # 1. seeded train-order replicate view (unchanged behavior)
    if fixed_train_order or mode != "trace":
        seq = sequence
    else:
        seq = [
            (reshuffle_train_loader(train_loader, seed, batch_size), eval_loader)
            for (train_loader, eval_loader) in sequence
        ]

    if not (crop_batches or sort_eval_batches):
        return seq  # fast path: exact legacy behavior

    out = []
    for train_loader, eval_loader in seq:
        ev = eval_loader
        if sort_eval_batches:
            ev = sort_eval_loader_by_length(ev, batch_size)
        if crop_batches:
            train_loader = crop_loader_padding(train_loader)
            ev = crop_loader_padding(ev)
        out.append((train_loader, ev))
    return out


def run(mode: str, arms: List[str], seeds: List[int], lambdas: List[float],
        n_tasks: int, lr: float, epochs: int,
        model_id: str = None, data_root=None,
        batch_size: int = 8, max_len: int = 512, ckpt_dir=None,
        baseline_tuning: Dict = None, fixed_train_order: bool = False,
        crop_batches: bool = False, sort_eval_batches: bool = False) -> Dict:
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

    # Build each arm's list of operating points along its plasticity frontier.
    #   * lambda-consuming arms  -> the full lambda grid (knob = lambda)
    #   * swept baselines (ER/MA)-> their native knob grid from BASELINE_SWEEPS
    #     (knob = rho/alpha), carried in the lambda-key slot; the tuned value
    #     from --tune-baselines is retained as the marked operating point.
    #   * other lambda-invariant arms (vanilla) -> a SINGLE point at lambdas[0]
    # Each point = (cell_key, construct_lam, sweep_override). cell_key is what
    # goes into results[...]["lambda_runs"] and the checkpoint filename.
    def _arm_points(arm_name):
        if arm_name in BASELINE_SWEEPS:
            knob, grid = BASELINE_SWEEPS[arm_name]
            values = [float(v) for v in grid]
            tuned = tuning.get(f"{arm_name}_{knob}")
            if tuned is not None and not any(
                abs(float(tuned) - v) < 1e-9 for v in values
            ):
                values.append(float(tuned))
            values.sort()
            return [(str(v), lambdas[0], (knob, v)) for v in values]
        if arm_name in LAMBDA_INVARIANT_ARMS:
            return [(str(lambdas[0]), lambdas[0], None)]
        return [(str(l), l, None) for l in lambdas]

    arm_points = {a: _arm_points(a) for a in arms}
    # Honest cell count: sum of each arm's operating points x seeds.
    total_cells = sum(len(pts) for pts in arm_points.values()) * len(seeds)
    done = 0
    t_matrix = time.time()

    for arm_name in arms:
        for cell_key, construct_lam, sweep_override in arm_points[arm_name]:
            lam = cell_key  # checkpoint/results key (lambda value or swept knob)
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
                # Per-seed training-example order => genuine replicates
                # (see _seeded_sequence / reshuffle_train_loader).
                seed_sequence = _seeded_sequence(
                    sequence, seed, mode, batch_size,
                    fixed_train_order=fixed_train_order,
                    crop_batches=crop_batches,
                    sort_eval_batches=sort_eval_batches,
                )

                t0 = time.time()
                print("[cell %d/%d] START lam=%s arm=%s seed=%s"
                      % (done, total_cells, lam, arm_name, seed), flush=True)

                ArmCls = ALL_ARMS[arm_name]
                arm = _construct_arm(
                    ArmCls, construct_lam, seed,
                    pad_token_id=pad_token_id,
                    baseline_tuning=tuning,
                    sweep_override=sweep_override,
                )
                try:
                    out = run_arm_over_sequence(
                        arm, model, seed_sequence, lr, epochs,
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
            cell_agg = {
                "bwt": mean_std(bwts),
                "target_task_acc": mean_std(targ),
                "per_seed": per_seed_metrics,
                "n_degenerate": sum(1 for m in per_seed_metrics if m.get("degenerate")),
                "metric_scope": METRIC_SCOPE,
            }
            # Honest logs: tag which knob this frontier point swept and whether
            # it is the tuned operating point retained from --tune-baselines.
            if sweep_override is not None:
                knob, val = sweep_override
                tuned_val = tuning.get(f"{arm_name}_{knob}")
                cell_agg["sweep_knob"] = knob
                cell_agg["sweep_value"] = float(val)
                cell_agg["is_tuned_operating_point"] = bool(
                    tuned_val is not None
                    and abs(float(tuned_val) - float(val)) < 1e-9
                )
            else:
                cell_agg["sweep_knob"] = "lambda"
            results[arm_name]["lambda_runs"][str(lam)] = cell_agg
    print("[run] matrix complete: %d/%d cells in %.1f min"
          % (done, total_cells, (time.time() - t_matrix) / 60), flush=True)
    results["_baseline_tuning"] = tuning
    results["_metric_scope"] = METRIC_SCOPE
    return results


def _construct_arm(ArmCls, lam: float, seed: int, pad_token_id=None,
                   baseline_tuning: Dict = None, sweep_override=None):
    """Construct an arm with λ, seed, pad id, and fixed weight-space knobs.

    Weight-space ``alpha`` (model averaging) and M-dynamics ``alpha`` share a
    name — only pass the weight-space value to ModelAveragingArm so we do not
    silently retune MyelinState.

    ``sweep_override`` (Fix #2): a (knob_name, value) pair that overrides the
    fixed tuned weight-space knob for this cell, used to trace a baseline's
    plasticity frontier (see BASELINE_SWEEPS). When set for elastic_reset it
    supplies ``rho``; for model_averaging it supplies ``alpha``. It NEVER
    touches mechanism dynamics.
    """
    tuning = baseline_tuning or _BASELINE_TUNING
    name = getattr(ArmCls, "name", "")
    kw = {
        "lambda_reg": lam,
        "seed": seed,
        "pad_token_id": pad_token_id,
    }
    ovr_knob, ovr_val = (sweep_override or (None, None))
    if name == "elastic_reset":
        kw["rho"] = (
            float(ovr_val) if ovr_knob == "rho"
            else tuning.get("elastic_reset_rho", DEFAULT_ELASTIC_RESET_RHO)
        )
    if name == "model_averaging":
        kw["alpha"] = (
            float(ovr_val) if ovr_knob == "alpha"
            else tuning.get(
                "model_averaging_alpha", DEFAULT_MODEL_AVERAGING_ALPHA
            )
        )
    # Mechanism arms: explicit signal_tuning dict suppresses auto filesystem load.
    if name in ("gradient_aligned_m", "raw_fisher_signal",
                "gradient_aligned_m_online"):
        if "signal_tuning" in tuning:
            kw["signal_tuning"] = tuning["signal_tuning"]
            kw["load_signal_tuning"] = False
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


def _frontier_dominance(
    mech_block: dict,
    base_block: dict,
    *,
    plasticity_tol: float = 0.02,
) -> Dict:
    """Matched-plasticity Pareto-frontier dominance of mechanism over a baseline.

    This is the honest replacement for the old single-``best_lambda`` comparison,
    which let each arm cherry-pick its own most-flattering operating point and so
    could report a "win" that was really just an unmatched-plasticity artifact.

    Both arms expose a plasticity frontier through their ``lambda_runs`` map --
    for the mechanism the knob is lambda; for the (formerly lambda-invariant)
    weight-space baselines the same key slot carries the swept rho/alpha value
    (see BASELINE_SWEEPS). For EACH mechanism operating point we:
      1. find the baseline point at MATCHED plasticity (min |delta target-task
         NTA|, and only if within ``plasticity_tol``), then
      2. test whether the mechanism's BWT beats that matched baseline point by
         >2 sigma at n>=5 (exceeds_2sigma).
    The mechanism DOMINATES at a point iff a matched baseline point exists and
    the 2-sigma BWT gate passes there. We report per-point results, the total
    number of dominated points, and the longest contiguous dominated run along
    the mechanism's sorted frontier (a Pareto-region summary, not a single dot).

    Returns ``available=0.0`` with a reason when either arm is missing/empty.
    """
    m_runs = (mech_block or {}).get("lambda_runs", {}) or {}
    b_runs = (base_block or {}).get("lambda_runs", {}) or {}
    if not m_runs or not b_runs:
        return {"available": 0.0, "reason": "missing arm or empty runs"}

    # Pre-extract baseline curve (BWT + plasticity per baseline knob point).
    base_points = []
    for b_key, b_run in b_runs.items():
        b_bwt = _live_seed_metrics(b_run, "bwt")
        b_targ = _live_seed_metrics(b_run, "target_task_acc")
        if not b_bwt or not b_targ:
            continue
        base_points.append({
            "key": b_key,
            "bwt": b_bwt,
            "targ": b_targ,
            "targ_mean": sum(b_targ) / len(b_targ),
        })
    if not base_points:
        return {"available": 0.0, "reason": "baseline has no live points"}

    per_point = []
    # Sort mechanism points numerically along its frontier for the contiguity run.
    def _as_float(k):
        try:
            return float(k)
        except (TypeError, ValueError):
            return float("inf")

    for m_key in sorted(m_runs.keys(), key=_as_float):
        m_run = m_runs[m_key]
        mech_bwt = _live_seed_metrics(m_run, "bwt")
        mech_t = _live_seed_metrics(m_run, "target_task_acc")
        if not mech_bwt or not mech_t:
            per_point.append({
                "mechanism_point": m_key, "matched": 0.0,
                "reason": "mechanism point empty", "dominates": 0.0,
            })
            continue
        mech_t_mean = sum(mech_t) / len(mech_t)
        # Closest baseline point in plasticity space.
        nearest = min(
            base_points, key=lambda bp: abs(bp["targ_mean"] - mech_t_mean)
        )
        plast = matched_plasticity(mech_t, nearest["targ"], tol=plasticity_tol)
        entry = {
            "mechanism_point": m_key,
            "matched_baseline_point": nearest["key"],
            **{f"plasticity_{k}": v for k, v in plast.items()},
        }
        if plast["matched"] <= 0:
            entry.update({"dominates": 0.0,
                          "reason": "no baseline point at matched plasticity"})
            per_point.append(entry)
            continue
        gate = exceeds_2sigma(mech_bwt, nearest["bwt"])
        entry.update({f"bwt_{k}": v for k, v in gate.items()})
        entry["dominates"] = float(gate["passes_2sigma_n5"] > 0)
        per_point.append(entry)

    dominated_flags = [p.get("dominates", 0.0) > 0 for p in per_point]
    n_dominated = sum(1 for f in dominated_flags if f)
    max_contig = 0
    cur = 0
    for f in dominated_flags:
        cur = cur + 1 if f else 0
        max_contig = max(max_contig, cur)

    return {
        "available": 1.0,
        "n_mechanism_points": len(per_point),
        "n_dominated_points": n_dominated,
        "max_contiguous_dominated": max_contig,
        # Headline: dominates if it wins a nontrivial contiguous frontier region
        # at matched plasticity + 2 sigma (not a single cherry-picked point).
        "passes_frontier_dominance": float(max_contig >= 2),
        "per_point": per_point,
    }


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
        mech_block = results.get("gradient_aligned_m", {})
        base_block = results.get(base_name, {})
        # HEADLINE: matched-plasticity frontier dominance across the whole curve
        # (replaces the old single best-lambda cherry-pick). See
        # _frontier_dominance for the honest-comparison rationale.
        frontier = _frontier_dominance(mech_block, base_block)
        if frontier.get("available", 0.0) <= 0:
            return frontier
        out = {
            "available": 1.0,
            "passes_beat_baseline": frontier["passes_frontier_dominance"],
            "frontier_dominance": frontier,
        }
        # SECONDARY (labeled, non-headline): the old best-point readout is kept
        # for continuity/diagnostics but is NOT the pass criterion, because
        # independently best-lambda per arm compares at unmatched plasticity.
        mb = best_lambda_run(mech_block)
        bb = best_lambda_run(base_block)
        if mb is not None and bb is not None:
            m_lam, m_run = mb
            b_lam, b_run = bb
            gate = exceeds_2sigma(
                _live_seed_metrics(m_run, "bwt"),
                _live_seed_metrics(b_run, "bwt"),
            )
            plast = matched_plasticity(
                _live_seed_metrics(m_run, "target_task_acc"),
                _live_seed_metrics(b_run, "target_task_acc"),
            )
            out["best_point_diagnostic"] = {
                "mechanism_best_lambda": m_lam,
                "baseline_best_lambda": b_lam,
                **gate,
                **plast,
                "note": (
                    "diagnostic only; each arm at its own best lambda -> "
                    "unmatched plasticity; not the pass criterion"
                ),
            }
        return out

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
    fixed_train_order: bool = False,
    crop_batches: bool = False,
    sort_eval_batches: bool = False,
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

    # Use the SAME per-seed training-example order the scored run uses at this
    # seed, so tuned baselines are selected under matched conditions. Match the
    # scored run's batch transforms too (crop / length-sort), so tuning selects
    # under identical batching.
    seed_sequence = _seeded_sequence(
        sequence, seed, mode, batch_size, fixed_train_order=fixed_train_order,
        crop_batches=crop_batches, sort_eval_batches=sort_eval_batches,
    )

    def _eval_arm(ArmCls, **kw):
        model.load_state_dict(pristine, strict=True)
        torch.manual_seed(seed)
        arm = ArmCls(lambda_reg=0.0, seed=seed, pad_token_id=pad_token_id, **kw)
        out = run_arm_over_sequence(
            arm, model, seed_sequence, lr, epochs, pad_token_id=pad_token_id,
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



def _bt_beta_for_breakeven(alpha: float, tau: float) -> float:
    """M-update break-even: tau = beta/(alpha+beta) => beta = alpha*tau/(1-tau)."""
    tau = min(max(tau, 1e-6), 1.0 - 1e-6)
    return alpha * tau / (1.0 - tau)


# Cap for BC host transfer: full M is [H] per-neuron (not per-weight), so
# typically <<1e6; still bound so BC (numpy skew/kurt) never scales with a
# future per-param signal mode. Mean/frac stats use full-vector torch ops.
_BT_M_STATS_BC_MAX_ELEMS = 50_000


def _bt_m_stats(myelin, bimodality_coefficient) -> Dict:
    """Distribution-level M stats over non-head params.

    Mean / mass-split fractions: torch reductions on the full concatenated
    vector (no Python float list). BC: bounded CPU subsample only — Sarle's
    coefficient is O(n) host numpy and does not need every element.

    Note: with grad_l2_per_neuron, M is per-neuron [H], not per-weight, so
    this is tens–hundreds of K floats, not gigabytes. Torch reductions are
    still the right idiom (faster, future-proof if signal_type changes).

    Uses myelin._param_role (private) for head skip — same established
    internal contract as check_degenerate_M / batch_utils (no public
    accessor exists; adding one is a separate cleanup across all call sites).
    """
    import torch

    flats = []
    for name, M in myelin.M.items():
        if myelin._param_role.get(name, "representation") == "head":
            continue
        if M is None or not torch.is_tensor(M) or M.numel() == 0:
            continue
        flats.append(M.detach().reshape(-1).float())
    if not flats:
        return dict(n=0, mean_M=None, frac_near_zero=None, frac_locked_hi=None,
                    frac_mid=None, bc=None)

    # Keep on the tensors' device for reductions; only BC subsample hits host.
    v = torch.cat(flats)
    n = int(v.numel())
    mean_M = float(v.mean().item())
    frac_near_zero = float((v < 0.01).float().mean().item())
    frac_locked_hi = float((v > 0.90).float().mean().item())
    frac_mid = float(((v >= 0.01) & (v <= 0.90)).float().mean().item())

    # Bounded subsample for BC (explicit CPU — never pass a CUDA tensor to
    # numpy-backed bimodality_coefficient).
    if n <= _BT_M_STATS_BC_MAX_ELEMS:
        bc_host = v.detach().cpu()
    else:
        # Even stride sample preserves rough mass shape without RNG coupling.
        step = max(1, n // _BT_M_STATS_BC_MAX_ELEMS)
        bc_host = v[::step][:_BT_M_STATS_BC_MAX_ELEMS].detach().cpu()
    try:
        bc = float(bimodality_coefficient(bc_host.numpy()))
    except Exception:
        bc = float("nan")

    return dict(
        n=n,
        mean_M=mean_M,
        frac_near_zero=frac_near_zero,
        frac_locked_hi=frac_locked_hi,
        frac_mid=frac_mid,
        bc=bc,
        bc_n_subsample=int(bc_host.numel()),
    )


def _bt_hard_degenerate(myelin) -> Optional[str]:
    """Hard-fail only empty / all-zero / NaN / Inf (not BC)."""
    from synthetic_myelination import check_m_hard_degenerate
    return check_m_hard_degenerate(myelin)


def _bt_mass_split_health(stats: Dict) -> Dict:
    """Health from mass split (DN02): need meaningful protected AND plastic mass."""
    hi = stats.get("frac_locked_hi")
    mid = stats.get("frac_mid")
    z = stats.get("frac_near_zero")
    if hi is None or mid is None or z is None:
        return {"mass_split_ok": False, "mass_split_note": "missing stats"}
    # Degenerate extremes: ~90% empty or ~90% ceiling.
    if z >= 0.90:
        return {"mass_split_ok": False, "mass_split_note": "near_zero_heavy"}
    if hi >= 0.90:
        return {"mass_split_ok": False, "mass_split_note": "locked_hi_heavy"}
    # Prefer some mid or balanced hi/zero.
    ok = (hi >= 0.05 and z <= 0.85) or (mid >= 0.10)
    return {
        "mass_split_ok": bool(ok),
        "mass_split_note": "ok" if ok else "unbalanced_split",
    }


DEFAULT_CALIB_TAUS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
# w=80 dropped from the DEFAULT grid (2026-07-25 review): even with the
# _CALIB_STEPS_HARD_CAP it is the single most expensive window (~35% of calib
# steps/task) AND is non-adoptable (> MAX_ADOPTABLE_EMA_WINDOW, see below), so
# it costs measurement budget for a point that can never win. It stays fully
# supported as an opt-in probe: `--ema-windows 1 5 15 40 80`.
DEFAULT_CALIB_EMA_WINDOWS = [1, 5, 15, 40]
# Pre-registered protected-fraction frontier endpoints (DN02).
PROTECTED_FRAC_TARGETS = [0.05, 0.10, 0.20, 0.40]
# Over-smoothing guard (data-grounded, DN02): an EMA window is only trustworthy
# if its memory is short relative to real per-task training length. TRACE trains
# 1 epoch over n_train=5000 examples at batch_size=8 => ~625 steps/task
# (trace_data.build_trace_sequence n_train=5000; run.py epochs=1, batch_size=8).
# Adoptable iff window <= ~10% of that budget (~62 steps): w in {1,5,15,40} pass
# (w=40 ~= 6.4% of a task). w=80 (~12.8%) is NOT in the default grid anymore
# (opt-in via --ema-windows); when probed it is mapped + on the frontier
# plot for curiosity/boundary behavior, but never selectable as a default-arm or
# EMA-on winner, because a large window mechanically shrinks protected_frac_std
# and would let the recommender adopt an over-smoothing artifact, not a better
# signal. If a rerun changes n_train/batch_size/epochs, recompute this ceiling.
MAX_ADOPTABLE_EMA_WINDOW = 40
# Under-populated std guard: single-task rows report std=None (unknown), not
# 0.0. Rows need at least this many tasks with a real protected fraction
# before they may win selection / EMA-on verdict (same "measured≠selectable"
# pattern as MAX_ADOPTABLE_EMA_WINDOW).
MIN_TASKS_FOR_VERDICT = 2
# Std must improve by at least this absolute amount, at comparable protected
# mass, before EMA-on may win on std alone (guards tiny noise wins).
EMA_ON_MATERIAL_STD_DELTA = 0.02
# Protected-fraction means must agree within this absolute band for a
# std-only EMA-on win to count as "comparable protected level".
EMA_ON_COMPARABLE_PROT_BAND = 0.10


def _ema_window_adoptable(w: int) -> bool:
    return int(w) <= int(MAX_ADOPTABLE_EMA_WINDOW)


def _row_selectable_for_verdict(t: dict) -> bool:
    """True iff a lookup row may win recommended / EMA-on (not just be mapped)."""
    if t.get("protected_frac_mean") is None:
        return False
    if t.get("n_hard_degenerate_tasks", 0) > 0:
        return False
    if not t.get("adoptable", _ema_window_adoptable(t.get("ema_window", 999))):
        return False
    if int(t.get("n_tasks") or 0) < int(MIN_TASKS_FOR_VERDICT):
        return False
    # Honest std required: None means under-populated, not "zero variance".
    if t.get("protected_frac_std") is None:
        return False
    return True


def _verdict_row_key(t: dict) -> tuple:
    """Single comparator for recommended / per-window / cross-window picks.

    Lower is better:
      1. fewer mass-split failures
      2. closer to 50% protected, with std penalty (mean±2σ style)

    Callers must only pass rows that pass ``_row_selectable_for_verdict``
    (real std guaranteed). Used by ``_build_lookup_and_recommend`` and
    ``infra/merge_calib_shards.py`` so priority order cannot drift.
    """
    return (
        int(t["n_mass_split_fail_tasks"]),
        abs(float(t["protected_frac_mean"]) - 0.5)
        + 2.0 * float(t["protected_frac_std"]),
    )


# Hard cap on per-point calibration steps. The warm-up rule (10*w) is what
# killed the 2026-07-25 run: w=80 wants 800 steps/point x 7 taus = 5,600 passes
# for that block alone, which crossed the pod wall-clock/resource ceiling and
# SIGKILL-ed all 8 pods simultaneously (no traceback). We clamp so no single
# point can run away with the pod budget. The cap is chosen ABOVE the warm-up
# need of every *adoptable* window (w<=MAX_ADOPTABLE_EMA_WINDOW=40 -> 400 steps)
# so adoptable rows stay fully warmed; only non-adoptable probes (w=80) are
# clamped. If a window needs more warm-up than the cap allows, it is by
# definition too large to be adoptable -- consistent with the over-smoothing
# guard. See docs/design_notes/02 (Results 2026-07-25).
_CALIB_STEPS_HARD_CAP = 450


def _calib_steps_for_window(base_steps: int, ema_window: int) -> int:
    """Warm EMA: steps = max(base, 10*w), clamped to _CALIB_STEPS_HARD_CAP.

    The 10*w warm-up keeps high-w rows from being under-warmed; the hard cap
    prevents a single expensive point (e.g. w=80 -> 800) from blowing the pod
    wall-clock budget. Adoptable windows (w<=40 -> <=400) are unaffected.
    """
    want = max(int(base_steps), 10 * int(ema_window))
    if want > _CALIB_STEPS_HARD_CAP:
        print(
            f"[calib] WARN: w={int(ema_window)} warm-up wants {want} steps > "
            f"cap {_CALIB_STEPS_HARD_CAP}; clamping. This window is beyond the "
            f"adoptable ceiling and is a non-adoptable probe; its stats are "
            f"under-warmed and must not be treated as authoritative.",
            flush=True,
        )
    return min(want, _CALIB_STEPS_HARD_CAP)


def _resolve_calib_task_ids(
    calib_tasks: Optional[List[int]],
    n_all: int,
) -> Dict:
    """Resolve --calib-tasks against TRACE length; fail-fast on empty/partial drop.

    Returns a dict with:
      full_run (bool): True iff caller requested all tasks (calib_tasks is None)
      requested_tasks, task_ids, dropped_tasks, shard_label

    Empty after filter OR any out-of-range index raises ValueError so a GPU
    pod never burns on an empty/incomplete grid with a colliding shard name.
    """
    if n_all <= 0:
        raise ValueError(
            f"--calibrate: TRACE sequence is empty (n_all={n_all}); "
            "cannot resolve task shards."
        )
    if calib_tasks is None:
        task_ids = list(range(n_all))
        return {
            "full_run": True,
            "requested_tasks": list(task_ids),
            "task_ids": task_ids,
            "dropped_tasks": [],
            "shard_label": "all",
        }

    requested = [int(t) for t in calib_tasks]
    if not requested:
        raise ValueError(
            "--calibrate: --calib-tasks was provided but empty. "
            "Omit the flag for a full all-task run, or pass valid indices."
        )
    dropped = [t for t in requested if not (0 <= t < n_all)]
    task_ids = [t for t in requested if 0 <= t < n_all]
    # Preserve order, de-dupe while keeping first occurrence.
    seen = set()
    deduped = []
    for t in task_ids:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    task_ids = deduped

    if dropped:
        raise ValueError(
            "--calibrate: --calib-tasks contains out-of-range indices "
            f"{dropped} (valid: 0..{n_all - 1}). Partial drops are fatal so "
            "a pod cannot silently run a smaller shard than requested "
            "(merge would under-count tasks). Fix the arm-spec / flag."
        )
    if not task_ids:
        raise ValueError(
            "--calibrate: --calib-tasks filtered to nothing "
            f"(requested={requested}, n_trace_tasks={n_all}). "
            "Refusing empty grid — would write a colliding/empty shard."
        )
    return {
        "full_run": False,
        "requested_tasks": requested,
        "task_ids": task_ids,
        "dropped_tasks": [],
        "shard_label": "".join(str(t) for t in task_ids),
    }


def _build_lookup_and_recommend(points, taus, ema_windows, alpha):
    """Aggregate points -> 2D lookup + nearest-50% default + full tau map.

    Full table/map retain every measured w (including non-adoptable 40/80)
    and under-populated single-task rows. Selection only considers rows that
    pass ``_row_selectable_for_verdict`` (adoptable w, n_tasks>=min, real std).
    """
    import statistics

    table = []
    for w in ema_windows:
        for tau in taus:
            rows = [
                p for p in points
                if abs(p["tau"] - round(float(tau), 4)) < 1e-9
                and int(p["ema_window"]) == int(w)
            ]
            hi = [p["frac_locked_hi"] for p in rows if p.get("frac_locked_hi") is not None]
            mid = [p["frac_mid"] for p in rows if p.get("frac_mid") is not None]
            z = [p["frac_near_zero"] for p in rows if p.get("frac_near_zero") is not None]
            a_ex = [
                p["frac_A_above_tau_mean"] for p in rows
                if p.get("frac_A_above_tau_mean") is not None
            ]
            hard_deg = sum(1 for p in rows if p.get("hard_degenerate"))
            mass_bad = sum(1 for p in rows if not p.get("mass_split_ok", True))
            bc_warn = sum(1 for p in rows if p.get("bc_warning"))
            # steps: surface heterogeneity across merged shards instead of
            # silently taking rows[0].
            step_vals = sorted({
                int(p["steps"]) for p in rows
                if p.get("steps") is not None
            })
            steps_heterogeneous = len(step_vals) > 1
            entry = dict(
                tau=round(float(tau), 4),
                ema_window=int(w),
                adoptable=_ema_window_adoptable(w),
                beta=round(_bt_beta_for_breakeven(alpha, float(tau)), 6),
                steps_used=(step_vals[0] if len(step_vals) == 1 else None),
                steps_used_values=step_vals,
                steps_heterogeneous=steps_heterogeneous,
                protected_frac_mean=statistics.mean(hi) if hi else None,
                # None = unknown (n<2), never fake 0.0 precision.
                protected_frac_std=(
                    statistics.pstdev(hi) if len(hi) > 1 else None
                ),
                plastic_mid_frac_mean=statistics.mean(mid) if mid else None,
                near_zero_frac_mean=statistics.mean(z) if z else None,
                frac_A_above_tau_mean=statistics.mean(a_ex) if a_ex else None,
                n_hard_degenerate_tasks=hard_deg,
                n_mass_split_fail_tasks=mass_bad,
                n_bc_warning_tasks=bc_warn,
                n_tasks=len(rows),
                n_tasks_with_protected=len(hi),
            )
            entry["selectable_for_verdict"] = _row_selectable_for_verdict(entry)
            table.append(entry)

    def _score(t):
        if not _row_selectable_for_verdict(t):
            return (1e9,)
        return _verdict_row_key(t)

    healthy = [t for t in table if _row_selectable_for_verdict(t)]
    recommended = min(healthy, key=_score) if healthy else None

    # Full tau -> protected-fraction map per ema_window (for frontier sweep).
    protected_frac_map = {}
    for w in ema_windows:
        series = []
        for tau in taus:
            rows = [
                t for t in table
                if int(t["ema_window"]) == int(w)
                and abs(t["tau"] - round(float(tau), 4)) < 1e-9
            ]
            if not rows:
                continue
            t0 = rows[0]
            series.append({
                "tau": t0["tau"],
                "beta": t0["beta"],
                "protected_frac_mean": t0["protected_frac_mean"],
                "protected_frac_std": t0["protected_frac_std"],
            })
        protected_frac_map[str(int(w))] = series

    # Nearest grid tau for each pre-registered protected-fraction target.
    frontier_endpoints = []
    if recommended is not None:
        w_star = int(recommended["ema_window"])
        series = protected_frac_map.get(str(w_star), [])
        for target in PROTECTED_FRAC_TARGETS:
            cand = [
                s for s in series
                if s["protected_frac_mean"] is not None
            ]
            if not cand:
                continue
            best = min(
                cand,
                key=lambda s: abs(s["protected_frac_mean"] - target),
            )
            frontier_endpoints.append({
                "target_protected_frac": target,
                "tau": best["tau"],
                "beta": best["beta"],
                "ema_window": w_star,
                "achieved_protected_frac_mean": best["protected_frac_mean"],
            })

    return table, recommended, protected_frac_map, frontier_endpoints


def _calib_summary(rec: Dict) -> str:
    """Dispatcher-visible results_summary.txt for calib shards."""
    A = rec["alpha"]
    lines = [
        "=== signal calibration (init-signal, rank)  alpha=%s ===" % A,
        "scope=%s  shard=%s  ema_windows=%s  taus=%s" % (
            rec.get("calibration_scope"),
            rec.get("task_shard"),
            rec.get("ema_windows"),
            rec.get("taus"),
        ),
        "tau".rjust(5) + "w".rjust(5) + "prot%".rjust(8) + "+/-".rjust(7)
        + "mid%".rjust(7) + "~0%".rjust(7) + "A>t%".rjust(7)
        + "hard".rjust(6) + "mass".rjust(6),
    ]
    for t in rec["lookup_table"]:
        pm = t["protected_frac_mean"]
        if pm is None:
            lines.append("%5.2f%5d  (no data)" % (t["tau"], t["ema_window"]))
            continue
        std = t.get("protected_frac_std")
        std_s = ("  n/a" if std is None else "%7.1f" % (100 * std))
        sel = "sel" if t.get("selectable_for_verdict") else "map"
        het = " steps!" if t.get("steps_heterogeneous") else ""
        lines.append(
            "%5.2f%5d%7.1f%s%7.1f%7.1f%7.1f%4d/%d%4d/%d %s%s" % (
                t["tau"], t["ema_window"],
                100 * pm, std_s,
                100 * (t["plastic_mid_frac_mean"] or 0),
                100 * (t["near_zero_frac_mean"] or 0),
                100 * (t["frac_A_above_tau_mean"] or 0),
                t["n_hard_degenerate_tasks"], t["n_tasks"],
                t["n_mass_split_fail_tasks"], t["n_tasks"],
                sel, het,
            )
        )
    rc = rec.get("recommended")
    if rc:
        std = rc.get("protected_frac_std")
        std_s = "n/a" if std is None else ("%.1f" % (100 * std))
        lines.append(
            "RECOMMENDED_DEFAULT_ARM tau=%.2f w=%d -> protected %.1f%% +/-%s, "
            "beta=%.4f (nearest-50%%; not the full frontier)" % (
                rc["tau"], rc["ema_window"],
                100 * rc["protected_frac_mean"],
                std_s,
                _bt_beta_for_breakeven(A, rc["tau"]),
            )
        )
    else:
        lines.append(
            "RECOMMENDED_DEFAULT_ARM: NONE selectable "
            "(need adoptable w, n_tasks>=%d, real std)."
            % MIN_TASKS_FOR_VERDICT
        )
    lines.append(
        "NOTE: init-signal calibration only. Do not lock tau*/w* into arms "
        "until a post-1-epoch-task0 confirmation cell agrees."
    )
    return "\n".join(lines)


def calibrate_signal(
    *,
    taus: List[float] = None,
    ema_windows: List[int] = None,
    alpha: float = 0.1,
    steps: int = 150,
    calib_tasks: List[int] = None,
    model_id: str = None,
    data_root=None,
    batch_size: int = 8,
    max_len: int = 512,
) -> Dict:
    """Init-signal calibration on real Llama+LoRA TRACE gradients (DN02).

    Sweeps the 2-D grid (tau x ema_window). Per point steps = max(base, 10*w)
    so EMA windows are warmed. No CL sequence / no BWT. Scope is explicitly
    init-weights; arm defaults require a post-task confirmation cell.
    """
    import torch

    from trace_data import build_real_model
    from arms_mechanism import _accumulate_signal_into_M
    from synthetic_myelination import MyelinState, bimodality_coefficient
    from fetch_data import TRACE_TASKS

    _SIGNAL_TYPE = "grad_l2_per_neuron"
    taus = list(taus or DEFAULT_CALIB_TAUS)
    ema_windows = [int(w) for w in (ema_windows or DEFAULT_CALIB_EMA_WINDOWS)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = build_real_model(model_id, device)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    sequence = build_task_sequence(
        "trace", tokenizer=tokenizer, data_root=data_root,
        batch_size=batch_size, max_len=max_len,
    )
    n_all = len(sequence)
    # Fail-fast before model work is wasted: empty or partially invalid
    # --calib-tasks must not produce a silent undersized / colliding shard.
    shard_meta = _resolve_calib_task_ids(calib_tasks, n_all)
    task_ids = shard_meta["task_ids"]

    print(
        f"[calib] INIT-SIGNAL scope device={device} taus={taus} "
        f"ema_windows={ema_windows} tasks={task_ids} "
        f"shard_label={shard_meta['shard_label']!r} "
        f"base_steps={steps} alpha={alpha} (of {n_all} TRACE tasks)",
        flush=True,
    )

    points = []
    for w in ema_windows:
        n_steps = _calib_steps_for_window(steps, w)
        for tau in taus:
            beta = _bt_beta_for_breakeven(alpha, tau)
            for ti in task_ids:
                for p in model.parameters():
                    p.grad = None
                model.train()
                myelin = MyelinState(
                    model, alpha=alpha, beta=beta,
                    use_asymptotic_lock=True, signal_type=_SIGNAL_TYPE,
                    normalization="rank",
                    ema_window=int(w),
                )
                a_diag = _accumulate_signal_into_M(
                    model, sequence[ti][0], myelin, torch,
                    n_steps=n_steps, pad_token_id=pad_token_id,
                    ema_window=int(w), track_tau=float(tau),
                ) or {}
                stats = _bt_m_stats(myelin, bimodality_coefficient)
                hard = _bt_hard_degenerate(myelin)
                mass = _bt_mass_split_health(stats)
                # BC is warning-only (DN02 / M2).
                bc_warn = None
                try:
                    from synthetic_myelination import check_degenerate_M
                    bc_warn = check_degenerate_M(myelin)
                except Exception:
                    bc_warn = None
                r = dict(
                    tau=round(float(tau), 4),
                    ema_window=int(w),
                    alpha=alpha,
                    beta=round(beta, 6),
                    task_idx=ti,
                    task=(
                        TRACE_TASKS[ti] if ti < len(TRACE_TASKS) else str(ti)
                    ),
                    steps=n_steps,
                    hard_degenerate=hard,
                    bc_warning=bc_warn,
                    mass_split_ok=mass["mass_split_ok"],
                    mass_split_note=mass["mass_split_note"],
                    frac_A_above_tau_mean=a_diag.get("frac_A_above_tau_mean"),
                    frac_A_above_tau_n_steps=a_diag.get(
                        "frac_A_above_tau_n_steps"
                    ),
                    **stats,
                )
                points.append(r)
                print(
                    f"[calib] tau={tau:.2f} w={w} task={ti} steps={n_steps} "
                    f"meanM={r['mean_M']} %~0={None if r['frac_near_zero'] is None else 100*r['frac_near_zero']:.1f} "
                    f"%hi={None if r['frac_locked_hi'] is None else 100*r['frac_locked_hi']:.1f} "
                    f"A>t={None if r['frac_A_above_tau_mean'] is None else 100*r['frac_A_above_tau_mean']:.1f} "
                    f"hard={'Y' if hard else 'n'} mass={'ok' if mass['mass_split_ok'] else mass['mass_split_note']}",
                    flush=True,
                )

    table, recommended, pmap, frontier = _build_lookup_and_recommend(
        points, taus, ema_windows, alpha,
    )

    return dict(
        harness="calibrate_signal",
        design_note="02_signal_normalization_pivot",
        calibration_scope="init_signal",
        calibration_scope_note=(
            "Measured on fresh LoRA-init weights with TRACE task batches. "
            "NOT post-task M. Do not lock tau*/w* into mechanism arms until a "
            "cheap post-1-epoch-task0 confirmation cell agrees on mass split."
        ),
        normalization="rank",
        signal_type=_SIGNAL_TYPE,
        alpha=alpha,
        base_steps=steps,
        taus=taus,
        ema_windows=ema_windows,
        task_shard=task_ids,
        task_shard_full_run=bool(shard_meta["full_run"]),
        task_shard_label=shard_meta["shard_label"],
        requested_tasks=list(shard_meta["requested_tasks"]),
        dropped_tasks=list(shard_meta["dropped_tasks"]),
        n_trace_tasks=n_all,
        points=points,
        lookup_table=table,
        protected_frac_map=pmap,
        frontier_endpoints=frontier,
        recommended=recommended,
        protected_frac_targets=PROTECTED_FRAC_TARGETS,
        note=(
            "Init-signal calibration: break-even x ema_window -> protected "
            "fraction on real TRACE gradients under rank conditioning. "
            "NOT a CL evaluation; no BWT claim. recommended = nearest-50% "
            "default-arm pick; protected_frac_map + frontier_endpoints feed "
            "the pre-registered {5,10,20,40}% axis. Merge shards with "
            "infra/merge_calib_shards.py before writing signal_tuning.json. "
            "Shard label is explicit ('all' only for intentional full runs)."
        ),
    )


def write_signal_tuning_json(recommended: Dict, alpha: float, path: Path) -> Path:
    """Write signal_tuning.json for mechanism-arm autowire (M3)."""
    if not recommended:
        raise ValueError("no recommended row to write")
    tau = float(recommended["tau"])
    payload = {
        "schema": "signal_tuning/v1",
        "design_note": "02_signal_normalization_pivot",
        "calibration_scope": "init_signal",
        "alpha": float(alpha),
        "beta": float(_bt_beta_for_breakeven(alpha, tau)),
        "tau": tau,
        "ema_window": int(recommended["ema_window"]),
        "normalization": "rank",
        "protected_frac_mean": recommended.get("protected_frac_mean"),
        "protected_frac_std": recommended.get("protected_frac_std"),
        "confirmation_required": True,
        "confirmation_note": (
            "Init-signal only. Run a post-1-epoch-task0 confirmation cell "
            "before treating these as locked arm defaults."
        ),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Post-task0 confirmation cell (DN02).
#
# The init-signal calibration (calibrate_signal) measures M on FRESH LoRA-init
# weights. That is enough to *select* an arm (tau*/w*/beta*) but NOT enough to
# lock it into a scored CL run: the mass split we care about is the one that
# holds on REAL post-task M, after the model has actually trained on a task.
# signal_tuning.json therefore ships with confirmation_required=true, and the
# arm-side gate (_apply_signal_tuning) refuses to apply it until this cell
# clears the flag.
#
# This cell is deliberately LIGHTWEIGHT RELATIVE TO A FULL RUN (not "laptop
# cheap": it still trains ONE full task-0 epoch of Llama-3.2-3B, ~625 steps,
# plus a measure phase of _calib_steps_for_window(150, w) = max(150, 10*w)
# steps -- i.e. >=150, scaling with the selected EMA window (150 at the
# recommended w=1; 400 at w=40) -- a few minutes on an A40, not something to run
# on a laptop). It trains ONLY task 0 for --epochs epochs
# (default 1), then measures M ONCE at the single recommended (tau*, w*, beta*)
# point -- not the whole grid. It reuses the exact same measurement path as
# calibrate_signal (MyelinState -> _accumulate_signal_into_M -> _bt_m_stats ->
# _bt_hard_degenerate -> _bt_mass_split_health) so there is no second, drifting
# definition of "healthy". The ONLY difference from calibration is the weights
# under measurement: post-1-epoch-task0 instead of init.
#
# Pass criteria (all required):
#   * not hard-degenerate,
#   * mass_split_ok == True on post-task M,
#   * protected_frac within CONFIRM_PROTECTED_TOL of the init-signal value
#     (the arm we picked must still behave like the arm we measured).
# On pass, the tuning file is rewritten with confirmation_required=false plus a
# full audit block. On fail, the flag is left true and the reason is reported;
# nothing is silently locked.
# ---------------------------------------------------------------------------
CONFIRM_PROTECTED_TOL = 0.12  # post-task protected_frac may drift +-12pp from
# the init-signal value and still confirm; wider than calibration's cross-task
# std because init->post-task is a genuine distribution shift, but tight enough
# that a collapsed (near_zero_heavy) or saturated (locked_hi_heavy) split fails.


def confirm_signal_tuning(
    tuning_path,
    *,
    epochs: int = 1,
    model_id: str = None,
    data_root=None,
    batch_size: int = 8,
    max_len: int = 512,
    lr: float = 1e-2,
    task_idx: int = 0,
) -> Dict:
    """Confirm init-signal tuning against post-1-epoch-task0 M (DN02).

    Trains task ``task_idx`` for ``epochs`` epochs, then measures M once at the
    recommended (tau, w, beta) from ``tuning_path`` using the SAME path as
    calibrate_signal. Rewrites the file with confirmation_required=false iff the
    post-task mass split is healthy and protected_frac is comparable to the
    init-signal value. Returns the confirmation record.
    """
    # Guard the gate's core premise: this is a POST-TASK confirmation. With
    # epochs<1 the training loop below is skipped, so M would be measured on
    # (essentially) init weights and could clear confirmation_required using the
    # very init-signal evidence the gate exists to supersede. Refuse it. (Same
    # "measured != selectable" discipline as the arm-side and merge-side gates.)
    if int(epochs) < 1:
        raise ValueError(
            f"confirmation requires epochs >= 1 (got {epochs}); epochs<1 would "
            "measure init-weight M and defeat the post-task gate"
        )
    if int(task_idx) < 0:
        raise ValueError(f"task_idx must be >= 0 (got {task_idx})")

    import json
    from pathlib import Path

    import torch

    from trace_data import build_real_model
    from arms_mechanism import _accumulate_signal_into_M
    from synthetic_myelination import (
        MyelinState, bimodality_coefficient, unpack_lm_batch,
    )

    _SIGNAL_TYPE = "grad_l2_per_neuron"

    tuning_path = Path(tuning_path)
    if not tuning_path.is_file():
        raise ValueError(f"tuning file not found: {tuning_path}")
    tuning = json.loads(tuning_path.read_text())
    tau = float(tuning["tau"])
    w = int(tuning["ema_window"])
    alpha = float(tuning.get("alpha", 0.1))
    beta = float(tuning.get("beta", _bt_beta_for_breakeven(alpha, tau)))
    init_prot = tuning.get("protected_frac_mean")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = build_real_model(model_id, device)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    sequence = build_task_sequence(
        "trace", tokenizer=tokenizer, data_root=data_root,
        batch_size=batch_size, max_len=max_len,
    )
    if task_idx >= len(sequence):
        raise ValueError(
            f"task_idx={task_idx} out of range (n_tasks={len(sequence)})"
        )

    print(
        f"[confirm] POST-TASK{task_idx} scope device={device} tau={tau} w={w} "
        f"alpha={alpha} beta={beta:.6f} epochs={epochs} "
        f"init_signal_protected={init_prot}",
        flush=True,
    )

    # --- train task_idx for `epochs` epochs (same inner loop as the harness) ---
    train_loader = sequence[task_idx][0]
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    model.train()
    n_batches = 0
    for _ep in range(int(epochs)):
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            dev = next(model.parameters()).device
            ids, labels, attention_mask = unpack_lm_batch(
                batch, dev, pad_token_id=pad_token_id,
            )
            try:
                loss = model(
                    ids, labels=labels, attention_mask=attention_mask,
                ).loss
            except TypeError:
                loss = model(ids, labels=labels).loss
            loss.backward()
            opt.step()
            n_batches += 1
    print(f"[confirm] trained task{task_idx}: {n_batches} batches", flush=True)

    # --- measure M ONCE at the recommended point (identical to calibration) ---
    for p in model.parameters():
        p.grad = None
    model.train()
    myelin = MyelinState(
        model, alpha=alpha, beta=beta,
        use_asymptotic_lock=True, signal_type=_SIGNAL_TYPE,
        normalization="rank", ema_window=int(w),
    )
    n_steps = _calib_steps_for_window(150, w)
    a_diag = _accumulate_signal_into_M(
        model, train_loader, myelin, torch,
        n_steps=n_steps, pad_token_id=pad_token_id,
        ema_window=int(w), track_tau=float(tau),
    ) or {}
    stats = _bt_m_stats(myelin, bimodality_coefficient)
    hard = _bt_hard_degenerate(myelin)
    mass = _bt_mass_split_health(stats)

    post_prot = stats.get("frac_locked_hi")
    prot_drift = (
        None if (post_prot is None or init_prot is None)
        else abs(float(post_prot) - float(init_prot))
    )

    reasons = []
    if hard:
        reasons.append(f"hard_degenerate={hard}")
    if not mass["mass_split_ok"]:
        reasons.append(f"mass_split={mass['mass_split_note']}")
    if prot_drift is None:
        reasons.append("protected_frac unavailable for drift check")
    elif prot_drift > CONFIRM_PROTECTED_TOL:
        reasons.append(
            f"protected_frac drift {prot_drift:.3f} > tol "
            f"{CONFIRM_PROTECTED_TOL} (init={init_prot}, post={post_prot})"
        )
    confirmed = not reasons

    hi_display = "na" if post_prot is None else f"{100 * post_prot:.1f}"
    mid = stats.get("frac_mid")
    mid_display = "na" if mid is None else f"{100 * mid:.1f}"
    near_zero = stats.get("frac_near_zero")
    near_zero_display = "na" if near_zero is None else f"{100 * near_zero:.1f}"
    print(
        f"[confirm] post-task M: %hi={hi_display} "
        f"%mid={mid_display} %~0={near_zero_display} "
        f"hard={'Y' if hard else 'n'} mass={mass['mass_split_note']} "
        f"drift={'na' if prot_drift is None else f'{prot_drift:.3f}'} "
        f"=> {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}",
        flush=True,
    )

    record = dict(
        harness="confirm_signal_tuning",
        design_note="02_signal_normalization_pivot",
        confirmation_scope=f"post_epoch{epochs}_task{task_idx}",
        tau=tau, ema_window=w, alpha=alpha, beta=beta,
        init_signal_protected_frac=init_prot,
        post_task_protected_frac=post_prot,
        post_task_frac_mid=stats.get("frac_mid"),
        post_task_frac_near_zero=stats.get("frac_near_zero"),
        protected_frac_drift=prot_drift,
        protected_frac_tol=CONFIRM_PROTECTED_TOL,
        hard_degenerate=hard,
        mass_split_ok=mass["mass_split_ok"],
        mass_split_note=mass["mass_split_note"],
        frac_A_above_tau_mean=a_diag.get("frac_A_above_tau_mean"),
        steps=n_steps,
        confirmed=confirmed,
        not_confirmed_reasons=reasons,
    )

    if confirmed:
        tuning["confirmation_required"] = False
        tuning["confirmation"] = record
        tuning["confirmation_note"] = (
            f"CONFIRMED against post-epoch{epochs}-task{task_idx} M on "
            f"{datetime.now(timezone.utc).date().isoformat()}: mass split "
            f"healthy ({mass['mass_split_note']}), protected_frac drift "
            f"{'na' if prot_drift is None else f'{prot_drift:.3f}'} within tol."
        )
        tuning_path.write_text(json.dumps(tuning, indent=2))
        print(
            f"[confirm] CONFIRMED — cleared confirmation_required in "
            f"{tuning_path}; arms may now apply this tuning by default.",
            flush=True,
        )
    else:
        # Leave the file untouched (confirmation_required stays true). Emit the
        # record next to it for the audit trail without poisoning the tuning.
        audit = tuning_path.with_suffix(".confirmation_failed.json")
        audit.write_text(json.dumps(record, indent=2))
        print(
            f"[confirm] NOT CONFIRMED — {'; '.join(reasons)}. Left "
            f"confirmation_required=true in {tuning_path}; wrote {audit}. "
            f"Arms will continue to refuse this tuning.",
            flush=True,
        )

    # Mirror outputs into the results dir so they survive an ephemeral GPU pod.
    # On the cloud the pod's filesystem (including tuning_path) is destroyed
    # after the run; only $SYNTH_RESULTS_DIR is tar-pulled back to local
    # staging. Without this, a CONFIRMED flag-clear would be lost with the pod.
    results_dir = os.environ.get("SYNTH_RESULTS_DIR")
    if results_dir:
        rd = Path(results_dir)
        try:
            rd.mkdir(parents=True, exist_ok=True)
            # the (possibly flag-cleared) tuning file, under its original name
            (rd / tuning_path.name).write_text(json.dumps(tuning, indent=2))
            # the confirmation record, always, for the audit trail
            (rd / "signal_tuning_confirmation.json").write_text(
                json.dumps(record, indent=2)
            )
            print(
                f"[confirm] mirrored tuning + record into {rd} "
                f"(survives pod teardown; pull these back to local staging).",
                flush=True,
            )
        except OSError as exc:
            print(f"[confirm] WARN: could not mirror to results dir: {exc}",
                  flush=True)
    return record


# ---------------------------------------------------------------------------
# Batch 2 preflight gate
# ---------------------------------------------------------------------------

BATCH2_PREFLIGHT_FILENAME = "_batch2_preflight.json"


def _current_git_commit() -> str:
    """Best-effort git commit for binding a Batch 2 preflight artifact.

    Preference order matches preflight_batch2._git_commit:
      1. ``SYNTH_GIT_COMMIT`` (dispatcher stamps operator HEAD; pods have no
         ``.git`` after tar-push).
      2. ``git rev-parse HEAD`` from this file's directory (local hand-runs).
      3. Empty string -- gate then matches model_id only (with a warning);
         a missing commit must not silently open the gate.
    """
    import subprocess
    env = (os.environ.get("SYNTH_GIT_COMMIT") or "").strip()
    if env:
        return env
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


def _check_batch2_preflight(out_dir: Path, model_id: str,
                            crop_batches: bool, sort_eval_batches: bool):
    """Enforce the Batch 2 equivalence gate for a SCORED run.

    A scored run with --crop-batches and/or --sort-eval-batches on must be able
    to point at a preflight artifact (written by preflight_batch2.py) that:
      * PASSED (crop logits/loss <=1e-5; sort aggregate accuracy exact), and
      * covers the transform(s) actually enabled, and
      * matches the current model_id, and
      * matches the current git commit (when both are available).

    Searches, in order: out_dir, out_dir.parent, and the run.py directory --
    so the artifact can live next to results or next to the code. Raises
    SystemExit(2) with an explicit reason if the gate is not satisfied. This is
    the load-bearing guarantee that scored numbers were proven equivalent
    BEFORE the fleet spent A40 hours producing them.
    """
    if not (crop_batches or sort_eval_batches):
        return  # nothing enabled; gate is inert

    search = [out_dir, out_dir.parent, Path(__file__).resolve().parent]
    art_path = None
    for d in search:
        cand = d / BATCH2_PREFLIGHT_FILENAME
        if cand.exists():
            art_path = cand
            break
    if art_path is None:
        print(
            "[batch2] REFUSING scored run: --crop-batches/--sort-eval-batches "
            "is set but no preflight artifact (%s) was found in %s. Run "
            "preflight_batch2.py on the target A40 first, or use "
            "--allow-batch2-without-preflight for a NON-scored probe only."
            % (BATCH2_PREFLIGHT_FILENAME, [str(d) for d in search]),
            file=sys.stderr, flush=True,
        )
        sys.exit(2)

    art = json.loads(art_path.read_text())
    problems = []
    if not art.get("passed", False):
        problems.append("artifact reports passed=false")
    if art.get("model_id") != model_id:
        problems.append(
            f"model_id mismatch: artifact={art.get('model_id')!r} run={model_id!r}"
        )
    cur_commit = _current_git_commit()
    art_commit = art.get("git_commit", "")
    if cur_commit and art_commit:
        if cur_commit != art_commit:
            problems.append(
                f"git commit mismatch: artifact={art_commit[:12]} "
                f"run={cur_commit[:12]} (re-run preflight after code changes)"
            )
    else:
        print(
            "[batch2] WARNING: git commit unavailable for %s; gate matched on "
            "model_id only. Ensure the preflight was produced from this exact "
            "code." % ("artifact" if not art_commit else "run"),
            file=sys.stderr, flush=True,
        )
    # Each enabled transform must be individually proven in the artifact.
    checks = art.get("checks", {})
    if crop_batches:
        c = checks.get("crop", {})
        if not c.get("passed", False):
            problems.append(
                "crop transform not proven in artifact (checks.crop.passed "
                "is not true)"
            )
    if sort_eval_batches:
        s = checks.get("sort_eval", {})
        if not s.get("passed", False):
            problems.append(
                "sort_eval transform not proven in artifact "
                "(checks.sort_eval.passed is not true)"
            )
    if problems:
        print(
            "[batch2] REFUSING scored run: preflight artifact %s does not "
            "authorize this configuration:\n  - %s\nRe-run preflight_batch2.py "
            "on the target A40 for this model + commit, or use "
            "--allow-batch2-without-preflight for a NON-scored probe."
            % (art_path, "\n  - ".join(problems)),
            file=sys.stderr, flush=True,
        )
        sys.exit(2)
    print(
        "[batch2] preflight OK: %s (crop=%s sort_eval=%s, model=%s, "
        "commit=%s) authorizes this scored run."
        % (art_path.name, crop_batches, sort_eval_batches, model_id,
           (art_commit[:12] or "n/a")),
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU preflight; single seed by default")
    ap.add_argument("--tune-baselines", action="store_true",
                    help="1-seed/2-task pre-study for elastic_reset.rho and "
                         "model_averaging.alpha; write chosen values and exit")
    ap.add_argument("--calibrate", action="store_true",
                    help="Design Note 02 init-signal calibration: tau x "
                         "ema_window grid on real TRACE gradients under rank. "
                         "Not a CL eval; emits no BWT.")
    ap.add_argument("--calib-tasks", type=int, nargs="+", default=None,
                    help="TRACE task indices for this shard (parallel fan-out). "
                         "Omit for intentional full all-task run (shard label "
                         "'all'). Out-of-range or empty specs are fatal "
                         "(no silent partial drop / empty grid).")
    ap.add_argument("--taus", type=float, nargs="+", default=None,
                    help="break-even thresholds (default 0.1..0.7).")
    ap.add_argument("--ema-windows", type=int, nargs="+", default=None,
                    help="EMA windows to sweep (default 1 5 15 40; w=80 is a "
                         "non-adoptable opt-in probe, add it explicitly to "
                         "measure the over-smoothing boundary).")
    ap.add_argument("--calib-alpha", type=float, default=0.1,
                    help="alpha held fixed while beta sets the break-even.")
    ap.add_argument("--calib-steps", type=int, default=150,
                    help="base signal-accumulation steps; actual steps per "
                         "point = max(this, 10*ema_window).")
    ap.add_argument("--baseline-tuning-json", type=Path, default=None,
                    help="load previously chosen rho/alpha from --tune-baselines")
    ap.add_argument("--signal-tuning-json", type=Path, default=None,
                    help="load signal_tuning.json (alpha/beta/ema_window) into "
                         "mechanism arms; also written by merge_calib_shards")
    ap.add_argument("--confirm-signal-tuning", type=Path, default=None,
                    help="DN02 post-1-epoch-task0 confirmation cell: train task "
                         "0 for --epochs, measure M at the recommended "
                         "(tau,w,beta) in this signal_tuning.json, and clear "
                         "confirmation_required iff the post-task mass split is "
                         "healthy and protected_frac is comparable. Cheap; "
                         "single point, single task. Nothing else runs.")
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
                    help="TRACE task-folder root (…/TRACE) or parent dir to "
                         "stage into (default: <bench>/data -> …/data/TRACE)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--fixed-train-order", dest="fixed_train_order",
                    action="store_true",
                    help="DEBUG ONLY: disable per-seed training-example "
                         "shuffling, so all seeds train on identical data "
                         "order (reproduces the old deterministic behavior; "
                         "collapses the seed axis to n=1 -- do NOT use for a "
                         "run of record).")
    ap.add_argument("--crop-batches", dest="crop_batches",
                    action="store_true",
                    help="EFFICIENCY (Batch 2): drop shared padding columns "
                         "from every train/eval batch. Forward-equivalent under "
                         "a correct attention mask. A SCORED run refuses to "
                         "start with this on unless preflight_batch2.py has "
                         "recorded a passing real-model equivalence proof for "
                         "this model_id + git commit (see --allow-batch2-"
                         "without-preflight to override for non-scored probes).")
    ap.add_argument("--sort-eval-batches", dest="sort_eval_batches",
                    action="store_true",
                    help="EFFICIENCY (Batch 2): regroup eval examples by real "
                         "length so cropping trims more (~halves eval "
                         "positions). Output-preserving for AGGREGATE eval "
                         "metrics ONLY -- unsafe with any per-example eval "
                         "metric. Same preflight gate as --crop-batches.")
    ap.add_argument("--allow-batch2-without-preflight",
                    dest="allow_batch2_without_preflight",
                    action="store_true",
                    help="Override the Batch 2 preflight gate. ONLY for "
                         "non-scored timing/smoke probes; a scored run with "
                         "this set is refused (the whole point of the gate is "
                         "that scored numbers are proven equivalent first).")
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
        # Tuning selects operating points that feed the scored run, so it must
        # use the SAME batch transforms. It is a short 1-seed/2-task pre-study
        # (not a scored matrix), so the hard preflight gate does not apply here;
        # but if the transforms are enabled we still print a reminder that the
        # scored run will require the preflight.
        if (args.crop_batches or args.sort_eval_batches):
            print("[batch2] tune-baselines using crop=%s sort_eval=%s to match "
                  "the scored run; remember the scored run itself still "
                  "requires a passing preflight artifact."
                  % (args.crop_batches, args.sort_eval_batches), flush=True)
        chosen = tune_baselines(
            mode, n_tasks=min(2, n_tasks), seed=(seeds[0] if seeds else 0),
            lr=args.lr, epochs=args.epochs, model_id=args.model_id,
            data_root=args.data_root, batch_size=args.batch_size,
            max_len=args.max_len, fixed_train_order=args.fixed_train_order,
            crop_batches=args.crop_batches,
            sort_eval_batches=args.sort_eval_batches,
        )
        tune_path = out.parent / "baseline_tuning.json"
        tune_path.write_text(json.dumps(chosen, indent=2))
        print(f"[tune] written: {tune_path}")
        return

    if args.confirm_signal_tuning is not None:
        # DN02 post-1-epoch-task0 confirmation cell. Cheap single-point check
        # that clears confirmation_required iff post-task M stays healthy.
        rec = confirm_signal_tuning(
            args.confirm_signal_tuning,
            epochs=args.epochs,
            model_id=args.model_id,
            data_root=args.data_root,
            batch_size=args.batch_size,
            max_len=args.max_len,
            lr=args.lr,
        )
        # Non-zero exit on failure so CI / make can gate on it.
        sys.exit(0 if rec.get("confirmed") else 3)

    if args.calibrate:
        # Design Note 02 init-signal calibration (tau x ema_window).
        # Task-sharded via --calib-tasks; no seed guard. Artifacts land in
        # results/ for B2/SCP. Merge shards before writing signal_tuning.json.
        try:
            rec = calibrate_signal(
                taus=args.taus,
                ema_windows=args.ema_windows,
                alpha=args.calib_alpha,
                steps=args.calib_steps,
                calib_tasks=args.calib_tasks,
                model_id=args.model_id,
                data_root=args.data_root,
                batch_size=args.batch_size,
                max_len=args.max_len,
            )
        except ValueError as exc:
            # Fail-fast shard resolution (empty / partial --calib-tasks).
            print(f"[calib] FATAL: {exc}", file=sys.stderr, flush=True)
            sys.exit(2)
        # Explicit label from resolver: "all" only for intentional full runs;
        # never `or "all"` fallback (collides empty-shard with full-run).
        shard = rec.get("task_shard_label")
        if not shard:
            shard = (
                "all" if rec.get("task_shard_full_run")
                else "".join(str(t) for t in (rec.get("task_shard") or []))
            )
        if not shard:
            print(
                "[calib] FATAL: empty task_shard_label after successful "
                "calibrate_signal (internal invariant broken).",
                file=sys.stderr, flush=True,
            )
            sys.exit(2)
        calib_path = out.parent / f"calib_signal_shard{shard}.json"
        calib_path.write_text(json.dumps(rec, indent=2))
        summary = _calib_summary(rec)
        (out.parent / "results_summary.txt").write_text(summary + "\n")
        print("\n" + summary, flush=True)
        print(f"[calib] written: {calib_path}", flush=True)
        print(
            "[calib] scope=init_signal — confirm post-1-epoch-task0 before "
            "locking arm defaults; merge shards with infra/merge_calib_shards.py",
            flush=True,
        )
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

    signal_tuning = None
    if args.signal_tuning_json is not None:
        from arms_mechanism import load_signal_tuning
        signal_tuning = load_signal_tuning(args.signal_tuning_json)
        if signal_tuning:
            print(f"[run] loaded signal tuning: {signal_tuning}", flush=True)
        else:
            print(f"[run] WARN: --signal-tuning-json not found/empty: "
                  f"{args.signal_tuning_json}", flush=True)

    # Cell checkpoints MUST sit next to the final JSON so B2/SCP capture them.
    ckpt_dir = out.parent / "cells"

    if signal_tuning:
        tuning["signal_tuning"] = signal_tuning
    if args.fixed_train_order and not args.smoke:
        print("[run] WARNING: --fixed-train-order set on a non-smoke run; the "
              "seed axis will collapse to identical replicates (n=1). This is "
              "NOT a valid run of record.", file=sys.stderr, flush=True)

    # Batch 2 gate: a SCORED run (non-smoke run of record) may not enable the
    # crop/sort transforms unless a matching preflight proof exists. --smoke is
    # a preflight itself (never evidence), so it is exempt. The explicit
    # override is allowed ONLY for non-scored probes and is refused here for a
    # scored run.
    scored_run = not args.smoke
    if scored_run and (args.crop_batches or args.sort_eval_batches):
        if args.allow_batch2_without_preflight:
            print("[batch2] REFUSING: --allow-batch2-without-preflight cannot "
                  "be used on a scored (non-smoke) run of record. Run "
                  "preflight_batch2.py on the target A40 and let the gate pass.",
                  file=sys.stderr, flush=True)
            sys.exit(2)
        _check_batch2_preflight(
            out.parent, args.model_id,
            args.crop_batches, args.sort_eval_batches,
        )
    elif (args.crop_batches or args.sort_eval_batches):
        # smoke / non-scored: allowed, but only silently if the override is set;
        # otherwise remind the operator this path is not gated.
        if not args.allow_batch2_without_preflight:
            print("[batch2] NOTE: crop/sort enabled on a non-scored (smoke) "
                  "run; equivalence is NOT gated here. Use this only for "
                  "timing/logic probes, never as evidence.", flush=True)

    results = run(mode, args.arms, seeds, lambdas, n_tasks, args.lr, args.epochs,
                  model_id=args.model_id, data_root=args.data_root,
                  batch_size=args.batch_size, max_len=args.max_len,
                  ckpt_dir=ckpt_dir, baseline_tuning=tuning,
                  fixed_train_order=args.fixed_train_order,
                  crop_batches=args.crop_batches,
                  sort_eval_batches=args.sort_eval_batches)
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
            "per_seed_train_shuffle": (not args.fixed_train_order),
            "crop_batches": bool(args.crop_batches),
            "sort_eval_batches": bool(args.sort_eval_batches),
            "batch2_git_commit": _current_git_commit(),
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
