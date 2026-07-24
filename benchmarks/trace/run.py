#!/usr/bin/env python3
"""Experiment 1 harness -- does the lock do anything?

Compares, on a continual-learning task sequence, the two mechanism arms
(gradient_aligned_m WITH lock vs raw_fisher_signal WITHOUT lock, identical
importance signal) against the full mandatory-baseline set (vanilla, ewc,
elastic_reset, model_averaging, replay).

Metric of record: Backward Transfer (BWT). See ../../eval/metrics.py and
../../docs/design_notes/01_temporal_fidelity.md.

Two run modes:

  --smoke   Tiny synthetic task sequence on a tiny GPT-2-ish model, CPU-safe,
            single seed by default. PREFLIGHT ONLY -- proves the harness
            logic, the arm interface, and the metric wiring. Never evidence.

  (default) Real TRACE task sequence via HuggingFace + a real base model on
            GPU. n>=5 seeds, full lambda sweep. This is the run of record and
            is dispatched through ../../infra on cloud GPUs.

The harness NEVER special-cases an arm: every arm is driven through the
baselines.base.Arm interface, which is what enforces "only the lock varies."
"""

from __future__ import annotations

import argparse
import json
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

from eval.metrics import summarize, mean_std, exceeds_2sigma        # noqa: E402
from baselines import ARM_REGISTRY as BASELINE_ARMS                 # noqa: E402
from arms_mechanism import ARM_REGISTRY as MECH_ARMS               # noqa: E402

ALL_ARMS = {**BASELINE_ARMS, **MECH_ARMS}

#: Arms reported in Experiment 1 (mechanism arms + mandatory baselines).
EXP1_ARMS = [
    "gradient_aligned_m",   # M + lock  (mechanism under test)
    "raw_fisher_signal",    # same signal, no lock (the isolating control)
    "ewc",                  # classical Fisher floor
    "elastic_reset",        # weight-space
    "model_averaging",      # weight-space, cheap
    "replay",               # rehearsal
    "vanilla",              # sanity floor
]

DEFAULT_SEEDS = [0, 1, 2, 3, 4]   # n>=5 per README


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

            def forward(self, input_ids, labels=None):
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

def evaluate(model, eval_loader) -> float:
    """Next-token accuracy on an eval loader."""
    import torch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in eval_loader:
            ids = batch if not isinstance(batch, dict) else batch["input_ids"]
            ids = ids.to(next(model.parameters()).device)
            logits = model(ids).logits
            pred = logits[:, :-1, :].argmax(-1)
            gold = ids[:, 1:]
            correct += (pred == gold).sum().item()
            total += gold.numel()
    return correct / total if total else 0.0


def run_arm_over_sequence(arm, model, sequence, lr: float, epochs: int) -> Dict:
    """Train `arm` sequentially over all tasks; fill accuracy matrix R.

    R[i][j] = accuracy on task j after finishing task i.
    """
    import torch

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    T = len(sequence)
    R = [[0.0] * T for _ in range(T)]

    arm.on_train_start(model)
    prev_loader = None
    for i, (train_loader, _eval_loader) in enumerate(sequence):
        arm.on_task_start(model, i, prev_loader)
        model.train()
        for _ep in range(epochs):
            for batch in train_loader:
                opt.zero_grad(set_to_none=True)
                ids = batch if not isinstance(batch, dict) else batch["input_ids"]
                ids = ids.to(next(model.parameters()).device)
                base = model(ids, labels=ids).loss
                aux = arm.aux_loss(model)
                loss = base if aux is None else (base + aux)
                loss.backward()
                opt.step()
                arm.on_step(model, i, batch)
        arm.on_task_end(model, i, train_loader)
        prev_loader = train_loader
        # evaluate on all tasks seen so far (and future = 0 by convention)
        for j, (_tl, ev) in enumerate(sequence):
            R[i][j] = evaluate(model, ev)

    arm.on_train_end(model)
    return {"R": R, "metrics": summarize(R), "telemetry": arm.telemetry()}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(mode: str, arms: List[str], seeds: List[int], lambdas: List[float],
        n_tasks: int, lr: float, epochs: int,
        model_id: str = None, data_root=None,
        batch_size: int = 8, max_len: int = 512, ckpt_dir=None) -> Dict:
    """Run the arms x seeds x lambdas matrix.

    Robustness (added 2026-07-24):
      1. Model + data built ONCE; pristine weights snapshotted to CPU and
         restored before each cell (was: full 3B reload per cell, 175x).
      2. Each completed cell checkpointed immediately to
         ckpt_dir/cell_<lam>_<arm>_<seed>.json + progress.jsonl, so an
         interrupted run loses at most the in-flight cell (was: lost all).
      3. On restart, checkpointed cells are loaded and skipped (resume).
         Flushed per-cell progress prints give live timing.
    """
    import copy
    import time
    import torch

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
    if mode == "trace":
        from trace_data import build_real_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[run] building base model ONCE (model_id={model_id}, device={device})", flush=True)
        model, tokenizer = build_real_model(model_id, device)
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
                arm = _construct_arm(ArmCls, lam, seed)
                out = run_arm_over_sequence(arm, model, sequence, lr, epochs)

                dt = time.time() - t0
                out["_meta"] = {"lam": lam, "arm": arm_name, "seed": seed,
                                "seconds": round(dt, 1)}
                # ---- FIX 1: checkpoint this cell immediately --------------
                f = _cell_file(lam, arm_name, seed)
                if f:
                    f.write_text(json.dumps(out, indent=2))
                    with open(progress_log, "a") as pl:
                        pl.write(json.dumps({"lam": lam, "arm": arm_name,
                                             "seed": seed, "seconds": round(dt, 1),
                                             "bwt": out["metrics"]["bwt"]}) + "\n")
                print("[cell %d/%d] DONE  lam=%s arm=%s seed=%s  bwt=%+.4f  %.1fs"
                      % (done, total_cells, lam, arm_name, seed,
                         out["metrics"]["bwt"], dt), flush=True)
                per_seed_metrics.append(out)

            bwts = [m["metrics"]["bwt"] for m in per_seed_metrics]
            targ = [m["metrics"]["target_task_acc"] for m in per_seed_metrics]
            results[arm_name]["lambda_runs"][str(lam)] = {
                "bwt": mean_std(bwts),
                "target_task_acc": mean_std(targ),
                "per_seed": per_seed_metrics,
            }
    print("[run] matrix complete: %d/%d cells in %.1f min"
          % (done, total_cells, (time.time() - t_matrix) / 60), flush=True)
    return results


def _construct_arm(ArmCls, lam: float, seed: int):
    """Construct an arm, passing lambda and (for replay) a seed. Arms ignore
    kwargs they do not use via **kw."""
    try:
        return ArmCls(lambda_reg=lam, seed=seed)
    except TypeError:
        return ArmCls(lambda_reg=lam)


def decide(results: Dict) -> Dict:
    """Apply the README 2-sigma gate: does gradient_aligned_m beat
    raw_fisher_signal on BWT by > 2x per-seed sigma at n>=5, at the best
    shared lambda? Reports the delta but does NOT invent a verdict keyword."""
    m = results.get("gradient_aligned_m", {}).get("lambda_runs", {})
    f = results.get("raw_fisher_signal", {}).get("lambda_runs", {})
    verdicts = {}
    for lam in sorted(set(m) & set(f), key=float):
        mech_bwt = [p["metrics"]["bwt"]
                    for p in m[lam]["per_seed"]]
        fish_bwt = [p["metrics"]["bwt"]
                    for p in f[lam]["per_seed"]]
        verdicts[lam] = exceeds_2sigma(mech_bwt, fish_bwt)
    return verdicts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU preflight; single seed by default")
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

    if not args.smoke and len(seeds) < 5:
        print("[run] REFUSING: run of record requires n>=5 seeds (README). "
              "Use --smoke for preflight.", file=sys.stderr)
        sys.exit(2)

    started = datetime.now(timezone.utc).isoformat()

    # Checkpoint dir: default alongside the final --out file (i.e. results/),
    # so per-cell JSONs are captured by the wrapper's partial-pull + B2 finalize.
    out = args.out or (ROOT / "lab_reports" /
                       f"exp1_{'smoke' if args.smoke else 'trace'}_{started[:19].replace(':','')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out.parent / "cells"

    results = run(mode, args.arms, seeds, lambdas, n_tasks, args.lr, args.epochs,
                  model_id=args.model_id, data_root=args.data_root,
                  batch_size=args.batch_size, max_len=args.max_len,
                  ckpt_dir=ckpt_dir)
    verdicts = decide(results)

    record = {
        "experiment": "exp1_does_the_lock_do_anything",
        "mode": mode,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"arms": args.arms, "seeds": seeds, "lambdas": lambdas,
                   "n_tasks": n_tasks, "lr": args.lr, "epochs": args.epochs},
        "results": results,
        "lock_vs_no_lock_2sigma_gate": verdicts,
        "note": ("smoke mode is PREFLIGHT ONLY, never evidence. n>=5 real "
                 "TRACE run is the run of record."),
    }

    out.write_text(json.dumps(record, indent=2))

    # concise console summary
    print(f"\n=== Experiment 1 ({mode}) ===")
    for arm in args.arms:
        lr_runs = results[arm]["lambda_runs"]
        best = max(lr_runs.items(),
                   key=lambda kv: kv[1]["bwt"]["mean"]) if lr_runs else None
        if best:
            lam, r = best
            print(f"  {arm:22s} best@λ={lam:>6}: "
                  f"BWT={r['bwt']['mean']:+.4f}±{r['bwt']['std']:.4f}  "
                  f"targetAcc={r['target_task_acc']['mean']:.3f}")
    print("\n  lock-vs-no-lock 2σ gate (mechanism − control):")
    for lam, v in verdicts.items():
        flag = "PASS" if v["passes_2sigma_n5"] else "no"
        print(f"    λ={lam:>6}: ΔBWT={v['delta_bwt']:+.4f}  "
              f"Δ/σ={v['delta_over_sigma']:+.2f}  [{flag}]")
    print(f"\n  written: {out}")


if __name__ == "__main__":
    main()
