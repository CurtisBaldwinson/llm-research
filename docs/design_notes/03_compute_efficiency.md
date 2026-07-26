# Design Note 03 — Compute Efficiency of the Experiment-1 Matrix

**Date:** 2026-07-25
**Author:** analysis + implementation pass (verified against `benchmarks/trace/run.py`, `benchmarks/trace/baselines/`, `benchmarks/trace/eval/metrics.py`)
**Status:** Two changes shipped. Both are *output-preserving* — they change cost, not results. The eval-triangle change is proven bitwise-identical by an executed smoke run (below).

---

## Motivation

A full completed calibration cell measured **~6,615 s (110 min)** per continual-learning
cell on an **A40 SECURE @ $0.44/hr** (anchor: `trace_run_20260724T175800Z.nohup.log`,
`--arms gradient_aligned_m --lambdas 0.1 --seeds 0 1 2 3 4`, 5 cells at
6624.7 / 6613.0 / 6613.2 / 6607.3 s). The full Experiment-1 matrix is
**7 arms × 5 lambdas × 5 seeds = 175 cells** (`run.py`: `total_cells = len(lambdas)*len(arms)*len(seeds)`),
i.e. **~321 GPU-hours ≈ $141**. That is ~3× the working budget, so we audited the
per-cell hot path and the matrix shape for waste that does **not** cost scientific
integrity.

Two independent inefficiencies were found and fixed. Neither alters any reported
number; both were verified against the actual code, and the larger one is backed by
an executed equivalence proof.

---

## Change 1 — Evaluate only the lower triangle of the R matrix

### The waste

`run_arm_over_sequence` (in `run.py`) fills the T×T accuracy matrix `R`, where
`R[i][j]` = accuracy on task *j* after finishing task *i*. The original loop
evaluated **every** task after **every** task:

```python
# OLD
for j, (_tl, ev) in enumerate(sequence):
    R[i][j] = evaluate(model, ev, ...)   # includes j > i (unseen future tasks)
```

But `R` is zero-initialised, and the strict upper triangle (`j > i`, i.e. tasks
not yet trained) is **never read by any metric**. Verified in `eval/metrics.py`:

| metric | entries it reads |
|---|---|
| `backward_transfer` | final row `R[T-1][j]` + diagonal `R[j][j]` |
| `average_accuracy` | final row `R[T-1][j]` |
| `learning_accuracy` | diagonal `R[j][j]` |
| `target_task_accuracy` | `R[T-1][T-1]` |

None read `R[i][j]` for `j > i`. So computing the upper triangle produces numbers
that are stored and discarded.

### The fix

```python
# NEW — evaluate only tasks seen so far; upper triangle stays 0.0 (same value
# the old "future = 0 by convention" full loop effectively produced downstream).
for j in range(i + 1):
    ev = sequence[j][1]
    R[i][j] = evaluate(model, ev, pad_token_id=pad_token_id)
```

At the real 8-task scale this removes **28 of 64 task-evals per cell (44%)**.
Because eval is roughly half of per-cell compute (forward-only vs.
forward+backward for training), this is **~23% off every cell**.

### Proof of equivalence (executed)

A CPU smoke harness (`smoke_proof/proof.py`) runs one CL cell twice from an
**identical weight snapshot, identical seed, identical optimizer trajectory** —
once with the shipped triangle loop, once with a byte-for-byte copy of the old
full-grid loop — then diffs the full `R` matrix and all four metrics. Run across
five arms (5-task smoke → 10/25 = 40% of evals skipped; the same mechanism gives
44% at 8 tasks):

| arm | lower-triangle bitwise identical | upper-triangle new = 0.0 | all 4 metrics bitwise identical |
|---|---|---|---|
| vanilla | ✅ | ✅ | ✅ |
| gradient_aligned_m | ✅ | ✅ | ✅ |
| ewc | ✅ | ✅ | ✅ |
| replay | ✅ | ✅ | ✅ |
| elastic_reset | ✅ | ✅ | ✅ |

Concrete example (`gradient_aligned_m`, 5-task smoke). The **only** cells that
differ between old and new are upper-triangle "future task" evals, and every
metric is unchanged:

```
metrics NEW : bwt=0.009896 acc=0.055833 target=0.033333 learning=0.047917
metrics OLD : bwt=0.009896 acc=0.055833 target=0.033333 learning=0.047917   (identical)

cells where NEW != OLD: 3, ALL with j > i (unused by every metric):
  R[0][2]: old=0.0021  new=0.0   (task 2 "measured" before it was trained)
  R[0][4]: old=0.0063  new=0.0
  R[1][4]: old=0.0063  new=0.0
```

Those old nonzero values are meaningless — they are the untrained model's accuracy
on a task it has not seen — and they never entered a metric. The new code simply
does not spend GPU time producing them.

Full per-arm output: `smoke_proof/proof_results.json`.

---

## Change 2 — Run lambda-invariant arms at a single lambda

### The waste

The matrix sweeps `lambda ∈ {0.0, 0.01, 0.1, 1.0, 10.0}` for **every** arm. But
three arms do not consume `lambda_reg` at all (verified in their source):

| arm | evidence lambda is ignored |
|---|---|
| `vanilla` | `VanillaArm.aux_loss` returns `None` — no penalty term |
| `elastic_reset` | `lambda_reg` declared `"unused; kept for uniform arm construction"`; driven by `rho` (fixed after `--tune-baselines`) |
| `model_averaging` | `self.lambda_reg` referenced **0×** in the body; driven by `alpha` |

For these, all 5 lambda values produce **bit-identical cells** — 4 of every 5 are
pure recomputation. That is **60 redundant cells** (12 unique × 4 wasted repeats),
each ~110 min.

Beyond cost, sweeping lambda for arms that ignore it makes the logs imply a lambda
sensitivity that does not exist. Running them once is also *more honest*.

### The fix

A documented set at the top of `run.py`:

```python
LAMBDA_INVARIANT_ARMS = frozenset({"vanilla", "elastic_reset", "model_averaging"})
```

and a skip in the matrix loop so these arms run only at the first lambda:

```python
if arm_name in LAMBDA_INVARIANT_ARMS and lam_idx != 0:
    continue
```

`total_cells` is recomputed to reflect this so progress counting stays honest.
The lambda-**consuming** arms (`gradient_aligned_m`, `raw_fisher_signal`, `ewc`,
`replay`) are deliberately excluded and keep their full sweep.

### Downstream safety (verified)

- `best_lambda_run` takes `max` over whatever lambda keys exist → a single-key dict
  returns that key. Safe.
- `lock_vs_control` iterates `set(m_runs) & set(f_runs)` over the two mechanism arms
  only — both lambda-consuming, so untouched.
- `beat_ewc_floor` uses `best_lambda_run` on baselines → single-lambda baselines
  return their one run as the floor, which is correct (all lambdas were identical).
- The summary printer uses `max(..., default=None)` → prints `best@λ=0.0` for
  invariant arms. Honest.

### Executed check

A tiny matrix (`arms=[vanilla, ewc]`, `lambdas=[0.0, 0.1, 1.0]`, 1 seed) produced:

```
vanilla  lambda_runs keys: ['0.0']            (count=1)  <- invariant, ran once
ewc      lambda_runs keys: ['0.0','0.1','1.0'] (count=3)  <- full sweep kept
[run] matrix complete: 4/4 cells   <- honest count (not 6)
```

---

## Combined impact

Grounded in the measured 110 min/cell @ $0.44/hr:

| stage | cells | GPU-hrs | cost |
|---|---|---|---|
| current code (full 7×5×5) | 175 | ~322 | ~$141 |
| after lambda-redundancy trim | 115 | ~211 | ~$93 |
| after eval-triangle fix too | 115 (each ~23% cheaper) | ~164 | ~$72 |

**≈ 49% reduction (~$141 → ~$72), zero change to any reported result.**

### Confidence / caveats

- The **λ-redundancy** figure (60 cells) is **exact** — it is a count of provably
  identical cells removed.
- The **eval-triangle** 23%-per-cell figure is a **compute-share model**
  (train batch ≈ 3× a forward pass for the backward; eval ≈ 1× forward). The
  *correctness* of the change is proven bitwise; the *magnitude* of the wall-clock
  saving is an estimate and could be somewhat lower if eval batches are cheaper
  per item than modeled or if fixed per-task overhead dominates. The honest way to
  bank the exact number is to compare a before/after cell timing on the pod.

## What was checked and found already efficient (no gains)

- Base model + tokenizer built **once**; weights reset from an in-memory `pristine`
  snapshot per cell (no per-cell HuggingFace reload).
- Tokenized task sequence built once and reused.
- Per-cell checkpoint/skip (`_load_cell`) already implemented for resumability.

## Reproduce the proof

```
cd smoke_proof/trace
PYTHONPATH=$(pwd) python3 ../proof.py
# -> prints the per-arm identical=True table; writes ../proof_results.json
```

## Files changed

- `benchmarks/trace/run.py`
  - `run_arm_over_sequence`: eval loop `for j in range(i + 1)` + rationale comment.
  - Module const `LAMBDA_INVARIANT_ARMS` + matrix-loop skip + honest `total_cells`.
