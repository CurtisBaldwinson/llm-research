# Design Note 02 — Signal normalization pivot: from max-normalized activity to smoothed-rank activity

Status: binding rationale for the M-update signal. Amends Design Note 01's
description of what M consumes. Read alongside `README.md` and `01_temporal_fidelity.md`.
Date: 2026-07-24

## Summary (what changed, in one line)

The per-neuron activity signal `A` fed to the M-update rule is changing from
**max-normalization** (`A = grad_l2 / max(grad_l2)`) to **smoothed-rank
normalization** (EMA of the raw signal, then per-neuron fractional rank),
because max-normalization made the *protected fraction* of parameters an
unstable, distribution-dependent accident of the α/β break-even rather than a
controllable experimental variable. This is a mechanism change, made in
response to Timing-Run evidence, and it is required for the project's central
claim (a Pareto-frontier improvement) to be defensible.

## When and why: the evidence trail

- **2026-07-24, timing run** (`trace_results_20260724-105800_thistle-crab`,
  `gradient_aligned_m`, λ=0.1, seeds 0–4, A40, ~1.84 hr/cell): all 5 cells
  completed cleanly (RC=0, per-cell checkpoints written, B2 finalize OK) but
  every cell logged a **degenerate-M warning**: BC ≈ 0.95, mean(M) ≈ 0.03–0.05,
  with **~88–92% of M mass at/near zero from task 0 onward**. The asymptotic
  lock fired heavily (`lock_activations` ≈ 15.6k) but latched onto a sliver of
  parameters — "the lock protects essentially no parameters."

- **Root-cause analysis** (CPU, no GPU): the M-update is
  `M ← clamp(M + α·A − β·(1−A), 0, 1)`, so M grows only where
  `A > β/(α+β)` (the "break-even"). The timing run used α=0.1, β=0.05 →
  break-even **0.33**. With max-normalization, a transformer's heavy-tailed
  per-neuron grad-L2, divided by its single largest element, places the vast
  majority of neurons **below 0.33**, so they are driven to zero every step.
  A CPU sweep of the *real* MyelinState/ActivityCapture confirmed the
  break-even is the dominant lever: lowering β 0.05→0.01 swung M from ~60%
  near-zero to ~89% *locked-high*. Normalization mode (max/rank/softmax) at
  fixed α/β barely moved the outcome — i.e. **the α/β threshold, not the
  chosen norm, drove the collapse.**

- **The deeper problem this exposed:** under max-normalization the protected
  fraction is a knife-edge function of α/β *and* the gradient tail. A 5× β
  change flips M from "protects almost nothing" (~91% empty) to "protects
  almost everything" (~96% saturated). **Both extremes are degenerate**: an
  empty M does nothing; a saturated M is indistinguishable from freezing the
  whole model and destroys plasticity. The healthy window is narrow and
  data-dependent — exactly the fragile-tuning regime a frontier claim cannot
  rest on.

## Why this matters for the thesis (and how the thesis moves)

Design Note 01 framed M as the *time-integral of activity* — continuous,
activity-dependent consolidation, contrasted against EWC's instantaneous
Fisher snapshot. That framing is **retained**. What changes is only the
*conditioning of the activity signal*:

- **Before:** A = per-neuron grad-L2, max-normalized per tensor. The magnitude
  scale is preserved but the distribution's shape (and thus the protected
  fraction) is uncontrolled and dataset-dependent.
- **After:** A = per-neuron grad-L2, EMA-smoothed, then fractional-rank
  normalized per tensor → A is approximately uniform on (0,1] **by
  construction**. The fraction of neurons above any break-even τ is ≈ (1−τ),
  **independent of the gradient tail**. The α/β break-even (or an equivalent
  target percentile) now sets the protected fraction *directly and stably*,
  and becomes a swept frontier axis rather than a tuned point.

The EMA (existing `ema_window` knob) is applied **before** ranking so the
ordering reflects *sustained* importance over time, not single-step gradient
noise. This preserves the "activity-dependent, path-dependent" character that
distinguishes M from Fisher — the mechanism is still driven by real gradients
integrated over time; we have only made the *fraction it protects*
distribution-invariant.

### Honest statement of what this is and isn't

- This is **not** hard top-k masking. M still accumulates continuously via the
  α/β dynamics; no parameter is hard-selected or hard-frozen by an external
  ranker. The rank reshape conditions the *input signal*, not the *decision*.
- It **is** a deliberate departure from pure magnitude-based activity. Rank
  normalization discards absolute magnitude differences among neurons,
  keeping only their ordering. For a frontier study of *which* parameters to
  protect this is appropriate, but it is a modeling choice, recorded here so
  no later reader mistakes it for the original max-norm design.
- The central claim it enables: **"a Pareto improvement on the
  retention–plasticity frontier, traced by a controlled protected-fraction
  axis"** — not "we found a β that made M look healthy."

## What this obligates us to do (pre-registration deltas)

1. The protected fraction becomes a **pre-registered, swept axis** (target
   fraction ∈ {5, 10, 20, 40}% via the break-even / percentile), reported as a
   frontier. We do **not** pick the single best fraction and report it as the
   result. Calibration emits the full \(\tau \mapsto\) protected-fraction map
   so the matrix can place those endpoints; the nearest-50% pick is only the
   *default arm* recommendation.
2. Health check per cell is no longer "BC alone" (M is always bimodal). The
   binding health signal is the **mass split**: a healthy cell has a
   meaningful fraction of M both protected *and* plastic — neither ~90% at
   zero nor ~90% at ceiling. Hard-abort remains empty/NaN/Inf only; BC is a
   warning. Calib scores `n_hard_degenerate` and mass-split separately.
3. Re-validate on real TRACE gradients before the fleet (**init-signal
   calibration**), then a cheap **post-1-epoch-task0 confirmation cell**
   before locking \(\tau^\*/w^\*\) into arms. Init-signal \(\tau^\*/w^\*\) are
   **not** arm defaults until confirmation agrees on mass split.
4. **EMA window \(w\) is a swept axis** alongside \(\tau\): default grid
   \(w \in \{1, 5, 15, 40, 80\}\). Per-point steps = \(\max(150, 10\cdot w)\)
   so high-\(w\) rows are warmed. **Adoptable range:** only \(w \le
   \texttt{MAX\_ADOPTABLE\_EMA\_WINDOW}\) (= 15) may be selected as default-arm
   or EMA-on winner. \(w \in \{40, 80\}\) stay in the measured map (curiosity /
   over-smoothing boundary) but cannot win — high-\(w\) shrinks cross-task std
   by washing signal, which is not a scientific improvement.
   **Verdict rule for EMA-on:** within the adoptable range, adopt \(w>1\) only
   if it **improves mass split**, or achieves a **material** std reduction
   (\(\ge 0.02\)) at a **comparable** protected fraction (within 0.10); std-alone
   / over-smoothed wins do not count. Otherwise record **"rank alone suffices
   on TRACE"**. **Under-populated std:** with \(n_{\mathrm{tasks}} <
   \texttt{MIN\_TASKS\_FOR\_VERDICT}\) (= 2), `protected_frac_std` is `None`
   (unknown), never fake `0.0`; such rows are mapped but not selectable. The
   std term of the adopt rule requires **both** sides to have a real std;
   otherwise decide on mass-split alone. \(\alpha\), `lock_consec_steps`,
   `theta_retain`, `update_interval` stay fixed.

## Status of prior conclusions

- The infrastructure conclusions from the timing run stand: pipeline,
  checkpointing, resume, and B2 finalize are proven.
- The scientific verdict from that run (`beat EWC floor: no`, `nan` deltas) is
  **void as evidence** — it was a single-arm timing cell with a degenerate
  signal, and answered no hypothesis. It is retained only as the trigger for
  this pivot.

## How calibration is run (operational)

Point 3 is a **first-class mode of `run.py`** (same launcher/dispatcher/B2
path as the eval fleet). `benchmarks/trace/calibrate_signal.py` is a thin
wrapper only — no second model-loading path.

- **Scope label:** `calibration_scope: "init_signal"` — fresh LoRA-init
  weights + TRACE batches. Not post-task M.
- **Grid (locked):** \(\tau \in \{0.1\ldots0.7\}\), \(w \in \{1,5,15,40,80\}\);
  one pod per TRACE task; each pod runs the full \(\tau \times w\) grid on
  its shard. Steps per point = \(\max(\texttt{--calib-steps}, 10w)\).
- **Outputs per shard:** `calib_signal_shard<i>.json` + `results_summary.txt`
  (lookup, A>\(\tau\) exceedance, mass split, BC warning).
- **Merge:** `infra/merge_calib_shards.py` / `make merge-calib` unions points,
  rebuilds the 2-D lookup, applies the EMA verdict rule, writes optional
  `signal_tuning.json` for mechanism-arm autowire (`--signal-tuning-json`).
- **No BWT claim.** Seed-of-record guard does not apply.

Commands:

    make run-parallel-calib BENCH=trace GPU=A40 CLOUD=SECURE
    make merge-calib
    # after post-1-epoch-task0 confirmation:
    make run-bg BENCH=trace GPU=A40 CLOUD=SECURE \
      EXTRA_FLAGS='--extra-args --signal-tuning-json lab_reports/_staging/signal_tuning.json'
