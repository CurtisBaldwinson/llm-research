# Synthetic2 — Myelin-Inspired Plasticity for Continual Learning

**Started:** 2026-07-22
**Predecessor:** `~/code/curtis/Synthetic` (retired — see "Why a clean start" below)
**Status:** Pre-experiment. This README is the anchor document. It is written *before* any experiment runs, and the Relevance Criteria and Kill Criterion below are binding.

---

## Thesis

The synthetic-myelination mechanism is, in its penalty form, a member of the
importance-weighted parameter-regularization family (EWC / MAS / SI): it applies
`Σ_p w(p)·(θ_p − θ_ref,p)²` with `w = M` instead of `w = Fisher`. That substitution
alone is **not** a contribution — the literature shows these importance measures are
approximations of one another.

Synthetic2 tests the **only two components of the myelin paradigm that EWC
structurally lacks**:

1. **The lock (asymptotic soft, penalty-based consolidation).** EWC's penalty is a
   continuous spring on every parameter. Myelination is closer to a discrete state
   change: a parameter is either plastic or consolidated. Hypothesis: an
   **asymptotic (soft, penalty-based) consolidation lock** that raises the
   importance-weight floor on a small, well-chosen subset — leaving the rest at low
   penalty — attacks EWC's ossification failure mode better than a purely continuous
   Fisher penalty does. Phase 1 does **not** hard-freeze parameters or gate gradients
   (`apply_gate` is deferred); the lock only changes the quadratic penalty weights.

2. **Demyelination (dynamic release).** Real myelin is *removed* and re-laid during
   learning. EWC only ever adds constraints; it has no mechanism to release a
   previously-important parameter when the task distribution shifts, which is why it
   ossifies over long task sequences. Hypothesis: an unlock rule that releases stale
   locked parameters preserves plasticity on late tasks better than monotonic EWC.

**If neither the lock nor demyelination beats a strong baseline under the Relevance
Criteria below, the myelin paradigm adds nothing over EWC and the program is retired.**
That is the whole point of Synthetic2: find out cheaply and honestly.

---

## Why a clean start (what went wrong in Synthetic v1)

The predecessor project had excellent process discipline but a structural flaw: it
optimized for keeping a weak effect alive rather than killing it. Concretely —

- The mechanism was only ever compared to **vanilla** (or, once, to a Fisher ablation
  that **beat it**). The strong-baseline comparison (EWC / replay / TRACE) was
  specified in Phase 3 and never run.
- Wins existed only on synthetic toy tasks (permuted-vocab GPT-2, Permuted-MNIST).
  The one standard benchmark attempted (CIFAR-100) was REFUTED.
- The single surviving positive result (KTO PP16) was seed-unstable, non-monotone in
  its own λ, and — per the PP20 ablation — possibly a property of KTO, not of M.
- A 15-step version ladder (v1.0 → v3.1) with escape-hatch verdicts and a mid-stream
  threshold amendment let a null-to-tiny effect survive ~109 compute-hours.
- There was **no program-level kill criterion**, so the program had no way to end.

Synthetic2 fixes each of these by construction (see below).

---

## Relevance Criteria (binding)

A result is *relevant* — i.e., worth reporting as a win — only if it meets **all four**:

1. **Beats the real baselines, not vanilla.** It must beat **both** (a) EWC / diagonal
   Fisher and (b) at least one weight-space method (Elastic Reset or model-averaging).
   Beating only vanilla is a null result.

2. **Exceeds seed noise.** The improvement must be **> 2× the per-seed standard
   deviation**, measured at **n ≥ 5 seeds**. Single-seed results are preflights only,
   never evidence.

3. **Pareto-frontier dominance, not a tuned point.** Sweep the regularization strength
   (λ for penalties, α for averaging) for **every** arm. The claim is valid only if the
   mechanism's stability–plasticity frontier **dominates the baselines over a
   contiguous, non-trivial region** of the trade-off — not at a single hand-tuned
   operating point.

4. **Survives cost adjustment.** Log wall-clock and $/run per arm. If the mechanism only
   ties a cheaper method, that is a loss. It must win on the capability/forgetting axis
   by enough to justify its overhead, or match it at lower cost.

**Metric units.**
- Continual learning (TRACE): **Backward Transfer (BWT)** as the forgetting metric;
  target-task accuracy as the plasticity metric. A defensible win is a **≥3–5 point BWT
  improvement at matched plasticity** (and always > 2σ).
- Alignment tax (KTO/DPO): capability retention (MMLU/ARC/TruthfulQA drop) vs
  reward/target performance, compared as **frontier dominance**, never single points.

---

## Pre-Registered Kill Criterion (binding)

> **The myelin paradigm is retired if:**
> Gradient-aligned M (= EWC importance signal + lock dynamics) does **not** beat raw
> diagonal Fisher on TRACE by **> 2× per-seed σ in BWT at matched target accuracy,
> n ≥ 5**, in Experiment 1 — **AND** — the demyelination/unlock rule does **not** beat
> monotonic EWC on late-task plasticity in Experiment 2 (with the gap widening as
> sequence length grows).
>
> If both fail, the lock and demyelination add nothing over EWC. Publish the
> comparative/negative result and stop. Do **not** open a v2.1, add a new decision-tree
> branch, or amend a threshold to rescue the effect.

Amending this criterion after seeing results is itself a program-failure signal. Any
change must be dated, justified in writing here, and made *before* the run it affects.

---

## Mandatory Baselines (every experiment, from run #1)

These are entry tickets, not deferred ablations. No experiment reports a result without
them present and tuned:

| Baseline | Family | Role |
|---|---|---|
| **vanilla** (no regularizer) | — | Sanity floor only; never the comparison of record |
| **EWC / diagonal Fisher** | importance-weighted reg. | The classical floor; the mechanism must beat this |
| **Elastic Reset** | weight-space / temporal | Strong alignment-tax contender |
| **Model / weight averaging** (AMA-style) | weight-space | Strong, cheap, ossification-free contender |
| **replay** (small buffer) | rehearsal | Standard CL reference on TRACE |

M-variant arms (`m_coupled`, `gradient_aligned_m`, `m_with_unlock`) are compared against
this full set, per-seed, with medians and > 2σ gating.

---

## Experiment Scope (only these two until one shows signal)

### Experiment 1 — Does the lock do anything?
**Question:** Does the M state machinery (asymptotic soft consolidation lock, bimodal
distribution) add value over raw diagonal Fisher when both use the *same*
gradient-derived importance signal?

- **Design (Phase-1 isolation screen):** `gradient_aligned_m` (same signal + asymptotic
  soft lock, **task-boundary** M build) **vs** `raw_fisher_signal` (same signal, no lock).
  The *only* structural variable is `use_asymptotic_lock`. This is an isolation screen,
  not a claim of continuous-during-training consolidation.
- **Phase-2 arm (registered, not default matrix):** `gradient_aligned_m_online` updates
  M/lock from `on_step` so consolidation is continuous and activity-dependent.
- **Dependent variable (Phase 1):** **BWT under a uniform next-token-accuracy (NTA)
  proxy** (pad-masked). Faithful per-task TRACE metrics (accuracy / ROUGE-L / SARI /
  edit_sim) are deferred to Phase 2. Records carry `metric_scope: "nta_proxy"`.
- **Benchmark:** TRACE task sequence (standard CL-for-LLMs data). Not a synthetic toy.
- **Arms:** gradient_aligned_m, raw_fisher_signal, cumulative EWC, + mandatory baselines.
  n ≥ 5 seeds, full λ sweep. Weight-space knobs (`rho`, `alpha`) are fixed after a short
  `--tune-baselines` pre-study — not swept in the main matrix.
- **M is not cumulative across tasks** the way online-EWC Fisher is; multi-task
  consolidation for M is carried by the asymptotic lock dynamics (the IV under test).
- **Relevant win:** all README relevance criteria in `decide()` (beat control, beat EWC,
  beat a weight-space method, matched plasticity, contiguous λ region).
- **If it fails:** the lock is inert. Half the kill criterion is met.

### Experiment 2 — Does demyelination beat ossification?
**Question:** Does an unlock/release rule preserve late-task plasticity better than
monotonic EWC over a long task sequence, where EWC is known to ossify?

- **Design:** M-with-unlock (locked params can be *released* when their reinforcement
  signal decays) **vs** monotonic EWC, over an **8+ task** sequence.
- **Relevant win:** higher plasticity (target-task accuracy on task N) at matched
  forgetting, with the **gap widening as N grows**. A flat gap = re-tuned λ, not a
  solved ossification problem.
- **If it fails:** demyelination adds nothing. Other half of the kill criterion met.

**Deferred until Exp 1 or 2 shows signal:** Hebbian / pathway-level M (Option 2 from the
v1 future-ideas memo). Interesting but expensive (O(H²)); it waits.

---

## What is being ported from Synthetic v1

**Infrastructure (port as-is — no scientific bias):**
- Cloud/GPU dispatch: `scripts/run_on_cloud.py`, `scripts/run_on_runpod.py`,
  `scripts/overnight_dispatch.sh`, `scripts/launch_parallel_sweep.py`,
  `scripts/monitor_run.py`, `scripts/pod_finalize.py`, `scripts/finalize_run.py`
- Cloud provider adapters: `scripts/cloud_providers/`
- Results persistence: B2 / Backblaze pivot (from dispatch layer)
- Report tooling: `LAB_REPORT_STANDARD.md`, `scripts/generate_lab_report.py`,
  `scripts/verify_report.py`, `scripts/lab_report_gen/`
- Eval harness wiring: lm-evaluation-harness integration + capability eval
  (`benchmarks/*/eval_capabilities.py`, `eval_utils.py`), IRA computation
- `COMPUTE_LEDGER.md` habit

**Mechanism (port selectively, bugs fixed FIRST):**
- `sm_src/synthetic_myelination/myelin_state.py` — M tensor, EMA update, lock dynamics
  (the novel part; keep)
- `sm_src/synthetic_myelination/m_coupled_regularizer.py` — regularizer *harness*, with
  `m_coupled` / `fisher` / `uniform` modes promoted to first-class arms
- **Bug fixes required before first run** (from v1 D1 review brief):
  1. L2-norm normalization collapse on wide tensors (apply the max-norm patch)
  2. LoRA-A zero-gradient path (PEFT zero-init on B leaves half of trainable params M=0)

**Deliberately NOT ported:**
- The DPO arc, the seven M-variants, the v1.0→v3.1 version ladder
- Patent-continuation framing as an experiment driver
- Escape-hatch verdicts (DIAGNOSTIC PASS, partial-PASS footnotes, threshold amendments)

---

## Proposed layout

```
Synthetic2/
  README.md              # this file — thesis, relevance criteria, kill criterion
  infra/                 # ported dispatch, cloud providers, B2, monitoring
  sm_core/               # myelin_state.py (lock), regularizer harness — BUGS FIXED
  baselines/             # ewc.py, elastic_reset.py, model_averaging.py, replay.py
  benchmarks/
    trace/               # Exp 1 + Exp 2 — the real turf
  eval/                  # lm-eval-harness wiring, capability + BWT metrics
  lab_reports/           # same format/standard, fresh history
  COMPUTE_LEDGER.md
```

---

## Operating rules (carried forward, tightened)

- Predeclare hypotheses and thresholds in the SPEC before the run; append-only lab
  reports; supersedes graph; compute ledger updated every run.
- **Single-seed = preflight, never evidence.** All verdicts require n ≥ 5.
- **No new verdict keywords, no new decision-tree branches, no threshold amendments**
  to rescue an effect. The verdict vocabulary is fixed at project start.
- Every run includes the full mandatory-baseline set. A run without EWC + a weight-space
  method is invalid by construction.
- Report frontiers, not points. Report deltas relative to 2σ, not absolute.

---

*References grounding the baseline choices and thresholds:
[EWC — Kirkpatrick et al., PNAS 2017](https://www.pnas.org/doi/10.1073/pnas.1611835114);
[Unifying Regularisation Methods for Continual Learning, AISTATS 2022](https://openreview.net/pdf/4c9a9c1dbe5bbcd060a6f7a170b10bc17538cdde.pdf);
[TRACE benchmark, ICLR](https://openreview.net/pdf?id=xelrLobW0n);
[Mitigating the Alignment Tax of RLHF (model averaging / AMA), EMNLP 2024](https://aclanthology.org/2024.emnlp-main.35.pdf);
[Language Model Alignment with Elastic Reset, 2023](https://arxiv.org/html/2312.07551v1).*
