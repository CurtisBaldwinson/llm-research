# Design Note 01 — Temporal fidelity: what EWC is, what M is, and what Experiment 1 actually tests

Status: binding rationale for Experiment 1. Read alongside `README.md`.
Date: 2026-07-23

## The claim this note pins down

A recurring intuition on this project is that EWC (and Fisher-based
consolidation generally) is a "grossly oversimplified" model of
myelination — "a toddler's 2D drawing of a 3D object." This note states
precisely where that intuition is **correct**, where it is a **trap**, and
how Experiment 1 is constructed to test the one difference that matters
without letting the metaphor expand the scope.

## 1. What EWC actually is, mechanically

EWC adds to the task-B loss a quadratic anchor to the end-of-task-A weights:

    L = L_new(θ) + Σ_i (λ/2) · F_i · (θ_i − θ_i*)²

- `θ_i*` — the weights at the instant task A finished.
- `F_i` — the diagonal of the Fisher information, **measured once**, at that
  same instant. It approximates how sharply the task-A loss depends on each
  weight (local curvature / sensitivity).

Two properties follow directly, and both match the "2D shadow" intuition:

1. **It is an instantaneous snapshot, not a temporal process.** `F_i` is a
   single sample at one timestep. It carries no information about *how* a
   weight became important — a weight hammered consistently across all of
   task A and a weight that spiked only in the final steps get identical
   `F_i` if they end at the same curvature. Biological myelination is the
   *integral of activity over time*; Fisher is one frame of a proxy for
   that integral.

2. **It represents importance as a scalar-at-an-instant, not a
   vector-over-time.** Vanilla EWC also does not accumulate cleanly across a
   long task sequence (the online-EWC / Fisher-saturation problem). The
   progressive, path-dependent consolidation the myelin metaphor evokes is
   genuinely absent from the base method.

**Conclusion for the narrow question:** as a model of *myelination over
time*, EWC's Fisher term is a second-order-curvature snapshot, not a
reinforcement integral. The intuition is correct on the mechanics.

## 2. What our M mechanism actually is

`MyelinState.update` integrates a per-neuron activity signal step by step:

    M ← clamp(M + α·A − β·(1 − A),  0, 1)

with an **asymptotic lock**: a neuron whose M stays above `theta_retain`
for `lock_consec_steps` consecutive steps is latched (discrete, bimodal
plasticity). This has exactly the property EWC lacks:

- It is a **temporal integral** of activity, not a single-instant sample.
- It is **path-dependent**: the lock can only fire after *sustained*
  reinforcement, so history matters.
- The lock is a **discrete consolidation event**, not a smooth stiffness.

So the M state machine is, by construction, closer to "myelination over
time" than Fisher is. **We already built the less-reduced model.** That is
the project's actual claim to novelty — not the substitution of M for
Fisher in the penalty (that part is EWC-family), but the *temporal*
components EWC structurally lacks:

    (1) the lock         — discrete/bimodal plasticity from sustained activity
    (2) demyelination    — dynamic release of stale locked parameters

## 3. The trap

The intuition becomes dangerous the moment "more faithful to myelin" is
treated as equivalent to "better at reducing catastrophic forgetting."
They are different claims:

- **Fidelity claim:** M models myelination-over-time more honestly than
  Fisher.  (Plausibly true.)
- **Performance claim:** that extra temporal fidelity *reduces forgetting
  measurably more than Fisher, at a cost worth paying.*  (Unproven — and
  the v1 lab-report history shows M+lock losing to plain diagonal Fisher
  once (imp05 v1.0) and never cleanly beating it.)

A richer, more biologically-honest mechanism that does not beat the simpler
one **on the metric** is worse, not better: it costs more and buys nothing
measurable. Biological faithfulness is not a result.

## 4. What Experiment 1 tests (and why it is the right test)

Experiment 1 isolates the *one* difference between the snapshot view and
the temporal view, holding everything else identical:

- **Importance signal held identical across both arms.** Both the M arm and
  the Fisher arm use the same `grad_l2_per_neuron` gradient-aligned signal.
  This removes "M vs Fisher as importance signals" as a confound — the v1
  work already conflated the signal choice with the mechanism.
- **The only thing that varies is the lock** (discrete temporal
  consolidation) vs. **raw diagonal Fisher** (instantaneous snapshot).
- Metric: **Backward Transfer (BWT)** on the TRACE task sequence at matched
  plasticity. A defensible win is a BWT improvement that clears the
  relevance bar in the README (beats BOTH Fisher/EWC and a weight-space
  method; > 2× per-seed σ at n ≥ 5; Pareto-frontier dominance over a
  contiguous λ region; survives cost adjustment).

If the temporal lock is a real improvement over the instantaneous snapshot,
**this is the experiment that shows it** — cleanly, with nothing else moving.
If it is not, the pre-registered kill criterion applies.

## 5. Pre-training as a myelin scaffold — deferred, and why

A stronger version of the intuition says consolidation should not be bolted
on at task boundaries but should run *continuously throughout learning,
including pre-training*, so the sheath structure reflects the entire
training history by the time task B arrives. Split the claim:

- **Weak form (already embodied):** consolidation needs time and
  reinforcement; you cannot snapshot importance at one instant and call it
  myelin. Our within-task M integral already does this.
- **Strong form (novel, expensive, hard to falsify):** the *pre-training
  phase itself* lays down the scaffold, continuously, rather than at
  discrete task boundaries. No method in the EWC lineage does this.

The strong form is exactly the appealing-but-unbounded direction the
pre-registered kill criterion exists to protect against. **It is deferred
to a possible Experiment 3, gated on Experiment 1.** Rationale: if the lock
does not beat Fisher even in the clean *within-task* comparison of Exp 1,
the continuous-pretraining elaboration is very unlikely to rescue it, and
scaling up compute to chase it would repeat the v1 failure mode. Exp 1
first; Exp 3 only if Exp 1 (and Exp 2) show signal.

## 6. One-line summary

EWC is an instantaneous curvature snapshot; M is a temporal integral with a
discrete lock. The lock is the thing under test. "More like myelin" is not a
result — beating Fisher on BWT at matched cost is. Run Experiment 1 before
letting the pre-training idea expand the scope.
