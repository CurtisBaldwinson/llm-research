"""Continual-learning metrics for Synthetic2 experiments.

The metric of record for Experiment 1 (per README) is **Backward Transfer
(BWT)** as the forgetting measure, with target-task accuracy as the
plasticity measure. Definitions follow Lopez-Paz & Ranzato (GEM, 2017) and
the TRACE benchmark conventions.

Given an accuracy matrix R where R[i, j] is the accuracy on task j after
finishing training on task i (0-indexed, i >= j means task j was seen at or
before step i):

    BWT   = (1 / (T-1)) * Σ_{j=0}^{T-2} ( R[T-1, j] - R[j, j] )
    ACC   = (1 / T)     * Σ_{j=0}^{T-1}   R[T-1, j]
    FWT   = (1 / (T-1)) * Σ_{j=1}^{T-1} ( R[j-1, j] - b[j] )   # needs baseline b

BWT < 0 is forgetting; BWT > 0 is positive backward transfer. The mechanism
"reduces forgetting" iff its BWT is *higher* (less negative) than a
baseline's at matched plasticity (matched final target-task accuracy).

No third-party deps — pure Python / lists so this imports on the CPU
smoke path without torch.
"""

from __future__ import annotations

from typing import List, Sequence, Dict


def _as_matrix(R: Sequence[Sequence[float]]) -> List[List[float]]:
    M = [list(row) for row in R]
    T = len(M)
    for row in M:
        if len(row) != T:
            raise ValueError(
                f"accuracy matrix must be square (T x T); got {T} rows, "
                f"a row of length {len(row)}"
            )
    return M


def backward_transfer(R: Sequence[Sequence[float]]) -> float:
    """BWT = mean over earlier tasks of (final acc - acc right after learning).

    Negative = forgetting. This is the forgetting metric of record.
    """
    M = _as_matrix(R)
    T = len(M)
    if T < 2:
        return 0.0
    last = M[T - 1]
    return sum(last[j] - M[j][j] for j in range(T - 1)) / (T - 1)


def average_accuracy(R: Sequence[Sequence[float]]) -> float:
    """ACC = mean final-row accuracy across all tasks (overall capability)."""
    M = _as_matrix(R)
    T = len(M)
    if T == 0:
        return 0.0
    last = M[T - 1]
    return sum(last[j] for j in range(T)) / T


def target_task_accuracy(R: Sequence[Sequence[float]]) -> float:
    """Plasticity proxy: accuracy on the LAST task right after learning it.

    This is R[T-1, T-1] — how well the model learned the newest task. Used
    as the "matched plasticity" axis: a forgetting win only counts if it is
    achieved without sacrificing this number (README Relevance Criterion:
    ">= 3-5 pt BWT improvement at matched plasticity").
    """
    M = _as_matrix(R)
    T = len(M)
    if T == 0:
        return 0.0
    return M[T - 1][T - 1]


def learning_accuracy(R: Sequence[Sequence[float]]) -> float:
    """Mean diagonal: average accuracy on each task right after learning it.

    A broader plasticity measure than target_task_accuracy (all tasks, not
    just the last).
    """
    M = _as_matrix(R)
    T = len(M)
    if T == 0:
        return 0.0
    return sum(M[j][j] for j in range(T)) / T


def summarize(R: Sequence[Sequence[float]]) -> Dict[str, float]:
    """All metrics for one accuracy matrix (one arm, one seed)."""
    return {
        "bwt": backward_transfer(R),
        "acc": average_accuracy(R),
        "target_task_acc": target_task_accuracy(R),
        "learning_acc": learning_accuracy(R),
        "n_tasks": float(len(R)),
    }


def mean_std(xs: Sequence[float]) -> Dict[str, float]:
    """Sample mean and (population) std, for per-seed aggregation."""
    n = len(xs)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0.0}
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n
    return {"mean": m, "std": var ** 0.5, "n": float(n)}


def exceeds_2sigma(
    mechanism_bwt: Sequence[float],
    baseline_bwt: Sequence[float],
) -> Dict[str, float]:
    """README gate #2: is the mechanism's BWT improvement > 2x per-seed sigma?

    Returns the delta, the pooled per-seed sigma, the ratio delta/sigma, and
    a pass flag (1.0 iff delta > 2 * sigma AND delta > 0). Single-seed inputs
    are preflight only and never pass (sigma undefined -> pass=0).
    """
    m = mean_std(mechanism_bwt)
    b = mean_std(baseline_bwt)
    delta = m["mean"] - b["mean"]
    # pooled per-seed sigma (the noise the improvement must clear)
    sigma = ((m["std"] ** 2 + b["std"] ** 2) / 2.0) ** 0.5
    ratio = (delta / sigma) if sigma > 0 else 0.0
    passes = 1.0 if (sigma > 0 and delta > 0 and delta > 2 * sigma
                     and m["n"] >= 5 and b["n"] >= 5) else 0.0
    return {
        "delta_bwt": delta,
        "per_seed_sigma": sigma,
        "delta_over_sigma": ratio,
        "passes_2sigma_n5": passes,
    }
