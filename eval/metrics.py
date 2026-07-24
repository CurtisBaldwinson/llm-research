"""Continual-learning metrics for Synthetic2 experiments.

Phase 1 dependent variable (binding label):
    **BWT under a uniform next-token-accuracy (NTA) proxy.**

All R[i,j] entries and derived BWT/ACC/plasticity numbers in Experiment 1
are computed from pad-masked next-token accuracy, not task-faithful TRACE
metrics (classification accuracy / ROUGE-L / SARI / edit_sim). Faithful
per-task TRACE scoring is deferred to Phase 2. Records and telemetry MUST
carry ``metric_scope: "nta_proxy"`` so a Phase-1 result is never misread
as a TRACE-leaderboard claim.

Definitions otherwise follow Lopez-Paz & Ranzato (GEM, 2017).

Given an accuracy matrix R where R[i, j] is the NTA-proxy score on task j
after finishing training on task i (0-indexed):

    BWT   = (1 / (T-1)) * Σ_{j=0}^{T-2} ( R[T-1, j] - R[j, j] )
    ACC   = (1 / T)     * Σ_{j=0}^{T-1}   R[T-1, j]

BWT < 0 is forgetting; BWT > 0 is positive backward transfer. The mechanism
"reduces forgetting" iff its BWT is *higher* (less negative) than a
baseline's at matched plasticity (matched final target-task NTA proxy).

No third-party deps — pure Python / lists so this imports on the CPU
smoke path without torch.
"""

from __future__ import annotations

from typing import List, Sequence, Dict, Optional


METRIC_SCOPE = "nta_proxy"
METRIC_SCOPE_NOTE = (
    "Phase 1 dependent variable is BWT under a uniform next-token-accuracy "
    "proxy (pad-masked). Faithful per-task TRACE metrics are Phase 2."
)


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

    Negative = forgetting. This is the forgetting metric of record
    (under the NTA proxy — see module docstring).
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
    """Plasticity proxy: NTA on the LAST task right after learning it.

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
    """Mean diagonal: average NTA on each task right after learning it."""
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
        "metric_scope": METRIC_SCOPE,
    }


def mean_std(xs: Sequence[float]) -> Dict[str, float]:
    """Sample mean and sample std (ddof=1 for n>=2; 0 for n<2)."""
    n = len(xs)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0.0}
    m = sum(xs) / n
    if n < 2:
        return {"mean": m, "std": 0.0, "n": float(n)}
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return {"mean": m, "std": var ** 0.5, "n": float(n)}


def exceeds_2sigma(
    mechanism_bwt: Sequence[float],
    baseline_bwt: Sequence[float],
) -> Dict[str, float]:
    """README gate: is the mechanism's BWT improvement > 2x per-seed sigma?

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


def matched_plasticity(
    mechanism_targ: Sequence[float],
    baseline_targ: Sequence[float],
    *,
    tol: float = 0.02,
) -> Dict[str, float]:
    """True if mechanism target-task NTA is within ``tol`` of baseline mean.

    README matched-plasticity axis: a BWT win only counts when plasticity is
    not sacrificed. Default tol=0.02 (2 NTA points).
    """
    m = mean_std(mechanism_targ)
    b = mean_std(baseline_targ)
    delta = m["mean"] - b["mean"]
    ok = 1.0 if abs(delta) <= tol else 0.0
    return {
        "mech_target_acc_mean": m["mean"],
        "base_target_acc_mean": b["mean"],
        "delta_target_acc": delta,
        "tol": tol,
        "matched": ok,
    }


def best_lambda_run(arm_block: dict) -> Optional[tuple]:
    """Return (lam_str, run_dict) with highest mean BWT, or None."""
    runs = arm_block.get("lambda_runs") or {}
    if not runs:
        return None
    return max(runs.items(), key=lambda kv: kv[1]["bwt"]["mean"])
