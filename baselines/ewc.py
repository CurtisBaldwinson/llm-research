"""Vanilla and EWC/diagonal-Fisher arms.

Both are thin configurations over the sm_core regularizer harness. EWC is
the *classical floor* the mechanism must beat (README Relevance Criterion
#1a). It estimates the diagonal Fisher at each task boundary and anchors
subsequent tasks with a cumulative (online) Fisher sum — the standard
multi-task EWC form, not last-task-only overwrite.

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural
networks," PNAS 2017. https://www.pnas.org/doi/10.1073/pnas.1611835114
"""

from __future__ import annotations

from .base import Arm, DegenerateSignalError


class VanillaArm(Arm):
    """No regularizer. Sanity floor only; never the comparison of record."""

    name = "vanilla"
    family = "none"

    def aux_loss(self, model):
        return None


class EWCArm(Arm):
    """Cumulative (online) diagonal-Fisher EWC, re-anchored at each boundary.

    At the end of each task we (1) estimate the diagonal Fisher on that
    task's data, (2) **add** it into a running-sum Fisher across all prior
    tasks (online/cumulative EWC — not last-task-only overwrite), and
    (3) snapshot the current weights as the new reference. The next task
    trains with  lambda * sum_n F_n (theta_n - theta_ref,n)^2.

    This is the classical floor for Experiment 1. Mechanism arms that only
    keep a last-task M snapshot are intentionally different: M's multi-task
    consolidation is carried by the asymptotic (soft, penalty-based) lock
    dynamics, which is the independent variable under test — not something
    to standardize onto the EWC accumulation form (see design note / README).
    """

    name = "ewc"
    family = "importance_reg"

    def __init__(self, lambda_reg: float = 0.1, fisher_steps: int = 50, **kw):
        super().__init__(lambda_reg=lambda_reg, **kw)
        self.fisher_steps = int(fisher_steps)
        self._reg = None
        self._fisher = {}  # cumulative sum across tasks
        self._n_tasks_accumulated = 0
        self._pad_token_id = kw.get("pad_token_id", None)

    def on_train_start(self, model) -> None:
        # Lazy import so the CPU metric-only path never needs torch/sm_core.
        from synthetic_myelination import (
            MCoupledRegularizer,
            estimate_fisher_diagonal,
            check_degenerate_importance,
        )
        self._MCoupledRegularizer = MCoupledRegularizer
        self._estimate_fisher = estimate_fisher_diagonal
        self._check_degenerate_importance = check_degenerate_importance
        self._fisher = {}
        self._n_tasks_accumulated = 0
        # Per-task hard-check log (None = ok). Survives into cell checkpoints.
        self._fisher_health_log: list = []

    def aux_loss(self, model):
        if self._reg is None:
            return None  # no reference yet (before first task boundary)
        return self._reg.penalty()

    def on_task_end(self, model, task_idx: int, task_loader) -> None:
        # Estimate Fisher on the task we just finished, accumulate, re-anchor.
        task_fisher = self._estimate_fisher(
            model,
            task_loader,
            n_steps=self.fisher_steps,
            pad_token_id=self._pad_token_id,
        )
        # Hard abort on empty / all-zero / NaN / Inf — same set as M hard check.
        viol = self._check_degenerate_importance(task_fisher, name="fisher")
        if viol is not None:
            self._fisher_health_log.append({
                "task_idx": int(task_idx), "stage": "task_fisher", "warning": viol,
            })
            raise DegenerateSignalError(viol, details={
                "arm": self.name,
                "task_idx": task_idx,
                "kind": "fisher",
                "fisher_health_log": list(self._fisher_health_log),
            })

        if not self._fisher:
            self._fisher = {k: v.detach().clone() for k, v in task_fisher.items()}
        else:
            for k, v in task_fisher.items():
                if k in self._fisher:
                    self._fisher[k] = self._fisher[k] + v.detach()
                else:
                    self._fisher[k] = v.detach().clone()
        self._n_tasks_accumulated += 1

        # Renormalize cumulative diagonal to [0, 1] per tensor so λ scale
        # stays comparable as tasks accumulate.
        for k in list(self._fisher.keys()):
            mx = float(self._fisher[k].max().item())
            if mx > 0:
                self._fisher[k] = self._fisher[k] / mx

        viol_c = self._check_degenerate_importance(
            self._fisher, name="fisher_cumulative"
        )
        if viol_c is not None:
            self._fisher_health_log.append({
                "task_idx": int(task_idx),
                "stage": "fisher_cumulative",
                "warning": viol_c,
            })
            raise DegenerateSignalError(viol_c, details={
                "arm": self.name,
                "task_idx": task_idx,
                "kind": "fisher_cumulative",
                "fisher_health_log": list(self._fisher_health_log),
            })

        self._fisher_health_log.append({
            "task_idx": int(task_idx), "stage": "ok", "warning": None,
        })

        self._reg = self._MCoupledRegularizer(
            model=model,
            fisher_diag=self._fisher,
            lambda_reg=self.lambda_reg,
            regularizer_type="fisher",
        )
        self._reg.snapshot_reference()

    def telemetry(self) -> dict:
        t = super().telemetry()
        t["fisher_steps"] = self.fisher_steps
        t["n_fisher_params"] = len(self._fisher)
        t["n_tasks_accumulated"] = self._n_tasks_accumulated
        t["ewc_mode"] = "cumulative_online"
        t["fisher_health_log"] = list(self._fisher_health_log)
        t["signal_health"] = "ok"
        t["importance_kind"] = "fisher_cumulative"
        t["m_distribution_warning"] = None  # N/A for EWC; field kept uniform
        t["m_distribution_warnings"] = []
        return t
