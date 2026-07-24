"""Vanilla and EWC/diagonal-Fisher arms.

Both are thin configurations over the sm_core regularizer harness. EWC is
the *classical floor* the mechanism must beat (README Relevance Criterion
#1a). It estimates the diagonal Fisher at each task boundary and anchors the
next task to the just-finished weights with the standard quadratic penalty.

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural
networks," PNAS 2017. https://www.pnas.org/doi/10.1073/pnas.1611835114
"""

from __future__ import annotations

from .base import Arm


class VanillaArm(Arm):
    """No regularizer. Sanity floor only; never the comparison of record."""

    name = "vanilla"
    family = "none"

    def aux_loss(self, model):
        return None


class EWCArm(Arm):
    """Diagonal-Fisher EWC penalty, re-anchored at each task boundary.

    At the end of each task we (1) estimate the diagonal Fisher on that
    task's data and (2) snapshot the current weights as the new reference.
    The next task trains with  lambda * sum_n F_n (theta_n - theta_ref,n)^2.

    This is the "raw diagonal Fisher" arm of Experiment 1 when driven with
    the SAME importance signal as the M arm; here it uses the canonical
    Fisher estimate so it also serves as the standalone EWC baseline.
    """

    name = "ewc"
    family = "importance_reg"

    def __init__(self, lambda_reg: float = 0.1, fisher_steps: int = 50, **kw):
        super().__init__(lambda_reg=lambda_reg, **kw)
        self.fisher_steps = int(fisher_steps)
        self._reg = None
        self._fisher = {}

    def on_train_start(self, model) -> None:
        # Lazy import so the CPU metric-only path never needs torch/sm_core.
        from synthetic_myelination import (
            MCoupledRegularizer,
            estimate_fisher_diagonal,
        )
        self._MCoupledRegularizer = MCoupledRegularizer
        self._estimate_fisher = estimate_fisher_diagonal

    def aux_loss(self, model):
        if self._reg is None:
            return None  # no reference yet (before first task boundary)
        return self._reg.penalty()

    def on_task_end(self, model, task_idx: int, task_loader) -> None:
        # Estimate Fisher on the task we just finished, then re-anchor.
        self._fisher = self._estimate_fisher(
            model, task_loader, n_steps=self.fisher_steps
        )
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
        return t
