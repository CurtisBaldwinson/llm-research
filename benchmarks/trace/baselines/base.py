"""Common arm protocol so the TRACE harness drives every method identically.

An *arm* is one method under test (a mechanism variant or a baseline). The
harness never special-cases an arm; it only calls this interface. That is
what enforces the Experiment-1 discipline in code: the ONLY thing that can
differ between the M arm and the Fisher arm is what happens inside these
hooks, and the harness applies the same training loop, same data, same
seeds, same eval to all of them.

Lifecycle (per seed):

    arm.on_train_start(model)
    for task_idx, task in enumerate(sequence):
        arm.on_task_start(model, task_idx, prev_loader)
        for step, batch in task_loader:
            loss = ce_loss(model, batch)
            loss = loss + arm.aux_loss(model)      # regularizer contribution
            loss.backward();  opt.step()
            arm.on_step(model, step, batch)        # state updates (M, EMA, ...)
        arm.on_task_end(model, task_idx, task_loader)
    arm.on_train_end(model)

Baselines that do nothing at a hook just inherit the no-op default.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class DegenerateSignalError(RuntimeError):
    """Raised when M or Fisher collapses to a mechanism-inert signal.

    The TRACE harness catches this per-cell, writes a
    ``{"degenerate": true, ...}`` checkpoint, and continues the matrix.
    """

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = dict(details or {})


class Arm:
    """Base arm: vanilla behavior (no auxiliary loss, no state)."""

    name: str = "arm"
    #: family label for reporting ("importance_reg" | "weight_space" |
    #: "rehearsal" | "mechanism" | "none")
    family: str = "none"

    def __init__(self, lambda_reg: float = 0.0, **kw: Any) -> None:
        self.lambda_reg = float(lambda_reg)
        self._cfg = dict(kw)

    # ---- lifecycle hooks (all optional; default no-op) -----------------

    def on_train_start(self, model) -> None:
        ...

    def on_task_start(self, model, task_idx: int, prev_loader) -> None:
        ...

    def aux_loss(self, model):
        """Return a differentiable scalar auxiliary loss (or None)."""
        return None

    def on_step(self, model, step: int, batch) -> None:
        ...

    def on_task_end(self, model, task_idx: int, task_loader) -> None:
        ...

    def on_train_end(self, model) -> None:
        ...

    # ---- reporting -----------------------------------------------------

    def telemetry(self) -> dict:
        """Per-arm diagnostic scalars merged into the run record."""
        return {"lambda_reg": self.lambda_reg}
