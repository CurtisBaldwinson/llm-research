"""Model / weight averaging baseline (AMA-style, weight-space).

Maintains a running average of the weights across the task sequence and, at
each task boundary, blends the live weights toward that running average:

    theta_avg  <-  (k * theta_avg + theta) / (k + 1)      (k = tasks seen)
    theta      <-  (1 - alpha) * theta + alpha * theta_avg  (at each boundary)

Like Elastic Reset it is purely weight-space (no importance weighting), but
it is the *cheap, ossification-free* contender: averaging never freezes a
parameter, so it cannot ossify, and it costs almost nothing. The README
calls this out as a strong, cheap baseline — a mechanism that only ties it
loses on the cost-adjustment criterion.

Reference: "Mitigating the Alignment Tax of RLHF" (AMA), EMNLP 2024.
https://aclanthology.org/2024.emnlp-main.35.pdf
"""

from __future__ import annotations

from .base import Arm


class ModelAveragingArm(Arm):
    name = "model_averaging"
    family = "weight_space"

    def __init__(self, lambda_reg: float = 0.0, alpha: float = 0.5, **kw):
        super().__init__(lambda_reg=lambda_reg, **kw)
        self.alpha = float(alpha)   # blend strength toward running average
        self._avg = {}
        self._k = 0

    def on_train_start(self, model) -> None:
        self._torch = __import__("torch")
        self._avg = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self._k = 1

    def on_task_end(self, model, task_idx: int, task_loader) -> None:
        torch = self._torch
        with torch.no_grad():
            # fold finished-task weights into the running average
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                a = self._avg.get(n)
                if a is None:
                    self._avg[n] = p.detach().clone()
                    continue
                a.mul_(self._k).add_(p.detach()).div_(self._k + 1)
            self._k += 1
            # blend live weights toward the average
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                a = self._avg.get(n)
                if a is None:
                    continue
                p.mul_(1.0 - self.alpha).add_(a, alpha=self.alpha)

    def telemetry(self) -> dict:
        t = super().telemetry()
        t.update({"alpha": self.alpha, "tasks_averaged": self._k})
        return t
