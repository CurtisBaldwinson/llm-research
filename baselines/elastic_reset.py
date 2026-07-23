"""Elastic Reset baseline (weight-space / temporal).

Elastic Reset keeps an exponential moving average (EMA) of the weights and,
at each task boundary, "resets" the live weights partway back toward that
EMA anchor:

    theta  <-  (1 - rho) * theta  +  rho * theta_ema        (at each boundary)
    theta_ema  <-  mu * theta_ema  +  (1 - mu) * theta       (each step)

This is a genuine weight-space method: it has NO importance weighting and NO
per-parameter penalty. It fights forgetting by periodically pulling the
model back toward a slow average rather than by anchoring individual
parameters. That structural difference is why the README requires beating a
weight-space method *in addition to* EWC — a mechanism that only beats EWC
might just be a better importance estimate, not a better paradigm.

Reference: "Language Model Alignment with Elastic Reset," 2023.
https://arxiv.org/html/2312.07551v1
"""

from __future__ import annotations

from .base import Arm


class ElasticResetArm(Arm):
    name = "elastic_reset"
    family = "weight_space"

    def __init__(
        self,
        lambda_reg: float = 0.0,   # unused; kept for uniform arm construction
        rho: float = 0.5,          # reset fraction toward EMA at each boundary
        mu: float = 0.99,          # EMA decay
        **kw,
    ):
        super().__init__(lambda_reg=lambda_reg, **kw)
        self.rho = float(rho)
        self.mu = float(mu)
        self._ema = {}
        self._resets = 0

    def on_train_start(self, model) -> None:
        import torch  # noqa: F401  (ensure torch present on this path)
        self._torch = __import__("torch")
        self._ema = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    def on_step(self, model, step: int, batch) -> None:
        torch = self._torch
        with torch.no_grad():
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                e = self._ema.get(n)
                if e is None:
                    self._ema[n] = p.detach().clone()
                else:
                    e.mul_(self.mu).add_(p.detach(), alpha=1.0 - self.mu)

    def on_task_end(self, model, task_idx: int, task_loader) -> None:
        torch = self._torch
        with torch.no_grad():
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                e = self._ema.get(n)
                if e is None:
                    continue
                # theta <- (1-rho) theta + rho theta_ema
                p.mul_(1.0 - self.rho).add_(e, alpha=self.rho)
        self._resets += 1

    def telemetry(self) -> dict:
        t = super().telemetry()
        t.update({"rho": self.rho, "mu": self.mu, "resets": self._resets})
        return t
