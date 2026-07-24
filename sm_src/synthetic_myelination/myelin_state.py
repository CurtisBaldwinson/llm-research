"""MyelinState — the M tensor and policy update rule."""

from __future__ import annotations

import inspect
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .attenuation import get_gate


def auto_assign_roles(model: nn.Module) -> dict[str, str]:
    """Mark the last ``nn.Linear`` as the output head; everything else is representation.

    Returns a ``{param_name: role}`` dict suitable for ``MyelinState(layer_roles=...)``.
    """
    linear_modules = [
        (name, mod) for name, mod in model.named_modules() if isinstance(mod, nn.Linear)
    ]
    if not linear_modules:
        return {}
    head_name, _head_mod = linear_modules[-1]
    roles: dict[str, str] = {}
    head_prefix = (head_name + ".") if head_name else ""
    for pname, _ in model.named_parameters():
        if head_prefix and pname.startswith(head_prefix):
            roles[pname] = "head"
        elif not head_prefix and "." not in pname:
            # Degenerate: head module is the model itself (no parents).
            roles[pname] = "head"
        else:
            roles[pname] = "representation"
    return roles


class MyelinState:
    """Maintains the per-parameter myelination tensor M aligned to a model.

    M is stored as float32 in [0, 1]. For every tracked parameter, a matching
    M tensor is kept. Calling :meth:`update` integrates a per-parameter
    activity signal using the policy update rule from the patent:

        M_new = clamp(M + alpha * A - beta * (1 - A), 0, 1)

    where A is broadcast to the parameter shape if supplied as a scalar.

    v2 extensions (all OFF by default, backward-compatible):
      - Asymptotic soft consolidation lock (``use_asymptotic_lock``): once M
        stays >= theta_retain for ``lock_consec_steps`` consecutive steps,
        the per-parameter floor is raised to theta_retain so M cannot decay
        below it. This raises the *penalty weight floor* in the regularizer
        path — it is **not** a hard freeze / gradient gate.
      - Participation gate (``use_participation_gate``): skip the M update
        entirely for parameters whose mean activity is below
        ``tau_relevance``.
      - Adaptive gate (``gate_fn="adaptive"``): uses the soft-wall gate with
        no fixed epsilon floor.

    v5 extensions:
      - ``signal_type="activation_l2"``: per-neuron M shape [H]; M is lazily
        initialized to match the signal shape on first update().
      - Per-neuron asymptotic lock using torch.where tensor ops.
      - apply_gate() broadcasts M[H] to weight grad [H, in] via unsqueeze(1).

    v7 extensions (Improvement 07 — Gradient-Aligned M):
      - ``signal_type="grad_l2_per_neuron"``: per-neuron M shape [H]; signal
        is derived from gradient L2 norm per output neuron rather than
        activation magnitude.  The MyelinState machinery itself is signal-
        agnostic — the lazy Tensor[H] re-initialization path already in v5
        handles this mode without further code change.  The downstream
        behavior (asymptotic lock, gate broadcasting, role-aware update) is
        identical between activation_l2 and grad_l2_per_neuron; the only
        difference is the activity signal the M state integrates.  See
        ``benchmarks/07_gradient_aligned_m/SPEC_v1.0.md`` and the
        ``ActivityCapture`` docstring in ``hooks.py`` for the signal-
        derivation details.
    """

    def __init__(
        self,
        model: nn.Module,
        gate_fn: str = "sigmoid",
        gate_kwargs: Optional[dict] = None,
        alpha: float = 0.05,
        beta: float = 0.003,
        epsilon: float = 1e-3,
        update_interval: int = 1,
        device: str = "cpu",
        # v2 knobs (all default-off for backward compat):
        use_asymptotic_lock: bool = False,
        use_participation_gate: bool = False,
        theta_retain: float = 0.7,
        lock_consec_steps: int = 20,
        tau_relevance: Optional[float] = None,
        # v3 knobs (forwarded to ActivityCapture by PolicyController):
        normalization: str = "max",
        ema_window: int = 1,
        # v4 knobs: per-parameter role (e.g. "representation" vs "head") with
        # optional per-role alpha/beta overrides. Head params stay at M=0.
        layer_roles: Optional[dict[str, str]] = None,
        layer_alpha: Optional[dict[str, float]] = None,
        layer_beta: Optional[dict[str, float]] = None,
        # v5 knobs:
        signal_type: str = "grad_norm",
    ) -> None:
        self.model = model
        self.gate_name = gate_fn
        self.gate_fn = get_gate(gate_fn)
        self.gate_kwargs = dict(gate_kwargs or {})
        # Only forward `epsilon` when the selected gate actually accepts it.
        if "epsilon" in inspect.signature(self.gate_fn).parameters:
            self.gate_kwargs.setdefault("epsilon", epsilon)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.update_interval = int(update_interval)
        self.device = torch.device(device)

        # v2 config
        self.use_asymptotic_lock = bool(use_asymptotic_lock)
        self.use_participation_gate = bool(use_participation_gate)
        self.theta_retain = float(theta_retain)
        self.lock_consec_steps = int(lock_consec_steps)

        # v3: normalization/ema_window are forwarded to ActivityCapture
        # via PolicyController; stored on self so the controller can read them.
        self.normalization = str(normalization)
        self.ema_window = int(ema_window)

        # v5: signal_type stored so PolicyController can forward it to ActivityCapture
        self.signal_type = str(signal_type)

        # tau_relevance auto-calibration for softmax normalization:
        # uniform softmax weight is 1/N over N tracked parameter tensors.
        # Setting threshold at 0.5/N means "skip any param getting less than
        # half of its fair share of attention".
        n_tracked = sum(
            1 for _name, p in model.named_parameters() if p.requires_grad
        )
        if tau_relevance is None:
            if self.normalization == "softmax" and n_tracked > 0:
                self.tau_relevance = 0.5 / float(n_tracked)
            else:
                self.tau_relevance = 0.1
        else:
            self.tau_relevance = float(tau_relevance)
        # v6: tau_relevance acts as a scale factor; actual per-layer tau = _tau_scale / sqrt(H)
        self._tau_scale: float = self.tau_relevance

        self._step_count: int = 0
        self.M: dict[str, Tensor] = {}
        self._M_floor: dict[str, Tensor] = {}
        # v5: _consec_above_retain is now Tensor[H] for per-neuron signal
        # modes (activation_l2, grad_l2_per_neuron), int for scalar modes.
        self._consec_above_retain: dict[str, int | Tensor] = {}
        # v4: per-parameter role + effective alpha/beta.
        self._param_role: dict[str, str] = {}
        self._param_alpha: dict[str, float] = {}
        self._param_beta: dict[str, float] = {}
        role_alpha = dict(layer_alpha or {})
        role_beta = dict(layer_beta or {})
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.M[name] = torch.zeros_like(
                    param, dtype=torch.float32, device=self.device
                )
                self._M_floor[name] = torch.zeros_like(self.M[name])
                self._consec_above_retain[name] = 0
                role = (layer_roles or {}).get(name, "representation")
                self._param_role[name] = role
                if role == "head":
                    self._param_alpha[name] = role_alpha.get("head", 0.0)
                    self._param_beta[name] = role_beta.get("head", 0.0)
                else:
                    self._param_alpha[name] = role_alpha.get(role, self.alpha)
                    self._param_beta[name] = role_beta.get(role, self.beta)

        # Diagnostic counter: how many (name, step) pairs the participation
        # gate suppressed. Reset externally with `reset_counters()`.
        self.participation_skips: int = 0
        self.lock_activations: int = 0  # how many params have had floor raised

    def state_dict(self) -> dict:
        """Return a serializable dict of M tensors and hyperparameters."""
        return {
            "M": {k: v.detach().clone().cpu() for k, v in self.M.items()},
            "M_floor": {k: v.detach().clone().cpu() for k, v in self._M_floor.items()},
            "consec_above_retain": dict(self._consec_above_retain),
            "alpha": self.alpha,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "gate_name": self.gate_name,
            "gate_kwargs": self.gate_kwargs,
            "update_interval": self.update_interval,
            "step_count": self._step_count,
            "use_asymptotic_lock": self.use_asymptotic_lock,
            "use_participation_gate": self.use_participation_gate,
            "theta_retain": self.theta_retain,
            "lock_consec_steps": self.lock_consec_steps,
            "tau_relevance": self.tau_relevance,
            "participation_skips": self.participation_skips,
            "lock_activations": self.lock_activations,
        }

    def load_state_dict(self, sd: dict) -> None:
        """Load M tensors and hyperparameters produced by :meth:`state_dict`."""
        for k, v in sd["M"].items():
            if k in self.M:
                self.M[k] = v.to(self.device).float()
        if "M_floor" in sd:
            for k, v in sd["M_floor"].items():
                if k in self._M_floor:
                    self._M_floor[k] = v.to(self.device).float()
        if "consec_above_retain" in sd:
            for k, v in sd["consec_above_retain"].items():
                if k not in self._consec_above_retain:
                    continue
                if torch.is_tensor(v):
                    self._consec_above_retain[k] = v.to(
                        self.device, dtype=torch.long
                    )
                elif torch.is_tensor(self._consec_above_retain[k]):
                    self._consec_above_retain[k] = torch.as_tensor(
                        v, dtype=torch.long, device=self.device
                    )
                else:
                    self._consec_above_retain[k] = int(v)
        self.alpha = float(sd.get("alpha", self.alpha))
        self.beta = float(sd.get("beta", self.beta))
        self.epsilon = float(sd.get("epsilon", self.epsilon))
        self.gate_name = sd.get("gate_name", self.gate_name)
        self.gate_fn = get_gate(self.gate_name)
        self.gate_kwargs = dict(sd.get("gate_kwargs", self.gate_kwargs))
        self.update_interval = int(sd.get("update_interval", self.update_interval))
        self._step_count = int(sd.get("step_count", 0))
        self.use_asymptotic_lock = bool(sd.get("use_asymptotic_lock", self.use_asymptotic_lock))
        self.use_participation_gate = bool(sd.get("use_participation_gate", self.use_participation_gate))
        self.theta_retain = float(sd.get("theta_retain", self.theta_retain))
        self.lock_consec_steps = int(sd.get("lock_consec_steps", self.lock_consec_steps))
        self.tau_relevance = float(sd.get("tau_relevance", self.tau_relevance))
        self.participation_skips = int(sd.get("participation_skips", self.participation_skips))
        self.lock_activations = int(sd.get("lock_activations", self.lock_activations))

    def reset_counters(self) -> None:
        """Reset diagnostic counters (participation skips, lock activations)."""
        self.participation_skips = 0
        self.lock_activations = sum(
            1 for name in self.M if self._M_floor[name].max().item() >= self.theta_retain - 1e-9
        )

    def update(self, activity_signals: dict) -> None:
        """Apply the policy update rule to M for each tracked parameter.

        activity_signals: mapping of parameter name to a scalar or tensor
        activity measure A in [0, 1] (e.g. per-layer grad norm normalized).
        Scalars are broadcast to the parameter shape.

        v5: if A is Tensor[H] and M was initialized to param shape, M is
        lazily re-initialized to shape [H] on first call.
        """
        self._step_count += 1
        if self._step_count % self.update_interval != 0:
            return
        for name in self.M:
            # v4: head-role params are never myelinated — M stays at 0.
            if self._param_role.get(name, "representation") == "head":
                if float(self.M[name].max().item()) != 0.0:
                    self.M[name] = torch.zeros_like(self.M[name])
                continue

            A_raw = activity_signals.get(name, None)
            if A_raw is None:
                continue

            if not torch.is_tensor(A_raw):
                A = torch.tensor(float(A_raw), device=self.M[name].device, dtype=torch.float32)
            else:
                A = A_raw.to(device=self.M[name].device, dtype=torch.float32)
            A = torch.clamp(A, 0.0, 1.0)

            # v5: lazy M shape initialization when signal is Tensor[H] (1-D+)
            # Only re-initialize when A is a non-scalar tensor with different shape.
            if A.ndim >= 1 and self.M[name].shape != A.shape:
                self.M[name] = torch.zeros_like(A)
                self._M_floor[name] = torch.zeros_like(A)
                if self.use_asymptotic_lock:
                    # Per-neuron counter as LongTensor
                    self._consec_above_retain[name] = torch.zeros(
                        A.shape[0], dtype=torch.long, device=A.device
                    )
            elif A.ndim == 0 and A.shape != self.M[name].shape:
                # Scalar signal: expand to match existing M shape (original behavior)
                A = A.expand_as(self.M[name])

            M = self.M[name]

            # H2: participation gate — skip this param entirely for this step.
            if self.use_participation_gate:
                H = A.shape[0]
                tau = self._tau_scale / math.sqrt(float(H))
                if float(A.mean().item()) < tau:
                    self.participation_skips += 1
                    continue

            alpha_i = self._param_alpha.get(name, self.alpha)
            beta_i = self._param_beta.get(name, self.beta)
            floor = self._M_floor[name]
            new_M = M + alpha_i * A - beta_i * (1.0 - A)
            new_M = torch.maximum(new_M, floor)
            new_M = torch.clamp(new_M, 0.0, 1.0)
            self.M[name] = new_M

            # H1: asymptotic lock
            if self.use_asymptotic_lock:
                consec = self._consec_above_retain[name]
                if torch.is_tensor(consec):
                    # v5: per-neuron tensor lock (soft penalty floor raise)
                    above = (new_M >= self.theta_retain)
                    new_consec = torch.where(
                        above,
                        consec + 1,
                        torch.zeros_like(consec)
                    )
                    self._consec_above_retain[name] = new_consec
                    lock_mask = new_consec >= self.lock_consec_steps
                    prev_floor = self._M_floor[name]
                    newly_locked = lock_mask & (
                        prev_floor < (self.theta_retain - 1e-9)
                    )
                    self.lock_activations += int(newly_locked.sum().item())
                    self._M_floor[name] = torch.where(
                        lock_mask,
                        torch.full_like(prev_floor, self.theta_retain),
                        prev_floor,
                    )
                else:
                    # Original scalar lock logic (backward compat)
                    mean_M = float(new_M.float().mean().item())
                    if mean_M >= self.theta_retain:
                        self._consec_above_retain[name] += 1
                    else:
                        self._consec_above_retain[name] = 0

                    if (
                        self._consec_above_retain[name] >= self.lock_consec_steps
                        and float(floor.max().item()) < self.theta_retain - 1e-9
                    ):
                        self._M_floor[name] = torch.full_like(M, self.theta_retain)
                        self.lock_activations += 1

    def apply_gate(self, param_name: str, step: Tensor) -> Tensor:
        """Return ``step * g(M[param_name])`` — the ΔW attenuation.

        v4: if the parameter is tagged with role ``"head"``, the gate is an
        identity (``g=1``) regardless of M or the configured gate function.
        This keeps output heads fully plastic even under adaptive gates.

        v5: if M is [H] and step is [H, in] (weight), broadcasts via unsqueeze(1).
        If M is [H] and step is [H] (bias), applies element-wise.

        v6: if M is [C] and step is [C, in, kH, kW] (Conv2d weight),
        broadcasts via unsqueeze(1).unsqueeze(2).unsqueeze(3). This adds CNN
        support without breaking the v5 MLP/transformer paths — the new
        case only fires for 4D step tensors, which v5 callers never produce.
        """
        if param_name not in self.M:
            return step
        if self._param_role.get(param_name, "representation") == "head":
            return step
        M = self.M[param_name]
        gM = self.gate_fn(M, **self.gate_kwargs)
        gM = gM.to(step.device, step.dtype)

        # v5 broadcast: M[H] → weight grad [H, in]
        if step.dim() == 2 and gM.dim() == 1 and gM.shape[0] == step.shape[0]:
            return step * gM.unsqueeze(1)
        # v5 broadcast: M[H] → bias grad [H]
        if step.dim() == 1 and gM.dim() == 1 and gM.shape == step.shape:
            return step * gM
        # v6 broadcast: M[C] → Conv2d weight grad [C_out, C_in, kH, kW]
        if step.dim() == 4 and gM.dim() == 1 and gM.shape[0] == step.shape[0]:
            return step * gM.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # Fallback: shapes already match or M is scalar/param-shaped
        return step * gM

    def get_M(self, param_name: str) -> Tensor:
        """Return a reference to M for the given parameter name."""
        return self.M[param_name]

    def summary_stats(self) -> dict:
        """Return mean/std of M, fractions high/low, floor stats, and per-role means.

        v5: counts locked *neurons* (entries where _M_floor > 0), not parameters.
        Adds std_M_repr to the returned dict.
        """
        if not self.M:
            return {
                "mean_M": 0.0,
                "std_M": 0.0,
                "fraction_high": 0.0,
                "fraction_low": 0.0,
                "mean_floor": 0.0,
                "params_locked": 0,
                "participation_skips": int(self.participation_skips),
                "mean_M_repr": 0.0,
                "std_M_repr": 0.0,
                "mean_M_head": 0.0,
            }

        repr_M_vals: list[float] = []
        head_M_vals: list[float] = []
        n_locked = 0

        for name, M in self.M.items():
            role = self._param_role.get(name, "representation")
            flat = M.flatten().float().tolist()
            if role == "head":
                head_M_vals.extend(flat)
            else:
                repr_M_vals.extend(flat)
            # v5: count locked neurons (entries in _M_floor > 0)
            if self.use_asymptotic_lock and name in self._M_floor:
                n_locked += int((self._M_floor[name] > 0).sum().item())

        all_vals = repr_M_vals + head_M_vals

        def safe_stats(vals: list) -> tuple:
            if not vals:
                return 0.0, 0.0, 0.0, 0.0
            t = torch.tensor(vals, dtype=torch.float32)
            return (
                float(t.mean().item()),
                float(t.std().item()) if len(vals) > 1 else 0.0,
                float((t > 0.7).float().mean().item()),
                float((t < 0.3).float().mean().item()),
            )

        m_repr, s_repr, fh, fl = safe_stats(repr_M_vals)
        m_head, _, _, _ = safe_stats(head_M_vals)
        m_all, s_all, _, _ = safe_stats(all_vals)

        # Compute mean_floor across all floor tensors
        if self._M_floor:
            floor_vals = [v for f in self._M_floor.values() for v in f.flatten().float().tolist()]
            mean_floor = float(sum(floor_vals) / len(floor_vals)) if floor_vals else 0.0
        else:
            mean_floor = 0.0

        # v6: compute per-layer tau values
        tau_per_layer: dict[str, float] = {}
        for name, m_tensor in self.M.items():
            H = m_tensor.shape[0]
            tau_per_layer[name] = self._tau_scale / math.sqrt(float(H))

        return {
            "mean_M": m_all,
            "std_M": s_all,
            "fraction_high": fh,
            "fraction_low": fl,
            "mean_floor": mean_floor,
            "params_locked": n_locked,
            "participation_skips": int(self.participation_skips),
            "mean_M_repr": m_repr,
            "std_M_repr": s_repr,
            "mean_M_head": m_head,
            "tau_per_layer": tau_per_layer,
        }

