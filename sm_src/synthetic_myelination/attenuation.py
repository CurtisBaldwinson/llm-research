"""Gate function library g(M).

All gate functions map a myelination tensor M in [0, 1] to an attenuation
factor in [epsilon, 1] that multiplies the gradient/update. Every gate is
monotonically non-increasing in M and clamped to a strictly positive floor
`epsilon`, so gradients are never completely blocked.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


def linear_gate(M: Tensor, epsilon: float = 1e-3) -> Tensor:
    """g(M) = max(epsilon, 1 - M)."""
    return torch.clamp(1.0 - M, min=epsilon, max=1.0)


def power_gate(M: Tensor, p: float = 2.0, epsilon: float = 1e-3) -> Tensor:
    """g(M) = max(epsilon, (1 - M) ** p)."""
    return torch.clamp((1.0 - M) ** p, min=epsilon, max=1.0)


def sigmoid_gate(
    M: Tensor, k: float = 10.0, m: float = 0.5, epsilon: float = 1e-3
) -> Tensor:
    """g(M) = max(epsilon, sigmoid(-k * (M - m)))."""
    return torch.clamp(torch.sigmoid(-k * (M - m)), min=epsilon, max=1.0)


def threshold_gate(
    M: Tensor,
    theta_lo: float = 0.3,
    theta_hi: float = 0.7,
    epsilon: float = 1e-3,
) -> Tensor:
    """Hysteresis-style threshold gate.

    M < theta_lo  -> 1.0
    M > theta_hi  -> epsilon
    otherwise     -> linear interpolation between 1.0 and epsilon
    """
    if theta_hi <= theta_lo:
        raise ValueError("threshold_gate requires theta_hi > theta_lo")
    out = torch.empty_like(M)
    low_mask = M < theta_lo
    high_mask = M > theta_hi
    mid_mask = ~(low_mask | high_mask)
    out[low_mask] = 1.0
    out[high_mask] = epsilon
    span = theta_hi - theta_lo
    frac = (M[mid_mask] - theta_lo) / span
    out[mid_mask] = 1.0 - (1.0 - epsilon) * frac
    return torch.clamp(out, min=epsilon, max=1.0)


def adaptive_gate(M: Tensor, p: float = 3.0) -> Tensor:
    """Adaptive epsilon gate: g(M) = (1 - M)^p with NO fixed floor.

    The floor shrinks to 0 as M -> 1 (g(1.0) = 0 exactly). For M=0.9, p=3 the
    gate is 1e-3; for M=0.99 it is ~1e-6. This is the v2 "soft wall" —
    functionally locks mastered parameters without a hard epsilon cliff.

    Patent embodiment: photonic analog attenuator drive -> 0 as stability -> max.
    """
    return (1.0 - M).clamp(min=0.0) ** p


_GATES: dict[str, Callable[..., Tensor]] = {
    "linear": linear_gate,
    "power": power_gate,
    "sigmoid": sigmoid_gate,
    "threshold": threshold_gate,
    "adaptive": adaptive_gate,
}


def get_gate(name: str) -> Callable[..., Tensor]:
    """Factory returning a gate function by name."""
    if name not in _GATES:
        raise KeyError(
            f"Unknown gate '{name}'. Valid: {sorted(_GATES.keys())}"
        )
    return _GATES[name]
