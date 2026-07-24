"""Synthetic Myelination (Synthetic2 core).

Lean re-port of the tested v1 mechanism: the M state machine with the
asymptotic lock, the activity-capture hooks (max-norm signal, gradient and
activation per-neuron modes), the gate library, and the M-coupled /
Fisher / uniform regularizer with diagonal-Fisher estimation.

Deliberately excludes the v1 metaphor-drift modules (ethos_state,
faith_state, circuit_faith_state, policy_controller, conflict_aware_gating,
inference_controller) — see Synthetic2/README.md "What is being ported".
"""

from .attenuation import (
    adaptive_gate,
    get_gate,
    linear_gate,
    power_gate,
    sigmoid_gate,
    threshold_gate,
)
from .hooks import ActivityCapture
from .batch_utils import (
    check_degenerate_importance,
    check_m_hard_degenerate,
    unpack_lm_batch,
)
from .m_coupled_regularizer import (
    MCoupledRegularizer,
    bimodality_coefficient,
    check_degenerate_M,
    estimate_fisher_diagonal,
)
from .myelin_state import (
    MyelinState,
    auto_assign_roles,
)

__all__ = [
    "ActivityCapture",
    "MCoupledRegularizer",
    "MyelinState",
    "adaptive_gate",
    "auto_assign_roles",
    "bimodality_coefficient",
    "check_degenerate_M",
    "check_degenerate_importance",
    "check_m_hard_degenerate",
    "estimate_fisher_diagonal",
    "get_gate",
    "linear_gate",
    "power_gate",
    "sigmoid_gate",
    "threshold_gate",
    "unpack_lm_batch",
]

__version__ = "0.2.0"
