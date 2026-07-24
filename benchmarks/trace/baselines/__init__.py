"""Mandatory baseline arms for every Synthetic2 experiment.

The README makes these entry tickets, not deferred ablations. Every run
reports results with the full set present and tuned:

    vanilla          - no regularizer (sanity floor only)
    ewc              - diagonal Fisher penalty (the classical floor to beat)
    elastic_reset    - weight-space / temporal (periodic reset toward an EMA)
    model_averaging  - weight-space (AMA-style running weight average)
    replay           - rehearsal (small buffer of prior-task examples)

Each baseline is exposed as a small, uniform *arm spec* so the TRACE harness
can drive them all through one loop. Two of them (ewc, vanilla) are thin
configurations over the sm_core regularizer harness; the two weight-space
methods and replay are genuine, self-contained mechanisms that do NOT reduce
to an M/Fisher penalty — that is the whole point of the "beats a weight-space
method too" relevance criterion.
"""

from .base import Arm, DegenerateSignalError
from .ewc import EWCArm, VanillaArm
from .elastic_reset import ElasticResetArm
from .model_averaging import ModelAveragingArm
from .replay import ReplayArm

# The mandatory set, in README table order.
MANDATORY_BASELINES = [
    "vanilla",
    "ewc",
    "elastic_reset",
    "model_averaging",
    "replay",
]

# Fixed after the weight-space pre-study (see run.py --tune-baselines).
# Do not sweep these in the main Exp 1 matrix.
DEFAULT_ELASTIC_RESET_RHO = 0.5
DEFAULT_MODEL_AVERAGING_ALPHA = 0.5

ARM_REGISTRY = {
    "vanilla": VanillaArm,
    "ewc": EWCArm,
    "elastic_reset": ElasticResetArm,
    "model_averaging": ModelAveragingArm,
    "replay": ReplayArm,
}

__all__ = [
    "MANDATORY_BASELINES",
    "ARM_REGISTRY",
    "Arm",
    "DegenerateSignalError",
    "VanillaArm",
    "EWCArm",
    "ElasticResetArm",
    "ModelAveragingArm",
    "ReplayArm",
    "DEFAULT_ELASTIC_RESET_RHO",
    "DEFAULT_MODEL_AVERAGING_ALPHA",
]
