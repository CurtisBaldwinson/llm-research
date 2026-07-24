"""Regression tests for the PEFT-init signal-coverage bug (v1.1 post-mortem).

Documents and pins two coupled failure modes that the 2026-05-21 consultant
review identified, and verifies that the new ``check_invariant_D5`` catches
the resulting degenerate M distribution.

The bug:
    PEFT initializes ``lora_B.weight = 0`` and ``lora_A.weight``
    Kaiming-uniform.  This produces a structural asymmetry that the
    synthetic_myelination activity-capture pipeline misreads:

    * Under ``signal_type="activation_l2"`` the hook fires on each
      ``nn.Linear`` submodule's forward output.  ``lora_B``'s output is
      ``lora_B.weight @ z = 0`` for any input ``z`` because ``lora_B.weight``
      is zero at init.  Phase A does no optimizer steps, so the signal stays
      at zero forever and ``M(lora_B) = 0`` for every lora_B parameter.

    * Under ``signal_type="grad_l2_per_neuron"`` the hook fires on each
      parameter's gradient.  ``∂L/∂lora_A = lora_B^T · ∂L/∂y · x^T = 0``
      whenever ``lora_B = 0``.  So ``M(lora_A) = 0`` instead.

    Same root cause, opposite affected factor, same downstream effect:
    M coverage is asymmetric and the M-coupled regularizer in Phase B
    becomes lopsided rather than per-parameter as intended.

The fix is non-trivial (warmup step or per-pathway M); these tests pin
the BUG so we can detect when a future fix actually closes it, and
verify that ``check_invariant_D5`` raises a useful error on the
degenerate distribution that the bug produces.

Tests run in seconds on CPU; no HF downloads, no GPU.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Make sm_src importable from this test file's location.
ROOT = Path(__file__).resolve().parents[2]
SM_SRC = ROOT / "sm_src"
if str(SM_SRC) not in sys.path:
    sys.path.insert(0, str(SM_SRC))

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402  (after importorskip)


# ----------------------------------------------------------------------------
# Synthetic PEFT-style layer
# ----------------------------------------------------------------------------

class FakeLoraLayer(nn.Module):
    """Minimal PEFT-LoRA-shaped wrapper.

    Mirrors PEFT's structure: ``lora_A`` and ``lora_B`` are ``nn.ModuleDict``
    holding a single ``"default"`` ``nn.Linear`` each.  Initialization
    matches PEFT exactly: lora_A.weight is Kaiming-uniform, lora_B.weight is
    zero.  Forward applies them sequentially.

    PyTorch's ``named_parameters()`` on this module yields names like
    ``lora_A.default.weight`` and ``lora_B.default.weight``, which is what
    the ``_categorize_param_name`` substring matcher expects.
    """

    def __init__(self, in_features: int, out_features: int, r: int = 4):
        super().__init__()
        self.lora_A = nn.ModuleDict({
            "default": nn.Linear(in_features, r, bias=False),
        })
        self.lora_B = nn.ModuleDict({
            "default": nn.Linear(r, out_features, bias=False),
        })
        # PEFT init exactly.
        nn.init.kaiming_uniform_(self.lora_A["default"].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B["default"].weight)

    def forward(self, x):
        return self.lora_B["default"](self.lora_A["default"](x))


# ----------------------------------------------------------------------------
# Bug pin #1: activation_l2 yields zero signal for lora_B
# ----------------------------------------------------------------------------

def test_activation_l2_lora_B_signal_is_zero_at_peft_init():
    """Under PEFT's lora_B=0 init, the activation_l2 hook captures exactly
    zero for every lora_B module."""
    from synthetic_myelination import ActivityCapture

    torch.manual_seed(0)
    model = FakeLoraLayer(in_features=4, out_features=8, r=2)
    activity = ActivityCapture(
        model,
        signal_type="activation_l2",
        normalization="max",
        ema_window=5,
    )
    activity.register()
    try:
        x = torch.randn(3, 4)
        _ = model(x)
        signals = activity.get_signals()
    finally:
        activity.remove()

    # Both lora_A and lora_B parameters should have signals stored.
    assert "lora_A.default.weight" in signals
    assert "lora_B.default.weight" in signals

    lora_A_sig = signals["lora_A.default.weight"]
    lora_B_sig = signals["lora_B.default.weight"]

    # lora_B's signal must be exactly zero (the bug we're pinning).
    assert torch.is_tensor(lora_B_sig)
    assert float(lora_B_sig.abs().max().item()) == 0.0, (
        f"Expected lora_B signal == 0 at PEFT init (bug), "
        f"got max abs = {lora_B_sig.abs().max().item()}"
    )

    # lora_A's signal should be non-zero (control: confirms the hook works).
    assert torch.is_tensor(lora_A_sig)
    assert float(lora_A_sig.abs().max().item()) > 0.0, (
        f"Expected lora_A signal > 0 (sanity check; if this fails the test "
        f"setup is wrong, not the bug)"
    )


# ----------------------------------------------------------------------------
# Bug pin #2: grad_l2_per_neuron yields zero signal for lora_A
# ----------------------------------------------------------------------------

def test_grad_per_neuron_lora_A_signal_is_zero_at_peft_init():
    """Under PEFT's lora_B=0 init, the grad_l2_per_neuron hook captures
    exactly zero for every lora_A parameter, because the gradient flowing
    back through lora_B (which has weight=0) is zero."""
    from synthetic_myelination import ActivityCapture

    torch.manual_seed(0)
    model = FakeLoraLayer(in_features=4, out_features=8, r=2)
    activity = ActivityCapture(
        model,
        signal_type="grad_l2_per_neuron",
        normalization="max",
        ema_window=5,
    )
    activity.register()
    try:
        x = torch.randn(3, 4)
        y = model(x)
        # Loss must have a non-zero gradient w.r.t. y for the dual bug to
        # manifest cleanly.  With `y.pow(2).mean()` the gradient is
        # 2y/N = 0 because y itself is zero (lora_B.weight = 0), which
        # makes BOTH lora_A and lora_B gradients zero and doesn't isolate
        # the asymmetry.  Use (y - target)^2 with a non-zero target so
        # dL/dy = 2(y - target)/N ≠ 0; then dL/dlora_B = a^T @ dL/dy ≠ 0
        # (because lora_A's forward output a is non-zero) but
        # dL/dlora_A = (lora_B^T @ dL/dy) @ x^T = 0 because lora_B = 0.
        target = torch.randn_like(y)
        loss = (y - target).pow(2).mean()
        loss.backward()
        signals = activity.get_signals()
    finally:
        activity.remove()

    assert "lora_A.default.weight" in signals
    assert "lora_B.default.weight" in signals

    lora_A_sig = signals["lora_A.default.weight"]
    lora_B_sig = signals["lora_B.default.weight"]

    # lora_A's gradient is zero because it flows back through lora_B (=0).
    assert torch.is_tensor(lora_A_sig)
    assert float(lora_A_sig.abs().max().item()) == 0.0, (
        f"Expected lora_A grad signal == 0 at PEFT init (bug), "
        f"got max abs = {lora_A_sig.abs().max().item()}"
    )

    # lora_B's gradient is non-zero (ones output → non-zero grad).
    assert torch.is_tensor(lora_B_sig)
    assert float(lora_B_sig.abs().max().item()) > 0.0, (
        f"Expected lora_B grad signal > 0 (sanity check; if this fails the "
        f"test setup is wrong, not the bug)"
    )


# ----------------------------------------------------------------------------
# Mock MyelinState for D5 tests (avoids dragging in the full MyelinState
# constructor and its model-tracking machinery).
# ----------------------------------------------------------------------------

class _MockMyelin:
    """Minimal MyelinState-shaped object exposing the attributes
    ``check_degenerate_M`` reads."""

    def __init__(self, M_dict, role_dict=None):
        self.M = M_dict
        self._param_role = role_dict or {}


def _build_v11_shaped_M(
    n_layers: int = 28,
    r: int = 16,
    out_features: int = 1024,
) -> dict:
    """Build an M dict shaped like v1.1's degenerate distribution.

    All ``lora_B`` params are zero (the bug).  All ``lora_A`` params are
    1.0 (locked, post-asymptotic-lock).  With the default Llama-shaped
    budget (r=16 vs out_features=1024, i.e. lora_A ~1.5% of elements),
    ``mean(M)`` lands well below the 0.10 degenerate-bimodal threshold,
    matching the v1.1 observed pathology.
    """
    M = {}
    for layer_idx in range(n_layers):
        # lora_A: shape [r], all 1.0 (locked).
        lora_A_name = f"model.layers.{layer_idx}.q_proj.lora_A.default.weight"
        M[lora_A_name] = torch.ones(r)
        # lora_B: shape [out_features], all 0 (degenerate).
        lora_B_name = f"model.layers.{layer_idx}.q_proj.lora_B.default.weight"
        M[lora_B_name] = torch.zeros(out_features)
    return M


def _build_healthy_M(n_layers: int = 8, r: int = 16, out_features: int = 128) -> dict:
    """Build a healthy M distribution: both lora_A and lora_B around 0.5,
    bimodal across the two halves of each tensor (lock vs no-lock)."""
    M = {}
    for layer_idx in range(n_layers):
        # lora_A: half at 1.0 (locked), half at 0.0 (unlocked).
        a = torch.zeros(r)
        a[: r // 2] = 1.0
        M[f"model.layers.{layer_idx}.q_proj.lora_A.default.weight"] = a
        # lora_B: same pattern.
        b = torch.zeros(out_features)
        b[: out_features // 2] = 1.0
        M[f"model.layers.{layer_idx}.q_proj.lora_B.default.weight"] = b
    return M


# ----------------------------------------------------------------------------
# Degenerate-M guard tests (setup-agnostic replacement for v1's D5)
# ----------------------------------------------------------------------------

def test_degenerate_M_catches_v11_degenerate_distribution():
    """The guard must raise on the v1.1-shaped distribution (0.4% locked at
    1, 99.6% stuck at 0): bimodal (BC ≈ 1) but mean(M) ≈ 0.004 << 0.10, so
    the 'degenerate-bimodal' branch fires."""
    from synthetic_myelination.m_coupled_regularizer import check_degenerate_M

    myelin = _MockMyelin(_build_v11_shaped_M())
    result = check_degenerate_M(myelin)

    assert result is not None, (
        "guard must flag v1.1-shaped degenerate distribution as a violation"
    )
    assert "VIOLATION" in result
    assert "degenerate-bimodal" in result, (
        f"Expected violation message to identify the degenerate-bimodal "
        f"failure mode; got: {result}"
    )


def test_degenerate_M_catches_unimodal_collapse():
    """The guard must raise when M never developed a bimodal lock
    distribution (all mass near a single value -> BC ≤ bc_min)."""
    from synthetic_myelination.m_coupled_regularizer import check_degenerate_M

    # All M near zero with tiny noise: unimodal, no lock ever formed.
    torch.manual_seed(0)
    M = {
        "model.layers.0.q_proj.weight": (torch.rand(256) * 0.02),
    }
    myelin = _MockMyelin(M)
    result = check_degenerate_M(myelin)

    assert result is not None, "guard must flag a unimodal collapse"
    assert "unimodal" in result, (
        f"Expected violation message to identify the unimodal failure mode; "
        f"got: {result}"
    )


def test_degenerate_M_passes_on_healthy_distribution():
    """The guard must NOT raise on a healthy bimodal distribution with a
    substantial fraction of mass at the lock."""
    from synthetic_myelination.m_coupled_regularizer import check_degenerate_M

    myelin = _MockMyelin(_build_healthy_M())
    result = check_degenerate_M(myelin)

    assert result is None, (
        f"guard must accept a healthy bimodal distribution (mean(M) ≈ 0.5); "
        f"got violation: {result}"
    )


def test_d5_replays_v11_observed_numbers():
    """Documentary test: the v1.1 observed M statistics (mean=0.004,
    std=0.063, BC≈1.0) match the closed-form prediction under the
    PEFT-init bug hypothesis.  This pins the diagnostic story so a future
    regression that recreates the same shape is recognizable."""
    from synthetic_myelination.m_coupled_regularizer import (
        bimodality_coefficient,
    )

    # Llama-3.2-3B-shaped budget: 28 layers, r=16, target_modules covering
    # both attention and MLP.  Aggregate per-layer:
    #   lora_A: 7 modules × r=16 = 112 elements
    #   lora_B: 3072 + 1024 + 1024 + 3072 + 8192 + 8192 + 3072 = 27,648
    # Across 28 layers:
    #   lora_A total: 3,136  (~0.40% of budget)
    #   lora_B total: 774,144
    n_lora_A = 28 * 112
    n_lora_B = 28 * 27648
    n_total = n_lora_A + n_lora_B
    lora_A_frac = n_lora_A / n_total
    assert lora_A_frac == pytest.approx(0.00403, abs=0.0001)

    # If all lora_A elements lock at 1.0 and all lora_B stay at 0:
    #   mean(M) = lora_A_frac = 0.00403
    all_vals = [1.0] * n_lora_A + [0.0] * n_lora_B
    mean_m = sum(all_vals) / n_total
    assert mean_m == pytest.approx(0.00403, abs=0.0001), (
        f"v1.1 expected mean(M) ≈ 0.00403, predicted {mean_m:.5f}"
    )
    # Observed in lab report: mean_M = 0.00400 — match to 3 sig figs.

    # std under Bernoulli(0.00403): √(p(1-p)) ≈ 0.0633
    # Observed in lab report: std_M = 0.0632
    import statistics as _st
    std_m = _st.pstdev(all_vals)
    assert std_m == pytest.approx(0.063, abs=0.001), (
        f"v1.1 expected std(M) ≈ 0.063 (Bernoulli), predicted {std_m:.4f}"
    )

    # BC on this distribution should be very high (sharp bimodal).
    # Use a small subsample for speed since BC is invariant to repetition.
    bc = bimodality_coefficient(all_vals[:10000])
    assert bc > 0.7, (
        f"v1.1 expected BC > 0.7 (the distribution IS bimodal, just "
        f"degenerately so); got BC={bc:.4f}"
    )
