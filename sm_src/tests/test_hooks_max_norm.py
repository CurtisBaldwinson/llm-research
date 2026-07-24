"""Tests for the per-tensor max-norm scheme in ``ActivityCapture``.

These pin the 2026-05-18 v7.1 fix that replaced the per-tensor L2-norm
divide with a per-tensor max divide in both per-neuron hook paths:

- ``_register_activation_l2_hooks``  (line ~163 in hooks.py)
- ``_register_grad_l2_per_neuron_hooks``  (line ~237 in hooks.py)

The fix's load-bearing property: the post-normalization peak per-neuron
entry is 1.0 regardless of the tensor's width H.  Under the old
L2-norm scheme the peak entry scaled as max/(c*sqrt(H)) for roughly-
uniform per-row magnitudes ``c``, dragging signals on wide tensors
below the MyelinState update rule's fixed point and producing the
degenerate near-zero M distribution observed in Improvement 07's
failing 2026-05-17 GPU smoke (and retrospectively in Improvement 05's
v1.0 headline run, where m_mean_after_a was ~0.004 with BC~1.0 — same
pathology).

Run with:

    pytest sm_src/tests/test_hooks_max_norm.py -v
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic_myelination.hooks import ActivityCapture  # noqa: E402


# ---------------------------------------------------------------------------
# grad_l2_per_neuron mode
# ---------------------------------------------------------------------------


def _make_linear(out_features: int, in_features: int = 8) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=False)
    layer.weight.requires_grad_(True)
    return layer


def _trigger_grad_hook(layer: nn.Linear, grad: torch.Tensor) -> None:
    """Invoke the param backward hook directly.

    ``ActivityCapture._register_grad_l2_per_neuron_hooks`` attaches a
    closure to ``param.register_hook``.  Torch fires that closure when
    the autograd engine produces a gradient.  Here we set ``param.grad``
    manually and re-fire the closure via a tiny no-op backward chain so
    the production code path is exercised end-to-end (not just the
    normalization math, but the param-name lookup, the signal-dict write,
    etc.).
    """
    # Use an autograd-traced forward so .backward() invokes the hook.
    x = torch.eye(layer.in_features, dtype=layer.weight.dtype)
    target = grad @ x  # shape [out_features, in_features]
    out = layer(x)
    # ``out`` is layer(x); we want grad(out)/grad(weight) = grad.
    # Use a custom autograd seed so the per-weight gradient equals
    # ``grad`` exactly.
    out.backward(gradient=target.t())


def test_grad_per_neuron_max_norm_peak_is_one_regardless_of_H():
    """For each H in {16, 256, 3072, 8192}: a roughly-uniform gradient
    produces a per-neuron signal whose peak entry is exactly 1.0 after
    the per-tensor normalization, and whose other entries are bounded
    by 1.0 (max-norm produces values in [0,1]).
    """
    for H in (16, 256, 3072, 8192):
        torch.manual_seed(H)
        layer = _make_linear(H, in_features=8)
        capture = ActivityCapture(
            layer,
            signal_type="grad_l2_per_neuron",
            normalization="max",
            ema_window=1,
        )
        capture.register()

        # Roughly-uniform per-row magnitudes with a small spread so the
        # max isn't degenerate.  Row i has base magnitude 1 + (i%4)*0.1.
        base = 1.0 + (torch.arange(H, dtype=torch.float32) % 4) * 0.1
        synthetic_grad = base.unsqueeze(1) * torch.ones(H, 8)
        layer.weight.grad = None
        _trigger_grad_hook(layer, synthetic_grad)

        signals = capture.get_signals()
        capture.remove()

        # The hook stored a [H]-shape signal under the param name.
        assert "weight" in signals, f"H={H}: signal not captured"
        s = signals["weight"]
        assert s.shape == (H,), f"H={H}: signal shape {s.shape}"

        # Max-norm property: peak entry is exactly 1.0, all entries <= 1.0.
        assert s.max().item() == pytest.approx(1.0, abs=1e-6), (
            f"H={H}: peak entry should be 1.0, got {s.max().item():.6f}"
        )
        assert s.min().item() >= 0.0, f"H={H}: signal went negative"
        assert s.max().item() <= 1.0 + 1e-6, f"H={H}: signal exceeded 1.0"


def test_grad_per_neuron_signal_does_not_collapse_to_one_over_sqrt_H():
    """Regression guard: the old L2-norm scheme had peak signal scaling
    as ~max/(c*sqrt(H)), i.e. peak signal would be ~0.018 for H=3072 with
    uniform rows of magnitude 1.  The max-norm scheme must keep the peak
    signal at 1.0.  This test catches a regression to L2-norm behavior.
    """
    H = 3072
    layer = _make_linear(H, in_features=8)
    capture = ActivityCapture(
        layer,
        signal_type="grad_l2_per_neuron",
        normalization="max",
        ema_window=1,
    )
    capture.register()
    # Perfectly uniform per-row magnitudes (max == mean), so L2-norm and
    # max-norm give very different results: max-norm -> 1.0 everywhere,
    # L2-norm -> 1/sqrt(H) everywhere.
    synthetic_grad = torch.ones(H, 8)
    layer.weight.grad = None
    _trigger_grad_hook(layer, synthetic_grad)
    s = capture.get_signals()["weight"]
    capture.remove()
    # Max-norm: every entry should be ~1.0 (uniform rows -> uniform normalized).
    # L2-norm would give ~1/sqrt(3072) = 0.018 — would catastrophically fail.
    assert s.mean().item() > 0.5, (
        f"signal mean {s.mean().item():.4f} suggests L2-norm regression "
        f"(expected ~1.0 under max-norm; ~0.018 under buggy L2-norm)"
    )
    assert s.mean().item() == pytest.approx(1.0, abs=1e-5)


def test_grad_per_neuron_handles_zero_gradient_without_nan():
    """If a tensor has all-zero gradient (e.g. lora_A under PEFT's
    lora_B=0 init), the per-tensor max is 0 and the +1e-8 epsilon guards
    the divide.  The signal should be all zeros, no NaN, no Inf.
    """
    H = 64
    layer = _make_linear(H, in_features=8)
    capture = ActivityCapture(
        layer,
        signal_type="grad_l2_per_neuron",
        normalization="max",
        ema_window=1,
    )
    capture.register()
    _trigger_grad_hook(layer, torch.zeros(H, 8))
    s = capture.get_signals()["weight"]
    capture.remove()
    assert torch.isfinite(s).all(), "signal contains NaN or Inf"
    assert (s == 0).all(), f"all-zero gradient should produce zero signal, got max {s.max().item()}"


# ---------------------------------------------------------------------------
# activation_l2 mode (same fix; verify symmetry)
# ---------------------------------------------------------------------------


def test_activation_l2_max_norm_peak_is_one():
    """Forward-pass hook path: per-neuron post-ReLU activation magnitudes
    should also normalize via max so peak signal is 1.0 regardless of H.
    """
    H = 2048
    layer = nn.Linear(8, H, bias=False)
    layer.weight.requires_grad_(True)
    capture = ActivityCapture(
        layer,
        signal_type="activation_l2",
        normalization="max",
        ema_window=1,
    )
    capture.register()
    # Forward with input of magnitude 1 across all in-features; activations
    # roughly track weight rows.  We seed the weight so per-row magnitudes
    # have a usable spread.
    torch.manual_seed(0)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(H, 8).abs())  # positive so ReLU is a no-op
    x = torch.ones(1, 8)
    _ = layer(x)
    s = capture.get_signals()["weight"]
    capture.remove()
    assert s.shape == (H,)
    assert s.max().item() == pytest.approx(1.0, abs=1e-6)
    assert s.min().item() >= 0.0


# ---------------------------------------------------------------------------
# MyelinState fixed-point smoke test (closed-form on the patch)
# ---------------------------------------------------------------------------


def test_max_norm_signal_clears_myelinstate_fixed_point():
    """The Phase A failure was driven by the L2-norm post-normalization
    signal sitting BELOW the MyelinState update fixed point
    `A* = beta/(alpha+beta) = 0.0566` (for default alpha=0.05, beta=0.003).

    Under max-norm, the peak signal is 1.0 — well above A*.  This is
    the empirical property that lets M actually grow during Phase A.
    Pin it explicitly so a future "improvement" to the normalization
    can't silently regress below A*.
    """
    alpha, beta = 0.05, 0.003
    A_star = beta / (alpha + beta)
    H = 4096
    layer = _make_linear(H, in_features=8)
    capture = ActivityCapture(
        layer,
        signal_type="grad_l2_per_neuron",
        normalization="max",
        ema_window=1,
    )
    capture.register()
    _trigger_grad_hook(layer, torch.ones(H, 8))
    s = capture.get_signals()["weight"]
    capture.remove()
    # Under max-norm with uniform gradients, every entry is 1.0.
    # That's well above A*.
    assert s.max().item() > A_star * 10, (
        f"peak signal {s.max().item():.4f} should be well above A*={A_star:.4f}"
    )
    assert s.mean().item() > A_star, (
        f"mean signal {s.mean().item():.4f} should clear A*={A_star:.4f} "
        f"so M can grow during the update rule"
    )
