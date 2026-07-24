"""Activity-capture forward/backward hooks.

Collects per-parameter activity signals that drive MyelinState updates.
Four signal modes are supported:

- ``grad_norm``:           L2 norm of each parameter's gradient after backward
                           (scalar per parameter).
- ``activation_l2``:       per-neuron activation magnitude (post-ReLU) per
                           Linear/Embedding module (Tensor[H] per parameter).
- ``participation``:       fraction of non-near-zero entries in the gradient
                           (scalar per parameter).
- ``grad_l2_per_neuron``:  per-neuron gradient L2 norm captured via parameter
                           backward hooks (Tensor[H] per parameter).
                           Added for Improvement 07 (gradient-aligned M); see
                           benchmarks/07_gradient_aligned_m/SPEC_v1.0.md.

v3 adds softmax normalization + EMA smoothing options to ``get_signals``.
v5 adds per-neuron (Tensor[H]) signals for activation_l2 mode.
v7 adds per-neuron (Tensor[H]) signals for the gradient-derived signal
mode ``grad_l2_per_neuron``.  The new mode is the gradient analogue of
activation_l2: same per-neuron Tensor[H] shape, same EMA smoothing path
in ``get_signals``, same broadcasting behavior downstream — the only
difference is the signal SOURCE (gradient L2 norm per row rather than
activation L2 norm per output neuron).

v7.1 (2026-05-18) replaces the per-tensor L2-norm normalization in both
per-neuron hook paths with per-tensor MAX-norm.  The previous L2-norm
divide systematically pushed per-neuron signals on wide tensors below
the MyelinState update fixed point A* = β/(α+β) = 0.0566 for any tensor
wider than ~310 elements (because the L2 norm of a length-H vector with
roughly-uniform entries scales as √H, so the post-normalization signal
scales as 1/√H).  This produced a degenerate near-zero M distribution
on Llama-3.2-3B's LoRA-B adapters (H ∈ {1024, 3072, 8192}) and caused
Improvement 07's Phase A to fail invariant D1 (`std(M) > 0.05`).  The
max-norm scheme is H-invariant: the peak per-neuron entry lands at 1.0
regardless of H, and the rest are proportional fractions of that.  See
``benchmarks/07_gradient_aligned_m/_d1_investigation.py`` and
``_d1_realgrad_investigation.py`` for the analysis and verification.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor


_TRACKED_TYPES = (nn.Linear, nn.Embedding)


# Signal modes that produce Tensor[H] per-neuron signals (vs scalar floats).
# Centralized so ``get_signals`` can route all such modes through the same
# per-neuron EMA path without scattered string comparisons.
_PER_NEURON_SIGNAL_TYPES = frozenset({
    "activation_l2",
    "activation_l2_wrapper",  # Option A' fix; see _register_activation_l2_wrapper_hooks
    "grad_l2_per_neuron",
})


def _rank_normalize(signals: dict[str, float]) -> dict[str, float]:
    """Within each layer group (module prefix), replace values by fractional rank.

    Group key = everything before the last '.' in the parameter name. Within each
    group, the smallest value gets rank 1/N and the largest gets N/N=1.0. Ties
    are broken by sorted-stable ordering. Result is in (0, 1].
    """
    from collections import defaultdict

    groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for name, val in signals.items():
        prefix = name.rsplit(".", 1)[0] if "." in name else name
        groups[prefix].append((name, val))

    out: dict[str, float] = {}
    for _prefix, items in groups.items():
        sorted_items = sorted(items, key=lambda x: x[1])
        n = len(sorted_items)
        for rank_idx, (name, _) in enumerate(sorted_items):
            out[name] = (rank_idx + 1) / n
    return out


class ActivityCapture:
    """Register forward/backward hooks to capture per-parameter activity."""

    def __init__(
        self,
        model: nn.Module,
        signal_type: str = "grad_norm",
        normalization: str = "max",
        ema_window: int = 1,
    ) -> None:
        allowed_signal_types = {
            "grad_norm",
            "activation_l2",
            "activation_l2_wrapper",  # Option A' fix; see _register_activation_l2_wrapper_hooks
                    "participation",
            "grad_l2_per_neuron",
        }
        if signal_type not in allowed_signal_types:
            raise ValueError(
                f"Unknown signal_type: {signal_type!r}. "
                f"Must be one of: {sorted(allowed_signal_types)}"
            )
        if normalization not in {"max", "softmax", "rank"}:
            raise ValueError(f"Unknown normalization: {normalization!r}")
        self.model = model
        self.signal_type = signal_type
        self.normalization = normalization
        self.ema_window = max(1, int(ema_window))
        self._handles: list = []
        # For per-neuron modes (activation_l2, grad_l2_per_neuron) values are
        # Tensor[H]; for scalar modes (grad_norm, participation) values are floats.
        self._signals: dict[str, Union[float, Tensor]] = {}
        self._module_to_param_names: dict[nn.Module, list[str]] = {}
        # EMA state mirrors the signal dtype (Tensor[H] or float).
        self._ema: dict[str, Union[float, Tensor]] = {}

    def _collect_param_names(self) -> None:
        self._module_to_param_names.clear()
        for mod_name, module in self.model.named_modules():
            if isinstance(module, _TRACKED_TYPES):
                names: list[str] = []
                for p_name, p in module.named_parameters(recurse=False):
                    if p.requires_grad:
                        full_name = f"{mod_name}.{p_name}" if mod_name else p_name
                        names.append(full_name)
                if names:
                    self._module_to_param_names[module] = names

    def register(self) -> None:
        """Attach hooks to all Linear/Embedding layers.

        Three registration paths:
          - ``activation_l2`` — forward hook on the module; per-neuron Tensor[H]
            signal computed from output activations (post-ReLU clamp).  Same
            signal stored for every parameter of the module (weight + bias
            share the neuron-level activity).
          - ``grad_l2_per_neuron`` — backward hook on each PARAMETER; per-neuron
            Tensor[H] signal computed from the L2 norm of each output neuron's
            gradient row.  Signal is parameter-specific (weight and bias of the
            same module get different signal vectors, both shape [H]).
          - ``grad_norm`` / ``participation`` — backward hook on each parameter
            with a scalar float signal.  Legacy paths preserved unchanged.
        """
        self.remove()
        self._collect_param_names()

        if self.signal_type == "activation_l2":
            self._register_activation_l2_hooks()
        elif self.signal_type == "activation_l2_wrapper":
            self._register_activation_l2_wrapper_hooks()
        elif self.signal_type == "grad_l2_per_neuron":
            self._register_grad_l2_per_neuron_hooks()
        else:
            self._register_scalar_grad_hooks()

    # ------------------------------------------------------------------
    # Per-mode hook registration helpers
    # ------------------------------------------------------------------

    def _register_activation_l2_hooks(self) -> None:
        """v5: per-neuron forward hook — stores Tensor[H] per param."""
        for module, param_names in self._module_to_param_names.items():
            names = list(param_names)

            def make_hook(pnames: list[str]):
                def hook(_m, _inp, out):
                    if isinstance(out, tuple):
                        out = out[0]
                    if not torch.is_tensor(out):
                        return
                    # Simulate post-ReLU by clamping to >= 0
                    act = out.detach().clamp(min=0.0)
                    # Per-neuron mean over batch (and time if 3D)
                    if act.dim() == 3:
                        neuron_act = act.mean(dim=(0, 1))  # [H]
                    else:
                        neuron_act = act.mean(dim=0)  # [H]
                    # v7.1: per-tensor MAX-norm.  Was per-tensor L2-norm,
                    # which collapsed signals on wide tensors below the
                    # MyelinState fixed point (A* = β/(α+β) = 0.0566 for the
                    # default α=0.05, β=0.003); max-norm is H-invariant.
                    mx = neuron_act.max()
                    neuron_act = (neuron_act / (mx + 1e-8)).clamp(0.0, 1.0)
                    # Store per-param signal — weight and bias share same neuron activity
                    for n in pnames:
                        self._signals[n] = neuron_act  # Tensor [H]

                return hook

            self._handles.append(module.register_forward_hook(make_hook(names)))

    @staticmethod
    def _is_peft_wrapper(module: nn.Module) -> bool:
        """Heuristic: a module is a PEFT LoRA wrapper if its named_children
        include both ``lora_A`` and ``lora_B``.  Matches both real PEFT
        wrappers (where ``lora_A`` / ``lora_B`` are ``ModuleDict``) and the
        synthetic ``FakeLoraLayer`` used in the regression tests.  Does NOT
        match other adapter families (DoRA, LoHa, etc.) which use different
        attribute names; those would fall through to the per-module fallback
        in ``_register_activation_l2_wrapper_hooks``.
        """
        children = dict(module.named_children())
        return "lora_A" in children and "lora_B" in children

    def _register_activation_l2_wrapper_hooks(self) -> None:
        """v8 (Option A', 2026-05-21): per-wrapper forward hook for PEFT LoRA.

        Background: under ``signal_type="activation_l2"`` the hook fires on
        each ``nn.Linear`` submodule with trainable params.  For a PEFT-
        wrapped Linear, that includes ``lora_A.default`` AND ``lora_B.default``
        as separate hook sites.  PEFT initializes ``lora_B.weight = 0``, so
        the forward output of ``lora_B.default`` is identically zero for the
        entire Phase A (no optimizer steps), which makes M(lora_B) stay at 0
        for ~99.6% of the M element budget at Llama-3.2-3B r=16.  This is
        the v1.1 "degenerate-bimodal" failure mode
        (see lab_reports/2026-05-21_0942_imp05-v11-headline-refuted/).

        Fix: hook on the PEFT wrapper module (the parent of ``lora_A`` /
        ``lora_B`` ModuleDicts) instead of on the inner Linears.  The
        wrapper's forward output is ``base_layer(x) + scaling *
        lora_B(lora_A(x))``.  At PEFT init the second term is zero, but the
        first term (``base_layer(x)``) is the unmodified base-model
        activation — **non-zero from step 1**, regardless of LoRA init.
        The same per-output-neuron signal is stored under every trainable
        parameter under the wrapper (both lora_A and lora_B), which gives
        balanced per-parameter M coverage by construction.

        Non-PEFT modules with trainable params (e.g. a fully fine-tuned
        layer with no LoRA wrapping) fall back to the standard
        ``activation_l2`` per-module path so this mode is a strict
        superset, not a replacement.

        Key properties vs Option A (warmup-step):
          * Preserves SPEC's "no optimizer step in Phase A" contract.
          * θ_ref captured after Phase A is still the LoRA-init state
            (lora_B == 0), not a post-warmup-SFT state.
          * Hook layer changes only; no run.py changes needed beyond
            switching the configured ``signal_type``.
          * lora_A and lora_B parameters share the SAME wrapper-level
            signal.  This is strictly between per-parameter M (v1.0/v1.1)
            and per-pathway M (Option B) on the granularity axis.

        EMPIRICAL RESULT (2026-05-21 pre-flight on Llama-3.2-1B, CPU):
        Option A' closes the structural bug — ``lora_A`` and ``lora_B``
        do receive equal, non-zero signals — but the wrapper's per-output-
        neuron signal averages across different active neurons across
        different inputs, so the per-element mean signal stays near the
        MyelinState fixed point A* = β/(α+β) = 0.0566 and M does NOT
        develop the bimodal lock distribution H3 requires.  Concretely:
        mean(M[lora_A]) = mean(M[lora_B]) = 0.062 (just above A*), BC =
        0.67 (below the 0.70 bimodality threshold).  D5 catches this as
        "unimodal" rather than "degenerate-bimodal".  Production v1.2
        therefore uses Option A (warmup) instead.  See
        ``benchmarks/05_m_coupled_kl_rlhf/preflight_report.txt`` for the
        comparative numbers and ``preflight.py`` to reproduce.

        Option A' is kept available as a signal_type for future work on
        a per-rank wrapper variant (capture lora_A's intermediate output
        of shape ``[batch, r]`` as the lora_A signal and the full wrapper
        output as the lora_B signal); that two-hook variant would have
        the correct per-factor signal shapes but is not yet implemented.
        """
        # Stage 1: identify PEFT wrapper modules and the trainable params
        # under each.  Track which param names get captured here so we
        # don't double-hook them in the fallback stage.
        wrappers: list[tuple[str, nn.Module, list[str]]] = []
        captured_param_names: set[str] = set()
        for mod_name, module in self.model.named_modules():
            if not self._is_peft_wrapper(module):
                continue
            wrapper_param_names: list[str] = []
            for p_name, p in module.named_parameters(recurse=True):
                if p.requires_grad:
                    full_name = f"{mod_name}.{p_name}" if mod_name else p_name
                    wrapper_param_names.append(full_name)
            if wrapper_param_names:
                wrappers.append((mod_name, module, wrapper_param_names))
                captured_param_names.update(wrapper_param_names)

        # Stage 2: register the wrapper-level hook.  Same signal-processing
        # logic as _register_activation_l2_hooks (post-ReLU clamp, per-neuron
        # mean, per-tensor max-norm); only the registration site differs.
        def _make_hook(captured_names: list[str]):
            def hook(_m, _inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                if not torch.is_tensor(out):
                    return
                act = out.detach().clamp(min=0.0)
                if act.dim() == 3:
                    neuron_act = act.mean(dim=(0, 1))  # [H]
                else:
                    neuron_act = act.mean(dim=0)  # [H]
                mx = neuron_act.max()
                neuron_act = (neuron_act / (mx + 1e-8)).clamp(0.0, 1.0)
                for n in captured_names:
                    self._signals[n] = neuron_act
            return hook

        for _, wrap_module, names in wrappers:
            self._handles.append(
                wrap_module.register_forward_hook(_make_hook(list(names)))
            )

        # Stage 3: per-module fallback for any trainable params NOT under
        # a PEFT wrapper.  Same hook as _register_activation_l2_hooks for
        # this subset.
        for module, param_names in self._module_to_param_names.items():
            remaining = [n for n in param_names if n not in captured_param_names]
            if not remaining:
                continue
            self._handles.append(
                module.register_forward_hook(_make_hook(list(remaining)))
            )

    def _register_grad_l2_per_neuron_hooks(self) -> None:
        """v7 (Improvement 07): per-neuron gradient L2 norm via parameter backward hooks.

        For each tracked parameter, register a ``param.register_hook`` callback
        that runs on backward and produces a Tensor[H] signal where H is the
        parameter's first dimension (the "output neuron" axis).

        Per-axis aggregation:
          - 1D (bias): each entry IS the per-neuron signal — take |grad|.
          - 2D (Linear weight [H_out, H_in]): per-row L2 norm — ``grad.norm(dim=1)``
            gives shape [H_out].  This is the LoRA case as well (lora_A weight
            shape [r, in_features] yields signal shape [r]; lora_B weight shape
            [out_features, r] yields signal shape [out_features]).
          - 4D (Conv2d weight [C_out, C_in, kH, kW]): per-output-channel L2 norm —
            ``grad.norm(dim=(1, 2, 3))`` gives shape [C_out].
          - Other shapes: fall back to flattened absolute values (rare; defensive).

        Per-tensor normalization (v7.1, 2026-05-18): divide by the per-neuron
        vector's max element and clamp to [0, 1].  Matches activation_l2's
        normalization for downstream consistency — both modes produce signals
        whose magnitudes are in [0, 1] regardless of the model's absolute
        gradient scale, AND are invariant to tensor width H (an earlier
        L2-norm-based scheme collapsed wide-tensor signals to ~1/√H, which
        sat below the MyelinState fixed point and produced degenerate M
        distributions; see module docstring).

        Accumulation: multiple backward passes between ``reset()`` calls are
        accumulated via elementwise maximum.  This matches the scalar
        ``grad_norm`` path's behavior and captures peak per-neuron gradient
        across micro-batches (relevant when callers use gradient accumulation
        without resetting between micro-batches).

        Backward-hook semantics: ``param.register_hook(fn)`` fires once per
        backward pass on the param's gradient.  The hook is registered against
        a frozen ``param`` reference captured in the closure — replacing
        ``param.data`` later (e.g. by ``model.eval()`` or device migration)
        does not detach the hook.  Hooks ARE detached when ``remove()`` is
        called or when the parameter object itself is garbage-collected.
        """
        for module, param_names in self._module_to_param_names.items():
            param_lookup = dict(module.named_parameters(recurse=False))
            for full_name in param_names:
                p_local = full_name.split(".")[-1]
                param = param_lookup[p_local]

                def make_hook(name: str):
                    def hook(grad: Tensor):
                        if grad is None:
                            return
                        g = grad.detach().float()
                        # Per-neuron L2 norm reduction depending on tensor rank
                        if g.dim() == 1:
                            # Bias case: signal is just |grad|
                            per_neuron = g.abs()
                        elif g.dim() == 2:
                            # Linear weight [H_out, H_in] (or LoRA A [r, in_features]
                            # or LoRA B [out_features, r]): per-row L2 norm.
                            per_neuron = g.norm(dim=1)
                        elif g.dim() == 4:
                            # Conv2d weight [C_out, C_in, kH, kW]: per-channel L2 norm.
                            per_neuron = g.norm(dim=(1, 2, 3))
                        else:
                            # Defensive fallback for other parameter shapes
                            # (e.g. nn.Parameter with arbitrary shape).  Use
                            # absolute values of the flattened tensor.
                            per_neuron = g.abs().flatten()
                        # Per-tensor MAX-norm to [0, 1].  +1e-8 guards
                        # against zero-grad tensors (signal stays at 0).
                        # See module docstring for why this is max-norm not
                        # L2-norm (the 2026-05-18 v7.1 fix).
                        mx = per_neuron.max()
                        per_neuron = (per_neuron / (mx + 1e-8)).clamp(0.0, 1.0)

                        # Accumulate via elementwise max with prior signal.
                        # This matches grad_norm's max-accumulation behavior
                        # and captures peak per-neuron gradient when multiple
                        # backwards happen before reset().
                        prev = self._signals.get(name)
                        if prev is None or not torch.is_tensor(prev):
                            self._signals[name] = per_neuron
                        elif prev.shape != per_neuron.shape:
                            # Shape mismatch (e.g. param was replaced) — overwrite.
                            self._signals[name] = per_neuron
                        else:
                            self._signals[name] = torch.maximum(prev, per_neuron)

                    return hook

                self._handles.append(param.register_hook(make_hook(full_name)))

    def _register_scalar_grad_hooks(self) -> None:
        """Legacy path: scalar-per-parameter backward hooks for grad_norm and participation."""
        for module, param_names in self._module_to_param_names.items():
            param_lookup = dict(module.named_parameters(recurse=False))
            for full_name in param_names:
                p_local = full_name.split(".")[-1]
                param = param_lookup[p_local]

                def make_hook(name: str):
                    signal_type = self.signal_type

                    def hook(grad: Tensor):
                        if signal_type == "grad_norm":
                            val = float(grad.detach().float().norm().item())
                        else:
                            nonzero = (grad.detach().abs() > 1e-8).float().mean()
                            val = float(nonzero.item())
                        prev = self._signals.get(name, 0.0)
                        # Ensure prev is a float (defensive against accidental
                        # mode switches that left Tensor[H] in _signals).
                        if not isinstance(prev, (int, float)):
                            prev = 0.0
                        self._signals[name] = max(prev, val)

                    return hook

                self._handles.append(param.register_hook(make_hook(full_name)))

    # ------------------------------------------------------------------
    # Hook lifecycle and signal retrieval
    # ------------------------------------------------------------------

    def remove(self) -> None:
        """Detach all registered hooks."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def get_signals(self, normalize: bool = True) -> dict[str, Union[float, Tensor]]:
        """Return the accumulated signals since the last :meth:`reset`.

        For per-neuron modes (activation_l2, grad_l2_per_neuron): returns
        ``{param_name: Tensor[H]}`` with EMA applied.
        For scalar modes (grad_norm, participation): returns
        ``{param_name: float}`` (normalized scalars).

        Behavior:
          - normalization="max", ema_window=1: divide by running max → [0,1]
            (v1/v2 backward-compatible).
          - normalization="softmax": softmax with temperature=10 over all
            tracked parameters; sums to 1.0.
          - ema_window>1: Return an EMA-smoothed version of the (normalized)
            signal with alpha = 2/(ema_window+1).

        If ``normalize`` is False, returns raw (un-normalized) signals without
        EMA. This is the backward-compatible escape hatch.
        """
        if not self._signals:
            return {}
        if not normalize:
            return dict(self._signals)

        # Per-neuron signal modes route through the Tensor[H] EMA path.  Both
        # activation_l2 and grad_l2_per_neuron produce Tensor[H] signals that
        # are already per-tensor-normalized to [0, 1] in the hook, so the EMA
        # is the only smoothing applied here.
        if self.signal_type in _PER_NEURON_SIGNAL_TYPES:
            return self._get_signals_per_neuron()

        # --- Scalar signal path (grad_norm, participation) ---
        if self.normalization == "softmax":
            names = sorted(self._signals.keys())
            raw = torch.tensor([self._signals[n] for n in names], dtype=torch.float32)
            weights = torch.softmax(raw * 10.0, dim=0)
            normed = {names[i]: float(weights[i].item()) for i in range(len(names))}
        elif self.normalization == "rank":
            normed = _rank_normalize(self._signals)
        else:
            max_val = max(self._signals.values())
            if max_val <= 0.0:
                normed = {k: 0.0 for k in self._signals}
            else:
                normed = {k: float(v / max_val) for k, v in self._signals.items()}

        if self.ema_window <= 1:
            return normed

        ema_alpha = 2.0 / (self.ema_window + 1)
        out: dict[str, float] = {}
        for name, val in normed.items():
            prev = self._ema.get(name, val)
            smoothed = ema_alpha * val + (1.0 - ema_alpha) * prev
            self._ema[name] = smoothed
            out[name] = smoothed
        return out

    def _get_signals_per_neuron(self) -> dict[str, Tensor]:
        """Per-neuron EMA for Tensor[H] signal modes.

        Used by both ``activation_l2`` and ``grad_l2_per_neuron``.  The hook
        already applied per-tensor normalization to [0, 1]; this method only
        applies the EMA smoothing controlled by ``ema_window``.
        """
        ema_alpha = 2.0 / (self.ema_window + 1) if self.ema_window > 1 else 1.0

        out: dict[str, Tensor] = {}
        for name, val in self._signals.items():
            if not torch.is_tensor(val):
                # Shouldn't happen in per-neuron modes, but be safe — convert
                # any scalar leakage from a previous mode into a 0-D tensor.
                val = torch.tensor(float(val))
            A = val.float()
            if name not in self._ema:
                # First time: initialize EMA to current signal
                self._ema[name] = A.clone()
            else:
                prev = self._ema[name]
                if not torch.is_tensor(prev) or prev.shape != A.shape:
                    # Tensor-shape change (or scalar leakage) — reset EMA to
                    # current value rather than corrupting downstream math.
                    self._ema[name] = A.clone()
                else:
                    # EMA update: new = alpha * current + (1 - alpha) * prev
                    smoothed = ema_alpha * A + (1.0 - ema_alpha) * prev
                    self._ema[name] = smoothed
                    A = smoothed
            out[name] = A
        return out

    # Backward-compatible alias for callers that referenced the activation_l2-
    # specific name directly.  Delegates to the unified per-neuron path.
    def _get_signals_activation_l2(self) -> dict[str, Tensor]:
        return self._get_signals_per_neuron()

    def reset(self) -> None:
        """Clear accumulated signals (keeps EMA state)."""
        self._signals.clear()

    def reset_ema(self) -> None:
        """Clear EMA state (e.g. between tasks)."""
        self._ema.clear()

    def __enter__(self) -> "ActivityCapture":
        self.register()
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:  # noqa: D401
        self.remove()
        return None
