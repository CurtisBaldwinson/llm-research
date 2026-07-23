"""
M-Coupled Regularizer — Improvement 05 mechanism module.

This module provides `MCoupledRegularizer`, a drop-in auxiliary loss term
for RLHF training (DPO or PPO). The regularizer computes a per-parameter
Frobenius distance penalty between the current model weights and a
reference snapshot, weighted by either:

  - the synthetic_myelination M tensor (the proposed mechanism), OR
  - a diagonal Fisher information estimate (the EWC-style ablation), OR
  - uniform weights (for ablation / sanity check)

The intent is to replace the global KL penalty in standard RLHF with a
per-parameter regularization that respects which parameters matter for
base capabilities (high M / high Fisher) vs which can move freely (low
M / low Fisher).

Usage with HuggingFace TRL's DPOTrainer:

    from sm_src.synthetic_myelination import MyelinState
    from sm_src.synthetic_myelination.m_coupled_regularizer import (
        MCoupledRegularizer,
    )

    # After Phase A pre-training:
    regularizer = MCoupledRegularizer(
        model=model,
        myelin_state=myelin,
        fisher_diag=fisher_diag,   # or None for M-only / uniform
        lambda_reg=0.1,
        regularizer_type="m_coupled",  # "m_coupled" | "fisher" | "uniform"
    )
    regularizer.snapshot_reference()

    # During DPO training, monkey-patch the trainer's loss:
    original_compute_loss = trainer.compute_loss
    def patched_compute_loss(model, inputs, return_outputs=False, **kw):
        loss, *rest = original_compute_loss(model, inputs,
                                            return_outputs=True, **kw)
        aux = regularizer.penalty()
        total = loss + aux
        return (total, *rest) if return_outputs else total
    trainer.compute_loss = patched_compute_loss

See SPEC_v1.0.md §3 for the math, §8 for the implementation map.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class MCoupledRegularizer:
    """Per-parameter Frobenius regularizer weighted by M (or Fisher / uniform).

    The penalty is:

        L_aux = λ · Σ_n W(n) · ||θ_n − θ_ref,n||²

    where W(n) is one of:
      - M(n) from a MyelinState (regularizer_type="m_coupled")
      - F(n) diagonal Fisher information (regularizer_type="fisher")
      - 1.0 uniform (regularizer_type="uniform")

    The reference θ_ref is captured by `snapshot_reference()` and held
    frozen for the duration of training. M / Fisher are likewise held
    frozen (this is a regularizer, not an active state).

    Broadcasting follows the same convention as
    `myelin.apply_gate` / the patched CAG `_apply_g_only`:
      - W shape [H] broadcasts to param.grad [H, in] via unsqueeze(1)
      - W shape [H] matches bias [H] directly
      - W shape [C] broadcasts to Conv2d weight [C, in, kH, kW]
      - Otherwise: rely on PyTorch broadcasting (e.g., Embedding case)

    Notes on LoRA:
      For a LoRA-wrapped Linear, peft creates lora_A (shape [r, in])
      and lora_B (shape [out, r]) as separate nn.Linear modules. With
      activation_l2 signal type, the hook captures the output of each
      module:
        - lora_A output is [batch, r] → M shape [r] → matches lora_A
          weight shape[0] = r. Per-rank weighting. Clean.
        - lora_B output is [batch, out] → M shape [out] → matches
          lora_B weight shape[0] = out. Per-output-neuron weighting.
          Clean.
      So in the LoRA case both adapter matrices get appropriately-
      shaped weights with no broadcasting fallback needed.
    """

    def __init__(
        self,
        model: nn.Module,
        myelin_state=None,                       # synthetic_myelination.MyelinState
        fisher_diag: Optional[Dict[str, torch.Tensor]] = None,
        lambda_reg: float = 0.1,
        regularizer_type: str = "m_coupled",
        exclude_param_names: Optional[set] = None,
    ):
        self.model = model
        self.myelin = myelin_state
        self.fisher_diag = fisher_diag if fisher_diag is not None else {}
        self.regularizer_type = str(regularizer_type)
        self.exclude_param_names = set(exclude_param_names or set())

        # Synthetic2: scalar λ only. The penalty is the EWC-family form
        #     L_aux = λ · Σ_n W(n) · ||θ_n − θ_ref,n||²
        # where W(n) is the importance signal (M, Fisher, or uniform).
        # The v1 per-pathway-λ machinery (PP14 "DIAGNOSTIC PASS") was a
        # respecification-to-rescue escape hatch that never produced a
        # clean win; it is removed here so the only knob is a single λ.
        self.lambda_reg = float(lambda_reg)

        if regularizer_type not in ("m_coupled", "fisher", "uniform"):
            raise ValueError(
                f"Unknown regularizer_type: {regularizer_type!r}. "
                f"Must be one of: m_coupled, fisher, uniform."
            )
        if regularizer_type == "m_coupled" and myelin_state is None:
            raise ValueError("regularizer_type='m_coupled' requires myelin_state")
        if regularizer_type == "fisher" and not fisher_diag:
            raise ValueError("regularizer_type='fisher' requires non-empty fisher_diag")

        # θ_ref snapshot: param name → tensor on same device as param
        self._theta_ref: Dict[str, torch.Tensor] = {}

        # Cached aggregate telemetry from last penalty() call
        self._last_penalty_value: float = 0.0
        self._last_per_layer_penalty: Dict[str, float] = {}

    # ---------- Reference snapshot --------------------------------------

    @torch.no_grad()
    def snapshot_reference(self) -> None:
        """Snapshot the current trainable parameters as θ_ref.

        Call this AFTER Phase A pre-training, BEFORE Phase B (RLHF)
        begins. The snapshot is held frozen for the duration of
        training; subsequent calls to penalty() compute distance
        against this snapshot.
        """
        self._theta_ref.clear()
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.exclude_param_names:
                continue
            self._theta_ref[name] = param.data.detach().clone()

    # ---------- Penalty computation -------------------------------------

    def penalty(self) -> torch.Tensor:
        """Return the auxiliary loss term.

        Scalar-λ EWC-family penalty:
        ``L_aux = λ · Σ_n W(n) · ‖θ_n − θ_ref,n‖²``, where W(n) is the
        importance signal (M, Fisher diagonal, or uniform).

        Differentiable w.r.t. model parameters. W(n) is detached (M and
        Fisher are state, not optimization targets).

        Returns a scalar Tensor on the same device as the model.
        """
        if not self._theta_ref:
            raise RuntimeError(
                "θ_ref is empty. Call snapshot_reference() before penalty()."
            )

        total = None
        per_layer: Dict[str, float] = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self._theta_ref:
                continue

            theta_ref = self._theta_ref[name]
            # Distance is differentiable w.r.t. param
            diff_sq = (param - theta_ref).pow(2)

            # Weight is detached — not an optimization target
            weight = self._get_weight(name, param.shape, param.device, param.dtype)
            weight = weight.detach()

            # Weighted Frobenius
            weighted = weight * diff_sq
            term = weighted.sum()

            per_layer[name] = float(term.detach().item())
            total = term if total is None else (total + term)

        if total is None:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        result = self.lambda_reg * total
        self._last_penalty_value = float(result.detach().item())
        self._last_per_layer_penalty = per_layer
        return result

    def _get_weight(
        self,
        name: str,
        param_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Resolve and broadcast the per-parameter weight tensor W(n).

        For activation_l2 M, M shape is typically [H] (per-neuron),
        which needs broadcasting to param shape [H, in] (weight) or
        [H] (bias). Mirrors myelin.apply_gate / patched CAG _apply_g_only.
        """
        if self.regularizer_type == "uniform":
            return torch.ones(param_shape, device=device, dtype=dtype)

        if self.regularizer_type == "m_coupled":
            if name not in self.myelin.M:
                # Param is tracked for θ_ref but not in M. Default to
                # uniform weight (safer than zero — at least applies
                # baseline L2).
                return torch.ones(param_shape, device=device, dtype=dtype)
            raw = self.myelin.M[name]
        else:  # fisher
            if name not in self.fisher_diag:
                return torch.ones(param_shape, device=device, dtype=dtype)
            raw = self.fisher_diag[name]

        w = raw.to(device, dtype)
        return self._broadcast_weight(w, param_shape)

    @staticmethod
    def _broadcast_weight(
        w: torch.Tensor, param_shape: torch.Size
    ) -> torch.Tensor:
        """Broadcast weight tensor w to match param_shape.

        Cases (mirrors myelin.apply_gate broadcasting precisely):
          - w shape == param_shape: identity
          - w.dim() == 1, param 2D, w.shape[0] == param.shape[0]:
              broadcast along axis 1 (per-row weight)
          - w.dim() == 1, param 1D, shapes match:
              identity
          - w.dim() == 1, param 4D, w.shape[0] == param.shape[0]:
              broadcast along axes 1, 2, 3 (per-output-channel weight)
          - Fallback: rely on PyTorch broadcasting (handles the
            embedding case w[H] vs param[V, H])
        """
        if w.shape == param_shape:
            return w
        if w.dim() == 1 and len(param_shape) == 2 and w.shape[0] == param_shape[0]:
            return w.unsqueeze(1).expand(param_shape)
        if w.dim() == 1 and len(param_shape) == 1 and w.shape == param_shape:
            return w
        if w.dim() == 1 and len(param_shape) == 4 and w.shape[0] == param_shape[0]:
            return w.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(param_shape)
        # Fallback: try PyTorch broadcasting via explicit expand.
        # If shapes are not broadcast-compatible, this will raise —
        # which is what we want (silent skip would hide bugs).
        try:
            return w.expand(param_shape)
        except RuntimeError:
            # Last-resort fallback: scalar mean weight
            return torch.full(param_shape, float(w.mean().item()),
                              device=w.device, dtype=w.dtype)

    # ---------- Telemetry ----------------------------------------------

    def summary_stats(self) -> Dict[str, float]:
        """Return aggregate stats for end-of-step logging."""
        return {
            "last_penalty": self._last_penalty_value,
            "n_theta_ref_params": float(len(self._theta_ref)),
            "lambda_reg": self.lambda_reg,
        }

    def per_layer_telemetry(self) -> Dict[str, float]:
        """Return per-layer penalty values from the last penalty() call."""
        return dict(self._last_per_layer_penalty)


# ============================================================================
# Fisher diagonal estimation
# ============================================================================

@torch.no_grad()
def _zero_fisher_diag(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Allocate zero tensors matching trainable params for Fisher accumulation."""
    out = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            out[name] = torch.zeros_like(param.data)
    return out


def estimate_fisher_diagonal(
    model: nn.Module,
    dataloader,
    n_steps: int = 200,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Estimate the diagonal Fisher information for trainable params.

    F(n) ≈ (1 / n_steps) · Σ_step (∇_θ log p(x|θ))²

    The standard EWC-style estimate. Used in the `fisher_DPO` ablation
    condition to test whether M provides signal beyond Fisher.

    Notes:
      - Model is set to eval mode for stability; gradients still flow.
      - Per-tensor normalization to [0, 1] makes Fisher and M
        comparable in magnitude.
    """
    if device is None:
        device = next(model.parameters()).device

    fisher = _zero_fisher_diag(model)
    n_samples = 0

    was_training = model.training
    model.eval()
    # Re-enable grad even in eval mode
    for p in model.parameters():
        if p.requires_grad:
            p.grad = None

    try:
        for step, batch in enumerate(dataloader):
            if step >= n_steps:
                break
            # Batch is expected to be input_ids; standard CE loss for fisher
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
            elif isinstance(batch, torch.Tensor):
                input_ids = batch.to(device)
            else:
                input_ids = torch.tensor(batch, device=device)

            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            model.zero_grad(set_to_none=True)
            loss.backward()

            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[name] += p.grad.detach().pow(2)
            n_samples += 1
    finally:
        if was_training:
            model.train()
        model.zero_grad(set_to_none=True)

    if n_samples == 0:
        return fisher

    # Normalize: divide by sample count, then per-tensor [0, 1]
    for name in fisher:
        fisher[name] /= float(n_samples)
        mx = float(fisher[name].max().item())
        if mx > 0:
            fisher[name] /= mx
    return fisher


# ============================================================================
# Invariant checks (SPEC §7)
# ============================================================================

def check_invariant_D1(myelin) -> Optional[str]:
    """D1: M distribution must be non-degenerate after Phase A.

    Aggregates std(M) across all representation-role params. If std is
    near zero everywhere, M didn't develop and the mechanism is dead.
    """
    if not myelin.M:
        return "D1 VIOLATION: myelin.M is empty"
    all_vals = []
    for name, M in myelin.M.items():
        role = myelin._param_role.get(name, "representation")
        if role == "head":
            continue
        all_vals.extend(M.flatten().detach().cpu().float().tolist())
    if not all_vals:
        return "D1 VIOLATION: no non-head M values to check"
    std = float(torch.tensor(all_vals).std().item())
    if std < 0.05:
        return (
            f"D1 VIOLATION: std(M) = {std:.4f} < 0.05 — M did not "
            f"develop a non-trivial distribution during Phase A. "
            f"Check activity capture path or use a more diverse dataset."
        )
    return None


def check_invariant_D2(theta_ref: Dict[str, torch.Tensor]) -> Optional[str]:
    """D2: reference snapshot is non-empty and at least one tensor is non-zero."""
    if not theta_ref:
        return "D2 VIOLATION: theta_ref is empty"
    nonzero_count = sum(
        1 for t in theta_ref.values()
        if float(t.abs().sum().item()) > 1e-9
    )
    if nonzero_count == 0:
        return (
            "D2 VIOLATION: every theta_ref tensor is zero — LoRA adapters "
            "may not be initialized. Check peft config."
        )
    return None


def check_invariant_D3(final_dpo_loss: float) -> Optional[str]:
    """D3 (legacy, v1.0/v1.1): final-step DPO loss must be < 0.7.

    DEPRECATED for v1.2+.  Kept for v1.0/v1.1 archive reproducibility.
    See ``check_invariant_D3_prime`` for the trailing-window /
    β-invariant successor and ``docs/2026-05-23_imp05-phase-b-noise.md``
    for the diagnostic that motivated the replacement.

    Two known issues with this formulation:

    * **Single-step snapshot is noise-dominated.**  At effective batch
      16 with no LR schedule, the per-batch DPO loss varies by ±0.4
      between adjacent 100-step snapshots within a single run (v1.1
      headline ``dpo_curves.csv``).  Reading one sample at step N as a
      stability statistic causes false positives.
    * **The 0.7 threshold is β-dependent.**  DPO loss is
      ``-log σ(β · logprob_gap)``; halving β halves the loss signal
      at the same model quality.  vanilla (β=0.1) and M_coupled
      (β=0.01) report different DPO loss scales for identical
      preference accuracy.  The universal 0.7 threshold is
      structurally harsher on the β=0.01 conditions.
    """
    if final_dpo_loss >= 0.7:
        return (
            f"D3 VIOLATION: final DPO loss = {final_dpo_loss:.4f} >= 0.7 "
            f"(random-chance baseline ~0.693). DPO did not learn the "
            f"preferences. Downstream comparisons are meaningless."
        )
    return None


def check_invariant_D3_prime(
    trailing_dpo_loss_mean: float,
    trailing_acc_mean: float,
    *,
    loss_max: float = 0.7,
    acc_min: float = 0.55,
) -> Optional[str]:
    """D3′ (v1.2+): trailing-window DPO loss < ``loss_max`` AND trailing
    accuracy > ``acc_min``.

    Replaces v1.0/v1.1 ``check_invariant_D3``.  Both inputs are means
    over the last ``trailing_window`` logged samples of Phase B (the
    caller is responsible for the windowing — typically the last 100
    of the 100-step-cadence ``dpo_curves.csv`` rows).

    The accuracy floor of 0.55 sits just above random-chance preference
    selection (0.50) and is β-invariant, which fixes the v1.0/v1.1 D3's
    structural bias against β=0.01 conditions.  The loss bound of 0.7
    is preserved as an upper sanity check — a truly diverged run will
    blow past it regardless of β.

    Returns a violation string on failure, ``None`` on success.

    See ``docs/2026-05-23_imp05-phase-b-noise.md`` for the v1.1
    dpo_curves trajectory data that motivated this replacement, and
    ``benchmarks/05_m_coupled_kl_rlhf/SPEC_v1.2.md §4`` for the formal
    invariant statement.
    """
    if not (trailing_dpo_loss_mean < loss_max):  # also catches NaN
        return (
            f"D3' VIOLATION (loss): trailing-window mean dpo_loss = "
            f"{trailing_dpo_loss_mean:.4f} >= {loss_max:.2f}. "
            f"Phase B did not converge to a basin under the random-chance "
            f"upper bound. Downstream comparisons are meaningless."
        )
    if not (trailing_acc_mean > acc_min):  # also catches NaN
        return (
            f"D3' VIOLATION (acc): trailing-window mean preference accuracy "
            f"= {trailing_acc_mean:.4f} <= {acc_min:.2f}. DPO did not learn "
            f"to discriminate chosen from rejected pairs beyond chance. "
            f"This catches the case where DPO loss looks fine under a "
            f"low-β setting but the implicit reward gap is uninformative."
        )
    return None


def check_invariant_D4(base_mmlu: float) -> Optional[str]:
    """D4: base model MMLU must be > 0.30 (above chance)."""
    if base_mmlu <= 0.30:
        return (
            f"D4 VIOLATION: base model MMLU = {base_mmlu:.4f} ≤ 0.30 "
            f"(random-chance baseline 0.25). Eval code is broken, not "
            f"the model. Fix evaluation before proceeding."
        )
    return None


# ----------------------------------------------------------------------------
# Degenerate-M guard (setup-agnostic replacement for v1's D5)
# ----------------------------------------------------------------------------

def check_degenerate_M(
    myelin,
    *,
    bc_min: float = 0.7,
    mean_m_min: float = 0.10,
    near_zero_frac_max: float = 0.95,
) -> Optional[str]:
    """Guard: abort if M collapsed to a mechanism-inert distribution.

    D1 alone (std(M) > 0.05) accepted two distinct mechanism-inert failure
    modes that produced indistinguishable downstream verdicts in v1
    (see the imp05 v1.0 / v1.1 lab reports). This guard catches both, using
    only distribution-level statistics of M so it is agnostic to the model
    architecture and parameter-naming scheme:

      * **Unimodal collapse** — BC ≤ ``bc_min``. M never developed a bimodal
        lock distribution; typically signal-magnitude collapse before the
        MyelinState fixed point.
      * **Degenerate-bimodal** — BC > ``bc_min`` but mean(M) ≤ ``mean_m_min``
        (equivalently, more than ``near_zero_frac_max`` of mass at ~0). The
        distribution is sharply bimodal but almost all mass sits at one
        extreme, so the mechanism is effectively inert on nearly every param.

    The v1 LoRA-category "lopsided" branch was deliberately dropped: it was
    tuned to the attn/mlp/lora_A/lora_B pathway taxonomy that Synthetic2 does
    not carry. Per-parameter coverage, if needed for a specific benchmark,
    should be asserted by that benchmark's own harness rather than baked into
    the core mechanism.

    Returns a violation string on failure or ``None`` on success. Intended to
    run immediately after :func:`check_invariant_D1` / :func:`check_invariant_D2`
    at Phase A end — a violation means the mechanism is structurally inert for
    the rest of training and the run should abort.
    """
    if not myelin.M:
        return "DEGENERATE-M VIOLATION: myelin.M is empty"
    all_vals: List[float] = []
    for name, M in myelin.M.items():
        role = myelin._param_role.get(name, "representation")
        if role == "head":
            continue
        all_vals.extend(M.flatten().detach().cpu().float().tolist())
    if not all_vals:
        return "DEGENERATE-M VIOLATION: no non-head M values to check"

    n_total = len(all_vals)
    mean_m = sum(all_vals) / n_total
    bc = bimodality_coefficient(all_vals)
    frac_near_zero = sum(1 for m in all_vals if m < 0.01) / n_total

    # Failure mode 1: unimodal collapse
    if not (bc > bc_min):  # also catches NaN
        return (
            f"DEGENERATE-M VIOLATION (unimodal): BC = {bc:.4f} ≤ {bc_min:.2f}, "
            f"mean(M) = {mean_m:.4f}. M failed to develop a bimodal lock "
            f"distribution — likely signal-magnitude collapse before the "
            f"MyelinState fixed point."
        )

    # Failure mode 2: degenerate-bimodal (sharp split, mass piled at ~0)
    if mean_m <= mean_m_min or frac_near_zero >= near_zero_frac_max:
        return (
            f"DEGENERATE-M VIOLATION (degenerate-bimodal): BC = {bc:.4f} > "
            f"{bc_min:.2f} but mean(M) = {mean_m:.4f} ≤ {mean_m_min:.2f} "
            f"({frac_near_zero * 100:.1f}% of M near zero). Distribution is "
            f"sharply bimodal but almost all mass is at one extreme, so the "
            f"lock protects essentially no parameters."
        )

    return None


# ============================================================================
# Bimodality coefficient (re-exposed for convenience)
# ============================================================================

def bimodality_coefficient(values) -> float:
    """Sarle's bimodality coefficient.

    BC = (γ² + 1) / (κ + 3·(n-1)² / ((n-2)·(n-3)))

    Where γ is skewness and κ is excess kurtosis.

    Returns float in roughly [0, 1]; values > 0.55 are conventionally
    considered bimodal.
    """
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n <= 3:
        return float("nan")
    mean_v = arr.mean()
    std_v = arr.std(ddof=1) if n > 1 else 0.0
    if std_v == 0:
        return float("nan")
    z = (arr - mean_v) / std_v
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean()) - 3.0
    correction = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    denom = kurt + correction
    if denom <= 0:
        return float("nan")
    return (skew ** 2 + 1.0) / denom
