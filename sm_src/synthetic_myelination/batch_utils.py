"""Shared LM batch unpacking with pad-safe labels.

Padding must never contribute to CE loss, Fisher estimates, or M activity
signals. Callers pass batches that are either:
  * bare LongTensor [B, T] of input_ids (smoke path; no padding assumed), or
  * dict with ``input_ids`` and optional ``attention_mask`` (TRACE path).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch


BatchLike = Union[torch.Tensor, Dict[str, Any]]


def unpack_lm_batch(
    batch: BatchLike,
    device: torch.device,
    *,
    pad_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(input_ids, labels, attention_mask)`` on ``device``.

    Labels equal input_ids except pad positions (attention_mask==0, or
    token_id==pad_token_id when mask is absent) which are set to ``-100``
    so ``CrossEntropyLoss(ignore_index=-100)`` / HF causal LM skip them.
    """
    if isinstance(batch, dict):
        input_ids = batch["input_ids"].to(device)
        if "attention_mask" in batch and batch["attention_mask"] is not None:
            attention_mask = batch["attention_mask"].to(device)
        else:
            attention_mask = torch.ones_like(input_ids)
            if pad_token_id is not None:
                attention_mask = (input_ids != pad_token_id).long()
    else:
        input_ids = batch.to(device) if torch.is_tensor(batch) else torch.tensor(
            batch, device=device
        )
        attention_mask = torch.ones_like(input_ids)
        if pad_token_id is not None:
            attention_mask = (input_ids != pad_token_id).long()

    labels = input_ids.clone()
    labels = labels.masked_fill(attention_mask == 0, -100)
    return input_ids, labels, attention_mask


def check_degenerate_importance(
    weights: Dict[str, torch.Tensor],
    *,
    name: str = "importance",
) -> Optional[str]:
    """Hard guard: empty / all-zero / NaN / Inf importance diagonal.

    Distinct from ``check_degenerate_M`` (bimodality / lock distribution),
    which is a softer distributional diagnostic. Returns a violation string
    or ``None``.
    """
    if not weights:
        return f"DEGENERATE-{name.upper()} VIOLATION: empty weight dict"
    n = 0
    n_nonzero = 0
    n_bad = 0
    for t in weights.values():
        flat = t.detach().float().reshape(-1)
        n += flat.numel()
        n_nonzero += int((flat.abs() > 1e-12).sum().item())
        n_bad += int((~torch.isfinite(flat)).sum().item())
    if n == 0:
        return f"DEGENERATE-{name.upper()} VIOLATION: zero elements"
    if n_bad > 0:
        return (
            f"DEGENERATE-{name.upper()} VIOLATION: {n_bad}/{n} non-finite values"
        )
    if n_nonzero == 0:
        return (
            f"DEGENERATE-{name.upper()} VIOLATION: all-zero diagonal "
            f"({n} elements)"
        )
    return None


def check_m_hard_degenerate(myelin) -> Optional[str]:
    """Hard abort conditions on M: empty / all-zero / NaN / Inf.

    Delegates to :func:`check_degenerate_importance` so M and Fisher share
    the **same corruption set** (empty dict, zero elements, non-finite,
    all-zero). Head-role params are excluded (they are never myelinated).

    The full ``check_degenerate_M`` BC/bimodality diagnostic is still run by
    callers for telemetry, but must not hard-abort the Phase-1 isolation
    screen: a short post-task integral often has not yet developed BC>0.7,
    and treating that as cell death would kill the entire matrix.
    """
    if not getattr(myelin, "M", None):
        return "DEGENERATE-M VIOLATION: empty weight dict"
    weights: Dict[str, torch.Tensor] = {}
    for name, M in myelin.M.items():
        role = myelin._param_role.get(name, "representation")
        if role == "head":
            continue
        weights[name] = M
    if not weights:
        return "DEGENERATE-M VIOLATION: empty weight dict"
    return check_degenerate_importance(weights, name="M")
