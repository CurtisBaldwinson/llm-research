"""Hard-degenerate guards share one corruption set across M and Fisher."""
from __future__ import annotations

import torch

from synthetic_myelination.batch_utils import (
    check_degenerate_importance,
    check_m_hard_degenerate,
)


class _FakeMyelin:
    def __init__(self, M: dict, roles: dict | None = None):
        self.M = M
        self._param_role = roles or {}


def test_fisher_and_m_agree_on_empty():
    assert check_degenerate_importance({}, name="fisher") is not None
    assert check_m_hard_degenerate(_FakeMyelin({})) is not None


def test_fisher_and_m_agree_on_all_zero():
    z = {"w": torch.zeros(4)}
    f = check_degenerate_importance(z, name="fisher")
    m = check_m_hard_degenerate(_FakeMyelin(z))
    assert f is not None and "all-zero" in f
    assert m is not None and "all-zero" in m


def test_fisher_and_m_agree_on_nan():
    t = torch.ones(3)
    t[1] = float("nan")
    w = {"w": t}
    f = check_degenerate_importance(w, name="fisher")
    m = check_m_hard_degenerate(_FakeMyelin(w))
    assert f is not None and "non-finite" in f
    assert m is not None and "non-finite" in m


def test_fisher_and_m_agree_on_inf():
    t = torch.ones(3)
    t[0] = float("inf")
    w = {"w": t}
    f = check_degenerate_importance(w, name="fisher")
    m = check_m_hard_degenerate(_FakeMyelin(w))
    assert f is not None and "non-finite" in f
    assert m is not None and "non-finite" in m


def test_healthy_passes():
    w = {"w": torch.tensor([0.0, 0.5, 1.0])}
    assert check_degenerate_importance(w, name="fisher") is None
    assert check_m_hard_degenerate(_FakeMyelin(w)) is None


def test_m_skips_head_role_only_then_empty():
    # Only head-role mass → empty non-head set → degenerate.
    m = _FakeMyelin(
        {"head.weight": torch.ones(3)},
        roles={"head.weight": "head"},
    )
    assert check_m_hard_degenerate(m) is not None
