"""REMOVED in Synthetic2 prune (see README "What is being ported").

This test covered the v1 per-pathway machinery (per-pathway-lambda,
per_pathway_M, the attn/mlp/lora_A/lora_B taxonomy, or the D5 lopsided
per-category coverage check) that Synthetic2 deliberately drops. The
mechanism it tested no longer exists in sm_core.

The sandbox write policy on this path does not permit file deletion, so
this file is emptied to a no-op skip rather than removed. It is safe to
`git rm` this file.
"""
import pytest

pytestmark = pytest.mark.skip(
    reason="pathway machinery removed in Synthetic2 prune; see README"
)
