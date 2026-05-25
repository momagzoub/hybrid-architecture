"""Tests for the per-head ↔ aggregate equivalence at the attention metrics layer.

The Phase 2 signature analysis depends on having per-(layer, head) granularity.
The Phase 1 functions already preserved leading dims; these tests pin that
contract down so a future "simplification" can't quietly aggregate too early.
"""

from __future__ import annotations

import pytest
import torch

from hybrid_arch.metrics import (
    aggregate_attention_concentration,
    aggregate_attention_entropy,
    attention_concentration,
    attention_entropy,
)


def _random_attention(L=3, B=2, H=4, S=5, seed=0):
    """A right-stochastic causal attention tensor: rows sum to 1, upper tri = 0."""
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(L, B, H, S, S, generator=g)
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1)


def test_attention_entropy_returns_per_head_shape():
    attn = _random_attention()
    h = attention_entropy(attn)
    assert h.shape == (3, 2, 4, 5)


def test_attention_concentration_returns_per_head_shape():
    attn = _random_attention()
    c = attention_concentration(attn, top_k=(1, 3))
    assert c.shape == (2, 3, 2, 4, 5)


def test_aggregate_entropy_is_mean_over_layers_and_heads():
    attn = _random_attention()
    per_head = attention_entropy(attn)              # [L, B, H, S]
    agg = aggregate_attention_entropy(per_head)     # [B, S]
    assert agg.shape == (2, 5)
    expected = per_head.mean(dim=(0, 2))
    assert torch.allclose(agg, expected, atol=1e-6)


def test_aggregate_concentration_is_mean_over_layers_and_heads():
    attn = _random_attention()
    per_head = attention_concentration(attn, top_k=(1, 3))  # [K, L, B, H, S]
    agg = aggregate_attention_concentration(per_head)        # [K, B, S]
    assert agg.shape == (2, 2, 5)
    expected = per_head.mean(dim=(1, 3))
    assert torch.allclose(agg, expected, atol=1e-6)


def test_aggregate_entropy_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\[L, B, H, S\]"):
        aggregate_attention_entropy(torch.zeros(2, 3, 4))


def test_aggregate_concentration_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\[K, L, B, H, S\]"):
        aggregate_attention_concentration(torch.zeros(2, 3, 4))


def test_per_head_has_no_nan_under_causal_softmax():
    """Causal masking puts exact zeros in the upper triangle. The xlogy-based
    entropy must handle them without NaN — this is the property that motivates
    the whole `attention.py` rewrite."""
    attn = _random_attention()
    assert not torch.isnan(attention_entropy(attn)).any()
    assert not torch.isnan(attention_concentration(attn, top_k=(1, 3, 5))).any()
