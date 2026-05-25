"""Deterministic tests for attention-side metrics.

Hand-built attention tensors whose entropy and concentration can be
computed by hand — assertions against known scalars, not against the
function's own output.
"""

from __future__ import annotations

import math

import pytest
import torch

from hybrid_arch.metrics import attention_concentration, attention_entropy


def _single_position_attention(seq: int, hot: int = 0) -> torch.Tensor:
    """Build a [seq, seq] attention matrix where every row points entirely
    at column `hot`. Entropy of every row = 0; top-1 concentration = 1.0."""
    attn = torch.zeros(seq, seq)
    attn[:, hot] = 1.0
    return attn


def _uniform_attention(seq: int) -> torch.Tensor:
    """[seq, seq] uniform attention (no causal mask): each row is 1/seq.
    Entropy per row = log(seq); top-1 = 1/seq."""
    return torch.full((seq, seq), 1.0 / seq)


def _causal_uniform_attention(seq: int) -> torch.Tensor:
    """Causally-masked uniform attention. Row t has 1/(t+1) on cols 0..t,
    0 elsewhere. Entropy of row t = log(t+1); top-1 of row t = 1/(t+1)."""
    attn = torch.zeros(seq, seq)
    for t in range(seq):
        attn[t, : t + 1] = 1.0 / (t + 1)
    return attn


# ---------- attention_entropy ----------

def test_entropy_single_position_is_zero():
    """All mass on one key → row entropy = 0."""
    attn = _single_position_attention(seq=8, hot=3)
    h = attention_entropy(attn)
    assert h.shape == (8,)
    assert torch.allclose(h, torch.zeros(8), atol=1e-6)


def test_entropy_uniform_attention():
    """Uniform attention over S keys → row entropy = log(S)."""
    seq = 16
    attn = _uniform_attention(seq)
    h = attention_entropy(attn)
    expected = torch.full((seq,), math.log(seq))
    assert torch.allclose(h, expected, atol=1e-6)


def test_entropy_causal_uniform_grows_with_position():
    """Under causal mask, row t has t+1 valid keys → H = log(t+1)."""
    seq = 8
    attn = _causal_uniform_attention(seq)
    h = attention_entropy(attn)
    expected = torch.tensor([math.log(t + 1) for t in range(seq)])
    assert torch.allclose(h, expected, atol=1e-6)
    # Position 0 specifically must be exactly 0 (the structural fact).
    assert h[0].item() == 0.0


def test_entropy_handles_zero_entries():
    """Zero entries in masked positions must NOT produce NaN (0·log0 = 0
    convention). This is the main reason for using torch.special.xlogy."""
    seq = 5
    attn = _causal_uniform_attention(seq)
    h = attention_entropy(attn)
    assert not torch.isnan(h).any()


def test_entropy_preserves_leading_dims():
    """[L, B, H, S, S] in → [L, B, H, S] out."""
    attn = torch.full((4, 2, 3, 6, 6), 1.0 / 6)
    h = attention_entropy(attn)
    assert h.shape == (4, 2, 3, 6)
    # All values should equal log(6).
    assert torch.allclose(h, torch.full_like(h, math.log(6)), atol=1e-6)


# ---------- attention_concentration ----------

def test_concentration_single_position_top1_is_one():
    """All mass on one key → top-1 = 1.0, top-3 = 1.0, top-5 = 1.0."""
    attn = _single_position_attention(seq=8, hot=4)
    concs = attention_concentration(attn, top_k=(1, 3, 5))
    assert concs.shape == (3, 8)
    assert torch.allclose(concs, torch.ones_like(concs), atol=1e-6)


def test_concentration_uniform_attention():
    """Uniform over S → top-k = k/S for k ≤ S."""
    seq = 10
    attn = _uniform_attention(seq)
    concs = attention_concentration(attn, top_k=(1, 3, 5))
    expected = torch.tensor([
        [1 / seq] * seq,
        [3 / seq] * seq,
        [5 / seq] * seq,
    ])
    assert torch.allclose(concs, expected, atol=1e-6)


def test_concentration_top_k_clamps_to_seq_len():
    """If k > seq_len, clamp to seq_len → result = 1.0 (full mass)."""
    seq = 3
    attn = _uniform_attention(seq)
    concs = attention_concentration(attn, top_k=(1, 3, 5))  # k=5 > seq=3
    # k=1 → 1/3, k=3 → 1.0, k=5 → clamped → 1.0
    assert torch.allclose(concs[0], torch.full((seq,), 1 / 3), atol=1e-6)
    assert torch.allclose(concs[1], torch.ones(seq), atol=1e-6)
    assert torch.allclose(concs[2], torch.ones(seq), atol=1e-6)


def test_concentration_monotone_in_k():
    """top-k mass is non-decreasing in k for any valid distribution."""
    attn = torch.softmax(torch.randn(4, 4), dim=-1)
    concs = attention_concentration(attn, top_k=(1, 2, 3, 4))
    diffs = concs[1:] - concs[:-1]
    assert (diffs >= -1e-6).all()


def test_concentration_known_two_value():
    """Hand-checked: row = [0.5, 0.3, 0.1, 0.1] → top-1 = 0.5, top-3 = 0.9."""
    row = torch.tensor([[0.5, 0.3, 0.1, 0.1]])  # [1, 4]
    concs = attention_concentration(row, top_k=(1, 3))
    assert torch.allclose(concs[0], torch.tensor([0.5]), atol=1e-6)
    assert torch.allclose(concs[1], torch.tensor([0.9]), atol=1e-6)


def test_concentration_preserves_leading_dims():
    """[L, B, H, S, S] in → [K, L, B, H, S] out."""
    attn = torch.full((4, 2, 3, 6, 6), 1.0 / 6)
    concs = attention_concentration(attn, top_k=(1, 3, 5))
    assert concs.shape == (3, 4, 2, 3, 6)


def test_concentration_rejects_empty_top_k():
    with pytest.raises(ValueError, match="at least one"):
        attention_concentration(_uniform_attention(4), top_k=())


def test_concentration_rejects_zero_k():
    with pytest.raises(ValueError, match=">= 1"):
        attention_concentration(_uniform_attention(4), top_k=(0, 1))
