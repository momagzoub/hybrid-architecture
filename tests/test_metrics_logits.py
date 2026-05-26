"""Deterministic tests for logit-side metrics.

Each test uses a hand-constructed logits tensor whose entropy / top-1
probability can be computed by hand, so the assertion is against a
known scalar — not against the function's own output.
"""

from __future__ import annotations

import math

import torch

from hybrid_arch.metrics import next_token_entropy, top1_probability

# ---------- next_token_entropy ----------

def test_entropy_uniform_distribution():
    """Uniform over V items → H = log(V)."""
    vocab = 10
    logits = torch.zeros(1, 1, vocab)  # [batch, seq, vocab]
    h = next_token_entropy(logits)
    assert h.shape == (1, 1)
    assert torch.allclose(h, torch.tensor([[math.log(vocab)]]), atol=1e-6)


def test_entropy_delta_distribution():
    """A very-peaked logit vector → H ≈ 0."""
    logits = torch.full((1, 1, 5), -1e9)
    logits[0, 0, 2] = 0.0  # only this position has finite logit
    h = next_token_entropy(logits)
    assert torch.allclose(h, torch.zeros_like(h), atol=1e-6)


def test_entropy_bernoulli_half():
    """Two-class uniform → H = log(2)."""
    logits = torch.zeros(1, 1, 2)
    h = next_token_entropy(logits)
    assert torch.allclose(h, torch.tensor([[math.log(2)]]), atol=1e-6)


def test_entropy_known_two_class():
    """p = 0.7 / 0.3 → H = -(0.7 log 0.7 + 0.3 log 0.3)."""
    p, q = 0.7, 0.3
    expected = -(p * math.log(p) + q * math.log(q))
    # log p, log q as logits → softmax gives back (p, q).
    logits = torch.tensor([[[math.log(p), math.log(q)]]])
    h = next_token_entropy(logits)
    assert torch.allclose(h, torch.tensor([[expected]]), atol=1e-6)


def test_entropy_shape_preserves_leading_dims():
    """[batch, seq, vocab] in → [batch, seq] out."""
    logits = torch.randn(3, 7, 100)
    h = next_token_entropy(logits)
    assert h.shape == (3, 7)


def test_entropy_is_non_negative():
    """Entropy of any valid distribution is ≥ 0."""
    logits = torch.randn(2, 4, 50)
    h = next_token_entropy(logits)
    assert (h >= 0).all()


def test_entropy_uniform_is_maximum():
    """log(V) is the upper bound; any other distribution should be lower."""
    vocab = 32
    uniform_logits = torch.zeros(1, 1, vocab)
    skewed_logits = torch.randn(1, 1, vocab) * 5  # likely peaked
    h_uniform = next_token_entropy(uniform_logits).item()
    h_skewed = next_token_entropy(skewed_logits).item()
    assert h_uniform >= h_skewed
    assert math.isclose(h_uniform, math.log(vocab), abs_tol=1e-6)


# ---------- top1_probability ----------

def test_top1_uniform():
    """Uniform over V → top1 = 1/V."""
    vocab = 8
    logits = torch.zeros(1, 1, vocab)
    p = top1_probability(logits)
    assert p.shape == (1, 1)
    assert torch.allclose(p, torch.tensor([[1.0 / vocab]]), atol=1e-6)


def test_top1_delta():
    """A near-delta distribution → top1 ≈ 1.0."""
    logits = torch.full((1, 1, 5), -1e9)
    logits[0, 0, 2] = 0.0
    p = top1_probability(logits)
    assert torch.allclose(p, torch.ones_like(p), atol=1e-6)


def test_top1_known_two_class():
    """logits = [log 0.7, log 0.3] → top1 = 0.7."""
    logits = torch.tensor([[[math.log(0.7), math.log(0.3)]]])
    p = top1_probability(logits)
    assert torch.allclose(p, torch.tensor([[0.7]]), atol=1e-6)


def test_top1_shape_preserves_leading_dims():
    logits = torch.randn(3, 7, 100)
    p = top1_probability(logits)
    assert p.shape == (3, 7)


def test_top1_in_unit_interval():
    """top-1 prob is always in (0, 1]."""
    logits = torch.randn(2, 4, 50)
    p = top1_probability(logits)
    assert (p > 0).all()
    assert (p <= 1).all()


def test_top1_at_least_one_over_vocab():
    """top-1 ≥ 1/V always (pigeonhole)."""
    vocab = 50
    logits = torch.randn(2, 4, vocab)
    p = top1_probability(logits)
    assert (p >= 1.0 / vocab - 1e-6).all()
