"""Tests for the Pythia checkpoint loader.

The `list_checkpoints` test is pure logic. The `load_pythia` smoke test
reuses the cached final checkpoint via the session fixture. The integration
test (early vs late entropy) downloads step0 on first run (~330 MB),
then runs ~1s on subsequent runs from the local HF cache.
"""

from __future__ import annotations

import math

import torch

from hybrid_arch.checkpoints import CANONICAL_STEPS, list_checkpoints, load_pythia
from hybrid_arch.metrics import next_token_entropy

# ---------- list_checkpoints ----------

def test_list_checkpoints_returns_list():
    steps = list_checkpoints()
    assert isinstance(steps, list)
    assert len(steps) >= 8, f"expected at least 8 steps, got {len(steps)}"


def test_list_checkpoints_monotone():
    steps = list_checkpoints()
    assert steps == sorted(steps), "steps should be monotone increasing"


def test_list_checkpoints_spans_full_training():
    steps = list_checkpoints()
    assert steps[0] == 0, "first step should be 0 (random init)"
    assert steps[-1] == 143000, "last step should be 143000 (final)"


def test_list_checkpoints_returns_fresh_copy():
    """Mutating the returned list must not affect future calls."""
    a = list_checkpoints()
    a.append(999999)
    b = list_checkpoints()
    assert 999999 not in b


def test_canonical_steps_constant_matches_default_list():
    """The CANONICAL_STEPS tuple is the source of truth."""
    assert list_checkpoints() == list(CANONICAL_STEPS)


# ---------- load_pythia ----------

def test_load_pythia_returns_model_and_tokenizer(pythia_model_and_tokenizer):
    """Smoke test against the session-cached final checkpoint."""
    model, tok = pythia_model_and_tokenizer
    assert hasattr(model, "config")
    assert tok is not None
    assert model.config.num_hidden_layers == 12
    assert model.config.hidden_size == 768


def test_load_pythia_early_step_high_entropy():
    """The project-critical sanity check: an early (random-init) checkpoint
    should produce near-uniform predictions, so its entropy is near
    `log(vocab_size)`. The final checkpoint should be far below that.

    This is the discriminator the metric battery relies on — if early and
    late checkpoints looked the same, there would be nothing to measure.
    """
    model_late, tok = load_pythia("160m", 143000)
    model_early, _ = load_pythia("160m", 0)  # ~330 MB on first run

    text = "The quick brown fox jumps over the lazy dog."
    input_ids = tok(text, return_tensors="pt").input_ids

    with torch.no_grad():
        late_logits = model_late(input_ids=input_ids).logits
        early_logits = model_early(input_ids=input_ids).logits

    late_h = next_token_entropy(late_logits)[0].mean().item()
    early_h = next_token_entropy(early_logits)[0].mean().item()

    vocab_size = model_late.config.vocab_size
    upper_bound = math.log(vocab_size)

    # Random init should be within a small margin of log(vocab).
    assert early_h > upper_bound - 1.0, (
        f"step 0 entropy {early_h:.3f} should be near upper bound "
        f"log({vocab_size}) = {upper_bound:.3f}"
    )
    # Trained model should be noticeably lower.
    assert late_h < early_h - 2.0, (
        f"step 143000 entropy ({late_h:.3f}) should be at least 2 nats below "
        f"step 0 ({early_h:.3f})"
    )
