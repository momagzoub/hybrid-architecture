"""Tests for parallel_prediction_agreement.

Two layers:
1. Deterministic unit tests with stub "models" whose teacher-forced and
   autoregressive behavior is exactly predictable. These pin down the
   function's logic without needing to download a real model.
2. One Pythia integration test: a repetitive prompt ("the the the …")
   should produce nearly-perfect agreement, since after a few tokens the
   model converges on predicting "the" regardless of context.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hybrid_arch.metrics import parallel_prediction_agreement

# ---------- stub models ----------


class _ConstantPredictionModel(nn.Module):
    """A 'model' whose argmax is always token 0, regardless of input.

    Teacher-forced predictions and autoregressive rollouts both produce
    token 0 at every position → agreement is always True. The attention
    mask kwarg is accepted but ignored.
    """

    def __init__(self, vocab_size: int = 16):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, self.vocab_size)
        logits[..., 0] = 100.0
        return SimpleNamespace(logits=logits)


class _CopyLastInputModel(nn.Module):
    """A 'model' whose prediction at position i is `input_ids[:, i]` itself
    (it copies whatever's in the input slot at position i). This makes the
    teacher-forced/AR divergence calculable by hand.

    At position t the teacher-forced prediction at step j is `input_ids[t+j]`
    (the real token at that position). The autoregressive rollout, starting
    from `input_ids[:t+1]`, predicts `input_ids[t]` and then keeps predicting
    that same token forever (since each AR step makes it the new last token).
    So agreement at step j = `input_ids[t+j] == input_ids[t]`.

    The attention_mask kwarg is accepted and ignored — the model is a pure
    function of `input_ids`, which is the right behavior for testing the
    batched parallel-prediction-agreement code path: the caller only looks
    at logits at each row's last *real* position (index `t+j`), and what
    the stub returns for padded slots is irrelevant to the comparison.
    """

    def __init__(self, vocab_size: int = 16):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        batch, seq = input_ids.shape
        logits = torch.full((batch, seq, self.vocab_size), -100.0)
        for b in range(batch):
            for i in range(seq):
                logits[b, i, input_ids[b, i]] = 100.0
        return SimpleNamespace(logits=logits)


# ---------- shape & contract ----------


def test_output_shape_and_dtype():
    """[B, S] in → [B, S-k, k] bool out. Uses batched=False since the
    batched implementation requires batch=1."""
    model = _ConstantPredictionModel(vocab_size=8)
    input_ids = torch.zeros(2, 10, dtype=torch.long)
    agreement = parallel_prediction_agreement(model, input_ids, k=3, batched=False)
    assert agreement.shape == (2, 7, 3)
    assert agreement.dtype == torch.bool


def test_output_shape_batched_path():
    """Same shape contract, but exercising the batched code path."""
    model = _ConstantPredictionModel(vocab_size=8)
    input_ids = torch.zeros(1, 10, dtype=torch.long)
    agreement = parallel_prediction_agreement(model, input_ids, k=3, batched=True)
    assert agreement.shape == (1, 7, 3)
    assert agreement.dtype == torch.bool


def test_rejects_k_too_large():
    model = _ConstantPredictionModel()
    input_ids = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="need seq > k"):
        parallel_prediction_agreement(model, input_ids, k=4)


def test_rejects_k_below_one():
    model = _ConstantPredictionModel()
    input_ids = torch.zeros(1, 5, dtype=torch.long)
    with pytest.raises(ValueError, match=">= 1"):
        parallel_prediction_agreement(model, input_ids, k=0)


# ---------- deterministic unit tests ----------


def test_constant_model_always_agrees():
    """If the model's argmax is constant, TF and AR predictions are identical
    everywhere → agreement is True at every (position, step)."""
    model = _ConstantPredictionModel(vocab_size=8)
    # Input has varied tokens; the model ignores them and always predicts 0.
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]])
    agreement = parallel_prediction_agreement(model, input_ids, k=3)
    assert agreement.all(), "constant-prediction model must give 100% agreement"


def test_copy_last_input_model_known_pattern():
    """For the copy-last-input stub, the model's argmax at every output
    position equals input_ids at that position. So:
      - TF predictions: tf_pred[t, j] = input_ids[t+j].
      - AR predictions: AR from ctx [0..t] predicts input_ids[t], then keeps
        predicting that same token (since it becomes the new rightmost slot).
        So ar_pred[t, j] = input_ids[t] for all j.
      - Agreement[t, j] = (input_ids[t+j] == input_ids[t]).

    For input [3, 3, 7, 7, 2, 2] and k=2, this gives:
      j=0 column: all True (both see identical context — structural).
      j=1 column: input[t+1] vs input[t] → (3==3, 3==7, 7==7, 7==2) = (T,F,T,F).
    """
    model = _CopyLastInputModel(vocab_size=16)
    input_ids = torch.tensor([[3, 3, 7, 7, 2, 2]])
    agreement = parallel_prediction_agreement(model, input_ids, k=2)
    expected = torch.tensor([[
        [True, True ],
        [True, False],
        [True, True ],
        [True, False],
    ]])
    assert torch.equal(agreement, expected)


def test_j0_always_agrees_structural_invariant():
    """At step j=0, teacher-forced and autoregressive predictions BOTH start
    from the same context `input_ids[..., 0..t]` and look at the same forward
    pass output position, so they must always agree. This holds for any
    model, not just our stubs. It's a strong structural correctness check."""
    model = _CopyLastInputModel(vocab_size=16)
    # Mix of patterns to make sure no edge case sneaks through.
    input_ids = torch.tensor([[5, 1, 1, 9, 2, 2, 3, 7]])
    agreement = parallel_prediction_agreement(model, input_ids, k=4)
    assert agreement[..., 0].all(), "j=0 must always be True (same context)"


def test_copy_last_input_constant_sequence_full_agreement():
    """A constant input means every TF prediction equals the AR prediction,
    so the copy-last stub gives 100% agreement on constant input."""
    model = _CopyLastInputModel(vocab_size=16)
    input_ids = torch.full((1, 6), 5, dtype=torch.long)
    agreement = parallel_prediction_agreement(model, input_ids, k=3)
    assert agreement.all()


# ---------- Pythia integration ----------


def test_pythia_j0_invariant(pythia_model_and_tokenizer):
    """The structural j=0 invariant must hold for a real model too."""
    model, tok = pythia_model_and_tokenizer
    text = "The quick brown fox jumps over the lazy dog."
    input_ids = tok(text, return_tensors="pt").input_ids
    agreement = parallel_prediction_agreement(model, input_ids, k=3)
    assert agreement[..., 0].all(), "j=0 must always be True (same context)"


def test_batched_matches_sequential_on_pythia(pythia_model_and_tokenizer):
    """The new batched implementation must produce element-wise identical
    output to the (already-tested) sequential implementation. This is the
    critical correctness check before scaling Phase 2 to 108 model runs.
    """
    model, tok = pythia_model_and_tokenizer
    text = "The quick brown fox jumps over the lazy dog and the cat watched."
    input_ids = tok(text, return_tensors="pt").input_ids

    seq = parallel_prediction_agreement(model, input_ids, k=3, batched=False)
    bat = parallel_prediction_agreement(model, input_ids, k=3, batched=True)

    assert torch.equal(seq, bat), (
        "batched and sequential disagree; "
        f"sequential mean = {seq.float().mean():.3f}, batched mean = {bat.float().mean():.3f}"
    )


def test_batched_matches_sequential_on_stubs():
    """Same agreement check, but with the stub models. Faster — no model
    download — and pins down padded-input handling in the batched path."""
    model = _CopyLastInputModel(vocab_size=16)
    input_ids = torch.tensor([[3, 3, 7, 7, 2, 2, 5, 1, 9]])
    seq = parallel_prediction_agreement(model, input_ids, k=4, batched=False)
    bat = parallel_prediction_agreement(model, input_ids, k=4, batched=True)
    assert torch.equal(seq, bat)


def test_batched_rejects_batch_above_one():
    model = _ConstantPredictionModel()
    input_ids = torch.zeros(2, 8, dtype=torch.long)
    with pytest.raises(NotImplementedError, match="batched=True"):
        parallel_prediction_agreement(model, input_ids, k=2, batched=True)


def test_pythia_repetitive_beats_diverse(pythia_model_and_tokenizer):
    """Project-thesis sanity check: a repetitive prompt should have STRICTLY
    HIGHER parallel-prediction agreement than a varied natural-language
    prompt of the same length. This is the parallelism signal the whole
    project is built on, so it had better show up at the metric level.
    """
    model, tok = pythia_model_and_tokenizer
    repetitive = "the the the the the the the the the the"
    diverse = "The quick brown fox jumps over the lazy sleeping dog."

    rep_ids = tok(repetitive, return_tensors="pt").input_ids
    div_ids = tok(diverse, return_tensors="pt").input_ids

    rep_rate = (
        parallel_prediction_agreement(model, rep_ids, k=3).float().mean().item()
    )
    div_rate = (
        parallel_prediction_agreement(model, div_ids, k=3).float().mean().item()
    )

    assert rep_rate > div_rate, (
        f"repetitive prompt should have higher agreement than diverse one; "
        f"got repetitive={rep_rate:.2%}, diverse={div_rate:.2%}"
    )
