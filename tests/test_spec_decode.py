"""Correctness tests for `hybrid_arch.spec_decode`.

The strongest invariant we can check without a second model: if the drafter
and the target are *the same model*, greedy speculative decoding must accept
every drafted token, because both sides compute identical argmaxes from
identical context. Any rejection in that setting is a bug in the
accept/reject bookkeeping (off-by-one in the position indexing being the
classic one).

We also pin the trace's shape contract, which the Phase 4 hybrid decoder
and the analysis scripts depend on.
"""

from __future__ import annotations

import torch

from hybrid_arch.spec_decode import spec_decode_capture


def test_self_drafting_accepts_everything(pythia_model_and_tokenizer):
    """drafter == target ⇒ accept rate 1.0 (the key correctness invariant)."""
    model, tok = pythia_model_and_tokenizer
    prompt = tok("The capital of France is", return_tensors="pt").input_ids
    trace = spec_decode_capture(model, model, prompt, n_steps=4, draft_k=3)
    assert trace.accept.all(), (
        f"self-drafting should accept all positions, got "
        f"accept_rate={trace.accept_rate:.3f}"
    )
    assert trace.accept_rate == 1.0
    # When everything is accepted, the drafter and target argmax tokens match.
    assert torch.equal(trace.drafter_token, trace.target_token)


def test_trace_shape_contract(pythia_model_and_tokenizer):
    model, tok = pythia_model_and_tokenizer
    prompt = tok("Once upon a time", return_tensors="pt").input_ids
    n_steps, draft_k = 3, 4
    trace = spec_decode_capture(model, model, prompt, n_steps=n_steps, draft_k=draft_k)
    n = n_steps * draft_k
    assert trace.accept.shape == (n,)
    assert trace.entropy.shape == (n,)
    assert trace.top1.shape == (n,)
    assert trace.drafter_token.shape == (n,)
    assert trace.target_token.shape == (n,)
    L = model.config.num_hidden_layers
    H = model.config.hidden_size
    assert trace.drafter_hidden_states.shape == (L, n, H)
    assert trace.n_drafted == n


def test_top1_in_unit_interval_and_entropy_nonneg(pythia_model_and_tokenizer):
    model, tok = pythia_model_and_tokenizer
    prompt = tok("A short prompt here", return_tensors="pt").input_ids
    trace = spec_decode_capture(model, model, prompt, n_steps=2, draft_k=3)
    assert torch.all(trace.top1 > 0) and torch.all(trace.top1 <= 1.0)
    assert torch.all(trace.entropy >= 0)


def test_draft_k_one(pythia_model_and_tokenizer):
    """draft_k=1 is the degenerate single-token case; must still self-accept."""
    model, tok = pythia_model_and_tokenizer
    prompt = tok("Edge case test", return_tensors="pt").input_ids
    trace = spec_decode_capture(model, model, prompt, n_steps=3, draft_k=1)
    assert trace.n_drafted == 3
    assert trace.accept.all()
