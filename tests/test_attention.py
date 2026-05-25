"""Integration test for forward-hook attention extraction.

This is the *spec* for `hybrid_arch.attention.extract_attention`. The function
must produce a NaN-free, properly normalized, causally-masked attention
tensor for any Pythia/GPTNeoX model — including the deep layers (9-11) where
the framework's `output_attentions=True` returns NaN due to softmax overflow.

Downloads Pythia-160M @ step143000 from HuggingFace on first run (~330 MB).
Subsequent runs use the local HF cache.
"""

from __future__ import annotations

import pytest
import torch

# Importing here so pytest collection works even if transformers is missing,
# though our pyproject.toml pins it as a hard dep.
transformers = pytest.importorskip("transformers")

from hybrid_arch.attention import extract_attention  # noqa: E402


@pytest.fixture(scope="module")
def attention_tensor(pythia_model_and_tokenizer):
    """One forward pass through a short prompt; reused across assertions."""
    model, tok = pythia_model_and_tokenizer
    text = "The quick brown fox jumps over the lazy dog."
    input_ids = tok(text, return_tensors="pt").input_ids
    attn = extract_attention(model, input_ids)
    return attn, model, input_ids


def test_attention_shape(attention_tensor):
    """(a) Output is [layers, batch, heads, seq, seq]."""
    attn, model, input_ids = attention_tensor
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    batch, seq = input_ids.shape
    assert attn.shape == (n_layers, batch, n_heads, seq, seq), (
        f"got {tuple(attn.shape)}, expected "
        f"({n_layers}, {batch}, {n_heads}, {seq}, {seq})"
    )


def test_attention_rows_sum_to_one(attention_tensor):
    """(b) Each row of the attention matrix is a probability distribution."""
    attn, _, _ = attention_tensor
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), (
        f"max deviation from 1.0: {(row_sums - 1.0).abs().max().item():.2e}"
    )


def test_attention_upper_triangle_is_zero(attention_tensor):
    """(c) Causal mask: query at position i cannot attend to keys at j > i."""
    attn, _, input_ids = attention_tensor
    seq = input_ids.shape[1]
    idx = torch.arange(seq)
    # True where key index > query index (strictly above diagonal).
    upper = idx.unsqueeze(0) > idx.unsqueeze(1)
    # attn shape [L, B, H, S, S] — apply mask to last two dims.
    upper_vals = attn[..., upper]
    assert (upper_vals == 0).all(), (
        f"upper triangle contained nonzero values; max = "
        f"{upper_vals.abs().max().item():.2e}"
    )


def test_attention_has_no_nan(attention_tensor):
    """(d) No NaN anywhere — especially in deep layers (9-11) where the
    framework's eager/SDPA softmax overflows on Pythia."""
    attn, _, _ = attention_tensor
    assert not torch.isnan(attn).any(), (
        f"NaN detected in layers: "
        f"{torch.isnan(attn).any(dim=(1, 2, 3, 4)).nonzero(as_tuple=True)[0].tolist()}"
    )


def test_attention_deep_layers_are_finite(attention_tensor):
    """Extra safeguard: the deep-layer NaN bug is the whole reason this
    module exists. Verify layers 9-11 specifically are finite."""
    attn, _, _ = attention_tensor
    for layer_idx in (9, 10, 11):
        layer_attn = attn[layer_idx]
        assert torch.isfinite(layer_attn).all(), (
            f"layer {layer_idx} contains non-finite values"
        )
