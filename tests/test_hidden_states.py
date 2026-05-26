"""Tests for `extract_hidden_states` and its integration with the cache.

The probe in Phase 3 consumes hidden states; if extraction is wrong or
silently shape-shifts, the probe trains on garbage. The tests below pin:

- shape `[L, B, S, H]`
- finite (no NaN/Inf) across all layers, including the deep ones that
  return NaN under the framework's `output_attentions=True` path
- equivalence with the framework's `output_hidden_states=True` view
  (last `len(layers)` entries, ignoring embedding)
- cache round-trip keeps fp16 dtype so on-disk size stays in budget
"""

from __future__ import annotations

import torch

from hybrid_arch import extract_hidden_states, metric_battery


def test_shape_and_finite(pythia_model_and_tokenizer):
    model, tok = pythia_model_and_tokenizer
    ids = tok("the quick brown fox", return_tensors="pt").input_ids
    hs = extract_hidden_states(model, ids)
    L = model.config.num_hidden_layers
    H = model.config.hidden_size
    assert hs.shape == (L, 1, ids.shape[1], H)
    assert hs.dtype == torch.float32
    assert torch.isfinite(hs).all()


def test_matches_framework_hidden_states(pythia_model_and_tokenizer):
    """The hook captures each `GPTNeoXLayer`'s raw output (the residual stream).

    `model(output_hidden_states=True).hidden_states` is laid out as
    `(embedding, layer_0_input, ..., layer_{L-1}_input, final_post_LN)`,
    i.e. each entry is the *input* to the next layer except for the last,
    which is `final_layer_norm` applied to the loop's final residual.

    So our hook[i] equals framework[i+1] for i in 0..L-2. The final layer
    won't match because the framework slot is post-final-LN; we apply
    that LN here and check equality.
    """
    model, tok = pythia_model_and_tokenizer
    ids = tok("a short test", return_tensors="pt").input_ids
    with torch.no_grad():
        framework = model(input_ids=ids, output_hidden_states=True).hidden_states
    hook = extract_hidden_states(model, ids)
    L = model.config.num_hidden_layers
    assert len(framework) == L + 1

    # Intermediate layers: hook[i] is the input to the next layer.
    for i in range(L - 1):
        assert torch.allclose(framework[i + 1], hook[i], atol=1e-5), f"layer {i} mismatch"

    # Final layer: the framework slot is the post-final-LN view of
    # `hook[L-1]`.
    final_ln = model.gpt_neox.final_layer_norm
    with torch.no_grad():
        ln_applied = final_ln(hook[L - 1].to(next(final_ln.parameters()).dtype))
    assert torch.allclose(framework[L], ln_applied, atol=1e-5)


def test_cache_stores_fp16(pythia_model_and_tokenizer, tmp_path):
    model, tok = pythia_model_and_tokenizer
    ids = tok("the quick brown fox jumps over the lazy dog", return_tensors="pt").input_ids[:, :8]
    out = metric_battery(
        "160m", 143000, "testset", ids,
        k_parallel=2, cache_dir=tmp_path, model=model,
    )
    assert "hidden_states_per_layer" in out
    L = model.config.num_hidden_layers
    H = model.config.hidden_size
    hs = out["hidden_states_per_layer"]
    assert hs.shape == (L, 1, 8, H)
    assert hs.dtype == torch.float16

    # Cache hit returns identical fp16 tensor.
    out2 = metric_battery(
        "160m", 143000, "testset", ids,
        k_parallel=2, cache_dir=tmp_path,
    )
    assert torch.equal(hs, out2["hidden_states_per_layer"])


def test_cache_opt_out(pythia_model_and_tokenizer, tmp_path):
    model, tok = pythia_model_and_tokenizer
    ids = tok("the quick brown fox jumps over the lazy dog", return_tensors="pt").input_ids[:, :8]
    out = metric_battery(
        "160m", 143000, "testset_no_hs", ids,
        k_parallel=2, cache_dir=tmp_path, model=model,
        store_hidden_states=False,
    )
    assert "hidden_states_per_layer" not in out
