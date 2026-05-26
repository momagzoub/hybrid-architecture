"""Tests for the metric-battery cache layer (`hybrid_arch.cache`).

The cache underwrites every Phase 2 experiment; correctness matters more
than coverage breadth. We exercise:

- slice_hash determinism and shape-sensitivity
- cache-miss writes the npz + manifest
- cache-hit returns tensors equal to the cache-miss result
- force_recompute bypasses the cache
- missing directories are created automatically
- manifest records model revision / slice hash / tokenizer

A single pythia forward pass is shared across tests via the session-scoped
fixture, so the suite stays fast on CPU (~30s end-to-end on a laptop).
"""

from __future__ import annotations

import json

import pytest
import torch

from hybrid_arch.cache import metric_battery, slice_hash

# ---------- slice_hash ----------

def test_slice_hash_deterministic():
    ids = torch.arange(16).reshape(1, 16)
    assert slice_hash(ids) == slice_hash(ids.clone())


def test_slice_hash_differs_on_content():
    a = torch.arange(16).reshape(1, 16)
    b = a.clone()
    b[0, 5] += 1
    assert slice_hash(a) != slice_hash(b)


def test_slice_hash_differs_on_shape():
    a = torch.arange(16).reshape(1, 16)
    b = torch.arange(16).reshape(16, 1)
    assert slice_hash(a) != slice_hash(b)


def test_slice_hash_is_hex_sha256():
    h = slice_hash(torch.arange(4).reshape(1, 4))
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ---------- metric_battery ----------

# Use a small input so parallel_prediction_agreement stays fast on CPU.
TEST_SEQ_LEN = 12
TEST_K = 3


@pytest.fixture
def tiny_input_ids(pythia_model_and_tokenizer):
    _, tok = pythia_model_and_tokenizer
    text = "the quick brown fox jumps over the lazy dog and continues running"
    ids = tok(text, return_tensors="pt").input_ids[:, :TEST_SEQ_LEN]
    assert ids.shape == (1, TEST_SEQ_LEN)
    return ids


def _run(model, input_ids, tmp_path, *, force=False):
    return metric_battery(
        model_size="160m",
        step=143000,
        dataset_name="testset",
        input_ids=input_ids,
        k_parallel=TEST_K,
        force_recompute=force,
        cache_dir=tmp_path,
        model=model,
        tokenizer_name="EleutherAI/pythia-160m",
    )


def test_battery_creates_cache_files(pythia_model_and_tokenizer, tiny_input_ids, tmp_path):
    model, _ = pythia_model_and_tokenizer
    out = _run(model, tiny_input_ids, tmp_path)

    # Cache files exist where expected.
    cached = list(tmp_path.rglob("*.npz"))
    manifests = list(tmp_path.rglob("*.manifest.json"))
    assert len(cached) == 1
    assert len(manifests) == 1
    assert cached[0].parent == manifests[0].parent
    assert cached[0].parent.parts[-2:] == ("testset", "160m")
    assert cached[0].name.startswith("step143000_")

    # Return dict has the expected keys and shapes.
    assert set(out.keys()) >= {
        "next_token_entropy",
        "top1_probability",
        "attention_entropy_per_head",
        "attention_concentration_per_head",
        "parallel_agreement",
        "input_ids",
    }
    assert out["next_token_entropy"].shape == (1, TEST_SEQ_LEN)
    assert out["top1_probability"].shape == (1, TEST_SEQ_LEN)
    L, H = out["attention_entropy_per_head"].shape[0], out["attention_entropy_per_head"].shape[2]
    assert out["attention_entropy_per_head"].shape == (L, 1, H, TEST_SEQ_LEN)
    assert out["attention_concentration_per_head"].shape == (3, L, 1, H, TEST_SEQ_LEN)
    assert out["parallel_agreement"].shape == (1, TEST_SEQ_LEN - TEST_K, TEST_K)
    assert out["parallel_agreement"].dtype == torch.bool


def test_cache_hit_matches_miss(pythia_model_and_tokenizer, tiny_input_ids, tmp_path):
    model, _ = pythia_model_and_tokenizer
    first = _run(model, tiny_input_ids, tmp_path)
    # Second call with the same inputs but model=None — must NOT need to load
    # the model, since the cache is already populated. We deliberately pass
    # model=None to prove the cache hit short-circuits before load_pythia.
    second = metric_battery(
        model_size="160m",
        step=143000,
        dataset_name="testset",
        input_ids=tiny_input_ids,
        k_parallel=TEST_K,
        cache_dir=tmp_path,
        model=None,
    )
    for key in first:
        assert torch.equal(first[key], second[key]), f"mismatch on {key}"


def test_force_recompute_overwrites(pythia_model_and_tokenizer, tiny_input_ids, tmp_path):
    model, _ = pythia_model_and_tokenizer
    first = _run(model, tiny_input_ids, tmp_path)
    cached_path = next(tmp_path.rglob("*.npz"))
    original_mtime = cached_path.stat().st_mtime_ns

    # Bump mtime artificially backwards so we can detect a rewrite.
    import os
    import time
    time.sleep(0.01)
    os.utime(cached_path, ns=(original_mtime - 1_000_000_000, original_mtime - 1_000_000_000))

    second = _run(model, tiny_input_ids, tmp_path, force=True)
    new_mtime = cached_path.stat().st_mtime_ns
    assert new_mtime > original_mtime - 1_000_000_000, "force_recompute did not rewrite the file"
    # Numerical content is unchanged (same model, same inputs, deterministic).
    for key in first:
        assert torch.equal(first[key], second[key])


def test_creates_missing_cache_directory(pythia_model_and_tokenizer, tiny_input_ids, tmp_path):
    nested = tmp_path / "does" / "not" / "exist" / "yet"
    assert not nested.exists()
    model, _ = pythia_model_and_tokenizer
    _run(model, tiny_input_ids, nested)
    assert nested.exists()
    assert any(nested.rglob("*.npz"))


def test_manifest_records_provenance(pythia_model_and_tokenizer, tiny_input_ids, tmp_path):
    model, _ = pythia_model_and_tokenizer
    _run(model, tiny_input_ids, tmp_path)
    manifest_path = next(tmp_path.rglob("*.manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["model_name"] == "EleutherAI/pythia-160m"
    assert manifest["model_revision"] == "step143000"
    assert manifest["step"] == 143000
    assert manifest["model_size"] == "160m"
    assert manifest["dataset"] == "testset"
    assert manifest["k_parallel"] == TEST_K
    assert manifest["tokenizer_name"] == "EleutherAI/pythia-160m"
    assert manifest["slice_sha256"] == slice_hash(tiny_input_ids)
    assert manifest["n_tokens"] == TEST_SEQ_LEN
    assert manifest["shape"] == [1, TEST_SEQ_LEN]
    assert "next_token_entropy" in manifest["tensor_keys"]


def test_different_slices_do_not_collide(pythia_model_and_tokenizer, tiny_input_ids, tmp_path):
    model, _ = pythia_model_and_tokenizer
    _run(model, tiny_input_ids, tmp_path)
    other = tiny_input_ids.clone()
    other[0, 0] = (other[0, 0] + 1) % 100  # different content, same shape
    _run(model, other, tmp_path)
    cached = list(tmp_path.rglob("*.npz"))
    assert len(cached) == 2, "two distinct slices should produce two cache files"
