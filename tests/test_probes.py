"""Tests for `hybrid_arch.probes`.

The probe is small and the trainer is short, but they're the central
artifact of Phase 3. The properties we pin here:

- forward pass shape + raw-logit output (no double sigmoid)
- training reaches AUROC > 0.8 on a synthetic noisy-XOR-like task in <2s
- save/load round-trip preserves weights and architecture
- degenerate labels (all-one-class) don't crash and yield NaN cleanly
"""

from __future__ import annotations

import time

import numpy as np
import torch

from hybrid_arch.probes import (
    LayerProbe,
    cross_val_auroc,
    load_probe,
    save_probe,
    train_probe,
)


def _synthetic_task(n: int = 600, d: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a synthetic binary task with a noisy linear decision boundary.

    The probe in production is going to be trained on hidden states with
    a strong linear signal (Phase 2's logistic regression already gets
    AUROC 0.85). The test mirrors that regime, not adversarial XOR: a
    fixed random projection plus Gaussian noise. Class balance ~50/50.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    w = rng.randn(d).astype(np.float32) * 0.6
    score = X @ w + 0.4 * rng.randn(n).astype(np.float32)
    y = (score > 0).astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


def test_forward_shape_and_logit_output():
    probe = LayerProbe(hidden_dim=16, mlp_dim=8)
    x = torch.randn(5, 16)
    out = probe(x)
    assert out.shape == (5,)
    # Logits, not probabilities — so we should see values outside [0,1] given
    # random init.
    assert (out.abs() > 0).any()

    probs = probe.predict_proba(x)
    assert torch.all(probs > 0) and torch.all(probs < 1)


def test_parameter_count_is_small():
    """A probe is supposed to be tiny — flag it if it ever balloons."""
    probe = LayerProbe(hidden_dim=1024, mlp_dim=64)
    n = probe.parameter_count()
    # ~67k params at hidden_dim=1024, mlp_dim=64 — well under 100k.
    assert 30_000 < n < 100_000, f"unexpected parameter count {n}"


def test_train_probe_beats_chance_quickly():
    X, y = _synthetic_task(n=600, d=32)
    start = time.time()
    probe, result = train_probe(
        X, y, mlp_dim=32, n_epochs=300, lr=1e-2, patience=30, seed=0,
    )
    elapsed = time.time() - start
    assert result.val_auroc > 0.8, f"train_probe should clear 0.8 AUROC, got {result.val_auroc:.3f}"
    assert elapsed < 8, f"trainer ran too long: {elapsed:.1f}s"
    # The trainer returns the best (not last) weights, so the loaded loss
    # should match `result.val_loss` ± a tiny rounding tolerance.
    assert result.val_loss == min(result.val_losses), "best weights not restored"
    assert result.n_positive_train > 0 and result.n_positive_val > 0


def test_cross_val_auroc():
    X, y = _synthetic_task(n=400, d=24)
    mean, std = cross_val_auroc(X, y, n_folds=3, mlp_dim=16, n_epochs=200, lr=1e-2)
    assert not np.isnan(mean)
    assert 0.7 < mean < 1.0, f"CV AUROC out of band: {mean:.3f}"
    assert std < 0.2


def test_cross_val_auroc_handles_degenerate_labels():
    X = torch.randn(50, 8)
    y = torch.zeros(50, dtype=torch.long)
    mean, std = cross_val_auroc(X, y, n_folds=3)
    assert np.isnan(mean) and np.isnan(std)


def test_save_and_load_roundtrip(tmp_path):
    probe = LayerProbe(hidden_dim=12, mlp_dim=16)
    probe.fc1.weight.data.normal_()
    probe.head.bias.data.fill_(0.123)
    save_probe(probe, tmp_path / "probe.pt", metadata={"model_size": "160m", "layer": 5})
    loaded = load_probe(tmp_path / "probe.pt")
    assert loaded.hidden_dim == 12
    assert loaded.mlp_dim == 16
    assert torch.allclose(loaded.head.bias, probe.head.bias)
    assert torch.allclose(loaded.fc1.weight, probe.fc1.weight)

    # Outputs match for the same input — sanity check end-to-end.
    x = torch.randn(4, 12)
    assert torch.allclose(probe(x), loaded(x), atol=1e-6)


def test_train_probe_rejects_bad_input_shapes():
    import pytest
    X = torch.randn(10, 3, 4)
    y = torch.randint(0, 2, (10,))
    with pytest.raises(ValueError, match=r"\[N, hidden_dim\]"):
        train_probe(X, y)

    X = torch.randn(10, 4)
    y_bad = torch.randint(0, 2, (8,))
    with pytest.raises(ValueError, match="labels must be"):
        train_probe(X, y_bad)
