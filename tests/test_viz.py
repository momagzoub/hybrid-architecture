"""Smoke tests for the visualization helpers.

These don't validate visual fidelity — they verify the functions accept
the expected inputs, return a `matplotlib.figure.Figure`, and handle
common shape errors. Visual sanity is left to the demo notebook.
"""

from __future__ import annotations

import matplotlib

# Use a non-interactive backend so tests run headless without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from hybrid_arch.viz import attention_track, entropy_heatmap  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any figures created by a test to avoid 'too many figures' warnings."""
    yield
    plt.close("all")


# ---------- entropy_heatmap ----------

def test_entropy_heatmap_returns_figure_from_tensor():
    tokens = ["The", " quick", " brown", " fox"]
    entropies = torch.tensor([0.1, 1.4, 0.9, 2.3])
    fig = entropy_heatmap(tokens, entropies)
    assert isinstance(fig, Figure)


def test_entropy_heatmap_returns_figure_from_list():
    tokens = ["a", "b", "c"]
    entropies = [0.0, 0.5, 1.0]
    fig = entropy_heatmap(tokens, entropies)
    assert isinstance(fig, Figure)


def test_entropy_heatmap_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        entropy_heatmap(["a", "b"], torch.tensor([0.0, 0.5, 1.0]))


def test_entropy_heatmap_rejects_non_1d():
    with pytest.raises(ValueError, match="1D"):
        entropy_heatmap(["a", "b"], torch.zeros(2, 3))


def test_entropy_heatmap_accepts_ax():
    fig, ax = plt.subplots()
    out = entropy_heatmap(["a", "b", "c"], [0.1, 0.5, 0.9], ax=ax)
    assert out is fig


# ---------- attention_track ----------

def test_attention_track_returns_figure_from_tensor():
    tokens = ["a", "b", "c", "d"]
    metric = torch.tensor([0.5, 0.7, 0.4, 0.9])
    fig = attention_track(tokens, metric)
    assert isinstance(fig, Figure)


def test_attention_track_returns_figure_from_list():
    fig = attention_track(["a", "b"], [0.1, 0.9])
    assert isinstance(fig, Figure)


def test_attention_track_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        attention_track(["a", "b"], [0.1, 0.5, 0.9])


def test_attention_track_accepts_ax():
    fig, ax = plt.subplots()
    out = attention_track(["a", "b", "c"], [0.1, 0.5, 0.9], ax=ax)
    assert out is fig


def test_attention_track_with_custom_labels():
    fig = attention_track(
        ["x", "y", "z"],
        [0.1, 0.2, 0.3],
        title="custom title",
        ylabel="custom ylabel",
    )
    assert isinstance(fig, Figure)
