"""Plotting helpers for the metric library.

One global `STYLE` dict, set once and shared across functions, so every
plot in the project has a consistent look. Per AGENTS.md §4: "settle on
a color palette now and never change it."

Public API:
    entropy_heatmap(tokens, entropies, ax=None) -> Figure
    attention_track(tokens, metric, ax=None)    -> Figure
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# One central place for visual choices. Don't touch matplotlib defaults
# elsewhere — call into this dict instead.
STYLE = {
    "sequential_cmap": "viridis",      # for entropy heatmaps, attention masses
    "diverging_cmap": "coolwarm",      # for calibration / diff plots
    "track_color": "#1f77b4",
    "track_linewidth": 1.5,
    "marker_size": 4,
    "token_fontsize": 8,
    "title_fontsize": 11,
    "label_fontsize": 9,
    "row_height": 0.6,                 # inches per row in heatmap strip
    "column_width": 0.35,              # inches per token column
    "dpi": 110,
}


def _to_numpy_1d(x) -> np.ndarray:
    """Accept Tensor / list / ndarray, return a 1D float64 numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().to(torch.float32).numpy()
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected 1D array, got shape {arr.shape}")
    return arr


def entropy_heatmap(
    tokens: Sequence[str],
    entropies,
    ax: Axes | None = None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Figure:
    """Color-coded strip showing per-token entropy with token labels.

    Each token gets one colored cell; cell color = entropy value, viridis
    colormap. The token text is drawn on top. Useful for spotting which
    positions in a prompt the model finds hard.

    Args:
        tokens: list of token strings (length S).
        entropies: 1D tensor/array of entropy values (length S), in nats.
        ax: optional matplotlib Axes to draw on. If None, a new figure is
            created with a width proportional to the number of tokens.
        title: optional title.
        vmin, vmax: color scale bounds. If None, derived from `entropies`.

    Returns:
        The matplotlib Figure containing the heatmap.
    """
    entropies = _to_numpy_1d(entropies)
    if len(tokens) != len(entropies):
        raise ValueError(
            f"tokens and entropies must have same length; "
            f"got {len(tokens)} and {len(entropies)}"
        )

    n = len(tokens)
    if ax is None:
        width = max(4.0, n * STYLE["column_width"])
        fig, ax = plt.subplots(figsize=(width, STYLE["row_height"] * 2.2), dpi=STYLE["dpi"])
    else:
        fig = ax.figure  # type: ignore[assignment]

    # One-row image: shape (1, n).
    data = entropies.reshape(1, n)
    im = ax.imshow(
        data,
        aspect="auto",
        cmap=STYLE["sequential_cmap"],
        vmin=vmin,
        vmax=vmax,
    )

    # Token labels under each column.
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=STYLE["token_fontsize"])
    ax.set_yticks([])
    ax.set_xlabel("position", fontsize=STYLE["label_fontsize"])
    if title:
        ax.set_title(title, fontsize=STYLE["title_fontsize"])

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="entropy (nats)")
    fig.tight_layout()
    return fig


def attention_track(
    tokens: Sequence[str],
    metric,
    ax: Axes | None = None,
    title: str | None = None,
    ylabel: str = "metric",
) -> Figure:
    """Line plot of a per-token metric (e.g. attention entropy per query
    position) against position index, with tokens labelled along the x-axis.

    Args:
        tokens: list of token strings (length S).
        metric: 1D tensor/array of values (length S).
        ax: optional matplotlib Axes.
        title: optional title.
        ylabel: label for the y-axis.

    Returns:
        The matplotlib Figure.
    """
    metric = _to_numpy_1d(metric)
    if len(tokens) != len(metric):
        raise ValueError(
            f"tokens and metric must have same length; "
            f"got {len(tokens)} and {len(metric)}"
        )

    n = len(tokens)
    if ax is None:
        width = max(4.0, n * STYLE["column_width"])
        fig, ax = plt.subplots(figsize=(width, 3.0), dpi=STYLE["dpi"])
    else:
        fig = ax.figure  # type: ignore[assignment]

    ax.plot(
        np.arange(n),
        metric,
        color=STYLE["track_color"],
        linewidth=STYLE["track_linewidth"],
        marker="o",
        markersize=STYLE["marker_size"],
    )
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=STYLE["token_fontsize"])
    ax.set_xlabel("position", fontsize=STYLE["label_fontsize"])
    ax.set_ylabel(ylabel, fontsize=STYLE["label_fontsize"])
    if title:
        ax.set_title(title, fontsize=STYLE["title_fontsize"])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
