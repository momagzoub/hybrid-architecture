"""Phase 2 Step 6 — parallel-safety broken down by token type.

Categorises each token in the 256-token WikiText slice using cheap rules
(closed list for function words, regex for numbers/punctuation, alphabetic
match for content words, BPE-fragment for anything else). Reads every
cached `metric_battery` output from Step 4 and computes the per-category
parallel-safety fraction across the canonical grid.

No new compute. Reads from `data/cache/...` only.

Writes:
    docs/results/05_token_type_breakdown.csv
    docs/results/05_token_type_breakdown.manifest.json
    docs/results/figures/05_token_type_breakdown.png
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from hybrid_arch import list_checkpoints, metric_battery, slice_hash  # noqa: E402


SIZES = ("70m", "160m", "410m")
DATASET = "wikitext"
N_TOKENS = 256
K_PARALLEL = 4
THRESHOLD = 0.9
SLICE_PATH = _REPO_ROOT / "data" / "dataset_slices" / f"{DATASET}_slice_{N_TOKENS}.pt"
CACHE_DIR = _REPO_ROOT / "data" / "cache"
RESULTS_DIR = _REPO_ROOT / "docs" / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
CSV_PATH = RESULTS_DIR / "05_token_type_breakdown.csv"
MANIFEST_PATH = RESULTS_DIR / "05_token_type_breakdown.manifest.json"
FIG_PATH = FIGURE_DIR / "05_token_type_breakdown.png"


# 50 most common English function words. Source: standard lexical list,
# trimmed to closed-class items unlikely to overlap with content meanings.
FUNCTION_WORDS = frozenset({
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "with",
    "by", "from", "as", "into", "about", "against", "between", "through",
    "and", "or", "but", "nor", "so", "yet", "if", "while", "because",
    "although", "though", "since", "until", "when", "where", "whereas",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his",
    "she", "her", "hers", "it", "its", "we", "us", "our", "ours",
    "they", "them", "their", "theirs", "this", "that", "these", "those",
    "who", "whom", "whose", "which", "what", "where", "when", "why", "how",
    "not", "no", "yes",
})


CATEGORIES = ("function", "content", "number", "punct", "fragment")


_NUMBER_RE = re.compile(r"^\d[\d,.]*$")
_ALPHA_RE = re.compile(r"^[A-Za-z]+$")
_PUNCT_RE = re.compile(r"^[^\w\s]+$")


def categorize(token_str: str) -> str:
    """Classify one decoded BPE token into a coarse category.

    The Pythia tokenizer emits leading-space markers (often via `Ġ` or a
    literal leading space depending on decode); we strip whitespace before
    matching.
    """
    s = token_str.strip()
    if not s:
        return "fragment"
    if _NUMBER_RE.match(s):
        return "number"
    if _PUNCT_RE.match(s):
        return "punct"
    if _ALPHA_RE.match(s) and s.lower() in FUNCTION_WORDS:
        return "function"
    if _ALPHA_RE.match(s):
        return "content"
    return "fragment"


def per_position_parallel_safe(agreement: torch.Tensor) -> np.ndarray:
    """Bool[n_positions] — mean over j=1..k-1, threshold at THRESHOLD."""
    per_position = agreement[0, :, 1:].float().mean(dim=-1).numpy()
    return per_position >= THRESHOLD


def load_cell(size: str, step: int, input_ids: torch.Tensor) -> dict[str, torch.Tensor] | None:
    from hybrid_arch.cache import _cache_paths
    slice12 = slice_hash(input_ids)[:12]
    npz, mf = _cache_paths(CACHE_DIR, DATASET, size, step, slice12)
    if not (npz.exists() and mf.exists()):
        return None
    return metric_battery(size, step, DATASET, input_ids,
                          k_parallel=K_PARALLEL, cache_dir=CACHE_DIR)


def run() -> None:
    from transformers import AutoTokenizer

    input_ids = torch.load(SLICE_PATH, weights_only=False)
    tok = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-160m", revision="step143000"
    )

    # Categorize each token in the slice. parallel_agreement covers positions
    # 0..n_positions-1, so we only need categories for those positions.
    n_positions = input_ids.shape[1] - K_PARALLEL
    decoded = [tok.decode([int(x)]) for x in input_ids[0, :n_positions]]
    cats = [categorize(s) for s in decoded]
    cat_array = np.array(cats)
    counts = {c: int((cat_array == c).sum()) for c in CATEGORIES}
    print(f"Token-type counts (over {n_positions} positions): {counts}\n")

    rows: list[dict] = []
    for size in SIZES:
        for step in list_checkpoints():
            out = load_cell(size, step, input_ids)
            if out is None:
                continue
            safe = per_position_parallel_safe(out["parallel_agreement"])
            for cat in CATEGORIES:
                mask = cat_array == cat
                if mask.sum() == 0:
                    psf = float("nan"); n = 0
                else:
                    psf = float(safe[mask].mean())
                    n = int(mask.sum())
                rows.append({
                    "size": size, "step": step, "category": cat,
                    "parallel_safety_fraction": psf, "n_positions": n,
                })

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    MANIFEST_PATH.write_text(json.dumps({
        "experiment": "Phase 2 Step 6 — token-type breakdown",
        "categories": list(CATEGORIES),
        "category_counts": counts,
        "function_word_list_size": len(FUNCTION_WORDS),
        "dataset": DATASET,
        "n_tokens": N_TOKENS,
        "n_positions": n_positions,
        "k_parallel": K_PARALLEL,
        "threshold": THRESHOLD,
        "slice_sha256": slice_hash(input_ids),
        "csv": str(CSV_PATH.relative_to(_REPO_ROOT)),
        "figure": str(FIG_PATH.relative_to(_REPO_ROOT)),
    }, indent=2, sort_keys=True))

    # ---- plot: small multiples, one panel per category ----
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(CATEGORIES), figsize=(18, 4), dpi=120, sharey=True)
    for ax, cat in zip(axes, CATEGORIES):
        for size in SIZES:
            sub = [r for r in rows if r["size"] == size and r["category"] == cat]
            sub.sort(key=lambda r: r["step"])
            xs = [max(r["step"], 0.5) for r in sub]
            ys = [r["parallel_safety_fraction"] for r in sub]
            ax.plot(xs, ys, marker="o", label=f"Pythia-{size}")
        ax.set_xscale("log")
        ax.set_title(f"{cat}  (n={counts[cat]})")
        ax.set_xlabel("training step")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(f"parallel-safety fraction\n(mean agreement j≥1 ≥{THRESHOLD})")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Parallel-safety by token type, across pretraining")
    fig.tight_layout()
    fig.savefig(FIG_PATH)
    plt.close(fig)

    print(f"Wrote {CSV_PATH}\nWrote {MANIFEST_PATH}\nWrote {FIG_PATH}")


if __name__ == "__main__":
    run()
