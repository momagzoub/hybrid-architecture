"""Phase 2 Step 5 — the signature analysis.

Tests the Phase 1 puzzle directly: aggregate attention metrics correlated
at |r| < 0.11 with parallel-safety. Hypothesis: specific (layer, head)
pairs are highly predictive but their mean is noise.

For each (model_size, checkpoint) cell from Step 4's cache, build a
per-position feature matrix containing:

    - 2 logit-side metrics:    next_token_entropy, top1_probability
    - L × H attention entropies
    - L × H × 3 attention top-k concentrations (top_1, top_3, top_5)

…and a binary "parallel-safe" label per position (`mean_j>=1(agreement) >= 0.9`).

Train an L2-regularized `LogisticRegression(class_weight="balanced")` with
5-fold stratified CV, score by mean ROC-AUC. Skip cells with degenerate
labels (n_positive < 5 or n_positive > n - 5).

For each model size, fit one *final-checkpoint* classifier on full data
and extract the top-10 attention features by absolute coefficient
(features standardised first so the magnitudes are comparable).

Writes:
    docs/results/03_signature_auroc.csv
    docs/results/04_top_features.csv
    docs/results/02_signature_analysis.manifest.json
    docs/results/figures/03_signature_auroc.png
    docs/results/figures/04_top_features.png
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from hybrid_arch import (  # noqa: E402
    list_checkpoints,
    metric_battery,
    slice_hash,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


SIZES = ("70m", "160m", "410m")
DATASET = "wikitext"
N_TOKENS = 256
K_PARALLEL = 4
THRESHOLD = 0.9
N_FOLDS = 5
TOP_FEATURES = 10
SLICE_PATH = _REPO_ROOT / "data" / "dataset_slices" / f"{DATASET}_slice_{N_TOKENS}.pt"
CACHE_DIR = _REPO_ROOT / "data" / "cache"
RESULTS_DIR = _REPO_ROOT / "docs" / "results"
FIGURE_DIR = RESULTS_DIR / "figures"

AUROC_CSV = RESULTS_DIR / "03_signature_auroc.csv"
TOPF_CSV = RESULTS_DIR / "04_top_features.csv"
MANIFEST = RESULTS_DIR / "02_signature_analysis.manifest.json"
AUROC_FIG = FIGURE_DIR / "03_signature_auroc.png"
TOPF_FIG = FIGURE_DIR / "04_top_features.png"


def build_features(out: dict[str, torch.Tensor]) -> tuple[np.ndarray, list[str]]:
    """Assemble the per-position feature matrix from a `metric_battery` output.

    Drops position 0 (structurally zero attention entropy under causal mask).
    Returns (X, feature_names) with X.shape = [n_positions - 1, n_features].
    """
    nte = out["next_token_entropy"][0].numpy()                # [S]
    top1 = out["top1_probability"][0].numpy()                 # [S]
    attn_ent = out["attention_entropy_per_head"][:, 0].numpy()  # [L, H, S]
    attn_conc = out["attention_concentration_per_head"][:, :, 0].numpy()  # [K, L, H, S]

    L, H = attn_ent.shape[:2]
    K = attn_conc.shape[0]

    # Stack along the feature axis at every position, then drop t=0.
    rows: list[np.ndarray] = []
    names: list[str] = []

    rows.append(nte[None])
    names.append("next_token_entropy")
    rows.append(top1[None])
    names.append("top1_probability")

    for li in range(L):
        for hi in range(H):
            rows.append(attn_ent[li, hi][None])
            names.append(f"attn_ent_L{li}_H{hi}")
    top_k_vals = (1, 3, 5)
    for ki in range(K):
        for li in range(L):
            for hi in range(H):
                rows.append(attn_conc[ki, li, hi][None])
                names.append(f"attn_conc_top{top_k_vals[ki]}_L{li}_H{hi}")

    X = np.concatenate(rows, axis=0).T   # [S, n_features]
    # Align with parallel_agreement: it has n_positions = S - k rows starting
    # at position 0. Caller is responsible for slicing to the agreement range.
    return X.astype(np.float32), names


def parallel_safe_labels(agreement: torch.Tensor) -> np.ndarray:
    """Binary parallel-safe label per position (mean over j=1..k-1)."""
    per_position = agreement[0, :, 1:].float().mean(dim=-1).numpy()  # [n_positions]
    return (per_position >= THRESHOLD).astype(np.int32)


def align_X_y(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Take features at positions 1..n_positions-1, labels at same range.

    Drops position 0 (structurally-zero attention entropy under causal mask)
    from both. X has S rows; y has n_positions = S - k rows. Align both at
    positions 1..n_positions-1.
    """
    n_positions = y.size
    return X[1:n_positions], y[1:]


def cell_auroc(X: np.ndarray, y: np.ndarray) -> tuple[float, float, int, int]:
    """5-fold stratified CV ROC-AUC. Returns (mean, std, n_pos, n_total)."""
    n_pos = int(y.sum())
    n_total = int(y.size)
    if n_pos < N_FOLDS or n_pos > n_total - N_FOLDS:
        return float("nan"), float("nan"), n_pos, n_total

    aurocs: list[float] = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        Xs_tr = scaler.transform(X[tr])
        Xs_te = scaler.transform(X[te])
        clf = LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=1000, solver="liblinear"
        )
        clf.fit(Xs_tr, y[tr])
        scores = clf.decision_function(Xs_te)
        # Manual ROC-AUC to avoid an extra sklearn import path.
        from sklearn.metrics import roc_auc_score
        aurocs.append(roc_auc_score(y[te], scores))
    return float(np.mean(aurocs)), float(np.std(aurocs)), n_pos, n_total


def top_features(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[tuple[str, float]]:
    """Train one classifier on the full data, return top-k by |coef|."""
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(
        class_weight="balanced", C=1.0, max_iter=2000, solver="liblinear"
    )
    clf.fit(scaler.transform(X), y)
    coefs = clf.coef_[0]
    order = np.argsort(-np.abs(coefs))[:TOP_FEATURES]
    return [(names[i], float(coefs[i])) for i in order]


def load_cell(size: str, step: int, input_ids: torch.Tensor) -> dict[str, torch.Tensor] | None:
    """Cache-only load — never recompute. Returns None if not cached."""
    from hybrid_arch.cache import _cache_paths
    slice12 = slice_hash(input_ids)[:12]
    npz_path, manifest_path = _cache_paths(CACHE_DIR, DATASET, size, step, slice12)
    if not (npz_path.exists() and manifest_path.exists()):
        return None
    return metric_battery(size, step, DATASET, input_ids,
                          k_parallel=K_PARALLEL, cache_dir=CACHE_DIR)


def run() -> None:
    input_ids = torch.load(SLICE_PATH, weights_only=False)
    print(f"Slice sha256: {slice_hash(input_ids)}\n")

    auroc_rows: list[dict] = []
    topf_rows: list[dict] = []

    final_step = max(list_checkpoints())
    final_features: dict[str, list[tuple[str, float]]] = {}

    for size in SIZES:
        for step in list_checkpoints():
            out = load_cell(size, step, input_ids)
            if out is None:
                print(f"[skip] size={size} step={step}: no cache")
                continue
            X_full, names = build_features(out)
            y_full = parallel_safe_labels(out["parallel_agreement"])
            X, y = align_X_y(X_full, y_full)
            auc_m, auc_s, n_pos, n_total = cell_auroc(X, y)
            print(f"size={size:5s} step={step:>6d}: n_pos={n_pos:3d}/{n_total} "
                  f"auroc={auc_m:.3f}±{auc_s:.3f}")
            auroc_rows.append({
                "size": size, "step": step,
                "auroc_mean": auc_m, "auroc_std": auc_s,
                "n_positive": n_pos, "n_total": n_total,
            })

            if step == final_step:
                top = top_features(X, y, names)
                final_features[size] = top
                for rank, (fname, coef) in enumerate(top, 1):
                    topf_rows.append({
                        "size": size, "rank": rank,
                        "feature": fname, "coef": coef,
                    })

    # ----- write CSVs -----
    AUROC_CSV.parent.mkdir(parents=True, exist_ok=True)
    with AUROC_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(auroc_rows[0].keys()))
        w.writeheader()
        w.writerows(auroc_rows)
    with TOPF_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(topf_rows[0].keys()))
        w.writeheader()
        w.writerows(topf_rows)

    MANIFEST.write_text(json.dumps({
        "experiment": "Phase 2 Step 5 — signature analysis",
        "sizes": list(SIZES),
        "dataset": DATASET,
        "n_tokens": N_TOKENS,
        "k_parallel": K_PARALLEL,
        "threshold": THRESHOLD,
        "n_folds": N_FOLDS,
        "feature_groups": [
            "next_token_entropy", "top1_probability",
            "attn_ent_L{l}_H{h}", "attn_conc_top{1,3,5}_L{l}_H{h}",
        ],
        "model": "LogisticRegression(penalty=l2, C=1, class_weight=balanced, liblinear)",
        "feature_standardisation": "per-fold StandardScaler",
        "csv_auroc": str(AUROC_CSV.relative_to(_REPO_ROOT)),
        "csv_top_features": str(TOPF_CSV.relative_to(_REPO_ROOT)),
        "figure_auroc": str(AUROC_FIG.relative_to(_REPO_ROOT)),
        "figure_top_features": str(TOPF_FIG.relative_to(_REPO_ROOT)),
        "slice_sha256": slice_hash(input_ids),
    }, indent=2, sort_keys=True))

    # ----- plots -----
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # AUROC curves: one line per size.
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    for size in SIZES:
        sub = [r for r in auroc_rows if r["size"] == size and not np.isnan(r["auroc_mean"])]
        sub.sort(key=lambda r: r["step"])
        xs = [max(r["step"], 0.5) for r in sub]
        ys = [r["auroc_mean"] for r in sub]
        es = [r["auroc_std"] for r in sub]
        ax.errorbar(xs, ys, yerr=es, marker="o", label=f"Pythia-{size}", capsize=3)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("Training step (log scale; 0 → 0.5 for display)")
    ax.set_ylabel(f"5-fold CV ROC-AUC\n(label: per-position parallel-safe at ≥{THRESHOLD})")
    ax.set_title("Predicting parallel-safety from per-(layer, head) attention + logit features")
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(AUROC_FIG)
    plt.close(fig)

    # Top features bar chart: one panel per size, top 10 by |coef|.
    fig, axes = plt.subplots(1, len(SIZES), figsize=(15, 4.5), dpi=120, sharex=False)
    for ax, size in zip(axes, SIZES):
        feats = final_features.get(size, [])
        if not feats:
            ax.set_title(f"Pythia-{size} — no data")
            continue
        ys = list(range(len(feats)))[::-1]
        coefs = [c for _, c in feats]
        labels = [n for n, _ in feats]
        colors = ["#1f77b4" if c >= 0 else "#d62728" for c in coefs]
        ax.barh(ys, coefs, color=colors)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title(f"Pythia-{size} @ step{final_step}")
        ax.set_xlabel("standardised coef")
    fig.suptitle("Top-10 most parallel-safety-predictive features (final checkpoint)")
    fig.tight_layout()
    fig.savefig(TOPF_FIG)
    plt.close(fig)

    print(f"\nWrote {AUROC_CSV}\nWrote {TOPF_CSV}\nWrote {MANIFEST}")
    print(f"Wrote {AUROC_FIG}\nWrote {TOPF_FIG}")


if __name__ == "__main__":
    run()
