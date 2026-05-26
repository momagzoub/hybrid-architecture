"""Phase 3 Step 3 — probe accuracy vs layer depth.

For each `(model_size, layer_index)` on Pythia at step 143000, train a
`LayerProbe` on the layer's hidden states and report 5-fold CV ROC-AUC
against the parallel-safety label from Step 2's cache.

Final-checkpoint only. Loads the cache; if `hidden_states_per_layer` is
missing, recomputes that one cell with `force_recompute=True` and the
hidden-state side-channel turned on.

Writes:
    docs/results/07_probe_layer_depth.csv
    docs/results/07_probe_layer_depth.manifest.json
    docs/results/figures/07_probe_layer_depth.png

    src/hybrid_arch/probe_checkpoints/<size>_step143000_L<idx>.pt
    src/hybrid_arch/probe_checkpoints/<size>_step143000_L<idx>.pt.json
    (committed: the trainable artifact Phase 4 calls into.)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from hybrid_arch import (  # noqa: E402
    cross_val_auroc,
    load_pythia,
    metric_battery,
    save_probe,
    slice_hash,
    train_probe,
)

SIZES = ("70m", "160m", "410m")
STEP = 143000
DATASET = "wikitext"
N_TOKENS = 256
K_PARALLEL = 4
THRESHOLD = 0.9
N_FOLDS = 5
SLICE_PATH = _REPO_ROOT / "data" / "dataset_slices" / f"{DATASET}_slice_{N_TOKENS}.pt"
CACHE_DIR = _REPO_ROOT / "data" / "cache"
RESULTS_DIR = _REPO_ROOT / "docs" / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
PROBE_DIR = _REPO_ROOT / "src" / "hybrid_arch" / "probe_checkpoints"

CSV_PATH = RESULTS_DIR / "07_probe_layer_depth.csv"
MANIFEST_PATH = RESULTS_DIR / "07_probe_layer_depth.manifest.json"
FIG_PATH = FIGURE_DIR / "07_probe_layer_depth.png"


def parallel_safe_labels(agreement: torch.Tensor) -> np.ndarray:
    """Mean over j=1..k-1, threshold at THRESHOLD. Returns bool[n_positions]."""
    pp = agreement[0, :, 1:].float().mean(dim=-1).numpy()
    return (pp >= THRESHOLD).astype(np.int64)


def ensure_cell_with_hidden(size: str, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return a metric_battery output that *includes* hidden_states_per_layer.

    The Phase 2 sweep ran before hidden states were part of the cache, so
    existing files for step 143000 may not have them. Probe `hidden_states_per_layer`
    and force-recompute if missing.
    """
    out = metric_battery(size, STEP, DATASET, input_ids,
                         k_parallel=K_PARALLEL, cache_dir=CACHE_DIR)
    if "hidden_states_per_layer" in out:
        return out
    print(f"  [recompute] {size} step{STEP}: cache lacks hidden states; reloading model...")
    model, _ = load_pythia(size, STEP)
    out = metric_battery(
        size, STEP, DATASET, input_ids,
        k_parallel=K_PARALLEL, cache_dir=CACHE_DIR,
        model=model, force_recompute=True,
        tokenizer_name=f"EleutherAI/pythia-{size}",
    )
    del model
    return out


def align_features(hs_layer: torch.Tensor, y: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """Drop position 0 and trim to the n_positions of the agreement labels.

    `hs_layer` is `[B, S, H]`; `y` is `[n_positions]` with n_positions = S - k.
    We use positions 1..n_positions-1 from both, matching the Step 5 protocol.
    """
    n_positions = y.size
    X = hs_layer[0, 1:n_positions].float()         # [n_positions - 1, H]
    return X, y[1:]


def run_size(size: str, input_ids: torch.Tensor) -> list[dict]:
    print(f"\n=== Pythia-{size} @ step{STEP} ===")
    out = ensure_cell_with_hidden(size, input_ids)
    hs = out["hidden_states_per_layer"]            # [L, B, S, H], fp16
    L, _, _, H = hs.shape
    y = parallel_safe_labels(out["parallel_agreement"])
    print(f"  layers={L}  hidden_dim={H}  n_pos_labels={int(y.sum())}/{y.size}")

    rows: list[dict] = []
    for layer_idx in range(L):
        X, y_aligned = align_features(hs[layer_idx], y)
        # cross-val on the per-layer hidden states.
        mean, std = cross_val_auroc(
            X, torch.from_numpy(y_aligned),
            n_folds=N_FOLDS, mlp_dim=32, n_epochs=200, lr=1e-2,
            weight_decay=1e-3, patience=20, seed=0,
        )
        # also fit a single probe on the full data and save it.
        probe, result = train_probe(
            X, torch.from_numpy(y_aligned),
            mlp_dim=32, n_epochs=200, lr=1e-2,
            weight_decay=1e-3, patience=20, seed=0,
        )
        ckpt_path = PROBE_DIR / f"{size}_step{STEP}_L{layer_idx}.pt"
        save_probe(probe, ckpt_path, metadata={
            "model_size": size,
            "step": STEP,
            "layer": layer_idx,
            "dataset": DATASET,
            "n_train": result.n_train,
            "n_val": result.n_val,
            "val_auroc": result.val_auroc,
            "cv_auroc_mean": mean,
            "cv_auroc_std": std,
        })
        print(f"  L{layer_idx:>2}  cv_auc={mean:.3f}±{std:.3f}  val_auc={result.val_auroc:.3f}  "
              f"saved {ckpt_path.name}")
        rows.append({
            "size": size, "step": STEP, "layer": layer_idx,
            "hidden_dim": H,
            "cv_auroc_mean": mean, "cv_auroc_std": std,
            "val_auroc": result.val_auroc,
            "n_positive": result.n_positive_train + result.n_positive_val,
            "n_total": result.n_train + result.n_val,
        })
    return rows


def write_outputs(all_rows: list[dict], input_ids: torch.Tensor) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    MANIFEST_PATH.write_text(json.dumps({
        "experiment": "Phase 3 Step 3 — probe AUROC vs layer depth",
        "sizes": list(SIZES),
        "step": STEP,
        "dataset": DATASET,
        "n_tokens": N_TOKENS,
        "k_parallel": K_PARALLEL,
        "threshold": THRESHOLD,
        "n_folds": N_FOLDS,
        "slice_sha256": slice_hash(input_ids),
        "probe_arch": "LayerProbe(mlp_dim=32)",
        "trainer": "Adam lr=1e-2 wd=1e-3 patience=20 max_epochs=200",
        "csv": str(CSV_PATH.relative_to(_REPO_ROOT)),
        "figure": str(FIG_PATH.relative_to(_REPO_ROOT)),
        "probe_checkpoint_dir": str(PROBE_DIR.relative_to(_REPO_ROOT)),
    }, indent=2, sort_keys=True))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    for size in SIZES:
        sub = [r for r in all_rows if r["size"] == size]
        sub.sort(key=lambda r: r["layer"])
        # Layer fraction lets us overlay sizes with different L on the same axis.
        max_layer = max(r["layer"] for r in sub)
        xs = [r["layer"] / max_layer if max_layer > 0 else 0 for r in sub]
        ys = [r["cv_auroc_mean"] for r in sub]
        es = [r["cv_auroc_std"] for r in sub]
        ax.errorbar(xs, ys, yerr=es, marker="o", label=f"Pythia-{size}", capsize=3)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, label="chance")
    ax.set_xlabel("Layer fraction (0 = embedding, 1 = final layer)")
    ax.set_ylabel(f"5-fold CV ROC-AUC\n(target: parallel-safety at ≥{THRESHOLD})")
    ax.set_title("MLP probe accuracy vs layer depth (Pythia step143000, WikiText)")
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_PATH)
    plt.close(fig)


def main() -> None:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    input_ids = torch.load(SLICE_PATH, weights_only=False)
    print(f"Slice sha256: {slice_hash(input_ids)}")
    all_rows: list[dict] = []
    for size in SIZES:
        all_rows.extend(run_size(size, input_ids))
    write_outputs(all_rows, input_ids)
    print(
        f"\nWrote {CSV_PATH}\nWrote {MANIFEST_PATH}\nWrote {FIG_PATH}\n"
        f"Probe checkpoints in {PROBE_DIR}"
    )


if __name__ == "__main__":
    main()
