"""Phase 2 Step 4 — the emergence curve.

For each (Pythia size, training step) in the canonical Phase 2 grid, run the
metric battery on a fixed 256-token WikiText slice and record the fraction
of positions where the next-`k` autoregressive predictions agree with the
teacher-forced ones at >= 90% rate. That fraction is our headline "parallel
safety" metric; plotting it vs. training step is the emergence curve.

Writes:
    docs/results/02_emergence_curve.csv
    docs/results/02_emergence_curve.manifest.json
    docs/results/figures/02_emergence_curve.png

All per-(size, step) metric batteries flow through `hybrid_arch.cache`, so
re-running the script after a crash skips finished cells in ms.

Conventions:
- Drop the j=0 column when computing the per-position agreement rate — it is
  structurally True (teacher-forced and AR start from identical context).
- Drop position 0 from any per-token attention-side aggregation (its
  attention row is a delta by causal-mask construction, so its entropy is
  0 mechanically, not by model behavior). The agreement fraction does not
  use position 0 either, because n_positions counts from t=0 but the metric
  itself is well-defined there — we keep it for the parallel-safety fraction.

Usage::

    python src/scripts/phase2_emergence_curve.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from hybrid_arch import (  # noqa: E402
    list_checkpoints,
    load_pythia,
    metric_battery,
    slice_hash,
)
from hybrid_arch.viz import STYLE  # noqa: E402

SIZES = ("70m", "160m", "410m")
DATASET = "wikitext"
N_TOKENS = 256
K_PARALLEL = 4
PARALLEL_SAFETY_THRESHOLD = 0.9
SLICE_PATH = _REPO_ROOT / "data" / "dataset_slices" / f"{DATASET}_slice_{N_TOKENS}.pt"
CACHE_DIR = _REPO_ROOT / "data" / "cache"
RESULTS_DIR = _REPO_ROOT / "docs" / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
CSV_PATH = RESULTS_DIR / "02_emergence_curve.csv"
MANIFEST_PATH = RESULTS_DIR / "02_emergence_curve.manifest.json"
FIGURE_PATH = FIGURE_DIR / "02_emergence_curve.png"


def parallel_safety_fraction(agreement: torch.Tensor) -> float:
    """Fraction of positions whose mean agreement over j=1..k-1 is >= threshold.

    `agreement` is `[B, n_positions, k]` bool. j=0 is structural — drop it.
    For each position, mean over j=1..k-1; count fraction >= threshold.
    """
    if agreement.shape[-1] < 2:
        raise ValueError("k must be >= 2 to compute a non-trivial safety fraction")
    per_position = agreement[..., 1:].float().mean(dim=-1)   # [B, n_positions]
    return (per_position >= PARALLEL_SAFETY_THRESHOLD).float().mean().item()


def run_grid(input_ids: torch.Tensor) -> list[dict]:
    rows: list[dict] = []
    steps = list_checkpoints()
    total = len(SIZES) * len(steps)
    done = 0
    for size in SIZES:
        for step in steps:
            done += 1
            cache_dir_cell = CACHE_DIR
            print(f"[{done}/{total}] size={size} step={step}: ", end="", flush=True)

            # Cache hit short-circuits before load_pythia, so check first.
            from hybrid_arch.cache import _cache_paths
            slice12 = slice_hash(input_ids)[:12]
            npz_path, manifest_path = _cache_paths(
                cache_dir_cell, DATASET, size, step, slice12
            )
            t0 = time.time()
            if npz_path.exists() and manifest_path.exists():
                out = metric_battery(
                    size, step, DATASET, input_ids,
                    k_parallel=K_PARALLEL, cache_dir=cache_dir_cell,
                )
                tag = "HIT"
                model = None
            else:
                model, _tok = load_pythia(size, step)
                out = metric_battery(
                    size, step, DATASET, input_ids,
                    k_parallel=K_PARALLEL, cache_dir=cache_dir_cell,
                    model=model, tokenizer_name=f"EleutherAI/pythia-{size}",
                )
                tag = "MISS"
            dt = time.time() - t0
            psf = parallel_safety_fraction(out["parallel_agreement"])
            mean_agree = out["parallel_agreement"][..., 1:].float().mean().item()
            print(f"{tag} {dt:5.1f}s  psf={psf:.3f}  mean_agree={mean_agree:.3f}")
            rows.append(
                {
                    "size": size,
                    "step": step,
                    "parallel_safety_fraction": psf,
                    "mean_agreement_j1_to_k": mean_agree,
                    "n_positions": int(out["parallel_agreement"].shape[1]),
                }
            )
            if model is not None:
                del model  # free RAM before loading the next checkpoint
    return rows


def write_csv(rows: list[dict]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(input_ids: torch.Tensor) -> None:
    manifest = {
        "experiment": "Phase 2 Step 4 — emergence curve",
        "sizes": list(SIZES),
        "steps": list_checkpoints(),
        "dataset": DATASET,
        "n_tokens": N_TOKENS,
        "k_parallel": K_PARALLEL,
        "parallel_safety_threshold": PARALLEL_SAFETY_THRESHOLD,
        "slice_sha256": slice_hash(input_ids),
        "slice_path": str(SLICE_PATH.relative_to(_REPO_ROOT)),
        "agreement_aggregation": (
            "fraction of positions with mean(agreement[j=1..k-1]) >= threshold"
        ),
        "csv": str(CSV_PATH.relative_to(_REPO_ROOT)),
        "figure": str(FIGURE_PATH.relative_to(_REPO_ROOT)),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def plot(rows: list[dict]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=STYLE.get("figsize_wide", (8, 5)), dpi=120)
    for size in SIZES:
        sub = [r for r in rows if r["size"] == size]
        sub.sort(key=lambda r: r["step"])
        xs = [max(r["step"], 0.5) for r in sub]  # log axis can't take 0; shift to 0.5
        ys = [r["parallel_safety_fraction"] for r in sub]
        ax.plot(xs, ys, marker="o", label=f"Pythia-{size}")
    ax.set_xscale("log")
    ax.set_xlabel("Training step (log scale; 0 → 0.5 for display)")
    ax.set_ylabel(f"Fraction of positions with\nmean agreement (j≥1) ≥ {PARALLEL_SAFETY_THRESHOLD}")
    ax.set_title(
        f"Emergence of parallel-safety across pretraining\n"
        f"({DATASET}, {N_TOKENS}-token slice, k={K_PARALLEL})"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_PATH)
    plt.close(fig)


def main() -> int:
    if not SLICE_PATH.exists():
        raise FileNotFoundError(
            f"WikiText slice missing at {SLICE_PATH}; generate it first."
        )
    input_ids = torch.load(SLICE_PATH, weights_only=False)
    if input_ids.shape != (1, N_TOKENS):
        raise ValueError(
            f"slice shape {tuple(input_ids.shape)} != expected (1, {N_TOKENS})"
        )

    print(f"Slice sha256: {slice_hash(input_ids)}")
    print(f"Grid: {len(SIZES)} sizes × {len(list_checkpoints())} steps "
          f"= {len(SIZES) * len(list_checkpoints())} cells\n")

    rows = run_grid(input_ids)
    write_csv(rows)
    write_manifest(input_ids)
    plot(rows)
    print(f"\nWrote {CSV_PATH}")
    print(f"Wrote {MANIFEST_PATH}")
    print(f"Wrote {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
