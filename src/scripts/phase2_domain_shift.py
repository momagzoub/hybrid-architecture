"""Phase 2 Step 7 — domain shift.

Final-checkpoint only (step143000), 3 sizes × 3 datasets (wikitext from
Step 4's cache + new MBPP and GSM8K slices). Six new metric-battery runs.

Writes:
    docs/results/06_domain_shift.csv
    docs/results/06_domain_shift.manifest.json
    docs/results/figures/06_domain_shift_heatmap.png
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from hybrid_arch import load_pythia, metric_battery, slice_hash  # noqa: E402


SIZES = ("70m", "160m", "410m")
DATASETS = ("wikitext", "mbpp", "gsm8k")
STEP = 143000
N_TOKENS = 256
K_PARALLEL = 4
THRESHOLD = 0.9
SLICE_DIR = _REPO_ROOT / "data" / "dataset_slices"
CACHE_DIR = _REPO_ROOT / "data" / "cache"
RESULTS_DIR = _REPO_ROOT / "docs" / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
CSV_PATH = RESULTS_DIR / "06_domain_shift.csv"
MANIFEST_PATH = RESULTS_DIR / "06_domain_shift.manifest.json"
FIG_PATH = FIGURE_DIR / "06_domain_shift_heatmap.png"


def _stream_mbpp():
    from datasets import load_dataset
    for item in load_dataset("google-research-datasets/mbpp", split="train", streaming=True):
        yield item.get("text", "") + "\n" + item.get("code", "")


def _stream_gsm8k():
    from datasets import load_dataset
    for item in load_dataset("openai/gsm8k", "main", split="train", streaming=True):
        yield item.get("question", "") + "\n" + item.get("answer", "")


STREAMS = {"mbpp": _stream_mbpp, "gsm8k": _stream_gsm8k}


def get_slice(name: str, tok) -> torch.Tensor:
    path = SLICE_DIR / f"{name}_slice_{N_TOKENS}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    pieces, chars = [], 0
    for txt in STREAMS[name]():
        pieces.append(txt); chars += len(txt)
        if chars > N_TOKENS * 8:
            break
    ids = tok(" ".join(pieces), return_tensors="pt", truncation=False).input_ids[:, :N_TOKENS].contiguous()
    if ids.shape[1] < N_TOKENS:
        raise RuntimeError(f"{name} yielded only {ids.shape[1]} tokens")
    torch.save(ids, path)
    return ids


def psf(agreement: torch.Tensor) -> float:
    pp = agreement[0, :, 1:].float().mean(dim=-1)
    return float((pp >= THRESHOLD).float().mean())


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m", revision=f"step{STEP}")

    slices = {}
    for ds in DATASETS:
        if ds == "wikitext":
            slices[ds] = torch.load(SLICE_DIR / f"wikitext_slice_{N_TOKENS}.pt", weights_only=False)
        else:
            slices[ds] = get_slice(ds, tok)
        print(f"  {ds}: shape={tuple(slices[ds].shape)}  sha={slice_hash(slices[ds])[:12]}")

    rows = []
    for size in SIZES:
        # Load once per size, reuse across datasets.
        from hybrid_arch.cache import _cache_paths
        slice12_wt = slice_hash(slices["wikitext"])[:12]
        wt_npz, wt_mf = _cache_paths(CACHE_DIR, "wikitext", size, STEP, slice12_wt)

        model = None
        for ds in DATASETS:
            slice12 = slice_hash(slices[ds])[:12]
            npz, mf = _cache_paths(CACHE_DIR, ds, size, STEP, slice12)
            cache_hit = npz.exists() and mf.exists()
            if not cache_hit and model is None:
                print(f"\nLoading Pythia-{size} step{STEP}...")
                t0 = time.time()
                model, _ = load_pythia(size, STEP)
                print(f"  load: {time.time()-t0:.1f}s")

            t0 = time.time()
            out = metric_battery(
                size, STEP, ds, slices[ds],
                k_parallel=K_PARALLEL, cache_dir=CACHE_DIR,
                model=model, tokenizer_name="EleutherAI/pythia-160m",
            )
            tag = "HIT" if cache_hit else "MISS"
            print(f"  size={size} ds={ds:8s} {tag} {time.time()-t0:5.1f}s  psf={psf(out['parallel_agreement']):.3f}")
            rows.append({
                "size": size, "dataset": ds, "step": STEP,
                "parallel_safety_fraction": psf(out["parallel_agreement"]),
                "mean_agreement_j1_to_k": float(out["parallel_agreement"][..., 1:].float().mean()),
                "n_positions": int(out["parallel_agreement"].shape[1]),
            })
        if model is not None:
            del model

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # 3x3 heatmap (size, domain → psf).
    M = np.zeros((len(SIZES), len(DATASETS)))
    for r in rows:
        M[SIZES.index(r["size"]), DATASETS.index(r["dataset"])] = r["parallel_safety_fraction"]

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=max(0.2, M.max()))
    ax.set_xticks(range(len(DATASETS)), DATASETS)
    ax.set_yticks(range(len(SIZES)), [f"Pythia-{s}" for s in SIZES])
    for i in range(len(SIZES)):
        for j in range(len(DATASETS)):
            ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                    color="white" if M[i,j] < M.max()*0.6 else "black", fontsize=9)
    ax.set_title(f"Parallel-safety fraction by size × domain (step{STEP}, k={K_PARALLEL})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_PATH)
    plt.close(fig)

    MANIFEST_PATH.write_text(json.dumps({
        "experiment": "Phase 2 Step 7 — domain shift",
        "sizes": list(SIZES),
        "datasets": list(DATASETS),
        "step": STEP,
        "n_tokens": N_TOKENS,
        "k_parallel": K_PARALLEL,
        "threshold": THRESHOLD,
        "slice_sha256_by_dataset": {ds: slice_hash(slices[ds]) for ds in DATASETS},
        "csv": str(CSV_PATH.relative_to(_REPO_ROOT)),
        "figure": str(FIG_PATH.relative_to(_REPO_ROOT)),
    }, indent=2, sort_keys=True))

    print(f"\nWrote {CSV_PATH}\nWrote {MANIFEST_PATH}\nWrote {FIG_PATH}")


if __name__ == "__main__":
    main()
