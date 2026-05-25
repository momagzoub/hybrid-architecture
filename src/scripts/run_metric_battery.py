"""Run the Phase 1 metric battery for one (size, step, dataset) cell.

Writes a per-token CSV and a manifest sidecar under
`<out-root>/<size>/step<N>/<dataset>/`. Skips work if the cache exists
and the manifest matches the requested (n_tokens, k).

Usage (single cell):
    python src/scripts/run_metric_battery.py \\
        --size 160m --step 143000 --dataset wikitext \\
        --n-tokens 1024 --k 4

Usage (sweep — run every cell in the canonical grid):
    python src/scripts/run_metric_battery.py --sweep \\
        --sizes 70m 160m 410m \\
        --datasets wikitext mbpp gsm8k \\
        --n-tokens 1024 --k 4

Sweep enumerates `list_checkpoints()` × sizes × datasets and runs each
serially. Cached cells are skipped, so the script is idempotent and
re-runnable after a Colab disconnect.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import torch

# Allow `python src/scripts/run_metric_battery.py` from repo root by adding
# src/ to sys.path. (When `pip install -e` is set up, this is unnecessary,
# but it makes the script work in fresh Colab cells without setup.)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from hybrid_arch import (  # noqa: E402
    attention_concentration,
    attention_entropy,
    extract_attention,
    list_checkpoints,
    load_pythia,
    next_token_entropy,
    parallel_prediction_agreement,
    top1_probability,
)


# ---------------------------------------------------------------------------
# Dataset slice loaders. Each yields raw text chunks; we concatenate, then
# tokenize and slice to the requested length. Cached on disk by (name,
# n_tokens) so re-runs are free.
# ---------------------------------------------------------------------------


def _stream_wikitext() -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )
    for item in ds:
        if item["text"].strip():
            yield item["text"]


def _stream_mbpp() -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", split="train", streaming=True)
    for item in ds:
        yield item.get("text", "") + "\n" + item.get("code", "")


def _stream_gsm8k() -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    for item in ds:
        yield item.get("question", "") + "\n" + item.get("answer", "")


DATASET_STREAMS = {
    "wikitext": _stream_wikitext,
    "mbpp": _stream_mbpp,
    "gsm8k": _stream_gsm8k,
}


def _get_dataset_slice(
    name: str, tok, n_tokens: int, cache_dir: Path
) -> torch.Tensor:
    """Return a tokenized [1, n_tokens] slice of the named dataset, cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{name}_slice_{n_tokens}.pt"
    if cache_path.exists():
        return torch.load(cache_path)

    pieces: list[str] = []
    chars = 0
    for txt in DATASET_STREAMS[name]():
        pieces.append(txt)
        chars += len(txt)
        if chars > n_tokens * 6:  # ~4 chars/token, with comfortable margin
            break
    text = " ".join(pieces)
    enc = tok(text, return_tensors="pt", truncation=False)
    input_ids = enc.input_ids[:, :n_tokens].contiguous()
    if input_ids.shape[1] < n_tokens:
        raise RuntimeError(
            f"dataset {name} yielded only {input_ids.shape[1]} tokens; "
            f"need {n_tokens}. Stream more text."
        )
    torch.save(input_ids, cache_path)
    return input_ids


# ---------------------------------------------------------------------------
# Core: run one cell.
# ---------------------------------------------------------------------------


def run_cell(
    size: str,
    step: int,
    dataset_name: str,
    n_tokens: int,
    k: int,
    out_root: Path,
    data_root: Path,
    *,
    verbose: bool = True,
) -> Path:
    """Run the metric battery for one (size, step, dataset) combination.

    Returns the path to the written CSV. If a valid cached CSV+manifest
    already exists for this combination, returns the cached path without
    re-running.
    """
    out_dir = out_root / size / f"step{step}" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    manifest_path = out_dir / "manifest.json"

    if csv_path.exists() and manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        if m.get("n_tokens") == n_tokens and m.get("k") == k:
            if verbose:
                print(f"  cached: {csv_path.relative_to(out_root.parent)}")
            return csv_path

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    t0 = time.perf_counter()
    model, tok = load_pythia(size, step)
    log(f"  loaded pythia-{size} @ step{step} in {time.perf_counter()-t0:.1f}s")

    input_ids = _get_dataset_slice(
        dataset_name, tok, n_tokens, data_root / "dataset_slices"
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    attn = extract_attention(model, input_ids)
    log(f"  forward + attn extract: {time.perf_counter()-t0:.1f}s")

    tok_entropy = next_token_entropy(logits)[0]
    tok_top1 = top1_probability(logits)[0]

    attn_h = attention_entropy(attn)
    attn_c = attention_concentration(attn, top_k=(1, 3, 5))
    attn_entropy_per_tok = attn_h.mean(dim=(0, 1, 2))
    attn_top1_per_tok = attn_c[0].mean(dim=(0, 1, 2))
    attn_top3_per_tok = attn_c[1].mean(dim=(0, 1, 2))
    attn_top5_per_tok = attn_c[2].mean(dim=(0, 1, 2))

    t0 = time.perf_counter()
    agreement = parallel_prediction_agreement(model, input_ids, k=k, batched=True)
    log(f"  parallel agreement (k={k}): {time.perf_counter()-t0:.1f}s")
    # Average over j > 0 to drop the structural j=0 column (always True).
    if k > 1:
        rate = agreement[0, :, 1:].float().mean(dim=-1)
    else:
        rate = agreement[0, :, 0].float()
    n_valid = rate.shape[0]

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "position", "token_id", "token_str",
            "next_token_entropy", "top1_probability",
            "attention_entropy",
            "attention_top1", "attention_top3", "attention_top5",
            "parallel_agreement_rate",
        ])
        for i in range(n_tokens):
            tid = int(input_ids[0, i].item())
            par_val = f"{rate[i].item():.6f}" if i < n_valid else "nan"
            w.writerow([
                i, tid, tok.decode([tid]),
                f"{tok_entropy[i].item():.6f}",
                f"{tok_top1[i].item():.6f}",
                f"{attn_entropy_per_tok[i].item():.6f}",
                f"{attn_top1_per_tok[i].item():.6f}",
                f"{attn_top3_per_tok[i].item():.6f}",
                f"{attn_top5_per_tok[i].item():.6f}",
                par_val,
            ])

    slice_hash = hashlib.sha256(input_ids.numpy().tobytes()).hexdigest()
    manifest = {
        "size": size,
        "step": step,
        "dataset": dataset_name,
        "n_tokens": n_tokens,
        "k": k,
        "slice_sha256": slice_hash,
        "model_revision": f"step{step}",
        "csv": str(csv_path.name),
        "parallel_agreement_excludes_j0": k > 1,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"  wrote {csv_path.relative_to(out_root.parent)}")
    return csv_path


# ---------------------------------------------------------------------------
# Sweep driver.
# ---------------------------------------------------------------------------


def run_sweep(
    sizes: list[str],
    datasets_: list[str],
    n_tokens: int,
    k: int,
    out_root: Path,
    data_root: Path,
    *,
    steps: list[int] | None = None,
) -> None:
    """Enumerate sizes × steps × datasets and run each cell.

    Cached cells are skipped. Failures on individual cells are caught and
    logged so one bad checkpoint doesn't abort the whole run.
    """
    if steps is None:
        steps = list_checkpoints()
    total = len(sizes) * len(steps) * len(datasets_)
    i = 0
    failures: list[tuple[str, int, str, str]] = []
    for size in sizes:
        for step in steps:
            for ds in datasets_:
                i += 1
                print(f"[{i}/{total}] {size} step{step} {ds}")
                try:
                    run_cell(size, step, ds, n_tokens, k, out_root, data_root)
                except Exception as e:
                    print(f"  FAILED: {type(e).__name__}: {e}")
                    failures.append((size, step, ds, str(e)))
    print(f"\nDone. {total - len(failures)}/{total} cells succeeded.")
    if failures:
        print("Failures:")
        for size, step, ds, err in failures:
            print(f"  {size} step{step} {ds}: {err}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sweep", action="store_true", help="enumerate the full grid")
    ap.add_argument("--size", help="single-cell mode: Pythia size (e.g. 160m)")
    ap.add_argument("--step", type=int, help="single-cell mode: training step")
    ap.add_argument(
        "--dataset", choices=list(DATASET_STREAMS), help="single-cell mode: dataset"
    )
    ap.add_argument(
        "--sizes",
        nargs="+",
        default=["70m", "160m", "410m"],
        help="sweep mode: sizes to evaluate",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_STREAMS),
        help="sweep mode: datasets to evaluate",
    )
    ap.add_argument(
        "--steps",
        nargs="*",
        type=int,
        default=None,
        help="sweep mode: training steps (default: list_checkpoints())",
    )
    ap.add_argument("--n-tokens", type=int, default=512)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out-root", default="data/metric_battery")
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    data_root = Path(args.data_root)

    if args.sweep:
        run_sweep(
            args.sizes,
            args.datasets,
            args.n_tokens,
            args.k,
            out_root,
            data_root,
            steps=args.steps,
        )
    else:
        if not (args.size and args.step is not None and args.dataset):
            ap.error(
                "single-cell mode requires --size, --step, and --dataset "
                "(or pass --sweep)"
            )
        run_cell(
            args.size, args.step, args.dataset, args.n_tokens, args.k,
            out_root, data_root,
        )


if __name__ == "__main__":
    main()
