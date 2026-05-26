"""On-disk cache for the per-token metric battery.

Phase 2 runs the Phase 1 metrics across ~36 (size, step) pairs × 3 datasets
= 108 metric-battery passes. Re-loading models and re-running the battery
on every notebook re-execution would blow the compute budget; this module
materializes the per-token outputs once and lets every downstream analysis
read them back in milliseconds.

Storage layout::

    data/cache/<dataset>/<size>/step<step>_<slice12>.npz
    data/cache/<dataset>/<size>/step<step>_<slice12>.manifest.json

The slice12 suffix is the first 12 hex chars of the input_ids sha256 — so
the same (size, step, dataset) at different token-slices coexist without
collision.

Public API::

    slice_hash(input_ids)               -> str (hex sha256)
    metric_battery(...)                 -> dict[str, Tensor]
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from hybrid_arch.attention import extract_attention
from hybrid_arch.metrics import (
    attention_concentration,
    attention_entropy,
    next_token_entropy,
    parallel_prediction_agreement,
    top1_probability,
)

DEFAULT_TOP_K = (1, 3, 5)
DEFAULT_K_PARALLEL = 4


def _project_root() -> Path:
    """Walk up from this file to the repo root (where `pyproject.toml` lives)."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[2]


def slice_hash(input_ids: Tensor) -> str:
    """Deterministic sha256 hex of an `input_ids` tensor.

    Hashes the int32 little-endian bytes plus the shape, so two tensors
    with the same content but different shapes (e.g., `[seq]` vs `[1, seq]`)
    hash distinctly. Matches the Phase 1 manifest format.
    """
    arr = input_ids.detach().to(torch.int32).cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(repr(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _cache_paths(
    cache_dir: Path,
    dataset_name: str,
    model_size: str,
    step: int,
    slice12: str,
) -> tuple[Path, Path]:
    folder = cache_dir / dataset_name / model_size
    stem = f"step{step}_{slice12}"
    return folder / f"{stem}.npz", folder / f"{stem}.manifest.json"


def _save_npz(npz_path: Path, arrays: dict[str, np.ndarray]) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: stage to .tmp then rename.
    tmp = npz_path.with_suffix(npz_path.suffix + ".tmp")
    # np.savez auto-appends ".npz" to string/Path args, so we pass an open
    # file handle to keep the exact filename we asked for.
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, npz_path)


def _save_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, manifest_path)


def _compute_battery(
    model: torch.nn.Module,
    input_ids: Tensor,
    *,
    k_parallel: int,
    top_k: tuple[int, ...],
) -> dict[str, Tensor]:
    """Run every metric in the battery on (`model`, `input_ids`). All tensors fp32/bool, on CPU."""
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits  # [B, S, V]
        nte = next_token_entropy(logits).cpu()              # [B, S]
        top1 = top1_probability(logits).cpu()               # [B, S]
        del logits

        attn = extract_attention(model, input_ids)          # [L, B, H, S, S]
        attn_ent_ph = attention_entropy(attn).cpu()         # [L, B, H, S]
        attn_conc_ph = attention_concentration(attn, top_k=top_k).cpu()  # [K, L, B, H, S]
        del attn

        pa = parallel_prediction_agreement(model, input_ids, k=k_parallel)
        pa = pa.cpu()

    return {
        "next_token_entropy": nte,
        "top1_probability": top1,
        "attention_entropy_per_head": attn_ent_ph,
        "attention_concentration_per_head": attn_conc_ph,
        "parallel_agreement": pa,
        "input_ids": input_ids.detach().cpu(),
    }


def _to_torch(arrays: dict[str, np.ndarray]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for key, arr in arrays.items():
        # np.savez wraps scalars in 0-d arrays; np.load returns them as such.
        # All our entries are at least 1-d, so we just convert.
        if arr.dtype == np.bool_:
            out[key] = torch.from_numpy(arr.copy())
        elif np.issubdtype(arr.dtype, np.integer):
            out[key] = torch.from_numpy(arr.astype(np.int64))
        else:
            out[key] = torch.from_numpy(arr.astype(np.float32))
    return out


def _to_numpy(tensors: dict[str, Tensor]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, t in tensors.items():
        if t.dtype == torch.bool:
            out[key] = t.cpu().numpy()
        elif t.dtype in (torch.int32, torch.int64, torch.long):
            out[key] = t.cpu().to(torch.int64).numpy()
        else:
            out[key] = t.cpu().to(torch.float32).numpy()
    return out


def metric_battery(
    model_size: str,
    step: int,
    dataset_name: str,
    input_ids: Tensor,
    *,
    k_parallel: int = DEFAULT_K_PARALLEL,
    top_k: tuple[int, ...] = DEFAULT_TOP_K,
    force_recompute: bool = False,
    cache_dir: Path | str | None = None,
    model: torch.nn.Module | None = None,
    tokenizer_name: str | None = None,
) -> dict[str, Tensor]:
    """Load or compute the per-token metric battery for one (size, step, slice).

    First checks the on-disk cache (keyed by model size, step, dataset, and the
    sha256 of `input_ids`). On cache hit, returns the saved tensors without
    touching the model. On miss, loads the model (or uses the one passed in),
    runs the full Phase 1 metric battery, writes both the `.npz` and the
    `.manifest.json` sidecar, and returns the tensors.

    Args:
        model_size: Pythia size, e.g. `"70m"`, `"160m"`, `"410m"`.
        step: Pythia training step.
        dataset_name: Short name used in the cache path, e.g. `"wikitext"`.
        input_ids: `[batch, seq]` long tensor — the slice to evaluate.
        k_parallel: lookahead horizon for `parallel_prediction_agreement`.
        top_k: `top_k` values for `attention_concentration`.
        force_recompute: If True, ignore any cached file and recompute.
        cache_dir: Override the default `<repo>/data/cache` location.
        model: Optional preloaded model. If None, calls `load_pythia(model_size, step)`.
            Always ignored on a cache hit.
        tokenizer_name: Optional name to record in the manifest (purely for
            provenance — not used to invalidate the cache).

    Returns:
        Dict with keys:
            ``next_token_entropy``        — `[B, S]` fp32
            ``top1_probability``          — `[B, S]` fp32
            ``attention_entropy_per_head``     — `[L, B, H, S]` fp32
            ``attention_concentration_per_head`` — `[K, L, B, H, S]` fp32
            ``parallel_agreement``        — `[B, S - k_parallel, k_parallel]` bool
            ``input_ids``                 — `[B, S]` int64
    """
    if cache_dir is None:
        cache_dir = _project_root() / "data" / "cache"
    cache_dir = Path(cache_dir)

    slice12 = slice_hash(input_ids)[:12]
    npz_path, manifest_path = _cache_paths(
        cache_dir, dataset_name, model_size, step, slice12
    )

    if not force_recompute and npz_path.exists() and manifest_path.exists():
        with np.load(npz_path) as data:
            arrays = {key: data[key] for key in data.files}
        return _to_torch(arrays)

    # --- Cache miss: compute. ---
    if model is None:
        from hybrid_arch.checkpoints import load_pythia
        model, _tok = load_pythia(model_size, step)  # type: ignore[arg-type]

    tensors = _compute_battery(
        model, input_ids, k_parallel=k_parallel, top_k=top_k
    )

    _save_npz(npz_path, _to_numpy(tensors))
    full_hash = slice_hash(input_ids)
    manifest = {
        "model_name": f"EleutherAI/pythia-{model_size}",
        "model_revision": f"step{step}",
        "model_size": model_size,
        "step": step,
        "dataset": dataset_name,
        "slice_sha256": full_hash,
        "n_tokens": int(input_ids.numel()),
        "shape": list(input_ids.shape),
        "k_parallel": k_parallel,
        "top_k": list(top_k),
        "tokenizer_name": tokenizer_name,
        "tensor_keys": sorted(tensors.keys()),
    }
    _save_manifest(manifest_path, manifest)
    return tensors
