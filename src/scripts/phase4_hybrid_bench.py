"""Phase 4 Step 3 — measurement harness for the hybrid decoder.

For each of three domains (WikiText, MBPP, GSM8K) and three decoding modes
(pure-target greedy, greedy spec-decode, hybrid-decode-with-default-router),
generate the same number of tokens and report:

- tokens per wall-second (the speed knob)
- divergence vs the pure-AR reference (the quality knob — how often does the
  hybrid output disagree with what the target alone would produce?)
- routing-fraction histograms (what does the router see at each position?)

Writes:
    docs/results/10_hybrid_decoder_bench.csv
    docs/results/10_hybrid_decoder_bench.manifest.json
    docs/results/figures/10_hybrid_throughput.png
    docs/results/figures/10_hybrid_routing_hist.png

This script intentionally does NOT try to beat .generate() on raw throughput
on free Colab — the demo is the point. We report numbers honestly.
"""

from __future__ import annotations

import csv
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from hybrid_arch import (  # noqa: E402
    hybrid_decode,
    load_pythia,
    slice_hash,
    spec_decode_capture,
    threshold_router,
)

DRAFTER_SIZE = "160m"
TARGET_SIZE = "1b"
STEP = 143000
N_PROMPT_TOKENS = 96
N_STEPS = 16
DRAFT_K = 4
ROUTER_THRESHOLD = 0.2
DOMAINS = ("wikitext", "mbpp", "gsm8k")
SLICE_DIR = _REPO_ROOT / "data" / "dataset_slices"
RESULTS_DIR = _REPO_ROOT / "docs" / "results"
FIGURE_DIR = RESULTS_DIR / "figures"

CSV_PATH = RESULTS_DIR / "10_hybrid_decoder_bench.csv"
MANIFEST_PATH = RESULTS_DIR / "10_hybrid_decoder_bench.manifest.json"
FIG_THROUGHPUT = FIGURE_DIR / "10_hybrid_throughput.png"
FIG_HIST = FIGURE_DIR / "10_hybrid_routing_hist.png"


@torch.no_grad()
def greedy_target_decode(target, prompt: torch.Tensor, n_new: int) -> tuple[torch.Tensor, float]:
    """Pure target-only greedy generation. Returns (generated_ids, seconds).

    `generated_ids` is the *new tokens only*, shape `[n_new]`. This is the
    quality reference we compare hybrid output against.
    """
    ctx = prompt.clone()
    out: list[int] = []
    t0 = time.perf_counter()
    for _ in range(n_new):
        last = target(input_ids=ctx).logits[0, -1]
        nxt = int(last.argmax().item())
        out.append(nxt)
        ctx = torch.cat([ctx, torch.tensor([[nxt]], dtype=ctx.dtype)], dim=1)
    return torch.tensor(out, dtype=torch.long), time.perf_counter() - t0


def reconstruct_decoded(prompt: torch.Tensor, trace) -> torch.Tensor:
    """Reconstruct the tokens that spec-decode actually emitted.

    Speculative decoding commits the drafted token if the target's argmax at
    the same position matches; otherwise it overrides with the target's pick.
    `trace.target_token[i]` holds the target's argmax for drafted position i.
    """
    out: list[int] = []
    for i in range(trace.n_drafted):
        out.append(int(trace.target_token[i].item())
                   if not bool(trace.accept[i].item())
                   else int(trace.drafter_token[i].item()))
    return torch.tensor(out, dtype=torch.long)


def reconstruct_hybrid(prompt: torch.Tensor, result) -> torch.Tensor:
    """Hybrid output sequence: when the router keeps a token, use the drafter's
    pick; otherwise use the target's pick (the verifier always runs in this
    benchmark, so the target's argmax at the position is available)."""
    out: list[int] = []
    for i in range(result.n):
        kept = bool(result.router_decision[i].item())
        if kept:
            out.append(int(result.spec_trace.drafter_token[i].item()))
        else:
            out.append(int(result.spec_trace.target_token[i].item()))
    return torch.tensor(out, dtype=torch.long)


def divergence(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of positions where two same-length sequences disagree."""
    n = min(a.numel(), b.numel())
    if n == 0:
        return 0.0
    return float((a[:n] != b[:n]).float().mean())


def ensure_slice(name: str, tok) -> torch.Tensor:
    p = SLICE_DIR / f"{name}_slice_256.pt"
    if p.exists():
        return torch.load(p, weights_only=False)
    # Reuse the streaming loaders embedded in the domain-shift script.
    from datasets import load_dataset
    if name == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                          split="train", streaming=True)
        gen = (it["text"] for it in ds if it.get("text", "").strip())
    elif name == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", split="train", streaming=True)
        gen = (it.get("text", "") + "\n" + it.get("code", "") for it in ds)
    elif name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
        gen = (it.get("question", "") + "\n" + it.get("answer", "") for it in ds)
    else:
        raise ValueError(name)
    pieces, chars = [], 0
    for txt in gen:
        pieces.append(txt)
        chars += len(txt)
        if chars > 256 * 8:
            break
    enc = tok(" ".join(pieces), return_tensors="pt", truncation=False)
    ids = enc.input_ids[:, :256].contiguous()
    if ids.shape[1] < 256:
        raise RuntimeError(f"{name} yielded only {ids.shape[1]} tokens")
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ids, p)
    return ids


def main() -> None:
    print(f"Loading drafter Pythia-{DRAFTER_SIZE}@step{STEP}...")
    drafter, tok = load_pythia(DRAFTER_SIZE, STEP)
    print(f"Loading target Pythia-{TARGET_SIZE}@step{STEP}...")
    target, _ = load_pythia(TARGET_SIZE, STEP)

    slices = {d: ensure_slice(d, tok) for d in DOMAINS}
    prompts = {d: slices[d][:, :N_PROMPT_TOKENS].contiguous() for d in DOMAINS}
    n_new = N_STEPS * DRAFT_K

    bench_rows: list[dict] = []
    hist_data: dict[str, dict] = {}

    for domain in DOMAINS:
        prompt = prompts[domain]
        print(f"\n=== domain={domain}  prompt={N_PROMPT_TOKENS}t  generating "
              f"{n_new} new tokens ===")

        # -- pure-target reference --
        print("  mode=pure_target   ", end="", flush=True)
        ref_tokens, ref_secs = greedy_target_decode(target, prompt, n_new)
        ref_tps = n_new / ref_secs
        print(f"{ref_secs:6.1f}s  {ref_tps:5.2f} tok/s")
        bench_rows.append({
            "domain": domain, "mode": "pure_target",
            "n_tokens": n_new, "wall_s": ref_secs, "tokens_per_sec": ref_tps,
            "accept_rate": 1.0, "router_keep_rate": 0.0, "false_keep_rate": 0.0,
            "divergence_vs_pure_target": 0.0,
        })

        # -- greedy spec-decode --
        print("  mode=spec_decode   ", end="", flush=True)
        t0 = time.perf_counter()
        spec_trace = spec_decode_capture(target, drafter, prompt,
                                         n_steps=N_STEPS, draft_k=DRAFT_K)
        spec_secs = time.perf_counter() - t0
        spec_tokens = reconstruct_decoded(prompt, spec_trace)
        spec_div = divergence(spec_tokens, ref_tokens)
        spec_tps = spec_trace.n_drafted / spec_secs
        print(f"{spec_secs:6.1f}s  {spec_tps:5.2f} tok/s  "
              f"accept={spec_trace.accept_rate:.3f}  div={spec_div:.3f}")
        bench_rows.append({
            "domain": domain, "mode": "spec_decode",
            "n_tokens": spec_trace.n_drafted, "wall_s": spec_secs,
            "tokens_per_sec": spec_tps,
            "accept_rate": spec_trace.accept_rate,
            "router_keep_rate": 0.0, "false_keep_rate": 0.0,
            "divergence_vs_pure_target": spec_div,
        })

        # -- hybrid decode (default router) --
        print("  mode=hybrid        ", end="", flush=True)
        router = threshold_router("one_minus_top1", threshold=ROUTER_THRESHOLD)
        t0 = time.perf_counter()
        result = hybrid_decode(target, drafter, prompt,
                               router=router, n_steps=N_STEPS, draft_k=DRAFT_K)
        hyb_secs = time.perf_counter() - t0
        hyb_tokens = reconstruct_hybrid(prompt, result)
        hyb_div = divergence(hyb_tokens, ref_tokens)
        hyb_tps = result.n / hyb_secs
        print(f"{hyb_secs:6.1f}s  {hyb_tps:5.2f} tok/s  "
              f"keep={result.router_keep_rate:.3f}  "
              f"false_keep={result.false_keep_rate:.3f}  div={hyb_div:.3f}")
        bench_rows.append({
            "domain": domain, "mode": "hybrid",
            "n_tokens": result.n, "wall_s": hyb_secs, "tokens_per_sec": hyb_tps,
            "accept_rate": result.spec_trace.accept_rate,
            "router_keep_rate": result.router_keep_rate,
            "false_keep_rate": result.false_keep_rate,
            "divergence_vs_pure_target": hyb_div,
        })

        # histogram data: one_minus_top1 split by accept/reject
        hist_data[domain] = {
            "one_minus_top1": (1.0 - result.spec_trace.top1).numpy(),
            "accept": result.spec_trace.accept.numpy().astype(bool),
        }

    del target, drafter
    gc.collect()

    # ----- CSV + manifest -----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(bench_rows[0].keys()))
        w.writeheader()
        w.writerows(bench_rows)

    MANIFEST_PATH.write_text(json.dumps({
        "experiment": "Phase 4 Step 3 — hybrid-decoder measurement harness",
        "target": f"EleutherAI/pythia-{TARGET_SIZE}",
        "drafter": f"EleutherAI/pythia-{DRAFTER_SIZE}",
        "step": STEP,
        "domains": list(DOMAINS),
        "prompt_tokens": N_PROMPT_TOKENS,
        "n_steps": N_STEPS,
        "draft_k": DRAFT_K,
        "router": f"threshold_router('one_minus_top1', {ROUTER_THRESHOLD})",
        "slice_sha256": {d: slice_hash(prompts[d]) for d in DOMAINS},
        "csv": str(CSV_PATH.relative_to(_REPO_ROOT)),
        "figures": [
            str(FIG_THROUGHPUT.relative_to(_REPO_ROOT)),
            str(FIG_HIST.relative_to(_REPO_ROOT)),
        ],
    }, indent=2, sort_keys=True))

    # ----- throughput bar chart -----
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    modes = ("pure_target", "spec_decode", "hybrid")
    width = 0.25
    x = np.arange(len(DOMAINS))
    for i, mode in enumerate(modes):
        tps = [next(r["tokens_per_sec"] for r in bench_rows
                    if r["domain"] == d and r["mode"] == mode) for d in DOMAINS]
        ax.bar(x + i * width, tps, width, label=mode)
    ax.set_xticks(x + width)
    ax.set_xticklabels(DOMAINS)
    ax.set_ylabel("Tokens per wall-second (CPU)")
    ax.set_title(
        f"Hybrid decoder throughput vs baselines\n"
        f"target Pythia-{TARGET_SIZE} / drafter Pythia-{DRAFTER_SIZE}, "
        f"router=`1-top1 < {ROUTER_THRESHOLD}`"
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_THROUGHPUT)
    plt.close(fig)

    # ----- routing histograms, small multiples by domain -----
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(15, 4), dpi=120, sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, domain in zip(axes, DOMAINS):
        data = hist_data[domain]
        v = data["one_minus_top1"]
        ax.hist(v[data["accept"]], bins=bins, alpha=0.6, label="target accepted")
        ax.hist(v[~data["accept"]], bins=bins, alpha=0.6, label="target rejected")
        ax.axvline(ROUTER_THRESHOLD, color="black", linestyle=":", linewidth=1)
        ax.set_title(f"{domain} (n={v.size})")
        ax.set_xlabel("1 − top1 (drafter)")
    axes[0].set_ylabel("count")
    axes[-1].legend(loc="upper right", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(
        f"What the router sees vs what the verifier said "
        f"(threshold dashed at {ROUTER_THRESHOLD})"
    )
    fig.tight_layout()
    fig.savefig(FIG_HIST)
    plt.close(fig)

    print(f"\nWrote {CSV_PATH}\nWrote {MANIFEST_PATH}\nWrote {FIG_THROUGHPUT}\nWrote {FIG_HIST}")


if __name__ == "__main__":
    main()
