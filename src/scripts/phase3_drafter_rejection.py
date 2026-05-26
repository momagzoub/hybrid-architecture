"""Phase 3 Step 4 — does the probe predict speculative-decoding rejections?

Greedy speculative decoding with Pythia-1B as the target and Pythia-160M as
the drafter. For each drafted position we know (a) the ground-truth accept
or reject bit from the verifier and (b) the drafter's cheap features:
next-token entropy, top-1 probability, and the per-layer hidden state.

We then evaluate four predictors of the reject event:

    - random                 (sanity baseline)
    - next_token_entropy     (Phase 1 baseline)
    - top1_probability       (Phase 1 baseline)
    - LayerProbe at L9       (the Phase 3 contender; loaded from disk)

All four are *not retrained* on this distribution — the probe was trained
on the offline parallel-safety label of the drafter on WikiText, not on
the actual 1B target's reject events. The interesting question is how well
that offline signal transfers.

Writes:
    docs/results/08_drafter_rejection.csv
    docs/results/08_drafter_rejection.manifest.json
    docs/results/figures/08_drafter_rejection_roc.png
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
from sklearn.metrics import roc_auc_score, roc_curve

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from hybrid_arch import load_probe, load_pythia, slice_hash  # noqa: E402
from hybrid_arch.spec_decode import spec_decode_capture  # noqa: E402

DRAFTER_SIZE = "160m"
TARGET_SIZE = "1b"
STEP = 143000
N_PROMPT_TOKENS = 96
N_STEPS = 16
DRAFT_K = 4
PROBE_LAYER = 9                 # best 160m layer per Step 3
SLICE_PATH = _REPO_ROOT / "data" / "dataset_slices" / "wikitext_slice_256.pt"
RESULTS_DIR = _REPO_ROOT / "docs" / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
PROBE_PATH = (_REPO_ROOT / "src" / "hybrid_arch" / "probe_checkpoints"
              / f"{DRAFTER_SIZE}_step{STEP}_L{PROBE_LAYER}.pt")

CSV_PATH = RESULTS_DIR / "08_drafter_rejection.csv"
MANIFEST_PATH = RESULTS_DIR / "08_drafter_rejection.manifest.json"
FIG_PATH = FIGURE_DIR / "08_drafter_rejection_roc.png"


def main() -> None:
    print(f"Loading drafter Pythia-{DRAFTER_SIZE}@step{STEP}...")
    t0 = time.time()
    drafter, _ = load_pythia(DRAFTER_SIZE, STEP)
    print(f"  load: {time.time()-t0:.1f}s")

    print(f"Loading target Pythia-{TARGET_SIZE}@step{STEP}...")
    t0 = time.time()
    target, _ = load_pythia(TARGET_SIZE, STEP)
    print(f"  load: {time.time()-t0:.1f}s")

    # Use the first N_PROMPT_TOKENS of the 256-token WikiText slice.
    slice_all = torch.load(SLICE_PATH, weights_only=False)
    prompt = slice_all[:, :N_PROMPT_TOKENS].contiguous()
    print(f"Prompt: {N_PROMPT_TOKENS} tokens, sha={slice_hash(prompt)[:12]}")

    print(f"Loading probe from {PROBE_PATH.name}...")
    probe = load_probe(PROBE_PATH)

    print(f"\nRunning {N_STEPS} steps × draft_k={DRAFT_K}...")
    t0 = time.time()
    trace = spec_decode_capture(target, drafter, prompt, n_steps=N_STEPS, draft_k=DRAFT_K)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")
    print(f"  drafted={trace.n_drafted}  accepted={trace.n_accepted}  "
          f"accept_rate={trace.accept_rate:.3f}")

    # Build the predictor scores. We're predicting REJECTION (label = ~accept).
    reject = (~trace.accept).numpy().astype(np.int64)
    n_reject = int(reject.sum())
    n_total = int(reject.size)
    print(f"\nreject events: {n_reject}/{n_total} ({n_reject/n_total*100:.1f}%)")
    if n_reject < 2 or n_reject > n_total - 2:
        print("Degenerate distribution — cannot compute AUROC.")
        return

    # Scores (higher = more likely to reject).
    rng = np.random.RandomState(0)
    random_score = rng.randn(n_total).astype(np.float32)
    entropy_score = trace.entropy.numpy()
    inverse_top1 = (1.0 - trace.top1.numpy())
    # Probe predicts parallel-safety. Higher probe → MORE safe → LESS reject.
    # So predicted-reject score = -probe_logit.
    hs_layer = trace.drafter_hidden_states[PROBE_LAYER].float()   # [N, H]
    with torch.no_grad():
        probe_logits = probe(hs_layer).cpu().numpy()
    probe_reject_score = -probe_logits

    predictors = {
        "random": random_score,
        "next_token_entropy": entropy_score,
        "1_minus_top1": inverse_top1,
        f"probe_L{PROBE_LAYER}": probe_reject_score,
    }

    print("\nAUROC for predicting REJECTION:")
    aurocs: dict[str, float] = {}
    for name, scores in predictors.items():
        auc = float(roc_auc_score(reject, scores))
        aurocs[name] = auc
        print(f"  {name:24s}  {auc:.3f}")

    # ----- CSV + manifest -----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {"predictor": name, "auroc_reject": auc, "n_total": n_total, "n_reject": n_reject}
        for name, auc in aurocs.items()
    ]
    with CSV_PATH.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    MANIFEST_PATH.write_text(json.dumps({
        "experiment": "Phase 3 Step 4 — drafter-rejection prediction",
        "target": f"EleutherAI/pythia-{TARGET_SIZE}",
        "drafter": f"EleutherAI/pythia-{DRAFTER_SIZE}",
        "step": STEP,
        "prompt_tokens": N_PROMPT_TOKENS,
        "n_steps": N_STEPS,
        "draft_k": DRAFT_K,
        "probe_layer": PROBE_LAYER,
        "probe_checkpoint": str(PROBE_PATH.relative_to(_REPO_ROOT)),
        "slice_sha256": slice_hash(prompt),
        "n_drafted": trace.n_drafted,
        "n_accepted": trace.n_accepted,
        "accept_rate": trace.accept_rate,
        "csv": str(CSV_PATH.relative_to(_REPO_ROOT)),
        "figure": str(FIG_PATH.relative_to(_REPO_ROOT)),
    }, indent=2, sort_keys=True))

    # ----- ROC plot -----
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    for name, scores in predictors.items():
        fpr, tpr, _ = roc_curve(reject, scores)
        ax.plot(fpr, tpr, label=f"{name}  (AUC={aurocs[name]:.2f})")
    ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (reject detection)")
    ax.set_title(
        f"Predicting rejection in greedy spec-decode\n"
        f"target Pythia-{TARGET_SIZE} / drafter Pythia-{DRAFTER_SIZE} "
        f"(n_drafted={trace.n_drafted}, reject={n_reject})"
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_PATH)
    plt.close(fig)
    print(f"\nWrote {CSV_PATH}\nWrote {MANIFEST_PATH}\nWrote {FIG_PATH}")


if __name__ == "__main__":
    main()
