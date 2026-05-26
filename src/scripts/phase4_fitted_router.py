"""Phase 4 Step 2 — does fitting on real rejection labels help, and does the
probe earn its place once the baselines are already in the regression?

Long greedy spec-decode run (Pythia-1b target / Pythia-160m drafter), 70/30
train/test split, four `LogisticRegression` ablations:

    1. `1 − top1`                        (the Phase 3 best baseline)
    2. `1 − top1 + entropy`              (the obvious extension)
    3. `1 − top1 + entropy + probe_L9`   (add Phase 3 probe)
    4. probe alone                       (sanity — does the probe carry
                                          anything once it's fitted on the
                                          right label?)

Phase 3 already showed the offline-trained probe predicts rejection at
chance (AUROC ~0.60). The question here is whether a probe *fitted on real
rejection labels* — even as a single feature — has predictive power, and
whether it adds anything on top of the cheap drafter-side baselines.

Writes:
    docs/results/09_router_coefficients.csv
    docs/results/09_router_coefficients.manifest.json
    docs/results/figures/09_router_roc.png
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from hybrid_arch import load_probe, load_pythia, slice_hash  # noqa: E402
from hybrid_arch.spec_decode import spec_decode_capture  # noqa: E402

DRAFTER_SIZE = "160m"
TARGET_SIZE = "1b"
STEP = 143000
N_PROMPT_TOKENS = 96
N_STEPS = 256          # 256 * 4 = 1024 drafted positions; budget ~60-90 min CPU
DRAFT_K = 4
PROBE_LAYER = 9
SEED = 0
SLICE_PATH = _REPO_ROOT / "data" / "dataset_slices" / "wikitext_slice_256.pt"
RESULTS_DIR = _REPO_ROOT / "docs" / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
PROBE_PATH = (_REPO_ROOT / "src" / "hybrid_arch" / "probe_checkpoints"
              / f"{DRAFTER_SIZE}_step{STEP}_L{PROBE_LAYER}.pt")
TRACE_CACHE = _REPO_ROOT / "data" / "cache" / "spec_decode_traces"

CSV_PATH = RESULTS_DIR / "09_router_coefficients.csv"
MANIFEST_PATH = RESULTS_DIR / "09_router_coefficients.manifest.json"
FIG_PATH = FIGURE_DIR / "09_router_roc.png"


def get_trace(prompt: torch.Tensor):
    """Run the long spec-decode once and cache the result.

    Each trace is the slow part of this script (~60-90 min on CPU); cache the
    raw outputs so re-running for plot tweaks is free.
    """
    TRACE_CACHE.mkdir(parents=True, exist_ok=True)
    key = (f"{TARGET_SIZE}vs{DRAFTER_SIZE}_step{STEP}_n{N_STEPS}_k{DRAFT_K}"
           f"_{slice_hash(prompt)[:12]}")
    path = TRACE_CACHE / f"{key}.pt"
    if path.exists():
        print(f"[trace cache hit] {path.name}")
        return torch.load(path, weights_only=False)

    print(f"Loading drafter Pythia-{DRAFTER_SIZE}@step{STEP}...")
    t0 = time.time()
    drafter, _ = load_pythia(DRAFTER_SIZE, STEP)
    print(f"  load: {time.time()-t0:.1f}s")

    print(f"Loading target Pythia-{TARGET_SIZE}@step{STEP}...")
    t0 = time.time()
    target, _ = load_pythia(TARGET_SIZE, STEP)
    print(f"  load: {time.time()-t0:.1f}s")

    print(f"\nRunning {N_STEPS} steps × draft_k={DRAFT_K} "
          f"= {N_STEPS * DRAFT_K} drafted positions...")
    t0 = time.time()
    trace = spec_decode_capture(target, drafter, prompt, n_steps=N_STEPS, draft_k=DRAFT_K)
    print(f"  done in {(time.time()-t0)/60:.1f} min  "
          f"accept_rate={trace.accept_rate:.3f}")

    torch.save(trace, path)
    return trace


def build_features(trace, probe) -> dict[str, np.ndarray]:
    one_minus_top1 = (1.0 - trace.top1.numpy()).astype(np.float32)
    entropy = trace.entropy.numpy().astype(np.float32)
    with torch.no_grad():
        probe_logit = probe(trace.drafter_hidden_states[PROBE_LAYER].float()).cpu().numpy()
    return {
        "one_minus_top1": one_minus_top1,
        "entropy": entropy,
        "probe_logit": probe_logit.astype(np.float32),
    }


ABLATIONS = [
    ("one_minus_top1", ["one_minus_top1"]),
    ("entropy_plus_top1", ["one_minus_top1", "entropy"]),
    ("baseline_plus_probe", ["one_minus_top1", "entropy", "probe_logit"]),
    ("probe_alone", ["probe_logit"]),
]


def fit_one(X_tr, y_tr, X_te, y_te) -> tuple[LogisticRegression, StandardScaler, float, float]:
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, solver="liblinear")
    clf.fit(scaler.transform(X_tr), y_tr)
    train_score = clf.decision_function(scaler.transform(X_tr))
    test_score = clf.decision_function(scaler.transform(X_te))
    auc_tr = float(roc_auc_score(y_tr, train_score))
    auc_te = float(roc_auc_score(y_te, test_score))
    return clf, scaler, auc_tr, auc_te


def main() -> None:
    slice_all = torch.load(SLICE_PATH, weights_only=False)
    prompt = slice_all[:, :N_PROMPT_TOKENS].contiguous()
    print(f"Prompt: {N_PROMPT_TOKENS} tokens, sha={slice_hash(prompt)[:12]}")

    trace = get_trace(prompt)
    print(f"\n  drafted={trace.n_drafted}  accepted={trace.n_accepted}  "
          f"accept_rate={trace.accept_rate:.3f}")

    probe = load_probe(PROBE_PATH)
    feats = build_features(trace, probe)
    reject = (~trace.accept).numpy().astype(np.int64)
    print(f"  reject events: {int(reject.sum())}/{reject.size} "
          f"({reject.mean()*100:.1f}%)")

    if reject.sum() < 5 or reject.sum() > reject.size - 5:
        print("Degenerate label distribution — aborting.")
        return

    # Single 70/30 train/test split, stratified.
    indices = np.arange(reject.size)
    idx_tr, idx_te = train_test_split(indices, test_size=0.30, random_state=SEED, stratify=reject)
    y_tr, y_te = reject[idx_tr], reject[idx_te]
    print(f"  train: {idx_tr.size}  test: {idx_te.size}\n")

    # Reference: Phase 3 unfitted `1 − top1` on the test split.
    baseline_unfitted = float(roc_auc_score(y_te, feats["one_minus_top1"][idx_te]))
    print(f"Reference: unfitted `1 − top1` test AUROC = {baseline_unfitted:.3f}\n")

    rows: list[dict] = []
    roc_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, feature_names in ABLATIONS:
        X = np.stack([feats[f] for f in feature_names], axis=1)
        X_tr, X_te = X[idx_tr], X[idx_te]
        clf, scaler, auc_tr, auc_te = fit_one(X_tr, y_tr, X_te, y_te)
        # Pull coefs back to the original (unstandardized) feature scale so they
        # are readable from the CSV.
        coef_std = clf.coef_[0]
        coef_orig = coef_std / scaler.scale_
        bias_orig = float(clf.intercept_[0] - (coef_std * scaler.mean_ / scaler.scale_).sum())

        # ROC curve on test split for plotting.
        score_te = clf.decision_function(scaler.transform(X_te))
        fpr, tpr, _ = roc_curve(y_te, score_te)
        roc_data[name] = (fpr, tpr)

        print(f"{name}")
        print(f"  features      = {feature_names}")
        print(f"  train AUROC   = {auc_tr:.3f}")
        print(f"  test  AUROC   = {auc_te:.3f}")
        print(f"  coef (orig)   = {dict(zip(feature_names, coef_orig.round(3)))}")
        print(f"  bias (orig)   = {bias_orig:.3f}\n")

        for fname, c in zip(feature_names, coef_orig):
            rows.append({
                "model": name,
                "features": "+".join(feature_names),
                "feature_name": fname,
                "coef": float(c),
                "bias": bias_orig,
                "train_auroc": auc_tr,
                "test_auroc": auc_te,
                "n_positive": int(y_tr.sum()) + int(y_te.sum()),
                "n_total": int(y_tr.size) + int(y_te.size),
            })

    # ----- CSV + manifest -----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    MANIFEST_PATH.write_text(json.dumps({
        "experiment": "Phase 4 Step 2 — fitted router on real rejection labels",
        "target": f"EleutherAI/pythia-{TARGET_SIZE}",
        "drafter": f"EleutherAI/pythia-{DRAFTER_SIZE}",
        "step": STEP,
        "prompt_tokens": N_PROMPT_TOKENS,
        "n_steps": N_STEPS,
        "draft_k": DRAFT_K,
        "n_drafted": trace.n_drafted,
        "n_accepted": trace.n_accepted,
        "accept_rate": trace.accept_rate,
        "split": "70/30 stratified, random_state=0",
        "model": "sklearn LogisticRegression(class_weight=balanced, C=1, liblinear)",
        "probe_layer": PROBE_LAYER,
        "probe_checkpoint": str(PROBE_PATH.relative_to(_REPO_ROOT)),
        "slice_sha256": slice_hash(prompt),
        "phase3_unfitted_one_minus_top1_test_auroc": baseline_unfitted,
        "csv": str(CSV_PATH.relative_to(_REPO_ROOT)),
        "figure": str(FIG_PATH.relative_to(_REPO_ROOT)),
    }, indent=2, sort_keys=True))

    # ----- ROC plot -----
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    test_aurocs = {r["model"]: r["test_auroc"] for r in rows}
    for name, (fpr, tpr) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name}  AUC={test_aurocs[name]:.2f}")
    ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="chance")
    ax.axhline(0, alpha=0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(
        f"Fitted-router ROC (test split, n={int(y_te.size)})\n"
        f"target Pythia-{TARGET_SIZE} / drafter Pythia-{DRAFTER_SIZE}, "
        f"Phase 3 unfitted top1 AUROC={baseline_unfitted:.2f}"
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_PATH)
    plt.close(fig)
    print(f"\nWrote {CSV_PATH}\nWrote {MANIFEST_PATH}\nWrote {FIG_PATH}")


if __name__ == "__main__":
    main()
