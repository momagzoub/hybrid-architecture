# AGENTS.md — Hybrid Architecture

> Project guide for Mohamed (`m0hamed@mit.edu`) and any engineer or coding
> assistant joining the project.
> Last refreshed: **2026-05-25** (Phase 2 close).

This file is the front door. Read it first. The roadmap lives in
[`PROJECT_PLAN.md`](./PROJECT_PLAN.md); the Phase 2 results in
[`docs/results/02_emergence_atlas.md`](./docs/results/02_emergence_atlas.md);
the Phase 3 brief in [`PHASE_3_HANDOFF.md`](./PHASE_3_HANDOFF.md).

---

## 1. The pitch

Speculative decoding, mixture-of-depths, mixture-of-recursions: every adaptive
inference technique hinges on a signal about *which tokens are easy*. We
(a) measure that signal directly from per-(layer, head) attention, (b) track
how it emerges across Pythia's 154 training checkpoints — a developmental
angle the big labs can't replicate because they only release final
checkpoints — and (c) ship a diagnostic toolkit other people can use.

**Working title:** *Parallelism Emerges: An attention-pattern atlas of how
language models learn what to compute in parallel.*

---

## 2. Research question (current form)

Across pretraining, how does the **separability** of "parallel-safe" vs
"sequential-needed" tokens emerge in small language models, and what
attention-pattern signatures predict it?

What "novel" means here: **not** "beat EAGLE-3 on speed" (lost on free Colab).
Novel means "produce evidence and tooling that nobody has because they all
start from final checkpoints of huge models."

The Phase 2 atlas answers the empirical version of this question for Pythia
70m-410m. Phase 3 turns the answer into a callable probe.

---

## 3. State of the art (as of 2026-05) — what we are NOT trying to beat

- **EAGLE-3 (2025)** — speculative decoding, 0.75-0.85 acceptance rate, 3-6× speedup. ([arXiv:2503.01840](https://arxiv.org/html/2503.01840v1))
- **Mixture-of-Recursions (NeurIPS 2025)** — per-token adaptive recursive depth + recursion-wise KV caching. ([arXiv:2507.10524](https://arxiv.org/abs/2507.10524))
- **Mirror Speculative Decoding (Oct 2025)** — parallel-draft variant. ([arXiv:2510.13161](https://arxiv.org/pdf/2510.13161))
- **DeepSeek MTP** — multi-token prediction baked into pretraining.
- **ADEPT (2026)** — draft model deciding per-token depth.

Our contribution lives upstream of all of these.

---

## 4. Repo conventions

```
hybrid-architecture/
├── AGENTS.md, README.md, PROJECT_PLAN.md
├── PHASE_3_HANDOFF.md                 ← next phase brief
├── pyproject.toml                     ← installable as `pip install -e ".[dev]"`
├── docs/results/                      ← atlas + CSVs + figures (tracked)
├── src/hybrid_arch/                   ← importable library
│   ├── attention.py, cache.py, checkpoints.py, metrics.py, viz.py
├── src/scripts/                       ← phase{N}_*.py sweep runners
├── tests/                             ← pytest, 81 tests
├── data/                              ← caches, dataset slices (gitignored)
└── results/                           ← raw experiment outputs (gitignored)
```

### Non-negotiables

- **Logic lives in `src/`.** Scripts are runners, not analysis.
- **Every experiment writes a `manifest.json`** alongside its outputs:
  model name + revision, seed, dataset slice hash, k, threshold, git commit if
  available.
- **Plots use one consistent style.** Set up in `viz.py`; don't touch
  matplotlib defaults outside it.
- **Commit hygiene.** Small commits, imperative messages (`add KV cache probe`).
- **No giant files in git.** Models, datasets, and result tensors go in
  `data/` and `results/`, both gitignored. Tracked outputs in
  `docs/results/` are CSVs, JSONs, MDs, and small PNGs only.

---

## 5. Compute discipline (free Colab edition)

- **Workhorse:** Pythia-160M fp32. 12 canonical training checkpoints per size.
- **Stretch:** Pythia-410M fp32 (CPU OK at 256 tokens), 1B fp16 single-sequence.
- **Datasets:** WikiText-103 (general), MBPP (code), GSM8K (math). Slices
  tokenized once, cached as `.pt` files.
- **Hard ceiling:** if a single experiment cell needs >4h of T4 time, it's
  mis-designed.
- **Caching is mandatory.** `metric_battery()` is the cache layer; reads are
  ms-scale. If anything breaks that property, that's a bug.

### Known gotchas

- **Pythia attention NaN.** `output_attentions=True` returns NaN in deep
  layers due to softmax overflow. Always go through
  `hybrid_arch.attention.extract_attention`.
- **`parallel_prediction_agreement` activation memory.** The batched
  implementation runs one forward on `[n_positions, n_positions + j]`. At
  n=2K on 410m this exceeds laptop RAM and the T4. Phase 2 used n=256.
- **macOS iCloud + Python 3.14.** `.venv` files get `UF_HIDDEN`; Python 3.14
  skips them. Fix: `chflags -R nohidden .venv` after every `pip install -e .`.

---

## 6. Phase status

| Phase | Status | Outcome |
|------:|--------|---------|
| 0 — Foundations | ✅ closed | Decode-from-scratch + KV cache walkthrough. |
| 1 — Measure | ✅ closed (2026-05-25) | Five metrics, NaN-free attention. |
| 2 — Patterns | ✅ closed (2026-05-25) | [Atlas](./docs/results/02_emergence_atlas.md). AUROC 0.85 on 410m; code 3.9× more parallel-safe than prose. |
| 3 — Probes & router | ✅ closed (2026-05-25) | [Probes](./docs/results/03_probes.md). 410m L12 probe AUROC 0.857; offline probe does NOT transfer to spec-decode rejection (1−top1 wins at 0.88). |
| 4 — Hybrid decoder demo | ⏭ next | See [`PHASE_4_HANDOFF.md`](./PHASE_4_HANDOFF.md). Natural parallel work — router, measurement harness, "what we'd need to scale" writeup are independent. |
| 5 — Polish & publish | pending | |

---

## 7. Reproducibility rules (non-negotiable)

- Seeds explicit at the top of every script.
- Pin model revisions by step (e.g., `step143000`), not by tag.
- Every published plot has a script that regenerates it from cached data.
- Every published number lives in a CSV or JSON under `docs/results/`.

---

## 8. How an assistant should help

- **Mohamed has internalized the Phase 0-1 material.** The teach-everything
  mode is no longer warranted at every turn; explain LLM internals when they
  *change the answer*, not by default.
- **Push back on scope creep.** Anything that requires >4h of T4 or a bigger
  model than Pythia-410m → flag and propose a smaller analogue.
- **Honor the SOTA boundary.** Don't propose "let's beat EAGLE-3." Do propose
  "let's see if our probes predict EAGLE-3 rejections."
- **Verification step on everything quantitative.** Every number that lands
  in a writeup must come from a regenerable script.
- **Autonomous execution within a defined plan.** Don't pause between steps
  of a multi-step handoff unless something's risky or genuinely ambiguous.

---

## 9. Living document policy

This file gets updated when:
- A phase closes (update §6).
- The thesis sharpens (rare).
- A SOTA change affects positioning (update §3).
- A convention turns out to be wrong (update §4).

Small fixes go straight in. Big changes get committed with a message
starting `AGENTS.md:` so they're easy to find in `git log`.
