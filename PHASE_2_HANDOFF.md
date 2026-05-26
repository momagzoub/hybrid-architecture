# Phase 2 Handoff — Hybrid Architecture

Paste-this-to-Claude-Code document. Self-contained: assume the reader is a fresh Claude Code session that has not seen Phase 1. Date authored: 2026-05-25, at the close of Phase 1.

## 0. How to use this document

You are a Claude Code session joining the Hybrid Architecture project. Your job is to execute Phase 2: the *Patterns* phase — running the Phase 1 metric battery across Pythia training checkpoints and producing the developmental atlas that is the project's headline contribution.

Read these before writing any code:

1. `CLAUDE.md` — project thesis, conventions, SOTA boundary, compute discipline.
2. `PROJECT_PLAN.md` §Phase 2 — canonical scope.
3. `PHASE_1_HANDOFF.md` and `docs/concepts/04_metrics.md` — what the metrics are and what Mohamed has internalized about them.
4. Skim `src/hybrid_arch/metrics.py`, `attention.py`, `viz.py` — the library you're going to be using.

## 1. Who Mohamed is and how to work with him

- MIT, GitHub `momagzoub`. Owner of the project, working solo.
- Some ML background, no LLM internals before this project. After Phase 1, he can: write an autoregressive decode loop unaided, reason about teacher-forcing vs autoregressive prediction, extract NaN-free attention via forward hooks, and apply the metric library to a model + dataset slice.
- Free Colab only — 16GB T4 when available, otherwise CPU. Phase 2's compute budget is real: see §3.
- Teach-as-you-go, scaffold-before-answer. Both still apply.
- He has explicitly opted into **autonomous execution** for multi-step plans — don't stop between steps unless something's risky or genuinely ambiguous. See [`memory/feedback_autonomous_execution.md`](/Users/mohamedmagzoub/.claude/projects/-Users-mohamedmagzoub-Documents-Claude-Projects-Hybrid-Architecture/memory/feedback_autonomous_execution.md).

## 2. State of the repo as of Phase 1 close

```
Hybrid Architecture/
├── CLAUDE.md, PROJECT_PLAN.md, README.md, GITHUB_GUIDE.md
├── PHASE_1_HANDOFF.md             ← Phase 1's brief, now historical
├── PHASE_2_HANDOFF.md             ← this file
├── pyproject.toml                 ← installable as `pip install -e ".[dev]"`
├── docs/
│   ├── reading_list.md
│   ├── concepts/
│   │   ├── 01_decoding_basics.md, 02_kv_cache.md, 03_attention.md   ← Phase 0
│   │   └── 04_metrics.md          ← Phase 1 — SCAFFOLD; Mohamed fills it in
│   └── results/
│       ├── 01_metric_correlations.csv             ← Phase 1 deliverable
│       └── 01_metric_correlations.manifest.json
├── notebooks/
│   ├── 00..02_*.ipynb             ← Phase 0
│   └── 03_metric_zoo.ipynb        ← Phase 1 demo
├── src/hybrid_arch/
│   ├── __init__.py                ← exposes the public API
│   ├── attention.py               ← extract_attention (forward-hook, NaN-free)
│   ├── metrics.py                 ← five per-token metrics
│   └── viz.py                     ← entropy_heatmap, attention_track, STYLE dict
└── tests/                         ← 63 tests, all passing
    ├── conftest.py                ← shared pythia fixture
    ├── test_attention.py          ← 5 tests
    ├── test_checkpoints.py        ← 7 tests
    ├── test_metrics_attention.py  ← 13 tests
    ├── test_metrics_logits.py     ← 14 tests
    ├── test_metrics_parallel.py   ← 13 tests
    ├── test_smoke.py              ← 2 tests
    └── test_viz.py                ← 9 tests
```

What's already in place that this doc previously listed as TO BUILD:

- ✅ `src/hybrid_arch/checkpoints.py` exists with `list_checkpoints()` and `load_pythia(size, step)`. Canonical step list is log-spaced: `[0, 1, 8, 128, 1000, 4000, 16000, 32000, 64000, 96000, 128000, 143000]`.
- ✅ `parallel_prediction_agreement` already has a `batched=True` flag (default). The performance trap mentioned in §5 was anticipated and fixed before Phase 2 began.
- ✅ `notebooks/04_emergence_atlas.ipynb` exists as an empty scaffold, waiting to be filled.

What still does *not* exist:

- The per-(layer, head) attention metrics primitive (Phase 1 only shipped aggregate versions; see §4 for why this matters).
- The metric-battery cache layer (`src/hybrid_arch/cache.py` or equivalent). Without this, Phase 2's compute budget overflows.
- Any of the cross-checkpoint experiments, plots, or the atlas writeup.
- `docs/results/figures/`.
- `docs/results/02_emergence_atlas.md`.

## 3. Phase 2 goal

From `PROJECT_PLAN.md`:

Run the Phase 1 metric battery across Pythia training checkpoints (sizes 70M, 160M, 410M, ~12 checkpoints each), on three datasets (WikiText, MBPP, GSM8K), and produce four headline analyses:

1. **The emergence curve.** For each (size, checkpoint) plot the fraction of tokens with `parallel_agreement ≥ 0.9`. Does parallel-safety emerge smoothly, in phases, or at a critical model size?
2. **The signature analysis.** Predict per-token parallel-safety from the other four metrics. Track the probe's accuracy across checkpoints.
3. **Token-type breakdown.** Categorize tokens (function words, content words, code syntax, numbers, names) and show how parallel-safety differs by category and evolves.
4. **Domain shift.** Same battery on WikiText / MBPP / GSM8K. Do parallel-safety signatures generalize across domains?

Deliverable: `docs/results/02_emergence_atlas.md` with 5-6 publication-quality plots in `docs/results/figures/`.

## 4. Critical inheritance from Phase 1

These four pieces of context must stay live across Phase 2:

1. **The Pythia attention NaN bug is solved, but only via the forward-hook path.** All attention-based metrics must go through `hybrid_arch.attention.extract_attention`, never `output_attentions=True`. The deep-layer overflow has been verified on every layer 0-11 of step143000.
2. **Position-0 (and early-position) entropy is structural.** Causal masking forces row 0 of attention to be a delta, so its entropy is 0 by construction, not by model behavior. For aggregate metrics in Phase 2, decide explicitly whether to drop these positions, log them separately, or report them — and document the choice in the atlas writeup. The Phase 1 `03_metric_zoo.ipynb` drops position 0 from the correlation, as one example.
3. **The j=0 column of `parallel_prediction_agreement` is always True.** It's a structural invariant (TF and AR see identical context), not a property of the model. If you report a single agreement rate per position, restrict the mean to `j > 0` or document that you're including the structurally-True column.
4. **Aggregate attention metrics correlate at \|r\| < 0.11 with parallel-safety.** This is the puzzle that defines Phase 2's most important experiment. The Phase 1 correlation analysis (256-token WikiText slice, Pythia-160M @ step143000) found that *averaging attention entropy/concentration across all 12 layers × 12 heads washes out any signal*. The thesis-relevant hypothesis: **specific (layer, head) pairs are highly predictive of parallel-safety, but the mean across them is noise.** The signature analysis (Step 5 below) tests this directly — and is the experiment whose result determines whether Phase 3's probe has anything to predict. If aggregate attention is really noise and per-head attention is also noise, the thesis is in trouble; budget time to dig into this carefully.

## 5. Performance — already fixed, plus the next bottleneck

The previous version of this handoff flagged `parallel_prediction_agreement` as a major performance problem (sequential rollouts would blow the budget at Phase 2 scale). **That's been fixed**: the function now defaults to `batched=True` and runs in `O(k)` batched forward passes instead of `O(n_positions × k)` sequential ones. `tests/test_metrics_parallel.py` includes a batched-vs-sequential equivalence test.

The new bottleneck at Phase 2 scale is **model loading and metric-battery caching**, not parallel-agreement compute. With 36 model loads × 3 datasets = 108 metric-battery runs and each load taking 30-60s on its own, *not caching the metric outputs is what would blow the budget*. Step 1 below builds that cache layer; it is non-negotiable.

## 6. Suggested sequence of work

Each step ends in something runnable. The first two steps from the previous version of this handoff are already done (`checkpoints.py` and batched `parallel_prediction_agreement`); the sequence below picks up from there. Don't move on until the prior step is green.

**Live progress (this session):**

- [x] Step 1 — `src/hybrid_arch/cache.py` + 10 cache tests (74 → 81 tests passing)
- [x] Step 2 — per-(layer, head) attention already preserved by Phase 1 primitives; added `aggregate_attention_*` helpers + 7 explicit equivalence tests
- [x] Step 3 — smoke verified on Pythia-70m and 410m @ step143000, 32 tokens, cache hit < 1ms
- [x] Step 4 — emergence curve sweep complete; `docs/results/02_emergence_curve.{csv,manifest.json,figures/02_emergence_curve.png}`. Headline: parallel-safety emerges between step128 and step1000 across all sizes; 410m converges to psf≈0.115 vs 70m ≈0.05 (mild size scaling). 410m@step8 has a psf=0.405 outlier — pre-collapse degenerate prediction (model emits the same token everywhere); flag in writeup.
- [ ] Step 5 — signature analysis: in progress
- [ ] Steps 6-9 — pending

**Slice-size deviation from the original plan.** The handoff calls for a 2K-token WikiText slice, but the batched `parallel_prediction_agreement` at n_positions ≈ 2K would create activations of ~25 GB per layer on Pythia-410M (the kernel is one big `[n_positions, n_positions+j]` forward pass). That exceeds laptop RAM and is over the T4's 16 GB. Stepped down to 256 tokens to match the Phase 1 slice; statistical power per cell is 252 × (k-1) = 756 binary observations, ample for the fraction-≥-0.9 statistic. Document this in the atlas; revisit at 1K on T4 if the laptop sweep finishes with budget headroom.

**Step 1 — Metric-battery cache layer (~3-4 hours)**

- File: `src/hybrid_arch/cache.py`.
- Function: `metric_battery(model_size, step, dataset_name, slice_hash, *, force_recompute=False) -> dict[str, Tensor]`. Loads cached outputs from disk if present; otherwise loads the model via `load_pythia`, runs the full metric battery, writes outputs, returns.
- Storage layout: `data/cache/<dataset>/<size>/<step>.npz` (named-array format keeps each metric retrievable independently). Add `data/cache/` to `.gitignore` if not already excluded by the `data/` rule.
- Tests: cache hit returns identical tensor to cache miss; `force_recompute=True` recomputes; missing cache directory is created automatically; manifest sidecar JSON records model revision, dataset slice hash, tokenizer hash.
- This step is **the single most important piece of Phase 2 engineering**. If caching is slow or has correctness bugs, every subsequent step pays the cost.

**Step 2 — Per-(layer, head) attention metrics (~2-3 hours)**

- Phase 1's `attention_entropy` and `attention_concentration` return shapes that aggregate across layers/heads (returning `[batch, seq]`). The signature analysis in Step 5 needs them at the full `[layers, heads, batch, seq]` granularity.
- Either add a `aggregate=False` flag to the existing functions, or create new `attention_entropy_per_head(...)` / `attention_concentration_per_head(...)` functions. Choose whichever doesn't break existing tests.
- Tests: shape correctness; `mean(per_head, dim=(0,1)) ≈ aggregate` to within float tolerance; no NaN in deep layers (already guaranteed by `extract_attention`).
- This is the **missing primitive that lets the signature analysis access the signal that aggregating washes out**.

**Step 3 — Smoke test the cache + per-head metrics at scale (~1 hour + compute)**

- Quick verification: load Pythia-70m at step143000, run the full per-token + per-(layer, head) metric battery on a 32-token slice, cache it, re-load from cache, confirm identical results. Then do the same for 410m to verify the cache layout handles the larger tensors.
- No new code; just exercise Steps 1-2 end-to-end.

**Step 4 — Experiment 1: emergence curve (~half a day + compute)**

- Run the metric battery across 12 canonical checkpoints × 3 model sizes (70m, 160m, 410m) on a 2K-token WikiText-103 slice (deterministic seed, hash documented in manifest). Use Step 1's cache.
- Notebook: fill in `notebooks/04_emergence_atlas.ipynb`.
- Output: per-checkpoint table of `parallel_safety_fraction` (fraction of positions where `mean(agreement[:, j=1:k]) >= 0.9`). Plot as 3 lines (one per model size) against training step on a log-x axis. Save the table to `docs/results/02_emergence_curve.csv` with manifest sidecar.

**Step 5 — Experiment 2: signature analysis (~1 day + compute)**

- For each (model_size, checkpoint), train a logistic regression: features = per-token logit-side metrics (entropy, top1) + per-(layer, head) attention metrics (12×12 = 144 features for Pythia-160M); target = binary `parallel_safety` label.
- Track classifier AUROC across checkpoints. Plot AUROC vs training step per model size.
- Extract feature importances from the final-checkpoint classifier. Identify the top-10 most predictive (layer, head) pairs. Plot as a small bar chart. *This is the experiment that answers the Phase 1 puzzle (§4 item 4).*
- Save: per-checkpoint AUROC in `docs/results/03_signature_auroc.csv`; top features in `docs/results/04_top_features.csv`.
- Use scikit-learn (`LogisticRegression` with `class_weight="balanced"`); don't overthink the modeling. The methodological pitfalls to watch for are documented in the Belinkov 2022 probing paper — read it before this step.

**Step 6 — Experiment 3: token-type breakdown (~half a day)**

- Categorize tokens using either spaCy POS tags or a regex/closed-list classifier (function words via closed list; content words via POS; numbers via regex). Keep it simple.
- For each token type, compute `parallel_safety_fraction` per checkpoint. Plot as a small-multiples figure (one panel per token type).
- No new compute. Reads from Step 4's cache.

**Step 7 — Experiment 4: domain shift (~half a day + compute)**

- Re-run the metric battery on MBPP (code) and a 2K-token slice of GSM8K (math). To stay within budget, do this only at the *final checkpoint* per model size (3 sizes × 2 new datasets = 6 new runs).
- Output: a 3×3 heatmap (model_size × domain → mean parallel-safety fraction), plus a per-domain emergence curve overlay if the time budget allows.

**Step 8 — Writeup (~1-2 days)**

- File: `docs/results/02_emergence_atlas.md`. Lead with the emergence-curve figure (it's the headline); then signature analysis, token-type, domain. Each section is one figure + one paragraph of interpretation. Pull figures into `docs/results/figures/` with consistent filenames.
- Audience: an inference researcher who hasn't read CLAUDE.md. They should understand the contribution in five minutes.

**Step 9 — Closeout (~30 min)**

- Update `PROJECT_PLAN.md` §Phase 2 status with a one-paragraph outcome summary (mirror the Phase 1 outcome paragraph).
- Update README.md hero section: pin the emergence-curve figure.
- Bump `__version__` to `"0.2.0"`.
- Write `PHASE_3_HANDOFF.md` for the probe-and-router work.

## 7. Compute discipline (reminder)

- Workhorse: Pythia-160M fp32 on T4 → ~50 ms per forward pass on short prompts.
- Stretch: Pythia-410M fp32 batch=1, or Pythia-1B fp16 single-sequence.
- Hard ceiling: if a single experiment cell needs >4 h of T4 time, it's mis-designed.
- Cache by `(model_size, step, dataset_slice_hash)` tuple. The dataset slice hash already exists in the Phase 1 manifest format — reuse it.

## 8. Reading-list pointers (start with these)

- *When Attention Sink Emerges* (ICLR 2025). The sink question is now empirical — Phase 2 should chart when the sink emerges in Pythia and whether it correlates with the parallel-safety transition.
- *Probing Classifiers: Promises, Shortcomings, and Advances* (Belinkov 2022). The signature analysis is a probe, so the methodological pitfalls in this paper apply.
- *Fast Inference from Transformers via Speculative Decoding* (Leviathan et al., 2022) §3. Pre-reading for Phase 3 but worth keeping in mind during Phase 2 — `parallel_agreement` is its offline cousin.

Full list in `docs/reading_list.md`.

## 9. Anti-patterns to avoid (Phase 1 incident report)

- **Asking "go" between every step.** Mohamed wants autonomous execution within a defined plan. See [feedback_autonomous_execution](memory).
- **Trusting the framework's attention.** Both eager and SDPA paths return NaN in Pythia deep layers. Always use `extract_attention`.
- **Reporting unverified numbers.** Phase 0 had one slip on this; Phase 1 didn't repeat it. Maintain the streak — every number in the atlas writeup must be regenerable from a script.
- **Letting `parallel_prediction_agreement` run sequentially at Phase 2 scale.** The performance trap is real and predicted. Batch it first.
- **Stripping notebook outputs.** Phase 0 and Phase 1 notebooks commit outputs intentionally — they document the run. Don't add a pre-commit hook that strips them.

## 10. Where to start

Open a fresh chat with Claude Code, point at this file, and say:

> Read `PHASE_2_HANDOFF.md`, `CLAUDE.md`, and `PROJECT_PLAN.md` §Phase 2. Note that `src/hybrid_arch/checkpoints.py` and the batched `parallel_prediction_agreement` are already shipped — *do not rebuild them*. Start with §6 Step 1: build `src/hybrid_arch/cache.py` with the `metric_battery()` cache function and its tests. Run continuously through the steps in order; only stop if you hit something genuinely risky or ambiguous, or if a step's compute estimate climbs above 4 hours (the CLAUDE.md §7 ceiling) — in those cases, flag for review before proceeding.

The "do not rebuild what already exists" sentence is load-bearing — the previous version of this handoff (before Phase 1 closed) described Steps 1 and 2 of the old sequence as work to do; that work is now done. Skipping it is the difference between Phase 2 finishing in 2-3 weeks and looping for a month on already-solved problems.

End of handoff. When Phase 2 is done, write a `PHASE_3_HANDOFF.md` in the same shape, anchoring on the signature-analysis output (Step 5) as the artifact that Phase 3's probe consumes.
