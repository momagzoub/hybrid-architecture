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
└── tests/                         ← 53 tests, all passing
    ├── conftest.py                ← shared pythia fixture
    ├── test_attention.py
    ├── test_metrics_{logits,attention,parallel}.py
    ├── test_smoke.py
    └── test_viz.py
```

What does *not* exist yet:

- `src/hybrid_arch/checkpoints.py` — the cross-checkpoint helper. Phase 2 builds it.
- A batched, fast version of `parallel_prediction_agreement`. See §5.
- Any of the cross-checkpoint experiments, plots, or the atlas.
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

These three pieces of context must stay live across Phase 2:

1. **The Pythia attention NaN bug is solved, but only via the forward-hook path.** All attention-based metrics must go through `hybrid_arch.attention.extract_attention`, never `output_attentions=True`. The deep-layer overflow has been verified on every layer 0-11 of step143000.
2. **Position-0 (and early-position) entropy is structural.** Causal masking forces row 0 of attention to be a delta, so its entropy is 0 by construction, not by model behavior. For aggregate metrics in Phase 2, decide explicitly whether to drop these positions, log them separately, or report them — and document the choice in the atlas writeup. The Phase 1 `03_metric_zoo.ipynb` drops position 0 from the correlation, as one example.
3. **The j=0 column of `parallel_prediction_agreement` is always True.** It's a structural invariant (TF and AR see identical context), not a property of the model. If you report a single agreement rate per position, restrict the mean to `j > 0` or document that you're including the structurally-True column.

## 5. The big performance problem and how to fix it

`parallel_prediction_agreement` in Phase 1 is `O(n_positions × k)` sequential CPU forward passes. On 256 tokens × k=4 that's ~3-5 min on a T4. On Phase 2 scale (5000 tokens × 36 model loads × 3 datasets) the same implementation would take ~70 hours, blowing the compute budget by an order of magnitude.

**Fix before launching the big experiment.** The batched version:

- The j=0 column is structural — skip the model call, just copy TF predictions.
- For j ≥ 1, batch all `n_positions` rollouts into one forward pass per step: build a padded `[n_positions, max_len]` tensor, run one model call, take argmax at each row's last unpadded position. With padding-aware attention masks, this collapses `n_positions × (k-1)` calls into `(k-1)` batched calls. Roughly **100× speedup** on long prompts.
- Keep `parallel_prediction_agreement` as the user-facing function; add a `batched=True` flag or a separate `parallel_prediction_agreement_batched` if the API divergence is too much.

The tests in `tests/test_metrics_parallel.py` already pin down the correct behavior — make the batched version pass them.

## 6. Suggested sequence of work

Each step ends in something runnable. Don't move on until the prior step is green.

**Step 1 — Checkpoint loader (~1 day)**

- File: `src/hybrid_arch/checkpoints.py`.
- Function: `load_pythia(size, step) → (model, tokenizer)`. Pins by revision string `stepNNNN`. Caches to disk so re-runs are free.
- Function: `list_checkpoints(size)` → list of canonical step values to evaluate (e.g., `[0, 1000, 4000, 16000, 32000, 64000, 96000, 128000, 143000]`). Document why these were chosen.
- Test: load one early and one late checkpoint, assert a meaningful difference on a sanity prompt (e.g., entropy at step 0 ≫ entropy at step 143000).

**Step 2 — Batched parallel_prediction_agreement (~half day)**

- Replace or extend the function in `metrics.py`. Existing tests must still pass.
- Add a Pythia integration test on a 100-token prompt: the batched and sequential versions must agree element-wise.
- Benchmark: report the speedup factor in the test or a comment.

**Step 3 — Metric-battery harness (~1 day)**

- Script (under `src/scripts/run_metric_battery.py` — make the `scripts/` dir): given `(size, step, dataset_slice_path)`, run all metrics, save per-token CSVs to `data/metric_battery/<size>/<step>/<dataset>.csv` plus a `manifest.json`. Skip work that already has cached outputs.

**Step 4 — Run the battery (~3-5 hours of T4 time)**

- 3 sizes × ~12 checkpoints × 3 datasets = ~108 runs. Each run produces one CSV. With caching, re-runs are free.
- After this step, ~108 CSV files exist on disk and are recoverable from manifest.

**Step 5 — Aggregation and plots (~2-3 days)**

- A separate notebook `notebooks/04_emergence_atlas.ipynb` reads from cached CSVs, generates the four headline plots (emergence curve, signature accuracy, token-type breakdown, domain shift).
- Use the `STYLE` dict in `viz.py`. Add new plot helpers (e.g., `emergence_curve`) to `viz.py` if reused.

**Step 6 — Writeup (~1-2 days)**

- `docs/results/02_emergence_atlas.md` — the headline document. One paragraph per plot, with the plot embedded. Include captions you'd be willing to staple to a workshop submission.

**Step 7 — Closeout**

- Update `PROJECT_PLAN.md` §Phase 2 status. Update `CLAUDE.md` §8. Write `PHASE_3_HANDOFF.md`.

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

> Read `PHASE_2_HANDOFF.md` and `CLAUDE.md`. Then start with §6 Step 1 — build `src/hybrid_arch/checkpoints.py` with `load_pythia` and `list_checkpoints`, including the sanity-check test. Run continuously through the steps; only stop for genuinely risky or ambiguous decisions.

End of handoff. When Phase 2 is done, write a `PHASE_3_HANDOFF.md` in the same shape.
