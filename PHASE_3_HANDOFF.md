# Phase 3 Handoff — Hybrid Architecture

Self-contained brief for the engineer (or assistant) picking up Phase 3.
Date authored: 2026-05-25, at the close of Phase 2.

## 0. How to use this document

Phase 2 produced the empirical claim that specific (layer, head) attention
features predict "parallel-safety" with AUROC ~0.85 on Pythia-410m using a
linear model. Phase 3 turns that observation into a *callable* artifact: a
small, pretrained probe an inference engineer can plug into a drafter as a
pre-verification step.

Read these before writing any code:

1. [`README.md`](./README.md) — the public pitch.
2. [`AGENTS.md`](./AGENTS.md) — project conventions, SOTA boundary, compute discipline.
3. [`PROJECT_PLAN.md`](./PROJECT_PLAN.md) §Phase 3 — canonical scope.
4. [`docs/results/02_emergence_atlas.md`](./docs/results/02_emergence_atlas.md) — Phase 2 results in narrative form. **The probe consumes the artifacts described here.**
5. Skim `src/hybrid_arch/{cache,metrics,attention}.py` and `src/scripts/phase2_signature_analysis.py` — the API and the script the probe will displace.

## 1. State of the repo at Phase 2 close

```
hybrid-architecture/
├── README.md, AGENTS.md, PROJECT_PLAN.md
├── PHASE_3_HANDOFF.md             ← this file
├── pyproject.toml                 ← installable as `pip install -e ".[dev]"`
├── docs/results/
│   ├── 02_emergence_atlas.md      ← Phase 2 atlas
│   ├── 02_emergence_curve.{csv,manifest.json}
│   ├── 02_signature_analysis.manifest.json
│   ├── 03_signature_auroc.csv
│   ├── 04_top_features.csv
│   ├── 05_token_type_breakdown.{csv,manifest.json}
│   ├── 06_domain_shift.{csv,manifest.json}
│   └── figures/02_emergence_curve.png, 03_signature_auroc.png,
│                04_top_features.png, 05_token_type_breakdown.png,
│                06_domain_shift_heatmap.png
├── src/hybrid_arch/
│   ├── attention.py               ← extract_attention (NaN-free)
│   ├── cache.py                   ← metric_battery + slice_hash
│   ├── checkpoints.py             ← list_checkpoints, load_pythia
│   ├── metrics.py                 ← 5 per-token metrics + aggregate helpers
│   └── viz.py                     ← STYLE, entropy_heatmap, attention_track
├── src/scripts/
│   ├── phase2_emergence_curve.py
│   ├── phase2_signature_analysis.py
│   ├── phase2_token_types.py
│   ├── phase2_domain_shift.py
│   └── run_metric_battery.py
└── tests/                         ← 81 tests, all passing
```

`data/cache/` (gitignored) holds 42 materialized metric-battery outputs covering
the full Phase 2 grid; Phase 3 reads them with no model load.

## 2. Phase 3 goal

From [`PROJECT_PLAN.md`](./PROJECT_PLAN.md):

1. **`src/hybrid_arch/probes.py`** — a 3-layer MLP probe (~10k params) that takes
   a hidden state from layer `L` at position `t` and predicts whether position
   `t+1` is parallel-safe. Train one probe per (model_size, layer) pair. Plot
   accuracy as a function of layer depth — at what point in the network does
   parallel-safety become predictable?
2. **The drafter-rejection prediction experiment.** Use HuggingFace's
   `assistant_model` speculative decoding on Pythia-1B with Pythia-160M as
   drafter. Log every accept/reject event. Test: does the Phase 2 entropy metric
   predict rejections? Does the Phase 3 probe predict them better? Report ROC AUC.
3. **The "what does the router look at" experiment.** Interpret probe weights to
   identify which features (attention entropy at head X, top-1 prob, etc.)
   matter most. Compare with the linear top-10 in `04_top_features.csv`.

Deliverable: `docs/results/03_probes.md` (probe accuracy curves, rejection-prediction
ROC, feature importance) plus a pretrained probe checkpoint committed to
`src/hybrid_arch/probe_checkpoints/`.

## 3. Critical inheritance from Phase 2

These pieces of context **must stay live across Phase 3**:

1. **The cache layer is mandatory.** Every Phase 2 experiment runs in ms because
   `metric_battery()` short-circuits before `load_pythia`. Phase 3's probe
   trainer should consume cached metric outputs directly, not re-extract
   attention. If you need a feature the cache doesn't have, add it to
   `_compute_battery` in `cache.py` and re-run the sweep.
2. **The agreement-vs-quality artifact at Pythia-410m@step8.** psf = 0.405 and
   AUROC = 0.569 because the undertrained model predicts the same token
   everywhere. Drop this cell before fitting probes or it will bias the layer
   depth analysis.
3. **The per-head primitives, not aggregates.** Phase 1 aggregated
   `attention_entropy` across (L, H) and got `|r| < 0.11`. Phase 2's signature
   analysis broke them apart and got AUROC 0.85. Use `attention_entropy_per_head`
   shapes, never `aggregate_attention_*` — the latter is kept only for the
   atlas's comparison plot.
4. **j=0 is structural.** `parallel_agreement[:, :, 0]` is always True (TF and
   AR start from identical context). Always compute the label from
   `mean(agreement[:, :, 1:k]) >= θ` and drop position 0 from features (its
   attention row is a delta by causal mask).
5. **Slice size is 256, not 2K.** Memory blocked the 2K plan. If a T4 is
   available, retry at 1K before going to 2K — the activation memory in
   batched `parallel_prediction_agreement` scales quadratically with n_positions.

## 4. Suggested sequence of work

Each step ends in something runnable.

**Step 1 — Hidden-state extraction (~2-3 hours)**

The Phase 2 cache stores per-(layer, head) *attention* features but not raw
hidden states. The probe takes a hidden state as input.

- Add `extract_hidden_states(model, input_ids) -> Tensor[L, B, S, H]` to
  `src/hybrid_arch/attention.py` (or a new `hidden.py`). Use a forward hook on
  each `GPTNeoXLayer`, capture the post-LN output. Verify NaN-free for all
  layers/checkpoints actually exercised in Phase 2.
- Extend `_compute_battery` in `cache.py` to also save `hidden_states` per
  layer. The `.npz` filename stays the same; the additional array key is
  `hidden_states_per_layer` with shape `[L, B, S, H]` (fp16 to keep file size
  reasonable — 410m × 256 × 1024 × 24 × 2B ≈ 12 MB per cell).
- Re-run the Phase 2 sweep with `force_recompute=True` to populate hidden
  states for all 42 cells. Budget: ~2 hr (same compute envelope as Phase 2
  Step 4).

**Step 2 — Probe model + trainer (~1 day)**

- `src/hybrid_arch/probes.py`: a `LayerProbe(nn.Module)` with a 3-layer MLP
  (e.g., `hidden_dim → 64 → 32 → 1`), `nn.LayerNorm` on the input, sigmoid
  output. Total params well under 100k for `hidden_dim=1024`.
- `train_probe(hidden, labels, n_epochs=200, lr=1e-3, weight_decay=1e-3) ->
  LayerProbe`. Adam, BCE loss with `pos_weight` to handle class imbalance,
  early stopping on a held-out 20% split.
- Tests: probe trains to ~AUROC 0.7+ on a synthetic noisy-XOR task in <5 s.
- Reproducible save/load via `state_dict()` + a JSON sidecar (input_dim,
  model_size, layer_index).

**Step 3 — Layer-depth sweep (~2-3 hours + compute)**

- For each `(size, layer)` in `{70m, 160m, 410m} × range(num_layers)`, train a
  probe on the final-checkpoint cache. 5-fold CV ROC-AUC, same protocol as
  Phase 2's signature analysis.
- Plot AUROC vs layer depth, one line per size. Save table to
  `docs/results/07_probe_layer_depth.csv`.

**Step 4 — Drafter-rejection experiment (~half a day + compute)**

- Use `transformers.AutoModelForCausalLM.generate(assistant_model=...)` with
  Pythia-1B target / Pythia-160M drafter on 500 tokens of WikiText.
  Capture per-position accept/reject via the `output_scores` + assistant
  hooks (transformers >=4.40 exposes the data needed; verify before assuming).
- Three baselines on the same positions:
  - random
  - `next_token_entropy` thresholded
  - the Phase 2 linear logistic regression
- Phase 3 contender: the layer-`L` probe (pick `L` from the layer-depth sweep).
- Plot ROC, report AUC. Save to `docs/results/08_drafter_rejection.csv`.

**Step 5 — Writeup (~1 day)**

- `docs/results/03_probes.md`. Same structure as the atlas: TL;DR, setup,
  three results, what Phase 4 inherits.
- Headline plot: probe AUROC vs drafter-rejection AUROC.

**Step 6 — Closeout**

- Bump `__version__` to `"0.3.0"`.
- Commit the trained probe checkpoint at `src/hybrid_arch/probe_checkpoints/<size>_L<layer>.pt`
  (a few hundred KB per probe — fits in git).
- Write `PHASE_4_HANDOFF.md` for the hybrid decoder demo. Note Phase 4 has
  natural parallel work — feature-importance write-up, the measurement
  harness (tokens/sec, perplexity), and the "what we'd need to scale"
  writeup are mostly independent and would fan out cleanly across agents.

## 5. Compute discipline (reminder)

- Workhorse: Pythia-160m fp32 on T4 → ~50 ms per forward pass on short prompts.
  Probe training itself is CPU-fine (small MLP, few thousand examples).
- Hard ceiling: if a single experiment cell needs >4 h of T4 time, it's
  mis-designed.
- Phase 2's cache layer extends here: every probe-input matrix is keyed by
  `(model_size, step, dataset_slice_hash, layer)` and re-derivable.

## 6. Anti-patterns to avoid

- **Recomputing attention from scratch.** Phase 2's cache files include
  per-(layer, head) attention metrics. Phase 3 should add hidden states to
  the cache, not re-extract attention.
- **Training one probe per (size, layer, step).** That's 3 × ~12 × 12 = 432
  probes. Final-checkpoint only for the layer-depth sweep; reserve the
  cross-checkpoint sweep for the writeup's "when does the probe become
  trainable?" footnote if time permits.
- **Reporting drafter-rejection AUC without a `next_token_entropy` baseline.**
  Phase 2 specifically left this open — the comparison is the publishable
  number.
- **Letting the 410m@step8 outlier enter probe training.** It's the one
  agreement-vs-quality cell in the cache; filter on `step >= 128`.

## 7. Reading list pointers

- Belinkov 2022 — *Probing Classifiers: Promises, Shortcomings, and Advances*.
  The probe is, well, a probe — read the methodological pitfalls before fitting.
- *Fast Inference from Transformers via Speculative Decoding* (Leviathan et al.,
  2022) §3 — the accept/reject mathematics you'll instrument in Step 4.
- *EAGLE-3* ([arXiv:2503.01840](https://arxiv.org/html/2503.01840v1)) — current
  SOTA acceptance rates on production models. Useful as a reference point;
  don't try to beat it on Pythia.

## 8. Where to start

Brief for the engineer or assistant kicking off Phase 3:

> Read `PHASE_3_HANDOFF.md`, `docs/results/02_emergence_atlas.md`, and
> `PROJECT_PLAN.md` §Phase 3. The Phase 2 cache, metric library, and 256-token
> slice are already shipped — *do not rebuild them*. Start with §4 Step 1:
> extend `cache.py` to materialize per-layer hidden states. Run continuously
> through the steps in order; only stop if a step's compute estimate climbs
> above 4 hours or you hit something ambiguous.
