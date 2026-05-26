# Phase 4 Handoff — Hybrid Architecture

Self-contained brief for the engineer (or assistant) picking up Phase 4.
Date authored: 2026-05-25, at the close of Phase 3.

## 0. How to use this document

Phase 3 produced two facts that anchor Phase 4:

1. The 410m middle-layer probe is a strong **offline** classifier of
   parallel-safety (AUROC 0.857).
2. The same probe is a **chance-level predictor of real spec-decode
   rejections**; the drafter's own `1 − top1` confidence hits AUROC 0.88
   on that task without any training.

Phase 4 builds the demo *that takes both facts seriously*: a hybrid
decoder that uses cheap drafter-side features for online routing, with
the offline probe as a diagnostic instrument (not the router).

Read these before writing code:

1. [`README.md`](./README.md) — public pitch.
2. [`AGENTS.md`](./AGENTS.md) — project conventions, SOTA boundary,
   compute discipline.
3. [`PROJECT_PLAN.md`](./PROJECT_PLAN.md) §Phase 4 — scope.
4. [`docs/results/02_emergence_atlas.md`](./docs/results/02_emergence_atlas.md) and [`docs/results/03_probes.md`](./docs/results/03_probes.md) — Phase 2 and 3 results in narrative form.
5. Skim `src/hybrid_arch/spec_decode.py` — the testbed Phase 4 extends.

## 1. State of the repo at Phase 3 close

```
hybrid-architecture/
├── README.md, AGENTS.md, PROJECT_PLAN.md
├── PHASE_3_HANDOFF.md, PHASE_4_HANDOFF.md
├── pyproject.toml                     ← v0.3.0, scikit-learn dep
├── docs/results/
│   ├── 02_emergence_atlas.md, 03_probes.md
│   ├── 02..08_*.{csv,manifest.json}
│   └── figures/*.png
├── src/hybrid_arch/
│   ├── attention.py, cache.py, checkpoints.py
│   ├── metrics.py, probes.py, spec_decode.py, viz.py
│   └── probe_checkpoints/    ← 42 (.pt + .json sidecar) pairs
├── src/scripts/
│   ├── phase2_emergence_curve.py
│   ├── phase2_signature_analysis.py
│   ├── phase2_token_types.py
│   ├── phase2_domain_shift.py
│   ├── phase3_layer_depth_sweep.py
│   └── phase3_drafter_rejection.py
├── tests/                             ← 92 tests, all passing
└── .github/workflows/ci.yml           ← ruff + pytest, Python 3.10/3.12
```

## 2. Phase 4 goal

From [`PROJECT_PLAN.md`](./PROJECT_PLAN.md):

> Show the probes inside a working decoder. Not to beat SOTA — to
> demonstrate the use case.

Concretely:

1. **`hybrid_decode()`** — a `transformers`-style generation function that
   uses a router to decide, per position, whether to commit a drafted
   token directly or fall through to the verifier. The router is a
   weighted vote of `1 − top1`, `entropy`, and the probe (Phase 3
   showed that the probe alone is not enough; the question is whether
   probe-on-top-of-baseline adds anything).
2. **Measurement harness** — tokens/second, perplexity vs the
   ground-truth autoregressive output, fraction of positions routed
   each way. Run on WikiText + MBPP + GSM8K from Phase 2.
3. **Honest writeup** — `docs/results/04_hybrid_decoder.md`. Lead with
   what the demo *can* do; close with the engineering work that would
   actually make this beat `.generate()` at scale (CUDA kernels,
   batched scheduling, KV-cache surgery — the parts we know we're not
   building on free Colab).

## 3. Parallel work in Phase 4

This phase has natural parallel work — three independent threads that
fan out cleanly across agents:

1. **Probe-feature router** — combine `1 − top1`, `entropy`, and the
   probe into a logistic regression *fit on real drafter-rejection
   labels*. Compare AUROC against `1 − top1` alone (the Phase 3
   baseline). Time budget: ~half a day.
2. **Measurement harness** — tokens/sec, perplexity, routing-fraction
   histograms. Independent of the router design. Time budget: ~half a
   day.
3. **"What we'd need to scale" writeup** — honest list of
   engineering work for a real productionization. Pure prose, depends
   on the other two only for numbers to cite. Time budget: ~half a day.

Suggested fan-out point: after Step 1 below produces the
`hybrid_decode` skeleton, the three threads can run concurrently.

## 4. Suggested sequence of work

**Step 1 — `hybrid_decode` skeleton** (~half a day)

- Extend `src/hybrid_arch/spec_decode.py` (or new
  `src/hybrid_arch/hybrid.py`) with a `hybrid_decode(target, drafter,
  prompt, router, draft_k, n_steps)` function. `router` is a callable
  that takes drafter features and returns a per-position routing
  decision.
- Default router: `1 − top1 < threshold` ⇒ keep drafted token;
  else ⇒ run target.
- Tests: routing decisions are exposed in the return value;
  perplexity vs pure-target output is bounded; tokens/sec ≥ pure-AR
  baseline on a 256-token slice.

**Step 2 — Fitted router** (~half a day, parallelizable)

- Run `spec_decode_capture` on a longer corpus (1024+ drafted
  positions) to get real labels.
- Fit a 3-feature logistic regression (`1 − top1`, `entropy`,
  `probe_L12_logit`) on the labels. Save coefficients to
  `docs/results/09_router_coefficients.csv`.
- Plot probe-augmented AUROC vs baseline AUROC.

**Step 3 — Measurement harness** (~half a day, parallelizable)

- `src/scripts/phase4_hybrid_decode_bench.py`. Run the hybrid decoder
  on WikiText / MBPP / GSM8K slices; compare tokens/sec to greedy
  spec-decode and to pure-target greedy. Track per-position routing
  histograms.
- Save: `docs/results/10_hybrid_decoder_bench.csv`, ROC + speedup plot.

**Step 4 — Writeup** (~half a day, parallelizable)

- `docs/results/04_hybrid_decoder.md`. Same shape as the Phase 3
  atlas: TL;DR, setup, two results, honest limitations.

**Step 5 — Closeout** (~30 min)

- Bump `__version__` to `"0.4.0"`.
- Pin one figure on the README (probably the speedup-vs-routing plot).
- Write `PHASE_5_HANDOFF.md` for the polish-and-publish phase.

## 5. Compute discipline

- The hybrid decoder runs Pythia-1b on CPU for the benchmark. Each
  256-token slice is ~5 minutes per dataset.
- Probe inference is ~0.1ms per position; not the bottleneck.
- Hard ceiling: 4h of T4 time per experiment; if you blow it, mis-design.

## 6. Anti-patterns to avoid

- **Trying to beat `.generate()` on raw throughput.** You won't, and that's
  fine. The demo is the point.
- **Re-doing Phase 2 or 3.** The cache + probes are committed. Reading them
  is the workflow.
- **Reporting wall-clock numbers without a perplexity check.** The router
  could trivially "speed up" by passing all drafted tokens through,
  catastrophically wrecking perplexity. Always report both axes together.

## 7. Where to start

Brief for the engineer or assistant kicking off Phase 4:

> Read `PHASE_4_HANDOFF.md`, `docs/results/03_probes.md`, and
> `PROJECT_PLAN.md` §Phase 4. The cache, probe library, and the spec-decode
> testbed are all shipped — *do not rebuild them*. Start with §4 Step 1:
> the `hybrid_decode` skeleton. Once that lands, the next three steps
> (fitted router, measurement harness, writeup) can fan out across agents.
