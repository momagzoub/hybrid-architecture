# Phase 1 Handoff — Hybrid Architecture

> Paste-this-to-Claude-Code document. Self-contained: assume the reader has just opened the repo for the first time.
> Date authored: 2026-05-25. Authored at the close of Phase 0.

---

## 0. How to use this document

You are a Claude Code session joining the Hybrid Architecture project. Your job is to execute **Phase 1: build the metric library**. This document tells you everything you need to start. Treat it as your top-level task brief; treat the other repo docs (`CLAUDE.md`, `PROJECT_PLAN.md`, `docs/`) as authoritative reference.

**Read these three things before writing any code:**
1. `CLAUDE.md` — project thesis, conventions, SOTA boundary, compute discipline.
2. `PROJECT_PLAN.md` §Phase 1 — the canonical scope of what we're building.
3. `docs/concepts/01_decoding_basics.md`, `02_kv_cache.md`, `03_attention.md` — what Mohamed (the project owner) has internalized so far. Calibrates how to explain things.

If you skip those, you'll repeat work or talk past Mohamed's current level.

---

## 1. Who Mohamed is and how to work with him

- **MIT, GitHub `momagzoub`.** Owner of the project, working solo.
- **Some ML background, no LLM internals before this project.** Phase 0 (which is done) was where he learned autoregressive decoding from scratch, KV caching, attention extraction. He can now write a decode loop unaided and reason about prefill/decode, softmax, attention-sink, and basic metric design.
- **Free Colab only.** Often CPU runtime, sometimes T4 when available. Design everything to fit a free T4 (16 GB) at the upper bound, and to run *something useful* on CPU at the lower bound.
- **First-time GitHub user.** The repo is local; he hasn't pushed it anywhere yet. If git becomes relevant, walk through commands (don't just run them); see `GITHUB_GUIDE.md` for the conventions.
- **Teach-as-we-go.** When you explain a new concept (e.g., what entropy is, what teacher-forced means), give the conceptual ground-up version inline, not just code.
- **Scaffold before answering.** This is the single most important working preference. When Mohamed says "I don't know how to start" or submits a half-formed answer, **don't write the polished version for him**. Invite his attempt, then give the smallest hint that gets him unstuck, then iterate. He learns from guided practice; reading polished material gives him false confidence without muscle memory. See `memory/feedback_scaffold_before_answer.md` if you're in the Cowork memory system; otherwise just internalize this rule.

---

## 2. State of the repo as of Phase 0 close

```
Hybrid Architecture/
├── CLAUDE.md                         ← read first
├── PROJECT_PLAN.md                   ← Phase 1 detail lives in §Phase 1
├── PHASE_1_HANDOFF.md                ← this file
├── README.md, GITHUB_GUIDE.md
├── docs/
│   ├── reading_list.md               ← curated, ordered concepts→papers
│   └── concepts/
│       ├── 01_decoding_basics.md     ← Mohamed's Phase 0 notes
│       ├── 02_kv_cache.md
│       └── 03_attention.md
├── notebooks/
│   ├── 00_warmup_decoding.ipynb      ← Phase 0, done
│   ├── 01_kv_cache_walkthrough.ipynb ← Phase 0, done
│   └── 02_attention_extraction.ipynb ← Phase 0, done (uses framework attention; broken for Pythia layers 9-11)
├── src/
│   └── hybrid_arch/
│       └── __init__.py               ← skeleton only; Phase 1 fills this out
├── tests/                            ← empty; Phase 1 starts populating
└── .gitignore
```

What's **not** yet in the repo:
- `pyproject.toml` — dependencies are pip-installed ad hoc on Colab; needs to be created so the package is `pip install -e .`-able for reproducibility.
- Tests of any kind.
- CI workflow (`.github/workflows/ci.yml`).
- Any of `metrics.py`, `viz.py`, `probes.py`, `checkpoints.py`, `decoding.py` under `src/hybrid_arch/`.

---

## 3. Phase 1 goal

From `PROJECT_PLAN.md`:

> Build the instrument. Every downstream analysis depends on these primitives, so this phase pays back forever.

Concretely: a small importable library (`src/hybrid_arch/metrics.py` and `viz.py`) plus one demonstration notebook (`notebooks/03_metric_zoo.ipynb`) that applies every metric to a fixed corpus slice and produces a correlation matrix CSV at `docs/results/01_metric_correlations.csv`.

The metrics, all per-token, all functions of a forward pass:

1. **`next_token_entropy(logits)`** — Shannon entropy `H = -Σ p log p` over the predicted next-token distribution at each position. High = model is uncertain about what comes next.
2. **`top1_probability(logits)`** — `softmax(logits).max(dim=-1)`. Simpler companion to entropy; useful for thresholding.
3. **`attention_entropy(attn_weights, layer, head)`** — entropy of a single head's attention distribution at each query position. Per (layer, head, query). Low entropy = focused attention; high entropy = diffuse.
4. **`attention_concentration(attn_weights)`** — sum of attention mass on top-1, top-3, top-5 attended positions, per (layer, head, query). Complementary to entropy.
5. **`parallel_prediction_agreement(model, tokens, k)`** — the project's headline metric. For each position `t`, predict tokens `t+1 … t+k` *in parallel* (one forward pass, treating positions independently as if you didn't have to be autoregressive) and compare to the actual autoregressive output for the same `k` steps. Report agreement rate per position. This is the offline "parallel-safety" label.

Plus:
- **`viz.entropy_heatmap(tokens, entropies)`** — token strip with color-coded heat per position.
- **`viz.attention_track(tokens, metric)`** — per-token line plot.
- A consistent matplotlib style set once and never touched again.

---

## 4. Critical constraint: the Pythia attention bug

In the current `transformers` release that Colab ships with, Pythia-160M's deeper attention layers (typically layers 9, 10, 11) return **NaN** in `outputs.attentions` for both the eager and SDPA attention paths. The cause is numerical overflow in the unfused softmax — deeper layers have larger residual-stream magnitudes, so `exp(Q·Kᵀ)` overflows, and `inf - inf` poisons the masked-position softmax. Both eager and SDPA paths inherit this. See `notebooks/02_attention_extraction.ipynb` and `docs/concepts/03_attention.md` for the full diagnostic.

**Implication for Phase 1:** *do not* rely on `model(..., output_attentions=True)` for any attention metric. The library must instead extract attention weights via **forward hooks** on the attention layers, computing softmax manually in fp32 to avoid the overflow. This is mentioned in `PROJECT_PLAN.md` as a Phase 1 deliverable: write `src/hybrid_arch/attention.py` (or fold into `metrics.py`) that gives back correct attention tensors for any Pythia/GPTNeoX model regardless of which transformers version is installed.

Forward-hook approach: register hooks on each `GPTNeoXAttention` layer that capture the (Q, K) tensors after rotary embedding application; recompute attention scores manually as `Q @ K.transpose(-2, -1) / sqrt(head_dim)`; apply the causal mask; apply softmax in fp32; return the result. Avoid materializing the full attention matrix at long sequence lengths — keep this honest: extract attention for the prompts the metric library will actually use (≤512 tokens), not for arbitrary lengths.

This work is **the first thing Phase 1 builds** because every attention-based metric depends on it.

---

## 5. Suggested sequence of work

Each step ends in something runnable. Don't move on until the prior step is green.

### Step 1 — Project skeleton (~30 min)
- Create `pyproject.toml` with deps: `torch`, `transformers>=4.40`, `datasets`, `numpy`, `matplotlib`, `pytest`, `ruff`, `pyright`. Use `uv` if you're partial to it; otherwise vanilla.
- Update `src/hybrid_arch/__init__.py` to export what we're about to build.
- Add a minimal `tests/test_smoke.py` that imports the package and checks `__version__`.
- Verify: `pytest -q` passes.

### Step 2 — Forward-hook attention extraction (~half a day)
- File: `src/hybrid_arch/attention.py`.
- Function: `extract_attention(model, input_ids) -> Tensor` of shape `[layers, batch, heads, seq, seq]`, with attention weights computed manually from hooked Q and K, masked causally, softmaxed in fp32, NaN-free.
- Test: load Pythia-160M, run on a short prompt, assert (a) shape correctness, (b) row sums equal 1.0 to within `1e-4`, (c) upper triangle is exactly zero, (d) no NaN anywhere.
- This test is the spec — write it first, then make it pass.

### Step 3 — Logit-side metrics (~1 hour)
- File: `src/hybrid_arch/metrics.py`.
- Functions: `next_token_entropy(logits)`, `top1_probability(logits)`. Both accept the full `[batch, seq, vocab]` logits tensor and return per-position scalars of shape `[batch, seq]`.
- Test: a hand-constructed deterministic logits tensor with known entropy values; assert numerical match.

### Step 4 — Attention-side metrics (~2-3 hours)
- Functions in `metrics.py`: `attention_entropy(attn_weights)`, `attention_concentration(attn_weights, top_k=(1,3,5))`. Operate on `[layers, batch, heads, seq, seq]` tensors; return `[layers, batch, heads, seq]` (one number per query position, per head, per layer).
- Test: hand-constructed attention with a single-position mass (entropy=0, concentration top-1=1.0) and uniform attention (entropy=log(seq), concentration top-1=1/seq).

### Step 5 — Parallel-prediction agreement (~1 day)
- The hardest one because it requires running the model twice and comparing.
- Function: `parallel_prediction_agreement(model, input_ids, k=4)`. For each position `t` in the prompt, predict tokens `t+1 … t+k` in parallel (one forward pass on prefix `0..t`, take the top-1 prediction from positions `t, t+1, …, t+k-1` as the "parallel guess" — note these come from the same forward pass), then run real autoregressive decoding for `k` steps starting from position `t+1`, and compare. Return per-position agreement counts `[batch, seq, k]`.
- Subtle: "parallel prediction" here means "what does the model say in one shot when forced into teacher-forced mode" vs "what does it generate when fed its own outputs sequentially." The distinction is whether the model has seen ground-truth or its own predictions at intermediate positions.
- Test: trivial sequence ("the the the the the") should have high agreement; random tokens should have lower.

### Step 6 — Visualization helpers (~2 hours)
- File: `src/hybrid_arch/viz.py`.
- `entropy_heatmap(tokens, entropies, ax=None)` — colored strip with token labels.
- `attention_track(tokens, metric, ax=None)` — line plot, x=position, y=metric.
- One global `STYLE` dict at the top of the file; both functions read from it. Use `viridis` for sequential, `coolwarm` for diverging.
- Test: smoke test that the functions produce a `matplotlib.figure.Figure` on hand-built inputs without raising.

### Step 7 — Demonstration notebook (~half a day)
- File: `notebooks/03_metric_zoo.ipynb`.
- Load Pythia-160M @ `step143000`, tokenize a fixed 1000-token slice of WikiText-103 (cache it under `data/`, gitignored), apply every metric, write the correlation matrix CSV to `docs/results/01_metric_correlations.csv`, render two or three small figures using the viz helpers.
- This is Mohamed's deliverable, so structure it so he can run it cell-by-cell on Colab CPU in ~5 minutes total.

### Step 8 — Closeout (~30 min)
- Update `PROJECT_PLAN.md` §Phase 1 status to "complete."
- Update `CLAUDE.md` §8 phase summary if the timeline shifted.
- Make sure the README's "Status" line reflects Phase 1 being done.

---

## 6. Definition of done

Phase 1 is complete when **all** of the following hold:

- `src/hybrid_arch/{attention,metrics,viz}.py` exist, import cleanly, and have docstrings.
- `pytest -q` passes, with at least one test per metric and per viz function.
- `notebooks/03_metric_zoo.ipynb` runs end-to-end on a fresh Colab CPU runtime in ≤10 minutes and produces the correlation CSV.
- `docs/results/01_metric_correlations.csv` exists and is committed (small file, gitignored exception is OK).
- One paragraph in `docs/concepts/04_metrics.md` summarizing what each metric measures, written by Mohamed (with you scaffolding) — this matches the Phase 0 pattern.

---

## 7. Compute discipline (reminder)

- Default model: **Pythia-160M @ step143000** in fp32. This fits free CPU with headroom.
- Stretch: Pythia-410M with care (fp32, batch=1, short prompts).
- Anything larger requires a paid GPU — flag it instead of attempting.
- Cache anything expensive. Hidden states, attention weights, logits for a fixed `(model, checkpoint, dataset-slice)` triple go in `data/` (gitignored).
- Every experiment writes a `manifest.json` next to its outputs: model name + revision hash, seed, dataset slice hash, git commit, wall clock. Reproducibility is non-negotiable per `CLAUDE.md` §9.

---

## 8. Anti-patterns to avoid (Phase 0 incident report)

These tripped Phase 0; don't let them trip Phase 1.

- **`pip install torch` in a Colab notebook.** Colab ships with torch already imported. Reinstalling poisons the kernel and produces `partially initialized module 'torch'`. Only ever install packages that aren't already present, and even then, restart the runtime afterward.
- **Reading from the HF cache object's `.key_cache` / `.value_cache` attributes.** The DynamicCache API has changed three times in the last year. Use `cache.to_legacy_cache()` if you must inspect, or extract via forward hooks.
- **Trusting `output_attentions=True` for Pythia.** See §4. It returns NaN in deep layers.
- **Quantitative hints without verification.** `CLAUDE.md` §10 explicitly says "verify everything quantitative." Phase 0 had one slip on this (an "expected answer" of 8.6 GB that was actually 2 GB) — don't repeat.
- **Writing polished refinements of Mohamed's answers without scaffolding first.** See §1. He gets false confidence from reading good answers; he gets real understanding from writing his own with hints.

---

## 9. Where to start

Open a fresh chat with Claude Code, point at this file, and say:

> *Read `PHASE_1_HANDOFF.md` and `CLAUDE.md`. Then start with §5 Step 1 — set up `pyproject.toml`, update `__init__.py`, add the smoke test. Don't start Step 2 until I've reviewed Step 1's diff.*

Don't let it batch all the steps into one mega-PR. Phase 1 is the foundation everything else stands on; review each step before letting it move forward.

---

## 10. Reading-list pointers (don't read these yet, but know they exist)

For attention entropy as a probe: *On Next-Token Prediction in LLMs* (May 2025), arXiv:2505.11183.
For attention sinks (the Phase 0 reflection question): *When Attention Sink Emerges in Language Models*, ICLR 2025.
For probing methodology before Phase 3: *Probing Classifiers: Promises, Shortcomings, and Advances*, Belinkov 2022.
For the broader "parallel safety" framing: *Fast Inference from Transformers via Speculative Decoding* (Leviathan et al., 2022) — the parallel-prediction-agreement metric is the offline cousin of speculative-decoding acceptance rate.

Full list in `docs/reading_list.md`.

---

**End of handoff.** When Phase 1 is done, write a `PHASE_2_HANDOFF.md` in the same shape and hand it to the next Claude Code session.
