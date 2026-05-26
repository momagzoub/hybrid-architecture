# PROJECT_PLAN.md — Hybrid Architecture

> The 20-week roadmap. Read [`AGENTS.md`](./AGENTS.md) first for context.

This plan assumes **~10-15 hours/week** over 5 months. Each phase ends with a tangible deliverable you can show to someone. If a phase slips, *cut scope, do not slip the deliverable*. The deliverables compound: if Phase 3 doesn't ship, Phase 4 has nothing to build on.

The phases are not equal in difficulty. Phase 0 is mostly learning. Phase 2 is the conceptual heart. Phase 4 is the engineering peak. Phase 5 is what makes the whole thing visible.

---

## Phase 0 — Foundations (weeks 1-3)

**Goal:** stop being a stranger to LLM internals. By the end, you should be able to draw the inference loop on a napkin and explain why a KV cache exists.

### What to learn first
1. The Illustrated Transformer (Alammar). One afternoon.
2. Andrej Karpathy's "Let's build GPT" video. Code along. **Do not skip the manual implementation.**
3. KV cache: read one good blog post (HuggingFace's, or Lilian Weng's inference survey).
4. Speculative decoding original paper (Leviathan et al., 2022). Just the intro and §3.

### What to build
- `notebooks/00_warmup_decoding.ipynb` — load `EleutherAI/pythia-160m`, generate 50 tokens with a hand-written greedy loop (no `.generate()`). Print the per-step logits for the top-5 candidates.
- `notebooks/01_kv_cache_walkthrough.ipynb` — same as above, but compare wall-clock with and without `use_cache=True`. Plot the difference vs sequence length.
- `src/hybrid_arch/decoding.py` — extract the loop into a clean function with type hints.

### Phase deliverable
A short writeup (`docs/concepts/01_decoding_basics.md`) explaining autoregressive decoding and KV caching *in your own words*, with a plot you generated. Audience: a smart friend who knows Python but not ML.

### Compute
Negligible. T4 once or twice for the wall-clock comparison.

### Industry mapping
None yet — this is foundation. The mapping starts in Phase 1.

### How you'll know you're done
You can explain to a non-ML friend why decoding is sequential, why KV caches exist, and what the per-token cost of a 7B model actually is. Without notes.

---

## Phase 1 — Measure (weeks 4-7) — **COMPLETE (2026-05-25)**

**Goal:** build the instrument. Every downstream analysis depends on these primitives, so this phase pays back forever.

**Outcome.** The metric library exists at `src/hybrid_arch/{attention,metrics,viz}.py`, with 53 tests passing. The demo notebook `notebooks/03_metric_zoo.ipynb` runs end-to-end on Pythia-160M @ step143000 over a 256-token WikiText slice in ~5 minutes on CPU, producing `docs/results/01_metric_correlations.csv` with a manifest sidecar. The Pythia deep-layer attention NaN bug is solved via a forward-hook re-implementation in `attention.py`. First signal of the project thesis is already visible: `parallel_agreement` correlates with `next_token_entropy` at r ≈ -0.43 — strong enough to be a real effect, weak enough to leave room for the Phase 3 probe to add value. Aggregate attention metrics did not predict parallel safety on this slice (|r| < 0.11), which Phase 2 should investigate at the (layer, head) level.

### What to learn first
- Attention entropy as a measurability concept (`-Σ p log p` over a single head's attention distribution at a position).
- Attention sinks paper (ICLR 2025, "When Attention Sink Emerges").
- The difference between *teacher-forced* logits (model sees ground truth) and *free-generation* logits (model sees its own output). This matters more than you'd expect.

### What to build (`src/hybrid_arch/metrics.py`)
Each of these is a function that takes a model forward pass and returns one number per token:

1. `next_token_entropy(logits)` — Shannon entropy of the predicted distribution. High = model is uncertain.
2. `top1_probability(logits)` — easy companion to entropy. Useful for thresholding.
3. `attention_entropy(attn_weights, layer, head)` — how spread out is each head's attention at each position?
4. `attention_concentration(attn_weights)` — mass on the top-1, top-3, and top-5 attended positions.
5. `parallel_prediction_agreement(model, tokens, k)` — *the key one*. Predict tokens `t+1…t+k` in parallel (one forward pass treating positions as independent), then run real autoregressive decoding for `k` steps, then measure how often they agree. This is your "parallel-safety" ground truth.

### What to build (`src/hybrid_arch/viz.py`)
- `entropy_heatmap(sequence, entropies)` — token strip with color-coded heat.
- `attention_track(sequence, attention_metric)` — per-token line plot.
- Settle on a color palette now and never change it. Suggest: matplotlib's `viridis` for sequential, a custom diverging map for calibration plots.

### Phase deliverable
- `notebooks/02_metric_zoo.ipynb` — apply every metric to the same 1000 tokens of WikiText, sanity-check correlations, show pretty plots.
- One CSV in `docs/results/01_metric_correlations.csv` with the correlation matrix.

### Compute
Pythia-160M on 10k tokens of WikiText. Maybe 30 min of T4 time per metric. Cache the hidden states + attention weights once — every subsequent metric reads from cache.

### Industry mapping
Metric #5 (parallel-prediction agreement) is *directly* the offline analogue of speculative decoding acceptance. The literature uses it implicitly but rarely measures it cleanly across models. This is the first thing you can show an inference engineer.

### How you'll know you're done
You can take any (model, dataset) pair and produce a per-token annotated CSV with all five metrics in under 5 minutes of human time.

---

## Phase 2 — Patterns (weeks 8-11) — **COMPLETE (2026-05-25)**

**Outcome.** The atlas at [`docs/results/02_emergence_atlas.md`](./docs/results/02_emergence_atlas.md) lands four results: (1) parallel-safety emerges between training steps 128 and 1000 across all three Pythia sizes and barely moves afterwards; (2) the Phase 1 aggregate-attention null is resolved by per-(layer, head) logistic regression, which hits AUROC 0.845 ± 0.083 on Pythia-410m at the final checkpoint; (3) MBPP code is 3.9× more parallel-safe than WikiText on Pythia-410m (psf 0.452 vs 0.115), with GSM8K math in between; (4) content words show *higher* parallel-safety than function words at every size, a counterintuitive finding worth re-examining at `k=1`. All compute flows through the new `metric_battery` cache layer, making the full sweep re-derivable from `data/cache/` in milliseconds.

**Goal:** the analytical heart of the project. Find the structure that nobody else has the checkpoints to find.

### The setup
EleutherAI publishes **154 training checkpoints** for each Pythia model — from step 0 to step 143,000. You'll work with three sizes (70M, 160M, 410M) at ~12 checkpoints each (e.g., steps 0, 1k, 4k, 16k, 32k, 64k, 96k, 128k, 143k). That's 36 model loads. Each one runs your Phase 1 metric battery on a fixed 5k-token evaluation set.

### Core experiments
1. **The emergence curve.** For each (model size, checkpoint) plot the fraction of tokens with `parallel_prediction_agreement >= 0.9`. Does parallel-safety emerge smoothly? In phases? At a critical model size?
2. **The signature analysis.** Pick "easy" and "hard" tokens at the final checkpoint. What distinguishes them in earlier checkpoints? Train a logistic regression on the four other metrics → predict parallel-safety. Track its accuracy across checkpoints.
3. **Token-type breakdown.** Categorize tokens (function words, content words, code syntax, numbers, names) and show how parallel-safety differs by category and evolves.
4. **Domain shift.** Same battery, but on three datasets: WikiText (general), MBPP (code), GSM8K (math). Do parallel-safety signatures generalize across domains?

### What to build (`src/hybrid_arch/checkpoints.py`)
Helper to load Pythia at any revision by step number (`step3000`, `step143000`, etc.) and run the metric battery. Cache by `(model_size, step, dataset_slice)` hash.

### Phase deliverable
- `docs/results/02_emergence_atlas.md` — the headline document. Contains the emergence curve, the signature analysis, the domain-shift plot, with prose interpretation.
- 5-6 publication-quality plots in `docs/results/figures/`.

### Compute
This is your peak compute spend. Budget: 20-30 hours of T4 time across the phase, but heavily cached so you only pay it once. If a re-run costs more than 30 min, you broke a cache somewhere — fix it before continuing.

### Industry mapping
The atlas plot is the GitHub README hero image. Any inference researcher who looks at the repo should immediately get *the point* from that plot alone. This phase is what makes the project share-able.

### How you'll know you're done
The atlas plot has a caption you'd be willing to staple to a workshop submission.

---

## Phase 3 — Probes & router (weeks 12-15)

**Goal:** turn the analysis into something you can call from code.

### What to build
1. `src/hybrid_arch/probes.py` — a small probe (3-layer MLP, ~10k params) that takes a hidden state from layer `L` at position `t` and predicts whether position `t+1` is parallel-safe. Train one probe per (model_size, layer) pair. Plot accuracy as a function of layer depth — at what point in the network does parallel-safety become predictable?
2. **The drafter-rejection prediction experiment.** Use HuggingFace's `assistant_model` speculative decoding on Pythia-1B with Pythia-160M as drafter. Log every accept/reject event. Test: does the Phase 2 entropy metric predict rejections? Does the Phase 3 probe predict them better? Report ROC AUC.
3. **The "what does the router look at" experiment.** Train a linear probe on top of MLP probes — interpret weights to identify which features (attention entropy at head X, top-1 prob, etc.) matter most.

### Phase deliverable
- `docs/results/03_probes.md` — probe accuracy curves, rejection-prediction ROC, feature importance.
- A pretrained probe checkpoint (~50KB) committed to `src/hybrid_arch/checkpoints_probes/` so people can use it.

### Compute
Moderate. Probe training is tiny (CPU is fine). The drafter-rejection experiment needs a few hours of T4.

### Industry mapping
This is **the** phase that lands. "We trained a sub-10k-param probe that predicts EAGLE/spec-decoding rejections with AUC 0.X" — if X is real, that's the headline of the blog post.

### How you'll know you're done
You can hand the probe to someone and they can use it in 5 lines of code.

---

## Phase 4 — Hybrid decoder demo (weeks 16-18)

**Goal:** show the probes inside a working decoder. Not to beat SOTA — to demonstrate the use case.

### What to build
- `src/hybrid_arch/decoding.py` (extension) — a `hybrid_decode()` function that uses the probe to decide, per position, whether to use single-step decoding or a parallel multi-token guess (verified by a one-shot pass).
- A measurement harness: tokens/second, perplexity vs the ground-truth autoregressive output, fraction of positions routed each way.

### Honest expectations
You will probably *not* beat the existing `.generate()` on raw throughput. That's fine — your model is 160M, and PyTorch's overhead dominates. What you *can* show: the probe correctly identifies the right fraction of parallel-safe tokens, and on a controlled synthetic benchmark, the hybrid decoder matches the predicted speedup.

### Phase deliverable
- `src/scripts/phase4_hybrid_decoder.py` — end-to-end demo. Generates a paragraph, color-coded by routing decision.
- A short writeup: "what we'd need to make this beat SOTA at scale" — honest list of the engineering work (CUDA kernels, batched scheduling, KV cache surgery) you'd need to actually compete.

### Compute
Light. Phase 4 is mostly engineering on Phase 3's outputs.

### Industry mapping
The "what we'd need to scale" writeup is the part inference engineers will read. Showing you *know what's hard* about productionizing this is itself a hiring signal.

### How you'll know you're done
You can run a one-liner and get a colored printout of a routed generation.

---

## Phase 5 — Polish & publish (weeks 19-20+)

**Goal:** the project doesn't exist until someone has read it.

### What to do
1. **Repo polish.** README hero image (the atlas plot), badges, install instructions tested in a fresh Colab, citation block.
2. **Blog post.** Long-form writeup, 2000-4000 words, with embedded plots. Publish on a static site (GitHub Pages, or your own domain). Link from the README.
3. **Tweet/post thread.** One headline plot + one paragraph + link. Tag relevant labs (politely) if you have a real result.
4. **Optional workshop submission.** ICLR/NeurIPS workshops on efficient inference, ML systems, or interpretability often accept shorter writeups. Check deadlines — ICLR workshops are typically Nov-Feb, NeurIPS workshops Sep-Oct, ICML workshops Apr-Jun.

### Phase deliverable
- A repo you'd link from your CV.
- A blog post you'd link from your LinkedIn.
- Optional: a workshop submission.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Pythia checkpoints don't show interesting emergence patterns | Medium | Pivot to "domain shift on final checkpoint only" — still a clean contribution |
| Probes don't predict EAGLE rejections better than entropy baseline | Medium-high | That's still a result. "Simple baselines work" is publishable; frame as such |
| Free Colab runs out / gets throttled | Medium | Caching discipline (Phase 1) makes this annoying but not fatal |
| Scope creeps into "rebuild EAGLE on small models" | Medium | AGENTS.md §8 warns any assistant to push back |
| 5 months turns into 8 months | High | The phases are gated by deliverables; don't proceed without the prior deliverable |

---

## A note on what "novel" looks like for a portfolio

You will not invent a new architecture in 5 months on free Colab. You don't need to. What you need is:

- A **clean question** that nobody else has bothered to ask carefully.
- **Rigorous answers**, with reproducible code.
- A **diagnostic tool** other people can use.

The combination of those three is what differentiates a portfolio that lands you interviews from a portfolio that doesn't. Inference teams don't hire people who claim to have beaten SOTA on free Colab (they know that's not real). They hire people who clearly *understand the problem space* and can produce useful artifacts. That's the bar this plan is aimed at.
