# CLAUDE.md — Hybrid Architecture

> Project guide for Mohamed (`m0hamed@mit.edu`) and any Claude session that joins the project.
> Last refreshed: **2026-05-24**.

This file is the *front door* to the project. Read it first. It explains what we're trying to learn, why, and how the repo is organized. The detailed roadmap lives in [`PROJECT_PLAN.md`](./PROJECT_PLAN.md); the curated reading list lives in [`docs/reading_list.md`](./docs/reading_list.md).

---

## 1. The one-paragraph pitch

Training a large language model is embarrassingly parallel — every position in a training batch is computed at the same time. **Inference, however, is stubbornly sequential**: to generate the next token, the model needs the previous one. That asymmetry is the single biggest driver of inference cost on the planet right now. This project asks a tighter version of the question that the field has been chasing: *not every token is equally "sequential" — some are obvious from context (the `the` in "the cat sat on the…"), others genuinely require the model to think.* We want to (a) **measure** that asymmetry directly from attention patterns, (b) **track how it emerges across pretraining** using Pythia's 154 training checkpoints — a developmental angle the big labs can't easily replicate because they only ship final checkpoints — and (c) **release a diagnostic toolkit** that adaptive-inference researchers can plug into their own routers and drafters.

**Working title:** *Parallelism Emerges: An attention-pattern atlas of how language models learn what to compute in parallel.*

---

## 2. What you need to know about how AI inference works

> This section exists because the rest of the doc assumes these mental models. If you already know them, skim and move on.

### 2.1 Autoregressive decoding

A language model is a function that takes a sequence of tokens `x₁, …, xₜ` and produces a probability distribution over the next token `xₜ₊₁`. To generate text, you:

1. Run the model on the prompt → get `P(xₜ₊₁ | x₁…xₜ)`.
2. Pick a token (greedy = argmax; sampling = draw from the distribution).
3. **Append** that token to the sequence.
4. Go to step 1.

Step 3 is the sequential bottleneck. You cannot compute step `t+2` until you've finished step `t+1`. Every token is one forward pass through the entire model — on a 70B model, that's ~140GB of weights touched per token.

### 2.2 KV caching — the optimization that makes inference tolerable

Naively, decoding token `t+1` would recompute attention over all `t` previous tokens from scratch. Instead, modern decoders cache the **K**ey and **V**alue projections of every previous token in every attention layer. New token's query attends over the cached K/V. This turns the per-step cost from `O(t²)` to `O(t)`. The cache itself is huge (often >50% of memory at long context) and managing it is half the job of an inference engineer.

### 2.3 Why training is parallel and inference is not

In training, you know the entire ground-truth sequence ahead of time, so you can compute the loss at every position in one forward pass using a causal mask. In inference, you don't know `xₜ₊₁` until you've computed it.

The whole adaptive-inference literature — speculative decoding, Medusa, EAGLE-3, Mixture-of-Depths, Mixture-of-Recursions, multi-token prediction — is variations on the same theme: **find shortcuts that recover some of training's parallelism at inference time, without losing quality.**

### 2.4 The asymmetry we exploit

Some tokens are highly predictable from context. The next token after `"the United States of"` is almost certainly `"America"`. The model "knows" this with high confidence; it doesn't need to deeply integrate context to decide. Other tokens — the first word of a sentence, a person's name, a numerical answer — require the model to actually think. **If we can tell these apart before running the full forward pass, we can route the easy ones through a cheaper path.** That's the field's bet. Our contribution is characterizing *which tokens are which* and *when during training the model learns to make that distinction crisply*.

---

## 3. The research question (refined)

> The first version of this project was "build a router that picks parallel vs sequential decoding per token." That's already been done at industry scale (see §6). We pivoted.

**Primary question:** Across pretraining, how does the *separability* of "parallel-safe" vs "sequential-needed" tokens emerge in small language models, and what attention-pattern signatures predict it?

**Secondary questions:**
- Can we build a probe that, from a single layer's hidden state at position `t`, predicts whether position `t+1` is parallel-safe — and how early in pretraining does that probe become trainable?
- How do these signatures shift on out-of-distribution text (e.g., code, biomedical, math)?
- Does the small-model attention-pattern atlas correlate with EAGLE-3 / MoR routing decisions on larger models? (i.e., does the small-model story generalize?)

**What "novel" means here:** Not "beat EAGLE-3 on speed." That game is lost on free Colab. Novel means "produce evidence and tooling that nobody currently has because they all start from final checkpoints of huge models."

---

## 4. Why this maps to real inference-engineer problems

The output of this project should land on an inference engineer's desk and *be useful*. Concretely:

1. **Drafter calibration.** EAGLE-3 acceptance rates plateau around 0.75-0.85. Every rejected draft is wasted compute. Our probes are direct candidates for *pre-verification* (predict rejection before running the verifier). Even a small win compounds at scale.
2. **MoR/MoD routing supervision.** Mixture-of-Recursions and Mixture-of-Depths train routers end-to-end with a sparse signal. Our offline-derived "parallel-safety" labels could serve as an auxiliary supervision signal — a paper-worthy contribution if it improves routing convergence.
3. **Diagnostic dashboards.** Inference teams flying blind on *why* their adaptive techniques fail on certain inputs. A reusable visualization layer (per-token entropy heatmaps, attention concentration tracks, calibration curves) is genuinely useful.

Whenever a phase deliverable lands, the question to ask is: *"if I emailed this to someone on the vLLM or TGI team, would they reply?"* If the answer is no, sharpen the framing or kill the phase.

---

## 5. Repo conventions

```
hybrid-architecture/
├── CLAUDE.md                       # this file — read first
├── PROJECT_PLAN.md                 # 5-phase roadmap with milestones
├── README.md                       # public-facing pitch (GitHub front page)
├── GITHUB_GUIDE.md                 # first-time Git/GitHub walkthrough
├── docs/
│   ├── reading_list.md             # ordered concepts → papers
│   ├── concepts/                   # short notes as I learn each idea
│   └── results/                    # final plots, tables, writeups
├── notebooks/                      # exploratory, numbered
│   ├── 00_warmup_decoding.ipynb    # write a decode loop from scratch
│   ├── 01_kv_cache_walkthrough.ipynb
│   └── …
├── src/
│   ├── hybrid_arch/                # importable code
│   │   ├── decoding.py             # vanilla + experimental decoders
│   │   ├── probes.py               # attention-pattern probes
│   │   ├── metrics.py              # entropy, concentration, calibration
│   │   ├── checkpoints.py          # Pythia checkpoint loading helpers
│   │   └── viz.py                  # plotting utilities
│   └── scripts/                    # one-off batch runs
├── data/                           # cached datasets (gitignored)
├── results/                        # raw experiment outputs (gitignored)
├── tests/                          # pytest, fast unit tests on tiny inputs
├── pyproject.toml                  # uv/poetry config + ruff + pyright
└── .github/workflows/ci.yml        # runs tests + linter on push
```

### Conventions, not preferences
- **Notebooks are exploratory; logic lives in `src/`.** Anything that runs twice gets extracted into a function in `src/hybrid_arch/`. Notebooks should be readable end-to-end without scrolling through helper code.
- **Every experiment writes a `manifest.json`** alongside its outputs: model name + revision, seed, dataset slice hash, git commit, wall clock. We must be able to re-run any plot we publish.
- **Plots use one consistent style.** Set this up once in `viz.py` and never touch matplotlib defaults again.
- **Commit hygiene:** small commits, present-tense imperative messages (`add KV cache probe`, not `added KV cache probe`). One concept per commit.
- **No giant files in git.** Models, datasets, and result tensors go in `data/` and `results/`, both gitignored. Use `huggingface-cli` to pull when needed.

---

## 6. State of the art as of 2026-05 (what we are NOT trying to beat)

Mentioning this for two reasons: (a) so Claude doesn't propose "let's reinvent EAGLE" in a future session, (b) so the README can position the work honestly.

- **EAGLE-3 (2025)** — speculative decoding, 0.75-0.85 acceptance rate, 3-6x speedup, productionized in vLLM. ([arXiv:2503.01840](https://arxiv.org/html/2503.01840v1))
- **Mixture-of-Recursions (NeurIPS 2025)** — per-token adaptive recursive depth + recursion-wise KV caching. ~2x throughput at parity quality. ([arXiv:2507.10524](https://arxiv.org/abs/2507.10524))
- **Mirror Speculative Decoding (Oct 2025)** — parallel-draft variant. ([arXiv:2510.13161](https://arxiv.org/pdf/2510.13161))
- **DeepSeek MTP** — multi-token prediction baked into pretraining.
- **ADEPT (2026)** — draft model deciding per-token depth.

Our contribution lives upstream of all of these: better characterization of *what makes a token easy*, with a developmental lens nobody else has.

---

## 7. Compute discipline (free Colab edition)

We have a 16GB T4 on free Colab, and a laptop. That's it. Decisions follow from this:

- **Default workhorse model:** Pythia-160M. Loads in fp32, full attention extraction fits comfortably, 154 training checkpoints freely available.
- **Stretch models:** Pythia-410M (fp32, smaller batch), Pythia-1B (fp16, single-sequence).
- **Datasets:** WikiText-103 for general, MBPP/HumanEval for code, GSM8K for math, a small slice of PubMed abstracts for biomedical (no PHI). Tokenize once, cache as `.pt` files.
- **Experiment budgets:** if a single experiment needs >4 hours of T4 time, redesign it. Bigger experiments are a sign of insufficient analytical thinking, not of ambition.
- **Caching is mandatory.** Hidden states, attention weights, and logits for fixed (model, checkpoint, dataset-slice) tuples get cached to disk. Re-running plots should never recompute the model pass.

---

## 8. The 5-phase plan (summary)

Full detail in [`PROJECT_PLAN.md`](./PROJECT_PLAN.md). Headline only:

1. **Foundations (weeks 1-3)** — implement decoding from scratch, KV cache walkthrough, get one Pythia checkpoint loaded and generating text. ✅ **Complete**.
2. **Measure (weeks 4-7)** — build the metric library: per-token next-token entropy, attention concentration, attention-sink patterns, parallel-prediction agreement. ✅ **Complete (2026-05-25)** — library + 53 tests + demo notebook + first correlation CSV land in `src/hybrid_arch/`, `tests/`, `notebooks/03_metric_zoo.ipynb`, and `docs/results/01_metric_correlations.csv`.
3. **Patterns (weeks 8-11)** — the core analysis. Cross-checkpoint, cross-domain. Produce the first version of the atlas.
4. **Probes & router (weeks 12-15)** — train shallow probes for parallel-safety; evaluate as drafter pre-verifiers and as MoR routing supervision.
5. **Polish & publish (weeks 16-20)** — writeup, blog post, GitHub repo polish, optional workshop submission.

---

## 9. Reproducibility rules (non-negotiable)

- Seeds set explicitly at the top of every script (Python, NumPy, Torch).
- Pin model revisions by commit hash, not by tag. (HuggingFace lets you do this.)
- Every published plot has a script that regenerates it from cached data.
- Every published number lives in a CSV or JSON in `docs/results/`.

---

## 10. How Claude should help

When a future Claude session works on this project:

- **Default to teaching mode.** Mohamed is new to LLM internals; explain inference concepts (KV cache, attention masking, sampling) inline when they come up, even if the immediate task is just coding.
- **Push back on scope creep.** If something would need >4hr of T4 time or a bigger model than Pythia-410M, flag it and propose a smaller analogue.
- **Honor the SOTA boundary.** Don't propose "let's beat EAGLE-3." Do propose "let's see if our probes predict EAGLE-3 rejections."
- **Verification step on everything quantitative.** Math, calibration curves, acceptance-rate numbers — verify programmatically before they go in a writeup.
- **GitHub guidance is welcome.** Mohamed is a first-time GitHub user (account: [github.com/momagzoub](https://github.com/momagzoub)); if a task involves `git` or repo hygiene, walk through it rather than just running commands.
- **When uncertain, ask before doing.** Cowork's `AskUserQuestion` is preferred over guessing direction on multi-step work.

---

## 11. Living document policy

This file gets updated when:
- The thesis sharpens or pivots (rare — needs explicit conversation).
- A phase completes (update §8 status).
- We discover a SOTA change that affects positioning (update §6).
- A convention turns out to be wrong (update §5).

Small fixes go straight in. Big changes get committed with a message starting `CLAUDE.md:` so they're easy to find in `git log`.
