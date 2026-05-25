# Reading list

> Ordered by *when in the project you'll want to know this*. Don't read everything before starting — read Phase 0 items now, the rest as their phases approach.

Each entry has a one-line "why" so you can decide whether to read it deeply or skim. Items marked **★** are required; others are background that pays off later.

---

## Phase 0 foundations — read these first

★ **The Illustrated Transformer** — Jay Alammar
[jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)
*Why:* the canonical "what is attention, what is a transformer" visual explainer. Build the mental model here.

★ **Let's build GPT: from scratch, in code, spelled out** — Andrej Karpathy
[YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY)
*Why:* 2 hours, you code along. Nothing teaches the inference loop like writing it yourself.

★ **The Annotated Transformer** — Sasha Rush et al.
[nlp.seas.harvard.edu/annotated-transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
*Why:* line-by-line PyTorch walkthrough. Use as a reference when Karpathy's code is too compressed.

★ **Transformer Inference Arithmetic** — kipp.ly
[kipp.ly/transformer-inference-arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
*Why:* the back-of-envelope cost model for every inference question. *Read twice.*

**LLM Inference Series** — Hugging Face
[huggingface.co/blog/llm-inference-survey-1](https://huggingface.co/blog/llm-inference-survey-1) (and parts 2, 3, 4)
*Why:* KV caching, batching, paged attention — explained at the level you need.

---

## Phase 1 metrics — building the instrument

★ **Fast Inference from Transformers via Speculative Decoding** — Leviathan, Kalman, Matias (2022)
[arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
*Why:* the foundational paper. The "agreement between draft and target" idea underlies our parallel-prediction-agreement metric.

★ **Accelerating Large Language Model Decoding with Speculative Sampling** — Chen et al. (2023, DeepMind)
[arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
*Why:* the same idea, contemporary, with cleaner math. Read alongside Leviathan.

**When Attention Sink Emerges in Language Models: An Empirical View** — ICLR 2025
[OpenReview](https://proceedings.iclr.cc/paper_files/paper/2025/file/f1b04face60081b689ba740d39ea8f37-Paper-Conference.pdf)
*Why:* one of the few papers that studies attention patterns developmentally. Methodological inspiration.

**On Next-Token Prediction in LLMs: How End-Goals Determine the Consistency of Decoding Algorithms** — May 2025
[arXiv:2505.11183](https://arxiv.org/pdf/2505.11183)
*Why:* careful framing of how entropy interacts with decoding choices.

---

## Phase 2 atlas — developmental analysis

★ **Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling** — Biderman et al., 2023
[arXiv:2304.01373](https://arxiv.org/abs/2304.01373)
*Why:* the suite you'll use. Read §3 carefully — it tells you exactly which checkpoints exist.

**Mistral, Pythia, and Falcon, Oh My: An Empirical Study of Open Language Models** — various follow-ups
*Why:* cross-model context. Skim once Pythia is comfortable.

**The Quantization Model of Neural Scaling** — Michaud et al., 2023
[arXiv:2303.13506](https://arxiv.org/abs/2303.13506)
*Why:* one framework for thinking about *what* gets learned at *which* training step. Useful conceptual scaffolding for the emergence-curve narrative.

**Phase Transitions in the Output Distribution of Large Language Models** — 2024
*Why:* if your emergence curves show phase-transition-like behavior, this is the prior art to engage with.

---

## Phase 3 probes & routers

★ **EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test** — 2025
[arXiv:2503.01840](https://arxiv.org/html/2503.01840v1)
*Why:* current SOTA. You need to understand precisely what it does so you can position your probes against it.

★ **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads** — Cai et al., 2024
[arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
*Why:* the conceptual ancestor of EAGLE. Lighter read than EAGLE-3.

**Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation** — NeurIPS 2025
[arXiv:2507.10524](https://arxiv.org/abs/2507.10524)
*Why:* the closest contemporary cousin of your project's framing. Your probes should be evaluated as a supervision signal for MoR-style routers.

**Mixture-of-Depths: Dynamically allocating compute in transformer-based language models** — Raposo et al., 2024
[arXiv:2404.02258](https://arxiv.org/abs/2404.02258)
*Why:* the predecessor to MoR. Cleaner expository.

**Probing Classifiers: Promises, Shortcomings, and Advances** — Belinkov, 2022
[arXiv:2102.12452](https://arxiv.org/abs/2102.12452)
*Why:* probing methodology — what they tell you, what they don't, common pitfalls. Read before you trust your own probe accuracy numbers.

---

## Phase 4 hybrid decoder

**SpecInfer / Mirror Speculative Decoding / P-EAGLE** — recent parallel-draft work
[arXiv:2510.13161](https://arxiv.org/pdf/2510.13161) (Mirror), [AWS P-EAGLE blog](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/)
*Why:* shows what production-grade variants of "decode in parallel where you can" look like. Cite as the "what we'd need to scale this" comparison.

**vLLM paper** — Kwon et al., 2023 (Paged Attention)
[arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
*Why:* if you're going to write "here's what you'd need to make this fast at scale," you need to know what vLLM actually does.

---

## Phase 5 writing & publishing

**How to Write a Great Research Paper** — Simon Peyton Jones (lecture)
[YouTube](https://www.youtube.com/watch?v=1AYxMbYZQ1Y)
*Why:* one of the best 50-minute crash courses in research writing.

**Distill's Essay-style ML papers** — various
[distill.pub](https://distill.pub/)
*Why:* the gold standard for ML writeups that recruit attention. Even if you publish on your own blog, copy their structural choices.

---

## Tools & libraries (read the docs when you first reach for them)

- **HuggingFace `transformers`** — your default model loader. The "Inference Optimization" and "Generation Strategies" docs are required reading before Phase 1.
- **HuggingFace `datasets`** — Wikitext, MBPP, GSM8K. Cache locally.
- **PyTorch hooks** — `register_forward_hook` is how you extract attention weights cleanly. Read the official docs once.
- **`einops`** — once you find yourself writing `.reshape(b, h, t, d)` for the third time, install this.
- **`uv`** — the fastest Python package manager. Install once, never touch `pip` again.
- **`matplotlib` + `seaborn`** — defaults. If you reach for Plotly, ask yourself if you really need interactivity (rarely, for static writeups).

---

## When to stop reading and start coding

Generally: read the **★** items for the current phase before you start the phase. Read the unmarked items as references when you hit a question. **Do not** try to read the whole list before Phase 0 — you'll never start. The goal is grounded understanding, not encyclopedic coverage.
