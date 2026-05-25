# Parallelism Emerges

*An attention-pattern atlas of how language models learn what to compute in parallel.*

> **Status:** Active research — Phase 1 (Measure) complete; Phase 2 (Patterns) next. See [`PROJECT_PLAN.md`](./PROJECT_PLAN.md) for the roadmap.

---

## TL;DR

Modern language model inference is *sequential* — one token at a time — even though training is fully parallel. A growing body of work (speculative decoding, Medusa, EAGLE-3, Mixture-of-Recursions) shows that **not every token actually needs sequential processing**. This project asks: when, during a model's pretraining, does the capacity for per-token parallelism *emerge*, and what attention-pattern signatures predict it?

We use [EleutherAI's Pythia](https://github.com/EleutherAI/pythia) — 154 training checkpoints per model, sizes 70M-410M — to study the developmental trajectory that big-lab work cannot, because they release only final checkpoints. The output is a diagnostic toolkit and an *atlas* of parallel-safety signatures, designed to be useful upstream of any adaptive-inference router.

## Why this matters

Inference cost dominates the economics of deployed AI. Adaptive techniques (speculative decoding, mixture-of-depths) save 2-6× at the cost of complex routers that are largely opaque. We don't yet have good tools to ask: *which tokens are easy, why are they easy, and how reliably can we tell?* This project is one attempt to build those tools.

## What's here

```
.
├── CLAUDE.md            ← project guide (start here)
├── PROJECT_PLAN.md      ← 20-week roadmap
├── GITHUB_GUIDE.md      ← first-time Git/GitHub usage (for me)
├── docs/
│   ├── reading_list.md  ← ordered concepts → papers
│   └── results/         ← final plots, tables, writeups
├── notebooks/           ← exploratory analysis (numbered)
└── src/hybrid_arch/     ← importable library code
```

## Reproducing the results

> Hero plot and reproduction details land at the end of Phase 2. For now, see individual notebooks.

```bash
# Coming soon
git clone https://github.com/momagzoub/hybrid-architecture.git
cd hybrid-architecture
uv sync                      # or: pip install -e .
jupyter lab notebooks/
```

## Relation to prior work

This project is **upstream** of, not competitive with, the following:

- **EAGLE-3** ([arXiv:2503.01840](https://arxiv.org/html/2503.01840v1)) — speculative decoding with 0.75-0.85 acceptance, 3-6× speedup.
- **Mixture-of-Recursions** ([NeurIPS 2025](https://arxiv.org/abs/2507.10524)) — per-token adaptive recursive depth.
- **Mixture-of-Depths** ([Raposo et al., 2024](https://arxiv.org/pdf/2404.02258)) — per-token layer skipping.
- **DeepSeek MTP** — multi-token prediction at scale.

We aim to produce *diagnostic tooling and developmental analysis* that complements these systems — not to compete on raw throughput.

## Citing

Once published. For now, the repo URL is the citation.

## Author

Mohamed Magzoub ([m0hamed@mit.edu](mailto:m0hamed@mit.edu)) · [github.com/momagzoub](https://github.com/momagzoub)

## License

MIT — see [`LICENSE`](./LICENSE).
