# hybrid-arch

[![CI](https://github.com/momagzoub/hybrid-architecture/actions/workflows/ci.yml/badge.svg)](https://github.com/momagzoub/hybrid-architecture/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A diagnostic toolkit for studying **parallel-safety** in language models — when, during
pretraining, a model learns which tokens are easy enough to skip the full sequential decode.

![Emergence curve](docs/results/figures/02_emergence_curve.png)

## Findings (Pythia 70m–1b, all reproducible)

- **Emerges early and abruptly** — between training steps 128 and 1000, then flat for 142k more.
- **Predictable from per-(layer, head) attention** — AUROC 0.85 on Pythia-410m; the *aggregate* is noise (|r| < 0.11), the signal lives in specific heads, mid-network.
- **Code ≫ prose** — 3.9× more parallel-safe on MBPP than WikiText.
- **Honest null** — in real speculative decoding, the drafter's own `1 − top1` predicts rejections at AUROC 0.88; a learned probe adds nothing, even fitted on the labels.

**→ Read the [blog post](docs/blog/parallelism-emerges.md)** for the full story (~10 min). Detailed writeups: [emergence](docs/results/02_emergence_atlas.md) · [probes](docs/results/03_probes.md) · [hybrid decoder](docs/results/04_hybrid_decoder.md).

## Install & quickstart

```bash
git clone https://github.com/momagzoub/hybrid-architecture.git
cd hybrid-architecture && pip install -e ".[dev]" && pytest -q
```

```python
from hybrid_arch import load_pythia, metric_battery

model, tok = load_pythia("160m", step=143000)
ids = tok("The quick brown fox jumps over the", return_tensors="pt").input_ids
out = metric_battery("160m", 143000, "demo", ids, k_parallel=4, model=model)
print("parallel-safety rate:", out["parallel_agreement"][..., 1:].float().mean().item())
```

Python 3.10+, CPU-only is fine. Every figure and CSV in `docs/results/` regenerates from
the `src/scripts/phase*.py` runners (idempotent over an on-disk cache).

## Layout

```
src/hybrid_arch/   library — attention extraction, metrics, cache, probes, hybrid decoder
src/scripts/       phase{2,3,4}_*.py experiment runners
docs/              blog post, result atlases + figures, lab notes (concepts + bug log)
tests/             107 tests — metric correctness, cache, NaN-free attention, spec-decode exactness
```

## Relation to prior work

Upstream of — not competitive with — [EAGLE-3](https://arxiv.org/html/2503.01840v1),
[Mixture-of-Recursions](https://arxiv.org/abs/2507.10524), and
[Mixture-of-Depths](https://arxiv.org/pdf/2404.02258). Those systems consume a per-token
difficulty signal; this repo characterizes one, shows when it emerges, and reports honestly
where it does and doesn't pay off.

## License

MIT · [Mohamed Magzoub](mailto:m0hamed@mit.edu) · [github.com/momagzoub](https://github.com/momagzoub)
