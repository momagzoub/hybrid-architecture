# Phase 5 Handoff — Hybrid Architecture

Self-contained brief for the engineer (or assistant) picking up Phase 5 —
the polish-and-publish phase. Date authored: 2026-05-25, at the close of
Phase 4.

## 0. How to use this document

Phases 0-4 are done. The research is finished and the artifacts exist; what
remains is making the project *legible to someone who finds it cold*. Phase 5
is not about new results — it's about the README, the blog post, and the
one-paragraph pitch that makes an inference researcher star the repo.

Read the four atlases in `docs/results/` before writing any prose — the blog
post is a distillation of them, not a new document.

## 1. State of the repo at Phase 4 close

```
hybrid-architecture/   (v0.4.0, 107 tests, CI green on 3.10 + 3.12)
├── README.md, AGENTS.md, PROJECT_PLAN.md
├── PHASE_5_HANDOFF.md
├── docs/results/
│   ├── 02_emergence_atlas.md     ← Phase 2
│   ├── 03_probes.md              ← Phase 3
│   ├── 04_hybrid_decoder.md      ← Phase 4
│   ├── 0*.{csv,manifest.json}    ← every number, regenerable
│   └── figures/*.png             ← 10 publication figures
├── src/hybrid_arch/
│   ├── attention.py   cache.py   checkpoints.py   metrics.py
│   ├── probes.py      spec_decode.py   hybrid.py   viz.py
│   └── probe_checkpoints/   ← 42 trained probes (~5 MB)
├── src/scripts/   ← phase2_*.py, phase3_*.py, phase4_*.py (all idempotent)
├── tests/         ← 107 tests
└── .github/workflows/ci.yml
```

## 2. The story, in one paragraph (for the blog post)

We measured *parallel-safety* — how often a small LM's greedy prediction
matches what it would produce under teacher-forcing — across Pythia's
training checkpoints. It emerges abruptly between training steps 128 and
1000; it's predictable from per-(layer, head) attention (AUROC 0.85 on
410m, with the middle layer carrying the signal); it's 3.9× higher on code
than prose. But when we tested whether that offline signal predicts *real*
speculative-decoding rejections, it failed — the drafter's own `1 − top1`
confidence is the signal that matters, and a probe adds nothing even when
fitted directly on the rejection labels. The honest punchline: parallel-
safety is a real, measurable, emergent property with a clean attention
signature, and the cheap thing (logit confidence) is already the right
router for the obvious application.

## 3. Phase 5 tasks

1. **README hero polish.** The emergence-curve figure is already the hero.
   Add a one-line "what is this" above the fold and a 3-bullet "findings"
   block. Tested install in a *fresh* Colab (the `pip install -e ".[dev]"`
   path + `pytest -q`).
2. **Blog post** (2000-4000 words). Long-form distillation of the four
   atlases. Lead with the emergence curve, end with the honest spec-decode
   null — the null is the most credible part of the story. Publish on
   GitHub Pages or a personal domain; link from the README.
3. **One-figure social thread.** Emergence curve + one paragraph + repo
   link. The null result is the hook ("we built a probe, it didn't beat a
   one-line baseline, here's why that's the interesting part").
4. **Optional workshop submission.** ICLR/NeurIPS efficient-inference or
   interpretability workshops. The developmental-checkpoint angle is the
   novel contribution; frame around that, not the (negative) router result.

## 4. What NOT to do

- **Don't run new experiments to "improve" the null.** The spec-decode null
  is a feature of the writeup, not a bug to fix. Fitting a bigger probe to
  beat `1 − top1` would be the exact over-claiming the project has avoided
  for five phases.
- **Don't add notebooks.** The repo is scripts + library + atlases; keep it
  that way.
- **Don't inflate the throughput numbers.** The honest framing (spec-decode
  wins iff acceptance is high; CPU eager-mode dominates) is load-bearing for
  credibility with inference engineers.

## 5. Where to start

> Read `PHASE_5_HANDOFF.md` and the four atlases in `docs/results/`. The
> research is done — Phase 5 is polish only. Start with the README
> above-the-fold block and a fresh-Colab install test, then draft the blog
> post as a distillation of the atlases. Do not run new experiments.
