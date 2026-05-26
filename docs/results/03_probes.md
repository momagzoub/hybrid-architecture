# Probing for parallel-safety, and what survives contact with a real verifier

*Hybrid Architecture, Phase 3. Pythia 70m / 160m / 410m × all layers, plus a
greedy speculative-decode benchmark on Pythia-1B + Pythia-160m. All numbers
regenerable from `src/scripts/phase3_*.py`.*

---

## TL;DR

Two results, one positive, one negative.

1. **Middle layers carry the parallel-safety signal.** A 50k-parameter MLP probe
   trained on Pythia-410m's L12 hidden state reaches 5-fold CV AUROC of
   **0.857 ± 0.106** on the offline parallel-safety label — slightly *above*
   the Phase 2 logistic-regression-on-attention baseline (0.845). The
   accuracy-vs-layer curve has the classic middle-layer peak: early layers
   carry noise, late layers carry the post-LN logit-side answer rather than
   the structural one.
2. **The offline probe does not transfer to real drafter-rejection.** Running
   greedy speculative decoding with Pythia-1B as the target and Pythia-160m
   as the drafter (64 drafted positions, 28 rejections, 0.562 accept rate),
   the pretrained L9 probe predicts rejection at AUROC **0.603 — essentially
   chance**. Two simple logit-side baselines beat it cleanly: `next_token_entropy`
   = 0.839, `1 − top1_probability` = **0.884**.

The polite read: "the offline metric is a clean diagnostic but a poor router."
The blunt read: if you only care about predicting spec-decode rejections, you
don't need a probe — you need the drafter's own confidence.

Both results matter for Phase 4.

---

## Setup

- **Models.** Pythia 70m (6 layers, 512-dim), 160m (12 layers, 768-dim),
  410m (24 layers, 1024-dim), 1b (16 layers, 2048-dim — *target only*).
  All fp32 on CPU, step 143000.
- **Probe.** `LayerProbe`: `LayerNorm → Linear(d→32) → GELU → Linear(32→16)
  → GELU → Linear(16→1)`. About 67k params at `d=1024`. BCE-with-logits,
  Adam + weight decay 1e-3, early stop on 80/20 val split.
- **Label.** The offline parallel-safety target from the Phase 2 cache:
  `mean(parallel_agreement[j=1..k−1]) ≥ 0.9` per position, on the 256-token
  WikiText slice. j=0 dropped because it's structurally True; position 0
  dropped because its attention row is a delta under causal masking.
- **Reproducing.** `pytest -q tests/test_probes.py tests/test_hidden_states.py`
  then `python src/scripts/phase3_{layer_depth_sweep,drafter_rejection}.py`.

---

## Result 1 — Layer-depth sweep: middle wins

![Probe AUROC vs layer depth](figures/07_probe_layer_depth.png)

For every `(size, layer)` we train one probe and run 5-fold stratified CV.

| Size  | Layers | Best layer | Layer fraction | Best CV AUROC | Final-layer AUROC |
|-------|------:|----------:|---------------:|--------------:|------------------:|
| 70m   |     6 |     1     | 0.17           | 0.831         | 0.777             |
| 160m  |    12 |     9     | 0.75           | 0.745         | 0.694             |
| 410m  |    24 |    12     | 0.50           | **0.857**     | 0.769             |

**Three observations.**

- **The 410m curve has the classic middle-layer hump.** AUROC climbs from
  ~0.70 at L0 to **0.857 at L12** (exact halfway), then drops back to ~0.77
  at the final layer. This matches the broader probing-classifier folklore
  that linguistic / structural properties peak around the middle of a
  transformer and that the final layers specialize toward the output-space
  prediction.
- **The final layer is not the right layer.** Naively you'd expect "the most
  informative state" to be the last one. It isn't. On 410m the gap between
  the middle probe and the final-layer probe is 0.088 AUROC — a *lot* for an
  effectively-free probe.
- **70m has its peak at L1 of 5.** With only 6 layers, "middle" is L2-L3,
  but the noise margin on 70m (n_positive ≈ 9 of 251) is large; the 0.83
  AUROC carries a sizeable error bar. 160m's curve is messier and the
  weakest of the three; that cell needs a larger eval slice before claiming
  much.

The 410m middle-layer result is the headline. It is also the most directly
useful one for downstream routing: if you're running a drafter and want a
cheap signal of which tokens will likely be parallel-safe, tap the
middle-layer residual — not the final logits.

---

## Result 2 — Real drafter-rejection: the probe doesn't transfer

The previous experiment trained the probe on the drafter's *self-agreement*:
"does the drafter's argmax at position `t` match what it would have
predicted under teacher-forcing?" That's an offline, label-free signal.

Phase 3 Step 4 asks the harder question: **does the same probe predict the
real accept/reject events of speculative decoding?** Target = Pythia-1b,
drafter = Pythia-160m, greedy on both sides, draft_k = 4, 16 steps, prompt
= 96 tokens of WikiText.

Outcome of the spec-decode run:

| metric              | value |
|---------------------|------:|
| drafted positions   |    64 |
| accepted            |    36 |
| accept rate         | 0.562 |
| reject events       |    28 |

The drafter's offline parallel-safety probe was trained on 160m's own
self-prediction label. We now use it (untouched) to predict rejection
against the 1b target.

| Predictor                      | AUROC for rejection |
|--------------------------------|--------------------:|
| random                         |               0.604 |
| Phase 3 probe (160m, L9)       |               0.603 |
| Phase 1 baseline (`entropy`)   |               0.839 |
| **1 − top1 probability**       |           **0.884** |

![Drafter-rejection ROC](figures/08_drafter_rejection_roc.png)

The probe is at chance. Both simple logit-side baselines clear AUROC 0.84.
A reviewer asked to bet on this would not pick the probe.

**Why the gap?** Three candidate explanations, all plausible:

1. **Different label distributions.** Self-agreement is a property of the
   *drafter*. Drafter-rejection is a property of the *target's* opinion
   about the drafter. The two distributions overlap but are not the same;
   a token can be drafter-confident and target-disagreeing if the target
   simply prefers a different completion, even one of equal quality.
2. **The probe overfits structure the target doesn't share.** The drafter's
   middle-layer geometry is shaped by its own training; the target's
   notion of "right answer" is shaped by 6× more parameters. Linear
   probes of one model's representations onto another model's labels are
   classically unreliable.
3. **Sample size.** 64 drafted positions is small. The AUROC error bars
   are wide and the "0.603 ≈ random" reading might soften with 10× more
   data. We'll know once Phase 4 streams a longer evaluation.

The cleanest take: when you have direct access to the drafter's logits (which
you do during spec-decode), the drafter's *own* confidence is the cheap
predictor that already works. The probe earns its keep elsewhere — for
example, in the offline analysis use case the Phase 2 atlas demonstrates.

---

## What Phase 4 inherits

Three artifacts the hybrid-decoder demo can consume directly:

1. **`src/hybrid_arch/spec_decode.py`** — a minimal greedy spec-decode
   implementation that records per-position accept/reject and the
   drafter's hidden states. Useful as a controlled testbed regardless of
   what router Phase 4 ends up using.
2. **42 pretrained `LayerProbe` checkpoints** in
   `src/hybrid_arch/probe_checkpoints/`, each ~50 KB, self-describing
   via JSON sidecar. The 410m L12 probe is the strongest offline
   diagnostic the project ships; it's also explicitly *not* the right
   router for spec-decoding on its own.
3. **The clear empirical fact that `1 − top1` ≈ AUROC 0.88 on
   drafter-rejection.** Any router Phase 4 builds should include the
   drafter's own logit confidence as its baseline feature. The "what does
   the probe add on top of that?" question is the Phase 4 ablation.

The phrasing of Phase 4 should reflect Phase 3's null: *we are not
shipping a router that beats `1 − top1`*. We are shipping a tooling
stack and an honest report on which signals predict what.

## Limitations

- One 256-token WikiText slice for training and one 96-token slice for
  the spec-decode benchmark. Phase 4 wants a longer corpus before any
  router claim.
- Single seed throughout; no robustness check across
  `(seed, slice_position)` pairs.
- `k = 4`, `θ = 0.9`, `draft_k = 4` are not swept. The
  agreement-vs-quality artifact at Pythia-410m@step8 in the Phase 2
  cache is filtered out here (step 143000 only).
- No probe has been trained on the drafter-rejection labels directly.
  If Phase 4 wants a probe that actually beats `1 − top1`, fitting on
  the right label is the obvious first move; we deliberately did not
  do that here because the value of the offline probe is itself the
  question.
