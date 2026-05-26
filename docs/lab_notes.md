# Lab notes — concepts learned and bugs encountered

One file, the whole project. The conceptual notes are the load-bearing
mental models I built in Phases 0-1; the bug log is every non-obvious thing
that cost real time across Phases 0-4. Both are written for *future me* and
for anyone reading the repo who wants to know where the bodies are buried.

---

## Part 1 — Concepts that changed how I think about inference

### The two-regime asymmetry is the whole project

One forward pass produces a logit distribution at *every* position. Training
uses all of them in parallel (the loss compares each position's prediction to
the known next token) — that's why training is parallel. Inference uses only
the *last* position's logits; the earlier ones are byproducts — that's why
inference is sequential. The entire field of adaptive inference is an attempt
to claw back some of that lost parallelism at decode time. Everything in this
repo is downstream of that one asymmetry.

### Two distinct wastes in naive decoding

- **Waste A — re-processing known tokens.** Naive greedy decoding recomputes
  K/V for the whole prefix every step. Fixed by the **KV cache**: per-step
  cost drops from O(t²) to O(t). Pure engineering.
- **Waste B — using the full model for an easy token.** Even with a perfect
  cache, every new token costs one full forward pass, including tokens a
  10× smaller model could have predicted. Fixed by **speculative decoding /
  Medusa / EAGLE / mixture-of-recursions**. Open research — this is what the
  project studies.

### Uncertainty is entropy, not top-1

Top-1 probability is a noisy proxy for uncertainty: two distributions with
the same top-1 = 0.5 can have wildly different tails. The honest measure is
Shannon entropy `H = -Σ p log p`. Confidence ≠ accuracy — the gap is
*calibration*, and it's why a probe might (in principle) beat raw confidence.
It mostly didn't (Part 1, Phase 3/4 result), but the distinction is real.

### "Easy" is a post-hoc label until you can predict it

A token having top-1 > 0.9 doesn't mean its compute was *unnecessary* — the
compute is *how you found out* it was easy. The interesting question is
whether you can predict easiness *before* paying the cost. ~70-85% of natural
text tokens are predictable enough to skip; that's the empirical fact that
makes speculative decoding economically real.

### Parallel-safety, defined precisely

For each position `t`, compare the model's `k` teacher-forced argmaxes
(the prefix sees ground truth at every step) against `k` autoregressive
greedy steps (the model sees only its own output). Position `t` is
**parallel-safe at threshold θ** if `mean_{j=1..k-1} agreement[t,j] ≥ θ`.
Two structural facts that bit repeatedly:

- The **j=0 column is always True** — TF and AR share identical context at
  step 0. Always restrict the reported rate to `j > 0`.
- **Position 0's attention entropy is structurally 0** under causal masking
  (one key to attend to), position 1 caps at `log 2`, etc. This is geometry,
  not model behavior. Drop position 0 from per-token attention aggregates.

### Agreement is not quality

The headline trap of the whole metric. `parallel_agreement` measures whether
two decode paths *agree*, not whether they're *correct*. An undertrained model
that emits the same token everywhere trivially "agrees with itself" — see the
Pythia-410m @ step 8 outlier (psf 0.405, mean-agreement 0.765) while every
other diagnostic screams degenerate. Always sanity-check agreement spikes
against an independent quality signal before believing them.

---

## Part 2 — Bug log (chronological-ish, by phase)

### B1. KV-cache memory arithmetic — trust the formula, not the asserted answer

An early reflection asserted a 7B model's KV cache at 4K context was "~8.6 GB".
Wrong. The formula `2 · L · H · head_dim · T · bytes` gives ~2.0 GB
(`2·32·32·128·4096·2`). The lesson became a project rule: **every quantitative
claim must come from a regenerable computation, never an asserted number.**
This rule later caught B6 and B7.

### B2. Pythia deep-layer attention returns NaN

`output_attentions=True` returns NaN in Pythia's deeper layers (≈9-11) on
both the eager and SDPA paths. Cause: the unfused softmax overflows — residual
magnitudes grow with depth, so `exp(QKᵀ)` overflows in fp16/bf16 and
`inf − inf` poisons the masked-position softmax. **Fix:** never use the
framework's attention. `hybrid_arch.attention.extract_attention` captures each
layer's input via a forward pre-hook and replays QKV + rotary + masked softmax
in fp32. Verified NaN-free on every layer of every checkpoint used.

### B3. macOS iCloud + Python 3.14 silently hides `.pth` files

The project lives under iCloud-synced `~/Documents`. iCloud marks files in
`.venv/` with `UF_HIDDEN`, and Python 3.14's `site.addpackage()` *silently
skips* hidden `.pth` files — so `import hybrid_arch` fails with
`ModuleNotFoundError` even though `pip install -e .` succeeded. Symptom:
the whole test suite errors at collection with "No module named hybrid_arch."
**Fix:** `chflags -R nohidden .venv` after every editable reinstall (iCloud
re-hides on every `pip install`). Linux/Colab is unaffected. This recurred
~4 times across the project; it's now reflexive.

### B4. `np.savez` silently appends `.npz` to the filename

The cache layer writes atomically (stage to `.tmp`, then `os.replace`). But
`np.savez("foo.npz.tmp", ...)` writes to `foo.npz.tmp.npz`, so the
`os.replace` then fails with FileNotFoundError. **Fix:** pass an open file
handle to `np.savez` so it keeps the exact filename.

### B5. The 2K-token slice doesn't fit in memory

The plan called for a 2K-token WikiText slice. The batched
`parallel_prediction_agreement` kernel runs one forward pass on a
`[n_positions, n_positions + j]` tensor; at n=2000 on Pythia-410m that's
~25 GB of activations per layer — over laptop RAM and the T4's 16 GB.
**Fix / deviation:** stepped down to 256 tokens (matches the Phase 1 slice;
252 × (k-1) = 756 binary observations per cell). Documented in every atlas.

### B6. Parallel HF pre-fetch made the sweep *slower*

During the Phase 2 410m sweep I tried to speed things up by pre-fetching the
remaining checkpoints in a second process. Both processes then contended for
the same unauthenticated HF download bandwidth, and one sweep cell ballooned
from ~150 s to **30 minutes**. **Lesson:** a single network origin doesn't
parallelize; the "optimization" was net-negative. Killed it; cells returned
to ~100-150 s once each checkpoint was already local.

### B7. The hybrid-bench `divergence` column was meaningless

First-pass bench reported `divergence_vs_pure_target ≈ 0.94`, which
contradicts the math: greedy speculative decoding is *provably exact* (its
committed output equals pure-target greedy, divergence 0). The bug: the
reconstruction iterated over *all drafted positions*, including the
rejected-step tails that spec-decode discards, and compared that garbage
against the pure-AR sequence. **Fix:** added `SpecDecodeTrace.committed_tokens`
(the actually-emitted stream), replaced the metric with `committed_divergence`
(now 0.000 on all domains — validates exactness), and added a test pinning
"committed stream == pure greedy decoding." Caught precisely because the
number violated a known invariant (rule from B1).

### B8. Agent worktrees forked from the wrong branch

When I tried to fan Phase 4 out to parallel sub-agents, the worktree
isolation created each worktree off `main` (pre-Phase-2) instead of the
feature branch, so none of the prerequisites existed. **Lesson:** for a
single-CPU compute-bound job there's no real parallelism to win anyway;
ran the two experiments as one chained background job instead. Don't reach
for multi-agent fan-out when the bottleneck is one machine's CPU.

---

## Part 3 — Reading that actually mattered

- *Fast Inference from Transformers via Speculative Decoding* (Leviathan et
  al., 2022), §3 — the accept/reject mathematics; `parallel_agreement` is its
  offline cousin.
- *When Attention Sink Emerges* (ICLR 2025) — the sink emerges in the same
  step window (128-1000) as parallel-safety; worth charting jointly.
- *Probing Classifiers: Promises, Shortcomings, and Advances* (Belinkov,
  2022) — why a probe should be small (a big probe recovers the property from
  anywhere and tells you nothing), and why "the probe is at chance" is a
  legitimate, informative result.
- EAGLE-3 (arXiv:2503.01840) — the SOTA acceptance-rate reference point. Not
  something to beat on free Colab; useful for calibrating expectations.
