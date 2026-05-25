# 01 — Decoding basics

> Concept notes for Phase 0, Notebook 00 (`notebooks/00_warmup_decoding.ipynb`).
> Format: my first-pass answer in my own words, then notes from review.
> Date: 2026-05-25.

The point of these notes is not to be polished — it's to capture *what I knew before reading anything carefully* so I can come back later and see what shifted. Future me: trust the "first-pass" sections to tell you where your intuition was, and the "notes from review" sections to tell you where it needed to sharpen.

---

## Q1. The forward pass produces a logit distribution at every position, not just the last. Why don't we use the others during decoding?

**My first-pass answer.** Logits are unnormalized numerical values that represent the "probability" that this token is a more likely candidate for the next word. Since our prompt is broken down into preprocessed tokens that we know exactly what they are, our target is the subsequent token — which is the only one we use the logit distribution for.

**Notes from review.**
The instinct is right: we already know the prompt tokens, so their predictions are unused at decode time. The hint about the causal mask was pointing at the bigger picture: because of the causal mask, the logits at position `t` are the model's prediction for *what comes at position* `t+1`, computed using only positions `0…t`. At decode time, positions `0…t-1` are tokens I already chose, so I don't care what the model thought should go there.

The deeper insight to remember: the same forward pass has **two regimes**.
- **Training:** every position's logits are used. The loss compares each position's prediction to the known next ground-truth token, all positions in parallel. *That's why training is parallel.*
- **Inference:** only the last position's logits matter. The earlier positions are byproducts. *That's why inference is sequential.*

The asymmetry between these two regimes is the entire project.

---

## Q2. What is the relationship between `chosen_prob` in the log above and the model's uncertainty?

**My first-pass answer.** The higher the probability is, the lesser the model's uncertainty is.

**Notes from review.**
Direction is right, but top-1 probability is a *noisy* proxy for uncertainty. Two distributions with the same top-1 = 0.5 can be very different:
- Remaining 0.5 spread across 50,000 vocabulary entries → very uncertain, almost guessing.
- Remaining 0.5 sitting entirely on one runner-up → relatively certain, just deciding between two candidates.

The proper measure is **Shannon entropy** of the full distribution: `H = -Σ p log p`. That's why `src/hybrid_arch/metrics.py` will compute entropy, not just top-1, when we get to Phase 1.

Second subtlety, file for later: a confident model can still be *wrong*. Confidence and accuracy are different axes. The gap between them is called **calibration**, and it'll matter when we evaluate probes in Phase 3.

---

## Q3. Suppose a generation has 50 tokens, and 40 of them have `chosen_prob > 0.9`. What does that suggest about how much of the model's compute was "necessary"?

**My first-pass answer.** If the chosen_prob for 40 tokens is > 0.9, that means the model is certain about that chosen token, which deems the compute for that unnecessary.

**Notes from review.**
This is the thesis seed and I found it — but there's a precision to add: the compute wasn't *unnecessary*, it's *what told me the model was confident*. I don't know a token is easy until I've already done the work to find out.

The harder, actually-interesting question the field is built around: **can we predict which tokens will be easy before running the full forward pass?** If yes, route those through a cheaper path (a tiny drafter model, a parallel multi-token guess, a shallower computation). For natural language, roughly **70-85%** of tokens turn out to be predictable enough to skip — which is why speculative decoding can hit 3-6× speedups in production.

My project asks the developmental version of this: *what makes those tokens predictable, and when during training does the model learn to make them predictable?*

---

## Q4. Why is this loop so slow? Where exactly is the wasted work?

**My first-pass answer.** The wasted work is carried out when computing tokens the model already "has the answer to".

**Notes from review.**
I conflated two distinct wastes. The literature treats them as different problems with different solutions, and learning to keep them separate is the conceptual scaffold of the whole field:

**Waste A — re-processing known tokens.** Every iteration of `greedy_decode_naive`, the model recomputes the K and V projections for the *entire* prefix: token 0, token 1, …, all of them. Nothing about those earlier tokens has changed between iterations. The K/V values for token 0 in layer 5 in iteration 30 are identical to what they were in iteration 1. Pure waste.

*Fix:* **KV caching.** Cache K and V for every previous token at every layer, and the new token's query attends over the cached values. Per-step cost drops from O(t²) to O(t). That's notebook 01.

**Waste B — using the full model for an easy token.** Even with a perfect KV cache, I'm still doing one full forward pass per new token. Many of those tokens are "easy" — a 10× smaller model could have predicted them. Running the full big model on them is wasted capacity.

*Fix:* **Speculative decoding / Medusa / EAGLE / Mixture-of-Recursions.** This is the active research frontier and what my project studies.

My first-pass answer pointed at Waste B. Waste A is the bigger and easier-to-fix culprit, and notebook 01 will measure the speedup of fixing it (probably 5-10× on this small model).

---

## What I want to remember from this notebook

1. **One forward pass produces logits at every position.** Training uses all of them in parallel; inference uses only the last. That asymmetry is the project.
2. **Top-1 probability is a crude uncertainty proxy.** Use entropy. Confidence ≠ accuracy.
3. **A token being "easy" doesn't mean the compute was unnecessary** — it means the compute *was the way I found out the token was easy*. The interesting question is whether I can predict easiness *before* paying the cost.
4. **There are two wastes in naive decoding.** KV caches fix the first one (engineering). Adaptive inference research fixes the second one (open problem).
5. **70-85% of tokens in natural text are predictable enough to skip** with a smaller drafter. That's the empirical fact that makes adaptive inference economically real.

Next notebook: `01_kv_cache_walkthrough.ipynb` — see Waste A get fixed, with timing measurements.
