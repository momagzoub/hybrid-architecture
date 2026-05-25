# 04 — The metric library

> Concept notes for Phase 1 (`src/hybrid_arch/metrics.py` + `attention.py` + `viz.py`).
> Format: same as Phase 0 — my first-pass answer, then notes from review, then a cleaned answer.
> Date: 2026-05-25.

This is the Phase 1 writeup of what each metric *means*. Phase 0 docs were about understanding mechanisms (decoding, KV cache, attention). Phase 1 is about understanding *measurements* — what number do we actually compute, and what does a high or low value tell us about the model's state.

Skeleton below — questions only. I fill in the first-pass answers in my own words, then bring them back for review. Resist the urge to look at the function implementations before answering: the point is to write what I think it means before I check.

---

## Q1. `next_token_entropy(logits)` returns a number per position. What does that number actually tell you about the model's state at that position? Give a concrete example of when it would be near 0, and one when it would be near `log(50257) ≈ 10.8`.

**My first-pass answer.**

(write your answer here)

**Notes from review.**

(left blank — fill in after review)

**Cleaned answer.**

(left blank)

---

## Q2. In the metric zoo correlation matrix, `next_token_entropy` and `top1_probability` correlate at r = -0.92 — very strongly. Does that mean one of them is redundant? Can you think of a position where one would be informative and the other wouldn't, or are they essentially the same measurement?

**My first-pass answer.**

(write your answer here)

**Notes from review.**

(left blank)

**Cleaned answer.**

(left blank)

---

## Q3. `attention_entropy` and `attention_concentration` are companion metrics — one is `H = -Σ p log p`, the other is the cumulative mass on the top-k attended positions. In the metric zoo, `attention_entropy` and `attention_top3` correlate at r = -0.99 — nearly perfectly. Why are they so close? What is `attention_top1` actually measuring that `attention_entropy` isn't?

**My first-pass answer.**

(write your answer here)

**Notes from review.**

(left blank)

**Cleaned answer.**

(left blank)

---

## Q4. Position 0 has *structural* attention entropy of exactly 0 under causal masking — it's a fact about geometry, not about the model. Position 1 is capped at `log(2)`, position 2 at `log(3)`, and so on. When you do the Phase 2 cross-checkpoint analysis, how does this constraint affect what you can and can't compare between positions? What would be the wrong way to handle it?

**My first-pass answer.**

(write your answer here)

**Notes from review.**

(left blank)

**Cleaned answer.**

(left blank)

---

## Q5. `parallel_prediction_agreement(model, input_ids, k)` returns a boolean per (position, lookahead step). The j=0 column is *always* True — the test suite confirms this as a structural invariant. In your own words: why is that, and what does it tell you about how to *report* parallel-agreement as a single number? Should you average over all (t, j), or restrict to j > 0?

**My first-pass answer.**

(write your answer here)

**Notes from review.**

(left blank)

**Cleaned answer.**

(left blank)

---

## Q6. The correlation matrix shows `parallel_agreement` correlated with `next_token_entropy` at r ≈ -0.43, but with *all* attention-side metrics at |r| < 0.11 in aggregate. If that pattern holds across more checkpoints and datasets in Phase 2, what would that mean for the project thesis? And — separately — what does the small-but-nonzero correlation with entropy (-0.43, not -1.0) suggest about whether entropy alone could replace a learned probe in Phase 3?

**My first-pass answer.**

(write your answer here)

**Notes from review.**

(left blank)

**Cleaned answer.**

(left blank)

---

## What I want to remember from this phase

(write 4-6 bullet points after answering the questions above)

---

**Phase 1 is done when this doc is filled in.** Until then, the metric library exists and works, but my understanding of what it measures is captured only in code — not in my own words. Code that I can't explain back is fragile knowledge.
