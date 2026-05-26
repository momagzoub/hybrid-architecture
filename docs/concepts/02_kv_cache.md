# 02 — KV cache, prefill, and decode

> Concept notes for Phase 0, Notebook 01 (`notebooks/01_kv_cache_walkthrough.ipynb`).
> Format: my first-pass answer in my own words, then notes from review.
> Date: 2026-05-25.

I started these answers "entirely unsure." That's the right frame — these questions are about quantitative inference engineering and I'm seeing it for the first time. The review notes are where the real learning lives.

---

## Q1. The cached version still gets slower as the sequence grows. Why?

**My first-pass answer.** As we see from the curve, it grows linearly since the prefill phase which is needed to populate the task performs a 0(n) operation, where n is the token length.

**Notes from review.**
The instinct ("linearly") is right. The *mechanism* I named is wrong: prefill is a one-time cost paid before the loop starts, so it can't be what makes the curve climb.

The real reason — the one the hint was pointing at — is that **every decode step has to read the entire KV cache** to compute attention. At step `t` the cache holds K and V for `t` previous tokens, so the per-step cost is `O(t)`. Per-step work grows linearly with how much I've already generated. Cumulative wall-clock over `T` steps is `O(T²)` — a small quadratic with a tiny constant, which is why my cached curve looks nearly linear on the visible scale.

Corrected mental model: the cache is a moving target, not a fixed cost. Every step does slightly more attention work than the last.

---

## Q2. KV cache size for a 7B model at 4K context, fp16

**My first-pass answer.** No idea.

**Notes from review.**
"No idea" is too pessimistic. It's just arithmetic:

```
KV cache size = 2 × L × H × head_dim × T × bytes_per_element

with L=32, H=32, head_dim=128, T=4096, fp16 = 2 bytes:

= 2 × 32 × 32 × 128 × 4096 × 2 bytes
= 2,147,483,648 bytes
≈ 2.0 GB
```

The leading 2 is because both K and V are stored. The trailing 2 is bytes per fp16 element.

**Comparison to model weights:** the 7B model itself is ~14 GB at fp16. So at 4K context, the cache is ~14% of the weight memory.

**The crossover question:** when does the cache equal the weights in memory? Cache scales linearly with `T`; weights are constant. Per-token cache cost = `2 × 32 × 32 × 128 × 2 ≈ 512 KB/token`. So `cache = weights` when `T ≈ 14 GB / 512 KB ≈ 27,000 tokens`. At 100K context the cache is ~4× the model itself.

**Why this matters:** this is *the* fact that unlocks intuition for long-context inference. Every serving system spends huge effort on cache management precisely because the cache dominates memory at long context. It also motivates architectural responses: Grouped-Query Attention (Llama-2 70B, Llama-3) shares K/V across query heads to shrink the cache; Multi-Latent Attention (DeepSeek) compresses it differently; sliding windows just throw old tokens away.

> *Note: an early reflection on this concept asserted an "expected answer of ~8.6 GB" — that was wrong. The correct number is ~2.0 GB. Trust the formula, not an asserted answer.*

---

## Q3. Prefill vs decode — which is bandwidth-bound, which is compute-bound?

**My first-pass answer.** Decode is sensitive to memory bandwidth since the cache memory depends on the size of the prompt passed in. The computational hardship is something that affects the decode, since we pass in new tokens that we rely on the cache for computation.

**Notes from review.**
I got the bandwidth half-right (decode is bandwidth-bound) but the reasoning is muddled and I conflated decode's compute and bandwidth bottlenecks.

The clean framing:

**Prefill is compute-bound.** One forward pass on a prompt of length `T` does roughly `T × P` FLOPs (where `P` is the parameter count) but reads the model weights from VRAM only *once* — that read is amortized over `T` positions of work. Arithmetic intensity (FLOPs per byte read) is high. The GPU's compute units saturate.

**Decode is memory-bandwidth-bound.** Each decode step processes exactly one new token. To process it, the GPU still has to read *every model weight* from VRAM (~14 GB for a 7B model in fp16). But it does only `P` FLOPs of useful work on that token. Arithmetic intensity is low — most time is spent reading weights, very little computing. The memory bus is the bottleneck, not the compute units.

**Why this matters operationally:** decode can be sped up dramatically by **batching** — running many sequences' decode steps in one kernel launch, so each weight read is amortized across N tokens of useful work. This is half of what vLLM/TGI/TensorRT-LLM optimize for. Prefill doesn't benefit from batching as much because it's already compute-saturated.

---

## Q4. Sketch a scenario where you'd drop tokens from the KV cache. What's the cost?

**My first-pass answer.** When we drop tokens, we in turn allow the model to lose some of its inference ability. Simply put, if the inference procedure populates tokens that are populated in the cache, the model's cognitive ability relies on the bandwidth of that cache, but it dramatically decreases when we drop some of the tokens to save memory.

**Notes from review.**
Intuition is in the right place — dropping tokens degrades model behavior — but vague. Sharpening:

The mechanical cost of dropping a token: **the model can no longer attend to that position.** Whatever information was there is gone from the model's view. The interesting question is *which* tokens are safe to drop.

Four named strategies the field uses:

**Sliding window** — keep only the most recent `K` tokens, drop everything older. Cheap, predictable. Cost: the model can't reference anything past `K` tokens ago. Long-range reasoning collapses.

**Attention sinks** — keep the first 4-ish tokens *plus* a recent window. Empirically the first few tokens act as "anchors" that other positions attend to regardless of content. Drop them and attention destabilizes. (This is the "When Attention Sink Emerges" paper on my reading list.) Cost: medium-distance context is lost but model behavior stays stable.

**Eviction by attention score** — track which tokens have been most attended-to, drop the rarely-attended ones. Smarter heuristic, costs runtime bookkeeping. Cost: greedy, can be wrong about which tokens future queries will need.

**Quantization (not dropping)** — store K/V in int8 or int4 instead of fp16. Cost: precision loss in attention scores; some tasks degrade.

**A concrete scenario:** a 100K-token chat session is eating VRAM, but the user is now asking short questions about the most recent exchange. Sliding-window + attention-sink eviction frees the memory and answers their question well. The cost lands later, when they suddenly ask "what was the first thing I told you?" — that information has been thrown away.

---

## What I want to remember from this notebook

1. **The KV cache is a moving target.** Per-step decode cost is `O(t)`, not constant, because the cache itself is being read every step. Cumulative cost is quadratic in tokens generated.
2. **`2 × L × H × head_dim × T × bytes` is the memory formula** for the KV cache. Burn it in. For a 7B model in fp16, that's ~512 KB per token of context — and that's *per sequence* in a batch.
3. **Cache dominates weights at long context.** Crossover for 7B is around 27K tokens. At 100K context, cache is ~4× the model. This is why long-context inference is so memory-hungry.
4. **Prefill is compute-bound; decode is memory-bandwidth-bound.** Decode benefits enormously from batching; prefill less so. This is half of what every serving system optimizes for.
5. **Dropping cache tokens trades memory for forgetting.** Sliding window, attention sinks, score-based eviction, quantization — each has a different cost profile. The interesting research is in *which tokens are safe to drop* and *when*.
6. **Verify quantitative claims before trusting them.** "Expected answer" lines can be wrong; the formula is the ground truth.

Next notebook: `02_attention_extraction.ipynb` — extracting attention weights by hand, building intuition for Phase 1's metric library.
