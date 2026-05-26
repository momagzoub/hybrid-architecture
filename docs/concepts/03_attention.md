# 03 — Attention extraction, softmax, sinks, and head patterns

> Concept notes for Phase 0, Notebook 02 (`notebooks/02_attention_extraction.ipynb`).
> Format: my first-pass answer in my own words, then notes from review, then the cleaned answer.
> Date: 2026-05-25.

This is the one where the scaffold-before-answer pattern really kicked in. I started off mostly unsure on every question; the review process walked me through them by hinting rather than answering, and I wrote each final version myself. Preserving the first-pass attempts here is the point — they show where my intuition was *before* I had the concepts in hand, which is what makes the review notes useful.

---

## Q1. Pick any cell of the attention tensor, say `attentions[3][0, 3, 4, 2]`. In plain English, what does this number represent?

**My first-pass answer.** It represents the softmaxed value of the logit for that specific cell.

**Notes from review.**
Technically true but doesn't *answer the question*. Every cell in any attention tensor is a post-softmax value — that description doesn't distinguish *this* cell from any other. The question is asking me to decode the indices and describe what *this specific cell* means semantically.

The shape of the indexing is `attentions[layer][batch, head, query_position, key_position]`. Decoding:

- outer `[3]` = layer 3 of the model
- inner `[0]` = batch 0 (the only sequence in this batch)
- inner `[3]` = attention head 3 of that layer
- inner `[4]` = query position 4 (the token doing the attending)
- inner `[2]` = key position 2 (the token being attended to)

The actor in attention is the *query*, not the batch. Attention has direction: from a query position TO a key position.

**Cleaned answer.**
> *"This is the attention weight that query position 4 (in batch 0) placed on key position 2, in head 3 of layer 3."*

---

## Q2. Why must row sums be 1.0 but column sums aren't constrained to anything in particular?

**My first-pass answer.** You group the rows into a tuple by converting each attention value from NaN to a number with a default equal to zero.

**Notes from review.**
Off — I pattern-matched on the `tuple(torch.nan_to_num(...))` workaround code, but that line is patching over a transformers bug. It's not what *causes* row sums to be 1.0; it's just a cleanup step. Without the bug, no NaN, no `nan_to_num`, and the row sums would *still* be 1.0.

The actual reason lives inside the attention computation, in the model itself: **softmax**. I'd seen softmax in notebook 00 (the `F.softmax(next_token_logits, dim=-1)` line that turned logits into probabilities) but I hadn't internalized it. Softmax takes a vector of any real numbers and turns it into a probability distribution — exponentiate each entry, then divide each by the sum of all the exponentials. The denominator is shared across entries, which is what forces the outputs to sum to exactly 1.

In attention specifically: the raw scores `Q @ K^T` are real numbers (can be huge or tiny, positive or negative). Softmax is applied **per row** (over the key axis). That makes each row a probability distribution.

Why columns aren't constrained: because softmax was applied *per row*, the rows are tied together internally (entries share a denominator) but the rows are *independent of each other*. Nothing normalizes across rows, so columns have no constraint — multiple queries could all attend strongly to position 0, or no query might attend to position 5. Both are allowed.

**Cleaned answer.**
> *"Each row of the attention matrix is the output of a softmax across the keys, so its entries form a probability distribution that sums to 1.0. Column sums aren't constrained because each column's entries came from different rows, each softmaxed independently of the others — no operation ties columns together."*

---

## Q3. The first column tends to be bright across many rows (the attention-sink phenomenon). If you dropped the first token from the KV cache, what failure mode would you expect, and how would you measure it?

**My first-pass answer.**
- (a) The attention alters in multiple cells, causing their attentions to increase.
- (b) Compute the diagonal's attentions?

**Notes from review.**
(a) — I'd described the *mechanical* consequence (yes, softmax renormalizes when a key disappears) but not the *failure mode* in the model's behavior. Many attention heads across many layers rely on the sink as a stable anchor. Removing it doesn't just shift weights; it *destabilizes* the attention pattern the model learned to depend on. The empirical finding (Xiao et al., "Efficient Streaming Language Models with Attention Sinks") is that the failure is *uniform and severe* — generated text often collapses into gibberish, not patchy or partially-coherent output. The reason it's uniform: the sink is relied on in many layers and heads simultaneously, so the disruption is everywhere at once.

(b) — The diagonal is a structural feature (self-attention strength), not a measure of output quality. What I want is a comparison of the model's *generated outputs* with vs without the sink, condensed into a number. The conventional metric is **perplexity** — a scalar that captures "how surprised was the model by this sequence" (lower = the model assigned higher probability to the tokens, meaning it was more confident). Comparing perplexity of generations with sink intact vs sink dropped would quantify the quality collapse. My step-by-step logit-divergence idea was a valid alternative formulation; perplexity is just the compact single-scalar summary the field uses.

**Cleaned answer.**
> *"(a) Generated output would degrade — likely collapse into incoherent or gibberish text — because many attention heads across many layers rely on the first-position sink, and removing it forces softmax to redistribute that weight unpredictably onto other tokens across all layers simultaneously.*
> *(b) Run the same decoder with and without the first token's K/V in cache, generate continuations of identical length, and compute perplexity (or compare per-step logit distributions) on both. Sharply higher perplexity, or large step-by-step divergence, would confirm the failure."*

---

## Q4. Looking at the four-head grid (layer 3, heads 0/3/7/11): pick two heads with visually different patterns and describe what each appears to be tracking.

**My first-pass answer.** Head 3 appears to be tracking the same token, meaning it places all its attention on one token for each column. Head 0 seems to be placing attention on the previous token for each column, except the first and last.

**Notes from review.**
Both observations correspond to *named* attention-head archetypes in the interpretability literature.

**Head 0** is a "previous-token head": the bright sub-diagonal means query position `i` attends most strongly to key position `i−1`. The first row is an exception because position 0 has no previous token to attend to. These heads are extremely common in transformers — they're useful for local pattern matching, bigram-like relationships, and propagating information from adjacent tokens forward.

**Head 3** is a *self-attending* (sometimes called "identity") head: my initial description was a bit imprecise; on closer look the bright pattern is the **main diagonal**, meaning query position `i` attends most strongly to itself. Functionally, this gives the network a way to preserve a token's representation while *other* heads in the same layer mix in context. Think of it as one head saying "keep this token's information intact," while heads 0, 7, 11 say "mix in the previous token" or "look back at the sink." Different heads, complementary roles.

**Cleaned answer.**
> *"In layer 3 head 0, the bright sub-diagonal indicates a 'previous-token' head — query position i attends most strongly to key position i−1. These are common in transformers and useful for local bigram-style relationships. In layer 3 head 3, the bright main diagonal indicates a 'self-attending' or identity head — query position i attends most strongly to itself, effectively preserving its own representation while other heads in the layer mix in context."*

---

## What I want to remember from this notebook

1. **Attention tensor axes:** `[batch, head, query, key]`. Rows = query (attending FROM); columns = key (attended TO). The query is the actor, the key is the target. Get the direction right.
2. **Softmax is row-wise.** It's applied across the key axis, one independent softmax per query. Rows sum to 1; columns don't, because nothing ties rows together.
3. **Softmax math, internalized:** `exp(x_i) / Σ_j exp(x_j)`. Exponentiate, then divide by the sum of all exponentials. Shared denominator = entries sum to 1. This is everywhere in ML; learn it once.
4. **Attention sinks aren't decorative.** Many heads in many layers rely on position 0 as a stable anchor; dropping it causes uniform quality collapse across the network, not patchy degradation.
5. **Perplexity** is the standard scalar for "how surprised was the model by this sequence." Lower = more confident. Use it when you need to quantify language-model quality changes.
6. **Named attention archetypes exist.** "Previous-token heads" (bright sub-diagonal), "attention-sink heads" (bright first column), "self-attending / identity heads" (bright main diagonal). When I see a visual pattern, naming it is the first step toward interpretation.
7. **Frameworks break.** The Pythia layers 9–11 NaN bug taught me that the path from "library returns a tensor" to "the tensor is correct" is not free — it has to be verified. Workaround now (use clean layers); fix properly in Phase 1 with forward hooks.

---

**Phase 0 is done.** I can now:

- Write an autoregressive decoder loop from scratch (debugged through real bugs).
- Explain what a KV cache buys and what it costs, with a benchmark to prove it.
- Pull attention weights out of a HuggingFace causal LM, visualize them, and work around the framework's quirks.
- Articulate the prefill/decode distinction and why batching helps decode.
- Read attention patterns visually and name common archetypes.

Next: Phase 1 — `src/hybrid_arch/metrics.py`, starting with `next_token_entropy` and forward-hook-based attention extraction.
