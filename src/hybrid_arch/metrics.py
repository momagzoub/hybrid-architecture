"""Per-token metrics derived from a model's forward pass.

All functions in this module return one number per position (or per
(layer, head, position) for attention-side metrics), so they can be lined
up against tokens for analysis and visualization.

Entropies are reported in **nats** (natural log). Convert to bits by
dividing by `math.log(2)` if needed.

Public API:
    next_token_entropy(logits)        -> Tensor
    top1_probability(logits)          -> Tensor
    attention_entropy(attn_weights)   -> Tensor                   # per-head [L,B,H,S]
    attention_concentration(attn_weights, top_k) -> Tensor        # per-head [K,L,B,H,S]
    aggregate_attention_entropy(per_head)       -> Tensor          # [B,S]
    aggregate_attention_concentration(per_head) -> Tensor          # [K,B,S]
    parallel_prediction_agreement(model, input_ids, k) -> Tensor
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def next_token_entropy(logits: Tensor) -> Tensor:
    """Shannon entropy of the predicted next-token distribution.

    `H = -Σᵢ pᵢ log(pᵢ)` in nats, where `p = softmax(logits)`.
    High entropy = model is uncertain about what comes next.

    Landmarks:
        Uniform over V items   → H = log(V)
        Delta (one item certain) → H = 0
        Bernoulli p=0.5        → H = log(2) ≈ 0.693

    Args:
        logits: shape `[..., vocab]`. Typically `[batch, seq, vocab]`
            from `model(input_ids).logits`.

    Returns:
        Tensor of shape `[...]` (i.e., logits without the last dim).
        Non-negative, in nats.
    """
    log_probs = F.log_softmax(logits.to(torch.float32), dim=-1)
    probs = log_probs.exp()
    # For finite logits, softmax produces strictly positive probs, so
    # `probs * log_probs` is finite everywhere. -inf logits would cause
    # NaN here, but a language model's logits are never -inf in practice.
    return -(probs * log_probs).sum(dim=-1)


def top1_probability(logits: Tensor) -> Tensor:
    """Probability mass on the single most likely next token at each position.

    Simpler companion to `next_token_entropy`: useful as a threshold
    ("is the model >90% confident?") and as a sanity check on entropy.

    Args:
        logits: shape `[..., vocab]`.

    Returns:
        Tensor of shape `[...]`, values in `(0, 1]`.
    """
    return F.softmax(logits.to(torch.float32), dim=-1).max(dim=-1).values


def attention_entropy(attn_weights: Tensor) -> Tensor:
    """Shannon entropy of each attention distribution at each query position.

    Each row of an attention matrix is a probability distribution over keys
    (this is enforced by the softmax inside attention itself), so we can
    apply the same Shannon formula `H = -Σⱼ pⱼ log pⱼ` row-wise.

    Low entropy = the head focuses on a few keys; high entropy = the head
    spreads its attention diffusely.

    Note: under a causal mask, position 0 has exactly one nonzero entry (the
    diagonal), so its entropy is structurally 0. Position 1 caps at log(2),
    etc. This is a property of causality, not of the model.

    Args:
        attn_weights: shape `[..., S, S]`. Last two dims are (query, key).
            Typically `[L, B, H, S, S]` from `extract_attention`.

    Returns:
        Tensor of shape `[..., S]` — one entropy per query position.
        Non-negative, in nats.
    """
    # xlogy(p, p) = p * log(p) with the convention 0 * log(0) = 0, which
    # matters here because causally-masked entries are exactly 0.
    return -torch.special.xlogy(attn_weights, attn_weights).sum(dim=-1)


def attention_concentration(
    attn_weights: Tensor,
    top_k: Sequence[int] = (1, 3, 5),
) -> Tensor:
    """Cumulative attention mass on the top-k attended keys per query.

    A simpler, more interpretable companion to `attention_entropy`. Top-1
    near 1.0 means the head is essentially pointing at one key; top-5 near
    1.0 means the head's attention is contained in a handful of keys
    regardless of how it's distributed among them.

    Args:
        attn_weights: shape `[..., S, S]`.
        top_k: sequence of integers, the k values to compute. Defaults to
            `(1, 3, 5)`. If any `k > S`, it is clamped to `S`.

    Returns:
        Tensor of shape `[len(top_k), ..., S]`. `result[i]` corresponds to
        `top_k[i]`. Values in `[0, 1]`.
    """
    if len(top_k) == 0:
        raise ValueError("top_k must contain at least one integer")
    if any(k < 1 for k in top_k):
        raise ValueError(f"top_k values must be >= 1, got {top_k}")

    seq_len = attn_weights.shape[-1]
    max_k_needed = min(max(top_k), seq_len)

    # top_values: [..., S, max_k_needed], sorted descending along last dim.
    top_values, _ = attn_weights.topk(max_k_needed, dim=-1)
    cumsum = top_values.cumsum(dim=-1)  # cumsum along the k axis

    # For each requested k, take cumsum[..., k-1]; clamp k to seq_len.
    per_k = [cumsum[..., min(k, seq_len) - 1] for k in top_k]
    return torch.stack(per_k, dim=0)


def aggregate_attention_entropy(per_head: Tensor) -> Tensor:
    """Average a per-(layer, head) attention-entropy tensor across (L, H).

    Phase 1 stored only this aggregate and saw |r| < 0.11 against parallel-safety.
    Phase 2's signature analysis uses the full per-head tensor instead. This
    helper exists so callers who need the aggregate can compute it without
    re-deriving the axis convention.

    Args:
        per_head: `[L, B, H, S]` from `attention_entropy`.

    Returns:
        `[B, S]` — the mean over layers and heads.
    """
    if per_head.ndim != 4:
        raise ValueError(
            f"expected [L, B, H, S], got shape {tuple(per_head.shape)}"
        )
    return per_head.mean(dim=(0, 2))


def aggregate_attention_concentration(per_head: Tensor) -> Tensor:
    """Average a per-(layer, head) concentration tensor across (L, H).

    Args:
        per_head: `[K, L, B, H, S]` from `attention_concentration`.

    Returns:
        `[K, B, S]` — the mean over layers and heads.
    """
    if per_head.ndim != 5:
        raise ValueError(
            f"expected [K, L, B, H, S], got shape {tuple(per_head.shape)}"
        )
    return per_head.mean(dim=(1, 3))


@torch.no_grad()
def parallel_prediction_agreement(
    model: torch.nn.Module,
    input_ids: Tensor,
    k: int = 4,
    *,
    batched: bool = True,
) -> Tensor:
    """The headline metric: how often does the model predict the same tokens
    in parallel as it does when forced to decode autoregressively?

    For each position `t` in the prompt, we compare two predictions of the
    next `k` tokens starting from context `input_ids[..., 0..t]`:

    1. **Teacher-forced (parallel).** A single forward pass on the prefix
       `input_ids[..., 0..t+k]`; we take argmax at output positions
       `t, t+1, …, t+k-1`. The model sees the ground-truth tokens at every
       intermediate step.
    2. **Autoregressive (sequential).** Start from `input_ids[..., 0..t]`,
       greedily generate one token at a time, feeding each output back as
       the next input. The model sees only its own predictions at
       intermediate steps.

    Agreement at step `j` (where `j ∈ 0..k-1`) is whether the two top-1
    predictions match. High agreement → "parallel-safe": the real future
    tokens didn't carry information the model needed.

    This is the offline analogue of speculative-decoding acceptance rate.

    Args:
        model: causal LM. Must accept `input_ids` and return `.logits`.
        input_ids: `[batch, seq]`. Must satisfy `seq > k` so there's at least
            one position with `k` real tokens ahead to compare against.
            The batched implementation currently requires `batch == 1`.
        k: lookahead horizon. Defaults to 4.
        batched: if True (default), use the batched O(k)-forward-passes
            implementation. If False, fall back to the O(n_positions × k)
            sequential implementation (slower, kept for verification).

    Returns:
        BoolTensor of shape `[batch, n_positions, k]` where
        `n_positions = seq - k`. `result[b, t, j] = True` iff the
        teacher-forced and autoregressive predictions match at step `j`
        starting from position `t`. Per-position agreement *rate* is the
        mean over the last dim.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    batch, seq = input_ids.shape
    if seq <= k:
        raise ValueError(
            f"input_ids has seq={seq} but k={k}; need seq > k so each "
            "position has at least k real tokens ahead to compare against."
        )

    was_training = model.training
    model.eval()
    try:
        if batched:
            if batch != 1:
                raise NotImplementedError(
                    "batched=True currently requires batch=1; "
                    "pass batched=False for multi-batch input."
                )
            return _parallel_prediction_agreement_batched(model, input_ids, k)
        return _parallel_prediction_agreement_sequential(model, input_ids, k)
    finally:
        if was_training:
            model.train()


def _parallel_prediction_agreement_sequential(
    model: torch.nn.Module, input_ids: Tensor, k: int
) -> Tensor:
    """Naive O(n_positions × k) implementation. Kept for correctness tests."""
    batch, seq = input_ids.shape
    n_positions = seq - k

    tf_logits = model(input_ids=input_ids).logits   # [B, S, V]
    tf_argmax = tf_logits.argmax(dim=-1)            # [B, S]
    tf_pred = tf_argmax.unfold(dimension=1, size=k, step=1)[:, :n_positions]

    ar_pred = torch.zeros(batch, n_positions, k, dtype=torch.long)
    for t in range(n_positions):
        ctx = input_ids[:, : t + 1].clone()
        for j in range(k):
            logits = model(input_ids=ctx).logits[:, -1, :]
            next_tok = logits.argmax(dim=-1, keepdim=True)
            ar_pred[:, t, j] = next_tok.squeeze(-1)
            ctx = torch.cat([ctx, next_tok], dim=1)

    return tf_pred == ar_pred


def _parallel_prediction_agreement_batched(
    model: torch.nn.Module, input_ids: Tensor, k: int
) -> Tensor:
    """Batched O(k)-forward-pass implementation.

    Speedup vs the sequential version comes from two observations:

    1. **j=0 is structural.** At step 0, teacher-forced and autoregressive
       predictions share identical context `input_ids[0..t]`, so they agree
       by construction. We skip the model call and copy the TF result.
    2. **For j ≥ 1, all n_positions rollouts can be batched.** At step j,
       each rollout's context is `input_ids[0..t] + ar_pred[t, 0..j-1]`,
       which has length `t + 1 + j`. We pack all rollouts into a single
       `[n_positions, max_len]` right-padded tensor and run one forward
       pass per step.

    Right-padded GPTNeoX with `attention_mask` produces correct logits at
    each row's last-real-position regardless of the padded tail (causal +
    padding mask combine).
    """
    _, seq = input_ids.shape  # batch must be 1 (checked by caller)
    n_positions = seq - k
    device = input_ids.device

    # --- 1. Teacher-forced predictions: one forward pass on the whole prompt.
    tf_logits = model(input_ids=input_ids).logits   # [1, seq, vocab]
    tf_argmax = tf_logits.argmax(dim=-1)            # [1, seq]
    tf_pred = tf_argmax.unfold(dimension=1, size=k, step=1)[:, :n_positions]

    # --- 2. AR predictions. j=0 column equals tf_argmax[0, t] for each t
    # (same context, so same prediction — the structural invariant).
    ar_pred = torch.zeros(1, n_positions, k, dtype=torch.long, device=device)
    ar_pred[0, :, 0] = tf_argmax[0, :n_positions]

    if k == 1:
        return tf_pred == ar_pred

    # j >= 1: batched rollouts.
    # At step j, row t holds [input_ids[0..t], ar_pred[t, 0..j-1]], length t+1+j.
    # The longest row is at t = n_positions - 1, so max_len = n_positions + j.
    pad_id = 0
    for j in range(1, k):
        max_len = n_positions + j
        ctx_batch = torch.full(
            (n_positions, max_len), pad_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros(
            (n_positions, max_len), dtype=torch.long, device=device
        )

        # Vectorize the row-filling. Each row t looks like:
        #   ctx_batch[t, : t+1]      = input_ids[0, : t+1]
        #   ctx_batch[t, t+1 : t+1+j] = ar_pred[0, t, : j]
        # Using a Python loop is fine — it's O(n_positions) cheap setup
        # compared to the O(n_positions × seq × hidden) forward pass.
        for t in range(n_positions):
            ctx_len = t + 1 + j
            ctx_batch[t, : t + 1] = input_ids[0, : t + 1]
            ctx_batch[t, t + 1 : ctx_len] = ar_pred[0, t, :j]
            attention_mask[t, :ctx_len] = 1

        logits = model(input_ids=ctx_batch, attention_mask=attention_mask).logits
        # Take logits at each row's last-real-position: index (ctx_len - 1) = (t + j).
        last_idx = torch.arange(n_positions, device=device) + j
        row_idx = torch.arange(n_positions, device=device)
        next_token = logits[row_idx, last_idx, :].argmax(dim=-1)
        ar_pred[0, :, j] = next_token

    return tf_pred == ar_pred
