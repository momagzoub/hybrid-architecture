"""Greedy speculative decoding with per-position accept/reject capture.

A minimal, inline implementation — we don't try to compete with
`transformers.generate(assistant_model=...)` on speed. The point is that
*every* drafted position emits a row of structured data the analysis layer
can use: the drafter's hidden states, its logit entropy, its top-1
probability, the position's argmax tokens from both models, and the
ground-truth accept/reject bit.

Greedy on both sides keeps the analysis clean: a "reject" is exactly the
event that an offline parallel-safety probe should predict.

Public API::

    spec_decode_capture(target, drafter, prompt_ids,
                        n_steps, draft_k) -> SpecDecodeTrace
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from hybrid_arch.attention import extract_hidden_states


@dataclass
class SpecDecodeTrace:
    """Per-drafted-position record from one `spec_decode_capture` run.

    Lengths are aligned: `accept[i]` corresponds to `drafter_hidden_states[:, i]`,
    `entropy[i]`, `top1[i]`, `drafter_token[i]`, `target_token[i]`.

    `accept` is True when the target's greedy argmax at the drafted position
    matches the drafter's argmax — i.e. the drafted token is what the target
    would have produced.
    """

    accept: Tensor                      # bool [N]
    entropy: Tensor                     # fp32 [N]
    top1: Tensor                        # fp32 [N]
    drafter_token: Tensor               # int64 [N]
    target_token: Tensor                # int64 [N]
    drafter_hidden_states: Tensor       # fp32 [L_drafter, N, H_drafter]
    n_steps: int = 0
    n_drafted: int = 0
    n_accepted: int = 0
    rejection_step_positions: list[int] = field(default_factory=list)

    @property
    def accept_rate(self) -> float:
        return float(self.accept.float().mean()) if self.accept.numel() else float("nan")


@torch.no_grad()
def _greedy_argmax(model: torch.nn.Module, input_ids: Tensor) -> Tensor:
    """One forward pass; return `argmax` of the logits at every position. `[B, S]`."""
    return model(input_ids=input_ids).logits.argmax(dim=-1)


@torch.no_grad()
def _drafter_step(
    drafter: torch.nn.Module,
    prefix: Tensor,
    draft_k: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Greedy-extend `prefix` by `draft_k` tokens with `drafter`.

    Returns:
      - `drafted`: `[draft_k]` of new token ids
      - `entropies`: `[draft_k]` of next-token entropies *at the step that produced each*
      - `top1`: `[draft_k]` of top-1 probabilities at each step
      - `hidden_per_layer_at_predictor`: `[L, draft_k, H]` — the drafter's hidden
        state at the *last* position of each prefix that produced the next token.
    """
    # The simplest correct (if not fastest) version: run the drafter once after
    # each new token, capture per-layer hidden states at the predictor position.
    drafted: list[int] = []
    entropies: list[float] = []
    top1s: list[float] = []
    hidden_columns: list[Tensor] = []   # each is [L, H]
    ctx = prefix.clone()
    for _ in range(draft_k):
        # Logits at the last position predict the next token.
        logits_full = drafter(input_ids=ctx).logits     # [1, S_ctx, V]
        last = logits_full[0, -1]                       # [V]
        log_probs = F.log_softmax(last.float(), dim=-1)
        probs = log_probs.exp()
        entropies.append(float(-(probs * log_probs).sum().item()))
        top1s.append(float(probs.max().item()))
        next_tok = int(last.argmax().item())
        drafted.append(next_tok)
        # Capture hidden states at the predictor position via the dedicated hook.
        hs = extract_hidden_states(drafter, ctx)        # [L, 1, S_ctx, H]
        hidden_columns.append(hs[:, 0, -1, :].clone())  # [L, H]
        next_t = torch.tensor([[next_tok]], dtype=ctx.dtype, device=ctx.device)
        ctx = torch.cat([ctx, next_t], dim=1)

    drafted_t = torch.tensor(drafted, dtype=torch.long)
    entropies_t = torch.tensor(entropies, dtype=torch.float32)
    top1_t = torch.tensor(top1s, dtype=torch.float32)
    hidden_stack = torch.stack(hidden_columns, dim=1)   # [L, draft_k, H]
    return drafted_t, entropies_t, top1_t, hidden_stack


@torch.no_grad()
def spec_decode_capture(
    target: torch.nn.Module,
    drafter: torch.nn.Module,
    prompt_ids: Tensor,
    *,
    n_steps: int,
    draft_k: int,
) -> SpecDecodeTrace:
    """Run greedy speculative decoding and capture per-drafted-position data.

    At each speculative-decoding *step*:
      1. The drafter greedy-extends the current prefix by `draft_k` tokens.
      2. The target evaluates the resulting prefix in one forward pass; we
         take its greedy argmax at the drafted positions.
      3. Compare drafter argmax vs target argmax position-by-position. The
         first mismatch is the rejection point.
      4. We commit the prefix up to *and including* the first rejected
         position, replaced by the target's argmax (the standard greedy
         spec-decode bookkeeping). All `draft_k` drafted positions still
         emit one row of trace data — accepted or not.

    Args:
        target:   the bigger verifier model (eval mode).
        drafter:  the smaller draft model (eval mode).
        prompt_ids: `[1, S0]` int64 starter prompt.
        n_steps:  how many speculative-decoding *steps* to run. Effective
                  generated-token count is bounded by `n_steps * draft_k`
                  but is typically smaller because of rejections.
        draft_k:  how many tokens the drafter proposes per step.

    Returns:
        `SpecDecodeTrace` with one row per drafted position across all
        `n_steps` steps.
    """
    if prompt_ids.dim() != 2 or prompt_ids.shape[0] != 1:
        raise ValueError(f"prompt_ids must be [1, S]; got {tuple(prompt_ids.shape)}")
    target.eval()
    drafter.eval()

    accept_rows: list[bool] = []
    ent_rows: list[float] = []
    top1_rows: list[float] = []
    drafter_tok_rows: list[int] = []
    target_tok_rows: list[int] = []
    hs_rows: list[Tensor] = []          # each [L, draft_k, H]
    rejection_positions: list[int] = []

    prefix = prompt_ids.clone()
    n_accepted_total = 0
    n_drafted_total = 0
    for step in range(n_steps):
        drafted, ents, top1, hs = _drafter_step(drafter, prefix, draft_k)
        # Extend prefix with drafted tokens, then ask the target what it would
        # have predicted at each of those positions.
        extended = torch.cat([prefix, drafted.unsqueeze(0)], dim=1)        # [1, S0 + k]
        target_argmax = _greedy_argmax(target, extended)[0]                # [S0 + k]
        # The target's argmax at output position i predicts the token AT i+1
        # if we treat extended as the input. We want target's prediction at
        # the positions of the drafted tokens. Drafter tokens were placed at
        # positions S0..S0+k-1 of `extended`; the target's prediction for the
        # token AT position S0+j is `target_argmax[S0 + j - 1]` (i.e., its
        # output at the previous position). Standard LM "argmax at t predicts
        # token t+1" convention.
        S0 = prefix.shape[1]
        target_preds = target_argmax[S0 - 1 : S0 - 1 + draft_k]            # [draft_k]

        # Compare position-by-position; first mismatch is the rejection.
        accepts = (target_preds == drafted)                                # [draft_k] bool
        first_reject = int((~accepts).nonzero()[0].item()) if (~accepts).any() else draft_k

        # Record one row per drafted position.
        for j in range(draft_k):
            accept_rows.append(bool(accepts[j].item()))
            ent_rows.append(float(ents[j].item()))
            top1_rows.append(float(top1[j].item()))
            drafter_tok_rows.append(int(drafted[j].item()))
            target_tok_rows.append(int(target_preds[j].item()))
            hs_rows.append(hs[:, j, :])                                     # [L, H]
        if first_reject < draft_k:
            rejection_positions.append(len(accept_rows) - draft_k + first_reject)

        # Commit: accepted prefix + target's correction at first reject (or all k
        # if none rejected). The "+1" extra token from a fully-accepted draft
        # is the bonus target token; we ignore it for simplicity, taking only
        # the accepted draft tokens through.
        if first_reject == draft_k:
            new_prefix = extended
        else:
            committed = drafted[:first_reject].tolist() + [int(target_preds[first_reject].item())]
            new_prefix = torch.cat([
                prefix,
                torch.tensor([committed], dtype=prefix.dtype, device=prefix.device),
            ], dim=1)
        prefix = new_prefix

        n_drafted_total += draft_k
        n_accepted_total += int(accepts.sum().item())

    L = hs_rows[0].shape[0]
    H = hs_rows[0].shape[1]
    hidden_per_layer = torch.stack(hs_rows, dim=1) if hs_rows else torch.zeros(L, 0, H)
    # hidden_per_layer now [L, N, H] where N = n_steps * draft_k
    return SpecDecodeTrace(
        accept=torch.tensor(accept_rows, dtype=torch.bool),
        entropy=torch.tensor(ent_rows, dtype=torch.float32),
        top1=torch.tensor(top1_rows, dtype=torch.float32),
        drafter_token=torch.tensor(drafter_tok_rows, dtype=torch.long),
        target_token=torch.tensor(target_tok_rows, dtype=torch.long),
        drafter_hidden_states=hidden_per_layer,
        n_steps=n_steps,
        n_drafted=n_drafted_total,
        n_accepted=n_accepted_total,
        rejection_step_positions=rejection_positions,
    )
