"""Forward-hook attention extraction for Pythia / GPTNeoX models.

The framework's `output_attentions=True` path returns NaN in Pythia's deeper
layers (typically 9-11) because the unfused softmax overflows: residual-stream
magnitudes grow with depth, so `exp(Q @ K^T)` overflows in fp16/bf16, and
`inf - inf` poisons the masked-position softmax. Both eager and SDPA paths
inherit this.

The fix here is to capture each attention layer's inputs via a forward
pre-hook, replay the QKV projection and rotary embedding using the layer's
own weights, then compute attention scores and softmax in fp32 with the
causal mask applied. Numerically clean for all layers.

Public API:
    extract_attention(model, input_ids)       -> Tensor[L, B, H, S, S]
    extract_hidden_states(model, input_ids)   -> Tensor[L, B, S, H]
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _gptneox_apply_rotary(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply rotary embeddings to q, k. Mirrors `apply_rotary_pos_emb` in
    transformers' GPTNeoX, kept local so this module doesn't depend on a
    specific transformers internal."""
    cos = cos.unsqueeze(1)  # broadcast over heads
    sin = sin.unsqueeze(1)
    rot_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]

    def rotate_half(x: Tensor) -> Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = q_rot * cos + rotate_half(q_rot) * sin
    k_embed = k_rot * cos + rotate_half(k_rot) * sin
    q_out = torch.cat([q_embed, q_pass], dim=-1)
    k_out = torch.cat([k_embed, k_pass], dim=-1)
    return q_out, k_out


def _find_gptneox_attention_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Return all `GPTNeoXAttention` submodules in document order."""
    layers: list[torch.nn.Module] = []
    for module in model.modules():
        if type(module).__name__ == "GPTNeoXAttention":
            layers.append(module)
    return layers


@torch.no_grad()
def extract_attention(
    model: torch.nn.Module,
    input_ids: Tensor,
    *,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Run `model` on `input_ids` and return per-layer attention weights.

    Captures each attention layer's post-layernorm input and rotary
    position embeddings via a forward pre-hook, then replays the QKV
    projection + rotary application + causally-masked softmax in fp32 to
    avoid the NaN bug in the framework's default attention path.

    Args:
        model: A GPTNeoX-family causal LM (e.g., any Pythia checkpoint).
        input_ids: Long tensor of shape `[batch, seq]`.
        attention_mask: Optional padding mask. Not yet supported — extract
            attention is intended for unpadded prompts; raises if provided
            with non-uniform padding.

    Returns:
        Tensor of shape `[num_layers, batch, num_heads, seq, seq]`, in fp32.
        Row sums are 1.0; upper triangle is exactly 0; no NaN anywhere.

    Raises:
        RuntimeError: if the model contains no `GPTNeoXAttention` layers.
    """
    if attention_mask is not None and not torch.all(attention_mask == 1):
        raise NotImplementedError(
            "extract_attention does not yet support padded inputs; "
            "pass a single unpadded sequence per batch row."
        )

    layers = _find_gptneox_attention_layers(model)
    if not layers:
        raise RuntimeError(
            f"No GPTNeoXAttention layers found on {type(model).__name__}. "
            "extract_attention currently only supports GPTNeoX-family models."
        )

    # One slot per layer; filled by the pre-hook during the forward pass.
    captured: list[dict[str, Any]] = [{} for _ in layers]

    def make_hook(idx: int):
        def hook(_module, args, kwargs):
            # In current transformers, GPTNeoXLayer calls:
            #   attention(self.input_layernorm(hidden_states), ...,
            #             position_embeddings=position_embeddings, ...)
            # so hidden_states is args[0] and position_embeddings is a kwarg.
            hidden_states = args[0] if args else kwargs["hidden_states"]
            pos = kwargs.get("position_embeddings")
            if pos is None:
                raise RuntimeError(
                    "position_embeddings kwarg not passed to attention layer; "
                    "transformers API may have changed."
                )
            cos, sin = pos
            captured[idx]["hidden_states"] = hidden_states.detach()
            captured[idx]["cos"] = cos.detach()
            captured[idx]["sin"] = sin.detach()
        return hook

    handles = [
        layer.register_forward_pre_hook(make_hook(i), with_kwargs=True)
        for i, layer in enumerate(layers)
    ]

    was_training = model.training
    try:
        model.eval()
        model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    # Replay QKV + rotary + masked softmax per layer, in fp32.
    per_layer: list[Tensor] = []
    for layer, cap in zip(layers, captured):
        hidden_states = cap["hidden_states"]
        cos = cap["cos"]
        sin = cap["sin"]

        input_shape = hidden_states.shape[:-1]  # (batch, seq)
        head_size = layer.head_size

        # query_key_value: Linear(hidden -> 3*hidden); reshape into heads.
        qkv = layer.query_key_value(hidden_states)
        # -> [batch, seq, num_heads, 3*head_size], transpose to put heads before seq
        qkv = qkv.view(*input_shape, -1, 3 * head_size).transpose(1, 2)
        # -> [batch, num_heads, seq, 3*head_size]
        q, k, _ = qkv.chunk(3, dim=-1)

        q, k = _gptneox_apply_rotary(q, k, cos, sin)

        # All score-and-softmax math in fp32 to avoid the overflow bug.
        q_f = q.to(torch.float32)
        k_f = k.to(torch.float32)
        scaling = head_size**-0.5
        scores = (q_f @ k_f.transpose(-2, -1)) * scaling  # [B, H, S, S]

        seq = scores.shape[-1]
        causal = torch.triu(
            torch.ones(seq, seq, dtype=torch.bool, device=scores.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal, float("-inf"))

        attn = torch.softmax(scores, dim=-1)  # fp32 softmax, masked rows OK
        per_layer.append(attn)

    return torch.stack(per_layer, dim=0)  # [L, B, H, S, S]


def _find_gptneox_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Return all `GPTNeoXLayer` submodules in document order."""
    layers: list[torch.nn.Module] = []
    for module in model.modules():
        if type(module).__name__ == "GPTNeoXLayer":
            layers.append(module)
    return layers


@torch.no_grad()
def extract_hidden_states(
    model: torch.nn.Module,
    input_ids: Tensor,
) -> Tensor:
    """Capture per-layer hidden states (the residual stream).

    Hooks each `GPTNeoXLayer`'s *output* so we get the post-block residual
    stream at every depth — the input that downstream probes consume.
    Equivalent to `model(output_hidden_states=True).hidden_states[1:]` but
    keeps the API in one place and remains insensitive to upstream wrapper
    changes.

    Args:
        model: A GPTNeoX-family causal LM.
        input_ids: Long tensor `[batch, seq]`.

    Returns:
        Tensor of shape `[num_layers, batch, seq, hidden_dim]` in fp32, on CPU.

    Raises:
        RuntimeError: if the model contains no `GPTNeoXLayer` blocks.
    """
    layers = _find_gptneox_layers(model)
    if not layers:
        raise RuntimeError(
            f"No GPTNeoXLayer blocks found on {type(model).__name__}. "
            "extract_hidden_states only supports GPTNeoX-family models."
        )

    captured: list[Tensor] = [None] * len(layers)  # type: ignore[list-item]

    def make_hook(idx: int):
        def hook(_module, _args, output):
            # GPTNeoXLayer.forward returns either a plain Tensor or a tuple
            # depending on transformers version; the residual hidden state
            # is always the first element.
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx] = hidden.detach().to(torch.float32).cpu()
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    was_training = model.training
    try:
        model.eval()
        model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    return torch.stack(captured, dim=0)  # [L, B, S, H]
