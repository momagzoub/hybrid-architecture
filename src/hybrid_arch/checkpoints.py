"""Helpers for loading Pythia models at specific training checkpoints.

Pythia ships 154 training checkpoints per model size. We evaluate a
log-spaced subset of 12 of them per the Phase 2 plan: dense early to
catch emergence, sparse late to track convergence.

Public API:
    list_checkpoints()             -> list[int]
    load_pythia(size, step, dtype) -> (model, tokenizer)
"""

from __future__ import annotations

from typing import Literal

import torch

PythiaSize = Literal["70m", "160m", "410m", "1b", "1.4b", "2.8b"]

#: Log-spaced step values used across Phase 2. All Pythia sizes train for
#: the same 143000 steps, so this list applies to every size.
CANONICAL_STEPS: tuple[int, ...] = (
    0,        # random init — entropy baseline
    1, 8, 128,
    1000,
    4000,
    16000,
    32000,
    64000,
    96000,
    128000,
    143000,   # final
)


def list_checkpoints() -> list[int]:
    """Return the canonical step values to evaluate in Phase 2.

    Log-spaced: dense early (to catch emergence), sparse late (to track
    convergence). The same list is used for every Pythia size since
    they share the 143000-step schedule.

    Returns:
        A new list (caller may mutate without affecting the module).
    """
    return list(CANONICAL_STEPS)


def load_pythia(
    size: PythiaSize,
    step: int,
    *,
    dtype: torch.dtype = torch.float32,
):
    """Load a Pythia model + tokenizer pinned to a specific training step.

    Args:
        size: Pythia size string, e.g. "70m", "160m", "410m". The model
            name resolves to "EleutherAI/pythia-{size}".
        step: training step. Must be a valid Pythia revision; common
            choices are in `CANONICAL_STEPS`.
        dtype: model dtype. Default fp32. Half-precision triggers a softmax
            overflow in Pythia's deep attention layers; the forward-hook
            extractor in `attention.py` works around it for analysis, but
            generation still benefits from fp32.

    Returns:
        `(model, tokenizer)`. The model is in `eval()` mode and on CPU.
        Move it to the desired device with `.to(...)` if needed.
    """
    # Local import — keeps `import hybrid_arch.checkpoints` cheap when only
    # `list_checkpoints` is needed.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = f"EleutherAI/pythia-{size}"
    revision = f"step{step}"

    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=dtype
    )
    model.eval()
    return model, tok
