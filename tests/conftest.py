"""Shared pytest fixtures for hybrid_arch tests."""

from __future__ import annotations

import pytest
import torch

PYTHIA_NAME = "EleutherAI/pythia-160m"
PYTHIA_REV = "step143000"


@pytest.fixture(scope="session")
def pythia_model_and_tokenizer():
    """Load Pythia-160M @ step143000 in fp32, cached for the whole session.

    Downloads ~330 MB on first run; subsequent runs use the local HF cache.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(PYTHIA_NAME, revision=PYTHIA_REV)
    model = AutoModelForCausalLM.from_pretrained(
        PYTHIA_NAME, revision=PYTHIA_REV, torch_dtype=torch.float32
    )
    model.eval()
    return model, tok
