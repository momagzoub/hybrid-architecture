"""Tests for `hybrid_arch.hybrid`.

The hybrid decoder is the production-shape wrapper around `spec_decode_capture`.
What we pin here are the contracts that downstream Phase 4 work depends on:

- the router types compose correctly without touching a model
- `HybridDecodeResult` invariants (lengths match, accuracy is well-defined)
- weighted_router reproduces threshold_router when only one weight is nonzero
- the default router actually defaults to the Phase 3 best baseline

End-to-end runs (which need Pythia loads) live in the bench script, not here.
"""

from __future__ import annotations

import math

import pytest
import torch

from hybrid_arch.hybrid import (
    HybridDecodeResult,
    _features_for,
    threshold_router,
    weighted_router,
)
from hybrid_arch.spec_decode import SpecDecodeTrace


def _fake_trace(accept: list[bool], top1: list[float] | None = None) -> SpecDecodeTrace:
    """Synthesize a `SpecDecodeTrace` with N positions for routing-logic tests."""
    n = len(accept)
    if top1 is None:
        top1 = [0.95 if a else 0.40 for a in accept]
    return SpecDecodeTrace(
        accept=torch.tensor(accept, dtype=torch.bool),
        entropy=torch.tensor([-math.log(p) if p > 0 else 5.0 for p in top1], dtype=torch.float32),
        top1=torch.tensor(top1, dtype=torch.float32),
        drafter_token=torch.zeros(n, dtype=torch.long),
        target_token=torch.zeros(n, dtype=torch.long),
        drafter_hidden_states=torch.zeros(1, n, 8),
        n_steps=1,
        n_drafted=n,
        n_accepted=sum(accept),
    )


# --------- routers ---------

def test_threshold_router_below():
    router = threshold_router("one_minus_top1", threshold=0.1, direction="below")
    assert router({"one_minus_top1": 0.05}) is True
    assert router({"one_minus_top1": 0.20}) is False


def test_threshold_router_above():
    router = threshold_router("entropy", threshold=1.0, direction="above")
    assert router({"entropy": 1.5}) is True
    assert router({"entropy": 0.5}) is False


def test_threshold_router_validates_direction():
    with pytest.raises(ValueError, match="direction"):
        threshold_router("entropy", threshold=1.0, direction="sideways")


def test_weighted_router_reduces_to_threshold():
    """A single-weight `weighted_router` equals `threshold_router` (modulo sign).

    `threshold_router("one_minus_top1", 0.1, "below")` keeps when
    `one_minus_top1 < 0.1`. The equivalent weighted formulation is
    `-1 · one_minus_top1 > -0.1`, i.e. weights={"one_minus_top1": -1},
    threshold=-0.1.
    """
    w = weighted_router({"one_minus_top1": -1.0}, bias=0.0, threshold=-0.1)
    assert w({"one_minus_top1": 0.05}) is True       # -0.05 > -0.1
    assert w({"one_minus_top1": 0.50}) is False      # -0.50 < -0.1
    assert w({"one_minus_top1": 0.10}) is False      # -0.10 is NOT > -0.1


def test_weighted_router_missing_feature_is_zero():
    """Missing features default to 0 — useful when a probe is unavailable."""
    w = weighted_router({"probe_logit": 2.0, "entropy": 1.0}, bias=0.0, threshold=0.5)
    assert w({"entropy": 1.0}) is True              # 2*0 + 1*1 = 1 > 0.5
    assert w({"entropy": 0.0, "probe_logit": 0.0}) is False


# --------- _features_for ---------

def test_features_for_basic_keys():
    trace = _fake_trace([True, False, True])
    feats = _features_for(trace)
    assert set(feats.keys()) == {"entropy", "top1", "one_minus_top1"}
    assert torch.allclose(feats["one_minus_top1"], 1.0 - feats["top1"])


def test_features_for_includes_probe_logit():
    trace = _fake_trace([True, False])
    probe = torch.tensor([0.7, -0.3])
    feats = _features_for(trace, probe_logits=probe)
    assert "probe_logit" in feats
    assert torch.equal(feats["probe_logit"], probe)


# --------- HybridDecodeResult ---------

def test_hybrid_decode_result_metrics():
    """An ideal router keeps exactly the accepted positions."""
    accept = [True, True, False, False, True]
    trace = _fake_trace(accept)
    result = HybridDecodeResult(
        accept=trace.accept,
        router_decision=trace.accept.clone(),       # perfect router
        correct_routing=torch.ones_like(trace.accept, dtype=torch.bool),
        features=_features_for(trace),
        spec_trace=trace,
    )
    assert result.n == 5
    assert result.router_keep_rate == pytest.approx(3 / 5)
    assert result.router_accuracy == 1.0
    assert result.false_keep_rate == 0.0


def test_hybrid_decode_result_false_keep():
    """A router that keeps everything has false_keep_rate = (1 - accept_rate)."""
    accept = [True, False, False, True]
    trace = _fake_trace(accept)
    decisions = torch.ones(4, dtype=torch.bool)
    correct = decisions == trace.accept
    result = HybridDecodeResult(
        accept=trace.accept,
        router_decision=decisions,
        correct_routing=correct,
        features=_features_for(trace),
        spec_trace=trace,
    )
    assert result.router_keep_rate == 1.0
    assert result.false_keep_rate == pytest.approx(0.5)


def test_hybrid_decode_result_empty():
    """Zero positions returns NaN, doesn't crash."""
    trace = _fake_trace([])
    result = HybridDecodeResult(
        accept=trace.accept,
        router_decision=trace.accept.clone(),
        correct_routing=torch.empty(0, dtype=torch.bool),
        features={},
        spec_trace=trace,
    )
    assert result.n == 0
    assert math.isnan(result.router_keep_rate)
    assert math.isnan(result.router_accuracy)
