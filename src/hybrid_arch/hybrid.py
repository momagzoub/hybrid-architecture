"""The hybrid decoder.

Wraps `spec_decode_capture` with a *router*: a callable that takes per-position
drafter features and returns a boolean "keep this drafted token without
running the target?" decision. The point of Phase 4 is the demo, not the
SOTA — so the implementation favors instrumentation (what the router said,
what the target would have said, perplexity vs the verifier) over raw
throughput.

Phase 3 Step 4 showed that the drafter's own `1 − top1` is a strong predictor
of rejection. The default router here is exactly that — anything else has
to beat the baseline.

Public API::

    Router (Protocol)
    threshold_router(feature_name, threshold) -> Router
    weighted_router(weights, bias, threshold) -> Router
    hybrid_decode(target, drafter, prompt, router, ...) -> HybridDecodeResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch
from torch import Tensor

from hybrid_arch.spec_decode import SpecDecodeTrace, spec_decode_capture


class Router(Protocol):
    """Protocol for per-position routing decisions.

    Given a feature dict for a *single* drafted position, return True if the
    hybrid decoder should "skip the verifier" and commit the drafted token.
    The feature dict is built from the most recent `spec_decode_capture`
    trace — see `_features_for` for the canonical keys.
    """

    def __call__(self, features: dict[str, float]) -> bool: ...


def threshold_router(
    feature_name: str = "one_minus_top1",
    threshold: float = 0.2,
    *,
    direction: str = "below",
) -> Router:
    """Route by thresholding a single feature.

    `direction` is `"below"` (keep if `features[name] < threshold`) or
    `"above"`. Defaulting to `"one_minus_top1 < 0.2"` reproduces the
    spirit of "keep tokens the drafter is at least 80% confident in",
    which is the Phase 3 best-baseline behavior.
    """
    if direction not in ("below", "above"):
        raise ValueError(f"direction must be 'below' or 'above', got {direction!r}")

    def router(features: dict[str, float]) -> bool:
        v = features[feature_name]
        return v < threshold if direction == "below" else v > threshold
    return router


def weighted_router(
    weights: dict[str, float],
    bias: float = 0.0,
    *,
    threshold: float = 0.0,
) -> Router:
    """Linear router: `keep if sum(w_i · x_i) + bias > threshold`.

    The natural endpoint of Phase 4's "fitted router" experiment — once a
    logistic regression on real drafter-rejection labels coughs up
    coefficients, pour them in here.
    """

    def router(features: dict[str, float]) -> bool:
        score = bias
        for k, w in weights.items():
            score += w * features.get(k, 0.0)
        return score > threshold
    return router


@dataclass
class HybridDecodeResult:
    """Per-position record from a hybrid-decode run.

    All fields are aligned across the same `n_drafted` positions:

    - `accept`: the *real* (target-side) accept/reject bit, identical to the
      one `spec_decode_capture` returns. The router decision is independent.
    - `router_decision`: True iff the router decided to commit the drafted
      token without checking the target.
    - `correct_routing`: True iff `router_decision == accept`. The router is
      "right" when it commits a drafted token that the target would also
      have produced, OR when it sends a position to the target that the
      target would have rejected anyway.
    """

    accept: Tensor                      # bool [N]
    router_decision: Tensor             # bool [N]
    correct_routing: Tensor             # bool [N]
    features: dict[str, Tensor]         # each [N]
    spec_trace: SpecDecodeTrace = field(repr=False)

    @property
    def n(self) -> int:
        return int(self.accept.numel())

    @property
    def router_keep_rate(self) -> float:
        return float(self.router_decision.float().mean()) if self.n else float("nan")

    @property
    def router_accuracy(self) -> float:
        """Fraction of positions where the router's decision matches the verifier."""
        return float(self.correct_routing.float().mean()) if self.n else float("nan")

    @property
    def false_keep_rate(self) -> float:
        """Fraction of positions the router *kept* but the target would have rejected.

        This is the quality-cost knob: false keeps are tokens that diverge from
        the target's preferred output. If `false_keep_rate` is high, the router
        is committing drafted tokens the verifier disagrees with — i.e., wrecking
        perplexity.
        """
        kept = self.router_decision
        if not kept.any():
            return 0.0
        return float((kept & ~self.accept).float().sum() / kept.float().sum())


def _features_for(trace: SpecDecodeTrace, probe_logits: Tensor | None = None) -> dict[str, Tensor]:
    """Canonical per-position feature dict from a spec-decode trace.

    Keys:
      - `entropy` — drafter's next-token entropy at the position
      - `top1` — drafter's top-1 probability
      - `one_minus_top1` — `1 - top1` (the Phase 3 best baseline)
      - `probe_logit` (optional) — `probe(hidden_states_at_layer)` if supplied
    """
    feats: dict[str, Tensor] = {
        "entropy": trace.entropy,
        "top1": trace.top1,
        "one_minus_top1": 1.0 - trace.top1,
    }
    if probe_logits is not None:
        feats["probe_logit"] = probe_logits
    return feats


def hybrid_decode(
    target: torch.nn.Module,
    drafter: torch.nn.Module,
    prompt_ids: Tensor,
    *,
    router: Router | None = None,
    probe_fn: Callable[[Tensor], Tensor] | None = None,
    probe_layer: int | None = None,
    n_steps: int = 16,
    draft_k: int = 4,
) -> HybridDecodeResult:
    """Run greedy speculative decoding + apply a router per-position.

    The decoder always runs the verifier (so we know the ground-truth
    accept/reject), but additionally records what the router *would* have
    done. This is the right framing for a Phase 4 demo: we measure
    routing quality against the real target, then report the cost.

    Args:
        target:  the larger verifier model.
        drafter: the smaller draft model.
        prompt_ids: `[1, S0]` int64.
        router:  callable taking a per-position feature dict and returning
            a keep/route boolean. Defaults to `threshold_router("one_minus_top1", 0.2)`.
        probe_fn: optional callable that maps a hidden state `[N, H]` to per-position
            logits `[N]`. If supplied with `probe_layer`, the `probe_logit`
            feature is included in the per-position dict.
        probe_layer: which layer's hidden state to feed the probe.
        n_steps, draft_k: passed to `spec_decode_capture`.

    Returns:
        `HybridDecodeResult` with the trace, the router's decisions, and
        the per-position correctness mask.
    """
    if router is None:
        router = threshold_router("one_minus_top1", threshold=0.2)

    trace = spec_decode_capture(target, drafter, prompt_ids, n_steps=n_steps, draft_k=draft_k)

    probe_logits: Tensor | None = None
    if probe_fn is not None:
        if probe_layer is None:
            raise ValueError("probe_layer must be provided when probe_fn is")
        hs = trace.drafter_hidden_states[probe_layer].float()  # [N, H]
        with torch.no_grad():
            probe_logits = probe_fn(hs)

    features = _features_for(trace, probe_logits=probe_logits)

    decisions = torch.zeros(trace.accept.shape, dtype=torch.bool)
    for i in range(trace.accept.numel()):
        per_pos = {k: float(v[i].item()) for k, v in features.items()}
        decisions[i] = bool(router(per_pos))

    correct = decisions == trace.accept
    return HybridDecodeResult(
        accept=trace.accept,
        router_decision=decisions,
        correct_routing=correct,
        features=features,
        spec_trace=trace,
    )
