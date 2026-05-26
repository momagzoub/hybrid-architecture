"""hybrid_arch — diagnostic toolkit for adaptive LLM inference research.

See AGENTS.md and PROJECT_PLAN.md for project context.

Public API (implemented across Phase 1):
  attention  — extract_attention()
  metrics    — next_token_entropy(), top1_probability(),
               attention_entropy(), attention_concentration(),
               parallel_prediction_agreement()
  viz        — entropy_heatmap(), attention_track()
"""

__version__ = "0.3.0"

from hybrid_arch.attention import extract_attention, extract_hidden_states
from hybrid_arch.cache import metric_battery, slice_hash
from hybrid_arch.checkpoints import list_checkpoints, load_pythia
from hybrid_arch.metrics import (
    aggregate_attention_concentration,
    aggregate_attention_entropy,
    attention_concentration,
    attention_entropy,
    next_token_entropy,
    parallel_prediction_agreement,
    top1_probability,
)
from hybrid_arch.probes import (
    LayerProbe,
    TrainResult,
    cross_val_auroc,
    load_probe,
    save_probe,
    train_probe,
)
from hybrid_arch.viz import attention_track, entropy_heatmap

__all__ = [
    "__version__",
    "extract_attention",
    "extract_hidden_states",
    "next_token_entropy",
    "top1_probability",
    "attention_entropy",
    "attention_concentration",
    "aggregate_attention_entropy",
    "aggregate_attention_concentration",
    "parallel_prediction_agreement",
    "entropy_heatmap",
    "attention_track",
    "list_checkpoints",
    "load_pythia",
    "metric_battery",
    "slice_hash",
    "LayerProbe",
    "TrainResult",
    "train_probe",
    "cross_val_auroc",
    "save_probe",
    "load_probe",
]

