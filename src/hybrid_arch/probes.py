"""Phase 3 — shallow MLP probes over hidden states.

A `LayerProbe` takes a single layer's hidden state at one position and
outputs a logit for the binary parallel-safety label. The architecture is
deliberately small (~50k params for `hidden_dim=1024`): an inference-time
pre-verifier needs to be cheap, and Belinkov 2022 warns that big probes
recover the property from anywhere, which tells you nothing about the
representation.

The trainer (`train_probe`) is standard: Adam, BCE-with-logits,
class-balanced positive weight, early stopping on a held-out split. No
schedule, no augmentation — the probe is the experiment, not the headline.

Public API::

    LayerProbe(hidden_dim, mlp_dim=64) -> nn.Module
    train_probe(features, labels, **kwargs) -> TrainResult
    cross_val_auroc(features, labels, n_folds=5, **kwargs) -> tuple[float, float]
    save_probe(probe, path)
    load_probe(path) -> LayerProbe
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import Tensor


class LayerProbe(nn.Module):
    """3-layer MLP: LayerNorm → Linear(d→m) → GELU → Linear(m→m//2) → GELU → Linear(m//2→1).

    Why this shape:
    - LayerNorm on the input absorbs the residual-stream scale drift across
      layers and across model sizes, so the same hyperparameters work on
      Pythia-70m's 512-dim states and Pythia-410m's 1024-dim states.
    - Two hidden layers is the minimum to capture any non-linear feature
      interactions; deeper probes leak capacity (Belinkov 2022 §4.3).
    - Sigmoid output via BCEWithLogitsLoss at training time, applied at
      inference via `predict_proba`.
    """

    def __init__(self, hidden_dim: int, mlp_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, mlp_dim // 2)
        self.head = nn.Linear(mlp_dim // 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        """`x`: `[..., hidden_dim]`. Returns raw logits of shape `[...]` (no sigmoid)."""
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = F.gelu(self.fc2(h))
        return self.head(h).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x: Tensor) -> Tensor:
        """Convenience: sigmoid of `forward`. Returns probabilities in `(0, 1)`."""
        return torch.sigmoid(self(x))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class TrainResult:
    """What `train_probe` returns. Self-contained — has everything the
    layer-depth sweep needs and nothing it doesn't."""

    val_auroc: float
    val_loss: float
    best_epoch: int
    train_losses: list[float]
    val_losses: list[float]
    n_train: int
    n_val: int
    n_positive_train: int
    n_positive_val: int


def _pos_weight(y: Tensor) -> Tensor:
    """Balanced positive-class weight for BCEWithLogitsLoss.

    `n_neg / n_pos` recovers the loss that `class_weight="balanced"` gives.
    Guards against `n_pos == 0` so the loss stays finite on degenerate
    splits.
    """
    n_pos = int(y.sum().item())
    n_neg = int(y.numel() - n_pos)
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(max(n_neg / n_pos, 1.0))


def train_probe(
    features: Tensor,
    labels: Tensor,
    *,
    hidden_dim: int | None = None,
    mlp_dim: int = 64,
    val_fraction: float = 0.2,
    n_epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    patience: int = 20,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[LayerProbe, TrainResult]:
    """Train a `LayerProbe` on `(features, labels)`.

    A single 80/20 train/val split (stratified manually so both halves keep
    the positive class). Adam, BCEWithLogitsLoss with `pos_weight`,
    early-stop on val loss after `patience` epochs without improvement.

    Args:
        features: `[N, hidden_dim]` fp32 tensor.
        labels: `[N]` long/bool tensor with values in `{0, 1}`.
        hidden_dim: explicit override; defaults to `features.shape[-1]`.
        mlp_dim: hidden dim of the probe's first MLP layer.
        val_fraction: held-out fraction for early stopping. 0.2 by default.
        n_epochs, lr, weight_decay, patience: standard knobs.
        seed: RNG seed for the train/val split and torch init.
        device: "cpu" or "cuda".
        verbose: print per-epoch losses.

    Returns:
        `(probe, TrainResult)`. The probe is returned with the *best*
        validation weights, not the final-epoch weights.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be [N, hidden_dim], got {tuple(features.shape)}")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError(
            f"labels must be [N] matching features rows; got {tuple(labels.shape)} "
            f"vs features {tuple(features.shape)}"
        )

    rng = np.random.RandomState(seed)
    y_np = labels.cpu().numpy().astype(np.int64)
    pos_idx = np.where(y_np == 1)[0]
    neg_idx = np.where(y_np == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    n_val_pos = max(int(round(len(pos_idx) * val_fraction)), 1) if len(pos_idx) > 1 else 0
    n_val_neg = max(int(round(len(neg_idx) * val_fraction)), 1) if len(neg_idx) > 1 else 0
    val_idx = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
    train_idx = np.concatenate([pos_idx[n_val_pos:], neg_idx[n_val_neg:]])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    if hidden_dim is None:
        hidden_dim = features.shape[-1]

    torch.manual_seed(seed)
    probe = LayerProbe(hidden_dim=hidden_dim, mlp_dim=mlp_dim).to(device)
    optim = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

    X = features.to(device).float()
    y = labels.to(device).float()
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]
    pos_w = _pos_weight(y_tr).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    best_val = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    patience_left = patience
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(n_epochs):
        probe.train()
        optim.zero_grad()
        logits_tr = probe(X_tr)
        loss_tr = loss_fn(logits_tr, y_tr)
        loss_tr.backward()
        optim.step()

        probe.eval()
        with torch.no_grad():
            logits_va = probe(X_va)
            loss_va = loss_fn(logits_va, y_va).item()

        train_losses.append(loss_tr.item())
        val_losses.append(loss_va)
        if verbose:
            print(f"epoch {epoch:3d}  train={loss_tr.item():.4f}  val={loss_va:.4f}")

        if loss_va < best_val - 1e-5:
            best_val = loss_va
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
            best_epoch = epoch
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        probe.load_state_dict(best_state)

    probe.eval()
    with torch.no_grad():
        logits_va = probe(X_va).cpu().numpy()
    y_va_np = y_va.cpu().numpy()
    if len(np.unique(y_va_np)) < 2:
        val_auroc = float("nan")
    else:
        val_auroc = float(roc_auc_score(y_va_np, logits_va))

    result = TrainResult(
        val_auroc=val_auroc,
        val_loss=best_val,
        best_epoch=best_epoch,
        train_losses=train_losses,
        val_losses=val_losses,
        n_train=int(X_tr.shape[0]),
        n_val=int(X_va.shape[0]),
        n_positive_train=int(y_tr.sum().item()),
        n_positive_val=int(y_va.sum().item()),
    )
    return probe, result


def cross_val_auroc(
    features: Tensor,
    labels: Tensor,
    *,
    n_folds: int = 5,
    seed: int = 0,
    **train_kwargs,
) -> tuple[float, float]:
    """Stratified K-fold ROC-AUC. Returns `(mean, std)`.

    Skips folds with degenerate validation labels (only one class).
    Returns `(nan, nan)` if no fold yields a valid AUROC.
    """
    y_np = labels.cpu().numpy().astype(np.int64)
    n_pos = int(y_np.sum())
    if n_pos < n_folds or n_pos > y_np.size - n_folds:
        return float("nan"), float("nan")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aurocs: list[float] = []
    for tr, te in skf.split(np.zeros((y_np.size, 1)), y_np):
        probe, _ = train_probe(
            features[tr], labels[tr],
            val_fraction=0.2, seed=seed, **train_kwargs,
        )
        with torch.no_grad():
            scores = probe(features[te].float()).cpu().numpy()
        if len(np.unique(y_np[te])) < 2:
            continue
        aurocs.append(float(roc_auc_score(y_np[te], scores)))
    if not aurocs:
        return float("nan"), float("nan")
    return float(np.mean(aurocs)), float(np.std(aurocs))


def save_probe(probe: LayerProbe, path: Path | str, *, metadata: dict | None = None) -> None:
    """Save weights + a small JSON sidecar describing the architecture.

    The sidecar makes the file self-describing without unpickling.
    `metadata` is merged into the sidecar — useful for stamping
    `(model_size, layer)` provenance.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), path)
    sidecar = path.with_suffix(path.suffix + ".json")
    info = {
        "hidden_dim": probe.hidden_dim,
        "mlp_dim": probe.mlp_dim,
        "parameter_count": probe.parameter_count(),
    }
    if metadata:
        info.update(metadata)
    sidecar.write_text(json.dumps(info, indent=2, sort_keys=True))


def load_probe(path: Path | str) -> LayerProbe:
    """Inverse of `save_probe`. Reads the JSON sidecar to rebuild the architecture."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.exists():
        raise FileNotFoundError(f"missing sidecar {sidecar}; cannot reconstruct architecture")
    info = json.loads(sidecar.read_text())
    probe = LayerProbe(hidden_dim=info["hidden_dim"], mlp_dim=info["mlp_dim"])
    probe.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    probe.eval()
    return probe


# Re-export dataclass helper for downstream use.
def train_result_as_dict(r: TrainResult) -> dict:
    return asdict(r)
