"""Multi-label ingredient classifier over RGB-D.

The geometric branch's oracle result shows that knowing *which* ingredients are
on the plate is worth ~10.6 points of mean PMAE. This model is the piece that
turns that oracle into a deployable number: predict the ingredient set from the
image, then feed it to
:class:`~n5k.geometric.calibrate.IngredientCalibration`.

Shares the 5-channel RGB-D stem with :mod:`n5k.models.direct` -- same backbone,
different head. The task is genuinely multi-label (a median dish has 4
ingredients, ~7 counting the frequent vocabulary), so the head is 133
independent sigmoids rather than a softmax.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision

from n5k.models.direct import IN_CHANNELS, _inflate_first_conv


class IngredientClassifier(nn.Module):
    """RGB-D -> per-ingredient presence logits."""

    def __init__(
        self,
        n_ingredients: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        in_channels: int = IN_CHANNELS,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        net = getattr(torchvision.models, backbone)(
            weights="DEFAULT" if pretrained else None
        )
        net.conv1 = _inflate_first_conv(net.conv1, in_channels)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, n_ingredients),
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor):
        x = torch.cat([rgb, depth, valid.float()], dim=1)
        return self.head(self.backbone(x))


def multilabel_metrics(
    probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Micro/macro F1, mAP, and set-level statistics.

    Micro-F1 is the headline: it weights by occurrence, matching what the
    downstream calibration cares about (getting the common ingredients right on
    most dishes). Macro-F1 is reported alongside because micro-F1 can look
    healthy while every rare class is dead.
    """
    pred = probs >= threshold
    tp = (pred & (labels > 0.5)).sum()
    fp = (pred & (labels < 0.5)).sum()
    fn = (~pred & (labels > 0.5)).sum()

    micro_p = tp / max(tp + fp, 1)
    micro_r = tp / max(tp + fn, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-9)

    f1s, aps = [], []
    for j in range(labels.shape[1]):
        y, p = labels[:, j] > 0.5, pred[:, j]
        if y.sum() == 0:
            continue
        t = (p & y).sum()
        prec = t / max(p.sum(), 1)
        rec = t / max(y.sum(), 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-9))
        aps.append(_average_precision(probs[:, j], y))

    return {
        "micro_f1": float(micro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "pred_per_dish": float(pred.sum(axis=1).mean()),
        "true_per_dish": float((labels > 0.5).sum(axis=1).mean()),
    }


def _average_precision(scores: np.ndarray, y: np.ndarray) -> float:
    """Area under the precision-recall curve for one class."""
    order = np.argsort(-scores)
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    n_pos = y.sum()
    return float((precision * y).sum() / n_pos) if n_pos else 0.0


def tune_threshold(
    probs: np.ndarray, labels: np.ndarray, grid: np.ndarray | None = None
) -> tuple[float, float]:
    """Pick the decision threshold maximising micro-F1.

    Must be called on **training** predictions. The threshold is a fitted
    parameter; choosing it on test would leak.
    """
    grid = grid if grid is not None else np.arange(0.05, 0.96, 0.05)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = multilabel_metrics(probs, labels, float(t))["micro_f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1
