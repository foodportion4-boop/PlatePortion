"""Direct RGB-D regression baseline: image -> five nutrition targets.

This is the learned counterpart to the geometric branch, and roughly what
Thames et al. call "depth as 4th channel". A single backbone consumes a 5-channel
stack and a linear head predicts all five targets jointly.

Input channels are ``[R, G, B, depth, valid]``. The validity channel is not
decoration: 11-33% of depth pixels per frame are sensor dropouts parked at the
far plane, and without an explicit mask the network cannot distinguish "far
surface" from "no measurement".

Targets are **standardised using training-split statistics** before the loss.
Calories reach ~3900 while fat is order 10, so an unnormalised multi-task L1 is
effectively a calories-only loss. Predictions are de-standardised before metrics
so reported MAE is in real units.

Sized for an 8 GB M1: ResNet-18 at 224px, batch 16 fits comfortably on MPS.
``--backbone resnet34`` is the next step up if more memory is available.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from n5k.data.metadata import TARGETS

IN_CHANNELS = 5  # R, G, B, depth, valid


def _inflate_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Widen a pretrained 3-channel stem to ``in_channels`` without losing it.

    The RGB filters are copied across; the extra channels are seeded with the
    mean RGB filter scaled by 3/in_channels, which preserves the layer's output
    magnitude so pretrained downstream statistics stay valid.
    """
    new = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        w = conv.weight  # (out, 3, k, k)
        new.weight[:, :3] = w
        if in_channels > 3:
            extra = w.mean(dim=1, keepdim=True).repeat(1, in_channels - 3, 1, 1)
            new.weight[:, 3:] = extra
        new.weight.mul_(3.0 / in_channels)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


class DirectNutritionNet(nn.Module):
    """Multi-task regressor over RGB-D."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        in_channels: int = IN_CHANNELS,
        n_targets: int = len(TARGETS),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        factory = getattr(torchvision.models, backbone)
        weights = "DEFAULT" if pretrained else None
        net = factory(weights=weights)

        net.conv1 = _inflate_first_conv(net.conv1, in_channels)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_targets),
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor):
        x = torch.cat([rgb, depth, valid.float()], dim=1)
        return self.head(self.backbone(x))


class TargetScaler:
    """Standardise/de-standardise targets with train-split statistics."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = std.clamp_min(1e-6)

    def to(self, device) -> "TargetScaler":
        return TargetScaler(self.mean.to(device), self.std.to(device))

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Nutrition is non-negative; clamping avoids nonsense like -30 kcal
        # inflating MAE when the head extrapolates below zero.
        return (z * self.std + self.mean).clamp_min(0.0)

    def state_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    @classmethod
    def from_state_dict(cls, d: dict) -> "TargetScaler":
        return cls(d["mean"], d["std"])


def pick_device(prefer: str = "auto") -> torch.device:
    """Choose a device, preferring Apple MPS on this machine."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
