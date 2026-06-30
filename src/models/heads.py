"""
CT2MAP-HN – Task-specific output heads.

Provides lightweight head modules that sit on top of backbone feature maps:

* **HeatmapHead** – voxel-wise metabolic risk regression (sigmoid output).
* **LesionHead** – binary lesion segmentation (sigmoid output).
* **TriageHead** – case-level triage classification (global pool → FC → sigmoid).
* **UncertaintyHead** – learnable log-variance for heteroscedastic uncertainty.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HeatmapHead
# ---------------------------------------------------------------------------
class HeatmapHead(nn.Module):
    """3-D voxel-wise regression head for metabolic risk heatmaps.

    Architecture: ``Conv3d(in_ch, mid_ch, 3) → BN → ReLU → Conv3d(mid_ch, out_ch, 1) → Sigmoid``.

    Args:
        in_channels: Number of input feature channels from the decoder.
        out_channels: Number of output channels (default 1 for single heatmap).
        mid_channels: Optional intermediate channel width.  Defaults to
            ``in_channels // 2`` (minimum 16).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        mid_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = max(in_channels // 2, 16)

        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=True),
        )
        self.activation = nn.Sigmoid()
        logger.debug(
            "HeatmapHead: in=%d  mid=%d  out=%d", in_channels, mid_channels, out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature tensor of shape ``(B, C, D, H, W)``.

        Returns:
            Heatmap predictions in ``[0, 1]``, shape ``(B, out_channels, D, H, W)``.
        """
        return self.activation(self.conv_block(x))


# ---------------------------------------------------------------------------
# LesionHead
# ---------------------------------------------------------------------------
class LesionHead(nn.Module):
    """3-D binary segmentation head for lesion detection.

    Architecture mirrors :class:`HeatmapHead` but is conceptually separate so
    that different loss functions / thresholds can be applied.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output channels (default 1).
        mid_channels: Optional intermediate width.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        mid_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = max(in_channels // 2, 16)

        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=True),
        )
        self.activation = nn.Sigmoid()
        logger.debug(
            "LesionHead: in=%d  mid=%d  out=%d", in_channels, mid_channels, out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature tensor ``(B, C, D, H, W)``.

        Returns:
            Binary segmentation probabilities ``(B, out_channels, D, H, W)``.
        """
        return self.activation(self.conv_block(x))


# ---------------------------------------------------------------------------
# TriageHead
# ---------------------------------------------------------------------------
class TriageHead(nn.Module):
    """Case-level triage classification head.

    Applies adaptive 3-D average pooling to collapse spatial dims, followed by
    a small MLP that outputs a per-case risk probability.

    Args:
        in_channels: Number of input feature channels.
        num_classes: Number of output classes (default 1 for binary triage).
        hidden_dim: Hidden layer width in the MLP.  Defaults to 128.
        dropout: Dropout probability applied before the final FC layer.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.activation = nn.Sigmoid()
        logger.debug(
            "TriageHead: in=%d  hidden=%d  classes=%d  drop=%.2f",
            in_channels, hidden_dim, num_classes, dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature tensor ``(B, C, D, H, W)``.

        Returns:
            Triage probabilities ``(B, num_classes)``.
        """
        pooled = self.pool(x)  # (B, C, 1, 1, 1)
        return self.activation(self.classifier(pooled))


# ---------------------------------------------------------------------------
# UncertaintyHead
# ---------------------------------------------------------------------------
class UncertaintyHead(nn.Module):
    """Learnable log-variance head for heteroscedastic aleatoric uncertainty.

    Predicts per-voxel ``log(sigma^2)`` which can be used in a Gaussian
    negative-log-likelihood loss to weigh uncertain voxels down.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output channels (default 1).
    """

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, out_channels, kernel_size=1, bias=True),
        )
        # Initialise bias so initial log-variance ≈ 0 → sigma ≈ 1
        nn.init.zeros_(self.conv[-1].bias)
        logger.debug("UncertaintyHead: in=%d  out=%d", in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature tensor ``(B, C, D, H, W)``.

        Returns:
            Log-variance map ``(B, out_channels, D, H, W)`` (unclamped).
        """
        return self.conv(x)
