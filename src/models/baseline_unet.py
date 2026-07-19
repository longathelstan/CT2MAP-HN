"""
CT2MAP-HN – Baseline 3-D UNet with multi-head output.

Uses MONAI's ``BasicUNet`` as the encoder–decoder backbone and attaches
three task-specific heads for heatmap regression, lesion segmentation, and
case-level triage classification.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from monai.networks.nets import BasicUNet

from src.models.heads import HeatmapHead, LesionHead, TriageHead, UncertaintyHead

logger = logging.getLogger(__name__)


class BaselineUNet(nn.Module):
    """Baseline 3-D UNet with multi-head output.

    The model accepts a single-channel CT volume and produces three outputs:

    * ``heatmap`` – voxel-wise metabolic risk map ``(B, 1, D, H, W)``.
    * ``lesion`` – binary lesion segmentation ``(B, 1, D, H, W)``.
    * ``triage`` – case-level risk probability ``(B, 1)``.

    Optionally an ``uncertainty`` map can be produced when
    ``use_uncertainty=True``.

    Args:
        in_channels: Number of input channels (1 for CT).
        features: Sequence of feature-map widths at each encoder level.
            Defaults to ``(32, 32, 64, 128, 256, 32)``, matching MONAI's
            BasicUNet.
        dropout: Dropout probability used throughout the backbone.
        use_uncertainty: Whether to attach an :class:`UncertaintyHead`.
        triage_hidden: Hidden-layer size for the :class:`TriageHead`.
        triage_dropout: Dropout probability for the :class:`TriageHead`.

    Example::

        model = BaselineUNet(in_channels=1, features=(32, 32, 64, 128, 256, 32))
        out = model(torch.randn(1, 1, 96, 96, 96))
        print(out["heatmap"].shape)   # (1, 1, 96, 96, 96)
        print(out["triage"].shape)    # (1, 1)
    """

    def __init__(
        self,
        in_channels: int = 1,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        dropout: float = 0.1,
        use_uncertainty: bool = False,
        triage_hidden: int = 128,
        triage_dropout: float = 0.3,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.use_uncertainty = use_uncertainty

        # Map string norm name to MONAI norm tuple (using eps=1e-4 for FP16 stability)
        _norm_map = {
            "instance": ("instance", {"affine": True, "eps": 1e-4}),
            "batch": ("batch", {"affine": True, "eps": 1e-4}),
            "group": ("group", {"num_groups": 8, "affine": True, "eps": 1e-4}),
        }
        norm_cfg = _norm_map.get(norm.lower(), ("batch", {"affine": True, "eps": 1e-4}))

        # ---- Backbone ----
        # BasicUNet outputs `features[-1]` channels.
        self.backbone = BasicUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=int(features[-1]),
            features=tuple(int(f) for f in features),
            dropout=dropout,
            act=("leakyrelu", {"negative_slope": 0.01, "inplace": True}),
            norm=norm_cfg,
            upsample="deconv",
        )

        decoder_out_ch = int(features[-1])

        # ---- Task heads ----
        self.heatmap_head = HeatmapHead(in_channels=decoder_out_ch, out_channels=1)
        self.lesion_head = LesionHead(in_channels=decoder_out_ch, out_channels=1)
        # Use the deepest encoder feature for triage (highest semantic info)
        self.triage_head = TriageHead(
            in_channels=decoder_out_ch,
            num_classes=1,
            hidden_dim=triage_hidden,
            dropout=triage_dropout,
        )

        if use_uncertainty:
            self.uncertainty_head = UncertaintyHead(in_channels=decoder_out_ch)
        else:
            self.uncertainty_head = None

        # Track intermediate features for distillation
        self._intermediate_features: List[torch.Tensor] = []

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "BaselineUNet initialised: in_ch=%d  features=%s  dropout=%.2f  "
            "params=%.2fM  uncertainty=%s",
            in_channels, features, dropout, n_params / 1e6, use_uncertainty,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input CT volume ``(B, 1, D, H, W)``.

        Returns:
            Dictionary with keys ``"heatmap"``, ``"lesion"``, ``"triage"``,
            and optionally ``"uncertainty"``.
        """
        # BasicUNet returns the final decoder output
        decoder_out = self.backbone(x)

        outputs: Dict[str, torch.Tensor] = {
            "heatmap": self.heatmap_head(decoder_out),
            "lesion": self.lesion_head(decoder_out),
            "triage": self.triage_head(decoder_out),
        }

        if self.uncertainty_head is not None:
            outputs["uncertainty"] = self.uncertainty_head(decoder_out)

        # Cache for potential distillation
        self._intermediate_features = [decoder_out]

        return outputs

    # ------------------------------------------------------------------
    # Distillation helpers
    # ------------------------------------------------------------------
    def get_intermediate_features(self) -> List[torch.Tensor]:
        """Return the most recently cached intermediate features.

        Returns:
            List containing the decoder output tensor.  Call :meth:`forward`
            first.
        """
        return self._intermediate_features
