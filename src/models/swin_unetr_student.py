"""
CT2MAP-HN – Swin UNETR student model with MC Dropout and distillation.

Wraps MONAI's ``SwinUNETR`` as a backbone encoder–decoder with multi-head
output identical to :class:`BaselineUNet`, plus first-class support for:

* **MC Dropout** – probabilistic inference via :meth:`enable_mc_dropout`.
* **Intermediate feature extraction** for teacher–student distillation.
* **Gradient checkpointing** for memory-efficient training on large volumes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from monai.networks.nets import SwinUNETR

from src.models.heads import HeatmapHead, LesionHead, TriageHead, UncertaintyHead

logger = logging.getLogger(__name__)


class SwinUNETRStudent(nn.Module):
    """Swin UNETR student model with multi-head output.

    The architecture mirrors :class:`BaselineUNet` in its outputs but replaces
    the backbone with MONAI's ``SwinUNETR``.

    Args:
        in_channels: Number of input channels (1 for CT).
        img_size: Spatial size of the input volume ``(D, H, W)``.
        feature_size: Base channel width inside SwinUNETR.
        patch_size: Patch size for the Swin Transformer tokeniser.
        depths: Number of Swin Transformer blocks at each stage.
        num_heads: Number of attention heads at each stage.
        use_checkpoint: Enable gradient checkpointing to save VRAM.
        dropout: Dropout probability applied inside the backbone and heads.
        use_uncertainty: Attach an :class:`UncertaintyHead`.
        triage_hidden: Hidden-layer width in :class:`TriageHead`.
        triage_dropout: Dropout in :class:`TriageHead`.
        spatial_dims: Number of spatial dimensions (always 3).

    Example::

        model = SwinUNETRStudent(in_channels=1, img_size=(96, 96, 96))
        out = model(torch.randn(1, 1, 96, 96, 96))
        print(out["heatmap"].shape)   # (1, 1, 96, 96, 96)
    """

    def __init__(
        self,
        in_channels: int = 1,
        img_size: Sequence[int] = (96, 96, 96),
        feature_size: int = 48,
        patch_size: Sequence[int] = (2, 2, 2),
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        use_checkpoint: bool = True,
        dropout: float = 0.1,
        use_uncertainty: bool = False,
        triage_hidden: int = 128,
        triage_dropout: float = 0.3,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.use_uncertainty = use_uncertainty
        self._mc_dropout_enabled = False

        # ---- Backbone ----
        # SwinUNETR out_channels = feature_size by default in its decoder
        self.backbone = SwinUNETR(
            img_size=tuple(int(s) for s in img_size),
            in_channels=in_channels,
            out_channels=feature_size,
            feature_size=feature_size,
            depths=tuple(int(d) for d in depths),
            num_heads=tuple(int(h) for h in num_heads),
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            drop_rate=dropout,
            attn_drop_rate=dropout,
        )

        decoder_out_ch = feature_size

        # ---- MC Dropout layer (placed after backbone, before heads) ----
        self.mc_dropout = nn.Dropout3d(p=dropout)

        # ---- Task heads ----
        self.heatmap_head = HeatmapHead(in_channels=decoder_out_ch, out_channels=1)
        self.lesion_head = LesionHead(in_channels=decoder_out_ch, out_channels=1)
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

        # Cache for distillation
        self._intermediate_features: List[torch.Tensor] = []

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "SwinUNETRStudent initialised: img_size=%s  feat=%d  "
            "depths=%s  heads=%s  params=%.2fM  checkpoint=%s",
            img_size, feature_size, depths, num_heads,
            n_params / 1e6, use_checkpoint,
        )

    # ------------------------------------------------------------------
    # MC Dropout control
    # ------------------------------------------------------------------
    def enable_mc_dropout(self) -> None:
        """Enable MC Dropout at inference (keeps dropout layers in train mode).

        After calling this, :meth:`forward` will produce stochastic outputs
        even when the model is in ``eval`` mode, enabling Monte Carlo sampling
        for uncertainty estimation.
        """
        self._mc_dropout_enabled = True
        # Set all Dropout layers to train mode
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()
        logger.info("MC Dropout ENABLED – dropout layers set to train mode.")

    def disable_mc_dropout(self) -> None:
        """Disable MC Dropout (restore normal eval behaviour)."""
        self._mc_dropout_enabled = False
        # If the whole model is in eval mode, dropout layers will follow
        if not self.training:
            for module in self.modules():
                if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                    module.eval()
        logger.info("MC Dropout DISABLED.")

    def train(self, mode: bool = True) -> "SwinUNETRStudent":
        """Override to preserve MC dropout state when switching modes."""
        super().train(mode)
        if not mode and self._mc_dropout_enabled:
            # Re-enable dropout layers even in eval
            for module in self.modules():
                if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                    module.train()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input CT volume ``(B, 1, D, H, W)``.

        Returns:
            Dict with ``"heatmap"``, ``"lesion"``, ``"triage"`` and
            optionally ``"uncertainty"`` tensors.
        """
        decoder_out = self.backbone(x)

        # Extract intermediate features from SwinUNETR's encoder
        # The swin_unetr stores hidden_states during forward
        self._extract_intermediate(x)

        # Apply MC dropout
        decoder_out = self.mc_dropout(decoder_out)

        outputs: Dict[str, torch.Tensor] = {
            "heatmap": self.heatmap_head(decoder_out),
            "lesion": self.lesion_head(decoder_out),
            "triage": self.triage_head(decoder_out),
        }

        if self.uncertainty_head is not None:
            outputs["uncertainty"] = self.uncertainty_head(decoder_out)

        return outputs

    # ------------------------------------------------------------------
    # Distillation helpers
    # ------------------------------------------------------------------
    def _extract_intermediate(self, x: torch.Tensor) -> None:
        """Extract and cache intermediate encoder features for distillation.

        Uses the SwinUNETR's ``swinViT`` sub-module to get hidden states at
        multiple resolutions.

        Args:
            x: Input volume ``(B, C, D, H, W)``.
        """
        try:
            hidden_states = self.backbone.swinViT(x, self.backbone.normalize)
            # hidden_states is a list of tensors at each stage
            self._intermediate_features = list(hidden_states)
        except Exception:
            # Graceful fallback: just use backbone output
            logger.debug(
                "Could not extract SwinViT hidden states; "
                "falling back to decoder output only."
            )
            self._intermediate_features = []

    def get_intermediate_features(self) -> List[torch.Tensor]:
        """Return cached intermediate features from the last forward pass.

        Returns:
            List of tensors at multiple encoder scales.  Call :meth:`forward`
            first.
        """
        return self._intermediate_features
