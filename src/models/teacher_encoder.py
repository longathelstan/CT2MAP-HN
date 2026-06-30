"""
CT2MAP-HN – Teacher encoder for knowledge distillation.

A dual-input (CT + PET) encoder that extracts PET-informed features at
multiple spatial scales.  After initial supervised training the encoder is
frozen and used solely to provide distillation targets for the student.

The architecture is a lightweight 3-D ResNet-style encoder with two input
branches that are fused early (channel concatenation + 1×1 conv).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------
class _ConvBlock(nn.Module):
    """Basic ``Conv3d → BN → ReLU`` block with optional residual skip."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.residual = residual
        if residual and (in_ch != out_ch or stride != 1):
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        elif residual:
            self.skip = nn.Identity()
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.skip is not None:
            out = out + self.skip(identity)
        return self.relu(out)


# -----------------------------------------------------------------------
# Teacher Encoder
# -----------------------------------------------------------------------
class TeacherEncoder(nn.Module):
    """Dual-input teacher encoder for CT + PET distillation.

    Architecture overview::

        CT  ──► stem_ct  (1 → base_ch)  ─┐
                                          ├── concat → fuse_conv → encoder stages
        PET ──► stem_pet (1 → base_ch)  ─┘

    Each encoder stage doubles the channel count and halves spatial resolution.
    Intermediate features at every scale are stored for distillation.

    After training, call :meth:`freeze` to lock all parameters.

    Args:
        in_channels_ct: CT input channels (default 1).
        in_channels_pet: PET input channels (default 1).
        base_channels: Channel width after the stem (default 32).
        num_stages: Number of downsampling encoder stages (default 4).
        use_residual: Use residual connections in conv blocks.

    Example::

        teacher = TeacherEncoder()
        feats = teacher.extract_features(ct, pet)  # list of 4 tensors
    """

    def __init__(
        self,
        in_channels_ct: int = 1,
        in_channels_pet: int = 1,
        base_channels: int = 32,
        num_stages: int = 4,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.num_stages = num_stages

        # Separate stems for CT and PET
        self.stem_ct = nn.Sequential(
            nn.Conv3d(in_channels_ct, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.stem_pet = nn.Sequential(
            nn.Conv3d(in_channels_pet, base_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Fuse the two branches
        self.fuse_conv = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels, 1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Encoder stages with progressive downsampling
        stages: list[nn.Module] = []
        ch_in = base_channels
        for i in range(num_stages):
            ch_out = ch_in * 2
            stages.append(
                _ConvBlock(ch_in, ch_out, kernel_size=3, stride=2, residual=use_residual)
            )
            ch_in = ch_out
        self.stages = nn.ModuleList(stages)

        # Output head (simple heatmap decoder for teacher pre-training)
        self.decoder_head: Optional[nn.Module] = None
        self._build_decoder(base_channels, num_stages)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "TeacherEncoder: base_ch=%d  stages=%d  params=%.2fM",
            base_channels, num_stages, n_params / 1e6,
        )

    def _build_decoder(self, base_channels: int, num_stages: int) -> None:
        """Build a lightweight decoder for teacher pre-training.

        The decoder mirrors the encoder with transposed convolutions.

        Args:
            base_channels: Base channel width.
            num_stages: Number of stages to upsample.
        """
        ch = base_channels * (2 ** num_stages)
        layers: list[nn.Module] = []
        for _ in range(num_stages):
            ch_out = ch // 2
            layers.extend([
                nn.ConvTranspose3d(ch, ch_out, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm3d(ch_out),
                nn.ReLU(inplace=True),
            ])
            ch = ch_out
        layers.append(nn.Conv3d(ch, 1, kernel_size=1))
        layers.append(nn.Sigmoid())
        self.decoder_head = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------
    def extract_features(
        self,
        ct: torch.Tensor,
        pet: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Extract PET-informed multi-scale features.

        Args:
            ct: CT volume ``(B, 1, D, H, W)``.
            pet: PET volume ``(B, 1, D, H, W)``.

        Returns:
            List of feature tensors at ``num_stages`` resolutions, from
            finest (after first stage) to coarsest.
        """
        ct_feat = self.stem_ct(ct)
        pet_feat = self.stem_pet(pet)
        fused = self.fuse_conv(torch.cat([ct_feat, pet_feat], dim=1))

        features: List[torch.Tensor] = []
        x = fused
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

    def forward(
        self,
        ct: torch.Tensor,
        pet: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass (encoder + decoder).

        Used during teacher pre-training to produce a heatmap prediction.
        After pre-training, use :meth:`extract_features` for distillation.

        Args:
            ct: CT volume ``(B, 1, D, H, W)``.
            pet: PET volume ``(B, 1, D, H, W)``.

        Returns:
            Dict with ``"heatmap"`` and ``"features"`` keys.
        """
        features = self.extract_features(ct, pet)
        heatmap = self.decoder_head(features[-1]) if self.decoder_head is not None else None

        # Upsample to match input size if needed
        if heatmap is not None and heatmap.shape[2:] != ct.shape[2:]:
            heatmap = F.interpolate(
                heatmap, size=ct.shape[2:], mode="trilinear", align_corners=False
            )

        return {
            "heatmap": heatmap,
            "features": features,
        }

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------
    def freeze(self) -> None:
        """Freeze all parameters (for distillation as teacher).

        Sets ``requires_grad=False`` on every parameter and switches to eval
        mode.
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        logger.info("TeacherEncoder frozen: all parameters locked, eval mode set.")

    def unfreeze(self) -> None:
        """Unfreeze all parameters (e.g., for fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()
        logger.info("TeacherEncoder unfrozen: all parameters unlocked, train mode set.")
