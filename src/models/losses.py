"""
CT2MAP-HN – Loss functions.

Provides task-specific and combined losses for the three prediction heads,
plus distillation losses for teacher–student training:

* :class:`HeatmapLoss` – voxel regression with optional focal weighting.
* :class:`LesionLoss` – Dice + BCE/Focal for binary segmentation.
* :class:`TriageLoss` – BCE for case-level classification.
* :class:`DistillationLoss` – feature-matching + output-consistency.
* :class:`CombinedLoss` – weighted sum of the three task losses.
* :class:`CombinedDistillLoss` – task loss plus distillation terms.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ====================================================================
# Utility helpers
# ====================================================================

def _focal_weight(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Compute per-voxel focal weight ``(1 - p_t)^gamma``.

    Args:
        pred: Predicted probabilities in ``[0, 1]``.
        target: Ground-truth binary labels.
        gamma: Focal exponent.

    Returns:
        Weight tensor with the same shape as *pred*.
    """
    p_t = pred * target + (1.0 - pred) * (1.0 - target)
    return (1.0 - p_t).pow(gamma)


def _dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Soft Dice coefficient (per-sample, averaged over batch).

    Args:
        pred: Predicted probabilities ``(B, 1, D, H, W)``.
        target: Ground-truth ``(B, 1, D, H, W)``.
        smooth: Smoothing constant to avoid division by zero.

    Returns:
        Scalar Dice score (higher is better).
    """
    # Flatten spatial dims
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()


# ====================================================================
# HeatmapLoss
# ====================================================================
class HeatmapLoss(nn.Module):
    """Voxel-wise regression loss for metabolic risk heatmaps.

    Supports MSE or L1 base loss with optional *focal* weighting that
    upweights voxels inside the lesion region.

    Args:
        base: ``"mse"`` or ``"l1"``.
        use_focal: Whether to apply focal weighting using a lesion mask.
        focal_gamma: Exponent for focal weighting.
        lesion_weight: Extra multiplicative weight for voxels inside the lesion.
    """

    def __init__(
        self,
        base: str = "mse",
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        lesion_weight: float = 5.0,
    ) -> None:
        super().__init__()
        self.base = base.lower()
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.lesion_weight = lesion_weight

        if self.base not in ("mse", "l1"):
            raise ValueError(f"Unsupported base loss '{self.base}'. Use 'mse' or 'l1'.")
        logger.info(
            "HeatmapLoss: base=%s  focal=%s  gamma=%.1f  lesion_w=%.1f",
            self.base, self.use_focal, self.focal_gamma, self.lesion_weight,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lesion_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute heatmap regression loss.

        Args:
            pred: Predicted heatmap ``(B, 1, D, H, W)`` in ``[0, 1]``.
            target: Ground-truth heatmap ``(B, 1, D, H, W)``.
            lesion_mask: Optional binary mask ``(B, 1, D, H, W)`` for focal
                weighting.  Required if ``use_focal=True``.

        Returns:
            Scalar loss value.
        """
        if self.base == "mse":
            loss_map = F.mse_loss(pred, target, reduction="none")
        else:
            loss_map = F.l1_loss(pred, target, reduction="none")

        if self.use_focal and lesion_mask is not None:
            # Upweight lesion voxels
            weight = torch.ones_like(loss_map)
            weight = weight + lesion_mask * (self.lesion_weight - 1.0)
            # Optional focal modulation on prediction error
            focal_w = _focal_weight(pred, target, self.focal_gamma)
            weight = weight * focal_w
            loss_map = loss_map * weight

        return loss_map.mean()


# ====================================================================
# LesionLoss
# ====================================================================
class LesionLoss(nn.Module):
    """Combined Dice + BCE/Focal loss for binary lesion segmentation.

    Args:
        bce_weight: Weight for the BCE (or focal) component.
        dice_weight: Weight for the Dice component.
        use_focal: If *True*, use focal BCE instead of standard BCE.
        focal_alpha: Alpha parameter for focal loss.
        focal_gamma: Gamma parameter for focal loss.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        use_focal: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        logger.info(
            "LesionLoss: bce_w=%.2f  dice_w=%.2f  focal=%s",
            bce_weight, dice_weight, use_focal,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute lesion segmentation loss.

        Args:
            pred: Predicted probabilities ``(B, 1, D, H, W)`` in ``[0, 1]``.
            target: Ground-truth binary mask ``(B, 1, D, H, W)``.

        Returns:
            Scalar combined loss value.
        """
        # --- BCE / Focal component ---
        if self.use_focal:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
            focal_w = _focal_weight(pred, target, self.focal_gamma)
            alpha_w = self.focal_alpha * target + (1.0 - self.focal_alpha) * (1.0 - target)
            bce_loss = (alpha_w * focal_w * bce).mean()
        else:
            bce_loss = F.binary_cross_entropy(pred, target, reduction="mean")

        # --- Dice component ---
        dice_loss = 1.0 - _dice_score(pred, target)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ====================================================================
# TriageLoss
# ====================================================================
class TriageLoss(nn.Module):
    """Binary cross-entropy loss for case-level triage classification.

    Args:
        pos_weight: Optional positive class weight to handle class imbalance.
    """

    def __init__(self, pos_weight: Optional[float] = None) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        if pos_weight is not None:
            logger.info("TriageLoss: pos_weight=%.2f", pos_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute triage loss.

        Args:
            pred: Predicted probabilities ``(B, 1)`` in ``[0, 1]``.
            target: Ground-truth labels ``(B, 1)`` in ``{0, 1}``.

        Returns:
            Scalar BCE loss.
        """
        if self.pos_weight is not None:
            # Use logit-based BCE with pos_weight for numerical stability.
            # We need logits: clamp pred then compute inverse-sigmoid.
            pred_clamped = pred.clamp(1e-6, 1.0 - 1e-6)
            logits = torch.log(pred_clamped / (1.0 - pred_clamped))
            pw = torch.tensor(
                [self.pos_weight], device=pred.device, dtype=pred.dtype
            )
            return F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pw
            )
        return F.binary_cross_entropy(pred, target)


# ====================================================================
# DistillationLoss
# ====================================================================
class DistillationLoss(nn.Module):
    """Knowledge distillation loss with feature matching + output consistency.

    Computes:

    1. **Feature-matching loss**: MSE between aligned intermediate features
       from the teacher and the student (at multiple scales).
    2. **Output-consistency loss**: MSE between the teacher's output heatmap
       and the student's output heatmap.

    Args:
        feature_weight: Weight for the feature-matching component.
        output_weight: Weight for the output-consistency component.
        temperature: Softening temperature (applied as divisor before
            recomputing sigmoid) for output consistency.
    """

    def __init__(
        self,
        feature_weight: float = 1.0,
        output_weight: float = 1.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.feature_weight = feature_weight
        self.output_weight = output_weight
        self.temperature = temperature
        self._adaptation_layers: nn.ModuleList | None = None
        logger.info(
            "DistillationLoss: feat_w=%.2f  out_w=%.2f  temp=%.1f",
            feature_weight, output_weight, temperature,
        )

    def _maybe_build_adaptors(
        self,
        student_feats: List[torch.Tensor],
        teacher_feats: List[torch.Tensor],
    ) -> None:
        """Lazily create 1×1×1 conv adaptors if channel dims differ."""
        if self._adaptation_layers is not None:
            return
        layers: list[nn.Module] = []
        for s_feat, t_feat in zip(student_feats, teacher_feats):
            s_ch = s_feat.shape[1]
            t_ch = t_feat.shape[1]
            if s_ch != t_ch:
                layers.append(
                    nn.Conv3d(s_ch, t_ch, kernel_size=1, bias=False).to(s_feat.device)
                )
            else:
                layers.append(nn.Identity())
        self._adaptation_layers = nn.ModuleList(layers)

    def forward(
        self,
        student_feats: List[torch.Tensor],
        teacher_feats: List[torch.Tensor],
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute distillation loss.

        Args:
            student_feats: Student intermediate features (list of tensors).
            teacher_feats: Teacher intermediate features (list of tensors,
                same length as *student_feats*).
            student_output: Student heatmap output ``(B, 1, D, H, W)``.
            teacher_output: Teacher heatmap output ``(B, 1, D, H, W)``.

        Returns:
            Dictionary with ``"feature_loss"``, ``"output_loss"``, and
            ``"total"`` scalar tensors.
        """
        # --- Feature matching ---
        self._maybe_build_adaptors(student_feats, teacher_feats)
        assert self._adaptation_layers is not None

        feat_loss = torch.tensor(0.0, device=student_output.device)
        n_scales = min(len(student_feats), len(teacher_feats))
        for i in range(n_scales):
            s_f = self._adaptation_layers[i](student_feats[i])
            t_f = teacher_feats[i].detach()
            # Spatial alignment if needed
            if s_f.shape[2:] != t_f.shape[2:]:
                s_f = F.interpolate(s_f, size=t_f.shape[2:], mode="trilinear", align_corners=False)
            feat_loss = feat_loss + F.mse_loss(s_f, t_f)
        if n_scales > 0:
            feat_loss = feat_loss / n_scales

        # --- Output consistency ---
        # Temperature-scaled soft targets
        if self.temperature != 1.0:
            # Convert sigmoid outputs back to logits, scale, re-sigmoid
            s_clamped = student_output.clamp(1e-6, 1 - 1e-6)
            t_clamped = teacher_output.clamp(1e-6, 1 - 1e-6)
            s_logits = torch.log(s_clamped / (1.0 - s_clamped)) / self.temperature
            t_logits = torch.log(t_clamped / (1.0 - t_clamped)) / self.temperature
            out_loss = F.mse_loss(torch.sigmoid(s_logits), torch.sigmoid(t_logits).detach())
        else:
            out_loss = F.mse_loss(student_output, teacher_output.detach())

        total = self.feature_weight * feat_loss + self.output_weight * out_loss
        return {
            "feature_loss": feat_loss,
            "output_loss": out_loss,
            "total": total,
        }


# ====================================================================
# CombinedLoss
# ====================================================================
class CombinedLoss(nn.Module):
    """Weighted combination of the three task losses.

    ``total = w_heatmap * L_heatmap + w_lesion * L_lesion + w_triage * L_triage``

    Args:
        heatmap_loss_cfg: Kwargs for :class:`HeatmapLoss`.
        lesion_loss_cfg: Kwargs for :class:`LesionLoss`.
        triage_loss_cfg: Kwargs for :class:`TriageLoss`.
        w_heatmap: Weight for heatmap loss.
        w_lesion: Weight for lesion loss.
        w_triage: Weight for triage loss.
    """

    def __init__(
        self,
        heatmap_loss_cfg: Optional[dict] = None,
        lesion_loss_cfg: Optional[dict] = None,
        triage_loss_cfg: Optional[dict] = None,
        w_heatmap: float = 1.0,
        w_lesion: float = 1.0,
        w_triage: float = 0.5,
    ) -> None:
        super().__init__()
        self.heatmap_loss = HeatmapLoss(**(heatmap_loss_cfg or {}))
        self.lesion_loss = LesionLoss(**(lesion_loss_cfg or {}))
        self.triage_loss = TriageLoss(**(triage_loss_cfg or {}))
        self.w_heatmap = w_heatmap
        self.w_lesion = w_lesion
        self.w_triage = w_triage
        logger.info(
            "CombinedLoss: w_heat=%.2f  w_les=%.2f  w_tri=%.2f",
            w_heatmap, w_lesion, w_triage,
        )

    def forward(
        self,
        pred_heatmap: torch.Tensor,
        target_heatmap: torch.Tensor,
        pred_lesion: torch.Tensor,
        target_lesion: torch.Tensor,
        pred_triage: torch.Tensor,
        target_triage: torch.Tensor,
        lesion_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            pred_heatmap: ``(B, 1, D, H, W)`` predicted heatmap.
            target_heatmap: ``(B, 1, D, H, W)`` target heatmap.
            pred_lesion: ``(B, 1, D, H, W)`` predicted lesion mask.
            target_lesion: ``(B, 1, D, H, W)`` target lesion mask.
            pred_triage: ``(B, 1)`` predicted triage probability.
            target_triage: ``(B, 1)`` target triage label.
            lesion_mask: Optional mask for focal heatmap weighting.

        Returns:
            Dictionary with per-component and ``"total"`` losses.
        """
        l_heat = self.heatmap_loss(pred_heatmap, target_heatmap, lesion_mask)
        l_les = self.lesion_loss(pred_lesion, target_lesion)
        l_tri = self.triage_loss(pred_triage, target_triage)

        total = (
            self.w_heatmap * l_heat
            + self.w_lesion * l_les
            + self.w_triage * l_tri
        )
        return {
            "heatmap_loss": l_heat,
            "lesion_loss": l_les,
            "triage_loss": l_tri,
            "total": total,
        }


# ====================================================================
# CombinedDistillLoss
# ====================================================================
class CombinedDistillLoss(nn.Module):
    """Task loss + distillation terms for teacher–student training.

    ``total = task_loss + lambda_feature * feature_loss + lambda_output * output_loss``

    The distillation weight can be ramped up via :meth:`set_distill_weight`.

    Args:
        combined_loss_cfg: Kwargs for :class:`CombinedLoss`.
        distill_loss_cfg: Kwargs for :class:`DistillationLoss`.
        lambda_feature: Global weight for feature-matching distillation.
        lambda_output: Global weight for output-consistency distillation.
    """

    def __init__(
        self,
        combined_loss_cfg: Optional[dict] = None,
        distill_loss_cfg: Optional[dict] = None,
        lambda_feature: float = 1.0,
        lambda_output: float = 1.0,
    ) -> None:
        super().__init__()
        self.task_loss = CombinedLoss(**(combined_loss_cfg or {}))
        self.distill_loss = DistillationLoss(**(distill_loss_cfg or {}))
        self.lambda_feature = lambda_feature
        self.lambda_output = lambda_output
        logger.info(
            "CombinedDistillLoss: lambda_feat=%.3f  lambda_out=%.3f",
            lambda_feature, lambda_output,
        )

    def set_distill_weight(self, alpha: float) -> None:
        """Scale both distillation lambdas by a common factor.

        Useful for warmup: start with ``alpha=0`` (task-only) and gradually
        increase to ``1.0``.

        Args:
            alpha: Multiplier in ``[0, 1]`` applied to both lambda values.
        """
        self._alpha = max(0.0, min(1.0, alpha))
        logger.debug("Distillation alpha set to %.4f", self._alpha)

    @property
    def _current_alpha(self) -> float:
        return getattr(self, "_alpha", 1.0)

    def forward(
        self,
        # Task outputs
        pred_heatmap: torch.Tensor,
        target_heatmap: torch.Tensor,
        pred_lesion: torch.Tensor,
        target_lesion: torch.Tensor,
        pred_triage: torch.Tensor,
        target_triage: torch.Tensor,
        # Distillation inputs
        student_feats: List[torch.Tensor],
        teacher_feats: List[torch.Tensor],
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        lesion_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined task + distillation loss.

        Args:
            pred_heatmap: Student heatmap prediction.
            target_heatmap: Ground-truth heatmap.
            pred_lesion: Student lesion prediction.
            target_lesion: Ground-truth lesion mask.
            pred_triage: Student triage prediction.
            target_triage: Triage label.
            student_feats: Student intermediate features for distillation.
            teacher_feats: Teacher intermediate features (detached).
            student_output: Student heatmap for output consistency.
            teacher_output: Teacher heatmap for output consistency.
            lesion_mask: Optional lesion mask for focal weighting.

        Returns:
            Dictionary with per-component losses, ``"task_total"``,
            ``"distill_total"``, and ``"total"``.
        """
        task = self.task_loss(
            pred_heatmap, target_heatmap,
            pred_lesion, target_lesion,
            pred_triage, target_triage,
            lesion_mask,
        )

        distill = self.distill_loss(
            student_feats, teacher_feats,
            student_output, teacher_output,
        )

        alpha = self._current_alpha
        total = (
            task["total"]
            + alpha * self.lambda_feature * distill["feature_loss"]
            + alpha * self.lambda_output * distill["output_loss"]
        )

        return {
            **task,
            "task_total": task["total"],
            "distill_feature_loss": distill["feature_loss"],
            "distill_output_loss": distill["output_loss"],
            "distill_total": distill["total"],
            "distill_alpha": torch.tensor(alpha),
            "total": total,
        }
