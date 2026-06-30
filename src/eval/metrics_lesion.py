# -*- coding: utf-8 -*-
"""CT2MAP-HN – Lesion / heatmap quality metrics.

Computes continuous-valued metrics that measure how closely a predicted
metabolic risk heatmap matches the ground-truth (PET-derived soft target):

* **Voxel MAE** – mean absolute error over the volume.
* **Voxel MSE** – mean squared error over the volume.
* **Voxel RMSE** – root-mean-squared error.
* **Pearson correlation** – linear correlation between predicted and target
  voxel intensities.
* **Spearman correlation** – rank-based correlation.
* **SSIM** – structural similarity index (3-D, computed slice-by-slice then
  averaged or via 3-D windowing).

All functions accept NumPy arrays or PyTorch tensors.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

_ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: _ArrayLike) -> np.ndarray:
    """Convert to NumPy float64 array.

    Args:
        x: Input tensor or array.

    Returns:
        NumPy array (float64).
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _flatten_pair(
    pred: _ArrayLike,
    target: _ArrayLike,
    mask: Optional[_ArrayLike] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten prediction and target, optionally applying a mask.

    Args:
        pred: Predicted values.
        target: Ground-truth values.
        mask: Optional binary mask to restrict metrics to a region.

    Returns:
        Tuple of flattened 1-D arrays ``(pred_flat, target_flat)``.

    Raises:
        ValueError: If shapes do not match.
    """
    p = _to_numpy(pred)
    t = _to_numpy(target)

    if p.shape != t.shape:
        raise ValueError(
            f"Shape mismatch: pred {p.shape} vs target {t.shape}"
        )

    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        if m.shape != p.shape:
            raise ValueError(
                f"Mask shape {m.shape} does not match pred shape {p.shape}"
            )
        p = p[m]
        t = t[m]

    return p.ravel(), t.ravel()


# ---------------------------------------------------------------------------
# MAE / MSE / RMSE
# ---------------------------------------------------------------------------

def voxel_mae(
    pred: _ArrayLike,
    target: _ArrayLike,
    mask: Optional[_ArrayLike] = None,
) -> float:
    """Compute voxel-wise mean absolute error.

    Args:
        pred: Predicted heatmap.
        target: Ground-truth heatmap.
        mask: Optional binary mask to restrict to a region of interest.

    Returns:
        Scalar MAE value.
    """
    p, t = _flatten_pair(pred, target, mask)
    if p.size == 0:
        logger.warning("voxel_mae: empty region after masking, returning 0.0")
        return 0.0
    return float(np.mean(np.abs(p - t)))


def voxel_mse(
    pred: _ArrayLike,
    target: _ArrayLike,
    mask: Optional[_ArrayLike] = None,
) -> float:
    """Compute voxel-wise mean squared error.

    Args:
        pred: Predicted heatmap.
        target: Ground-truth heatmap.
        mask: Optional binary mask.

    Returns:
        Scalar MSE value.
    """
    p, t = _flatten_pair(pred, target, mask)
    if p.size == 0:
        logger.warning("voxel_mse: empty region after masking, returning 0.0")
        return 0.0
    return float(np.mean((p - t) ** 2))


def voxel_rmse(
    pred: _ArrayLike,
    target: _ArrayLike,
    mask: Optional[_ArrayLike] = None,
) -> float:
    """Compute voxel-wise root-mean-squared error.

    Args:
        pred: Predicted heatmap.
        target: Ground-truth heatmap.
        mask: Optional binary mask.

    Returns:
        Scalar RMSE value.
    """
    return float(np.sqrt(voxel_mse(pred, target, mask)))


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def pearson_correlation(
    pred: _ArrayLike,
    target: _ArrayLike,
    mask: Optional[_ArrayLike] = None,
) -> float:
    """Compute Pearson correlation coefficient between two volumes.

    Args:
        pred: Predicted heatmap.
        target: Ground-truth heatmap.
        mask: Optional binary mask.

    Returns:
        Pearson *r* in ``[-1, 1]``.  Returns 0.0 if either volume has
        zero variance.
    """
    p, t = _flatten_pair(pred, target, mask)
    if p.size < 2:
        logger.warning(
            "pearson_correlation: fewer than 2 voxels, returning 0.0"
        )
        return 0.0

    p_std = np.std(p)
    t_std = np.std(t)
    if p_std < 1e-12 or t_std < 1e-12:
        logger.warning(
            "pearson_correlation: near-zero variance (pred_std=%.2e, "
            "target_std=%.2e), returning 0.0",
            p_std, t_std,
        )
        return 0.0

    corr_matrix = np.corrcoef(p, t)
    return float(corr_matrix[0, 1])


def spearman_correlation(
    pred: _ArrayLike,
    target: _ArrayLike,
    mask: Optional[_ArrayLike] = None,
) -> float:
    """Compute Spearman rank correlation between two volumes.

    Args:
        pred: Predicted heatmap.
        target: Ground-truth heatmap.
        mask: Optional binary mask.

    Returns:
        Spearman *ρ* in ``[-1, 1]``.  Returns 0.0 if fewer than 2 voxels.

    Raises:
        ImportError: If ``scipy`` is not installed.
    """
    try:
        from scipy.stats import spearmanr
    except ImportError as exc:
        raise ImportError(
            "scipy is required for spearman_correlation. "
            "Install with: pip install scipy"
        ) from exc

    p, t = _flatten_pair(pred, target, mask)
    if p.size < 2:
        logger.warning(
            "spearman_correlation: fewer than 2 voxels, returning 0.0"
        )
        return 0.0

    rho, _ = spearmanr(p, t)
    return float(rho) if np.isfinite(rho) else 0.0


# ---------------------------------------------------------------------------
# SSIM (3-D)
# ---------------------------------------------------------------------------

def ssim_3d(
    pred: _ArrayLike,
    target: _ArrayLike,
    win_size: int = 7,
    data_range: Optional[float] = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Compute mean structural similarity (SSIM) over a 3-D volume.

    Computes SSIM slice-by-slice along the depth axis and returns the mean.
    Uses the simplified SSIM formula with a uniform window.

    Args:
        pred: Predicted heatmap ``(D, H, W)`` or ``(1, D, H, W)``.
        target: Ground-truth heatmap with the same shape.
        win_size: Window size for local statistics (must be odd).
        data_range: Dynamic range of the data.  If *None*, inferred from
            the target as ``target.max() - target.min()``.
        k1: SSIM constant ``k1`` (stabilises luminance).
        k2: SSIM constant ``k2`` (stabilises contrast).

    Returns:
        Mean SSIM in ``[-1, 1]`` (typically ``[0, 1]`` for non-negative data).

    Raises:
        ImportError: If ``skimage`` is not installed.
    """
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for ssim_3d. "
            "Install with: pip install scikit-image"
        ) from exc

    p = _to_numpy(pred).squeeze()
    t = _to_numpy(target).squeeze()

    if p.ndim != 3:
        raise ValueError(
            f"ssim_3d expects 3-D volumes, got pred shape {p.shape}"
        )
    if p.shape != t.shape:
        raise ValueError(
            f"Shape mismatch: pred {p.shape} vs target {t.shape}"
        )

    if data_range is None:
        data_range = float(t.max() - t.min())
        if data_range < 1e-12:
            data_range = 1.0

    # Ensure win_size does not exceed any spatial dimension
    max_dim = min(p.shape)
    effective_win = min(win_size, max_dim)
    if effective_win % 2 == 0:
        effective_win = max(effective_win - 1, 1)

    ssim_val = structural_similarity(
        p,
        t,
        data_range=data_range,
        win_size=effective_win,
        channel_axis=None,  # 3-D volumetric, not multi-channel
        K1=k1,
        K2=k2,
    )

    return float(ssim_val)
