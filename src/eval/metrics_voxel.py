# -*- coding: utf-8 -*-
"""CT2MAP-HN – Voxel-level segmentation metrics.

Computes standard binary segmentation metrics between predicted and ground-truth
volumes:

* **Dice coefficient** (F1 score for binary overlap).
* **IoU / Jaccard index**.
* **Sensitivity** (recall / true positive rate).
* **Specificity** (true negative rate).
* **95th-percentile Hausdorff distance** (surface-based error metric).

All functions accept either NumPy arrays or PyTorch tensors and return Python
floats.  Tensors are moved to CPU automatically.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Type alias: anything we accept as a volume
_ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: _ArrayLike) -> np.ndarray:
    """Convert a tensor or array to a NumPy array on CPU.

    Args:
        x: Input tensor or NumPy array.

    Returns:
        NumPy array (float64).
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _binarise(
    x: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Binarise a probability map.

    Args:
        x: Probability map in ``[0, 1]``.
        threshold: Binarisation threshold.

    Returns:
        Binary array of dtype ``bool``.
    """
    return (x >= threshold).astype(bool)


# ---------------------------------------------------------------------------
# Dice
# ---------------------------------------------------------------------------

def dice_score(
    pred: _ArrayLike,
    target: _ArrayLike,
    threshold: float = 0.5,
    smooth: float = 1e-7,
) -> float:
    """Compute the Dice coefficient (F1) between two binary volumes.

    .. math::
        \\text{Dice} = \\frac{2 |P \\cap T|}{|P| + |T|}

    Args:
        pred: Predicted segmentation (probabilities or binary).
        target: Ground-truth segmentation (probabilities or binary).
        threshold: Threshold to binarise soft predictions.
        smooth: Smoothing constant to avoid division by zero.

    Returns:
        Dice score in ``[0, 1]`` (higher is better).

    Example:
        >>> dice_score(np.ones((10, 10, 10)), np.ones((10, 10, 10)))
        1.0
    """
    p = _binarise(_to_numpy(pred), threshold)
    t = _binarise(_to_numpy(target), threshold)

    intersection = np.sum(p & t)
    denom = np.sum(p) + np.sum(t)

    if denom == 0:
        # Both empty → perfect match
        return 1.0

    return float((2.0 * intersection + smooth) / (denom + smooth))


# ---------------------------------------------------------------------------
# IoU / Jaccard
# ---------------------------------------------------------------------------

def iou_score(
    pred: _ArrayLike,
    target: _ArrayLike,
    threshold: float = 0.5,
    smooth: float = 1e-7,
) -> float:
    """Compute the Intersection-over-Union (Jaccard index).

    .. math::
        \\text{IoU} = \\frac{|P \\cap T|}{|P \\cup T|}

    Args:
        pred: Predicted segmentation.
        target: Ground-truth segmentation.
        threshold: Binarisation threshold.
        smooth: Smoothing constant.

    Returns:
        IoU in ``[0, 1]``.
    """
    p = _binarise(_to_numpy(pred), threshold)
    t = _binarise(_to_numpy(target), threshold)

    intersection = np.sum(p & t)
    union = np.sum(p | t)

    if union == 0:
        return 1.0

    return float((intersection + smooth) / (union + smooth))


# ---------------------------------------------------------------------------
# Sensitivity & Specificity
# ---------------------------------------------------------------------------

def sensitivity_score(
    pred: _ArrayLike,
    target: _ArrayLike,
    threshold: float = 0.5,
    smooth: float = 1e-7,
) -> float:
    """Compute sensitivity (recall / true positive rate).

    .. math::
        \\text{Sensitivity} = \\frac{TP}{TP + FN}

    Args:
        pred: Predicted segmentation.
        target: Ground-truth segmentation.
        threshold: Binarisation threshold.
        smooth: Smoothing constant.

    Returns:
        Sensitivity in ``[0, 1]``.
    """
    p = _binarise(_to_numpy(pred), threshold)
    t = _binarise(_to_numpy(target), threshold)

    tp = np.sum(p & t)
    fn = np.sum(~p & t)

    if tp + fn == 0:
        # No positives in ground truth
        return 1.0

    return float((tp + smooth) / (tp + fn + smooth))


def specificity_score(
    pred: _ArrayLike,
    target: _ArrayLike,
    threshold: float = 0.5,
    smooth: float = 1e-7,
) -> float:
    """Compute specificity (true negative rate).

    .. math::
        \\text{Specificity} = \\frac{TN}{TN + FP}

    Args:
        pred: Predicted segmentation.
        target: Ground-truth segmentation.
        threshold: Binarisation threshold.
        smooth: Smoothing constant.

    Returns:
        Specificity in ``[0, 1]``.
    """
    p = _binarise(_to_numpy(pred), threshold)
    t = _binarise(_to_numpy(target), threshold)

    tn = np.sum(~p & ~t)
    fp = np.sum(p & ~t)

    if tn + fp == 0:
        return 1.0

    return float((tn + smooth) / (tn + fp + smooth))


# ---------------------------------------------------------------------------
# Hausdorff Distance (95th percentile)
# ---------------------------------------------------------------------------

def hausdorff_distance_95(
    pred: _ArrayLike,
    target: _ArrayLike,
    threshold: float = 0.5,
    voxel_spacing: Optional[tuple[float, ...]] = None,
) -> float:
    """Compute the 95th-percentile Hausdorff distance between surfaces.

    Uses ``scipy.ndimage`` to extract surface voxels (voxels adjacent to
    background), then computes directed distances between the two surfaces
    and returns the 95th percentile of the combined distance set.

    Args:
        pred: Predicted segmentation.
        target: Ground-truth segmentation.
        threshold: Binarisation threshold.
        voxel_spacing: Physical spacing per axis ``(sz, sy, sx)`` in mm.
            If *None*, unit spacing is assumed.

    Returns:
        HD95 in the units defined by *voxel_spacing* (mm if spacing is mm).
        Returns ``float('inf')`` if either prediction or target is empty.

    Raises:
        ImportError: If ``scipy`` is not installed.
    """
    try:
        from scipy.ndimage import binary_erosion, generate_binary_structure
        from scipy.spatial.distance import cdist
    except ImportError as exc:
        raise ImportError(
            "scipy is required for hausdorff_distance_95. "
            "Install it with: pip install scipy"
        ) from exc

    p = _binarise(_to_numpy(pred), threshold)
    t = _binarise(_to_numpy(target), threshold)

    if not np.any(p) or not np.any(t):
        logger.warning(
            "HD95: empty prediction or target – returning inf."
        )
        return float("inf")

    # Extract surface voxels via erosion
    ndim = p.ndim
    struct = generate_binary_structure(ndim, 1)

    p_surface = p & ~binary_erosion(p, structure=struct)
    t_surface = t & ~binary_erosion(t, structure=struct)

    if not np.any(p_surface) or not np.any(t_surface):
        # Fall back to all foreground voxels if surface extraction fails
        p_surface = p
        t_surface = t

    p_coords = np.argwhere(p_surface).astype(np.float64)
    t_coords = np.argwhere(t_surface).astype(np.float64)

    # Apply voxel spacing
    if voxel_spacing is not None:
        spacing = np.array(voxel_spacing, dtype=np.float64)
        if spacing.shape[0] != ndim:
            raise ValueError(
                f"voxel_spacing has {spacing.shape[0]} elements but volume "
                f"has {ndim} dimensions."
            )
        p_coords *= spacing
        t_coords *= spacing

    # Compute directed distances
    dist_p_to_t = cdist(p_coords, t_coords, metric="euclidean").min(axis=1)
    dist_t_to_p = cdist(t_coords, p_coords, metric="euclidean").min(axis=1)

    all_distances = np.concatenate([dist_p_to_t, dist_t_to_p])
    hd95 = float(np.percentile(all_distances, 95))

    return hd95
