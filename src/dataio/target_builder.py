# -*- coding: utf-8 -*-
"""Target heatmap generation for CT2MAP-HN.

Tạo các loại target từ mask và PET:
  - Binary target (nhị phân)
  - Gaussian-smoothed heatmap
  - PET-derived soft target
  - Case-level labels (high_risk, metabolic activity index)
"""

import logging
from typing import Any, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


def create_binary_target(mask_array: np.ndarray) -> np.ndarray:
    """Create a binary segmentation target from a lesion mask.

    Converts any non-zero value to 1.

    Args:
        mask_array: Input mask array (any shape). Values > 0 are foreground.

    Returns:
        Binary numpy array (float32) with values 0.0 or 1.0.
    """
    binary = (mask_array > 0).astype(np.float32)
    fg_ratio = binary.sum() / max(binary.size, 1)
    logger.debug(
        "Binary target: shape=%s, fg_ratio=%.6f",
        binary.shape, fg_ratio,
    )
    return binary


def create_gaussian_heatmap(
    mask_array: np.ndarray,
    sigma: float = 3.0,
) -> np.ndarray:
    """Create a Gaussian-smoothed heatmap from a binary mask.

    The Gaussian filter creates smooth distance-like falloff around lesion
    boundaries, providing a softer regression target than binary masks.

    Args:
        mask_array: Binary mask array (any shape). Values > 0 are foreground.
        sigma: Standard deviation of the Gaussian kernel in voxels.
            Larger values produce smoother heatmaps.

    Returns:
        Heatmap numpy array (float32) with values in [0.0, 1.0].

    Raises:
        ValueError: If sigma is non-positive.
    """
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}.")

    binary = (mask_array > 0).astype(np.float32)

    # Early return if mask is empty
    if binary.sum() == 0:
        logger.warning("Empty mask; returning zero heatmap.")
        return np.zeros_like(binary)

    heatmap = gaussian_filter(binary, sigma=sigma)

    # Normalize to [0, 1]
    hm_max = heatmap.max()
    if hm_max > 0:
        heatmap = heatmap / hm_max

    logger.debug(
        "Gaussian heatmap: sigma=%.1f, range=[%.4f, %.4f]",
        sigma, heatmap.min(), heatmap.max(),
    )
    return heatmap.astype(np.float32)


def create_pet_derived_target(
    pet_array: np.ndarray,
    mask_array: np.ndarray,
    normalize: bool = True,
    suv_min: float = 0.0,
    suv_max: float = 25.0,
) -> np.ndarray:
    """Create a PET-derived soft target using PET activity within mask regions.

    The target represents metabolic activity: PET intensity is masked and
    optionally normalized so the network learns to predict metabolic hotspots.

    Args:
        pet_array: PET image array (same shape as mask_array).
        mask_array: Binary mask array.
        normalize: Whether to normalize the target to [0, 1] range.
        suv_min: Lower SUV clipping bound.
        suv_max: Upper SUV clipping bound.

    Returns:
        PET-derived target array (float32).

    Raises:
        ValueError: If shapes don't match.
    """
    if pet_array.shape != mask_array.shape:
        raise ValueError(
            f"Shape mismatch: PET {pet_array.shape} vs mask {mask_array.shape}."
        )

    binary_mask = (mask_array > 0).astype(np.float32)

    # Clip PET values
    pet_clipped = np.clip(pet_array, suv_min, suv_max).astype(np.float32)

    # Mask PET to lesion regions
    target = pet_clipped * binary_mask

    if normalize and target.max() > 0:
        target = target / target.max()

    fg_voxels = binary_mask.sum()
    mean_activity = target[binary_mask > 0].mean() if fg_voxels > 0 else 0.0

    logger.debug(
        "PET-derived target: fg_voxels=%d, mean_activity=%.4f",
        int(fg_voxels), mean_activity,
    )
    return target


def create_case_label(
    mask_array: np.ndarray,
    pet_array: Optional[np.ndarray] = None,
    high_risk_threshold: float = 0.3,
) -> dict[str, Any]:
    """Create case-level classification labels from mask and PET.

    Labels include:
        - has_active_lesion: whether any foreground exists in the mask
        - high_risk: whether metabolic activity exceeds threshold
        - metabolic_activity_index: normalized mean PET within mask (0-1)

    Args:
        mask_array: Binary mask array.
        pet_array: Optional PET array for metabolic features.
        high_risk_threshold: Threshold for ``metabolic_activity_index``
            above which a case is considered high-risk.

    Returns:
        Dictionary with case-level labels:
            - has_active_lesion (bool)
            - has_primary (bool)
            - has_nodes (bool)
            - high_risk (bool)
            - metabolic_activity_index (float)
            - lesion_volume_voxels (int)
    """
    binary = (mask_array > 0)
    has_active_lesion = bool(binary.any())
    lesion_volume = int(binary.sum())

    # Primary vs node detection heuristic:
    # In HECKTOR, label 1 = primary GTV, label 2 = nodal GTV
    has_primary = bool((mask_array == 1).any())
    has_nodes = bool((mask_array == 2).any())

    # If mask is purely binary (0/1), treat all foreground as primary
    unique_labels = np.unique(mask_array)
    if set(unique_labels.tolist()) <= {0, 1}:
        has_primary = has_active_lesion
        has_nodes = False

    # Metabolic activity index
    metabolic_index = 0.0
    if pet_array is not None and has_active_lesion:
        pet_in_mask = pet_array[binary]
        # Normalize by PET global max (or use a fixed SUV ceiling)
        pet_max = max(pet_array.max(), 1e-8)
        metabolic_index = float(pet_in_mask.mean() / pet_max)

    high_risk = metabolic_index > high_risk_threshold

    label = {
        "has_active_lesion": has_active_lesion,
        "has_primary": has_primary,
        "has_nodes": has_nodes,
        "high_risk": high_risk,
        "metabolic_activity_index": round(metabolic_index, 6),
        "lesion_volume_voxels": lesion_volume,
    }

    logger.debug("Case label: %s", label)
    return label


def build_targets(
    case_dict: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Full target-building pipeline for a single case.

    Generates all configured target types and case-level labels from
    the preprocessed case data.

    Args:
        case_dict: Dictionary from ``preprocess_case`` or loaded data,
            expected keys: 'mask_array', optionally 'pet_array'.
        config: Target config dict (from preprocess.yaml['targets']).

    Returns:
        Dictionary with keys:
            - 'binary_target': np.ndarray or None
            - 'gaussian_heatmap': np.ndarray or None
            - 'pet_derived_target': np.ndarray or None
            - 'case_label': dict of case-level labels
            - 'target_types': list of generated target type names

    Raises:
        ValueError: If mask_array is missing from case_dict.
    """
    mask_array = case_dict.get("mask_array")
    if mask_array is None:
        raise ValueError("mask_array is required in case_dict to build targets.")

    pet_array = case_dict.get("pet_array")
    target_types = config.get("types", ["binary", "gaussian_heatmap"])
    sigma = config.get("gaussian_sigma", 3.0)
    pet_normalize = config.get("pet_normalize", True)
    suv_min = config.get("pet_suv_min", 0.0)
    suv_max = config.get("pet_suv_max", 25.0)
    risk_threshold = config.get("high_risk_threshold", 0.3)

    result: dict[str, Any] = {
        "binary_target": None,
        "gaussian_heatmap": None,
        "pet_derived_target": None,
        "case_label": {},
        "target_types": [],
    }

    # Generate requested targets
    if "binary" in target_types:
        result["binary_target"] = create_binary_target(mask_array)
        result["target_types"].append("binary")

    if "gaussian_heatmap" in target_types:
        result["gaussian_heatmap"] = create_gaussian_heatmap(mask_array, sigma=sigma)
        result["target_types"].append("gaussian_heatmap")

    if "pet_derived" in target_types:
        if pet_array is not None:
            result["pet_derived_target"] = create_pet_derived_target(
                pet_array, mask_array,
                normalize=pet_normalize,
                suv_min=suv_min,
                suv_max=suv_max,
            )
            result["target_types"].append("pet_derived")
        else:
            logger.warning(
                "PET array not available; skipping pet_derived target."
            )

    # Case-level label
    result["case_label"] = create_case_label(
        mask_array,
        pet_array=pet_array,
        high_risk_threshold=risk_threshold,
    )

    logger.info(
        "Built targets: types=%s, label=%s",
        result["target_types"], result["case_label"],
    )
    return result
