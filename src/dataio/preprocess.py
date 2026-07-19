# -*- coding: utf-8 -*-
"""Preprocessing functions for CT2MAP-HN.

Chuẩn hóa hướng ảnh, resampling, crop vùng đầu-cổ, clip HU,
và normalize cường độ. Tất cả phép biến đổi không gian dùng SimpleITK,
kết quả trả về numpy array.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Orientation Standardization
# --------------------------------------------------------------------------- #


def standardize_orientation(
    image_sitk: sitk.Image,
    target: str = "RAS",
) -> sitk.Image:
    """Re-orient a SimpleITK image to a standard anatomical orientation.

    Args:
        image_sitk: Input SimpleITK image.
        target: Target orientation code (e.g. 'RAS', 'LPS').
            Uses SimpleITK DICOMOrient filter.

    Returns:
        Re-oriented SimpleITK image.

    Raises:
        ValueError: If the target orientation code is invalid.
    """
    valid_axes = set("RLAPSI")
    if len(target) != 3 or not all(c in valid_axes for c in target.upper()):
        raise ValueError(
            f"Invalid orientation code '{target}'. "
            f"Must be 3 chars from {valid_axes}."
        )

    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation(target.upper())
    oriented = orienter.Execute(image_sitk)

    logger.debug(
        "Orientation: %s -> %s",
        sitk.DICOMOrientImageFilter().GetOrientationFromDirectionCosines(
            image_sitk.GetDirection()
        ),
        target,
    )
    return oriented


# --------------------------------------------------------------------------- #
#  Resampling
# --------------------------------------------------------------------------- #


def resample_volume(
    image_sitk: sitk.Image,
    target_spacing: tuple[float, float, float],
    interpolation: str = "linear",
) -> sitk.Image:
    """Resample a volume to a target isotropic/anisotropic spacing.

    Args:
        image_sitk: Input SimpleITK image.
        target_spacing: Desired voxel spacing (x, y, z) in mm.
        interpolation: Interpolation method — 'linear', 'nearest',
            'bspline', or 'gaussian'.

    Returns:
        Resampled SimpleITK image.

    Raises:
        ValueError: If interpolation method is unknown.
    """
    interp_map = {
        "linear": sitk.sitkLinear,
        "nearest": sitk.sitkNearestNeighbor,
        "bspline": sitk.sitkBSpline,
        "gaussian": sitk.sitkGaussian,
    }
    if interpolation not in interp_map:
        raise ValueError(
            f"Unknown interpolation '{interpolation}'. "
            f"Choose from {list(interp_map.keys())}."
        )

    original_spacing = image_sitk.GetSpacing()
    original_size = image_sitk.GetSize()

    # Compute new size
    new_size = [
        int(round(osz * osp / tsp))
        for osz, osp, tsp in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image_sitk.GetDirection())
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interp_map[interpolation])

    # Use appropriate default pixel value
    resampler.SetDefaultPixelValue(
        float(sitk.GetArrayViewFromImage(image_sitk).min())
    )

    resampled = resampler.Execute(image_sitk)

    logger.debug(
        "Resampled: spacing %s -> %s | size %s -> %s",
        original_spacing, target_spacing, original_size, new_size,
    )
    return resampled


# --------------------------------------------------------------------------- #
#  HU Clipping & Normalization
# --------------------------------------------------------------------------- #


def clip_and_normalize_hu(
    array: np.ndarray,
    hu_min: float = -1024.0,
    hu_max: float = 3071.0,
    method: str = "minmax",
    *,
    zscore_mean: Optional[float] = None,
    zscore_std: Optional[float] = None,
) -> np.ndarray:
    """Clip Hounsfield Units and normalize intensity values.

    Args:
        array: Input CT array (any shape).
        hu_min: Lower HU clipping bound.
        hu_max: Upper HU clipping bound.
        method: Normalization method — 'minmax' scales to [0, 1],
            'zscore' standardizes to zero mean / unit variance.
        zscore_mean: Mean for z-score normalization (required if method='zscore').
        zscore_std: Std for z-score normalization (required if method='zscore').

    Returns:
        Normalized numpy array as float32.

    Raises:
        ValueError: If method is unknown or z-score params are missing.
    """
    if method not in ("minmax", "zscore"):
        raise ValueError(f"Unknown normalize method '{method}'. Use 'minmax' or 'zscore'.")

    clipped = np.clip(array, hu_min, hu_max).astype(np.float32)

    if method == "minmax":
        denom = hu_max - hu_min
        if denom == 0:
            logger.warning("hu_min == hu_max; returning zeros.")
            return np.zeros_like(clipped)
        normalized = (clipped - hu_min) / denom
    else:
        if zscore_mean is None or zscore_std is None:
            raise ValueError(
                "zscore_mean and zscore_std are required for method='zscore'."
            )
        if zscore_std == 0:
            logger.warning("zscore_std == 0; returning zeros.")
            return np.zeros_like(clipped)
        normalized = (clipped - zscore_mean) / zscore_std

    return normalized


# --------------------------------------------------------------------------- #
#  Head-Neck ROI Cropping
# --------------------------------------------------------------------------- #


def compute_body_mask(
    ct_hu: np.ndarray,
    hu_threshold: float = -500.0,
    closing_iterations: int = 2,
) -> np.ndarray:
    """Derive a body mask from a CT volume using HU thresholding only.

    This is the CT-only replacement for the previous GT-lesion-mask-based ROI
    definition (H1).  Because it depends solely on the CT intensities, it is
    reproducible at inference time when no ground-truth mask exists.

    Algorithm:
        1. Threshold: voxels with HU > ``hu_threshold`` are candidate tissue
           (air is ≈ −1000 HU; soft tissue/bone are well above −500 HU).
        2. Morphological closing to bridge small gaps (e.g. skin/air noise).
        3. Keep the single largest 3D connected component (the patient body;
           discards disconnected air pockets and most of the scanner bore).
        4. Fill internal holes (air cavities inside the body — trachea,
           sinuses, oesophagus — so the ROI stays a solid block).

    Args:
        ct_hu: 3D CT array in **raw Hounsfield units** (D, H, W). Must be
            called BEFORE HU clipping/normalisation.
        hu_threshold: HU cutoff separating body/couch from air background.
        closing_iterations: Iterations of binary closing.

    Returns:
        Boolean 3D mask (D, H, W); True inside the body region. If no voxel
        exceeds the threshold (degenerate input), returns an all-True mask so
        the caller falls back to the full volume.
    """
    from scipy import ndimage  # local import: scipy is already a pipeline dep

    if ct_hu.ndim != 3:
        raise ValueError(f"Expected 3D CT array, got {ct_hu.ndim}D.")

    binary = ct_hu > hu_threshold
    if not binary.any():
        logger.warning("Body-mask threshold produced empty mask; using full volume.")
        return np.ones_like(ct_hu, dtype=bool)

    if closing_iterations > 0:
        binary = ndimage.binary_closing(binary, iterations=closing_iterations)

    labeled, n = ndimage.label(binary)
    if n == 0:
        logger.warning("Body-mask found 0 components; using full volume.")
        return np.ones_like(ct_hu, dtype=bool)

    # Largest connected component = patient body
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # ignore background label 0
    largest = int(np.argmax(counts))
    body = labeled == largest

    # Fill internal air cavities so the ROI is a solid block
    body = ndimage.binary_fill_holes(body)
    return body


def _crop_body(
    array: np.ndarray,
    margin: tuple[int, int, int],
    original_shape: tuple[int, ...],
    hu_threshold: float = -500.0,
    z_extent_mm: float = 360.0,
    z_spacing_mm: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Crop to a CT-only head-neck ROI (H1 fix).

    ``array`` must be the CT volume in raw HU (RAS orientation, so the highest
    z-index is the superior/head end). The ROI is derived entirely from the CT:

        1. Body mask via HU threshold (``compute_body_mask``).
        2. In-plane (y, x): tight bounding box of the body + margin — removes
           surrounding air and the scanner couch.
        3. Along z (superior→inferior): keep only the top ``z_extent_mm`` below
           the body-top landmark (top of head). This restricts whole-body
           PET/CT scans to the head-neck region without any GT mask, so the
           crop is reproducible at inference.

    The body-top landmark is stable: across the HECKTOR cohort the top of the
    body is anatomically the skull vertex (validated: narrower cross-section
    than mid-body in 97% of sampled cases), and every GTV sits ≤ ~311 mm below
    it (max observed), so a 360 mm window contains all lesions with margin.

    Args:
        array: CT volume in raw HU (D, H, W); D is the z axis.
        margin: Voxel margin (z, y, x) added around the in-plane body bbox.
        original_shape: Shape of ``array`` (for crop_info).
        hu_threshold: HU cutoff for the body mask.
        z_extent_mm: Superior→inferior window length in millimetres.
        z_spacing_mm: Physical z-spacing (mm/slice) of ``array`` after
            resampling — used to convert ``z_extent_mm`` to a slice count.

    Returns:
        (cropped array, crop_info). ``crop_info['slices']`` are (start, stop)
        index pairs on the ORIGINAL array, so the prediction can be pasted back
        to full CT geometry.
    """
    body = compute_body_mask(array, hu_threshold=hu_threshold)
    coords = np.argwhere(body)
    if coords.size == 0:
        crop_info = {
            "slices": [(0, s) for s in original_shape],
            "original_shape": original_shape,
            "method": "body_empty",
        }
        return array.copy(), crop_info

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    # z (axis 0): head-neck window measured DOWN from the body-top (head).
    # This is a deliberate anatomical cut, not a tight bbox, so no margin is
    # applied to the z bounds — the window size itself carries the safety
    # margin (360 mm vs ~311 mm deepest observed GTV).
    z_top = int(maxs[0])                       # superior-most body slice (head)
    z_body_bottom = int(mins[0])
    n_window = int(round(z_extent_mm / max(z_spacing_mm, 1e-6)))
    z_start = max(0, z_top - n_window)
    z_start = max(z_start, z_body_bottom)      # never extend past the body
    z_end = min(original_shape[0], z_top + 1)

    # y, x (axes 1, 2): tight body bbox + margin.
    yx_slices = []
    for dim in (1, 2):
        start = max(0, int(mins[dim]) - margin[dim])
        end = min(original_shape[dim], int(maxs[dim]) + margin[dim] + 1)
        yx_slices.append(slice(start, end))

    slices = [slice(z_start, z_end), yx_slices[0], yx_slices[1]]
    cropped = array[slices[0], slices[1], slices[2]].copy()
    crop_info = {
        "slices": [(s.start, s.stop) for s in slices],
        "original_shape": original_shape,
        "method": "body",
        "hu_threshold": hu_threshold,
        "z_extent_mm": z_extent_mm,
        "z_spacing_mm": z_spacing_mm,
        "body_z_range": [z_body_bottom, z_top],
    }
    logger.debug(
        "Body crop: %s -> %s (z_window=%dmm=%d slices, margin=%s)",
        original_shape, cropped.shape, int(z_extent_mm), n_window, margin,
    )
    return cropped, crop_info


def crop_head_neck_roi(
    array: np.ndarray,
    mask: Optional[np.ndarray] = None,
    margin: tuple[int, int, int] = (10, 10, 10),
    method: str = "bbox",
    fixed_size: Optional[tuple[int, int, int]] = None,
    hu_threshold: float = -500.0,
    z_extent_mm: float = 360.0,
    z_spacing_mm: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Crop the head-neck region of interest from a 3D volume.

    Supports three methods:
        - 'body': CT-only bounding box around an HU-thresholded body mask
          (see ``compute_body_mask``). Needs NO ground-truth mask and is
          reproducible at inference — this is the H1-fix method and the
          recommended default for the baseline.
        - 'bbox': tight bounding box around mask foreground + margin.
          REQUIRES the GT lesion mask, so it is NOT reproducible at inference.
        - 'fixed_size': center crop of fixed_size, centered on mask centroid.
          Also requires a mask.

    If method is 'bbox'/'fixed_size' and no mask is provided, the volume is
    returned as-is with identity crop info. ``array`` must be in raw HU when
    method='body'.

    Args:
        array: 3D numpy array (D, H, W) — z, y, x ordering.
        mask: Optional binary mask for guiding the crop.
        margin: Voxel margin to add around the bounding box (z, y, x).
        method: Crop method — 'bbox' or 'fixed_size'.
        fixed_size: Required if method='fixed_size'. Output size (D, H, W).

    Returns:
        Tuple of:
            - cropped: Cropped numpy array.
            - crop_info: Dict with keys 'slices', 'original_shape', 'method'
              for reversibility (un-cropping).

    Raises:
        ValueError: If method is unknown, or fixed_size missing.
    """
    if method not in ("bbox", "fixed_size", "body"):
        raise ValueError(
            f"Unknown crop method '{method}'. Use 'bbox', 'fixed_size', or 'body'."
        )

    original_shape = array.shape
    ndim = len(original_shape)
    if ndim != 3:
        raise ValueError(f"Expected 3D array, got {ndim}D.")

    # 'body': CT-only ROI (H1 fix) — derived from the CT itself, needs no mask
    # and is therefore reproducible at inference time.
    if method == "body":
        return _crop_body(
            array, margin, original_shape,
            hu_threshold=hu_threshold,
            z_extent_mm=z_extent_mm,
            z_spacing_mm=z_spacing_mm,
        )

    # 'bbox'/'fixed_size' require a mask. With no mask → return full volume.
    if mask is None:
        crop_info = {
            "slices": [slice(0, s) for s in original_shape],
            "original_shape": original_shape,
            "method": "identity",
        }
        logger.debug("No mask provided; returning full volume.")
        return array.copy(), crop_info

    if method == "bbox":
        return _crop_bbox(array, mask, margin, original_shape)
    else:
        if fixed_size is None:
            raise ValueError("fixed_size is required when method='fixed_size'.")
        return _crop_fixed_size(array, mask, fixed_size, original_shape)


def _crop_bbox(
    array: np.ndarray,
    mask: np.ndarray,
    margin: tuple[int, int, int],
    original_shape: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Crop using bounding box around mask foreground."""
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        logger.warning("Mask is empty; returning full volume.")
        crop_info = {
            "slices": [slice(0, s) for s in original_shape],
            "original_shape": original_shape,
            "method": "bbox_empty",
        }
        return array.copy(), crop_info

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    slices = []
    for dim in range(3):
        start = max(0, mins[dim] - margin[dim])
        end = min(original_shape[dim], maxs[dim] + margin[dim] + 1)
        slices.append(slice(int(start), int(end)))

    cropped = array[slices[0], slices[1], slices[2]].copy()
    crop_info = {
        "slices": [(s.start, s.stop) for s in slices],
        "original_shape": original_shape,
        "method": "bbox",
    }

    logger.debug(
        "BBox crop: %s -> %s (margin=%s)",
        original_shape, cropped.shape, margin,
    )
    return cropped, crop_info


def _crop_fixed_size(
    array: np.ndarray,
    mask: np.ndarray,
    fixed_size: tuple[int, int, int],
    original_shape: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Center crop of fixed size, centered on mask centroid."""
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        centroid = np.array(original_shape) // 2
        logger.warning("Mask is empty; centering crop on volume center.")
    else:
        centroid = coords.mean(axis=0).astype(int)

    slices = []
    for dim in range(3):
        half = fixed_size[dim] // 2
        start = max(0, centroid[dim] - half)
        end = start + fixed_size[dim]
        # Shift if exceeding bounds
        if end > original_shape[dim]:
            end = original_shape[dim]
            start = max(0, end - fixed_size[dim])
        slices.append(slice(int(start), int(end)))

    cropped = array[slices[0], slices[1], slices[2]].copy()
    crop_info = {
        "slices": [(s.start, s.stop) for s in slices],
        "original_shape": original_shape,
        "method": "fixed_size",
        "fixed_size": fixed_size,
    }

    logger.debug(
        "Fixed-size crop: %s -> %s (center=%s)",
        original_shape, cropped.shape, centroid.tolist(),
    )
    return cropped, crop_info


# --------------------------------------------------------------------------- #
#  Volume Alignment
# --------------------------------------------------------------------------- #


def align_volumes(
    ct_sitk: sitk.Image,
    pet_sitk: Optional[sitk.Image] = None,
    mask_sitk: Optional[sitk.Image] = None,
) -> tuple[sitk.Image, Optional[sitk.Image], Optional[sitk.Image]]:
    """Resample PET and mask into CT physical space for voxel-wise alignment.

    CT is the reference frame. PET is resampled with linear interpolation,
    mask with nearest-neighbor.

    Args:
        ct_sitk: Reference CT SimpleITK image.
        pet_sitk: Optional PET SimpleITK image to align.
        mask_sitk: Optional mask SimpleITK image to align.

    Returns:
        Tuple of (ct_sitk, aligned_pet_sitk, aligned_mask_sitk).
        Items that were None remain None.
    """
    aligned_pet = None
    aligned_mask = None

    if pet_sitk is not None:
        aligned_pet = sitk.Resample(
            pet_sitk,
            ct_sitk,
            sitk.Transform(),
            sitk.sitkLinear,
            0.0,
            pet_sitk.GetPixelID(),
        )
        logger.debug(
            "Aligned PET to CT: size %s -> %s",
            pet_sitk.GetSize(), aligned_pet.GetSize(),
        )

    if mask_sitk is not None:
        aligned_mask = sitk.Resample(
            mask_sitk,
            ct_sitk,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            mask_sitk.GetPixelID(),
        )
        logger.debug(
            "Aligned mask to CT: size %s -> %s",
            mask_sitk.GetSize(), aligned_mask.GetSize(),
        )

    return ct_sitk, aligned_pet, aligned_mask


# --------------------------------------------------------------------------- #
#  Full Preprocessing Pipeline for One Case
# --------------------------------------------------------------------------- #


def preprocess_case(
    ct_path: str | Path,
    mask_path: Optional[str | Path],
    pet_path: Optional[str | Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Full preprocessing pipeline for a single case.

    Pipeline steps:
        1. Load CT (+ optional PET, mask) as SimpleITK images.
        2. Standardize orientation to target (e.g. RAS).
        3. Align PET and mask to CT physical space.
        4. Resample all volumes to target spacing.
        5. Convert to numpy arrays.
        6. Crop head-neck ROI.
        7. Clip HU and normalize CT.

    Args:
        ct_path: Path to CT NIfTI file.
        mask_path: Optional path to mask NIfTI file.
        pet_path: Optional path to PET NIfTI file.
        config: Preprocessing config dict (from preprocess.yaml['preprocessing']).

    Returns:
        Dictionary with keys:
            - 'ct_array': preprocessed CT numpy array (float32)
            - 'pet_array': preprocessed PET numpy array or None
            - 'mask_array': preprocessed mask numpy array or None
            - 'ct_meta': metadata dict (spacing, origin, direction after resampling)
            - 'crop_info': crop information for reversibility
    """
    from src.dataio.readers import load_nifti

    prep = config
    target_orientation = prep.get("target_orientation", "RAS")
    target_spacing = tuple(prep.get("target_spacing", [1.0, 1.0, 1.0]))
    ct_interp = prep.get("ct_interpolation", "linear")
    pet_interp = prep.get("pet_interpolation", "linear")
    mask_interp = prep.get("mask_interpolation", "nearest")
    hu_min = prep.get("hu_min", -1024)
    hu_max = prep.get("hu_max", 3071)
    norm_method = prep.get("normalize_method", "minmax")
    crop_cfg = prep.get("crop", {})
    crop_method = crop_cfg.get("method", "bbox")
    crop_margin = tuple(crop_cfg.get("margin", [10, 10, 10]))
    crop_fixed_size = tuple(crop_cfg.get("fixed_size", [192, 192, 192]))
    crop_hu_threshold = float(crop_cfg.get("hu_threshold", -500.0))
    crop_z_extent_mm = float(crop_cfg.get("z_extent_mm", 360.0))

    # Step 1: Load
    logger.info("Preprocessing case: %s", Path(ct_path).name)
    _, ct_meta = load_nifti(ct_path)
    ct_sitk = ct_meta["sitk_image"]

    pet_sitk = None
    mask_sitk = None

    if pet_path is not None and Path(pet_path).exists():
        _, pet_meta_raw = load_nifti(pet_path)
        pet_sitk = pet_meta_raw["sitk_image"]

    if mask_path is not None and Path(mask_path).exists():
        _, mask_meta_raw = load_nifti(mask_path)
        mask_sitk = mask_meta_raw["sitk_image"]

    # Step 2: Standardize orientation
    ct_sitk = standardize_orientation(ct_sitk, target_orientation)
    if pet_sitk is not None:
        pet_sitk = standardize_orientation(pet_sitk, target_orientation)
    if mask_sitk is not None:
        mask_sitk = standardize_orientation(mask_sitk, target_orientation)

    # Step 3: Align PET & mask to CT space
    ct_sitk, pet_sitk, mask_sitk = align_volumes(ct_sitk, pet_sitk, mask_sitk)

    # Step 4: Resample to target spacing
    ct_sitk = resample_volume(ct_sitk, target_spacing, ct_interp)
    if pet_sitk is not None:
        pet_sitk = resample_volume(pet_sitk, target_spacing, pet_interp)
    if mask_sitk is not None:
        mask_sitk = resample_volume(mask_sitk, target_spacing, mask_interp)

    # Step 5: Convert to numpy
    ct_array = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)
    pet_array = (
        sitk.GetArrayFromImage(pet_sitk).astype(np.float32)
        if pet_sitk is not None else None
    )
    mask_array = (
        sitk.GetArrayFromImage(mask_sitk).astype(np.uint8)
        if mask_sitk is not None else None
    )

    # Step 6: Crop head-neck ROI
    ct_cropped, crop_info = crop_head_neck_roi(
        ct_array,
        mask=mask_array,
        margin=crop_margin,
        method=crop_method,
        fixed_size=crop_fixed_size if crop_method == "fixed_size" else None,
        hu_threshold=crop_hu_threshold,
        z_extent_mm=crop_z_extent_mm,
        z_spacing_mm=float(target_spacing[2]),
    )

    # Apply same crop to PET and mask
    slices_tuples = crop_info["slices"]
    if isinstance(slices_tuples[0], tuple):
        slices = [slice(s[0], s[1]) for s in slices_tuples]
    else:
        slices = slices_tuples

    if pet_array is not None:
        pet_cropped = pet_array[slices[0], slices[1], slices[2]].copy()
    else:
        pet_cropped = None

    if mask_array is not None:
        mask_cropped = mask_array[slices[0], slices[1], slices[2]].copy()
        # Safety check (prep-time only; mask unavailable at inference): the
        # CT-only crop must not clip the GT lesion. If it does, the z-window is
        # too small for this case — warn loudly instead of silently training on
        # a truncated target.
        gtv_before = int((mask_array > 0).sum())
        gtv_after = int((mask_cropped > 0).sum())
        if gtv_before > 0:
            retained = gtv_after / gtv_before
            crop_info["gtv_retained"] = retained
            if retained < 0.999:
                logger.warning(
                    "CROP CLIPPED GTV for %s: %.2f%% of lesion retained "
                    "(%d/%d voxels). Crop method=%s z_extent_mm=%s — consider "
                    "increasing z_extent_mm.",
                    Path(ct_path).name, 100.0 * retained,
                    gtv_after, gtv_before, crop_method, crop_z_extent_mm,
                )
    else:
        mask_cropped = None

    # Step 7: Clip HU and normalize CT
    ct_normalized = clip_and_normalize_hu(
        ct_cropped,
        hu_min=hu_min,
        hu_max=hu_max,
        method=norm_method,
        zscore_mean=prep.get("zscore_mean"),
        zscore_std=prep.get("zscore_std"),
    )

    # Build output metadata
    result_meta = {
        "spacing": ct_sitk.GetSpacing(),
        "origin": ct_sitk.GetOrigin(),
        "direction": ct_sitk.GetDirection(),
    }

    logger.info(
        "  Done: CT %s | PET %s | Mask %s",
        ct_normalized.shape,
        pet_cropped.shape if pet_cropped is not None else "N/A",
        mask_cropped.shape if mask_cropped is not None else "N/A",
    )

    return {
        "ct_array": ct_normalized,
        "pet_array": pet_cropped,
        "mask_array": mask_cropped,
        "ct_meta": result_meta,
        "crop_info": crop_info,
    }
