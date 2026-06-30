# -*- coding: utf-8 -*-
"""Medical image readers and writers for CT2MAP-HN.

Hỗ trợ đọc/ghi NIfTI (.nii, .nii.gz) và DICOM series sử dụng SimpleITK.
Tất cả hàm trả về numpy array kèm metadata (spacing, origin, direction)
để bảo toàn thông tin không gian.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def load_nifti(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a NIfTI file and return the image array with spatial metadata.

    Args:
        path: Path to the NIfTI file (.nii or .nii.gz).

    Returns:
        Tuple of (array, metadata) where:
            - array: numpy ndarray with shape (D, H, W) in (z, y, x) order.
            - metadata: dict with keys 'spacing', 'origin', 'direction',
              'size', 'dtype', 'sitk_image' (the original SimpleITK image).

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If SimpleITK fails to read the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    try:
        sitk_image = sitk.ReadImage(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to read NIfTI file {path}: {e}") from e

    array = sitk.GetArrayFromImage(sitk_image)  # (z, y, x) ordering

    metadata = {
        "spacing": sitk_image.GetSpacing(),       # (x, y, z)
        "origin": sitk_image.GetOrigin(),          # (x, y, z)
        "direction": sitk_image.GetDirection(),    # 9-element tuple
        "size": sitk_image.GetSize(),              # (x, y, z)
        "dtype": array.dtype,
        "sitk_image": sitk_image,
        "source_path": str(path),
    }

    logger.debug(
        "Loaded NIfTI: %s | shape=%s | spacing=%s",
        path.name, array.shape, metadata["spacing"],
    )
    return array, metadata


def load_dicom_series(
    directory: str | Path,
    series_id: Optional[str] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a DICOM series from a directory.

    Args:
        directory: Directory containing DICOM files.
        series_id: Optional specific series UID to load.
            If None, the first series found is used.

    Returns:
        Tuple of (array, metadata) — same format as ``load_nifti``.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no DICOM series is found in the directory.
        RuntimeError: If SimpleITK fails to read the series.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {directory}")

    reader = sitk.ImageSeriesReader()

    if series_id is None:
        series_ids = reader.GetGDCMSeriesIDs(str(directory))
        if not series_ids:
            raise ValueError(f"No DICOM series found in {directory}")
        series_id = series_ids[0]
        if len(series_ids) > 1:
            logger.warning(
                "Multiple DICOM series in %s. Using first: %s",
                directory, series_id,
            )

    dicom_names = reader.GetGDCMSeriesFileNames(str(directory), series_id)
    if not dicom_names:
        raise ValueError(
            f"No DICOM files for series {series_id} in {directory}"
        )

    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    try:
        sitk_image = reader.Execute()
    except Exception as e:
        raise RuntimeError(
            f"Failed to read DICOM series from {directory}: {e}"
        ) from e

    array = sitk.GetArrayFromImage(sitk_image)

    metadata = {
        "spacing": sitk_image.GetSpacing(),
        "origin": sitk_image.GetOrigin(),
        "direction": sitk_image.GetDirection(),
        "size": sitk_image.GetSize(),
        "dtype": array.dtype,
        "sitk_image": sitk_image,
        "source_path": str(directory),
        "series_id": series_id,
    }

    logger.debug(
        "Loaded DICOM series: %s | shape=%s | spacing=%s",
        directory.name, array.shape, metadata["spacing"],
    )
    return array, metadata


def save_nifti(
    array: np.ndarray,
    metadata: dict[str, Any],
    path: str | Path,
    *,
    use_compression: bool = True,
) -> Path:
    """Save a numpy array as a NIfTI file, preserving spatial metadata.

    Args:
        array: numpy ndarray with shape (D, H, W) in (z, y, x) order.
        metadata: dict with 'spacing', 'origin', 'direction' keys.
        path: Output file path (.nii or .nii.gz).
        use_compression: Whether to use gzip compression (default True).

    Returns:
        The Path to the saved file.

    Raises:
        ValueError: If metadata is missing required keys.
        RuntimeError: If writing fails.
    """
    path = Path(path)
    required_keys = {"spacing", "origin", "direction"}
    missing = required_keys - set(metadata.keys())
    if missing:
        raise ValueError(f"Metadata missing required keys: {missing}")

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    sitk_image = sitk.GetImageFromArray(array)
    sitk_image.SetSpacing(metadata["spacing"])
    sitk_image.SetOrigin(metadata["origin"])
    sitk_image.SetDirection(metadata["direction"])

    try:
        sitk.WriteImage(sitk_image, str(path), use_compression)
    except Exception as e:
        raise RuntimeError(f"Failed to write NIfTI to {path}: {e}") from e

    logger.debug("Saved NIfTI: %s | shape=%s", path.name, array.shape)
    return path


def load_case(
    ct_path: str | Path,
    pet_path: Optional[str | Path] = None,
    mask_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Load a complete case (CT + optional PET + optional mask).

    Args:
        ct_path: Path to CT NIfTI file.
        pet_path: Optional path to PET NIfTI file.
        mask_path: Optional path to lesion mask NIfTI file.

    Returns:
        Dictionary with keys:
            - 'ct_array': np.ndarray (z, y, x)
            - 'ct_meta': dict of CT metadata
            - 'pet_array': np.ndarray or None
            - 'pet_meta': dict or None
            - 'mask_array': np.ndarray or None
            - 'mask_meta': dict or None

    Raises:
        FileNotFoundError: If CT file is missing (PET/mask are optional).
    """
    ct_path = Path(ct_path)
    logger.info("Loading case: CT=%s", ct_path.name)

    ct_array, ct_meta = load_nifti(ct_path)

    case = {
        "ct_array": ct_array,
        "ct_meta": ct_meta,
        "pet_array": None,
        "pet_meta": None,
        "mask_array": None,
        "mask_meta": None,
    }

    if pet_path is not None:
        pet_path = Path(pet_path)
        if pet_path.exists():
            pet_array, pet_meta = load_nifti(pet_path)
            case["pet_array"] = pet_array
            case["pet_meta"] = pet_meta
            logger.debug("  PET loaded: shape=%s", pet_array.shape)
        else:
            logger.warning("PET file not found, skipping: %s", pet_path)

    if mask_path is not None:
        mask_path = Path(mask_path)
        if mask_path.exists():
            mask_array, mask_meta = load_nifti(mask_path)
            case["mask_array"] = mask_array
            case["mask_meta"] = mask_meta
            logger.debug("  Mask loaded: shape=%s", mask_array.shape)
        else:
            logger.warning("Mask file not found, skipping: %s", mask_path)

    return case
