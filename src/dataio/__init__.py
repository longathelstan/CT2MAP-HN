# -*- coding: utf-8 -*-
"""Data I/O package for CT2MAP-HN.

Cung cấp các module đọc/ghi ảnh y tế, tiền xử lý, tạo target, và chia dữ liệu.

Modules:
    readers: Đọc/ghi NIfTI, DICOM
    preprocess: Tiền xử lý ảnh CT (resampling, crop, normalize)
    target_builder: Tạo target heatmap từ mask và PET
    splits: Chia dữ liệu train/val/test
"""

from src.dataio.readers import load_nifti, save_nifti, load_dicom_series, load_case
from src.dataio.preprocess import (
    standardize_orientation,
    resample_volume,
    clip_and_normalize_hu,
    crop_head_neck_roi,
    preprocess_case,
    align_volumes,
)
from src.dataio.target_builder import (
    create_binary_target,
    create_gaussian_heatmap,
    create_pet_derived_target,
    create_case_label,
    build_targets,
)
from src.dataio.splits import (
    create_splits,
    save_splits,
    load_splits,
    validate_no_leak,
)

__all__ = [
    # readers
    "load_nifti",
    "save_nifti",
    "load_dicom_series",
    "load_case",
    # preprocess
    "standardize_orientation",
    "resample_volume",
    "clip_and_normalize_hu",
    "crop_head_neck_roi",
    "preprocess_case",
    "align_volumes",
    # target_builder
    "create_binary_target",
    "create_gaussian_heatmap",
    "create_pet_derived_target",
    "create_case_label",
    "build_targets",
    # splits
    "create_splits",
    "save_splits",
    "load_splits",
    "validate_no_leak",
]
