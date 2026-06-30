# -*- coding: utf-8 -*-
"""Dataset classes package for CT2MAP-HN.

Cung cấp PyTorch Dataset classes và MONAI transforms cho training/inference.

Classes:
    CTHeatmapDataset: Dataset chính cho student model (CT → heatmap)
    CTTeacherDataset: Dataset cho teacher model (CT + PET → heatmap)

Functions:
    get_train_transforms: Augmentation pipeline cho training
    get_val_transforms: Transform pipeline cho validation
    get_inference_transforms: Transform pipeline cho inference
"""

from src.datasets.ct_dataset import CTHeatmapDataset
from src.datasets.ct_teacher_dataset import CTTeacherDataset
from src.datasets.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_inference_transforms,
)

__all__ = [
    "CTHeatmapDataset",
    "CTTeacherDataset",
    "get_train_transforms",
    "get_val_transforms",
    "get_inference_transforms",
]
