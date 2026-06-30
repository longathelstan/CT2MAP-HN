# -*- coding: utf-8 -*-
"""MONAI transforms for CT2MAP-HN training and inference.

Định nghĩa các pipeline augmentation/transform cho train, val, và inference.
Sử dụng MONAI transforms API (dictionary-based) để tương thích với
Dataset trả về dict.
"""

import logging
from typing import Any

from monai.transforms import (
    Compose,
    EnsureTyped,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
)

logger = logging.getLogger(__name__)

# Keys used across all transforms
IMAGE_KEY = "image"
TARGET_KEY = "heatmap_target"
MASK_KEY = "lesion_mask"
ALL_KEYS = [IMAGE_KEY, TARGET_KEY, MASK_KEY]
SPATIAL_KEYS = [IMAGE_KEY, TARGET_KEY, MASK_KEY]


def get_train_transforms(config: dict[str, Any]) -> Compose:
    """Build the training transform pipeline with data augmentation.

    Augmentations include:
        - Spatial padding (to ensure minimum size for cropping)
        - Random crop by positive/negative label ratio
        - Random axis flips
        - Random affine (rotation, scaling, translation)
        - Random intensity shift and scale
        - Random Gaussian noise

    Args:
        config: Transform config dict (from preprocess.yaml['transforms']['train']).

    Returns:
        MONAI Compose transform for training.
    """
    tcfg = config
    spatial_size = tcfg.get("spatial_size", [128, 128, 128])
    num_samples = tcfg.get("num_samples", 2)
    pos_ratio = tcfg.get("pos_ratio", 0.7)
    flip_prob = tcfg.get("flip_prob", 0.5)
    flip_axes = tcfg.get("flip_axes", [0, 1, 2])
    affine_prob = tcfg.get("affine_prob", 0.3)
    rotate_range = tcfg.get("rotate_range", [0.26, 0.26, 0.26])
    scale_range = tcfg.get("scale_range", [0.1, 0.1, 0.1])
    translate_range = tcfg.get("translate_range", [10, 10, 10])
    intensity_shift = tcfg.get("intensity_shift_range", 0.1)
    intensity_scale = tcfg.get("intensity_scale_range", 0.1)
    noise_prob = tcfg.get("gaussian_noise_prob", 0.2)
    noise_std = tcfg.get("gaussian_noise_std", 0.05)

    transforms = [
        # Ensure minimum spatial size for cropping
        SpatialPadd(
            keys=SPATIAL_KEYS,
            spatial_size=spatial_size,
            mode="constant",
        ),
        # Random crop guided by label map (positive region sampling)
        RandCropByPosNegLabeld(
            keys=SPATIAL_KEYS,
            label_key=MASK_KEY,
            spatial_size=spatial_size,
            pos=pos_ratio,
            neg=1.0 - pos_ratio,
            num_samples=num_samples,
            allow_smaller=False,
        ),
    ]

    # Random flips per axis
    for axis in flip_axes:
        transforms.append(
            RandFlipd(
                keys=SPATIAL_KEYS,
                prob=flip_prob,
                spatial_axis=axis,
            )
        )

    # Random affine (rotation + scale + translate)
    transforms.append(
        RandAffined(
            keys=SPATIAL_KEYS,
            prob=affine_prob,
            rotate_range=rotate_range,
            scale_range=scale_range,
            translate_range=translate_range,
            mode=["bilinear", "bilinear", "nearest"],
            padding_mode="zeros",
        )
    )

    # Intensity augmentation (CT only)
    transforms.extend([
        RandShiftIntensityd(
            keys=[IMAGE_KEY],
            offsets=intensity_shift,
            prob=0.5,
        ),
        RandScaleIntensityd(
            keys=[IMAGE_KEY],
            factors=intensity_scale,
            prob=0.5,
        ),
        RandGaussianNoised(
            keys=[IMAGE_KEY],
            prob=noise_prob,
            std=noise_std,
        ),
    ])

    # Ensure PyTorch tensor output
    transforms.append(
        EnsureTyped(keys=ALL_KEYS, dtype="float32"),
    )

    logger.debug(
        "Train transforms: spatial_size=%s, num_samples=%d, "
        "flip=%.1f, affine=%.1f, noise=%.1f",
        spatial_size, num_samples, flip_prob, affine_prob, noise_prob,
    )
    return Compose(transforms)


def get_val_transforms(config: dict[str, Any]) -> Compose:
    """Build the validation transform pipeline (deterministic only).

    Only applies spatial padding (no augmentation) to ensure consistent
    sizes, then converts to tensors.

    Args:
        config: Transform config dict (from preprocess.yaml['transforms']['val']).

    Returns:
        MONAI Compose transform for validation.
    """
    vcfg = config
    spatial_size = vcfg.get("spatial_size", [128, 128, 128])

    transforms = [
        SpatialPadd(
            keys=SPATIAL_KEYS,
            spatial_size=spatial_size,
            mode="constant",
        ),
        EnsureTyped(keys=ALL_KEYS, dtype="float32"),
    ]

    logger.debug("Val transforms: spatial_size=%s", spatial_size)
    return Compose(transforms)


def get_inference_transforms(config: dict[str, Any]) -> Compose:
    """Build the inference transform pipeline.

    Minimal transforms: just type conversion. Sliding-window inference
    parameters (roi_size, overlap, etc.) are handled by the inference
    engine, not by transforms.

    Args:
        config: Transform config dict
            (from preprocess.yaml['transforms']['inference']).

    Returns:
        MONAI Compose transform for inference.
    """
    transforms = [
        EnsureTyped(keys=[IMAGE_KEY], dtype="float32"),
    ]

    logger.debug(
        "Inference transforms: roi_size=%s, overlap=%.2f",
        config.get("roi_size", [128, 128, 128]),
        config.get("overlap", 0.5),
    )
    return Compose(transforms)
