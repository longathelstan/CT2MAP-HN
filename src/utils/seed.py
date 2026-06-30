# -*- coding: utf-8 -*-
"""Reproducibility utilities for seeding all random number generators.

This module provides a single entry point ``set_seed`` that deterministically
seeds every source of randomness used throughout the CT2MAP-HN pipeline
(stdlib random, NumPy, PyTorch CPU/CUDA, cuDNN, and Python hash seed).

Example:
    >>> from src.utils.seed import set_seed
    >>> set_seed(42)
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all random number generators for reproducibility.

    Sets seeds for:
        - ``random`` (Python stdlib)
        - ``numpy.random``
        - ``torch`` (CPU)
        - ``torch.cuda`` (all GPUs)
        - ``torch.backends.cudnn`` (deterministic mode + benchmark off)
        - ``os.environ['PYTHONHASHSEED']``

    Args:
        seed: Integer seed value. Must be non-negative.
        deterministic: If ``True`` (default), sets cuDNN to deterministic
            mode and disables benchmark. This can slightly reduce
            performance but guarantees reproducibility.

    Raises:
        ValueError: If *seed* is negative.

    Example:
        >>> set_seed(42)
        >>> set_seed(0, deterministic=False)  # faster, less reproducible
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    # Python stdlib
    random.seed(seed)

    # Python hash seed (must be string)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN determinism
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    logger.info(
        "Random seed set to %d (deterministic=%s)", seed, deterministic
    )
