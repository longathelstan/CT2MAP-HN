# -*- coding: utf-8 -*-
"""Data splitting utilities for CT2MAP-HN.

Chia dữ liệu train/val/test với hỗ trợ stratification, lưu/đọc JSON,
và kiểm tra data leakage giữa các split.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def create_splits(
    manifest_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_col: Optional[str] = None,
    case_id_col: str = "case_id",
) -> dict[str, list[str]]:
    """Split cases into train / val / test sets.

    Uses sklearn's ``train_test_split`` with optional stratification.
    The split is done at the case (patient) level to prevent data leakage.

    Args:
        manifest_df: DataFrame with at least a ``case_id`` column.
        train_ratio: Fraction of data for training (0-1).
        val_ratio: Fraction of data for validation (0-1).
        test_ratio: Fraction of data for testing (0-1).
        seed: Random seed for reproducibility.
        stratify_col: Optional column name for stratified splitting
            (e.g. 'site', 'has_primary').
        case_id_col: Name of the case ID column.

    Returns:
        Dictionary ``{'train': [...], 'val': [...], 'test': [...]}`` of
        case ID lists.

    Raises:
        ValueError: If ratios don't sum to ~1.0, or columns are missing.
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.6f} "
            f"({train_ratio} + {val_ratio} + {test_ratio})."
        )

    if case_id_col not in manifest_df.columns:
        raise ValueError(f"Column '{case_id_col}' not found in manifest.")

    case_ids = manifest_df[case_id_col].unique().tolist()
    n_cases = len(case_ids)
    logger.info("Splitting %d cases: train=%.0f%% val=%.0f%% test=%.0f%%",
                n_cases, train_ratio * 100, val_ratio * 100, test_ratio * 100)

    # Stratify labels (per unique case_id)
    stratify_labels = None
    if stratify_col is not None:
        if stratify_col not in manifest_df.columns:
            logger.warning(
                "Stratify column '%s' not found; falling back to random split.",
                stratify_col,
            )
        else:
            # Get one label per case_id
            case_df = manifest_df.drop_duplicates(subset=case_id_col)
            case_df = case_df.set_index(case_id_col)
            stratify_labels = [
                case_df.loc[cid, stratify_col] for cid in case_ids
            ]

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_ids, val_test_ids = train_test_split(
        case_ids,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=stratify_labels,
    )

    # Second split: val vs test
    if stratify_labels is not None:
        case_df = manifest_df.drop_duplicates(subset=case_id_col)
        case_df = case_df.set_index(case_id_col)
        vt_stratify = [case_df.loc[cid, stratify_col] for cid in val_test_ids]
    else:
        vt_stratify = None

    relative_test = test_ratio / val_test_ratio
    val_ids, test_ids = train_test_split(
        val_test_ids,
        test_size=relative_test,
        random_state=seed,
        stratify=vt_stratify,
    )

    splits = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }

    logger.info(
        "Split result: train=%d, val=%d, test=%d",
        len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )
    return splits


def save_splits(
    splits_dict: dict[str, list[str]],
    path: str | Path,
) -> Path:
    """Save splits dictionary to a JSON file.

    Args:
        splits_dict: Splits dictionary from ``create_splits``.
        path: Output JSON file path.

    Returns:
        Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits_dict, f, indent=2, ensure_ascii=False)

    logger.info("Saved splits to %s", path)
    return path


def load_splits(path: str | Path) -> dict[str, list[str]]:
    """Load splits dictionary from a JSON file.

    Args:
        path: Path to the splits JSON file.

    Returns:
        Splits dictionary ``{'train': [...], 'val': [...], 'test': [...]}``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON structure is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Splits file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    required_keys = {"train", "val", "test"}
    missing = required_keys - set(splits.keys())
    if missing:
        raise ValueError(f"Splits JSON missing required keys: {missing}")

    for key in required_keys:
        if not isinstance(splits[key], list):
            raise ValueError(f"splits['{key}'] must be a list, got {type(splits[key])}")

    logger.info(
        "Loaded splits from %s: train=%d, val=%d, test=%d",
        path, len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )
    return splits


def validate_no_leak(splits_dict: dict[str, list[str]]) -> bool:
    """Validate that there is no patient/case ID overlap between splits.

    Args:
        splits_dict: Splits dictionary.

    Returns:
        True if no leakage is detected.

    Raises:
        ValueError: If data leakage (overlap) is found.
    """
    train_set = set(splits_dict.get("train", []))
    val_set = set(splits_dict.get("val", []))
    test_set = set(splits_dict.get("test", []))

    train_val = train_set & val_set
    train_test = train_set & test_set
    val_test = val_set & test_set

    leaks = []
    if train_val:
        leaks.append(f"train∩val: {train_val}")
    if train_test:
        leaks.append(f"train∩test: {train_test}")
    if val_test:
        leaks.append(f"val∩test: {val_test}")

    if leaks:
        msg = f"DATA LEAKAGE detected! Overlapping cases: {'; '.join(leaks)}"
        logger.error(msg)
        raise ValueError(msg)

    total = len(train_set) + len(val_set) + len(test_set)
    logger.info("No data leakage detected. Total unique cases: %d", total)
    return True
