#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create train/val/test splits stratified by clinical site.

Usage:
    python scripts/create_splits.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.dataio.splits import create_splits, save_splits, validate_no_leak

logger = setup_logger("create_splits", level="INFO")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test data splits.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data.yaml",
        help="Path to data YAML config.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    cfg = load_config(str(config_path))
    data_cfg = cfg.get("data", {})

    manifest_path = Path(data_cfg.get("manifest_path", "data/processed/manifests/manifest.csv"))
    split_file = Path(data_cfg.get("split_file", "data/processed/manifests/splits.json"))
    train_ratio = float(data_cfg.get("train_ratio", 0.70))
    val_ratio = float(data_cfg.get("val_ratio", 0.10))
    test_ratio = float(data_cfg.get("test_ratio", 0.20))
    seed = int(data_cfg.get("seed", 42))

    if not manifest_path.exists():
        logger.error("Manifest file not found: %s. Run build_manifest.py first.", manifest_path)
        sys.exit(1)

    # Read manifest
    df = pd.read_csv(manifest_path)
    logger.info("Loaded manifest from %s with %d cases", manifest_path, len(df))

    # Create splits stratified by site
    splits = create_splits(
        manifest_df=df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratify_col="site",
        case_id_col="case_id",
    )

    # Validate leakage
    validate_no_leak(splits)

    # Save to JSON
    save_splits(splits, split_file)

    # Update manifest split column
    df["split"] = ""
    for split_name, case_ids in splits.items():
        df.loc[df["case_id"].isin(case_ids), "split"] = split_name

    df.to_csv(manifest_path, index=False)
    logger.info("Updated manifest with split assignments and saved to %s", manifest_path)


if __name__ == "__main__":
    main()
