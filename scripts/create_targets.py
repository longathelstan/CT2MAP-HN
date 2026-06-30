#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create training targets (heatmaps + case-level labels) from preprocessed data.

Tạo target cho huấn luyện từ mask và PET đã tiền xử lý:
  - Binary target (0/1 lesion mask)
  - Gaussian-smoothed heatmap
  - PET-derived soft target (for knowledge distillation)
  - Case-level labels (has_active_lesion, high_risk)

Usage:
    python scripts/create_targets.py --config configs/data.yaml
    python scripts/create_targets.py --config configs/data.yaml --target-types binary gaussian_heatmap pet_derived
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataio.target_builder import (  # noqa: E402
    build_targets,
    create_binary_target,
    create_case_label,
    create_gaussian_heatmap,
    create_pet_derived_target,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

logger = setup_logger("create_targets", level="INFO")


# ======================================================================== #
# Single-case worker                                                       #
# ======================================================================== #


def _create_targets_for_case(
    case_id: str,
    processed_dir: Path,
    target_dir: Path,
    target_config: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    """Create all target types for a single case.

    Args:
        case_id: Case identifier.
        processed_dir: Directory with preprocessed .npy arrays.
        target_dir: Output directory for target arrays.
        target_config: Target generation config dict.
        overwrite: Whether to overwrite existing target files.

    Returns:
        Status dict with case_id, status, message.
    """
    # Check for existing targets (resume support)
    label_out = target_dir / f"{case_id}_label.json"
    if label_out.exists() and not overwrite:
        return {"case_id": case_id, "status": "skipped", "message": "already exists"}

    # Load preprocessed mask
    mask_path = processed_dir / f"{case_id}_mask.npy"
    if not mask_path.exists():
        return {
            "case_id": case_id,
            "status": "error",
            "message": f"Mask not found: {mask_path}",
        }

    try:
        mask_array = np.load(str(mask_path))
    except Exception as exc:
        return {
            "case_id": case_id,
            "status": "error",
            "message": f"Cannot load mask: {exc}",
        }

    # Load preprocessed PET (optional)
    pet_path = processed_dir / f"{case_id}_pet.npy"
    pet_array = None
    if pet_path.exists():
        try:
            pet_array = np.load(str(pet_path))
        except Exception as exc:
            logger.warning("Cannot load PET for %s: %s", case_id, exc)

    # Build case dict for target builder
    case_dict = {
        "mask_array": mask_array,
        "pet_array": pet_array,
    }

    # Build targets using the unified pipeline
    try:
        targets = build_targets(case_dict, target_config)
    except Exception as exc:
        tb = traceback.format_exc()
        return {
            "case_id": case_id,
            "status": "error",
            "message": f"build_targets failed: {exc}\n{tb}",
        }

    # Save target arrays
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_types: list[str] = []

    if targets["binary_target"] is not None:
        np.save(str(target_dir / f"{case_id}_binary.npy"), targets["binary_target"])
        saved_types.append("binary")

    if targets["gaussian_heatmap"] is not None:
        np.save(
            str(target_dir / f"{case_id}_gaussian_heatmap.npy"),
            targets["gaussian_heatmap"],
        )
        saved_types.append("gaussian_heatmap")

    if targets["pet_derived_target"] is not None:
        np.save(
            str(target_dir / f"{case_id}_pet_derived.npy"),
            targets["pet_derived_target"],
        )
        saved_types.append("pet_derived")

    # Save case-level label
    case_label = targets.get("case_label", {})
    _save_case_label(case_label, target_dir / f"{case_id}_label.json")
    saved_types.append("label")

    return {
        "case_id": case_id,
        "status": "ok",
        "message": f"saved: {', '.join(saved_types)}",
    }


def _save_case_label(label: dict[str, Any], path: Path) -> None:
    """Save case-level label dict to JSON.

    Args:
        label: Label dictionary.
        path: Output JSON path.
    """
    # Convert numpy types for JSON
    clean = {}
    for k, v in label.items():
        if isinstance(v, (np.bool_, np.integer)):
            clean[k] = int(v)
        elif isinstance(v, np.floating):
            clean[k] = float(v)
        elif isinstance(v, bool):
            clean[k] = v
        else:
            clean[k] = v

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)


# ======================================================================== #
# Worker wrapper for multiprocessing                                        #
# ======================================================================== #


def _worker_wrapper(args: tuple) -> dict[str, Any]:
    """Unpack and call _create_targets_for_case.

    Args:
        args: Tuple of (case_id, processed_dir, target_dir, target_config, overwrite).

    Returns:
        Status dict.
    """
    case_id, processed_dir, target_dir, target_config, overwrite = args
    return _create_targets_for_case(
        case_id, processed_dir, target_dir, target_config, overwrite
    )


# ======================================================================== #
# Main pipeline                                                            #
# ======================================================================== #


def create_all_targets(
    manifest_path: str | Path,
    processed_dir: str | Path,
    target_dir: str | Path,
    target_config: dict[str, Any],
    *,
    num_workers: int = 4,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Create targets for all cases in the manifest.

    Args:
        manifest_path: Path to manifest CSV.
        processed_dir: Directory with preprocessed .npy arrays.
        target_dir: Output directory for targets.
        target_config: Target generation config dict.
        num_workers: Number of parallel workers.
        overwrite: Whether to overwrite existing files.

    Returns:
        DataFrame with processing status per case.
    """
    manifest_path = Path(manifest_path)
    processed_dir = Path(processed_dir)
    target_dir = Path(target_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    df = pd.read_csv(manifest_path)
    case_ids = df["case_id"].unique().tolist()
    n_cases = len(case_ids)

    logger.info("Creating targets for %d cases", n_cases)
    logger.info("  Processed dir: %s", processed_dir)
    logger.info("  Target dir   : %s", target_dir)
    logger.info("  Target types : %s", target_config.get("types", []))
    logger.info("  Workers      : %d", num_workers)

    target_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments
    task_args = [
        (cid, processed_dir, target_dir, target_config, overwrite)
        for cid in case_ids
    ]

    results: list[dict[str, Any]] = []

    if num_workers <= 1:
        try:
            from tqdm import tqdm
            iterator = tqdm(task_args, desc="Creating targets", unit="case")
        except ImportError:
            iterator = task_args

        for args in iterator:
            result = _worker_wrapper(args)
            results.append(result)
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(status=result["status"], case=result["case_id"])
    else:
        try:
            from tqdm import tqdm
            with Pool(processes=num_workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(_worker_wrapper, task_args),
                    total=n_cases,
                    desc="Creating targets",
                    unit="case",
                ):
                    results.append(result)
        except ImportError:
            with Pool(processes=num_workers) as pool:
                results = pool.map(_worker_wrapper, task_args)

    # Summary
    results_df = pd.DataFrame(results)
    n_ok = (results_df["status"] == "ok").sum()
    n_skipped = (results_df["status"] == "skipped").sum()
    n_error = (results_df["status"] == "error").sum()

    logger.info("=" * 60)
    logger.info("Target creation complete!")
    logger.info("  OK     : %d", n_ok)
    logger.info("  Skipped: %d", n_skipped)
    logger.info("  Errors : %d", n_error)
    logger.info("=" * 60)

    if n_error > 0:
        error_cases = results_df[results_df["status"] == "error"]
        for _, erow in error_cases.iterrows():
            logger.error("  FAILED: %s — %s", erow["case_id"], erow["message"][:200])

    # Save report
    report_path = target_dir / "target_creation_report.csv"
    results_df.to_csv(report_path, index=False)
    logger.info("Report saved to %s", report_path)

    # Create a summary of case-level labels
    _create_label_summary(target_dir, case_ids)

    return results_df


def _create_label_summary(target_dir: Path, case_ids: list[str]) -> None:
    """Aggregate all case-level labels into a single CSV summary.

    Args:
        target_dir: Directory containing per-case label JSON files.
        case_ids: List of case identifiers.
    """
    label_rows: list[dict[str, Any]] = []
    for cid in case_ids:
        label_path = target_dir / f"{cid}_label.json"
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                label = json.load(f)
            label["case_id"] = cid
            label_rows.append(label)

    if not label_rows:
        return

    labels_df = pd.DataFrame(label_rows)
    summary_path = target_dir / "case_labels_summary.csv"
    labels_df.to_csv(summary_path, index=False)

    # Log distribution
    n_active = labels_df.get("has_active_lesion", pd.Series()).sum()
    n_high_risk = labels_df.get("high_risk", pd.Series()).sum()
    n_primary = labels_df.get("has_primary", pd.Series()).sum()
    n_nodes = labels_df.get("has_nodes", pd.Series()).sum()

    logger.info("Label summary (%d cases):", len(labels_df))
    logger.info("  Active lesion: %d", n_active)
    logger.info("  High risk    : %d", n_high_risk)
    logger.info("  Has primary  : %d", n_primary)
    logger.info("  Has nodes    : %d", n_nodes)
    logger.info("Saved to %s", summary_path)


# ======================================================================== #
# CLI                                                                      #
# ======================================================================== #


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Create training targets from preprocessed HECKTOR data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data.yaml",
        help="Path to data YAML config.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Override: path to manifest CSV.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Override: directory with preprocessed .npy arrays.",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="Override: output directory for targets.",
    )
    parser.add_argument(
        "--target-types",
        nargs="+",
        default=None,
        help="Override: target types to generate (e.g. binary gaussian_heatmap pet_derived).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Override: Gaussian sigma for heatmap smoothing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing target files.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for create_targets script."""
    args = parse_args()
    t0 = time.time()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        cfg = load_config(str(config_path))
    else:
        logger.warning("Config %s not found; using defaults and CLI.", config_path)
        cfg = {}

    data_cfg = cfg.get("data", {})

    # Resolve paths
    manifest_path = args.manifest or data_cfg.get(
        "manifest_path", "data/processed/manifests/manifest.csv"
    )
    processed_dir = args.processed_dir or data_cfg.get(
        "processed_dir", "data/processed"
    )
    target_dir = args.target_dir or str(
        Path(processed_dir) / "targets"
    )

    # Build target config
    default_types = ["binary", "gaussian_heatmap"]
    target_type_str = data_cfg.get("target_type", "gaussian")

    # Map simplified names to internal names
    if target_type_str == "gaussian":
        default_types = ["binary", "gaussian_heatmap"]
    elif target_type_str == "pet_derived":
        default_types = ["binary", "gaussian_heatmap", "pet_derived"]
    elif target_type_str == "binary":
        default_types = ["binary"]

    target_config: dict[str, Any] = {
        "types": args.target_types or default_types,
        "gaussian_sigma": args.sigma or data_cfg.get("gaussian_sigma", 3.0),
        "pet_normalize": True,
        "pet_suv_min": 0.0,
        "pet_suv_max": 25.0,
        "high_risk_threshold": 0.3,
    }

    logger.info("Config: %s", args.config)
    logger.info("Target config: %s", target_config)

    create_all_targets(
        manifest_path=manifest_path,
        processed_dir=processed_dir,
        target_dir=target_dir,
        target_config=target_config,
        num_workers=args.workers,
        overwrite=args.overwrite,
    )

    elapsed = time.time() - t0
    logger.info("Total time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
