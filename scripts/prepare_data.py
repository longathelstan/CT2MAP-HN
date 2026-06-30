#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Preprocess raw HECKTOR data (CT/PET/mask) into standardised volumes.

Pipeline cho mỗi case:
  1. Load CT → standardize orientation → resample → crop ROI → clip HU → normalize
  2. Optionally process PET and mask in the same spatial frame
  3. Save processed arrays as .npy to output directory

Hỗ trợ đa luồng (multiprocessing), resume (skip đã xử lý), progress bar.

Usage:
    python scripts/prepare_data.py --config configs/preprocess.yaml
    python scripts/prepare_data.py --config configs/preprocess.yaml --workers 16 --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataio.preprocess import preprocess_case  # noqa: E402
from src.dataio.readers import save_nifti  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

logger = setup_logger("prepare_data", level="INFO")


# ======================================================================== #
# Single-case worker                                                       #
# ======================================================================== #


def _process_single_case(
    row: dict[str, Any],
    preprocess_cfg: dict[str, Any],
    output_dir: Path,
    save_format: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Preprocess a single case and save results.

    This function is designed to be called from a multiprocessing pool.
    It must be a top-level function (not a closure) for pickling.

    Args:
        row: A manifest row dict with case_id, ct_path, pet_path, mask_path.
        preprocess_cfg: Preprocessing config dict (from preprocess.yaml).
        output_dir: Directory for saving processed files.
        save_format: Output format — ``'npy'`` or ``'nifti'``.
        overwrite: Whether to overwrite existing files.

    Returns:
        Status dict with case_id, status ('ok'|'skipped'|'error'), and message.
    """
    case_id = row["case_id"]
    ct_path = row.get("ct_path", "")
    pet_path = row.get("pet_path", "") or None
    mask_path = row.get("mask_path", "") or None

    # Check if already processed (resume support)
    ct_out = output_dir / f"{case_id}_ct.npy"
    if ct_out.exists() and not overwrite:
        return {"case_id": case_id, "status": "skipped", "message": "already processed"}

    if not ct_path or not Path(ct_path).exists():
        return {"case_id": case_id, "status": "error", "message": f"CT not found: {ct_path}"}

    # Resolve optional paths
    if pet_path and not Path(pet_path).exists():
        pet_path = None
    if mask_path and not Path(mask_path).exists():
        mask_path = None

    try:
        result = preprocess_case(
            ct_path=ct_path,
            mask_path=mask_path,
            pet_path=pet_path,
            config=preprocess_cfg,
        )

        # Save processed arrays
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_format == "npy":
            np.save(str(output_dir / f"{case_id}_ct.npy"), result["ct_array"])

            if result["pet_array"] is not None:
                np.save(str(output_dir / f"{case_id}_pet.npy"), result["pet_array"])

            if result["mask_array"] is not None:
                np.save(str(output_dir / f"{case_id}_mask.npy"), result["mask_array"])
        else:
            # Save as NIfTI
            meta = result["ct_meta"]
            save_nifti(
                result["ct_array"], meta,
                output_dir / f"{case_id}_ct.nii.gz",
            )
            if result["pet_array"] is not None:
                save_nifti(
                    result["pet_array"], meta,
                    output_dir / f"{case_id}_pet.nii.gz",
                )
            if result["mask_array"] is not None:
                save_nifti(
                    result["mask_array"].astype(np.float32), meta,
                    output_dir / f"{case_id}_mask.nii.gz",
                )

        # Save crop info for reversibility
        crop_info = result.get("crop_info", {})
        # Convert numpy types for JSON serialisation
        crop_json = _sanitise_for_json(crop_info)
        with open(output_dir / f"{case_id}_crop_info.json", "w", encoding="utf-8") as f:
            json.dump(crop_json, f, indent=2)

        return {
            "case_id": case_id,
            "status": "ok",
            "message": f"shape={result['ct_array'].shape}",
        }

    except Exception as exc:
        tb = traceback.format_exc()
        return {
            "case_id": case_id,
            "status": "error",
            "message": f"{exc}\n{tb}",
        }


def _sanitise_for_json(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON.

    Args:
        obj: Object to sanitise.

    Returns:
        JSON-serialisable version of the object.
    """
    if isinstance(obj, dict):
        return {k: _sanitise_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitise_for_json(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, slice):
        return {"start": obj.start, "stop": obj.stop}
    return obj


# ======================================================================== #
# Wrapper for multiprocessing (top-level for pickling)                      #
# ======================================================================== #


def _worker_wrapper(args: tuple) -> dict[str, Any]:
    """Unpack arguments and call _process_single_case.

    Args:
        args: Tuple of (row_dict, preprocess_cfg, output_dir, save_format, overwrite).

    Returns:
        Status dict from _process_single_case.
    """
    row, preprocess_cfg, output_dir, save_format, overwrite = args
    return _process_single_case(row, preprocess_cfg, output_dir, save_format, overwrite)


# ======================================================================== #
# Main pipeline                                                            #
# ======================================================================== #


def prepare_data(
    manifest_path: str | Path,
    preprocess_cfg: dict[str, Any],
    output_dir: str | Path,
    *,
    num_workers: int = 4,
    overwrite: bool = False,
    save_format: str = "npy",
) -> pd.DataFrame:
    """Run preprocessing pipeline on all cases in the manifest.

    Args:
        manifest_path: Path to manifest CSV.
        preprocess_cfg: Preprocessing config dict.
        output_dir: Directory for processed output files.
        num_workers: Number of parallel workers (0 = sequential).
        overwrite: Overwrite existing processed files.
        save_format: Output format — ``'npy'`` or ``'nifti'``.

    Returns:
        DataFrame with processing status per case.
    """
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    n_cases = len(df)
    logger.info("Loaded manifest with %d cases from %s", n_cases, manifest_path)
    logger.info("Output directory: %s", output_dir)
    logger.info("Workers: %d | Overwrite: %s | Format: %s",
                num_workers, overwrite, save_format)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments
    rows = df.to_dict(orient="records")
    task_args = [
        (row, preprocess_cfg, output_dir, save_format, overwrite)
        for row in rows
    ]

    # Process
    results: list[dict[str, Any]] = []

    if num_workers <= 0 or num_workers == 1:
        # Sequential processing with tqdm
        try:
            from tqdm import tqdm
            iterator = tqdm(task_args, desc="Preprocessing", unit="case")
        except ImportError:
            iterator = task_args
            logger.info("Install tqdm for progress bars: pip install tqdm")

        for args in iterator:
            result = _worker_wrapper(args)
            results.append(result)
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(status=result["status"], case=result["case_id"])
    else:
        # Parallel processing
        try:
            from tqdm import tqdm
            with Pool(processes=num_workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(_worker_wrapper, task_args),
                    total=n_cases,
                    desc="Preprocessing",
                    unit="case",
                ):
                    results.append(result)
        except ImportError:
            logger.info("Install tqdm for progress bars. Running without...")
            with Pool(processes=num_workers) as pool:
                results = pool.map(_worker_wrapper, task_args)

    # Summary
    results_df = pd.DataFrame(results)
    n_ok = (results_df["status"] == "ok").sum()
    n_skipped = (results_df["status"] == "skipped").sum()
    n_error = (results_df["status"] == "error").sum()

    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info("  OK     : %d", n_ok)
    logger.info("  Skipped: %d (already processed)", n_skipped)
    logger.info("  Errors : %d", n_error)
    logger.info("=" * 60)

    if n_error > 0:
        error_cases = results_df[results_df["status"] == "error"]
        for _, erow in error_cases.iterrows():
            logger.error(
                "  FAILED: %s — %s", erow["case_id"], erow["message"][:200]
            )

    # Save processing report
    report_path = output_dir / "preprocessing_report.csv"
    results_df.to_csv(report_path, index=False)
    logger.info("Report saved to %s", report_path)

    return results_df


# ======================================================================== #
# CLI                                                                      #
# ======================================================================== #


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess raw HECKTOR data to standardised volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocess.yaml",
        help="Path to preprocessing YAML config.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Override: path to manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override: output directory for processed files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override: number of parallel workers (0 = sequential).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed files.",
    )
    parser.add_argument(
        "--save-format",
        type=str,
        choices=["npy", "nifti"],
        default="npy",
        help="Output file format.",
    )
    return parser.parse_args()


def _build_preprocess_config(prep_cfg: dict[str, Any]) -> dict[str, Any]:
    """Map preprocess.yaml structure to the config dict expected by preprocess_case().

    Args:
        prep_cfg: Raw ``preprocess:`` section from the YAML config.

    Returns:
        Flat config dict compatible with ``preprocess_case()``.
    """
    hu_clip = prep_cfg.get("hu_clip", [-200, 300])
    crop_margin = prep_cfg.get("crop_margin", [10, 10, 10])
    crop_size = prep_cfg.get("crop_size", [192, 192, 192])

    return {
        "target_orientation": prep_cfg.get("orientation", "RAS"),
        "target_spacing": prep_cfg.get("target_spacing", [1.0, 1.0, 1.0]),
        "ct_interpolation": "linear",
        "pet_interpolation": "linear",
        "mask_interpolation": "nearest",
        "hu_min": hu_clip[0],
        "hu_max": hu_clip[1],
        "normalize_method": prep_cfg.get("normalize_method", "minmax"),
        "crop": {
            "method": prep_cfg.get("crop_method", "bbox"),
            "margin": crop_margin,
            "fixed_size": crop_size,
        },
    }


def main() -> None:
    """Entry point for prepare_data script."""
    args = parse_args()
    t0 = time.time()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        cfg = load_config(str(config_path))
    else:
        logger.warning("Config %s not found; using defaults and CLI.", config_path)
        cfg = {}

    prep_raw = cfg.get("preprocess", {})

    # Resolve parameters (CLI overrides > config)
    manifest_path = args.manifest or cfg.get("data", {}).get(
        "manifest_path", "data/processed/manifests/manifest.csv"
    )
    output_dir = args.output_dir or prep_raw.get("output_dir", "data/processed")
    num_workers = args.workers if args.workers is not None else prep_raw.get("num_workers", 4)
    overwrite = args.overwrite or prep_raw.get("overwrite", False)

    # Build preprocess config for preprocess_case()
    preprocess_cfg = _build_preprocess_config(prep_raw)

    logger.info("Config: %s", args.config)
    logger.info("Manifest: %s", manifest_path)
    logger.info("Output: %s", output_dir)

    prepare_data(
        manifest_path=manifest_path,
        preprocess_cfg=preprocess_cfg,
        output_dir=output_dir,
        num_workers=num_workers,
        overwrite=overwrite,
        save_format=args.save_format,
    )

    elapsed = time.time() - t0
    logger.info("Total time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
