#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a data manifest CSV from raw HECKTOR data.

Quét thư mục dữ liệu HECKTOR để tìm CT, PET, mask; trích xuất metadata
(spacing, size, has_primary, has_nodes, site); tạo manifest.csv.

Usage:
    python scripts/build_manifest.py --config configs/data.yaml
    python scripts/build_manifest.py --data-root data/raw/hecktor --output data/processed/manifests/manifest.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src` is importable.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

logger = setup_logger("build_manifest", level="INFO")

# ======================================================================== #
# HECKTOR file-name conventions                                            #
# ======================================================================== #
# HECKTOR 2021: {CASE_ID}_ct.nii.gz, {CASE_ID}_pt.nii.gz,
#               {CASE_ID}_gt.nii.gz  (ground truth mask)
# HECKTOR 2022: {CASE_ID}__CT.nii.gz, {CASE_ID}__PT.nii.gz,
#               {CASE_ID}.nii.gz (mask in labelsTr/)
# We support both conventions automatically.

_CT_SUFFIXES = ("_ct.nii.gz", "_CT.nii.gz", "__CT.nii.gz")
_PET_SUFFIXES = ("_pt.nii.gz", "_PT.nii.gz", "__PT.nii.gz")
_MASK_SUFFIXES = ("_gt.nii.gz", "_GTVp.nii.gz", ".nii.gz")


def _find_file(
    base_dirs: list[Path],
    case_id: str,
    suffixes: tuple[str, ...],
    *,
    mask_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Search for a file matching case_id + one of the suffixes.

    Args:
        base_dirs: Directories to search in.
        case_id: Patient/case identifier.
        suffixes: Candidate file-name suffixes to try.
        mask_dir: If provided, also search in this directory (for masks
            stored separately, e.g. HECKTOR 2022 ``labelsTr/``).

    Returns:
        Resolved Path if found, else None.
    """
    search_dirs = list(base_dirs)
    if mask_dir is not None:
        search_dirs.append(mask_dir)

    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for suffix in suffixes:
            candidate = directory / f"{case_id}{suffix}"
            if candidate.exists():
                return candidate
    return None


def _extract_site(case_id: str) -> str:
    """Heuristically extract the clinical site from a HECKTOR case ID.

    HECKTOR IDs typically start with a site code, e.g. ``CHGJ001``,
    ``CHUS042``, ``CHUM019``, ``MDA201``.

    Args:
        case_id: The case identifier.

    Returns:
        Site code string, or ``'unknown'`` if pattern not recognized.
    """
    # Strip trailing digits to get the site prefix
    prefix = ""
    for ch in case_id:
        if ch.isalpha():
            prefix += ch
        else:
            break
    return prefix if prefix else "unknown"


def _extract_metadata_sitk(path: Path) -> dict[str, Any]:
    """Read a NIfTI header with SimpleITK and return spatial metadata.

    Args:
        path: Path to the NIfTI file.

    Returns:
        Dict with spacing_x/y/z, size_x/y/z, origin, direction.

    Raises:
        RuntimeError: If SimpleITK cannot read the file.
    """
    try:
        img = sitk.ReadImage(str(path))
    except Exception as exc:
        raise RuntimeError(f"Cannot read {path}: {exc}") from exc

    spacing = img.GetSpacing()  # (x, y, z)
    size = img.GetSize()        # (x, y, z)

    return {
        "spacing_x": round(spacing[0], 4),
        "spacing_y": round(spacing[1], 4),
        "spacing_z": round(spacing[2], 4),
        "size_x": size[0],
        "size_y": size[1],
        "size_z": size[2],
    }


def _extract_mask_info(mask_path: Path) -> dict[str, Any]:
    """Analyse a lesion mask to determine primary / node presence.

    In HECKTOR, label 1 = primary GTV (GTVp), label 2 = nodal GTV (GTVn).
    Some datasets use only binary (0/1).

    Args:
        mask_path: Path to the mask NIfTI file.

    Returns:
        Dict with has_primary, has_nodes, num_lesion_voxels.
    """
    try:
        mask_img = sitk.ReadImage(str(mask_path))
        mask_arr = sitk.GetArrayFromImage(mask_img)
    except Exception as exc:
        logger.warning("Cannot read mask %s: %s", mask_path, exc)
        return {
            "has_primary": False,
            "has_nodes": False,
            "num_lesion_voxels": 0,
        }

    unique_labels = set(np.unique(mask_arr).tolist())

    # HECKTOR convention: 1 = primary, 2 = node
    has_primary = bool(1 in unique_labels)
    has_nodes = bool(2 in unique_labels)

    # If mask is purely binary 0/1, treat all foreground as primary
    if unique_labels <= {0, 1}:
        has_primary = bool(mask_arr.any())
        has_nodes = False

    num_lesion_voxels = int((mask_arr > 0).sum())

    return {
        "has_primary": has_primary,
        "has_nodes": has_nodes,
        "num_lesion_voxels": num_lesion_voxels,
    }


def _discover_case_ids(data_root: Path) -> list[str]:
    """Discover unique case IDs by scanning for CT NIfTI files.

    Args:
        data_root: Root directory of raw HECKTOR data.

    Returns:
        Sorted list of unique case ID strings.
    """
    case_ids: set[str] = set()

    # Recursively find NIfTI files that match CT suffixes
    for nifti_path in data_root.rglob("*.nii.gz"):
        name = nifti_path.name
        for suffix in _CT_SUFFIXES:
            if name.endswith(suffix):
                cid = name[: -len(suffix)]
                case_ids.add(cid)
                break

    return sorted(case_ids)


def _quality_flag(row: dict[str, Any]) -> str:
    """Assign a quality flag to a case based on completeness.

    Args:
        row: Partial manifest row dict.

    Returns:
        One of: 'ok', 'missing_pet', 'missing_mask', 'missing_both',
        'empty_mask'.
    """
    flags: list[str] = []
    if not row.get("pet_path"):
        flags.append("missing_pet")
    if not row.get("mask_path"):
        flags.append("missing_mask")
    if row.get("num_lesion_voxels", 0) == 0 and row.get("mask_path"):
        flags.append("empty_mask")
    return "|".join(flags) if flags else "ok"


def build_manifest(
    data_root: str | Path,
    output_path: str | Path,
    *,
    mask_subdir: str = "labelsTr",
) -> pd.DataFrame:
    """Scan HECKTOR data directory and build a manifest CSV.

    Args:
        data_root: Root directory containing raw HECKTOR data
            (CT, PET, mask files).
        output_path: Where to save the manifest CSV.
        mask_subdir: Name of the subdirectory containing masks
            (HECKTOR 2022 convention). Set to empty string to skip.

    Returns:
        DataFrame with manifest data.
    """
    data_root = Path(data_root)
    output_path = Path(output_path)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    # Directories to search within
    search_dirs = [data_root]
    # Also search common sub-directories
    for sub in ("imagesTr", "imagesTs", "images", "hecktor_nii"):
        sub_path = data_root / sub
        if sub_path.is_dir():
            search_dirs.append(sub_path)

    # Mask directory (HECKTOR 2022)
    mask_dir = data_root / mask_subdir if mask_subdir else None
    if mask_dir is not None and not mask_dir.is_dir():
        mask_dir = None

    # Discover case IDs
    case_ids = _discover_case_ids(data_root)
    if not case_ids:
        logger.error("No cases found under %s. Check file naming.", data_root)
        raise ValueError(f"No HECKTOR CT files found under {data_root}")

    logger.info("Discovered %d case IDs under %s", len(case_ids), data_root)

    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for idx, case_id in enumerate(case_ids):
        if (idx + 1) % 50 == 0 or idx == 0:
            logger.info("Processing case %d/%d: %s", idx + 1, len(case_ids), case_id)

        row: dict[str, Any] = {"case_id": case_id}

        # --- Locate files ---
        ct_path = _find_file(search_dirs, case_id, _CT_SUFFIXES)
        pet_path = _find_file(search_dirs, case_id, _PET_SUFFIXES)
        mask_path = _find_file(
            search_dirs, case_id, _MASK_SUFFIXES, mask_dir=mask_dir
        )

        if ct_path is None:
            errors.append(f"CT not found for {case_id}")
            continue

        row["ct_path"] = str(ct_path)
        row["pet_path"] = str(pet_path) if pet_path else ""
        row["mask_path"] = str(mask_path) if mask_path else ""

        # --- Extract spatial metadata from CT ---
        try:
            meta = _extract_metadata_sitk(ct_path)
            row.update(meta)
        except RuntimeError as exc:
            errors.append(str(exc))
            row.update({
                "spacing_x": 0, "spacing_y": 0, "spacing_z": 0,
                "size_x": 0, "size_y": 0, "size_z": 0,
            })

        # --- Extract mask info ---
        if mask_path is not None:
            mask_info = _extract_mask_info(mask_path)
            row.update(mask_info)
        else:
            row.update({
                "has_primary": False,
                "has_nodes": False,
                "num_lesion_voxels": 0,
            })

        # --- Site extraction ---
        row["site"] = _extract_site(case_id)

        # --- Split placeholder (filled later by create_splits) ---
        row["split"] = ""

        # --- Quality flag ---
        row["quality_flag"] = _quality_flag(row)

        rows.append(row)

    if errors:
        logger.warning("Encountered %d error(s) during scanning:", len(errors))
        for err in errors[:20]:
            logger.warning("  %s", err)
        if len(errors) > 20:
            logger.warning("  ... and %d more.", len(errors) - 20)

    if not rows:
        raise RuntimeError("No valid cases found. Cannot build manifest.")

    # Build DataFrame with ordered columns
    column_order = [
        "case_id", "ct_path", "pet_path", "mask_path",
        "split", "site",
        "spacing_x", "spacing_y", "spacing_z",
        "size_x", "size_y", "size_z",
        "has_primary", "has_nodes", "num_lesion_voxels",
        "quality_flag",
    ]
    df = pd.DataFrame(rows)
    # Ensure all expected columns exist
    for col in column_order:
        if col not in df.columns:
            df[col] = ""
    df = df[column_order]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Summary statistics
    n_total = len(df)
    n_with_pet = (df["pet_path"] != "").sum()
    n_with_mask = (df["mask_path"] != "").sum()
    n_ok = (df["quality_flag"] == "ok").sum()
    sites = df["site"].nunique()

    logger.info("=" * 60)
    logger.info("Manifest built successfully!")
    logger.info("  Output   : %s", output_path)
    logger.info("  Cases    : %d total", n_total)
    logger.info("  With PET : %d (%.1f%%)", n_with_pet, 100 * n_with_pet / n_total)
    logger.info("  With mask: %d (%.1f%%)", n_with_mask, 100 * n_with_mask / n_total)
    logger.info("  Quality  : %d ok, %d flagged", n_ok, n_total - n_ok)
    logger.info("  Sites    : %d unique", sites)
    logger.info("=" * 60)

    return df


# ======================================================================== #
# CLI                                                                      #
# ======================================================================== #


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Build data manifest from raw HECKTOR directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data.yaml",
        help="Path to data YAML config (reads data.root and data.manifest_path).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override: root directory with raw HECKTOR files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override: output manifest CSV path.",
    )
    parser.add_argument(
        "--mask-subdir",
        type=str,
        default="labelsTr",
        help="Subdirectory name for mask files (HECKTOR 2022 convention).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for build_manifest script."""
    args = parse_args()
    t0 = time.time()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        cfg = load_config(str(config_path))
    else:
        logger.warning("Config %s not found; using CLI args only.", config_path)
        cfg = {}

    data_cfg = cfg.get("data", {})

    # Resolve paths (CLI overrides > config)
    data_root = args.data_root or data_cfg.get("root", "data/raw/hecktor")
    output_path = args.output or data_cfg.get(
        "manifest_path", "data/processed/manifests/manifest.csv"
    )

    logger.info("Building manifest from: %s", data_root)
    logger.info("Output manifest: %s", output_path)

    build_manifest(
        data_root=data_root,
        output_path=output_path,
        mask_subdir=args.mask_subdir,
    )

    elapsed = time.time() - t0
    logger.info("Total time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
