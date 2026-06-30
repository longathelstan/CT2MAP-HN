# -*- coding: utf-8 -*-
"""Batch inference for CT2MAP-HN.

Processes all CT cases listed in a manifest file (test split), saves
per-case results and a summary CSV.

CLI usage:
    $ python -m src.inference.infer_batch \\
        --manifest data/manifest.csv \\
        --ckpt outputs/checkpoints/best.pt \\
        --config configs/demo.yaml \\
        --out outputs/batch_predictions/

Library usage:
    >>> from src.inference.infer_batch import batch_infer
    >>> batch_infer("data/manifest.csv", "ckpt/best.pt", config, "out/")
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _load_manifest(
    manifest_path: str,
    split: str = "test",
) -> List[Dict[str, str]]:
    """Load a case manifest CSV and filter to the requested split.

    Expected CSV columns: ``case_id``, ``ct_path``, ``split``
    (at minimum).  Extra columns are preserved.

    Args:
        manifest_path: Path to the manifest CSV.
        split: Split to select (e.g. ``"test"``).

    Returns:
        List of row-dicts for the requested split.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        ValueError: If required columns are missing.
    """
    mpath = Path(manifest_path)
    if not mpath.exists():
        raise FileNotFoundError(f"Manifest not found: {mpath}")

    with open(mpath, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []

        for required in ("case_id", "ct_path"):
            if required not in fieldnames:
                raise ValueError(
                    f"Manifest is missing required column '{required}'. "
                    f"Found columns: {fieldnames}"
                )

        rows = list(reader)

    # Filter by split if the column exists
    if "split" in fieldnames:
        rows = [r for r in rows if r.get("split", "").lower() == split.lower()]
        logger.info(
            "Manifest: %d cases in '%s' split (total rows read: %d)",
            len(rows),
            split,
            len(rows),
        )
    else:
        logger.warning(
            "Manifest has no 'split' column — using all %d rows", len(rows)
        )

    return rows


def batch_infer(
    manifest_path: str,
    checkpoint_path: str,
    config: Union[dict, Any],
    output_dir: str,
    *,
    split: str = "test",
    device: str = "cuda",
    mc_passes: int = 0,
) -> List[Dict[str, Any]]:
    """Run inference on all cases in a manifest file.

    Args:
        manifest_path: Path to the manifest CSV.
        checkpoint_path: Path to the model checkpoint.
        config: Configuration dict or Config object.
        output_dir: Root directory for saving results.  Each case gets a
            subdirectory named by its ``case_id``.
        split: Manifest split to process.
        device: Torch device string.
        mc_passes: Number of MC Dropout passes (0 = deterministic).

    Returns:
        List of per-case summary dicts.

    Example:
        >>> results = batch_infer("manifest.csv", "best.pt", cfg, "out/")
    """
    from src.inference.infer_case import CaseInferencer
    from src.utils.io import ensure_dir

    try:
        from tqdm import tqdm  # type: ignore[import-untyped]

        has_tqdm = True
    except ImportError:
        has_tqdm = False

    # Load manifest
    cases = _load_manifest(manifest_path, split=split)
    if not cases:
        logger.warning("No cases found in manifest for split '%s'", split)
        return []

    # Initialise inferencer (loads model once)
    inferencer = CaseInferencer(checkpoint_path, config, device=device)

    # Output directory
    out_root = ensure_dir(output_dir)
    summaries: List[Dict[str, Any]] = []

    # Progress bar
    iterator = tqdm(cases, desc="Batch inference") if has_tqdm else cases

    t0_total = time.time()
    for case in iterator:
        case_id = case["case_id"]
        ct_path = case["ct_path"]

        if not Path(ct_path).exists():
            logger.error("CT file not found: %s (case %s) — skipping", ct_path, case_id)
            summaries.append(
                {
                    "case_id": case_id,
                    "status": "error",
                    "error": f"CT file not found: {ct_path}",
                }
            )
            continue

        case_out_dir = str(out_root / case_id)
        try:
            if mc_passes > 0:
                results = inferencer.infer_with_mc_dropout(
                    ct_path, num_passes=mc_passes
                )
            else:
                results = inferencer.infer(ct_path)

            # Save outputs
            inferencer.save_outputs(
                results, case_out_dir, reference_ct_path=ct_path
            )

            summary = {
                "case_id": case_id,
                "status": "success",
                "triage_score": results["triage_score"],
                "uncertainty_score": results["uncertainty_score"],
                "num_candidates": len(results["lesion_candidates"]),
                "elapsed_seconds": results.get("elapsed_seconds"),
            }

        except Exception as exc:
            logger.error("Error processing case %s: %s", case_id, exc, exc_info=True)
            summary = {
                "case_id": case_id,
                "status": "error",
                "error": str(exc),
            }

        summaries.append(summary)

        if has_tqdm:
            iterator.set_postfix(  # type: ignore[union-attr]
                case=case_id,
                score=summary.get("triage_score", "N/A"),
            )

    total_elapsed = time.time() - t0_total

    # Save summary CSV
    _save_summary_csv(summaries, str(out_root / "batch_summary.csv"))

    # Save summary JSON
    summary_json = {
        "total_cases": len(cases),
        "successful": sum(1 for s in summaries if s.get("status") == "success"),
        "failed": sum(1 for s in summaries if s.get("status") == "error"),
        "total_elapsed_seconds": total_elapsed,
        "cases": summaries,
    }
    json_path = out_root / "batch_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary_json, fh, indent=2, ensure_ascii=False, default=str)

    logger.info(
        "Batch inference complete: %d/%d successful in %.1fs",
        summary_json["successful"],
        summary_json["total_cases"],
        total_elapsed,
    )
    return summaries


def _save_summary_csv(
    summaries: List[Dict[str, Any]],
    path: str,
) -> None:
    """Save batch summary to a CSV file.

    Args:
        summaries: List of per-case summary dicts.
        path: Output CSV path.
    """
    if not summaries:
        return

    fieldnames = list(summaries[0].keys())
    # Ensure all keys from all dicts are included
    for s in summaries:
        for k in s:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    logger.info("Saved batch summary CSV to %s", path)


# ======================================================================
# CLI
# ======================================================================


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for batch inference."""
    parser = argparse.ArgumentParser(
        description="CT2MAP-HN: Batch metabolic risk heatmap inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to the manifest CSV file.",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to the model checkpoint (.pt).",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for batch results.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Manifest split to process.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--mc-passes",
        type=int,
        default=0,
        help="Number of MC Dropout passes (0 = deterministic).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for batch inference."""
    from src.utils.config import load_config
    from src.utils.logger import setup_logger

    args = _parse_args(argv)
    setup_logger("ct2map", level="INFO")

    config = load_config(args.config)
    batch_infer(
        manifest_path=args.manifest,
        checkpoint_path=args.ckpt,
        config=config,
        output_dir=args.out,
        split=args.split,
        device=args.device,
        mc_passes=args.mc_passes,
    )


if __name__ == "__main__":
    main()
