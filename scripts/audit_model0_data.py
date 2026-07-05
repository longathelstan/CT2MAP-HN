"""
Audit script to check the processed data before Model 0 training.
"""

import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--manifest", type=str, default="data/processed/manifests/manifest.csv")
    parser.add_argument("--splits", type=str, default="data/processed/manifests/splits.json")
    parser.add_argument("--out_dir", type=str, default="results/model0_unet_v2")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    with open(args.splits, "r") as f:
        splits = json.load(f)
    
    records = []
    
    for split_name, case_ids in splits.items():
        for case_id in case_ids:
            ct_path = processed_dir / f"{case_id}_ct.npy"
            mask_path = processed_dir / "targets" / f"{case_id}_binary.npy"
            target_path = processed_dir / "targets" / f"{case_id}_gaussian_heatmap.npy"
            
            if not ct_path.exists():
                logger.warning(f"Missing CT for {case_id} at {ct_path}")
                continue
                
            ct = np.load(ct_path)
            mask = np.load(mask_path) if mask_path.exists() else None
            target = np.load(target_path) if target_path.exists() else None
            
            record = {
                "case_id": case_id,
                "split": split_name,
                "ct_shape": ct.shape,
                "ct_min": float(ct.min()),
                "ct_max": float(ct.max()),
                "ct_median": float(np.median(ct)),
                "has_mask": mask is not None,
                "has_target": target is not None,
            }
            
            if mask is not None:
                record["mask_shape"] = mask.shape
                record["foreground_fraction"] = float(mask.sum()) / mask.size
                
            if target is not None:
                record["target_shape"] = target.shape
                record["target_min"] = float(target.min())
                record["target_max"] = float(target.max())
                
            records.append(record)
            logger.info(f"Audited {case_id}")

    audit_df = pd.DataFrame(records)
    
    # Check for shape mismatches
    shape_mismatch = False
    for _, row in audit_df.iterrows():
        if row["has_mask"] and row["ct_shape"] != row["mask_shape"]:
            shape_mismatch = True
            logger.error(f"Shape mismatch for {row['case_id']}: {row['ct_shape']} != {row['mask_shape']}")
    
    summary = {
        "total_cases": len(audit_df),
        "train_cases": len(audit_df[audit_df["split"] == "train"]),
        "val_cases": len(audit_df[audit_df["split"] == "val"]),
        "test_cases": len(audit_df[audit_df["split"] == "test"]),
        "cases_with_mask": int(audit_df["has_mask"].sum()),
        "cases_with_target": int(audit_df["has_target"].sum()),
        "shape_mismatch_detected": shape_mismatch,
    }
    
    with open(out_dir / "data_audit.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    audit_df.to_csv(out_dir / "data_audit.csv", index=False)
    logger.info("Audit complete. Saved to %s", out_dir)


if __name__ == "__main__":
    main()
