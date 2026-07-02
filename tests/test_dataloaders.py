# -*- coding: utf-8 -*-
"""Test dataloader initialization and batch loading for CT2MAP-HN."""

import sys
from pathlib import Path
import yaml

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.train.train_baseline import _build_dataloaders

def test_dataloaders():
    print("Loading config...")
    cfg_path = _PROJECT_ROOT / "configs/baseline_nnunet.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    data_cfg = cfg.get("data", {})
    print("Initializing dataloaders...")
    train_loader, val_loader = _build_dataloaders(data_cfg)
    
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    
    print("Fetching first training batch...")
    batch = next(iter(train_loader))
    print("Batch keys:", list(batch.keys()))
    print("ct shape:", batch["ct"].shape)
    print("heatmap shape:", batch["heatmap"].shape)
    print("lesion_mask shape:", batch["lesion_mask"].shape)
    print("triage_label shape:", batch["triage_label"].shape)
    print("case_id count:", len(batch["case_id"]))
    print("Dataloader test passed successfully!")

if __name__ == "__main__":
    test_dataloaders()
