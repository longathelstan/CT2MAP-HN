"""
Smoke test script to overfit Model 0 Baseline v2 on a few cases.
Checks for lesion Dice > 0.8 and proper heatmap quality.
"""

import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import yaml
import numpy as np
import matplotlib.pyplot as plt

from src.models import build_model
from src.train.train_baseline import _build_dataloaders, _build_optimizer, _build_loss, _seed_everything
from src.inference.postprocess import extract_connected_components, quality_check_heatmap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model0_v2.yaml")
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Force reproducible settings
    _seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Restrict batch size and num workers for overfit
    config["training"]["batch_size"] = 2
    config["data"]["num_workers"] = 2
    
    # Enable faster convergence for overfitting smoke test
    config.setdefault("optimizer", {})
    config["optimizer"]["lr"] = 5.0e-3
    config["data"].setdefault("augmentation", {})
    config["data"]["augmentation"]["num_samples"] = 1
    # Overfit smoke test MUST sample lesion-containing patches only.
    # Default pos_ratio=0.7 leaves a 30% chance of a background-only patch
    # (empty target -> Dice=0.0), which is a test artifact, not a model failure.
    config["data"]["augmentation"]["pos_ratio"] = 1.0
    
    config.setdefault("model", {})
    config["model"]["name"] = "baseline_unet"

    # Build components
    model = build_model(config).to(device)
    optimizer = _build_optimizer(model, config.get("optimizer", {}))
    loss_fn = _build_loss(config.get("loss", {}))

    train_loader, _ = _build_dataloaders(config.get("data", {}), training_cfg=config.get("training", {}))

    logger.info(f"Starting overfit smoke test for {args.iters} iterations...")
    model.train()
    
    # Grab one fixed batch to overfit
    batch = next(iter(train_loader))
    ct = batch["ct"].to(device)
    heatmap_target = batch["heatmap"].to(device)
    lesion_target = batch["lesion_mask"].to(device)

    for i in range(args.iters):
        optimizer.zero_grad()
        
        preds = model(ct)
        pred_heatmap = preds["heatmap"]
        pred_lesion = preds["lesion"]
        pred_triage = preds.get("triage")
        
        losses = loss_fn(
            pred_heatmap=pred_heatmap,
            target_heatmap=heatmap_target,
            pred_lesion=pred_lesion,
            target_lesion=lesion_target,
            pred_triage=pred_triage,
            target_triage=None, # Will be ignored since w_triage is 0
            lesion_mask=lesion_target
        )
        
        loss = losses["total"]
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 20 == 0:
            logger.info(f"Iter {i+1:03d} | Total Loss: {loss.item():.4f} | Heatmap: {losses['heatmap_loss'].item():.4f} | Lesion: {losses['lesion_loss'].item():.4f}")

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        preds = model(ct)
        pred_heatmap = preds["heatmap"].cpu().numpy()
        pred_lesion = preds["lesion"].cpu().numpy()
    
    heatmap_target_np = heatmap_target.cpu().numpy()
    lesion_target_np = lesion_target.cpu().numpy()

    out_dir = Path(config["paths"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {"passed": True, "cases": []}
    
    for b in range(ct.shape[0]):
        # Check Lesion Dice
        p_lesion = (pred_lesion[b, 0] > 0.5).astype(np.float32)
        t_lesion = lesion_target_np[b, 0]
        
        intersection = (p_lesion * t_lesion).sum()
        dice = (2.0 * intersection) / (p_lesion.sum() + t_lesion.sum() + 1e-6)
        
        # Quality check heatmap
        p_heat = pred_heatmap[b, 0]
        binary_heat = (p_heat >= 0.5).astype(np.uint8)
        components = extract_connected_components(binary_heat)
        flags = quality_check_heatmap(p_heat, components)
        
        case_id = batch["case_id"][b] if "case_id" in batch else f"sample_{b}"
        
        res = {
            "case_id": case_id,
            "lesion_dice": float(dice),
            "heatmap_flags": flags,
            "passed": bool(dice > 0.8 and len(flags) == 0)
        }
        summary["cases"].append(res)
        if not res["passed"]:
            summary["passed"] = False
            
        # Plot center slice overlay
        z_center = p_heat.shape[0] // 2
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(ct[b, 0, z_center].cpu().numpy(), cmap='gray')
        axes[0].set_title("CT")
        axes[1].imshow(t_lesion[z_center], cmap='gray')
        axes[1].set_title("Target Lesion")
        axes[2].imshow(p_heat[z_center], cmap='hot')
        axes[2].set_title("Pred Heatmap")
        plt.savefig(out_dir / f"overfit_{case_id}.png")
        plt.close()

    with open(out_dir / "overfit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    logger.info("Overfit check complete.")
    logger.info(json.dumps(summary, indent=2))
    
    if summary["passed"]:
        logger.info("✅ OVERFIT SMOKE TEST PASSED. Ready for full train.")
    else:
        logger.error("❌ OVERFIT SMOKE TEST FAILED.")


if __name__ == "__main__":
    main()
