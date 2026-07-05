"""
CT2MAP-HN – Baseline UNet training script.

Usage::

    python -m src.train.train_baseline --config configs/baseline.yaml

Loads config, builds dataset / model / optimizer / scheduler, and runs the
full training loop via :class:`src.train.engine.Trainer`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

# Project imports
from src.models import build_model
from src.models.losses import CombinedLoss
from src.train.engine import Trainer

logger = logging.getLogger(__name__)


# ====================================================================
# Reproducibility
# ====================================================================
def _seed_everything(seed: int) -> None:
    """Set seeds for full reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("All seeds set to %d.", seed)


# ====================================================================
# Builder helpers
# ====================================================================
def _build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Build optimizer from config.

    Args:
        model: The model whose parameters will be optimised.
        cfg: Optimizer config dict with ``name``, ``lr``, ``weight_decay``, etc.

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: If the optimizer name is not supported.
    """
    name = cfg.get("name", "adamw").lower()
    lr = float(cfg.get("lr", 1e-4))
    wd = float(cfg.get("weight_decay", 1e-5))

    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        return SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    num_epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """Build LR scheduler from config.

    Args:
        optimizer: The optimizer to schedule.
        cfg: Scheduler config dict with ``name`` and scheduler-specific params.
        num_epochs: Total number of training epochs (for cosine annealing).

    Returns:
        Scheduler instance, or *None* if ``name`` is ``"none"``.
    """
    name = cfg.get("name", "cosine").lower()

    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=float(cfg.get("eta_min", 1e-7)),
        )
    elif name == "step":
        return StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", 30)),
            gamma=float(cfg.get("gamma", 0.1)),
        )
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {name}")


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate config schema and fail fast on missing required keys.

    Also warns about legacy keys that are silently ignored.

    Args:
        config: Full experiment config dict.

    Raises:
        ValueError: If required top-level sections or keys are missing,
            or if legacy keys are detected.
    """
    # --- Required top-level sections ---
    required_sections = ["optimizer", "loss", "paths", "training"]
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(
            f"Config missing required sections: {missing}. "
            f"Make sure you are using the correct config file (e.g. configs/model0_v2.yaml)."
        )

    # --- Required keys within sections ---
    required_keys = {
        "training": ["num_epochs"],
        "optimizer": ["name", "lr"],
        "paths": ["checkpoint_dir"],
    }
    for section, keys in required_keys.items():
        cfg_section = config.get(section, {})
        missing_keys = [k for k in keys if k not in cfg_section]
        if missing_keys:
            raise ValueError(
                f"Config section '{section}' missing required keys: {missing_keys}"
            )

    # --- Detect legacy keys that would be silently ignored ---
    legacy_detections = []
    training_cfg = config.get("training", {})
    if "epochs" in training_cfg and "num_epochs" not in training_cfg:
        legacy_detections.append(
            "training.epochs → should be training.num_epochs"
        )
    if "lr" in training_cfg:
        legacy_detections.append(
            "training.lr → should be optimizer.lr"
        )
    if "optimizer" in training_cfg and isinstance(training_cfg["optimizer"], str):
        legacy_detections.append(
            "training.optimizer → should be optimizer.name (top-level)"
        )
    if "loss" in training_cfg:
        legacy_detections.append(
            "training.loss → should be loss (top-level section)"
        )
    if "scheduler" in training_cfg and isinstance(training_cfg["scheduler"], dict):
        sched = training_cfg["scheduler"]
        if "type" in sched and "name" not in config.get("scheduler", {}):
            legacy_detections.append(
                "training.scheduler.type → should be scheduler.name (top-level)"
            )

    if legacy_detections:
        msg = "Legacy config keys detected (these would be SILENTLY IGNORED):\n"
        for d in legacy_detections:
            msg += f"  - {d}\n"
        msg += "Please use configs/model0_v2.yaml with correct schema."
        raise ValueError(msg)

    logger.info("Config validation passed.")


def _print_resolved_config(config: Dict[str, Any]) -> None:
    """Print resolved hyperparameters before training starts.

    Args:
        config: Full experiment config dict.
    """
    training = config.get("training", {})
    opt = config.get("optimizer", {})
    sched = config.get("scheduler", {})
    loss = config.get("loss", {})
    paths = config.get("paths", {})
    preproc = config.get("preprocessing", {})

    # Batch size: training.batch_size is source-of-truth
    batch_size = training.get("batch_size", config.get("data", {}).get("batch_size", 4))
    accum = training.get("grad_accumulation_steps", 1)

    info = f"""
╔══════════════════════════════════════════════╗
║         RESOLVED TRAINING CONFIG             ║
╠══════════════════════════════════════════════╣
║  Epochs:           {training.get('num_epochs', '?'):<25}║
║  Batch size:       {batch_size:<25}║
║  Grad accumulation:{accum:<25}║
║  Effective batch:  {batch_size * accum:<25}║
║  AMP:              {training.get('use_amp', False)!s:<25}║
║  Grad clip norm:   {training.get('clip_grad_norm', 'None')!s:<25}║
║  Optimizer:        {opt.get('name', '?'):<25}║
║  LR:               {opt.get('lr', '?')!s:<25}║
║  Weight decay:     {opt.get('weight_decay', '?')!s:<25}║
║  Scheduler:        {sched.get('name', '?'):<25}║
║  Loss weights:     H={loss.get('w_heatmap', 1.0)} L={loss.get('w_lesion', 1.0)} T={loss.get('w_triage', 0.5):<5}║
║  Heatmap loss:     {loss.get('heatmap', {}).get('base', 'mse'):<25}║
║  HU range:         [{preproc.get('hu_min', -1024)}, {preproc.get('hu_max', 1024)}]{'':<14}║
║  Checkpoint dir:   {paths.get('checkpoint_dir', '?'):<25}║
╚══════════════════════════════════════════════╝"""
    logger.info(info)


def _build_dataloaders(
    cfg: Dict[str, Any],
    training_cfg: Dict[str, Any] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders using real HECKTOR data.

    Args:
        cfg: Data config dict from config.get("data", {}).
        training_cfg: Training config dict. If provided, ``batch_size``
            is read from here as source-of-truth, with fallback to
            ``cfg["batch_size"]``.

    Returns:
        Tuple of ``(train_loader, val_loader)``.
    """
    from src.datasets.ct_dataset import CTHeatmapDataset
    from src.datasets.transforms import get_train_transforms, get_val_transforms
    from src.utils.config import load_config

    # Load the data configuration yaml
    data_config_path = cfg.get("config", "configs/data.yaml")
    data_cfg = load_config(data_config_path).get("data", {})

    manifest_path = data_cfg.get("manifest_path", "data/processed/manifests/manifest.csv")
    processed_dir = data_cfg.get("processed_dir", "data/processed")
    target_dir = data_cfg.get("target_dir", str(Path(processed_dir) / "targets"))
    splits_path = data_cfg.get("split_file", "data/processed/manifests/splits.json")
    target_type = data_cfg.get("target_type", "gaussian")

    # Map baseline yaml targets: 'gaussian' -> 'gaussian_heatmap'
    if target_type == "gaussian":
        dataset_target_type = "gaussian_heatmap"
    else:
        dataset_target_type = target_type

    # Batch size: training.batch_size is source-of-truth
    if training_cfg is not None and "batch_size" in training_cfg:
        batch_size = int(training_cfg["batch_size"])
    else:
        batch_size = int(cfg.get("batch_size", 4))
    logger.info("DataLoader batch_size = %d (source: %s)",
                batch_size,
                "training.batch_size" if training_cfg and "batch_size" in training_cfg else "data.batch_size")

    num_workers = int(cfg.get("num_workers", 8))
    pin_memory = bool(cfg.get("pin_memory", True))

    # Build transforms
    aug_cfg = cfg.get("augmentation", {})
    train_transform_config = {
        "spatial_size": aug_cfg.get("patch_size", [128, 128, 128]),
        "flip_prob": aug_cfg.get("random_flip_prob", 0.5),
        "flip_axes": [0, 1, 2],
        "num_samples": 2,  # number of crops per volume
        "pos_ratio": 0.7,
        "affine_prob": 0.3,
        "gaussian_noise_prob": 0.2,
    }
    val_transform_config = {
        "spatial_size": cfg.get("val_patch_size", [128, 128, 128]),
    }

    train_transforms = get_train_transforms(train_transform_config)
    val_transforms = get_val_transforms(val_transform_config)

    # Initialize datasets
    train_ds = CTHeatmapDataset(
        manifest_path=manifest_path,
        split="train",
        processed_dir=processed_dir,
        target_dir=target_dir,
        splits_path=splits_path,
        transform=train_transforms,
        target_type=dataset_target_type,
    )

    val_ds = CTHeatmapDataset(
        manifest_path=manifest_path,
        split="val",
        processed_dir=processed_dir,
        target_dir=target_dir,
        splits_path=splits_path,
        transform=val_transforms,
        target_type=dataset_target_type,
    )

    # Map keys from CTHeatmapDataset to Trainer expectations:
    # 'image' -> 'ct', 'heatmap_target' -> 'heatmap'
    class _MappedDataset(torch.utils.data.Dataset):
        def __init__(self, ds: CTHeatmapDataset) -> None:
            self.ds = ds

        def __len__(self) -> int:
            return len(self.ds)

        def __getitem__(self, idx: int) -> dict[str, Any] | list[dict[str, Any]]:
            sample = self.ds[idx]
            # Handle list of samples returned by RandCropByPosNegLabeld (due to num_samples=2)
            if isinstance(sample, list):
                mapped_list = []
                for s in sample:
                    mapped = {
                        "ct": s["image"],
                        "heatmap": s["heatmap_target"],
                        "lesion_mask": s["lesion_mask"],
                        "triage_label": torch.tensor([float(s["case_label"].get("high_risk", 0.0))]),
                        "case_id": s["case_id"],
                    }
                    mapped_list.append(mapped)
                return mapped_list
            else:
                return {
                    "ct": sample["image"],
                    "heatmap": sample["heatmap_target"],
                    "lesion_mask": sample["lesion_mask"],
                    "triage_label": torch.tensor([float(sample["case_label"].get("high_risk", 0.0))]),
                    "case_id": sample["case_id"],
                }

    # Custom collate function to handle lists of dicts returned by RandCropByPosNegLabeld
    def _collate_fn(batch):
        flat_batch = []
        for item in batch:
            if isinstance(item, list):
                flat_batch.extend(item)
            else:
                flat_batch.append(item)
        
        collated = {}
        for key in flat_batch[0].keys():
            if isinstance(flat_batch[0][key], torch.Tensor):
                collated[key] = torch.stack([x[key] for x in flat_batch])
            else:
                collated[key] = [x[key] for x in flat_batch]
        return collated

    train_loader = DataLoader(
        _MappedDataset(train_ds),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=_collate_fn,
    )

    val_loader = DataLoader(
        _MappedDataset(val_ds),
        batch_size=1,  # Set to 1 to support variable spatial sizes in validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=_collate_fn,
    )

    return train_loader, val_loader

# ====================================================================
# Loss builder
# ====================================================================
def _build_loss(cfg: Dict[str, Any]) -> CombinedLoss:
    """Build the combined task loss from config.

    Args:
        cfg: Loss config dict.

    Returns:
        :class:`CombinedLoss` instance.
    """
    return CombinedLoss(
        heatmap_loss_cfg=cfg.get("heatmap", {}),
        lesion_loss_cfg=cfg.get("lesion", {}),
        triage_loss_cfg=cfg.get("triage", {}),
        w_heatmap=float(cfg.get("w_heatmap", 1.0)),
        w_lesion=float(cfg.get("w_lesion", 1.0)),
        w_triage=float(cfg.get("w_triage", 0.5)),
    )


# ====================================================================
# Main
# ====================================================================
def main(args: argparse.Namespace | None = None) -> None:
    """Entry point for baseline training.

    Args:
        args: Parsed CLI arguments.  If *None*, ``sys.argv`` is parsed.
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Train Baseline UNet")
        parser.add_argument(
            "--config", type=str, required=True, help="Path to YAML config."
        )
        parser.add_argument(
            "--resume", type=str, default=None, help="Checkpoint path to resume from."
        )
        args = parser.parse_args()

    # ---- Load config ----
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # ---- Logging ----
    log_level = config.get("logging", {}).get("level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Config loaded from: %s", config_path)

    # ---- Validate config schema (fail fast) ----
    _validate_config(config)

    # ---- Print resolved config ----
    _print_resolved_config(config)

    # ---- Seed ----
    seed = int(config.get("seed", 42))
    _seed_everything(seed)

    # ---- Device ----
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s  |  CUDA available: %s", device, torch.cuda.is_available())

    # ---- Ensure model config specifies baseline_unet ----
    config.setdefault("model", {})
    config["model"].setdefault("name", "baseline_unet")

    # ---- Build components ----
    model = build_model(config)
    logger.info("Model: %s", type(model).__name__)

    train_loader, val_loader = _build_dataloaders(
        config.get("data", {}),
        training_cfg=config.get("training", {}),
    )

    optimizer = _build_optimizer(model, config.get("optimizer", {}))
    num_epochs = int(config.get("training", {}).get("num_epochs", 100))
    scheduler = _build_scheduler(
        optimizer, config.get("scheduler", {}), num_epochs
    )
    loss_fn = _build_loss(config.get("loss", {}))

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # ---- Resume ----
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    # ---- Train ----
    results = trainer.fit()

    # ---- Save final metrics ----
    metrics_dir = config.get("paths", {}).get("metrics_dir", "results")
    metrics_path = Path(metrics_dir) / "baseline_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_loss": results["best_val_loss"],
                "total_epochs": results["total_epochs"],
            },
            f,
            indent=2,
        )
    logger.info("Final metrics saved to: %s", metrics_path)
    logger.info("Training complete.  Best val loss: %.6f", results["best_val_loss"])


if __name__ == "__main__":
    main()
