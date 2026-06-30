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


def _build_dataloaders(
    cfg: Dict[str, Any],
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders.

    This is a **placeholder** that creates synthetic dummy data.  Replace the
    body with your actual dataset loading logic (e.g., HECKTOR dataset).

    Args:
        cfg: Data config dict.

    Returns:
        Tuple of ``(train_loader, val_loader)``.
    """
    from torch.utils.data import TensorDataset

    batch_size = int(cfg.get("batch_size", 2))
    patch_size = tuple(cfg.get("patch_size", [96, 96, 96]))
    num_train = int(cfg.get("num_train_samples", 20))
    num_val = int(cfg.get("num_val_samples", 5))
    num_workers = int(cfg.get("num_workers", 4))

    logger.warning(
        "Using SYNTHETIC data (%d train, %d val) – replace with real dataset.",
        num_train, num_val,
    )

    def _make_loader(n: int, shuffle: bool) -> DataLoader:
        ct = torch.randn(n, 1, *patch_size)
        heatmap = torch.rand(n, 1, *patch_size)
        lesion = (torch.rand(n, 1, *patch_size) > 0.9).float()
        triage = (torch.rand(n, 1) > 0.5).float()
        ds = TensorDataset(ct, heatmap, lesion, triage)

        # Wrap to return dicts
        class _DictDataset(torch.utils.data.Dataset):
            def __init__(self, tensor_ds: TensorDataset) -> None:
                self._ds = tensor_ds

            def __len__(self) -> int:
                return len(self._ds)

            def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
                ct_t, hm, les, tri = self._ds[idx]
                return {
                    "ct": ct_t,
                    "heatmap": hm,
                    "lesion_mask": les,
                    "triage_label": tri,
                }

        return DataLoader(
            _DictDataset(ds),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=shuffle,
        )

    return _make_loader(num_train, True), _make_loader(num_val, False)


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

    # ---- Seed ----
    seed = int(config.get("seed", 42))
    _seed_everything(seed)

    # ---- Device ----
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s  |  CUDA available: %s", device, torch.cuda.is_available())

    # ---- Ensure model config specifies baseline_unet ----
    config.setdefault("model", {})
    config["model"]["name"] = "baseline_unet"

    # ---- Build components ----
    model = build_model(config)
    logger.info("Model: %s", type(model).__name__)

    train_loader, val_loader = _build_dataloaders(config.get("data", {}))

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
    metrics_path = Path(
        config.get("paths", {}).get("metrics_dir", "results")
    ) / "baseline_metrics.json"
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
