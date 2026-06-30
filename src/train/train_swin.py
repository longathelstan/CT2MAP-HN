"""
CT2MAP-HN – Swin UNETR training script.

Usage::

    python -m src.train.train_swin --config configs/swin.yaml

Same overall structure as :mod:`train_baseline` but instantiates
:class:`SwinUNETRStudent` with gradient-checkpointing enabled by default,
and uses a larger default patch size tuned for 48 GB A6000 GPUs.
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

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
        model: Model to optimise.
        cfg: Optimizer config.

    Returns:
        Configured optimizer.
    """
    lr = float(cfg.get("lr", 2e-4))
    wd = float(cfg.get("weight_decay", 1e-5))
    return AdamW(model.parameters(), lr=lr, weight_decay=wd)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    num_epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build cosine-annealing scheduler (default for Swin).

    Args:
        optimizer: Optimizer to schedule.
        cfg: Scheduler config.
        num_epochs: Total epochs.

    Returns:
        CosineAnnealingLR scheduler.
    """
    warmup_epochs = int(cfg.get("warmup_epochs", 5))
    eta_min = float(cfg.get("eta_min", 1e-7))
    return CosineAnnealingLR(optimizer, T_max=max(num_epochs - warmup_epochs, 1), eta_min=eta_min)


def _build_loss(cfg: Dict[str, Any]) -> CombinedLoss:
    """Build combined loss function.

    Args:
        cfg: Loss config dict.

    Returns:
        CombinedLoss instance.
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
    """Build train / val DataLoaders.

    Placeholder using synthetic data.  Replace with real HECKTOR loading.

    Args:
        cfg: Data config dict.

    Returns:
        ``(train_loader, val_loader)`` tuple.
    """
    from torch.utils.data import TensorDataset

    batch_size = int(cfg.get("batch_size", 1))  # Swin needs more memory → smaller batch
    patch_size = tuple(cfg.get("patch_size", [128, 128, 128]))  # Larger for A6000
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

        class _DictDataset(torch.utils.data.Dataset):
            def __init__(self, ct_t: torch.Tensor, hm: torch.Tensor,
                         les: torch.Tensor, tri: torch.Tensor) -> None:
                self.ct = ct_t
                self.hm = hm
                self.les = les
                self.tri = tri

            def __len__(self) -> int:
                return self.ct.size(0)

            def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
                return {
                    "ct": self.ct[idx],
                    "heatmap": self.hm[idx],
                    "lesion_mask": self.les[idx],
                    "triage_label": self.tri[idx],
                }

        return DataLoader(
            _DictDataset(ct, heatmap, lesion, triage),
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
    """Entry point for Swin UNETR training.

    Args:
        args: Parsed CLI arguments.  If *None*, ``sys.argv`` is parsed.
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Train Swin UNETR Student")
        parser.add_argument(
            "--config", type=str, required=True, help="Path to YAML config."
        )
        parser.add_argument(
            "--resume", type=str, default=None, help="Checkpoint to resume from."
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

    # ---- Seed ----
    seed = int(config.get("seed", 42))
    _seed_everything(seed)

    # ---- Device ----
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # ---- Force model name ----
    config.setdefault("model", {})
    config["model"]["name"] = "swin_unetr_student"
    # Ensure gradient checkpointing is on
    config["model"].setdefault("use_checkpoint", True)

    # ---- Build ----
    model = build_model(config)
    logger.info("Model: %s", type(model).__name__)

    train_loader, val_loader = _build_dataloaders(config.get("data", {}))
    optimizer = _build_optimizer(model, config.get("optimizer", {}))
    num_epochs = int(config.get("training", {}).get("num_epochs", 200))
    scheduler = _build_scheduler(optimizer, config.get("scheduler", {}), num_epochs)
    loss_fn = _build_loss(config.get("loss", {}))

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

    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    results = trainer.fit()

    # ---- Save metrics ----
    metrics_path = Path(
        config.get("paths", {}).get("metrics_dir", "results")
    ) / "swin_metrics.json"
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
