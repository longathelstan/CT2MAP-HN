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
from torch.utils.data.distributed import DistributedSampler

from src.models import build_model
from src.models.losses import CombinedLoss
from src.train.engine import Trainer, setup_ddp, cleanup_ddp, is_ddp_active, is_main_process

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
    use_distributed: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders using real HECKTOR data.

    Args:
        cfg: Data config dict from swin_unetr.yaml (typically config.get("data", {})).
        use_distributed: If True, use DistributedSampler for training.

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

    batch_size = int(cfg.get("batch_size", 1))
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

    # Build samplers
    train_dataset = _MappedDataset(train_ds)
    val_dataset = _MappedDataset(val_ds)

    if use_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False  # sampler handles shuffling
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Set to 1 to support variable spatial sizes in validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=_collate_fn,
    )

    return train_loader, val_loader


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
        parser.add_argument(
            "--local_rank", type=int, default=-1,
            help="Local rank for DDP (set automatically by torchrun).",
        )
        args = parser.parse_args()

    # ---- DDP setup ----
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    use_distributed = local_rank >= 0

    if use_distributed:
        local_rank = setup_ddp(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load config ----
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # ---- Logging (only rank 0) ----
    log_level = config.get("logging", {}).get("level", "INFO")
    if is_main_process():
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    # ---- Seed ----
    seed = int(config.get("seed", 42))
    _seed_everything(seed)

    # ---- Force model name ----
    config.setdefault("model", {})
    config["model"]["name"] = "swin_unetr_student"
    # Ensure gradient checkpointing is on
    config["model"].setdefault("use_checkpoint", True)

    # ---- Build ----
    model = build_model(config)
    if is_main_process():
        logger.info("Model: %s  |  Device: %s  |  DDP: %s", type(model).__name__, device, use_distributed)

    train_loader, val_loader = _build_dataloaders(
        config.get("data", {}),
        use_distributed=use_distributed,
    )
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
        local_rank=local_rank,
    )

    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    try:
        results = trainer.fit()

        # ---- Save metrics (rank 0 only) ----
        if is_main_process():
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
    finally:
        if use_distributed:
            cleanup_ddp()


if __name__ == "__main__":
    main()
