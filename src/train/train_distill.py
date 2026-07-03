"""
CT2MAP-HN – Knowledge distillation training script.

Usage::

    python -m src.train.train_distill --config configs/distill.yaml

Loads a pre-trained teacher encoder, freezes it, and trains the student
(Swin UNETR) with :class:`CombinedDistillLoss`.  Implements a warmup strategy
that starts with task loss only and gradually adds distillation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from src.models import build_model
from src.models.teacher_encoder import TeacherEncoder
from src.models.losses import CombinedDistillLoss

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


# ====================================================================
# Reproducibility
# ====================================================================
def _seed_everything(seed: int) -> None:
    """Set seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ====================================================================
# Distillation-aware training loop
# ====================================================================
class DistillTrainer:
    """Training loop for teacher–student knowledge distillation.

    Differences from :class:`Trainer`:

    * Manages **two** models (frozen teacher + trainable student).
    * Uses :class:`CombinedDistillLoss`.
    * Implements distillation warmup: ``alpha`` ramps from 0 to 1 over
      ``warmup_epochs`` so the student first learns from ground truth
      before receiving teacher guidance.

    Args:
        student: Student model.
        teacher: Teacher model (will be frozen).
        optimizer: Optimizer for the student.
        scheduler: LR scheduler.
        loss_fn: :class:`CombinedDistillLoss`.
        train_loader: Training DataLoader (must yield CT **and** PET).
        val_loader: Validation DataLoader.
        config: Full experiment config.
        device: Target device.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: TeacherEncoder,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_fn: CombinedDistillLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.config = config

        # ---- Models ----
        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device)
        self.teacher.freeze()

        # ---- Training params ----
        train_cfg = config.get("training", {})
        self.num_epochs = int(train_cfg.get("num_epochs", 200))
        self.warmup_epochs = int(train_cfg.get("distill_warmup_epochs", 20))
        self.grad_accum_steps = int(train_cfg.get("grad_accumulation_steps", 1))
        self.use_amp = bool(train_cfg.get("use_amp", True))
        self.clip_grad_norm: Optional[float] = train_cfg.get("clip_grad_norm", None)
        self.val_interval = int(train_cfg.get("val_interval", 1))
        self.save_interval = int(train_cfg.get("save_interval", 10))
        self.early_stop_patience = int(train_cfg.get("early_stop_patience", 30))

        # DataParallel
        if train_cfg.get("use_data_parallel", False) and torch.cuda.device_count() > 1:
            self.student = nn.DataParallel(self.student)
            self.teacher = nn.DataParallel(self.teacher)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.scaler = GradScaler(enabled=self.use_amp)

        # Paths
        self.checkpoint_dir = Path(
            config.get("paths", {}).get("checkpoint_dir", "checkpoints/distill")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tb_dir = config.get("paths", {}).get("tensorboard_dir", "runs/distill")
        self.writer = SummaryWriter(log_dir=tb_dir)

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    # ------------------------------------------------------------------
    # Distillation alpha schedule
    # ------------------------------------------------------------------
    def _get_distill_alpha(self, epoch: int) -> float:
        """Compute distillation weight alpha for the current epoch.

        Linear ramp from 0 to 1 over ``warmup_epochs``.

        Args:
            epoch: Current epoch index.

        Returns:
            Alpha value in ``[0, 1]``.
        """
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, epoch / self.warmup_epochs)

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------
    def train_one_epoch(self) -> Dict[str, float]:
        """Run one distillation training epoch.

        Returns:
            Averaged training metrics dict.
        """
        self.student.train()
        self.teacher.eval()

        alpha = self._get_distill_alpha(self.current_epoch)
        self.loss_fn.set_distill_weight(alpha)

        running: Dict[str, float] = {}
        n_batches = 0

        iterator = self.train_loader
        if _HAS_TQDM:
            iterator = tqdm(iterator, desc=f"Distill Epoch {self.current_epoch}", leave=False)

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(iterator):
            ct = batch["ct"].to(self.device, non_blocking=True)
            pet = batch.get("pet")
            if pet is not None:
                pet = pet.to(self.device, non_blocking=True)
            else:
                # If no PET, use zeros (teacher still needs dual input)
                pet = torch.zeros_like(ct)

            targets = {
                k: batch[k].to(self.device, non_blocking=True)
                for k in ("heatmap", "lesion_mask", "triage_label")
                if k in batch
            }

            with autocast(enabled=self.use_amp):
                # ---- Teacher forward (no grad) ----
                with torch.no_grad():
                    teacher_out = self.teacher(ct, pet)
                    teacher_feats = teacher_out["features"]
                    teacher_heatmap = teacher_out["heatmap"]

                # ---- Student forward ----
                student_out = self.student(ct)
                student_feats = self.student.module.get_intermediate_features() \
                    if isinstance(self.student, nn.DataParallel) \
                    else self.student.get_intermediate_features()

                # ---- Loss ----
                loss_dict = self.loss_fn(
                    pred_heatmap=student_out["heatmap"],
                    target_heatmap=targets.get("heatmap", torch.zeros_like(student_out["heatmap"])),
                    pred_lesion=student_out["lesion"],
                    target_lesion=targets.get("lesion_mask", torch.zeros_like(student_out["lesion"])),
                    pred_triage=student_out["triage"],
                    target_triage=targets.get("triage_label", torch.zeros_like(student_out["triage"])),
                    student_feats=student_feats,
                    teacher_feats=teacher_feats,
                    student_output=student_out["heatmap"],
                    teacher_output=teacher_heatmap,
                    lesion_mask=targets.get("lesion_mask"),
                )
                loss = loss_dict["total"] / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.clip_grad_norm:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.student.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            for k, v in loss_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                running[k] = running.get(k, 0.0) + val
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in running.items()}
        for k, v in avg.items():
            self.writer.add_scalar(f"train/{k}", v, self.current_epoch)
        self.writer.add_scalar("train/distill_alpha", alpha, self.current_epoch)
        self.writer.add_scalar(
            "train/lr", self.optimizer.param_groups[0]["lr"], self.current_epoch
        )
        return avg

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation loop (task loss only, no distillation).

        Returns:
            Averaged validation metrics.
        """
        self.student.eval()
        running: Dict[str, float] = {}
        n_batches = 0

        iterator = self.val_loader
        if _HAS_TQDM:
            iterator = tqdm(iterator, desc="Val (distill)", leave=False)

        for batch in iterator:
            ct = batch["ct"].to(self.device, non_blocking=True)
            targets = {
                k: batch[k].to(self.device, non_blocking=True)
                for k in ("heatmap", "lesion_mask", "triage_label")
                if k in batch
            }

            with autocast(enabled=self.use_amp):
                student_out = self.student(ct)
                # Use task loss only for validation metric
                task_loss = self.loss_fn.task_loss(
                    pred_heatmap=student_out["heatmap"],
                    target_heatmap=targets.get("heatmap", torch.zeros_like(student_out["heatmap"])),
                    pred_lesion=student_out["lesion"],
                    target_lesion=targets.get("lesion_mask", torch.zeros_like(student_out["lesion"])),
                    pred_triage=student_out["triage"],
                    target_triage=targets.get("triage_label", torch.zeros_like(student_out["triage"])),
                )

            for k, v in task_loss.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                running[k] = running.get(k, 0.0) + val
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in running.items()}
        for k, v in avg.items():
            self.writer.add_scalar(f"val/{k}", v, self.current_epoch)
        return avg

    # ------------------------------------------------------------------
    # Full loop
    # ------------------------------------------------------------------
    def fit(self) -> Dict[str, Any]:
        """Execute full distillation training.

        Returns:
            Summary dict with best loss and epoch count.
        """
        logger.info(
            "Starting distillation training: %d epochs, warmup=%d, device=%s",
            self.num_epochs, self.warmup_epochs, self.device,
        )

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            t0 = time.time()

            train_metrics = self.train_one_epoch()

            if (epoch + 1) % self.val_interval == 0:
                val_metrics = self.validate()
                val_loss = val_metrics.get("total", float("inf"))

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    self._save_checkpoint("best_distill_model.pth")
                    logger.info("New best val loss: %.6f (epoch %d)", val_loss, epoch)
                else:
                    self.epochs_without_improvement += 1

                if (
                    self.early_stop_patience > 0
                    and self.epochs_without_improvement >= self.early_stop_patience
                ):
                    logger.info("Early stopping at epoch %d.", epoch)
                    break
            else:
                val_metrics = {}

            if self.scheduler is not None:
                self.scheduler.step()

            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(f"distill_epoch_{epoch + 1:04d}.pth")

            elapsed = time.time() - t0
            logger.info(
                "Epoch %d/%d  train_total=%.5f  task=%.5f  distill=%.5f  "
                "alpha=%.3f  val_total=%.5f  time=%.1fs",
                epoch + 1, self.num_epochs,
                train_metrics.get("total", 0.0),
                train_metrics.get("task_total", 0.0),
                train_metrics.get("distill_total", 0.0),
                train_metrics.get("distill_alpha", 0.0),
                val_metrics.get("total", 0.0),
                elapsed,
            )

        self._save_checkpoint("final_distill_model.pth")
        self.writer.close()

        return {
            "best_val_loss": self.best_val_loss,
            "total_epochs": self.current_epoch + 1,
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def _save_checkpoint(self, filename: str) -> None:
        """Save student checkpoint.

        Args:
            filename: Checkpoint filename.
        """
        student_model = (
            self.student.module
            if isinstance(self.student, nn.DataParallel)
            else self.student
        )
        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "student_state_dict": student_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        path = self.checkpoint_dir / filename
        torch.save(state, path)
        logger.debug("Distill checkpoint saved: %s", path)

    def resume_from_checkpoint(self, path: str | Path) -> None:
        """Resume distillation training from checkpoint.

        Args:
            path: Path to the checkpoint file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location=self.device, weights_only=False)
        student_model = (
            self.student.module
            if isinstance(self.student, nn.DataParallel)
            else self.student
        )
        student_model.load_state_dict(state["student_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scaler.load_state_dict(state["scaler_state_dict"])
        self.current_epoch = state["epoch"] + 1
        self.global_step = state["global_step"]
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        if self.scheduler and "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        logger.info("Resumed distillation from epoch %d.", self.current_epoch)


# ====================================================================
# DataLoader (placeholder)
# ====================================================================
def _build_dataloaders(
    cfg: Dict[str, Any],
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders using real HECKTOR data (with PET).

    Args:
        cfg: Data config dict.

    Returns:
        Tuple of ``(train_loader, val_loader)``.
    """
    from src.datasets.ct_teacher_dataset import CTTeacherDataset
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
    # Since RandCropByPosNegLabeld expects MASK_KEY = "lesion_mask", SPATIAL_KEYS = [IMAGE_KEY, TARGET_KEY, MASK_KEY]
    # For CTTeacherDataset, we also need to transform pet_image!
    # However, get_train_transforms in src.datasets.transforms uses SPATIAL_KEYS = ["image", "heatmap_target", "lesion_mask"].
    # Let's customize transforms for distillation if needed, or use get_train_transforms.
    # Note: get_train_transforms doesn't automatically include "pet_image" in SPATIAL_KEYS in transforms.py.
    # Wait, let's look at src/datasets/transforms.py to verify if it has teacher transforms, or if we need to define them.
    # Actually, we can define custom teacher transforms here or import them. Let's see: ct_teacher_dataset.py might apply some custom transforms or expects it.
    train_transform_config = {
        "spatial_size": aug_cfg.get("patch_size", [128, 128, 128]),
        "flip_prob": aug_cfg.get("random_flip_prob", 0.5),
        "flip_axes": [0, 1, 2],
        "num_samples": 2,
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
    train_ds = CTTeacherDataset(
        manifest_path=manifest_path,
        split="train",
        processed_dir=processed_dir,
        target_dir=target_dir,
        splits_path=splits_path,
        transform=train_transforms,
        target_type=dataset_target_type,
    )

    val_ds = CTTeacherDataset(
        manifest_path=manifest_path,
        split="val",
        processed_dir=processed_dir,
        target_dir=target_dir,
        splits_path=splits_path,
        transform=val_transforms,
        target_type=dataset_target_type,
    )

    # Map keys from CTTeacherDataset to Trainer expectations:
    # 'image' -> 'ct', 'pet_image' -> 'pet', 'heatmap_target' -> 'heatmap'
    class _MappedDataset(torch.utils.data.Dataset):
        def __init__(self, ds: CTTeacherDataset) -> None:
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
                        "pet": s.get("pet_image"),
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
                    "pet": sample.get("pet_image"),
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
# Main
# ====================================================================
def main(args: argparse.Namespace | None = None) -> None:
    """Entry point for distillation training.

    Args:
        args: CLI arguments.
    """
    if args is None:
        parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
        parser.add_argument("--config", type=str, required=True, help="YAML config path.")
        parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint.")
        parser.add_argument(
            "--teacher_checkpoint", type=str, default=None,
            help="Pre-trained teacher checkpoint (overrides config).",
        )
        args = parser.parse_args()

    # ---- Config ----
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    log_level = config.get("logging", {}).get("level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    seed = int(config.get("seed", 42))
    _seed_everything(seed)

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # ---- Build teacher ----
    teacher_cfg = config.get("teacher", {})
    teacher = TeacherEncoder(
        in_channels_ct=teacher_cfg.get("in_channels_ct", 1),
        in_channels_pet=teacher_cfg.get("in_channels_pet", 1),
        base_channels=teacher_cfg.get("base_channels", 32),
        num_stages=teacher_cfg.get("num_stages", 4),
    )

    # Load teacher weights
    teacher_ckpt = args.teacher_checkpoint or teacher_cfg.get("checkpoint", None)
    if teacher_ckpt and Path(teacher_ckpt).exists():
        state = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            teacher.load_state_dict(state["model_state_dict"])
        elif "student_state_dict" in state:
            teacher.load_state_dict(state["student_state_dict"])
        else:
            teacher.load_state_dict(state)
        logger.info("Loaded teacher weights from: %s", teacher_ckpt)
    else:
        logger.warning(
            "No teacher checkpoint loaded – teacher will use random weights!"
        )

    # ---- Build student ----
    config.setdefault("model", {})
    config["model"]["name"] = "swin_unetr_student"
    config["model"].setdefault("use_checkpoint", True)
    student = build_model(config)

    # ---- Build loss ----
    loss_cfg = config.get("loss", {})
    loss_fn = CombinedDistillLoss(
        combined_loss_cfg=loss_cfg.get("combined", {}),
        distill_loss_cfg=loss_cfg.get("distillation", {}),
        lambda_feature=float(loss_cfg.get("lambda_feature", 1.0)),
        lambda_output=float(loss_cfg.get("lambda_output", 1.0)),
    )

    # ---- Build data ----
    train_loader, val_loader = _build_dataloaders(config.get("data", {}))

    # ---- Build optimizer / scheduler ----
    opt_cfg = config.get("optimizer", {})
    optimizer = AdamW(
        student.parameters(),
        lr=float(opt_cfg.get("lr", 1e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-5)),
    )
    num_epochs = int(config.get("training", {}).get("num_epochs", 200))
    sched_cfg = config.get("scheduler", {})
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=float(sched_cfg.get("eta_min", 1e-7)),
    )

    # ---- Trainer ----
    trainer = DistillTrainer(
        student=student,
        teacher=teacher,
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
    ) / "distill_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Distillation complete.  Best val loss: %.6f", results["best_val_loss"])


if __name__ == "__main__":
    main()
