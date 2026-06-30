"""
CT2MAP-HN – Training engine.

Provides :class:`Trainer`, a reusable training loop with:

* Mixed-precision training (``torch.cuda.amp``).
* Gradient accumulation.
* Multi-GPU via ``DataParallel``.
* TensorBoard logging.
* Checkpoint saving (best + periodic).
* Early stopping.
* Resume from checkpoint.
* Rich/tqdm progress bars.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from tqdm.auto import tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

logger = logging.getLogger(__name__)


class Trainer:
    """Configurable training engine for CT2MAP-HN models.

    Args:
        model: The ``nn.Module`` to train.
        optimizer: PyTorch optimizer.
        scheduler: Optional learning-rate scheduler (stepped per **epoch**).
        loss_fn: Callable that returns a dict with at least ``"total"`` key.
        train_loader: Training :class:`DataLoader`.
        val_loader: Validation :class:`DataLoader`.
        config: Full experiment config dictionary.
        device: Target device (e.g. ``"cuda:0"``).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        self.config = config
        self.device = torch.device(device)

        # ---- Training hyper-params from config ----
        train_cfg = config.get("training", {})
        self.num_epochs: int = train_cfg.get("num_epochs", 100)
        self.grad_accum_steps: int = train_cfg.get("grad_accumulation_steps", 1)
        self.use_amp: bool = train_cfg.get("use_amp", True)
        self.clip_grad_norm: Optional[float] = train_cfg.get("clip_grad_norm", None)
        self.val_interval: int = train_cfg.get("val_interval", 1)
        self.save_interval: int = train_cfg.get("save_interval", 10)
        self.early_stop_patience: int = train_cfg.get("early_stop_patience", 20)

        # ---- Multi-GPU ----
        self.use_data_parallel: bool = train_cfg.get("use_data_parallel", False)
        if self.use_data_parallel and torch.cuda.device_count() > 1:
            logger.info(
                "Wrapping model in DataParallel (%d GPUs).",
                torch.cuda.device_count(),
            )
            model = nn.DataParallel(model)
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader

        # ---- AMP scaler ----
        self.scaler = GradScaler(enabled=self.use_amp)

        # ---- Checkpoint dir ----
        self.checkpoint_dir = Path(
            config.get("paths", {}).get("checkpoint_dir", "checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ---- TensorBoard ----
        tb_dir = config.get("paths", {}).get("tensorboard_dir", "runs")
        self.writer = SummaryWriter(log_dir=tb_dir)

        # ---- State tracking ----
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.best_val_loss: float = float("inf")
        self.epochs_without_improvement: int = 0
        self.train_history: List[Dict[str, float]] = []
        self.val_history: List[Dict[str, float]] = []

    # ==================================================================
    # Single epoch
    # ==================================================================
    def train_one_epoch(self) -> Dict[str, float]:
        """Run one training epoch with mixed precision and gradient accumulation.

        Returns:
            Dictionary of averaged training metrics for this epoch.
        """
        self.model.train()
        running: Dict[str, float] = {}
        n_batches = 0

        iterator = self.train_loader
        if _HAS_TQDM:
            iterator = tqdm(
                iterator,
                desc=f"Train Epoch {self.current_epoch}",
                leave=False,
            )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(iterator):
            # Move data to device
            inputs = batch["ct"].to(self.device, non_blocking=True)
            targets = self._move_targets(batch)

            # Forward with AMP
            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss_dict = self._compute_loss(outputs, targets)
                loss = loss_dict["total"] / self.grad_accum_steps

            # Backward
            self.scaler.scale(loss).backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Accumulate metrics
            for k, v in loss_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                running[k] = running.get(k, 0.0) + val
            n_batches += 1

            # Update progress bar
            if _HAS_TQDM and isinstance(iterator, tqdm):
                iterator.set_postfix(
                    loss=f"{loss_dict['total'].item():.4f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )

        # Average metrics
        avg_metrics = {k: v / max(n_batches, 1) for k, v in running.items()}

        # Log to TensorBoard
        for k, v in avg_metrics.items():
            self.writer.add_scalar(f"train/{k}", v, self.current_epoch)
        self.writer.add_scalar(
            "train/lr", self.optimizer.param_groups[0]["lr"], self.current_epoch
        )

        return avg_metrics

    # ==================================================================
    # Validation
    # ==================================================================
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation loop.

        Returns:
            Dictionary of averaged validation metrics.
        """
        self.model.eval()
        running: Dict[str, float] = {}
        n_batches = 0

        iterator = self.val_loader
        if _HAS_TQDM:
            iterator = tqdm(iterator, desc="Validate", leave=False)

        for batch in iterator:
            inputs = batch["ct"].to(self.device, non_blocking=True)
            targets = self._move_targets(batch)

            with autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss_dict = self._compute_loss(outputs, targets)

            for k, v in loss_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                running[k] = running.get(k, 0.0) + val
            n_batches += 1

        avg_metrics = {k: v / max(n_batches, 1) for k, v in running.items()}

        for k, v in avg_metrics.items():
            self.writer.add_scalar(f"val/{k}", v, self.current_epoch)

        return avg_metrics

    # ==================================================================
    # Full training loop
    # ==================================================================
    def fit(self, num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """Execute the full training loop.

        Args:
            num_epochs: Override the number of epochs from config.

        Returns:
            Dictionary with training summary: best_val_loss, total_epochs,
            and train/val history.
        """
        if num_epochs is not None:
            self.num_epochs = num_epochs

        logger.info(
            "Starting training: %d epochs, device=%s, AMP=%s, "
            "grad_accum=%d, early_stop=%d",
            self.num_epochs, self.device, self.use_amp,
            self.grad_accum_steps, self.early_stop_patience,
        )

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # ---- Train ----
            train_metrics = self.train_one_epoch()
            self.train_history.append(train_metrics)

            # ---- Validate ----
            if (epoch + 1) % self.val_interval == 0:
                val_metrics = self.validate()
                self.val_history.append(val_metrics)

                val_loss = val_metrics.get("total", float("inf"))

                # Best model tracking
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    self._save_checkpoint("best_model.pth", is_best=True)
                    logger.info(
                        "New best val loss: %.6f (epoch %d)", val_loss, epoch
                    )
                else:
                    self.epochs_without_improvement += 1

                # Early stopping
                if (
                    self.early_stop_patience > 0
                    and self.epochs_without_improvement >= self.early_stop_patience
                ):
                    logger.info(
                        "Early stopping triggered after %d epochs without "
                        "improvement.",
                        self.epochs_without_improvement,
                    )
                    break
            else:
                val_metrics = {}

            # ---- Scheduler step ----
            if self.scheduler is not None:
                self.scheduler.step()

            # ---- Periodic checkpoint ----
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch + 1:04d}.pth")

            # ---- Epoch summary ----
            elapsed = time.time() - epoch_start
            logger.info(
                "Epoch %d/%d  train_loss=%.5f  val_loss=%.5f  "
                "lr=%.2e  time=%.1fs",
                epoch + 1,
                self.num_epochs,
                train_metrics.get("total", 0.0),
                val_metrics.get("total", 0.0),
                self.optimizer.param_groups[0]["lr"],
                elapsed,
            )

        # Final checkpoint
        self._save_checkpoint("final_model.pth")
        self.writer.close()

        return {
            "best_val_loss": self.best_val_loss,
            "total_epochs": self.current_epoch + 1,
            "train_history": self.train_history,
            "val_history": self.val_history,
        }

    # ==================================================================
    # Checkpoint management
    # ==================================================================
    def _save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save a training checkpoint.

        Args:
            filename: Checkpoint filename.
            is_best: Whether this is the best model so far.
        """
        model_to_save = (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )
        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(state, path)
        logger.debug("Checkpoint saved: %s", path)

    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Resume training from a saved checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint ``.pth`` file.

        Raises:
            FileNotFoundError: If *checkpoint_path* does not exist.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        logger.info("Resuming from checkpoint: %s", checkpoint_path)
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        model_to_load = (
            self.model.module
            if isinstance(self.model, nn.DataParallel)
            else self.model
        )
        model_to_load.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scaler.load_state_dict(state["scaler_state_dict"])
        self.current_epoch = state["epoch"] + 1
        self.global_step = state["global_step"]
        self.best_val_loss = state.get("best_val_loss", float("inf"))

        if self.scheduler is not None and "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        logger.info(
            "Resumed: epoch=%d  global_step=%d  best_val_loss=%.6f",
            self.current_epoch, self.global_step, self.best_val_loss,
        )

    # ==================================================================
    # Internal helpers
    # ==================================================================
    def _move_targets(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Move target tensors from batch dict to the training device.

        Expects the batch dict to contain any subset of:
        ``"heatmap"``, ``"lesion_mask"``, ``"triage_label"``.

        Args:
            batch: Raw batch from DataLoader.

        Returns:
            Dictionary of target tensors on ``self.device``.
        """
        targets: Dict[str, torch.Tensor] = {}
        for key in ("heatmap", "lesion_mask", "triage_label"):
            if key in batch:
                targets[key] = batch[key].to(self.device, non_blocking=True)
        return targets

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute loss using the configured loss function.

        Handles the mapping from model output keys to loss function arguments.

        Args:
            outputs: Model output dictionary.
            targets: Target dictionary.

        Returns:
            Loss dictionary with at least a ``"total"`` key.
        """
        # Build kwargs matching CombinedLoss.forward signature
        kwargs: Dict[str, Any] = {}

        if "heatmap" in outputs and "heatmap" in targets:
            kwargs["pred_heatmap"] = outputs["heatmap"]
            kwargs["target_heatmap"] = targets["heatmap"]

        if "lesion" in outputs and "lesion_mask" in targets:
            kwargs["pred_lesion"] = outputs["lesion"]
            kwargs["target_lesion"] = targets["lesion_mask"]

        if "triage" in outputs and "triage_label" in targets:
            kwargs["pred_triage"] = outputs["triage"]
            kwargs["target_triage"] = targets["triage_label"]

        if "lesion_mask" in targets:
            kwargs["lesion_mask"] = targets["lesion_mask"]

        result = self.loss_fn(**kwargs)

        # If loss_fn returns a plain tensor, wrap it
        if isinstance(result, torch.Tensor):
            return {"total": result}
        return result
