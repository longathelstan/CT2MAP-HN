# -*- coding: utf-8 -*-
"""I/O utilities for checkpoints, metrics, and file management.

Provides helpers for saving / loading PyTorch checkpoints, recording
training metrics in JSON or CSV, and copying configuration files to
output directories for reproducibility.

Example:
    >>> from src.utils.io import save_checkpoint, load_checkpoint
    >>> save_checkpoint(model.state_dict(), optimizer, epoch=10,
    ...                 metrics={"val_loss": 0.42}, path="ckpt/best.pt")
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> Path:
    """Create a directory (and parents) if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved ``Path`` object.

    Example:
        >>> ensure_dir("outputs/predictions/case_001")
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_checkpoint(
    state_dict: dict,
    optimizer: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    *,
    scheduler: Optional[Any] = None,
    extra: Optional[dict] = None,
) -> None:
    """Save a training checkpoint.

    The checkpoint dict contains:
        - ``model_state_dict``
        - ``optimizer_state_dict`` (if optimizer is not None)
        - ``scheduler_state_dict`` (if scheduler is not None)
        - ``epoch``
        - ``metrics``
        - ``timestamp`` (ISO-8601)
        - any keys supplied via *extra*

    Args:
        state_dict: Model ``state_dict()``.
        optimizer: PyTorch optimizer (or ``None``).
        epoch: Current epoch number.
        metrics: Dictionary of metric names → values.
        path: Destination file path (e.g. ``"ckpt/best.pt"``).
        scheduler: Optional LR scheduler.
        extra: Optional dict of additional metadata.

    Example:
        >>> save_checkpoint(model.state_dict(), optimizer, 10,
        ...                 {"val_loss": 0.42}, "ckpt/best.pt")
    """
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "model_state_dict": state_dict,
        "epoch": epoch,
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, ckpt_path)
    logger.info(
        "Saved checkpoint to %s (epoch=%d, metrics=%s)",
        ckpt_path,
        epoch,
        metrics,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    scheduler: Optional[Any] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> dict:
    """Load a training checkpoint and restore model / optimizer state.

    Args:
        path: Path to the ``.pt`` checkpoint file.
        model: PyTorch model whose ``state_dict`` will be loaded.
        optimizer: Optional optimizer to restore. If ``None``, optimizer
            state in the checkpoint is ignored.
        scheduler: Optional LR scheduler to restore.
        map_location: Passed to ``torch.load`` (e.g. ``"cpu"``).
        strict: Passed to ``model.load_state_dict``.

    Returns:
        A dict with metadata from the checkpoint:
        ``{"epoch": int, "metrics": dict, "timestamp": str, ...}``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If the checkpoint is missing ``model_state_dict``.

    Example:
        >>> meta = load_checkpoint("ckpt/best.pt", model, optimizer)
        >>> print(meta["epoch"], meta["metrics"])
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(
        ckpt_path,
        map_location=map_location,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint at {ckpt_path} is missing 'model_state_dict' key. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    logger.info("Loaded model weights from %s", ckpt_path)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Restored optimizer state from %s", ckpt_path)

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Restored scheduler state from %s", ckpt_path)

    meta = {
        k: v
        for k, v in checkpoint.items()
        if k
        not in {
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
        }
    }
    return meta


def save_metrics(
    metrics: Union[Dict[str, Any], List[Dict[str, Any]]],
    path: str,
) -> None:
    """Save metrics to JSON or CSV, determined by file extension.

    Supported extensions:
        - ``.json``: writes pretty-printed JSON.
        - ``.csv``: writes a CSV with header (one row per dict if *metrics*
          is a list, otherwise a single row).

    Args:
        metrics: Dictionary (single snapshot) or list of dicts
            (e.g. epoch-level logs).
        path: Destination file path. Extension determines format.

    Raises:
        ValueError: If the extension is not ``.json`` or ``.csv``.

    Example:
        >>> save_metrics({"epoch": 1, "loss": 0.5}, "logs/metrics.json")
        >>> save_metrics([{"epoch": 1}, {"epoch": 2}], "logs/history.csv")
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()

    if ext == ".json":
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=False, default=str)
        logger.info("Saved metrics (JSON) to %s", out_path)

    elif ext == ".csv":
        rows: List[dict] = (
            metrics if isinstance(metrics, list) else [metrics]
        )
        if not rows:
            logger.warning("Empty metrics list; nothing written to %s", out_path)
            return
        fieldnames = list(rows[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved metrics (CSV, %d rows) to %s", len(rows), out_path)

    else:
        raise ValueError(
            f"Unsupported metrics format '{ext}'. Use .json or .csv."
        )


def load_metrics(path: str) -> Union[dict, list]:
    """Load metrics from a JSON or CSV file.

    Args:
        path: Source file path (``.json`` or ``.csv``).

    Returns:
        Parsed metrics. JSON files return a dict or list; CSV files always
        return a list of row-dicts.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the extension is not ``.json`` or ``.csv``.

    Example:
        >>> metrics = load_metrics("logs/metrics.json")
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {in_path}")

    ext = in_path.suffix.lower()

    if ext == ".json":
        with open(in_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Loaded metrics (JSON) from %s", in_path)
        return data

    elif ext == ".csv":
        with open(in_path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        logger.info("Loaded metrics (CSV, %d rows) from %s", len(rows), in_path)
        return rows

    else:
        raise ValueError(
            f"Unsupported metrics format '{ext}'. Use .json or .csv."
        )


def copy_config_to_output(config_path: str, output_dir: str) -> Path:
    """Copy a config file into an output directory for reproducibility.

    A timestamp suffix is appended to avoid overwriting existing copies.

    Args:
        config_path: Path to the configuration file to copy.
        output_dir: Destination directory.

    Returns:
        Path to the copied file.

    Raises:
        FileNotFoundError: If *config_path* does not exist.

    Example:
        >>> copy_config_to_output("configs/train.yaml", "outputs/exp01")
    """
    src = Path(config_path)
    if not src.exists():
        raise FileNotFoundError(f"Config file not found: {src}")

    dst_dir = ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = dst_dir / f"{src.stem}_{timestamp}{src.suffix}"

    shutil.copy2(src, dst)
    logger.info("Copied config %s → %s", src, dst)
    return dst
