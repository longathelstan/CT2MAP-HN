# -*- coding: utf-8 -*-
"""Centralized logging configuration for CT2MAP-HN.

Sets up loggers with both console (optionally Rich-formatted) and file
handlers.  Calling ``setup_logger`` multiple times with the same *name*
returns the same logger without adding duplicate handlers.

Example:
    >>> from src.utils.logger import setup_logger
    >>> logger = setup_logger("training", log_file="outputs/train.log")
    >>> logger.info("Training started")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

# Cache to avoid duplicate handler attachment
_CONFIGURED_LOGGERS: dict[str, logging.Logger] = {}

# Default log format
_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
)
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _level_from_string(level: Union[str, int]) -> int:
    """Convert a level name string to a logging level int.

    Args:
        level: Level name (``'DEBUG'``, ``'INFO'``, …) or integer.

    Returns:
        Corresponding ``logging`` level constant.
    """
    if isinstance(level, int):
        return level
    return getattr(logging, level.upper(), logging.INFO)


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: Union[str, int] = "INFO",
    *,
    propagate: bool = False,
) -> logging.Logger:
    """Create or retrieve a configured logger.

    On the first call for a given *name*, handlers are attached:

    1. **Console handler** – uses ``rich.logging.RichHandler`` if the
       ``rich`` package is installed, otherwise a plain
       ``StreamHandler(sys.stdout)``.
    2. **File handler** (optional) – if *log_file* is provided, a
       ``FileHandler`` is added with UTF-8 encoding.

    Subsequent calls with the same *name* return the existing logger
    without adding more handlers (idempotent).

    Args:
        name: Logger name, typically the module path
            (e.g. ``"ct2map.training"``).
        log_file: Optional path to a log file.  Parent directories are
            created automatically.
        level: Logging level as a string (``'DEBUG'``, ``'INFO'``, …)
            or ``int``.
        propagate: Whether to propagate to parent loggers.

    Returns:
        Configured ``logging.Logger`` instance.

    Example:
        >>> log = setup_logger("train", "outputs/train.log", "DEBUG")
        >>> log.debug("Detailed message")
    """
    # Return cached logger if already configured
    if name in _CONFIGURED_LOGGERS:
        return _CONFIGURED_LOGGERS[name]

    log_level = _level_from_string(level)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = propagate

    # --- Console handler ---
    try:
        from rich.logging import RichHandler  # type: ignore[import-untyped]

        console_handler = RichHandler(
            level=log_level,
            rich_tracebacks=True,
            show_time=True,
            show_path=True,
        )
        # RichHandler supplies its own formatting; keep message-only
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    except ImportError:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(
            logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)
        )

    logger.addHandler(console_handler)

    # --- File handler ---
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_path, mode="a", encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)
        )
        logger.addHandler(file_handler)

    _CONFIGURED_LOGGERS[name] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """Retrieve an existing logger by name.

    If the logger has not been set up via ``setup_logger`` yet, this
    function returns a basic ``logging.getLogger(name)`` so that callers
    never receive ``None``.

    Args:
        name: Logger name.

    Returns:
        ``logging.Logger`` instance.

    Example:
        >>> log = get_logger("train")
    """
    if name in _CONFIGURED_LOGGERS:
        return _CONFIGURED_LOGGERS[name]
    return logging.getLogger(name)
