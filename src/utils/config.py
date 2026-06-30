# -*- coding: utf-8 -*-
"""YAML-based configuration management with dot-notation access.

Provides helpers for loading, merging, and saving YAML configurations, plus
a ``Config`` wrapper that supports attribute-style access
(e.g. ``config.model.type``).

Example:
    >>> from src.utils.config import Config
    >>> cfg = Config("configs/base.yaml")
    >>> print(cfg.model.type)
    'swin_unetr'
    >>> print(cfg.get("training.lr", 1e-4))
    0.001
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file. Supports both ``/`` and ``\\``
            separators (cross-platform).

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If *path* does not exist.
        yaml.YAMLError: If the file contains invalid YAML.

    Example:
        >>> cfg = load_config("configs/base.yaml")
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if config is None:
        logger.warning("Config file %s is empty, returning empty dict", path)
        return {}

    logger.info("Loaded config from %s", config_path)
    return config


def merge_configs(base: dict, override: dict) -> dict:
    """Deep-merge *override* into *base* (recursively).

    Values in *override* take precedence.  Both dicts are left unmodified;
    a new dict is returned.

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary whose values take
            precedence over *base*.

    Returns:
        Merged configuration dictionary (deep copy).

    Example:
        >>> base = {"model": {"type": "unet", "channels": 32}}
        >>> override = {"model": {"type": "swin_unetr"}, "lr": 1e-3}
        >>> merged = merge_configs(base, override)
        >>> merged["model"]["type"]
        'swin_unetr'
        >>> merged["model"]["channels"]
        32
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def save_config(config: dict, path: str) -> None:
    """Save a configuration dictionary to a YAML file.

    Parent directories are created automatically if they do not exist.

    Args:
        config: Configuration dictionary to save.
        path: Destination file path.

    Example:
        >>> save_config({"model": {"type": "swin_unetr"}}, "out/config.yaml")
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.dump(
            config,
            fh,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    logger.info("Saved config to %s", out_path)


# ---------------------------------------------------------------------------
# Config class with dot-notation access
# ---------------------------------------------------------------------------


class Config:
    """Dot-notation wrapper around a configuration dictionary.

    Supports both ``Config(yaml_path)`` (load from file) and
    ``Config(dict)`` (wrap existing dict).

    Nested dicts are wrapped lazily so that ``config.model.type`` works.

    Args:
        source: Either a ``str`` / ``Path`` pointing to a YAML file, or a
            plain ``dict`` to wrap.

    Raises:
        TypeError: If *source* is neither a path-like nor a ``dict``.
        FileNotFoundError: If a file path is given but does not exist.

    Example:
        >>> cfg = Config({"model": {"type": "swin_unetr", "channels": 48}})
        >>> cfg.model.type
        'swin_unetr'
        >>> cfg.get("model.channels", 32)
        48
        >>> cfg.to_dict()
        {'model': {'type': 'swin_unetr', 'channels': 48}}
    """

    def __init__(self, source: Union[str, Path, dict]) -> None:
        if isinstance(source, (str, Path)):
            self._data: dict = load_config(str(source))
        elif isinstance(source, dict):
            self._data = copy.deepcopy(source)
        else:
            raise TypeError(
                f"Config source must be a path (str/Path) or dict, "
                f"got {type(source).__name__}"
            )

    # ----- attribute access -----

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(
                f"Config has no attribute '{name}'. "
                f"Available keys: {list(self._data.keys())}"
            )
        if isinstance(value, dict):
            return Config(value)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    # ----- dict-like helpers -----

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by (optionally dotted) key.

        Args:
            key: Key string, supporting nested lookup with dots.
                For example ``"model.type"`` drills into
                ``self._data["model"]["type"]``.
            default: Value returned when the key is not found.

        Returns:
            The looked-up value, or *default*.

        Example:
            >>> cfg.get("model.type", "unet")
            'swin_unetr'
        """
        parts = key.split(".")
        current: Any = self._data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def to_dict(self) -> dict:
        """Return a deep copy of the underlying data as a plain dict.

        Returns:
            dict: Deep copy of configuration data.
        """
        return copy.deepcopy(self._data)

    def keys(self):
        """Return top-level keys."""
        return self._data.keys()

    def items(self):
        """Return top-level (key, value) pairs."""
        return self._data.items()

    def update(self, other: Union[dict, "Config"]) -> None:
        """Deep-merge another dict or Config into this one.

        Args:
            other: Values to merge in (takes precedence).
        """
        if isinstance(other, Config):
            other = other.to_dict()
        self._data = merge_configs(self._data, other)
