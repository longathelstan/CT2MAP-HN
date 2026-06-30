# -*- coding: utf-8 -*-
"""PyTorch Dataset for CT heatmap prediction (student model).

Dataset chính cho CT2MAP-HN: đọc CT đã tiền xử lý và target heatmap,
trả về dict phù hợp với MONAI training loop.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CTHeatmapDataset(Dataset):
    """PyTorch Dataset for CT → metabolic heatmap prediction.

    Loads preprocessed CT volumes and their corresponding target heatmaps.
    Supports both pre-computed disk-based loading and on-the-fly preprocessing.

    Each sample is a dictionary compatible with MONAI transforms:
        - ``image``: CT volume tensor (1, D, H, W)
        - ``heatmap_target``: Heatmap target tensor (1, D, H, W)
        - ``lesion_mask``: Binary mask tensor (1, D, H, W) or None
        - ``case_label``: Case-level label dict
        - ``case_id``: String identifier
        - ``meta``: Metadata dict

    Args:
        manifest_path: Path to manifest CSV.
        split: Which split to use — 'train', 'val', or 'test'.
        processed_dir: Directory containing preprocessed CT volumes (.npy).
        target_dir: Directory containing target heatmaps (.npy).
        splits_path: Path to splits JSON (maps split name → case IDs).
        transform: Optional callable (e.g. MONAI Compose) applied to
            each sample dict.
        target_type: Which target to load — 'gaussian_heatmap', 'binary',
            or 'pet_derived'. Default 'gaussian_heatmap'.
        allow_missing_pet: If True, samples without PET are included.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        processed_dir: str | Path,
        target_dir: str | Path,
        splits_path: Optional[str | Path] = None,
        transform: Optional[Callable] = None,
        target_type: str = "gaussian_heatmap",
        allow_missing_pet: bool = True,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.processed_dir = Path(processed_dir)
        self.target_dir = Path(target_dir)
        self.transform = transform
        self.target_type = target_type
        self.allow_missing_pet = allow_missing_pet

        # Load manifest
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )
        self.manifest = pd.read_csv(self.manifest_path)

        # Filter by split
        if splits_path is not None:
            splits_path = Path(splits_path)
            if splits_path.exists():
                with open(splits_path, "r", encoding="utf-8") as f:
                    splits = json.load(f)
                if split not in splits:
                    raise ValueError(
                        f"Split '{split}' not found in {splits_path}. "
                        f"Available: {list(splits.keys())}"
                    )
                split_ids = set(splits[split])
                self.manifest = self.manifest[
                    self.manifest["case_id"].isin(split_ids)
                ].reset_index(drop=True)
            else:
                logger.warning(
                    "Splits file %s not found; using 'split' column "
                    "in manifest.", splits_path,
                )
                self.manifest = self.manifest[
                    self.manifest["split"] == split
                ].reset_index(drop=True)
        else:
            # Fallback: use 'split' column in manifest
            if "split" in self.manifest.columns:
                self.manifest = self.manifest[
                    self.manifest["split"] == split
                ].reset_index(drop=True)

        self.case_ids = self.manifest["case_id"].tolist()

        if len(self.case_ids) == 0:
            logger.warning(
                "No cases found for split '%s'. Check manifest and splits.",
                split,
            )

        logger.info(
            "CTHeatmapDataset initialized: split=%s, n_cases=%d, "
            "target_type=%s",
            split, len(self.case_ids), target_type,
        )

    def __len__(self) -> int:
        """Return the number of cases in this split."""
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load a single case and return a sample dictionary.

        Args:
            index: Sample index.

        Returns:
            Dictionary with 'image', 'heatmap_target', 'lesion_mask',
            'case_label', 'case_id', 'meta'.
        """
        case_id = self.case_ids[index]

        # Load preprocessed CT
        ct_path = self.processed_dir / f"{case_id}_ct.npy"
        ct_array = self._load_array(ct_path, case_id, "CT")

        # Load target heatmap
        target_path = self.target_dir / f"{case_id}_{self.target_type}.npy"
        target_array = self._load_array(target_path, case_id, "target")

        # Load lesion mask (optional)
        mask_path = self.target_dir / f"{case_id}_binary.npy"
        mask_array = self._load_array_optional(mask_path)

        # Load case label (optional)
        label_path = self.target_dir / f"{case_id}_label.json"
        case_label = self._load_label(label_path)

        # Build sample dict (MONAI-compatible with channel dim)
        sample = {
            "image": self._to_tensor(ct_array),
            "heatmap_target": self._to_tensor(target_array),
            "lesion_mask": (
                self._to_tensor(mask_array) if mask_array is not None
                else torch.zeros_like(self._to_tensor(ct_array))
            ),
            "case_label": case_label,
            "case_id": case_id,
            "meta": self._get_meta(case_id),
        }

        # Apply transforms
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _load_array(
        self, path: Path, case_id: str, name: str,
    ) -> np.ndarray:
        """Load a numpy array, raising informative error if missing."""
        if not path.exists():
            raise FileNotFoundError(
                f"{name} file not found for case '{case_id}': {path}"
            )
        return np.load(str(path)).astype(np.float32)

    def _load_array_optional(self, path: Path) -> Optional[np.ndarray]:
        """Load a numpy array if it exists, else return None."""
        if path.exists():
            return np.load(str(path)).astype(np.float32)
        return None

    def _load_label(self, path: Path) -> dict[str, Any]:
        """Load case label JSON if it exists."""
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert (D, H, W) array to (1, D, H, W) float tensor."""
        if array.ndim == 3:
            array = array[np.newaxis, ...]  # Add channel dim
        return torch.from_numpy(array.copy()).float()

    def _get_meta(self, case_id: str) -> dict[str, Any]:
        """Extract metadata for a case from the manifest."""
        row = self.manifest[self.manifest["case_id"] == case_id]
        if row.empty:
            return {"case_id": case_id}
        row = row.iloc[0]
        meta = {"case_id": case_id}
        for col in ["site", "spacing_x", "spacing_y", "spacing_z",
                     "has_primary", "has_nodes"]:
            if col in row.index:
                val = row[col]
                # Convert numpy types to Python native for JSON compatibility
                if hasattr(val, "item"):
                    val = val.item()
                meta[col] = val
        return meta
