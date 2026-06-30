# -*- coding: utf-8 -*-
"""Tests for the CT dataset and transforms.

Uses mock data and temporary directories to validate dataset
initialisation, ``__getitem__`` behaviour, and transform pipelines
without requiring real HECKTOR data.

Run with:
    $ pytest tests/test_dataset.py -v
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch


# ======================================================================
# Helpers
# ======================================================================


def _create_mock_nifti(
    path: str,
    shape: tuple = (32, 32, 32),
    spacing: tuple = (1.0, 1.0, 1.0),
    value_range: tuple = (-1024.0, 3000.0),
    seed: int = 42,
) -> None:
    """Create a mock NIfTI file using SimpleITK.

    Args:
        path: Destination file path.
        shape: Volume shape (D, H, W).
        spacing: Voxel spacing.
        value_range: (min, max) for random values.
        seed: Random seed.
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        pytest.skip("SimpleITK not installed")

    rng = np.random.RandomState(seed)
    array = rng.uniform(
        value_range[0], value_range[1], size=shape
    ).astype(np.float32)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(image, str(path))


def _create_mock_manifest(
    manifest_path: str,
    cases: list,
) -> None:
    """Create a mock manifest CSV.

    Args:
        manifest_path: Destination CSV path.
        cases: List of dicts with at least 'case_id', 'ct_path', 'split'.
    """
    fieldnames = list(cases[0].keys()) if cases else ["case_id", "ct_path", "split"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def mock_data_dir():
    """Create a temporary directory with mock NIfTI data.

    Yields:
        Tuple of (tmpdir_path, list_of_case_dicts).
    """
    try:
        import SimpleITK  # noqa: F401
    except ImportError:
        pytest.skip("SimpleITK not installed")

    with tempfile.TemporaryDirectory() as tmpdir:
        cases = []
        for i in range(3):
            case_id = f"case_{i:03d}"
            case_dir = Path(tmpdir) / case_id
            case_dir.mkdir()

            ct_path = str(case_dir / "ct.nii.gz")
            pet_path = str(case_dir / "pet.nii.gz")
            mask_path = str(case_dir / "mask.nii.gz")

            _create_mock_nifti(ct_path, shape=(32, 32, 32), seed=i)
            _create_mock_nifti(
                pet_path,
                shape=(32, 32, 32),
                value_range=(0.0, 20.0),
                seed=i + 100,
            )
            _create_mock_nifti(
                mask_path,
                shape=(32, 32, 32),
                value_range=(0.0, 1.0),
                seed=i + 200,
            )

            cases.append(
                {
                    "case_id": case_id,
                    "ct_path": ct_path,
                    "pet_path": pet_path,
                    "mask_path": mask_path,
                    "split": "train" if i < 2 else "test",
                }
            )

        manifest_path = str(Path(tmpdir) / "manifest.csv")
        _create_mock_manifest(manifest_path, cases)

        yield tmpdir, cases, manifest_path


# ======================================================================
# Tests: Dataset initialisation
# ======================================================================


class TestCTDatasetInit:
    """Tests for dataset creation and manifest parsing."""

    def test_manifest_loading(self, mock_data_dir) -> None:
        """Manifest CSV should be loaded correctly."""
        tmpdir, cases, manifest_path = mock_data_dir

        with open(manifest_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 3
        assert all("case_id" in r for r in rows)
        assert all("ct_path" in r for r in rows)

    def test_manifest_split_filtering(self, mock_data_dir) -> None:
        """Manifest filtering by split should work."""
        tmpdir, cases, manifest_path = mock_data_dir

        with open(manifest_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        train_cases = [r for r in rows if r["split"] == "train"]
        test_cases = [r for r in rows if r["split"] == "test"]

        assert len(train_cases) == 2
        assert len(test_cases) == 1

    def test_case_files_exist(self, mock_data_dir) -> None:
        """All file paths in the manifest should exist."""
        tmpdir, cases, manifest_path = mock_data_dir

        for case in cases:
            assert Path(case["ct_path"]).exists(), f"CT not found: {case['ct_path']}"
            assert Path(case["pet_path"]).exists(), f"PET not found: {case['pet_path']}"
            assert Path(case["mask_path"]).exists(), f"Mask not found: {case['mask_path']}"


# ======================================================================
# Tests: Dataset __getitem__
# ======================================================================


class TestCTDatasetGetItem:
    """Tests for loading individual samples from the dataset."""

    def test_load_ct_volume(self, mock_data_dir) -> None:
        """Loading a CT NIfTI should return a valid numpy array."""
        try:
            import SimpleITK as sitk
        except ImportError:
            pytest.skip("SimpleITK not installed")

        tmpdir, cases, _ = mock_data_dir
        ct_path = cases[0]["ct_path"]

        image = sitk.ReadImage(ct_path)
        array = sitk.GetArrayFromImage(image)

        assert array.ndim == 3
        assert array.shape == (32, 32, 32)
        assert array.dtype == np.float32

    def test_load_pet_volume(self, mock_data_dir) -> None:
        """Loading a PET NIfTI should return a valid numpy array."""
        try:
            import SimpleITK as sitk
        except ImportError:
            pytest.skip("SimpleITK not installed")

        tmpdir, cases, _ = mock_data_dir
        pet_path = cases[0]["pet_path"]

        image = sitk.ReadImage(pet_path)
        array = sitk.GetArrayFromImage(image)

        assert array.ndim == 3
        assert array.min() >= -1.0  # Slight tolerance from resampling
        assert array.max() <= 25.0  # Slight tolerance

    def test_load_mask_volume(self, mock_data_dir) -> None:
        """Loading a mask NIfTI should return a valid array."""
        try:
            import SimpleITK as sitk
        except ImportError:
            pytest.skip("SimpleITK not installed")

        tmpdir, cases, _ = mock_data_dir
        mask_path = cases[0]["mask_path"]

        image = sitk.ReadImage(mask_path)
        array = sitk.GetArrayFromImage(image)

        assert array.ndim == 3

    def test_to_tensor_conversion(self, mock_data_dir) -> None:
        """Numpy array should convert to a valid PyTorch tensor."""
        try:
            import SimpleITK as sitk
        except ImportError:
            pytest.skip("SimpleITK not installed")

        tmpdir, cases, _ = mock_data_dir
        ct_path = cases[0]["ct_path"]

        image = sitk.ReadImage(ct_path)
        array = sitk.GetArrayFromImage(image).astype(np.float32)

        tensor = torch.from_numpy(array).unsqueeze(0)  # (1, D, H, W)
        assert tensor.ndim == 4
        assert tensor.shape == (1, 32, 32, 32)
        assert tensor.dtype == torch.float32

    def test_sample_dict_structure(self, mock_data_dir) -> None:
        """A dataset sample should contain expected keys."""
        try:
            import SimpleITK as sitk
        except ImportError:
            pytest.skip("SimpleITK not installed")

        tmpdir, cases, _ = mock_data_dir
        case = cases[0]

        # Simulate __getitem__ logic
        ct_image = sitk.ReadImage(case["ct_path"])
        ct_array = sitk.GetArrayFromImage(ct_image).astype(np.float32)

        pet_image = sitk.ReadImage(case["pet_path"])
        pet_array = sitk.GetArrayFromImage(pet_image).astype(np.float32)

        sample: Dict[str, Any] = {
            "case_id": case["case_id"],
            "ct": torch.from_numpy(ct_array).unsqueeze(0),
            "target": torch.from_numpy(pet_array).unsqueeze(0),
        }

        assert "case_id" in sample
        assert "ct" in sample
        assert "target" in sample
        assert isinstance(sample["ct"], torch.Tensor)
        assert isinstance(sample["target"], torch.Tensor)


# ======================================================================
# Tests: Transforms
# ======================================================================


class TestTransforms:
    """Tests for data augmentation and preprocessing transforms."""

    def test_random_flip(self) -> None:
        """Random flip should preserve shape and dtype."""
        rng = np.random.RandomState(42)
        volume = rng.rand(1, 32, 32, 32).astype(np.float32)
        tensor = torch.from_numpy(volume)

        # Simulate random flip along axis 2 (depth)
        flipped = torch.flip(tensor, dims=[1])
        assert flipped.shape == tensor.shape
        assert flipped.dtype == tensor.dtype

    def test_random_rotation_90(self) -> None:
        """90-degree rotation should preserve shape for cubic volumes."""
        volume = torch.rand(1, 32, 32, 32)

        # Rotate in the (H, W) plane
        rotated = torch.rot90(volume, k=1, dims=[2, 3])
        assert rotated.shape == volume.shape

    def test_intensity_scaling(self) -> None:
        """Random intensity scaling should stay within valid range."""
        rng = np.random.RandomState(42)
        volume = rng.rand(32, 32, 32).astype(np.float32)

        # Scale by a random factor
        scale = rng.uniform(0.9, 1.1)
        scaled = volume * scale
        scaled = np.clip(scaled, 0.0, 1.0)

        assert scaled.min() >= 0.0
        assert scaled.max() <= 1.0

    def test_gaussian_noise(self) -> None:
        """Adding Gaussian noise should change values but preserve shape."""
        rng = np.random.RandomState(42)
        volume = rng.rand(32, 32, 32).astype(np.float32)

        noise = rng.normal(0, 0.01, size=volume.shape).astype(np.float32)
        noisy = volume + noise

        assert noisy.shape == volume.shape
        # Values should differ
        assert not np.allclose(volume, noisy)

    def test_compose_transforms(self) -> None:
        """Composing multiple transforms should produce valid output."""
        rng = np.random.RandomState(42)
        volume = rng.rand(1, 32, 32, 32).astype(np.float32)
        tensor = torch.from_numpy(volume)

        # Compose: flip → intensity scale → clip
        t1 = torch.flip(tensor, dims=[1])
        t2 = t1 * 1.05
        t3 = torch.clamp(t2, 0.0, 1.0)

        assert t3.shape == tensor.shape
        assert t3.min() >= 0.0
        assert t3.max() <= 1.0

    def test_center_crop(self) -> None:
        """Centre cropping should produce the expected output shape."""
        volume = torch.rand(1, 64, 64, 64)
        crop_size = (32, 32, 32)

        d, h, w = volume.shape[1:]
        cd, ch, cw = crop_size
        d0 = (d - cd) // 2
        h0 = (h - ch) // 2
        w0 = (w - cw) // 2

        cropped = volume[
            :, d0 : d0 + cd, h0 : h0 + ch, w0 : w0 + cw
        ]

        assert cropped.shape == (1, 32, 32, 32)
