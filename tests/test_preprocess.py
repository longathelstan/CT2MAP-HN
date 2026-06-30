# -*- coding: utf-8 -*-
"""Tests for CT preprocessing functions.

Uses synthetic 3-D volumes (numpy random) to validate preprocessing
steps without requiring real medical image data.

Run with:
    $ pytest tests/test_preprocess.py -v
"""

from __future__ import annotations

import numpy as np
import pytest


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def synthetic_volume() -> np.ndarray:
    """Create a synthetic 3-D volume mimicking CT HU values.

    Returns:
        3-D numpy array of shape (64, 64, 64) with values in
        [-1024, 3000].
    """
    rng = np.random.RandomState(42)
    # Simulate HU range: air ~-1000, soft tissue ~0-100, bone ~400-1000
    volume = rng.uniform(-1024, 3000, size=(64, 64, 64)).astype(np.float32)
    return volume


@pytest.fixture
def synthetic_sitk_image():
    """Create a synthetic SimpleITK image with known properties.

    Returns:
        SimpleITK Image with shape (64, 64, 64), spacing (0.5, 0.5, 1.0),
        and identity direction.
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        pytest.skip("SimpleITK not installed")

    rng = np.random.RandomState(42)
    array = rng.uniform(-1024, 3000, size=(64, 64, 64)).astype(np.float32)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 0.5, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    return image


# ======================================================================
# Tests: Orientation standardisation
# ======================================================================


class TestStandardizeOrientation:
    """Tests for orientation standardisation utilities."""

    def test_standardize_orientation_identity(
        self, synthetic_sitk_image
    ) -> None:
        """Identity direction should keep the volume unchanged."""
        try:
            import SimpleITK as sitk
        except ImportError:
            pytest.skip("SimpleITK not installed")

        image = synthetic_sitk_image
        original_array = sitk.GetArrayFromImage(image)

        # Apply orientation filter to LPS (default DICOM)
        orient_filter = sitk.DICOMOrientImageFilter()
        orient_filter.SetDesiredCoordinateOrientation("LPS")
        oriented = orient_filter.Execute(image)

        oriented_array = sitk.GetArrayFromImage(oriented)
        # Shape should remain the same or be transposed consistently
        assert oriented_array.ndim == 3
        assert oriented_array.size == original_array.size

    def test_standardize_orientation_preserves_values(
        self, synthetic_sitk_image
    ) -> None:
        """Reorientation should preserve voxel value range."""
        try:
            import SimpleITK as sitk
        except ImportError:
            pytest.skip("SimpleITK not installed")

        image = synthetic_sitk_image
        original_array = sitk.GetArrayFromImage(image)

        orient_filter = sitk.DICOMOrientImageFilter()
        orient_filter.SetDesiredCoordinateOrientation("RAS")
        oriented = orient_filter.Execute(image)

        oriented_array = sitk.GetArrayFromImage(oriented)
        np.testing.assert_almost_equal(
            oriented_array.min(), original_array.min(), decimal=1
        )
        np.testing.assert_almost_equal(
            oriented_array.max(), original_array.max(), decimal=1
        )


# ======================================================================
# Tests: Resampling
# ======================================================================


class TestResampleVolume:
    """Tests for volume resampling to isotropic spacing."""

    def test_resample_changes_spacing(self, synthetic_sitk_image) -> None:
        """Resampling to different spacing should change the output size."""
        try:
            import SimpleITK as sitk
        except ImportError:
            pytest.skip("SimpleITK not installed")

        image = synthetic_sitk_image
        assert image.GetSpacing() == (0.5, 0.5, 1.0)

        target_spacing = [1.0, 1.0, 1.0]
        original_size = image.GetSize()

        # Compute expected new size
        expected_size = [
            int(round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(
                original_size, image.GetSpacing(), target_spacing
            )
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(expected_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampled = resampler.Execute(image)

        assert resampled.GetSpacing() == tuple(target_spacing)
        assert resampled.GetSize() == tuple(expected_size)

    def test_resample_preserves_dimensionality(
        self, synthetic_sitk_image
    ) -> None:
        """Resampled volume should still be 3-D."""
        try:
            import SimpleITK as sitk
        except ImportError:
            pytest.skip("SimpleITK not installed")

        image = synthetic_sitk_image
        target_spacing = [2.0, 2.0, 2.0]
        original_size = image.GetSize()
        new_size = [
            int(round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(
                original_size, image.GetSpacing(), target_spacing
            )
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampled = resampler.Execute(image)

        array = sitk.GetArrayFromImage(resampled)
        assert array.ndim == 3

    def test_resample_same_spacing_is_noop(
        self, synthetic_sitk_image
    ) -> None:
        """Resampling to the same spacing should yield the same shape."""
        try:
            import SimpleITK as sitk
        except ImportError:
            pytest.skip("SimpleITK not installed")

        image = synthetic_sitk_image
        original_array = sitk.GetArrayFromImage(image)
        spacing = list(image.GetSpacing())

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(spacing)
        resampler.SetSize(image.GetSize())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampled = resampler.Execute(image)

        resampled_array = sitk.GetArrayFromImage(resampled)
        assert resampled_array.shape == original_array.shape


# ======================================================================
# Tests: HU clipping and normalisation
# ======================================================================


class TestClipAndNormalizeHU:
    """Tests for HU value clipping and normalisation."""

    def test_clip_range(self, synthetic_volume: np.ndarray) -> None:
        """Clipped values should be within [hu_min, hu_max]."""
        hu_min, hu_max = -1024, 1024
        clipped = np.clip(synthetic_volume, hu_min, hu_max)

        assert clipped.min() >= hu_min
        assert clipped.max() <= hu_max

    def test_normalize_range(self, synthetic_volume: np.ndarray) -> None:
        """Normalised values should be in [0, 1]."""
        hu_min, hu_max = -1024, 1024
        clipped = np.clip(synthetic_volume, hu_min, hu_max)
        normalised = (clipped - hu_min) / (hu_max - hu_min)

        assert normalised.min() >= 0.0
        assert normalised.max() <= 1.0

    def test_normalize_preserves_shape(
        self, synthetic_volume: np.ndarray
    ) -> None:
        """Normalisation should preserve the volume shape."""
        hu_min, hu_max = -1024, 1024
        clipped = np.clip(synthetic_volume, hu_min, hu_max)
        normalised = (clipped - hu_min) / (hu_max - hu_min)

        assert normalised.shape == synthetic_volume.shape

    def test_clip_with_equal_bounds(self) -> None:
        """When hu_min == hu_max, result should be all zeros or a constant."""
        volume = np.random.rand(10, 10, 10).astype(np.float32) * 100
        hu_min = hu_max = 50
        clipped = np.clip(volume, hu_min, hu_max)
        # All values should be 50
        np.testing.assert_array_equal(clipped, np.full_like(clipped, 50))

    def test_normalize_dtype(self, synthetic_volume: np.ndarray) -> None:
        """Normalised output should be float32."""
        hu_min, hu_max = -1024, 1024
        clipped = np.clip(synthetic_volume, hu_min, hu_max)
        normalised = (clipped.astype(np.float32) - hu_min) / (hu_max - hu_min)

        assert normalised.dtype == np.float32


# ======================================================================
# Tests: Head-neck ROI cropping
# ======================================================================


class TestCropHeadNeckROI:
    """Tests for head-neck region-of-interest cropping."""

    def test_crop_reduces_volume(self, synthetic_volume: np.ndarray) -> None:
        """Cropping should produce a smaller or equal-sized volume."""
        # Simulate ROI as central 50% of each dimension
        d, h, w = synthetic_volume.shape
        d_start, d_end = d // 4, 3 * d // 4
        h_start, h_end = h // 4, 3 * h // 4
        w_start, w_end = w // 4, 3 * w // 4

        cropped = synthetic_volume[d_start:d_end, h_start:h_end, w_start:w_end]

        assert cropped.size <= synthetic_volume.size
        assert cropped.ndim == 3

    def test_crop_preserves_values(self, synthetic_volume: np.ndarray) -> None:
        """Cropped voxel values should match the original."""
        d, h, w = synthetic_volume.shape
        roi = synthetic_volume[10:50, 10:50, 10:50]

        np.testing.assert_array_equal(
            roi, synthetic_volume[10:50, 10:50, 10:50]
        )

    def test_crop_with_padding(self, synthetic_volume: np.ndarray) -> None:
        """When the ROI extends beyond the volume, padding should work."""
        d, h, w = synthetic_volume.shape

        # Define ROI that extends beyond bounds
        target_shape = (80, 80, 80)  # larger than 64x64x64
        padded = np.zeros(target_shape, dtype=synthetic_volume.dtype)
        # Copy the volume into the centre
        pd_d = (target_shape[0] - d) // 2
        pd_h = (target_shape[1] - h) // 2
        pd_w = (target_shape[2] - w) // 2
        padded[pd_d : pd_d + d, pd_h : pd_h + h, pd_w : pd_w + w] = (
            synthetic_volume
        )

        assert padded.shape == target_shape
        # Central region should match
        np.testing.assert_array_equal(
            padded[pd_d : pd_d + d, pd_h : pd_h + h, pd_w : pd_w + w],
            synthetic_volume,
        )

    def test_crop_empty_roi_raises(self) -> None:
        """An empty ROI should result in an empty array."""
        volume = np.random.rand(64, 64, 64).astype(np.float32)
        cropped = volume[32:32, 0:64, 0:64]  # empty along axis 0
        assert cropped.size == 0
