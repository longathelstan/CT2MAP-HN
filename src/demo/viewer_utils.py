# -*- coding: utf-8 -*-
"""Visualisation utilities for CT2MAP-HN demo.

Functions for extracting 2-D slices, overlaying heatmaps, creating
montages and multi-view displays, gauge / bar charts, and annotating
suspicious regions on CT images.

Example:
    >>> from src.demo.viewer_utils import create_3view, plot_triage_gauge
    >>> views = create_3view(ct_volume, heatmap)
    >>> fig = plot_triage_gauge(0.72)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# Slice extraction
# ======================================================================


def create_slice_viewer(
    volume_3d: np.ndarray,
    axis: str = "axial",
    slice_idx: Optional[int] = None,
) -> np.ndarray:
    """Extract a 2-D slice from a 3-D volume.

    The volume is expected in (D, H, W) order (depth-first / z-first).

    Args:
        volume_3d: 3-D numpy array.
        axis: Anatomical axis — ``'axial'`` (z), ``'coronal'`` (y),
            or ``'sagittal'`` (x).
        slice_idx: Index along the chosen axis.  If ``None``, the
            middle slice is used.

    Returns:
        2-D numpy array representing the selected slice.

    Raises:
        ValueError: If *axis* is not one of the supported values.

    Example:
        >>> slc = create_slice_viewer(volume, axis='coronal', slice_idx=128)
    """
    axis = axis.lower()
    axis_map = {"axial": 0, "coronal": 1, "sagittal": 2}

    if axis not in axis_map:
        raise ValueError(
            f"Unsupported axis '{axis}'. Choose from {list(axis_map.keys())}."
        )

    ax = axis_map[axis]
    n = volume_3d.shape[ax]

    if slice_idx is None:
        slice_idx = n // 2
    slice_idx = int(np.clip(slice_idx, 0, n - 1))

    if ax == 0:
        return volume_3d[slice_idx, :, :]
    elif ax == 1:
        return volume_3d[:, slice_idx, :]
    else:
        return volume_3d[:, :, slice_idx]


# ======================================================================
# Overlay
# ======================================================================


def create_overlay(
    ct_slice: np.ndarray,
    heatmap_slice: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay a heatmap on a CT slice.

    Args:
        ct_slice: 2-D grayscale CT slice.
        heatmap_slice: 2-D heatmap with values in [0, 1].
        alpha: Blending factor (0 = CT only, 1 = heatmap only).
        colormap: Matplotlib colormap name.

    Returns:
        RGB image (H×W×3) as ``np.uint8``.

    Example:
        >>> overlay = create_overlay(ct_axial, heatmap_axial, alpha=0.5)
    """
    import matplotlib.pyplot as plt

    # Normalise CT to [0, 1]
    ct_float = ct_slice.astype(np.float32)
    ct_min, ct_max = ct_float.min(), ct_float.max()
    if ct_max - ct_min > 0:
        ct_norm = (ct_float - ct_min) / (ct_max - ct_min)
    else:
        ct_norm = np.zeros_like(ct_float)

    ct_rgb = np.stack([ct_norm] * 3, axis=-1)

    # Colormapped heatmap
    cmap = plt.get_cmap(colormap)
    hm = np.clip(heatmap_slice.astype(np.float32), 0.0, 1.0)
    hm_rgb = cmap(hm)[:, :, :3]

    # Blend
    blended = (1.0 - alpha) * ct_rgb + alpha * hm_rgb
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


# ======================================================================
# Montage
# ======================================================================


def create_montage(
    volume_3d: np.ndarray,
    num_slices: int = 9,
    axis: str = "axial",
) -> np.ndarray:
    """Create a montage of representative slices from a 3-D volume.

    Slices are evenly spaced along the chosen axis. The montage is
    arranged in a grid as close to square as possible.

    Args:
        volume_3d: 3-D numpy array.
        num_slices: Number of slices to include.
        axis: Anatomical axis.

    Returns:
        2-D numpy array containing the montage.

    Example:
        >>> montage = create_montage(ct_volume, num_slices=16, axis='axial')
    """
    axis_lower = axis.lower()
    axis_map = {"axial": 0, "coronal": 1, "sagittal": 2}

    if axis_lower not in axis_map:
        raise ValueError(
            f"Unsupported axis '{axis}'. Choose from {list(axis_map.keys())}."
        )

    ax = axis_map[axis_lower]
    total = volume_3d.shape[ax]
    num_slices = min(num_slices, total)

    indices = np.linspace(0, total - 1, num_slices, dtype=int)
    slices = [
        create_slice_viewer(volume_3d, axis=axis_lower, slice_idx=int(i))
        for i in indices
    ]

    # Determine grid layout
    ncols = int(np.ceil(np.sqrt(num_slices)))
    nrows = int(np.ceil(num_slices / ncols))

    h, w = slices[0].shape[:2]
    # Handle RGB or grayscale
    if slices[0].ndim == 3:
        canvas = np.zeros((nrows * h, ncols * w, slices[0].shape[2]), dtype=slices[0].dtype)
    else:
        canvas = np.zeros((nrows * h, ncols * w), dtype=slices[0].dtype)

    for idx, slc in enumerate(slices):
        r, c = divmod(idx, ncols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = slc

    return canvas


# ======================================================================
# 3-view display
# ======================================================================


def create_3view(
    volume_3d: np.ndarray,
    heatmap_3d: Optional[np.ndarray] = None,
    slice_indices: Optional[Dict[str, int]] = None,
    alpha: float = 0.4,
) -> Dict[str, np.ndarray]:
    """Create axial, coronal, and sagittal views, optionally with overlay.

    Args:
        volume_3d: 3-D CT volume (D, H, W).
        heatmap_3d: Optional 3-D heatmap volume of the same shape.
        slice_indices: Dict mapping axis name to slice index.
            E.g. ``{'axial': 64, 'coronal': 128, 'sagittal': 128}``.
            Missing axes default to the middle slice.
        alpha: Heatmap overlay blending factor.

    Returns:
        Dict mapping axis name → 2-D numpy image (RGB if overlay,
        grayscale otherwise).

    Example:
        >>> views = create_3view(ct, heatmap, slice_indices={'axial': 50})
    """
    if slice_indices is None:
        slice_indices = {}

    views: Dict[str, np.ndarray] = {}
    for axis_name in ("axial", "coronal", "sagittal"):
        idx = slice_indices.get(axis_name, None)
        ct_slice = create_slice_viewer(volume_3d, axis=axis_name, slice_idx=idx)

        if heatmap_3d is not None:
            hm_slice = create_slice_viewer(heatmap_3d, axis=axis_name, slice_idx=idx)
            views[axis_name] = create_overlay(ct_slice, hm_slice, alpha=alpha)
        else:
            views[axis_name] = ct_slice

    return views


# ======================================================================
# Charts
# ======================================================================


def plot_triage_gauge(
    score: float,
    save_path: Optional[str] = None,
) -> Any:
    """Create a circular gauge chart for the triage score.

    Colour zones:
        - Green (0 – 0.4): LOW risk
        - Yellow (0.4 – 0.7): MEDIUM risk
        - Red (0.7 – 1.0): HIGH risk

    Args:
        score: Triage score in [0, 1].
        save_path: If provided, save the figure to this path.

    Returns:
        ``matplotlib.figure.Figure`` object.

    Example:
        >>> fig = plot_triage_gauge(0.72)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(4, 3), subplot_kw={"projection": "polar"})

    # Background arcs (half-circle gauge)
    theta_range = np.pi  # 180 degrees
    zones = [
        (0.0, 0.4, "#2ecc71", "LOW"),
        (0.4, 0.7, "#f39c12", "MEDIUM"),
        (0.7, 1.0, "#e74c3c", "HIGH"),
    ]

    for start, end, colour, _label in zones:
        theta_start = np.pi - end * np.pi
        theta_end = np.pi - start * np.pi
        theta = np.linspace(theta_start, theta_end, 50)
        ax.fill_between(theta, 0.6, 1.0, color=colour, alpha=0.3)

    # Needle
    needle_angle = np.pi - score * np.pi
    ax.plot(
        [needle_angle, needle_angle],
        [0, 0.9],
        color="black",
        linewidth=2,
        solid_capstyle="round",
    )
    ax.plot(needle_angle, 0.9, "o", color="black", markersize=6)

    # Labels
    ax.set_ylim(0, 1.2)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)

    # Score text
    if score >= 0.7:
        level, colour = "HIGH", "#e74c3c"
    elif score >= 0.4:
        level, colour = "MEDIUM", "#f39c12"
    else:
        level, colour = "LOW", "#2ecc71"

    ax.text(
        np.pi / 2,
        0.2,
        f"{score:.2f}",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=colour,
    )
    ax.text(
        np.pi / 2,
        -0.15,
        level,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=colour,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved triage gauge to %s", save_path)

    return fig


def plot_uncertainty_bar(
    score: float,
    save_path: Optional[str] = None,
) -> Any:
    """Create a horizontal bar chart showing uncertainty level.

    Args:
        score: Uncertainty score in [0, 1].
        save_path: If provided, save the figure to this path.

    Returns:
        ``matplotlib.figure.Figure`` object.

    Example:
        >>> fig = plot_uncertainty_bar(0.35)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 1.2))

    # Background bar
    ax.barh(0, 1.0, height=0.5, color="#ecf0f1", edgecolor="#bdc3c7")

    # Filled portion
    if score >= 0.6:
        colour = "#e74c3c"
    elif score >= 0.3:
        colour = "#f39c12"
    else:
        colour = "#2ecc71"

    ax.barh(0, score, height=0.5, color=colour, edgecolor="none")

    # Text
    ax.text(
        score + 0.02,
        0,
        f"{score:.3f}",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=colour,
    )

    ax.set_xlim(0, 1.15)
    ax.set_yticks([])
    ax.set_xlabel("Uncertainty")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved uncertainty bar to %s", save_path)

    return fig


# ======================================================================
# Annotation
# ======================================================================


def annotate_suspicious_regions(
    ct_slice: np.ndarray,
    regions: List[Dict[str, Any]],
    heatmap_slice: Optional[np.ndarray] = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """Draw bounding boxes around suspicious regions on a CT slice.

    Each region dict should have a ``bounding_box`` key with
    ``(slice_z, slice_y, slice_x)`` tuples from
    ``scipy.ndimage.find_objects``.

    Args:
        ct_slice: 2-D CT image (H×W) or RGB (H×W×3).
        regions: List of region dicts from ``extract_connected_components``.
        heatmap_slice: Optional heatmap to overlay before annotating.
        alpha: Overlay blending factor.

    Returns:
        Annotated RGB image (H×W×3) as ``np.uint8``.

    Example:
        >>> annotated = annotate_suspicious_regions(
        ...     ct[:, :, 64], candidates, heatmap[:, :, 64]
        ... )
    """
    import matplotlib.pyplot as plt

    # Build base image
    if heatmap_slice is not None:
        base = create_overlay(ct_slice, heatmap_slice, alpha=alpha)
    elif ct_slice.ndim == 2:
        # Grayscale → RGB
        ct_float = ct_slice.astype(np.float32)
        ct_min, ct_max = ct_float.min(), ct_float.max()
        if ct_max - ct_min > 0:
            ct_norm = (ct_float - ct_min) / (ct_max - ct_min)
        else:
            ct_norm = np.zeros_like(ct_float)
        base = (np.stack([ct_norm] * 3, axis=-1) * 255).astype(np.uint8)
    else:
        base = ct_slice.copy()

    # Draw rectangles using matplotlib
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(base)

    h, w = base.shape[:2]
    colours = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71", "#9b59b6"]

    for i, region in enumerate(regions):
        bbox = region.get("bounding_box")
        if bbox is None or len(bbox) < 2:
            continue

        # Extract y, x ranges from the bounding box slices
        # bbox is (slice_z, slice_y, slice_x) for 3D
        # For a 2D annotation we use the last two dimensions
        if len(bbox) >= 3:
            y_slice, x_slice = bbox[1], bbox[2]
        else:
            y_slice, x_slice = bbox[0], bbox[1]

        y0 = max(y_slice.start, 0)
        y1 = min(y_slice.stop, h)
        x0 = max(x_slice.start, 0)
        x1 = min(x_slice.stop, w)

        colour = colours[i % len(colours)]
        rect = plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2,
            edgecolor=colour,
            facecolor="none",
        )
        ax.add_patch(rect)

        label = f"R{i + 1} ({region.get('volume_voxels', '?')}v)"
        ax.text(
            x0,
            y0 - 3,
            label,
            color=colour,
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
        )

    ax.axis("off")
    fig.tight_layout(pad=0)

    # Render to numpy
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return buf
