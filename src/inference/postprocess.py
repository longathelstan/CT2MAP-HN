# -*- coding: utf-8 -*-
"""Post-processing utilities for metabolic risk heatmaps.

Functions for thresholding, connected-component extraction, triage scoring,
recommendation generation, and heatmap overlay visualisation.

Example:
    >>> from src.inference.postprocess import (
    ...     threshold_heatmap, extract_connected_components,
    ...     compute_triage_score, compute_triage_recommendation,
    ... )
    >>> binary = threshold_heatmap(heatmap, threshold=0.5)
    >>> candidates = extract_connected_components(binary, min_size=100)
    >>> score = compute_triage_score(heatmap, candidates)
    >>> rec = compute_triage_recommendation(score, uncertainty=0.3)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def threshold_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Create a binary mask from a continuous heatmap.

    Args:
        heatmap: 3-D array of predicted metabolic risk values (expected
            range [0, 1], but any float array is accepted).
        threshold: Activation threshold.  Voxels ≥ *threshold* are set
            to 1; all others to 0.

    Returns:
        Binary ``np.uint8`` array of the same shape as *heatmap*.

    Example:
        >>> mask = threshold_heatmap(heatmap, threshold=0.4)
        >>> mask.dtype
        dtype('uint8')
    """
    if heatmap.ndim != 3:
        raise ValueError(
            f"Expected 3-D heatmap, got shape {heatmap.shape}"
        )
    binary = (heatmap >= threshold).astype(np.uint8)
    logger.debug(
        "Thresholded heatmap at %.3f → %d positive voxels",
        threshold,
        int(binary.sum()),
    )
    return binary


def extract_connected_components(
    binary_mask: np.ndarray,
    min_size: int = 100,
) -> List[Dict[str, Any]]:
    """Extract connected components (lesion candidates) from a binary mask.

    Uses ``scipy.ndimage.label`` for 3-D connected-component analysis.
    Internally vectorised: ``find_objects`` and ``np.bincount`` are called
    once for *all* labels, and ``center_of_mass`` is computed only within
    the (small) bounding box of each qualifying component.

    Each returned dict contains:
        - ``label``: int – component label (1-indexed).
        - ``volume_voxels``: int – number of voxels.
        - ``centroid``: tuple[float, float, float] – centre of mass (z, y, x).
        - ``bounding_box``: tuple[slice, slice, slice].
        - ``max_intensity``: float – max value in the original heatmap
          (if the mask was derived from one).  Set to 1.0 when not
          available from the binary mask alone.

    Args:
        binary_mask: 3-D binary ``np.ndarray`` (0/1).
        min_size: Minimum number of voxels for a component to be kept.

    Returns:
        List of component-descriptor dicts, sorted by volume descending.

    Example:
        >>> comps = extract_connected_components(mask, min_size=50)
        >>> for c in comps:
        ...     print(c["volume_voxels"], c["centroid"])
    """
    try:
        from scipy import ndimage  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "scipy is required for connected-component analysis. "
            "Install it with: pip install scipy"
        ) from exc

    if binary_mask.ndim != 3:
        raise ValueError(
            f"Expected 3-D binary mask, got shape {binary_mask.shape}"
        )

    labelled, num_features = ndimage.label(binary_mask)
    logger.debug("Found %d raw connected components", num_features)

    if num_features == 0:
        return []

    # --- Vectorised volume computation (single pass over labelled) ---
    volumes = np.bincount(labelled.ravel(), minlength=num_features + 1)

    # --- Vectorised bounding boxes (single pass) ---
    all_bboxes = ndimage.find_objects(labelled)  # list of (slice, slice, slice)

    components: List[Dict[str, Any]] = []
    for lbl in range(1, num_features + 1):
        vol = int(volumes[lbl])
        if vol < min_size:
            continue

        bbox = all_bboxes[lbl - 1]  # find_objects returns 0-indexed list
        if bbox is None:
            continue

        # Compute centroid only within the small bounding-box region
        local_mask = labelled[bbox] == lbl
        local_centroid = ndimage.center_of_mass(local_mask)

        # Translate local centroid back to global coordinates
        centroid = tuple(
            float(lc + sl.start)
            for lc, sl in zip(local_centroid, bbox)
        )

        components.append(
            {
                "label": lbl,
                "volume_voxels": vol,
                "centroid": centroid,
                "bounding_box": bbox,
                "max_intensity": 1.0,
            }
        )

    # Sort by volume descending
    components.sort(key=lambda c: c["volume_voxels"], reverse=True)
    logger.info(
        "Extracted %d connected components (min_size=%d)",
        len(components),
        min_size,
    )
    return components


def _enrich_components_with_heatmap(
    components: List[Dict[str, Any]],
    heatmap: np.ndarray,
    labelled: np.ndarray,
) -> List[Dict[str, Any]]:
    """Add max/mean intensity from the heatmap to each component.

    Args:
        components: Output of ``extract_connected_components``.
        heatmap: Original continuous heatmap.
        labelled: Labelled array from ``scipy.ndimage.label``.

    Returns:
        Components list with updated intensity fields.
    """
    for comp in components:
        mask = labelled == comp["label"]
        values = heatmap[mask]
        comp["max_intensity"] = float(values.max())
        comp["mean_intensity"] = float(values.mean())
    return components


def compute_triage_score(
    heatmap: np.ndarray,
    lesion_candidates: List[Dict[str, Any]],
    method: str = "rule_based",
    *,
    uncertainty_map: Optional[np.ndarray] = None,
) -> float:
    """Compute a case-level triage score from the heatmap and candidates.

    The **rule-based** method (default) aggregates four sub-scores:

    1. ``max_activation``: peak value in the heatmap (0–1).
    2. ``volume_high_risk``: fraction of voxels above 0.7
       (higher → riskier).
    3. ``num_suspicious_components``: capped count of components
       normalised to [0, 1].
    4. ``mean_uncertainty_in_suspicious``: average uncertainty in
       high-activation regions (lower uncertainty ↔ higher confidence).

    The final score is a weighted combination in [0, 1], where
    higher = more urgent.

    Args:
        heatmap: 3-D metabolic risk heatmap.
        lesion_candidates: Output of ``extract_connected_components``.
        method: Scoring method.  Currently only ``"rule_based"`` is
            supported.
        uncertainty_map: Optional 3-D uncertainty map (same shape as
            *heatmap*).

    Returns:
        Triage score in [0, 1].

    Raises:
        ValueError: If *method* is unsupported.

    Example:
        >>> score = compute_triage_score(heatmap, candidates)
        >>> 0.0 <= score <= 1.0
        True
    """
    if method != "rule_based":
        raise ValueError(
            f"Unsupported triage method '{method}'. Use 'rule_based'."
        )

    # --- Sub-score 1: max activation ---
    max_activation = float(heatmap.max()) if heatmap.size > 0 else 0.0

    # --- Sub-score 2: volume of high-risk voxels ---
    high_risk_mask = heatmap >= 0.7
    total_voxels = max(heatmap.size, 1)
    volume_high_risk = float(high_risk_mask.sum()) / total_voxels
    # Normalise to [0, 1] with a practical cap
    volume_high_risk = min(volume_high_risk * 100.0, 1.0)

    # --- Sub-score 3: number of suspicious components ---
    num_components = len(lesion_candidates)
    # Sigmoid-like saturation: 5+ components → ~1.0
    num_suspicious_norm = min(num_components / 5.0, 1.0)

    # --- Sub-score 4: mean uncertainty in suspicious regions ---
    if uncertainty_map is not None and high_risk_mask.any():
        mean_unc = float(uncertainty_map[high_risk_mask].mean())
        # Lower uncertainty → higher confidence → score contribution
        uncertainty_score = 1.0 - min(mean_unc, 1.0)
    else:
        uncertainty_score = 0.5  # neutral when unavailable

    # --- Weighted combination ---
    weights = {
        "max_activation": 0.35,
        "volume_high_risk": 0.30,
        "num_suspicious": 0.20,
        "uncertainty_confidence": 0.15,
    }
    triage = (
        weights["max_activation"] * max_activation
        + weights["volume_high_risk"] * volume_high_risk
        + weights["num_suspicious"] * num_suspicious_norm
        + weights["uncertainty_confidence"] * uncertainty_score
    )
    triage = float(np.clip(triage, 0.0, 1.0))

    logger.info(
        "Triage score=%.4f  (max_act=%.3f, vol_hr=%.3f, "
        "n_sus=%.3f, unc_conf=%.3f)",
        triage,
        max_activation,
        volume_high_risk,
        num_suspicious_norm,
        uncertainty_score,
    )
    return triage


def compute_triage_recommendation(
    triage_score: float,
    uncertainty_score: float,
) -> str:
    """Generate a human-readable triage recommendation.

    Risk levels:
        - **HIGH** (score ≥ 0.7): immediate specialist review.
        - **MEDIUM** (0.4 ≤ score < 0.7): follow-up recommended.
        - **LOW** (score < 0.4): routine monitoring.

    If uncertainty is high (≥ 0.6), a caveat is appended.

    Args:
        triage_score: Case-level triage score in [0, 1].
        uncertainty_score: Overall uncertainty in [0, 1].

    Returns:
        Multi-line recommendation string.

    Example:
        >>> print(compute_triage_recommendation(0.85, 0.2))
        Risk Level: HIGH
        ...
    """
    # Determine risk level
    if triage_score >= 0.7:
        level = "HIGH"
        colour = "🔴"
        action = (
            "Immediate specialist review recommended. "
            "The metabolic risk map indicates significant suspicious activity."
        )
    elif triage_score >= 0.4:
        level = "MEDIUM"
        colour = "🟡"
        action = (
            "Follow-up imaging or clinical evaluation is recommended. "
            "The metabolic risk map shows moderate activity that warrants attention."
        )
    else:
        level = "LOW"
        colour = "🟢"
        action = (
            "Routine monitoring suggested. "
            "The metabolic risk map does not indicate significant suspicious activity."
        )

    lines = [
        f"Risk Level: {level} {colour}",
        f"Triage Score: {triage_score:.3f}",
        f"Uncertainty Score: {uncertainty_score:.3f}",
        "",
        f"Recommendation: {action}",
    ]

    if uncertainty_score >= 0.6:
        lines.append("")
        lines.append(
            "⚠️  Note: Model uncertainty is elevated. Results should be "
            "interpreted with caution and verified by a qualified clinician."
        )

    lines.append("")
    lines.append(
        "Disclaimer: This output is for research purposes only. "
        "It must not be used as the sole basis for clinical diagnosis."
    )

    return "\n".join(lines)


def overlay_heatmap_on_ct(
    ct_slice: np.ndarray,
    heatmap_slice: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay a heatmap on a CT slice for visualisation.

    Args:
        ct_slice: 2-D grayscale CT slice (H×W), any numeric dtype.
        heatmap_slice: 2-D heatmap (H×W) with values in [0, 1].
        alpha: Blending factor for the heatmap.  0 = CT only,
            1 = heatmap only.
        colormap: Matplotlib colormap name.

    Returns:
        RGB image (H×W×3) as ``np.uint8``.

    Example:
        >>> overlay = overlay_heatmap_on_ct(ct[:, :, 64], heatmap[:, :, 64])
        >>> overlay.shape
        (H, W, 3)
    """
    import matplotlib.pyplot as plt

    if ct_slice.ndim != 2 or heatmap_slice.ndim != 2:
        raise ValueError(
            f"Expected 2-D inputs, got ct={ct_slice.ndim}D, "
            f"heatmap={heatmap_slice.ndim}D"
        )

    # Normalise CT to [0, 1]
    ct_min, ct_max = float(ct_slice.min()), float(ct_slice.max())
    if ct_max - ct_min > 0:
        ct_norm = (ct_slice.astype(np.float32) - ct_min) / (ct_max - ct_min)
    else:
        ct_norm = np.zeros_like(ct_slice, dtype=np.float32)

    # Convert CT to RGB (grayscale → 3 channel)
    ct_rgb = np.stack([ct_norm] * 3, axis=-1)

    # Apply colormap to heatmap
    cmap = plt.get_cmap(colormap)
    heatmap_clipped = np.clip(heatmap_slice.astype(np.float32), 0.0, 1.0)
    heatmap_rgb = cmap(heatmap_clipped)[:, :, :3]  # drop alpha

    # Blend
    blended = (1.0 - alpha) * ct_rgb + alpha * heatmap_rgb
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    return blended
