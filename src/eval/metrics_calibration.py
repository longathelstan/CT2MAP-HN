# -*- coding: utf-8 -*-
"""CT2MAP-HN – Calibration metrics and reliability diagrams.

Assesses how well predicted probabilities match observed frequencies:

* **Expected Calibration Error (ECE)** – weighted average of per-bin
  calibration gaps.
* **Brier score** – mean squared error between predicted probabilities and
  binary labels.
* **Reliability diagram** – visual comparison of predicted vs observed
  frequency per bin.

These metrics are particularly important for the triage head where clinicians
rely on the reported risk probability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

_ArrayLike = Union[np.ndarray, "torch.Tensor"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy_1d(x: _ArrayLike) -> np.ndarray:
    """Flatten and convert to NumPy float64."""
    import torch

    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    return arr.ravel().astype(np.float64)


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: _ArrayLike,
    y_prob: _ArrayLike,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Expected Calibration Error (ECE).

    ECE measures the average gap between predicted confidence and actual
    accuracy, weighted by the number of samples in each bin.

    .. math::
        \\text{ECE} = \\sum_{b=1}^{B} \\frac{n_b}{N}
                      \\left| \\text{acc}(b) - \\text{conf}(b) \\right|

    Args:
        y_true: Ground-truth binary labels ``{0, 1}``.
        y_prob: Predicted probabilities in ``[0, 1]``.
        n_bins: Number of calibration bins.
        strategy: Binning strategy.

            * ``"uniform"`` – equally-spaced bins in ``[0, 1]``.
            * ``"quantile"`` – equal-count bins (adaptive).

    Returns:
        Tuple of:

        * ``ece`` – scalar ECE value.
        * ``bin_accs`` – per-bin accuracy.
        * ``bin_confs`` – per-bin mean confidence.
        * ``bin_counts`` – per-bin sample count.

    Raises:
        ValueError: If ``strategy`` is not ``"uniform"`` or ``"quantile"``.
    """
    yt = _to_numpy_1d(y_true)
    yp = _to_numpy_1d(y_prob)
    n = len(yt)

    if n == 0:
        logger.warning("ECE: empty inputs, returning 0.0")
        empty = np.zeros(n_bins)
        return 0.0, empty, empty, empty.astype(int)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(yp, quantiles)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Use 'uniform' or 'quantile'."
        )

    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == n_bins - 1:
            # Include right edge in last bin
            mask = (yp >= lo) & (yp <= hi)
        else:
            mask = (yp >= lo) & (yp < hi)

        count = int(np.sum(mask))
        bin_counts[b] = count

        if count > 0:
            bin_accs[b] = float(np.mean(yt[mask]))
            bin_confs[b] = float(np.mean(yp[mask]))

    # Weighted average of |acc - conf|
    weights = bin_counts / max(n, 1)
    ece = float(np.sum(weights * np.abs(bin_accs - bin_confs)))

    return ece, bin_accs, bin_confs, bin_counts


# ---------------------------------------------------------------------------
# Brier Score
# ---------------------------------------------------------------------------

def brier_score(
    y_true: _ArrayLike,
    y_prob: _ArrayLike,
) -> float:
    """Compute the Brier score (mean squared probability error).

    .. math::
        \\text{Brier} = \\frac{1}{N} \\sum_{i=1}^{N} (p_i - y_i)^2

    Lower is better (perfect = 0, worst = 1 for binary classification).

    Args:
        y_true: Ground-truth binary labels ``{0, 1}``.
        y_prob: Predicted probabilities in ``[0, 1]``.

    Returns:
        Scalar Brier score.
    """
    yt = _to_numpy_1d(y_true)
    yp = _to_numpy_1d(y_prob)

    if yt.size == 0:
        logger.warning("brier_score: empty inputs, returning 0.0")
        return 0.0

    return float(np.mean((yp - yt) ** 2))


# ---------------------------------------------------------------------------
# Reliability Diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    y_true: _ArrayLike,
    y_prob: _ArrayLike,
    save_path: Union[str, Path],
    n_bins: int = 15,
    strategy: str = "uniform",
    title: str = "Reliability Diagram",
    dpi: int = 150,
) -> Path:
    """Plot and save a reliability (calibration) diagram.

    Shows predicted confidence vs observed accuracy per bin, with a
    histogram of sample counts in a secondary axis, and annotates the
    ECE value.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities.
        save_path: File path for the saved figure.
        n_bins: Number of bins.
        strategy: ``"uniform"`` or ``"quantile"``.
        title: Figure title.
        dpi: Figure resolution.

    Returns:
        Absolute path to the saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ece, bin_accs, bin_confs, bin_counts = expected_calibration_error(
        y_true, y_prob, n_bins=n_bins, strategy=strategy,
    )

    bin_width = 1.0 / n_bins
    bin_centers = np.linspace(bin_width / 2, 1 - bin_width / 2, n_bins)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # --- Top: reliability diagram ---
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    # Bar chart of accuracy per bin
    mask = bin_counts > 0
    ax1.bar(
        bin_centers[mask],
        bin_accs[mask],
        width=bin_width * 0.8,
        color="steelblue",
        edgecolor="white",
        alpha=0.8,
        label="Observed accuracy",
    )

    # Gap shading
    for i in range(n_bins):
        if bin_counts[i] > 0:
            gap_lo = min(bin_accs[i], bin_confs[i])
            gap_hi = max(bin_accs[i], bin_confs[i])
            ax1.bar(
                bin_centers[i],
                gap_hi - gap_lo,
                bottom=gap_lo,
                width=bin_width * 0.8,
                color="coral",
                alpha=0.35,
                edgecolor="none",
            )

    ax1.set_ylabel("Fraction of Positives (Accuracy)")
    ax1.set_title(f"{title}  |  ECE = {ece:.4f}")
    ax1.legend(loc="upper left")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(alpha=0.3)

    # --- Bottom: histogram ---
    ax2.bar(
        bin_centers, bin_counts,
        width=bin_width * 0.8,
        color="gray", edgecolor="white", alpha=0.7,
    )
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("Reliability diagram saved to: %s (ECE=%.4f)", save_path, ece)
    return save_path.resolve()
