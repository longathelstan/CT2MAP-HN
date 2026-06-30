# -*- coding: utf-8 -*-
"""CT2MAP-HN – Triage-level classification metrics.

Computes case-level classification metrics for the binary triage head:

* **AUROC** – area under the receiver operating characteristic curve.
* **AUPRC** – area under the precision–recall curve.
* **Precision / Recall / F1** at a given threshold.
* **ROC curve plot** – saved to disk with optimal threshold annotated.
* **PR curve plot** – saved to disk with iso-F1 contours.

All metric functions accept 1-D arrays of predictions and labels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

_ArrayLike = Union[np.ndarray, "torch.Tensor"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy_1d(x: _ArrayLike) -> np.ndarray:
    """Ensure input is a flat NumPy float64 array.

    Args:
        x: 1-D predictions or labels.

    Returns:
        Flat NumPy array.
    """
    import torch

    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    return arr.ravel().astype(np.float64)


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------

def compute_auroc(
    y_true: _ArrayLike,
    y_score: _ArrayLike,
) -> float:
    """Compute the area under the ROC curve.

    Args:
        y_true: Ground-truth binary labels ``{0, 1}``.
        y_score: Predicted probabilities in ``[0, 1]``.

    Returns:
        AUROC value in ``[0, 1]``.

    Raises:
        ValueError: If all labels are the same class (ROC undefined).
    """
    from sklearn.metrics import roc_auc_score

    yt = _to_numpy_1d(y_true)
    ys = _to_numpy_1d(y_score)

    n_classes = len(np.unique(yt))
    if n_classes < 2:
        logger.warning(
            "compute_auroc: only %d class(es) present, returning 0.0",
            n_classes,
        )
        return 0.0

    return float(roc_auc_score(yt, ys))


# ---------------------------------------------------------------------------
# AUPRC
# ---------------------------------------------------------------------------

def compute_auprc(
    y_true: _ArrayLike,
    y_score: _ArrayLike,
) -> float:
    """Compute the area under the precision–recall curve.

    Args:
        y_true: Ground-truth binary labels ``{0, 1}``.
        y_score: Predicted probabilities in ``[0, 1]``.

    Returns:
        AUPRC value in ``[0, 1]``.
    """
    from sklearn.metrics import average_precision_score

    yt = _to_numpy_1d(y_true)
    ys = _to_numpy_1d(y_score)

    n_classes = len(np.unique(yt))
    if n_classes < 2:
        logger.warning(
            "compute_auprc: only %d class(es) present, returning 0.0",
            n_classes,
        )
        return 0.0

    return float(average_precision_score(yt, ys))


# ---------------------------------------------------------------------------
# Precision / Recall / F1
# ---------------------------------------------------------------------------

def compute_precision_recall_f1(
    y_true: _ArrayLike,
    y_score: _ArrayLike,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute precision, recall, and F1 score at a given threshold.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probabilities.
        threshold: Decision threshold for converting probabilities to labels.

    Returns:
        Dictionary with keys ``"precision"``, ``"recall"``, ``"f1"``,
        ``"threshold"``.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    yt = _to_numpy_1d(y_true).astype(int)
    y_pred = (_to_numpy_1d(y_score) >= threshold).astype(int)

    # Handle edge cases
    if len(np.unique(yt)) < 2 or len(np.unique(y_pred)) < 2:
        logger.warning(
            "compute_precision_recall_f1: degenerate predictions or labels. "
            "Using zero_division=0."
        )

    precision = float(precision_score(yt, y_pred, zero_division=0))
    recall = float(recall_score(yt, y_pred, zero_division=0))
    f1 = float(f1_score(yt, y_pred, zero_division=0))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# ROC Curve Plot
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: _ArrayLike,
    y_score: _ArrayLike,
    save_path: Union[str, Path],
    title: str = "ROC Curve",
    dpi: int = 150,
) -> Path:
    """Plot and save an ROC curve with the optimal operating point.

    The optimal threshold is identified by maximising Youden's J statistic
    (sensitivity + specificity − 1).

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probabilities.
        save_path: File path to save the figure (e.g. ``"roc.png"``).
        title: Figure title.
        dpi: Resolution for saved figure.

    Returns:
        Absolute path to the saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    yt = _to_numpy_1d(y_true)
    ys = _to_numpy_1d(y_score)

    fpr, tpr, thresholds = roc_curve(yt, ys)
    auc_val = roc_auc_score(yt, ys) if len(np.unique(yt)) >= 2 else 0.0

    # Optimal threshold (Youden's J)
    j_scores = tpr - fpr
    optimal_idx = int(np.argmax(j_scores))
    optimal_thresh = float(thresholds[optimal_idx]) if optimal_idx < len(thresholds) else 0.5

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.scatter(
        fpr[optimal_idx], tpr[optimal_idx],
        c="red", s=80, zorder=5,
        label=f"Optimal (t={optimal_thresh:.3f})",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("ROC curve saved to: %s (AUC=%.4f)", save_path, auc_val)
    return save_path.resolve()


# ---------------------------------------------------------------------------
# PR Curve Plot
# ---------------------------------------------------------------------------

def plot_pr_curve(
    y_true: _ArrayLike,
    y_score: _ArrayLike,
    save_path: Union[str, Path],
    title: str = "Precision-Recall Curve",
    dpi: int = 150,
) -> Path:
    """Plot and save a precision–recall curve with iso-F1 contours.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probabilities.
        save_path: File path to save the figure.
        title: Figure title.
        dpi: Resolution.

    Returns:
        Absolute path to the saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    yt = _to_numpy_1d(y_true)
    ys = _to_numpy_1d(y_score)

    precision, recall, thresholds = precision_recall_curve(yt, ys)
    ap = average_precision_score(yt, ys) if len(np.unique(yt)) >= 2 else 0.0

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Iso-F1 contours
    f1_values = np.linspace(0.2, 0.8, num=4)
    for f1 in f1_values:
        x = np.linspace(0.01, 1, 100)
        y = f1 * x / (2 * x - f1)
        valid = (y >= 0) & (y <= 1)
        ax.plot(
            x[valid], y[valid],
            color="gray", alpha=0.2, linestyle="--", lw=0.8,
        )
        # Label the F1 contour
        label_idx = np.argmin(np.abs(x[valid] - 0.9))
        if np.any(valid):
            ax.annotate(
                f"F1={f1:.1f}",
                (x[valid][label_idx], y[valid][label_idx]),
                fontsize=7, alpha=0.4,
            )

    ax.plot(recall, precision, "b-", lw=2, label=f"AP = {ap:.4f}")

    # Baseline: positive class prevalence
    prevalence = float(np.mean(yt))
    ax.axhline(y=prevalence, color="k", linestyle="--", lw=1, alpha=0.4, label=f"Baseline = {prevalence:.3f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("PR curve saved to: %s (AP=%.4f)", save_path, ap)
    return save_path.resolve()
