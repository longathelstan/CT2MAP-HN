# -*- coding: utf-8 -*-
"""CT2MAP-HN – Figure and report export utilities.

Provides functions to generate publication-quality outputs:

* **Metric CSV** – tabulated evaluation metrics for all test cases.
* **Learning curves** – training / validation loss over epochs from
  TensorBoard logs or history dicts.
* **Slice gallery** – visual comparison of CT, predicted heatmap,
  ground-truth heatmap, and lesion mask for selected cases.
* **Ablation table** – comparison of multiple experiment runs.

All plotting uses ``matplotlib`` with the ``Agg`` backend so that no
display server is required (suitable for headless servers).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric CSV export
# ---------------------------------------------------------------------------

def export_metric_csv(
    results: List[Dict[str, Any]],
    save_path: Union[str, Path],
    fieldnames: Optional[List[str]] = None,
) -> Path:
    """Export per-case evaluation metrics as a CSV file.

    Args:
        results: List of dicts, one per test case.  Each dict must have
            the same set of keys (e.g. ``"case_id"``, ``"dice"``,
            ``"mae"``, ``"auroc"``).
        save_path: Destination CSV path.
        fieldnames: Ordered column names.  If *None*, inferred from the
            first result dict.

    Returns:
        Absolute path to the saved CSV.

    Raises:
        ValueError: If *results* is empty.
    """
    if not results:
        raise ValueError("Cannot export empty results list to CSV.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = list(results[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Convert any non-string values for cleaner output
            cleaned = {}
            for k in fieldnames:
                val = row.get(k, "")
                if isinstance(val, float):
                    cleaned[k] = f"{val:.6f}"
                elif isinstance(val, np.floating):
                    cleaned[k] = f"{float(val):.6f}"
                else:
                    cleaned[k] = val
            writer.writerow(cleaned)

    # Summary row
    logger.info(
        "Exported %d rows × %d columns to: %s",
        len(results), len(fieldnames), save_path,
    )
    return save_path.resolve()


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def export_learning_curves(
    train_history: List[Dict[str, float]],
    val_history: List[Dict[str, float]],
    save_dir: Union[str, Path],
    metrics: Optional[List[str]] = None,
    dpi: int = 150,
) -> List[Path]:
    """Plot and save training / validation learning curves.

    Creates one figure per metric, with train and val curves overlaid.

    Args:
        train_history: List of dicts (one per epoch) with metric values.
        val_history: Same format as *train_history* for validation.
        save_dir: Directory to save figures.
        metrics: Which metrics to plot.  If *None*, all metrics that appear
            in both train and val history are plotted.
        dpi: Figure resolution.

    Returns:
        List of paths to saved figures.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not train_history:
        logger.warning("export_learning_curves: empty train history.")
        return []

    # Infer metrics
    if metrics is None:
        train_keys = set(train_history[0].keys()) if train_history else set()
        val_keys = set(val_history[0].keys()) if val_history else set()
        metrics = sorted(train_keys | val_keys)

    saved: List[Path] = []

    for metric_name in metrics:
        train_vals = [ep.get(metric_name) for ep in train_history]
        val_vals = [ep.get(metric_name) for ep in val_history]

        # Filter None values
        train_epochs = [i for i, v in enumerate(train_vals) if v is not None]
        train_vals_clean = [v for v in train_vals if v is not None]
        val_epochs = [i for i, v in enumerate(val_vals) if v is not None]
        val_vals_clean = [v for v in val_vals if v is not None]

        if not train_vals_clean and not val_vals_clean:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        if train_vals_clean:
            ax.plot(
                train_epochs, train_vals_clean,
                "b-", lw=1.5, alpha=0.8, label=f"Train {metric_name}",
            )
        if val_vals_clean:
            ax.plot(
                val_epochs, val_vals_clean,
                "r-", lw=1.5, alpha=0.8, label=f"Val {metric_name}",
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Learning Curve: {metric_name}")
        ax.legend()
        ax.grid(alpha=0.3)

        # Annotate best validation value
        if val_vals_clean:
            best_idx = int(np.argmin(val_vals_clean))
            best_val = val_vals_clean[best_idx]
            best_epoch = val_epochs[best_idx]
            ax.axvline(
                x=best_epoch, color="green", linestyle="--", alpha=0.5,
            )
            ax.annotate(
                f"Best: {best_val:.5f}\n(epoch {best_epoch})",
                xy=(best_epoch, best_val),
                fontsize=8,
                color="green",
                xytext=(10, 10),
                textcoords="offset points",
            )

        fig_path = save_dir / f"learning_curve_{metric_name}.png"
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(fig_path.resolve())

    logger.info("Saved %d learning curve figures to: %s", len(saved), save_dir)
    return saved


# ---------------------------------------------------------------------------
# Slice gallery
# ---------------------------------------------------------------------------

def export_slice_gallery(
    ct_volume: np.ndarray,
    pred_heatmap: np.ndarray,
    gt_heatmap: np.ndarray,
    lesion_mask: Optional[np.ndarray],
    save_path: Union[str, Path],
    slice_indices: Optional[List[int]] = None,
    n_slices: int = 5,
    axis: int = 0,
    case_id: str = "",
    dpi: int = 150,
) -> Path:
    """Generate a gallery figure comparing CT, prediction, and ground truth.

    For each selected axial (or other axis) slice, shows side-by-side panels:
    CT, predicted heatmap, ground-truth heatmap, and (optionally) lesion mask.

    Args:
        ct_volume: CT volume ``(D, H, W)``.
        pred_heatmap: Predicted heatmap ``(D, H, W)`` in ``[0, 1]``.
        gt_heatmap: Ground-truth heatmap ``(D, H, W)``.
        lesion_mask: Optional binary lesion mask ``(D, H, W)``.
        save_path: Destination image file.
        slice_indices: Explicit slice indices to display.  If *None*,
            ``n_slices`` evenly-spaced slices are selected.
        n_slices: Number of slices when *slice_indices* is *None*.
        axis: Axis along which to take slices (0 = axial, 1 = coronal,
            2 = sagittal).
        case_id: Identifier string for the title.
        dpi: Figure resolution.

    Returns:
        Absolute path to the saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Determine slices
    depth = ct_volume.shape[axis]
    if slice_indices is None:
        # Select slices that are likely to contain lesion
        if lesion_mask is not None:
            # Find slices with lesion content
            lesion_per_slice = np.array([
                np.sum(np.take(lesion_mask, i, axis=axis))
                for i in range(depth)
            ])
            nonzero = np.where(lesion_per_slice > 0)[0]
            if len(nonzero) >= n_slices:
                step = max(1, len(nonzero) // n_slices)
                slice_indices = list(nonzero[::step][:n_slices])
            else:
                # Fall back to evenly spaced
                slice_indices = list(
                    np.linspace(0, depth - 1, n_slices, dtype=int)
                )
        else:
            slice_indices = list(
                np.linspace(0, depth - 1, n_slices, dtype=int)
            )

    n_cols = 4 if lesion_mask is not None else 3
    n_rows = len(slice_indices)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5),
    )

    if n_rows == 1:
        axes = axes[np.newaxis, :]  # type: ignore[index]

    col_titles = ["CT", "Predicted Heatmap", "GT Heatmap"]
    if lesion_mask is not None:
        col_titles.append("Lesion Mask")

    for row_idx, sl_idx in enumerate(slice_indices):
        ct_slice = np.take(ct_volume, sl_idx, axis=axis)
        pred_slice = np.take(pred_heatmap, sl_idx, axis=axis)
        gt_slice = np.take(gt_heatmap, sl_idx, axis=axis)

        # CT slice (grayscale)
        axes[row_idx, 0].imshow(ct_slice.T, cmap="gray", origin="lower")
        axes[row_idx, 0].set_ylabel(f"Slice {sl_idx}", fontsize=8)

        # Predicted heatmap (jet overlay)
        axes[row_idx, 1].imshow(ct_slice.T, cmap="gray", origin="lower", alpha=0.3)
        im = axes[row_idx, 1].imshow(
            pred_slice.T, cmap="jet", origin="lower",
            alpha=0.7, vmin=0, vmax=1,
        )

        # Ground-truth heatmap
        axes[row_idx, 2].imshow(ct_slice.T, cmap="gray", origin="lower", alpha=0.3)
        axes[row_idx, 2].imshow(
            gt_slice.T, cmap="jet", origin="lower",
            alpha=0.7, vmin=0, vmax=1,
        )

        # Lesion mask
        if lesion_mask is not None:
            mask_slice = np.take(lesion_mask, sl_idx, axis=axis)
            axes[row_idx, 3].imshow(ct_slice.T, cmap="gray", origin="lower")
            axes[row_idx, 3].imshow(
                mask_slice.T, cmap="Reds", origin="lower",
                alpha=0.5, vmin=0, vmax=1,
            )

        # Column titles on first row
        if row_idx == 0:
            for col_idx, col_title in enumerate(col_titles):
                axes[0, col_idx].set_title(col_title, fontsize=10)

        # Remove ticks
        for col_idx in range(n_cols):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    title = f"Slice Gallery: {case_id}" if case_id else "Slice Gallery"
    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("Slice gallery saved to: %s (%d slices)", save_path, len(slice_indices))
    return save_path.resolve()


# ---------------------------------------------------------------------------
# Ablation table
# ---------------------------------------------------------------------------

def export_ablation_table(
    experiments: List[Dict[str, Any]],
    save_path: Union[str, Path],
    metrics: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    ascending: bool = True,
    latex: bool = False,
) -> Path:
    """Export a comparison table across multiple experiment runs.

    Generates a CSV (and optionally LaTeX) table showing metrics for each
    experiment, suitable for ablation studies.

    Args:
        experiments: List of dicts, each with ``"name"`` and metric values.
            Example::

                [
                    {"name": "Baseline UNet", "dice": 0.72, "mae": 0.15},
                    {"name": "Swin UNETR",    "dice": 0.78, "mae": 0.11},
                ]

        save_path: Destination CSV path.
        metrics: Columns to include (besides ``"name"``).  If *None*,
            all numeric columns from the first experiment are included.
        sort_by: Optional metric name to sort rows by.
        ascending: Sort order (lower-is-better vs higher-is-better).
        latex: If *True*, also save a ``.tex`` version alongside the CSV.

    Returns:
        Absolute path to the saved CSV.
    """
    if not experiments:
        raise ValueError("Cannot export empty experiments list.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine columns
    if metrics is None:
        metrics = [
            k for k in experiments[0].keys()
            if k != "name" and isinstance(experiments[0][k], (int, float, np.floating))
        ]

    # Sort if requested
    if sort_by and sort_by in metrics:
        experiments = sorted(
            experiments,
            key=lambda x: x.get(sort_by, float("inf")),
            reverse=not ascending,
        )

    fieldnames = ["name"] + metrics

    # CSV export
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for exp in experiments:
            row = {"name": exp.get("name", "?")}
            for m in metrics:
                val = exp.get(m, "")
                if isinstance(val, (float, np.floating)):
                    row[m] = f"{float(val):.4f}"
                else:
                    row[m] = str(val)
            writer.writerow(row)

    logger.info(
        "Ablation table exported: %d experiments × %d metrics → %s",
        len(experiments), len(metrics), save_path,
    )

    # Optional LaTeX
    if latex:
        tex_path = save_path.with_suffix(".tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            # Build LaTeX table
            col_spec = "l" + "c" * len(metrics)
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Ablation Study Results}\n")
            f.write("\\label{tab:ablation}\n")
            f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
            f.write("\\toprule\n")
            header = " & ".join(["Experiment"] + [m.replace("_", "\\_") for m in metrics])
            f.write(f"{header} \\\\\n")
            f.write("\\midrule\n")

            for exp in experiments:
                cells = [exp.get("name", "?").replace("_", "\\_")]
                for m in metrics:
                    val = exp.get(m, "")
                    if isinstance(val, (float, np.floating)):
                        cells.append(f"{float(val):.4f}")
                    else:
                        cells.append(str(val))
                f.write(" & ".join(cells) + " \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        logger.info("LaTeX ablation table saved to: %s", tex_path)

    return save_path.resolve()
