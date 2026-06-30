# -*- coding: utf-8 -*-
"""CT2MAP-HN – Evaluation metrics and visualization package.

Provides submodules for computing voxel-level, lesion-level, triage, and
calibration metrics, as well as figure export utilities.

Submodules:
    - :mod:`metrics_voxel` – Dice, IoU, sensitivity, Hausdorff distance.
    - :mod:`metrics_lesion` – MAE, correlation, voxel MSE for heatmap quality.
    - :mod:`metrics_triage` – AUROC, AUPRC, precision/recall/F1, ROC/PR plots.
    - :mod:`metrics_calibration` – ECE, Brier score, reliability diagrams.
    - :mod:`export_figures` – Gallery, learning curves, metric CSV, ablation tables.
"""

from src.eval.metrics_voxel import (
    dice_score,
    iou_score,
    sensitivity_score,
    specificity_score,
    hausdorff_distance_95,
)
from src.eval.metrics_lesion import (
    voxel_mae,
    voxel_mse,
    voxel_rmse,
    pearson_correlation,
    spearman_correlation,
    ssim_3d,
)
from src.eval.metrics_triage import (
    compute_auroc,
    compute_auprc,
    compute_precision_recall_f1,
    plot_roc_curve,
    plot_pr_curve,
)
from src.eval.metrics_calibration import (
    expected_calibration_error,
    brier_score,
    plot_reliability_diagram,
)
from src.eval.export_figures import (
    export_metric_csv,
    export_learning_curves,
    export_slice_gallery,
    export_ablation_table,
)

__all__ = [
    # Voxel metrics
    "dice_score",
    "iou_score",
    "sensitivity_score",
    "specificity_score",
    "hausdorff_distance_95",
    # Lesion/heatmap metrics
    "voxel_mae",
    "voxel_mse",
    "voxel_rmse",
    "pearson_correlation",
    "spearman_correlation",
    "ssim_3d",
    # Triage metrics
    "compute_auroc",
    "compute_auprc",
    "compute_precision_recall_f1",
    "plot_roc_curve",
    "plot_pr_curve",
    # Calibration metrics
    "expected_calibration_error",
    "brier_score",
    "plot_reliability_diagram",
    # Export utilities
    "export_metric_csv",
    "export_learning_curves",
    "export_slice_gallery",
    "export_ablation_table",
]
