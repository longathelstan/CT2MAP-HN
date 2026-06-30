# -*- coding: utf-8 -*-
"""Inference module for CT2MAP-HN.

Provides single-case inference, batch inference, and post-processing
utilities for generating metabolic risk heatmaps from head-neck CT scans.
"""

from src.inference.infer_case import CaseInferencer
from src.inference.infer_batch import batch_infer
from src.inference.postprocess import (
    threshold_heatmap,
    extract_connected_components,
    compute_triage_score,
    compute_triage_recommendation,
    overlay_heatmap_on_ct,
)

__all__ = [
    "CaseInferencer",
    "batch_infer",
    "threshold_heatmap",
    "extract_connected_components",
    "compute_triage_score",
    "compute_triage_recommendation",
    "overlay_heatmap_on_ct",
]
