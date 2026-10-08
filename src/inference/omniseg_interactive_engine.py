# -*- coding: utf-8 -*-
"""OmniSeg-CT Interactive Segmentation Engine (VISTA3D Backend).

Provides:
  1. Real-time point-prompt segmentation (Positive/Negative clicks) via VISTA3D on GPU 1.
  2. Algorithm 1 (Connected Components Refinement) to add missing tissue or excise over-segmentation.
  3. Seamless volume metrics (mL) and mask generation for real CT scans (example_1, example_2, example_3).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_VENV_PYTHON = Path.home() / ".cache" / "nvidia-skills" / "venvs" / "nv-segment-ct-f9f5f51" / "bin" / "python"
BUNDLE_DIR = Path.home() / "longspec" / ".agents" / "skills" / "nv-segment-ct" / "bundle"


def apply_algorithm_1_cc_refinement(
    m_auto: np.ndarray,
    m_point: np.ndarray,
    pos_points: List[Tuple[int, int, int]],
    neg_points: List[Tuple[int, int, int]],
) -> np.ndarray:
    """Algorithm 1 from OmniSeg-CT paper: Connected Components interactive refinement.

    Args:
        m_auto: Current base mask shape (Z, Y, X).
        m_point: Point head prediction mask shape (Z, Y, X).
        pos_points: Positive prompt coords (z, y, x).
        neg_points: Negative prompt coords (z, y, x).

    Returns:
        Refined binary mask M_final = M_a + M_final_add - M_final_rm.
    """
    m_a = (m_auto > 0).astype(np.uint8)
    m_p = (m_point > 0).astype(np.uint8)

    # 1. Connected components to add: (M_p - M_a) > 0
    diff_add = ((m_p.astype(np.int32) - m_a.astype(np.int32)) > 0).astype(np.uint8)
    labeled_add, num_add = ndi.label(diff_add)

    m_final_add = np.zeros_like(m_a, dtype=np.uint8)
    for p in pos_points:
        z, y, x = int(p[0]), int(p[1]), int(p[2])
        if 0 <= z < labeled_add.shape[0] and 0 <= y < labeled_add.shape[1] and 0 <= x < labeled_add.shape[2]:
            lbl = labeled_add[z, y, x]
            if lbl > 0:
                m_final_add[labeled_add == lbl] = 1

    # Also add local M_p around positive clicks if they fall on edges of M_a
    for p in pos_points:
        z, y, x = int(p[0]), int(p[1]), int(p[2])
        if 0 <= z < m_p.shape[0] and 0 <= y < m_p.shape[1] and 0 <= x < m_p.shape[2]:
            if m_p[z, y, x] > 0:
                labeled_mp, _ = ndi.label(m_p)
                lbl_mp = labeled_mp[z, y, x]
                if lbl_mp > 0:
                    m_final_add[labeled_mp == lbl_mp] = 1

    # 2. Connected components to remove: (M_a - M_p) > 0
    diff_rm = ((m_a.astype(np.int32) - m_p.astype(np.int32)) > 0).astype(np.uint8)
    labeled_rm, num_rm = ndi.label(diff_rm)

    m_final_rm = np.zeros_like(m_a, dtype=np.uint8)
    for p in neg_points:
        z, y, x = int(p[0]), int(p[1]), int(p[2])
        if 0 <= z < labeled_rm.shape[0] and 0 <= y < labeled_rm.shape[1] and 0 <= x < labeled_rm.shape[2]:
            lbl = labeled_rm[z, y, x]
            if lbl > 0:
                m_final_rm[labeled_rm == lbl] = 1

    # Fallback excision around negative point in M_a
    for p in neg_points:
        z, y, x = int(p[0]), int(p[1]), int(p[2])
        if 0 <= z < m_a.shape[0] and 0 <= y < m_a.shape[1] and 0 <= x < m_a.shape[2]:
            if m_a[z, y, x] > 0:
                z_min, z_max = max(0, z - 3), min(m_a.shape[0], z + 4)
                y_min, y_max = max(0, y - 8), min(m_a.shape[1], y + 9)
                x_min, x_max = max(0, x - 8), min(m_a.shape[2], x + 9)
                m_final_rm[z_min:z_max, y_min:y_max, x_min:x_max] = 1

    m_final = np.clip(m_a + m_final_add - m_final_rm, 0, 1).astype(np.uint8)
    return m_final


class OmniSegInteractiveRunner:
    """Runner for interactive point prompts and live model inference on GPU 1."""

    def __init__(
        self,
        venv_python: Optional[Union[str, Path]] = None,
        bundle_dir: Optional[Union[str, Path]] = None,
        gpu_device: str = "1",
    ) -> None:
        self.venv_python = Path(venv_python) if venv_python else DEFAULT_VENV_PYTHON
        self.bundle_dir = Path(bundle_dir) if bundle_dir else BUNDLE_DIR
        self.gpu_device = str(gpu_device)

    def is_ready(self) -> Tuple[bool, str]:
        if not self.venv_python.exists():
            return False, f"Không tìm thấy Python venv: {self.venv_python}"
        if not (self.bundle_dir / "vista3d_pretrained_model").exists():
            return False, f"Không tìm thấy weights bundle: {self.bundle_dir}"
        return True, "OmniSeg-CT interactive engine sẵn sàng."

    def run_point_inference(
        self,
        ct_path: Union[str, Path],
        points_zyx: List[Tuple[int, int, int]],
        point_labels: List[int],
        output_mask_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Execute live VISTA3D point prompt inference on GPU 1.

        Args:
            ct_path: Path to CT NIfTI file.
            points_zyx: List of (z, y, x) coordinates in voxel space.
            point_labels: List of 1 (positive) or 0 (negative).
            output_mask_path: Target path to save clean output NIfTI.
        """
        ct_path = Path(ct_path).resolve()
        output_mask_path = Path(output_mask_path).resolve()
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert z,y,x voxel to x,y,z as expected by VISTA3D pipeline
        points_xyz = [[int(p[2]), int(p[1]), int(p[0])] for p in points_zyx]
        tmp_run_dir = output_mask_path.parent / "tmp_run"
        tmp_run_dir.mkdir(parents=True, exist_ok=True)

        script_code = f"""
import sys, os, torch, json
sys.path.insert(0, '{str(self.bundle_dir)}')
from hugging_face_pipeline import HuggingFacePipelineHelper

helper = HuggingFacePipelineHelper('vista3d')
pipeline = helper.init_pipeline('{str(self.bundle_dir / "vista3d_pretrained_model")}', device=torch.device('cuda:0'))

inputs = [{{
    'image': '{str(ct_path)}',
    'points': {json.dumps(points_xyz)},
    'point_labels': {json.dumps(point_labels)}
}}]

pipeline(inputs, output_dir='{str(tmp_run_dir)}')
print('SUCCESS_VISTA3D')
"""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu_device
        env["PYTHONUNBUFFERED"] = "1"
        env["MLFLOW_DISABLE_AGENT_HINT"] = "1"

        t0 = time.time()
        proc = subprocess.run(
            [str(self.venv_python), "-c", script_code],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        elapsed = time.time() - t0

        if proc.returncode != 0:
            logger.error("Interactive VISTA3D failed: %s", proc.stderr)
            raise RuntimeError(f"Lỗi VISTA3D interactive (code {proc.returncode}): {proc.stderr[:1000]}")

        # Find raw output file in tmp_run_dir
        found_mask = None
        for cand in tmp_run_dir.rglob("*_seg.nii.gz"):
            found_mask = cand
            break
        if not found_mask:
            for cand in tmp_run_dir.rglob("*.nii.gz"):
                found_mask = cand
                break

        if not found_mask or not found_mask.exists():
            raise FileNotFoundError(f"Không tìm thấy file kết quả sau khi chạy VISTA3D tại {tmp_run_dir}")

        # Load raw mask, clean background (keep only foreground label == 1), and save clean mask
        raw_img = nib.load(str(found_mask))
        raw_arr = raw_img.get_fdata()
        clean_arr = (raw_arr == 1).astype(np.uint8)

        spacing = raw_img.header.get_zooms()[:3]
        voxel_count = int(np.sum(clean_arr))
        vol_ml = round(float(voxel_count * np.prod(spacing) / 1000.0), 3)

        clean_img = nib.Nifti1Image(clean_arr, raw_img.affine, raw_img.header)
        nib.save(clean_img, str(output_mask_path))

        # Cleanup tmp_run_dir
        try:
            shutil.rmtree(tmp_run_dir, ignore_errors=True)
        except Exception:
            pass

        return {
            "elapsed_seconds": round(elapsed, 2),
            "mask_path": str(output_mask_path),
            "volume_ml": vol_ml,
            "voxel_count": voxel_count,
            "points": points_xyz,
            "point_labels": point_labels,
        }
