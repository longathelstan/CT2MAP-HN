# -*- coding: utf-8 -*-
"""Inference runner for NVIDIA NV-Segment-CT (VISTA3D 132-class Foundation Model).

Runs VISTA3D in an isolated Python 3.10 virtual environment on a dedicated GPU
(default CUDA_VISIBLE_DEVICES=1) to prevent VRAM competition and dependency conflicts
with the host Streamlit / OmniReason-CT environment.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_VENV = Path.home() / ".cache" / "nvidia-skills" / "venvs" / "nv-segment-ct-f9f5f51"
DEFAULT_SKILL_DIR = Path.home() / "longspec" / ".agents" / "skills" / "nv-segment-ct"


class NVSegmentRunner:
    """Orchestrator for NV-Segment-CT (VISTA3D) 3D multi-organ CT segmentation."""

    def __init__(
        self,
        venv_path: Optional[Union[str, Path]] = None,
        skill_dir: Optional[Union[str, Path]] = None,
        gpu_device: str = "1",
    ) -> None:
        """Initialize runner with paths and target GPU.

        Args:
            venv_path: Path to isolated Python 3.10 venv.
            skill_dir: Path to nv-segment-ct skill directory.
            gpu_device: GPU index for CUDA_VISIBLE_DEVICES (default '1').
        """
        self.venv_path = Path(venv_path) if venv_path else DEFAULT_VENV
        self.skill_dir = Path(skill_dir) if skill_dir else DEFAULT_SKILL_DIR
        self.gpu_device = str(gpu_device)

        self.python_bin = self.venv_path / "bin" / "python"
        self.entrypoint = self.skill_dir / "scripts" / "run_vista3d.py"
        self.bundle_dir = self.skill_dir / "bundle"
        self.label_dict_file = self.bundle_dir / "label_dict.json"

        self._label_dict: Optional[Dict[str, int]] = None
        self._inv_label_dict: Optional[Dict[int, str]] = None

    def is_ready(self) -> Tuple[bool, str]:
        """Check whether the isolated environment, script, and weights are present.

        Returns:
            (is_ready, message): True if all components exist, else False with reason.
        """
        if not self.python_bin.exists():
            return False, f"Python executable không tồn tại tại: {self.python_bin}"

        if not self.entrypoint.exists():
            return False, f"Script run_vista3d.py không tồn tại tại: {self.entrypoint}"

        safetensors = self.bundle_dir / "vista3d_pretrained_model" / "model.safetensors"
        model_pt = self.bundle_dir / "vista3d_pretrained_model" / "model.pt"
        if not (safetensors.exists() or model_pt.exists()):
            return False, f"Trọng số VISTA3D không tìm thấy tại {self.bundle_dir / 'vista3d_pretrained_model'}"

        return True, "Hệ thống NV-Segment-CT (VISTA3D) đã sẵn sàng."

    def get_label_dict(self) -> Dict[str, int]:
        """Load and return label name to ID mapping (132 anatomical classes)."""
        if self._label_dict is None:
            if self.label_dict_file.exists():
                try:
                    with open(self.label_dict_file, "r", encoding="utf-8") as f:
                        self._label_dict = json.load(f)
                except Exception as e:
                    logger.warning("Không thể đọc label_dict.json: %s", e)
                    self._label_dict = {}
            else:
                self._label_dict = {}
        return self._label_dict

    def get_inv_label_dict(self) -> Dict[int, str]:
        """Return label ID to name mapping."""
        if self._inv_label_dict is None:
            ld = self.get_label_dict()
            self._inv_label_dict = {int(v): k for k, v in ld.items()}
        return self._inv_label_dict

    @staticmethod
    def get_supported_presets() -> Dict[str, Dict[str, Any]]:
        """Return clinical presets for quick anatomical segmentation."""
        return {
            "abdominal_4": {
                "name": "Ổ bụng - 4 tạng chính",
                "label_ids": [1, 3, 5, 14],
                "description": "Gan, Lách, Thận phải, Thận trái",
                "organ_names": ["liver", "spleen", "right kidney", "left kidney"],
                "badge": "⭐ Khuyến nghị (Abdomen)",
            },
            "chest_full": {
                "name": "Toàn bộ lồng ngực & Phổi",
                "label_ids": [28, 29, 30, 31, 32, 57, 115],
                "description": "5 thùy phổi, Khí quản, Tim",
                "organ_names": [
                    "left lung upper lobe",
                    "left lung lower lobe",
                    "right lung upper lobe",
                    "right lung middle lobe",
                    "right lung lower lobe",
                    "trachea",
                    "heart",
                ],
                "badge": "🫁 Khuyến nghị (Chest)",
            },
            "abdominal_all": {
                "name": "Ổ bụng toàn diện (8 tạng)",
                "label_ids": [1, 3, 4, 5, 6, 10, 12, 14],
                "description": "Gan, Lách, Tụy, Thận P/T, Động mạch chủ, Túi mật, Dạ dày",
                "organ_names": [
                    "liver",
                    "spleen",
                    "pancreas",
                    "right kidney",
                    "aorta",
                    "gallbladder",
                    "stomach",
                    "left kidney",
                ],
                "badge": "🔬 Chuyên sâu",
            },
        }

    def segment(
        self,
        ct_path: Union[str, Path],
        label_ids: Union[List[int], str],
        output_dir: Optional[Union[str, Path]] = None,
        force_rerun: bool = False,
    ) -> Dict[str, Any]:
        """Perform 3D CT multi-organ segmentation via isolated subprocess.

        Args:
            ct_path: Path to CT NIfTI file (.nii or .nii.gz).
            label_ids: List of integer class IDs or comma-separated string (e.g. [1, 3, 5, 14]).
            output_dir: Optional output directory. Default: outputs/vista3d/<case_stem>.
            force_rerun: If True, bypass cache and recompute.

        Returns:
            Dictionary matching the output format of run_vista3d.py, with extra metadata.
        """
        ct_path = Path(ct_path).resolve()
        if not ct_path.is_file():
            raise FileNotFoundError(f"File CT không tồn tại: {ct_path}")

        ready, msg = self.is_ready()
        if not ready:
            raise RuntimeError(f"NV-Segment-CT chưa sẵn sàng: {msg}")

        # Normalize label IDs
        if isinstance(label_ids, str):
            clean_ids = [int(x.strip()) for x in label_ids.split(",") if x.strip()]
        else:
            clean_ids = [int(x) for x in label_ids]
        clean_ids = sorted(list(set(clean_ids)))
        label_str = ",".join(str(x) for x in clean_ids)
        label_tag = "_".join(str(x) for x in clean_ids)

        # Determine output directory
        if output_dir is None:
            case_stem = ct_path.name
            for sfx in (".nii.gz", ".nii"):
                if case_stem.endswith(sfx):
                    case_stem = case_stem[: -len(sfx)]
                    break
            # Use parent case name if ct_path is inside a named directory like example_2/ct.nii.gz
            if ct_path.parent.name and ct_path.parent.name not in ("demo", "data", "CT2MAP-HN"):
                case_stem = ct_path.parent.name
            out_path = PROJECT_ROOT / "outputs" / "vista3d" / case_stem
        else:
            out_path = Path(output_dir).resolve()

        out_path.mkdir(parents=True, exist_ok=True)
        cache_file = out_path / f"result_labels_{label_tag}.json"
        legacy_cache_file = out_path / "result.json"

        # Check cache
        if not force_rerun:
            for c_candidate in (cache_file, legacy_cache_file):
                if c_candidate.exists():
                    try:
                        with open(c_candidate, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        req_labels = sorted(data.get("output", {}).get("label_prompts_requested", []))
                        mask_p = data.get("output", {}).get("path")
                        if req_labels == clean_ids and mask_p and Path(mask_p).exists():
                            logger.info("Sử dụng kết quả VISTA3D đã cache tại: %s", c_candidate)
                            data["cached"] = True
                            data["cache_file"] = str(c_candidate)
                            return data
                    except Exception as e:
                        logger.warning("Không thể đọc cache file %s: %s", c_candidate, e)

        # Prepare subprocess execution
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu_device
        env["PYTHONUNBUFFERED"] = "1"
        env["MLFLOW_DISABLE_AGENT_HINT"] = "1"

        cmd = [
            str(self.python_bin),
            str(self.entrypoint),
            str(ct_path),
            "--label-prompts",
            label_str,
            "--output-dir",
            str(out_path),
            "--device",
            "cuda",
        ]

        logger.info(
            "Khởi chạy VISTA3D trên GPU %s: %s (labels: %s)...",
            self.gpu_device,
            ct_path.name,
            label_str,
        )
        t0 = time.time()

        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t0

        if proc.returncode != 0:
            err_msg = proc.stderr.strip() or proc.stdout.strip()
            logger.error("Lỗi khi chạy run_vista3d.py (code %d): %s", proc.returncode, err_msg)
            raise RuntimeError(f"VISTA3D inference thất bại (mã {proc.returncode}): {err_msg}")

        # Parse JSON from stdout
        stdout_text = proc.stdout.strip()
        data = None
        # Often stdout contains pure JSON, but handle any accidental log lines
        try:
            data = json.loads(stdout_text)
        except json.JSONDecodeError:
            # Attempt to locate JSON block
            first_brace = stdout_text.find("{")
            last_brace = stdout_text.rfind("}")
            if first_brace != -1 and last_brace != -1:
                try:
                    data = json.loads(stdout_text[first_brace : last_brace + 1])
                except Exception as e:
                    raise RuntimeError(f"Không thể parse JSON từ VISTA3D output: {e}\nOutput: {stdout_text}")
            else:
                raise RuntimeError(f"VISTA3D không trả về định dạng JSON hợp lệ:\n{stdout_text}")

        # Save to cache
        data["cached"] = False
        data["cache_file"] = str(cache_file)
        data["total_runner_seconds"] = round(elapsed, 3)

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            with open(legacy_cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("Không thể lưu cache VISTA3D: %s", e)

        logger.info(
            "VISTA3D hoàn thành cho %s trong %.2fs (inference: %.2fs)",
            ct_path.name,
            elapsed,
            data.get("runtime", {}).get("inference_seconds", 0),
        )
        return data
