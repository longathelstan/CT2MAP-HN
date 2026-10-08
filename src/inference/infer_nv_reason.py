# -*- coding: utf-8 -*-
"""Inference runner for NVIDIA NV-Reason-CT (3D Vision-Language Model).

Provides deterministic or interactive inference on 3D CT NIfTI volumes (.nii.gz),
producing structured radiology reports, deep multi-step reasoning, and Q&A.
Optimized for high-VRAM GPUs (e.g., NVIDIA CMP 170HX 64GB).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class NVReasonInferencer:
    """Wrapper around NVIDIA NV-Reason-CT for 3D CT reasoning and report generation."""

    def __init__(
        self,
        model_dir: str = "checkpoints/NV-Reason-CT",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize NV-Reason-CT model and processor.

        Args:
            model_dir: Path to directory containing NV-Reason-CT checkpoints/code.
            device: Torch device (e.g. 'cuda:0', 'cuda:1', 'cpu').
            dtype: Model weights data type (default: torch.bfloat16).
        """
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        p = Path(model_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        self.model_dir = str(p.resolve())
        self.device = device
        self.dtype = dtype

        logger.info("Initializing NV-Reason-CT from %s on %s...", self.model_dir, self.device)

        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            dtype=self.dtype,
            attn_implementation="sdpa",
        ).eval().to(self.device)

        logger.info("NV-Reason-CT loaded successfully on %s.", self.device)

    def generate(
        self,
        ct_path: str,
        prompt_text: str,
        anatomy_region: str = "chest",
        enable_thinking: bool = False,
        max_new_tokens: int = 1024,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Generate clinical reasoning or report from a 3D CT volume with multi-turn support.

        Args:
            ct_path: Path to .nii.gz volume.
            prompt_text: User prompt or query text.
            anatomy_region: Region crop ('chest' or 'abdomen').
            enable_thinking: Whether to activate chain-of-thought reasoning tokens.
            max_new_tokens: Maximum tokens to generate.
            chat_history: Optional previous messages [{'role': 'user'|'assistant', 'content': str}]

        Returns:
            Dictionary with 'text', 'elapsed_seconds', 'peak_vram_gb', and 'anatomy_region'.
        """
        ct_path = str(Path(ct_path).resolve())
        if not Path(ct_path).is_file():
            raise FileNotFoundError(f"CT volume not found: {ct_path}")

        valid_regions = {"chest", "abdomen"}
        if anatomy_region not in valid_regions:
            anatomy_region = "chest"

        start_time = time.time()
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.reset_peak_memory_stats(self.device)

        # Build multi-turn messages
        messages = []
        if chat_history and len(chat_history) > 0:
            for idx, msg in enumerate(chat_history):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if idx == 0 and role == "user":
                    # First turn carries image token
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": content},
                        ],
                    })
                else:
                    messages.append({
                        "role": role,
                        "content": content,
                    })
            # Current turn
            messages.append({
                "role": "user",
                "content": prompt_text,
            })
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        inputs = self.processor(
            text=prompt,
            images3d=[ct_path],
            anatomy_region=anatomy_region,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
        response_text = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        elapsed = time.time() - start_time
        peak_vram = 0.0
        if torch.cuda.is_available() and "cuda" in self.device:
            peak_vram = torch.cuda.max_memory_allocated(self.device) / (1024**3)

        return {
            "text": response_text.strip(),
            "elapsed_seconds": elapsed,
            "peak_vram_gb": peak_vram,
            "anatomy_region": anatomy_region,
            "prompt": prompt_text,
        }
