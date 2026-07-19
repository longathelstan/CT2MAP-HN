# -*- coding: utf-8 -*-
"""Single-case inference for CT2MAP-HN.

Loads a trained model checkpoint, preprocesses a CT volume, runs inference
(optionally with MC Dropout for uncertainty estimation), and saves
structured outputs (heatmap, lesion candidates, triage report).

Can be used as a library:
    >>> from src.inference.infer_case import CaseInferencer
    >>> inferencer = CaseInferencer("ckpt/best.pt", config)
    >>> results = inferencer.infer("data/case_001/ct.nii.gz")

Or as a CLI:
    $ python -m src.inference.infer_case \\
        --ct data/case_001/ct.nii.gz \\
        --ckpt outputs/checkpoints/best.pt \\
        --config configs/demo.yaml \\
        --out outputs/predictions/case_001/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class _HeatmapOutputWrapper(nn.Module):
    """Expose the heatmap tensor from a multi-head model's dict output.

    ``BaselineUNet.forward`` returns ``{"heatmap", "lesion", "triage", ...}``,
    but MONAI's ``SlidingWindowInferer`` expects the wrapped module to return a
    single tensor.  This adapter selects one key so the trained model can be
    driven by the inferer unchanged.
    """

    def __init__(self, model: nn.Module, key: str = "heatmap") -> None:
        super().__init__()
        self.model = model
        self.key = key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            return out[self.key]
        return out


class CaseInferencer:
    """End-to-end single-case inference engine.

    Handles model loading, CT preprocessing, forward pass, post-processing,
    and optional MC Dropout uncertainty estimation.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        config: Configuration dict or :class:`src.utils.config.Config`.
        device: Target device (``'cuda'``, ``'cpu'``, or ``'cuda:0'``).

    Attributes:
        model: The loaded PyTorch model (eval mode).
        device: The torch device used for inference.
        config: Resolved configuration dictionary.

    Example:
        >>> inferencer = CaseInferencer("ckpt/best.pt", cfg, device="cuda:0")
        >>> result = inferencer.infer("ct.nii.gz")
        >>> result.keys()
        dict_keys(['heatmap', 'lesion_candidates', 'triage_score',
                    'uncertainty_score', 'uncertainty_map'])
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: Union[dict, Any],
        device: str = "cuda",
    ) -> None:
        from src.utils.config import Config

        # Resolve config
        if isinstance(config, Config):
            self.config: dict = config.to_dict()
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError(
                f"config must be dict or Config, got {type(config).__name__}"
            )

        # Device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU")
            device = "cpu"
        self.device = torch.device(device)

        # Load model
        self.checkpoint_path = checkpoint_path
        self.model = self._load_model()

        # Whether the model's forward output is already a probability in [0, 1]
        # (i.e. a final Sigmoid lives inside the model).  BaselineUNet's heads
        # apply Sigmoid, so post-processing must NOT apply it again.  Set after
        # _load_model, which may replace self.config["model"] from the ckpt.
        model_name = str(
            self.config.get("model", {}).get(
                "name", self.config.get("model", {}).get("type", "baseline_unet")
            )
        ).lower().replace("_", "")
        self.model_outputs_probabilities = model_name == "baselineunet"
        logger.info(
            "CaseInferencer initialised (device=%s, ckpt=%s)",
            self.device,
            checkpoint_path,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> nn.Module:
        """Build the model architecture and load checkpoint weights.

        Returns:
            Model in eval mode on ``self.device``.
        """
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        # Prefer the config embedded in the checkpoint so that the model
        # architecture and preprocessing always match the trained weights.
        # This prevents architecture drift between the training and inference
        # code paths (the original cause of the uniform-heatmap failure).
        ckpt_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
        if isinstance(ckpt_config, dict):
            if "model" in ckpt_config:
                self.config["model"] = ckpt_config["model"]
            if "preprocessing" in ckpt_config:
                # Only fill in preprocessing keys the caller did not override.
                merged = dict(ckpt_config["preprocessing"])
                merged.update(self.config.get("preprocessing", {}) or {})
                self.config["preprocessing"] = merged
            logger.info(
                "Using model/preprocessing config embedded in checkpoint "
                "(model=%s, preprocessing=%s)",
                self.config.get("model", {}).get("name"),
                self.config.get("preprocessing"),
            )

        model = self._build_model()

        # Support both raw state_dict and wrapped checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Load strictly.  The only weights we tolerate being absent are the
        # optional uncertainty head (not present in the baseline checkpoint).
        result = model.load_state_dict(state_dict, strict=False)
        unexpected = list(result.unexpected_keys)
        missing = [
            k for k in result.missing_keys if not k.startswith("uncertainty_head")
        ]
        if unexpected or missing:
            raise RuntimeError(
                "Checkpoint does not match the reconstructed architecture — "
                f"{len(missing)} missing and {len(unexpected)} unexpected keys. "
                "This indicates the inference model differs from the trained "
                "model.  First missing: "
                f"{missing[:3]}; first unexpected: {unexpected[:3]}"
            )
        n_loaded = len(state_dict) - len(unexpected)
        logger.info(
            "Loaded %d/%d checkpoint tensors into the model", n_loaded, len(state_dict)
        )

        # BaselineUNet returns a dict; wrap so the sliding-window inferer sees
        # a single heatmap tensor.
        model = _HeatmapOutputWrapper(model, key="heatmap")

        model = model.to(self.device)
        model.eval()
        logger.info("Model loaded from %s", self.checkpoint_path)
        return model

    def _build_model(self) -> nn.Module:
        """Build model architecture from config.

        Supports:
            - ``swin_unetr``: MONAI SwinUNETR
            - ``basic_unet``: MONAI BasicUNet
            - ``unet``: MONAI UNet

        Returns:
            Uninitialised model (weights will be loaded separately).

        Raises:
            ValueError: If model type is unsupported.
        """
        model_cfg = self.config.get("model", {})
        # Accept both "name" (training config key) and "type" (legacy inference
        # key).  Default to baseline_unet — the Model 0 architecture actually
        # trained — rather than a bare BasicUNet.
        model_type = str(
            model_cfg.get("name", model_cfg.get("type", "baseline_unet"))
        ).lower()
        # Normalise aliases: "basicunet" → "basic_unet", "swinunetr" → "swin_unetr"
        model_type_normalised = model_type.replace("_", "")

        in_channels = model_cfg.get("in_channels", 1)
        out_channels = model_cfg.get("out_channels", 1)
        spatial_dims = model_cfg.get("spatial_dims", 3)

        if model_type_normalised == "baselineunet":
            from src.models.baseline_unet import BaselineUNet

            features = model_cfg.get("features", (32, 32, 64, 128, 256, 32))
            if isinstance(features, list):
                features = tuple(features)
            dropout = model_cfg.get("dropout", 0.0)
            norm = model_cfg.get("norm", "instance")

            model = BaselineUNet(
                in_channels=in_channels,
                features=features,
                dropout=dropout,
                use_uncertainty=model_cfg.get("use_uncertainty", False),
                norm=norm,
            )
            logger.info(
                "Built baseline_unet model (features=%s, norm=%s)", features, norm
            )
            return model

        if model_type_normalised == "swinunetr":
            from monai.networks.nets import SwinUNETR  # type: ignore[import-untyped]

            img_size = model_cfg.get("img_size", (96, 96, 96))
            if isinstance(img_size, list):
                img_size = tuple(img_size)
            feature_size = model_cfg.get("feature_size", 48)
            drop_rate = model_cfg.get("drop_rate", 0.0)
            attn_drop_rate = model_cfg.get("attn_drop_rate", 0.0)
            dropout_path_rate = model_cfg.get("dropout_path_rate", 0.0)

            model = SwinUNETR(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=feature_size,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                dropout_path_rate=dropout_path_rate,
                spatial_dims=spatial_dims,
            )

        elif model_type_normalised == "basicunet":
            from monai.networks.nets import BasicUNet  # type: ignore[import-untyped]

            features = model_cfg.get(
                "features", (32, 32, 64, 128, 256, 32)
            )
            if isinstance(features, list):
                features = tuple(features)
            dropout = model_cfg.get("dropout", 0.0)

            model = BasicUNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                features=features,
                dropout=dropout,
            )

        elif model_type_normalised == "unet":
            from monai.networks.nets import UNet  # type: ignore[import-untyped]

            channels = model_cfg.get("channels", (16, 32, 64, 128, 256))
            if isinstance(channels, list):
                channels = tuple(channels)
            strides = model_cfg.get("strides", (2, 2, 2, 2))
            if isinstance(strides, list):
                strides = tuple(strides)

            model = UNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                strides=strides,
            )
        else:
            raise ValueError(
                f"Unsupported model type '{model_type}'. "
                f"Choose from: swin_unetr, basic_unet, unet."
            )

        logger.info("Built %s model (in=%d, out=%d)", model_type, in_channels, out_channels)
        return model

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, ct_path: str) -> Tuple[torch.Tensor, dict]:
        """Load and preprocess a CT volume.

        Steps:
            1. Load NIfTI with SimpleITK.
            2. Resample to isotropic spacing.
            3. Clip HU range and normalise to [0, 1].
            4. Convert to torch tensor (1×D×H×W).

        Args:
            ct_path: Path to the CT NIfTI file.

        Returns:
            Tuple of (input tensor, metadata dict).
        """
        try:
            import SimpleITK as sitk  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "SimpleITK is required for loading NIfTI files. "
                "Install with: pip install SimpleITK"
            ) from exc

        # Load
        ct_image = sitk.ReadImage(str(ct_path))
        ct_array = sitk.GetArrayFromImage(ct_image).astype(np.float32)

        metadata = {
            "original_spacing": ct_image.GetSpacing(),
            "original_origin": ct_image.GetOrigin(),
            "original_direction": ct_image.GetDirection(),
            "original_size": ct_image.GetSize(),
            "original_shape": ct_array.shape,
        }

        # Resample to target spacing
        preproc_cfg = self.config.get("preprocessing", {})
        target_spacing = preproc_cfg.get("target_spacing", [1.0, 1.0, 1.0])

        current_spacing = ct_image.GetSpacing()
        if list(current_spacing) != list(target_spacing):
            ct_image = self._resample_sitk(ct_image, target_spacing)
            ct_array = sitk.GetArrayFromImage(ct_image).astype(np.float32)

        # Crop to CT-only body ROI (H1 fix) — MUST match training preprocessing.
        # Performed on RAW HU (before clip/normalise), using the same body-mask
        # logic as scripts/prepare_data.py. The crop slices are recorded in the
        # metadata so the predicted heatmap can be pasted back to full CT
        # geometry (Task 12). Controlled by preprocessing.crop.method:
        #   'body'  -> CT-only body-mask bbox (reproducible, recommended)
        #   'none'/absent -> no crop (full volume; legacy behaviour)
        crop_cfg = preproc_cfg.get("crop", {})
        crop_method = crop_cfg.get("method", "body")
        crop_info = {
            "slices": [(0, s) for s in ct_array.shape],
            "original_shape": ct_array.shape,
            "method": "identity",
        }
        if crop_method == "body":
            from src.dataio.preprocess import crop_head_neck_roi

            margin = tuple(crop_cfg.get("margin", [10, 10, 10]))
            hu_threshold = float(crop_cfg.get("hu_threshold", -500.0))
            z_extent_mm = float(crop_cfg.get("z_extent_mm", 360.0))
            ct_array, crop_info = crop_head_neck_roi(
                ct_array,
                mask=None,
                margin=margin,
                method="body",
                hu_threshold=hu_threshold,
                z_extent_mm=z_extent_mm,
                z_spacing_mm=float(target_spacing[2]),
            )
        metadata["crop_info"] = crop_info

        # Clip HU
        hu_min = preproc_cfg.get("hu_min", -1024)
        hu_max = preproc_cfg.get("hu_max", 1024)
        ct_array = np.clip(ct_array, hu_min, hu_max)

        # Normalise to [0, 1]
        ct_array = (ct_array - hu_min) / (hu_max - hu_min + 1e-8)

        # To tensor: (1, D, H, W)
        tensor = torch.from_numpy(ct_array).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=torch.float32)

        logger.debug(
            "Preprocessed CT: shape=%s, spacing=%s",
            tensor.shape,
            target_spacing,
        )
        return tensor, metadata

    @staticmethod
    def _resample_sitk(
        image: Any,
        target_spacing: List[float],
    ) -> Any:
        """Resample a SimpleITK image to the target spacing.

        Args:
            image: SimpleITK image.
            target_spacing: Target voxel spacing [x, y, z].

        Returns:
            Resampled SimpleITK image.
        """
        import SimpleITK as sitk  # type: ignore[import-untyped]

        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        new_size = [
            int(round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(
                original_size, original_spacing, target_spacing
            )
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(float(sitk.GetArrayFromImage(image).min()))
        resampler.SetInterpolator(sitk.sitkLinear)

        return resampler.Execute(image)

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        output: torch.Tensor,
        metadata: dict,
    ) -> np.ndarray:
        """Convert raw model output to a heatmap array.

        Applies sigmoid (if needed) and squeezes batch/channel dimensions.

        Args:
            output: Raw model output tensor (B×C×D×H×W or B×1×D×H×W).
            metadata: Preprocessing metadata (used for potential resizing).

        Returns:
            3-D numpy heatmap in [0, 1].
        """
        with torch.no_grad():
            # The BaselineUNet HeatmapHead already applies a final Sigmoid, so
            # its output is a probability in [0, 1].  Applying sigmoid a second
            # time floors the map into [0.5, 0.73] and destroys contrast — this
            # was the cause of the "everything ≈ 0.5" heatmap.  Only apply the
            # activation when the model emits raw logits.
            if self.model_outputs_probabilities:
                heatmap = output.squeeze(0).squeeze(0)
            else:
                heatmap = torch.sigmoid(output).squeeze(0).squeeze(0)
            heatmap = heatmap.cpu().numpy().astype(np.float32)

        heatmap = np.clip(heatmap, 0.0, 1.0)
        logger.debug("Postprocessed heatmap: shape=%s, range=[%.3f, %.3f]",
                      heatmap.shape, heatmap.min(), heatmap.max())
        return heatmap

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer(self, ct_path: str) -> Dict[str, Any]:
        """Run sliding-window inference on a CT volume.

        Uses MONAI ``SlidingWindowInferer`` to process the volume in
        small patches (default 128³ with 50 % overlap), keeping VRAM
        usage bounded.  AMP (float16) is used when running on CUDA.

        Args:
            ct_path: Path to the CT NIfTI file.

        Returns:
            Dict with keys:
                - ``heatmap``: 3-D numpy array (metabolic risk map).
                - ``lesion_candidates``: list of candidate dicts.
                - ``triage_score``: float in [0, 1].
                - ``uncertainty_score``: float (0 for deterministic).
                - ``uncertainty_map``: None (deterministic mode).
        """
        from src.inference.postprocess import (
            threshold_heatmap,
            extract_connected_components,
            compute_triage_score,
        )

        logger.info("Running deterministic inference on %s", ct_path)
        t0 = time.time()

        # Preprocess
        t_step = time.time()
        input_tensor, metadata = self._preprocess(ct_path)
        logger.info("  [timing] preprocess: %.2fs", time.time() - t_step)

        # Forward pass — sliding window
        t_step = time.time()
        self.model.eval()
        output = self._sliding_window_forward(input_tensor)
        heatmap = self._postprocess(output, metadata)
        logger.info("  [timing] forward + postprocess: %.2fs", time.time() - t_step)

        # Post-process
        t_step = time.time()
        postproc_cfg = self.config.get("postprocessing", {})
        threshold = postproc_cfg.get("threshold", 0.5)
        min_component_size = postproc_cfg.get("min_component_size", 100)

        binary = threshold_heatmap(heatmap, threshold=threshold)
        candidates = extract_connected_components(
            binary, min_size=min_component_size
        )

        # Enrich candidates with actual heatmap intensities
        for cand in candidates:
            bbox = cand["bounding_box"]
            region = heatmap[bbox]
            cand["max_intensity"] = float(region.max())
            cand["mean_intensity"] = float(region.mean())

        triage = compute_triage_score(heatmap, candidates)
        logger.info("  [timing] postprocess (CC + triage): %.2fs", time.time() - t_step)

        elapsed = time.time() - t0
        logger.info("Inference completed in %.2fs (triage=%.4f)", elapsed, triage)

        return {
            "heatmap": heatmap,
            "lesion_candidates": candidates,
            "triage_score": triage,
            "uncertainty_score": 0.0,
            "uncertainty_map": None,
            "metadata": metadata,
            "elapsed_seconds": elapsed,
        }

    def _sliding_window_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run a forward pass using MONAI SlidingWindowInferer + AMP.

        Reads ``roi_size`` and ``overlap`` from config (falls back to
        128³ / 0.5).  Uses float16 AMP on CUDA for ~2× speedup.

        Args:
            input_tensor: Preprocessed CT tensor (B×1×D×H×W).

        Returns:
            Raw model output tensor (same spatial dims as input).
        """
        from monai.inferers import SlidingWindowInferer  # type: ignore[import-untyped]

        inf_cfg = self.config.get("inference", {})
        roi_size = inf_cfg.get("roi_size", [128, 128, 128])
        if isinstance(roi_size, list):
            roi_size = tuple(roi_size)
        overlap = inf_cfg.get("overlap", 0.25)
        sw_batch_size = inf_cfg.get("sw_batch_size", 4)

        inferer = SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            mode="gaussian",
            progress=True,
        )

        use_amp = self.device.type == "cuda"
        if use_amp:
            with torch.amp.autocast("cuda"):
                output = inferer(input_tensor, self.model)
        else:
            output = inferer(input_tensor, self.model)

        return output

    def infer_with_mc_dropout(
        self,
        ct_path: str,
        num_passes: int = 20,
    ) -> Dict[str, Any]:
        """Run MC Dropout inference for uncertainty estimation.

        Enables dropout at inference time and runs *num_passes* stochastic
        forward passes using sliding window.  The mean prediction is used
        as the heatmap and the voxel-wise standard deviation as the
        uncertainty map.

        Args:
            ct_path: Path to the CT NIfTI file.
            num_passes: Number of stochastic forward passes.

        Returns:
            Dict with keys:
                - ``heatmap``: Mean heatmap (3-D numpy).
                - ``lesion_candidates``: list of candidate dicts.
                - ``triage_score``: float in [0, 1].
                - ``uncertainty_score``: scalar uncertainty (mean of map).
                - ``uncertainty_map``: 3-D numpy (voxel-wise std dev).
        """
        from src.inference.postprocess import (
            threshold_heatmap,
            extract_connected_components,
            compute_triage_score,
        )

        logger.info(
            "Running MC Dropout inference (%d passes) on %s",
            num_passes,
            ct_path,
        )
        t0 = time.time()

        # Preprocess
        input_tensor, metadata = self._preprocess(ct_path)

        # Enable dropout for MC sampling
        self._enable_dropout(self.model)

        predictions: List[np.ndarray] = []
        for i in range(num_passes):
            with torch.no_grad():
                output = self._sliding_window_forward(input_tensor)
            pred = self._postprocess(output, metadata)
            predictions.append(pred)
            if (i + 1) % 5 == 0:
                logger.debug("MC pass %d/%d completed", i + 1, num_passes)

        # Restore eval mode (dropout off)
        self.model.eval()

        # Aggregate
        stacked = np.stack(predictions, axis=0)  # (N, D, H, W)
        heatmap = stacked.mean(axis=0)
        uncertainty_map = stacked.std(axis=0)
        uncertainty_score = float(uncertainty_map.mean())

        # Post-process
        postproc_cfg = self.config.get("postprocessing", {})
        threshold = postproc_cfg.get("threshold", 0.5)
        min_component_size = postproc_cfg.get("min_component_size", 100)

        binary = threshold_heatmap(heatmap, threshold=threshold)
        candidates = extract_connected_components(
            binary, min_size=min_component_size
        )
        for cand in candidates:
            bbox = cand["bounding_box"]
            region = heatmap[bbox]
            cand["max_intensity"] = float(region.max())
            cand["mean_intensity"] = float(region.mean())

        triage = compute_triage_score(
            heatmap,
            candidates,
            uncertainty_map=uncertainty_map,
        )

        elapsed = time.time() - t0
        logger.info(
            "MC Dropout inference completed in %.2fs "
            "(triage=%.4f, uncertainty=%.4f)",
            elapsed,
            triage,
            uncertainty_score,
        )

        return {
            "heatmap": heatmap,
            "lesion_candidates": candidates,
            "triage_score": triage,
            "uncertainty_score": uncertainty_score,
            "uncertainty_map": uncertainty_map,
            "metadata": metadata,
            "elapsed_seconds": elapsed,
        }

    @staticmethod
    def _enable_dropout(model: nn.Module) -> None:
        """Set all Dropout layers to training mode for MC sampling.

        Args:
            model: PyTorch model. Dropout modules are switched to
                ``.train()`` while everything else stays in ``.eval()``.
        """
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    def save_outputs(
        self,
        results: Dict[str, Any],
        output_dir: str,
        *,
        reference_ct_path: Optional[str] = None,
    ) -> None:
        """Save inference results to disk.

        Saves:
            - ``heatmap.nii.gz``
            - ``lesion_candidates.nii.gz`` (thresholded binary)
            - ``uncertainty_map.nii.gz`` (if available)
            - ``summary.json``

        Args:
            results: Output of ``infer`` or ``infer_with_mc_dropout``.
            output_dir: Directory to save outputs.
            reference_ct_path: Optional path to reference CT for copying
                spatial metadata (spacing, origin, direction).
        """
        try:
            import SimpleITK as sitk  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "SimpleITK is required for saving NIfTI outputs. "
                "Install with: pip install SimpleITK"
            ) from exc

        from src.utils.io import ensure_dir

        out = ensure_dir(output_dir)

        # Reference spatial info
        if reference_ct_path is not None:
            ref_image = sitk.ReadImage(str(reference_ct_path))
        else:
            ref_image = None

        def _resample_to_reference(
            array: np.ndarray,
            ref: Any,
        ) -> np.ndarray:
            """Resample a 3-D array back to the reference image geometry.

            This ensures the output NIfTI has the same dimensions as the
            original CT so it can be overlaid directly in viewers like
            ITK-SNAP.
            """
            src_image = sitk.GetImageFromArray(array)
            # Inherit spacing from preprocessing target (isotropic)
            preproc_cfg = self.config.get("preprocessing", {})
            target_spacing = preproc_cfg.get("target_spacing", [1.0, 1.0, 1.0])
            src_image.SetSpacing(target_spacing)
            src_image.SetOrigin(ref.GetOrigin())
            src_image.SetDirection(ref.GetDirection())

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(ref)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0.0)
            resampled = resampler.Execute(src_image)
            return sitk.GetArrayFromImage(resampled).astype(array.dtype)

        def _save_nifti(array: np.ndarray, name: str) -> None:
            # Resample back to original CT space if dimensions differ
            if ref_image is not None:
                ref_size = ref_image.GetSize()  # (W, H, D) in SimpleITK
                arr_size = tuple(reversed(array.shape))  # numpy (D, H, W) → (W, H, D)
                if arr_size != ref_size:
                    logger.info(
                        "Resampling %s from %s to %s (original CT space)",
                        name, arr_size, ref_size,
                    )
                    array = _resample_to_reference(array, ref_image)

            image = sitk.GetImageFromArray(array)
            if ref_image is not None:
                if image.GetSize() == ref_image.GetSize():
                    image.CopyInformation(ref_image)
            sitk.WriteImage(image, str(out / name))
            logger.info("Saved %s to %s", name, out / name)

        # Heatmap
        _save_nifti(results["heatmap"].astype(np.float32), "heatmap.nii.gz")

        # Lesion candidates binary mask
        postproc_cfg = self.config.get("postprocessing", {})
        threshold = postproc_cfg.get("threshold", 0.5)
        from src.inference.postprocess import threshold_heatmap

        binary = threshold_heatmap(results["heatmap"], threshold=threshold)
        _save_nifti(binary, "lesion_candidates.nii.gz")

        # Uncertainty map
        if results.get("uncertainty_map") is not None:
            _save_nifti(
                results["uncertainty_map"].astype(np.float32),
                "uncertainty_map.nii.gz",
            )

        # Summary JSON
        # Make candidates JSON-serialisable (remove slice objects)
        serialisable_candidates = []
        for cand in results["lesion_candidates"]:
            sc = {k: v for k, v in cand.items() if k != "bounding_box"}
            if "bounding_box" in cand:
                bbox = cand["bounding_box"]
                sc["bounding_box"] = [
                    {"start": s.start, "stop": s.stop, "step": s.step}
                    for s in bbox
                ]
            serialisable_candidates.append(sc)

        summary = {
            "triage_score": results["triage_score"],
            "uncertainty_score": results["uncertainty_score"],
            "num_lesion_candidates": len(results["lesion_candidates"]),
            "lesion_candidates": serialisable_candidates,
            "heatmap_shape": list(results["heatmap"].shape),
            "elapsed_seconds": results.get("elapsed_seconds"),
        }

        summary_path = out / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
        logger.info("Saved summary to %s", summary_path)


# ======================================================================
# CLI entry point
# ======================================================================


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="CT2MAP-HN: Single-case metabolic risk heatmap inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ct",
        required=True,
        help="Path to the input CT NIfTI file (.nii.gz).",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to the model checkpoint (.pt).",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--mc-passes",
        type=int,
        default=0,
        help="Number of MC Dropout passes (0 = deterministic).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for single-case inference.

    Args:
        argv: Optional argument list.
    """
    from src.utils.config import load_config
    from src.utils.logger import setup_logger

    args = _parse_args(argv)
    setup_logger("ct2map", level="INFO")

    config = load_config(args.config)
    inferencer = CaseInferencer(args.ckpt, config, device=args.device)

    if args.mc_passes > 0:
        results = inferencer.infer_with_mc_dropout(
            args.ct, num_passes=args.mc_passes
        )
    else:
        results = inferencer.infer(args.ct)

    inferencer.save_outputs(results, args.out, reference_ct_path=args.ct)

    # Print recommendation
    from src.inference.postprocess import compute_triage_recommendation

    recommendation = compute_triage_recommendation(
        results["triage_score"],
        results["uncertainty_score"],
    )
    print("\n" + "=" * 60)
    print(recommendation)
    print("=" * 60)


if __name__ == "__main__":
    main()
