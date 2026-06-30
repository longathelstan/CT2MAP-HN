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
        model = self._build_model()
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        # Support both raw state_dict and wrapped checkpoint
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
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
        model_type = model_cfg.get("type", "basic_unet").lower()

        in_channels = model_cfg.get("in_channels", 1)
        out_channels = model_cfg.get("out_channels", 1)
        spatial_dims = model_cfg.get("spatial_dims", 3)

        if model_type == "swin_unetr":
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

        elif model_type == "basic_unet":
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

        elif model_type == "unet":
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
        resampler.SetInterpolator(sitk.sitkBSpline)

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
            # Sigmoid to [0, 1]
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
        """Run single-pass inference on a CT volume.

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
        input_tensor, metadata = self._preprocess(ct_path)

        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        heatmap = self._postprocess(output, metadata)

        # Post-process
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

    def infer_with_mc_dropout(
        self,
        ct_path: str,
        num_passes: int = 20,
    ) -> Dict[str, Any]:
        """Run MC Dropout inference for uncertainty estimation.

        Enables dropout at inference time and runs *num_passes* stochastic
        forward passes.  The mean prediction is used as the heatmap and the
        voxel-wise standard deviation as the uncertainty map.

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
                output = self.model(input_tensor)
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

        def _save_nifti(array: np.ndarray, name: str) -> None:
            image = sitk.GetImageFromArray(array)
            if ref_image is not None:
                # Only copy metadata if shapes match
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
