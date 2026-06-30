"""
CT2MAP-HN Models Package.

Exports all model classes and a factory function ``build_model(config)``
that instantiates the requested architecture from a YAML configuration dict.
"""

from src.models.baseline_unet import BaselineUNet
from src.models.swin_unetr_student import SwinUNETRStudent
from src.models.teacher_encoder import TeacherEncoder
from src.models.heads import HeatmapHead, LesionHead, TriageHead, UncertaintyHead
from src.models.losses import (
    HeatmapLoss,
    LesionLoss,
    TriageLoss,
    DistillationLoss,
    CombinedLoss,
    CombinedDistillLoss,
)

__all__ = [
    "BaselineUNet",
    "SwinUNETRStudent",
    "TeacherEncoder",
    "HeatmapHead",
    "LesionHead",
    "TriageHead",
    "UncertaintyHead",
    "HeatmapLoss",
    "LesionLoss",
    "TriageLoss",
    "DistillationLoss",
    "CombinedLoss",
    "CombinedDistillLoss",
    "build_model",
]

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: dict[str, type] = {
    "baseline_unet": BaselineUNet,
    "swin_unetr_student": SwinUNETRStudent,
    "teacher_encoder": TeacherEncoder,
}


def build_model(config: dict) -> "BaselineUNet | SwinUNETRStudent | TeacherEncoder":
    """Factory function that builds a model from a configuration dictionary.

    The ``config`` dict **must** contain a ``model`` sub-dict with at least a
    ``name`` key whose value matches one of the registered model names:

    * ``"baseline_unet"``
    * ``"swin_unetr_student"``
    * ``"teacher_encoder"``

    All remaining keys inside ``config["model"]`` (except ``"name"``) are
    forwarded as keyword arguments to the corresponding model constructor.

    Args:
        config: Full experiment configuration dictionary.  Expected structure::

            model:
              name: baseline_unet
              in_channels: 1
              features: [32, 64, 128, 256]
              dropout: 0.1
              ...

    Returns:
        An instantiated ``nn.Module`` subclass.

    Raises:
        ValueError: If ``config["model"]["name"]`` is not in the registry.
        KeyError: If required keys are missing from *config*.
    """
    model_cfg: dict = config["model"]
    model_name: str = model_cfg["name"]

    if model_name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model name '{model_name}'. Available: {available}"
        )

    # Pop 'name' so remaining keys are constructor kwargs.
    kwargs = {k: v for k, v in model_cfg.items() if k != "name"}
    return _MODEL_REGISTRY[model_name](**kwargs)
