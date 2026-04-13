"""Tiny model registry for A/B testing and version-pinning from config.

Example config (configs/inference.yaml):

    layer1:
      name: part_detector
      version: yolov8x_v1
      weights: yolov8x.pt
    layer2:
      name: damage_type
      version: convnextv2_large_v1
      weights: null   # pretrained only
    layer3:
      name: severity
      version: swinv2_large_v1
      weights: null
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from models.layer1_detector import PartDetector
from models.layer2_damage import DamageTypeClassifier
from models.layer3_severity import SeverityAssessor

Builder = Callable[..., Any]


MODEL_REGISTRY: dict[str, Builder] = {
    "part_detector": PartDetector,
    "damage_type": DamageTypeClassifier,
    "severity": SeverityAssessor,
}


def build_model(name: str, **kwargs: Any) -> Any:
    """Instantiate a registered model by name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def register_model(name: str, builder: Builder) -> None:
    """Register a custom model factory (e.g. distilled student, A/B variant)."""
    MODEL_REGISTRY[name] = builder
