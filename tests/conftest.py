"""Pytest fixtures. Keep CI light: stub out heavy models so tests don't require
GPU, real weights, or internet. Tests marked 'integration' exercise the real
models and should be run locally with `pytest -m integration`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

# Make the repo root importable.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# Stub classes for lightweight CPU testing.
# --------------------------------------------------------------------------- #
class StubDetection:
    part = "bumper"
    class_id = 0
    confidence = 0.85
    bbox_xyxy_norm = (0.20, 0.40, 0.60, 0.65)
    bbox_xyxy_px = (128, 192, 384, 312)
    image_width = 640
    image_height = 480

    def to_dict(self) -> dict[str, Any]:
        return {
            "part": self.part,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "bbox_xyxy_norm": self.bbox_xyxy_norm,
            "bbox_xyxy_px": self.bbox_xyxy_px,
            "image_width": self.image_width,
            "image_height": self.image_height,
        }


class StubDetector:
    """Drop-in replacement for PartDetector that returns a single bbox."""

    def __init__(self, *args, **kwargs):  # noqa: D401
        self.classes = kwargs.get("classes", ["bumper"])

    def predict(self, images, **kwargs):
        if isinstance(images, list):
            return [[StubDetection()] for _ in images]
        return [[StubDetection()]]

    def train(self, **kwargs):
        return None

    def export(self, format="onnx", **kwargs):
        return f"stub.{format}"


class StubClassifier(torch.nn.Module):
    """Tiny pretend-classifier for L2/L3 that yields deterministic outputs."""

    def __init__(self, num_classes: int, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        super().__init__()
        self.num_classes = num_classes
        self.mean = mean
        self.std = std
        self.input_size = 64
        self.fc = torch.nn.Conv2d(3, num_classes, kernel_size=1)

    def forward(self, x):
        out = self.fc(x).mean(dim=(-2, -1))
        return out


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture
def tmp_image_path(tmp_path: Path, sample_rgb_image: np.ndarray) -> Path:
    from PIL import Image

    p = tmp_path / "sample.jpg"
    Image.fromarray(sample_rgb_image).save(p, quality=85)
    return p


@pytest.fixture
def stub_assessor(monkeypatch):
    """Build a ClaimAssessor where all three layers are cheap stubs."""
    import inference.claim_assessor as mod
    from models.layer1_detector import DEFAULT_PART_CLASSES
    from models.layer2_damage import DEFAULT_DAMAGE_CLASSES
    from models.layer3_severity import DEFAULT_SEVERITY_GRADES, SeverityOutput

    monkeypatch.setattr(mod, "PartDetector", StubDetector)

    class L2Stub(StubClassifier):
        def __init__(self, *a, **kw):
            super().__init__(num_classes=len(DEFAULT_DAMAGE_CLASSES))
            self.classes = DEFAULT_DAMAGE_CLASSES

        def predict_proba(self, x):
            with torch.no_grad():
                return torch.sigmoid(self.forward(x))

    class L3Stub(StubClassifier):
        def __init__(self, *a, **kw):
            super().__init__(num_classes=len(DEFAULT_SEVERITY_GRADES) - 1)
            self.grades = DEFAULT_SEVERITY_GRADES
            self.num_classes = len(DEFAULT_SEVERITY_GRADES)

        def predict(self, x):
            B = x.shape[0]
            return [
                SeverityOutput(
                    grade="minor",
                    grade_index=0,
                    grade_confidence=0.7,
                    severity_probs={g: (0.7 if g == "minor" else 0.1) for g in self.grades},
                    repair_probability=0.7,
                    replace_probability=0.3,
                    recommendation="repair",
                )
                for _ in range(B)
            ]

    monkeypatch.setattr(mod, "DamageTypeClassifier", L2Stub)
    monkeypatch.setattr(mod, "SeverityAssessor", L3Stub)

    cfg = mod.AssessorConfig(
        device="cpu",
        l1_classes=list(DEFAULT_PART_CLASSES),
        l2_classes=list(DEFAULT_DAMAGE_CLASSES),
        l3_grades=list(DEFAULT_SEVERITY_GRADES),
        batch_size=2,
    )
    return mod.ClaimAssessor(cfg)


# Provide a CDP_TEST_MODE env for code paths that want to branch for tests.
os.environ.setdefault("CDP_TEST_MODE", "1")
