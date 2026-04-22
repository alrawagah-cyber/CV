"""ONNX Runtime wrappers for L2 and L3 inference.

When `.onnx` weight files are present alongside the `.pt` files, the
ClaimAssessor can use these wrappers for 2-3x faster CPU inference.

L1 (YOLO) already supports ONNX natively via ``ultralytics.YOLO("model.onnx")``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class OnnxDamageClassifier:
    """Drop-in replacement for DamageTypeClassifier.predict_proba using ONNX RT."""

    def __init__(self, onnx_path: str | Path, classes: list[str]):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self.classes = classes
        self.input_name = self.session.get_inputs()[0].name
        meta = self.session.get_inputs()[0]
        self.input_size = meta.shape[-1] if len(meta.shape) == 4 else 384
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        logger.info("Loaded ONNX L2 model from %s (input=%s)", onnx_path, meta.shape)

    def predict_proba(self, x) -> np.ndarray:
        """x: torch.Tensor [B,3,H,W] → numpy [B, num_classes] sigmoid probs."""
        x_np = x.cpu().numpy() if hasattr(x, "numpy") else np.asarray(x)
        (output,) = self.session.run(None, {self.input_name: x_np})
        return _sigmoid(output)


class OnnxSeverityAssessor:
    """Drop-in replacement for SeverityAssessor.predict using ONNX RT."""

    def __init__(self, onnx_path: str | Path, grades: list[str]):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self.grades = grades
        self.num_classes = len(grades)
        self.input_name = self.session.get_inputs()[0].name
        meta = self.session.get_inputs()[0]
        self.input_size = meta.shape[-1] if len(meta.shape) == 4 else 384
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        logger.info("Loaded ONNX L3 model from %s (input=%s)", onnx_path, meta.shape)

    def predict(self, x) -> list:
        """x: torch.Tensor [B,3,H,W] → list of SeverityOutput-like dicts."""
        from models.layer3_severity import SeverityOutput

        x_np = x.cpu().numpy() if hasattr(x, "numpy") else np.asarray(x)

        outputs = self.session.run(None, {self.input_name: x_np})
        # Model exports two outputs: ordinal_logits [B, K-1] and repair_logit [B, 1]
        if len(outputs) == 2:
            ordinal_logits, repair_logits = outputs
        else:
            ordinal_logits = outputs[0]
            repair_logits = np.zeros((ordinal_logits.shape[0], 1))

        results = []
        for i in range(ordinal_logits.shape[0]):
            probs = _sigmoid(ordinal_logits[i])
            rank = _coral_rank(probs)
            grade = self.grades[rank]
            grade_conf = float(_grade_confidence(probs, rank))
            sev_probs = {g: float(p) for g, p in zip(self.grades, _grade_probs(probs), strict=False)}
            rep = float(
                _sigmoid(repair_logits[i]).item() if repair_logits.ndim > 1 else _sigmoid(repair_logits[i])
            )
            results.append(
                SeverityOutput(
                    grade=grade,
                    grade_index=rank,
                    grade_confidence=grade_conf,
                    severity_probs=sev_probs,
                    repair_probability=1.0 - rep,
                    replace_probability=rep,
                    recommendation="replace" if rep > 0.5 else "repair",
                )
            )
        return results


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _coral_rank(cumulative_probs: np.ndarray) -> int:
    return int((cumulative_probs > 0.5).sum())


def _grade_confidence(cumulative_probs: np.ndarray, rank: int) -> float:
    if rank == 0:
        return float(1.0 - cumulative_probs[0])
    if rank >= len(cumulative_probs):
        return float(cumulative_probs[-1])
    return float(
        cumulative_probs[rank - 1] - (cumulative_probs[rank] if rank < len(cumulative_probs) else 0.0)
    )


def _grade_probs(cumulative_probs: np.ndarray) -> list[float]:
    """Convert cumulative sigmoid probs to per-grade probabilities."""
    K = len(cumulative_probs) + 1
    out = []
    for k in range(K):
        if k == 0:
            out.append(1.0 - cumulative_probs[0])
        elif k < K - 1:
            out.append(cumulative_probs[k - 1] - cumulative_probs[k])
        else:
            out.append(cumulative_probs[-1])
    return [max(0.0, p) for p in out]
