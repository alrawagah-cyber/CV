"""Model architectures for the 3-layer car damage assessment pipeline."""

from models.heads import CoralOrdinalHead, MultiLabelHead, coral_loss
from models.layer1_detector import Detection, PartDetector
from models.layer2_damage import DamageTypeClassifier
from models.layer3_severity import SeverityAssessor, SeverityOutput
from models.registry import MODEL_REGISTRY, build_model

__all__ = [
    "CoralOrdinalHead",
    "MultiLabelHead",
    "coral_loss",
    "Detection",
    "PartDetector",
    "DamageTypeClassifier",
    "SeverityAssessor",
    "SeverityOutput",
    "MODEL_REGISTRY",
    "build_model",
]
