"""ClaimAssessor: end-to-end orchestrator for the 3-layer pipeline.

Usage:
    from inference.claim_assessor import ClaimAssessor
    assessor = ClaimAssessor.from_config("configs/inference.yaml")
    report = assessor.assess("data/samples/test_car.jpg")

The assessor:
    1. Runs Layer 1 (part detector) on the full image.
    2. Crops each detection and runs Layer 2 (damage type) in one batch.
    3. Runs Layer 3 (severity + repair/replace) on the same crops.
    4. Aggregates into a single JSON-serializable ClaimReport dict.

If task-specific fine-tuned weights are not provided, the pipeline still
*runs* using pretrained backbones + untrained classifier heads. Such reports
are flagged `pretrained_baseline=true` so downstream consumers can discard or
mark them as baseline demos.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from inference.batching import chunked
from inference.postprocessing import build_part_assessment, build_report
from inference.preprocessing import (
    batch_tensor_from_crops,
    crop,
    expand_bbox,
    load_image,
)
from models.layer1_detector import DEFAULT_PART_CLASSES, Detection, PartDetector
from models.layer2_damage import DEFAULT_DAMAGE_CLASSES, DamageTypeClassifier
from models.layer3_severity import DEFAULT_SEVERITY_GRADES, SeverityAssessor

logger = logging.getLogger(__name__)


@dataclass
class AssessorConfig:
    # Layer 1
    l1_weights: str = "yolov8x.pt"
    l1_conf: float = 0.25
    l1_iou: float = 0.45
    l1_img_size: int = 640
    l1_classes: list[str] = field(default_factory=lambda: list(DEFAULT_PART_CLASSES))
    l1_use_default_classes_mapping: bool = True

    # Layer 2
    l2_backbone: str = "convnextv2_large.fcmae_ft_in22k_in1k"
    l2_weights: str | None = None
    l2_classes: list[str] = field(default_factory=lambda: list(DEFAULT_DAMAGE_CLASSES))
    l2_threshold: float = 0.5

    # Layer 3
    l3_backbone: str = "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k"
    l3_weights: str | None = None
    l3_grades: list[str] = field(default_factory=lambda: list(DEFAULT_SEVERITY_GRADES))

    # Runtime
    device: str = "cuda"
    batch_size: int = 8
    crop_margin: float = 0.1
    max_parts: int = 32
    use_rule_override: bool = False

    # Versioning for the report
    versions: dict[str, str] = field(
        default_factory=lambda: {
            "layer1": "yolov8x_v1",
            "layer2": "convnextv2_large_v1",
            "layer3": "swinv2_large_v1",
        }
    )

    @classmethod
    def from_file(cls, path: str | Path) -> AssessorConfig:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        l1 = cfg.get("layer1", {})
        l2 = cfg.get("layer2", {})
        l3 = cfg.get("layer3", {})
        runtime = cfg.get("runtime", {})
        versions = {
            "layer1": l1.get("version", "yolov8x_v1"),
            "layer2": l2.get("version", "convnextv2_large_v1"),
            "layer3": l3.get("version", "swinv2_large_v1"),
        }
        return cls(
            l1_weights=l1.get("weights", "yolov8x.pt"),
            l1_conf=l1.get("conf_threshold", 0.25),
            l1_iou=l1.get("iou_threshold", 0.45),
            l1_img_size=l1.get("img_size", 640),
            l1_classes=l1.get("classes", list(DEFAULT_PART_CLASSES)),
            l1_use_default_classes_mapping=l1.get("use_default_classes_mapping", True),
            l2_backbone=l2.get("backbone", "convnextv2_large.fcmae_ft_in22k_in1k"),
            l2_weights=l2.get("weights"),
            l2_classes=l2.get("classes", list(DEFAULT_DAMAGE_CLASSES)),
            l2_threshold=l2.get("threshold", 0.5),
            l3_backbone=l3.get("backbone", "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k"),
            l3_weights=l3.get("weights"),
            l3_grades=l3.get("grades", list(DEFAULT_SEVERITY_GRADES)),
            device=runtime.get("device", "cuda"),
            batch_size=runtime.get("batch_size", 8),
            crop_margin=runtime.get("crop_margin", 0.1),
            max_parts=runtime.get("max_parts", 32),
            use_rule_override=runtime.get("use_rule_override", False),
            versions=versions,
        )


class ClaimAssessor:
    """Chains Layer 1 -> Layer 2 -> Layer 3 end-to-end."""

    def __init__(self, cfg: AssessorConfig):
        self.cfg = cfg
        # Resolve device (CPU fallback if CUDA unavailable)
        requested = cfg.device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
            requested = "cpu"
        self.device = torch.device(requested)

        logger.info("Loading Layer 1 (detector)...")
        self.detector = PartDetector(
            weights=cfg.l1_weights,
            classes=cfg.l1_classes,
            conf_threshold=cfg.l1_conf,
            iou_threshold=cfg.l1_iou,
            device=requested,
            img_size=cfg.l1_img_size,
        )

        logger.info("Loading Layer 2 (damage type)...")
        self.damage_model = (
            DamageTypeClassifier(
                backbone=cfg.l2_backbone,
                classes=cfg.l2_classes,
                pretrained=cfg.l2_weights is None,  # if no weights provided, use pretrained backbone
            )
            .to(self.device)
            .eval()
        )
        self._l2_is_baseline = cfg.l2_weights is None
        if cfg.l2_weights:
            logger.info("Loading Layer 2 weights from %s", cfg.l2_weights)
            ckpt = torch.load(cfg.l2_weights, map_location=self.device)
            state = ckpt.get("state_dict", ckpt)
            self.damage_model.load_state_dict(state, strict=False)

        logger.info("Loading Layer 3 (severity)...")
        self.severity_model = (
            SeverityAssessor(
                backbone=cfg.l3_backbone,
                grades=cfg.l3_grades,
                pretrained=cfg.l3_weights is None,
            )
            .to(self.device)
            .eval()
        )
        self._l3_is_baseline = cfg.l3_weights is None
        if cfg.l3_weights:
            logger.info("Loading Layer 3 weights from %s", cfg.l3_weights)
            ckpt = torch.load(cfg.l3_weights, map_location=self.device)
            state = ckpt.get("state_dict", ckpt)
            self.severity_model.load_state_dict(state, strict=False)

        # Sanity: report uses "pretrained_baseline" if *either* classifier is not fine-tuned
        self.pretrained_baseline = self._l2_is_baseline or self._l3_is_baseline

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, path: str | Path) -> ClaimAssessor:
        return cls(AssessorConfig.from_file(path))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def assess(self, image: str | Path | np.ndarray | bytes, image_id: str | None = None) -> dict[str, Any]:
        return self.assess_batch([image], image_ids=[image_id] if image_id else None)[0]

    def assess_batch(
        self,
        images: list[str | Path | np.ndarray | bytes],
        image_ids: list[str | None] | None = None,
    ) -> list[dict[str, Any]]:
        if image_ids is None:
            image_ids = [None] * len(images)
        assert len(image_ids) == len(images)

        reports: list[dict[str, Any]] = []
        for src, maybe_id in zip(images, image_ids, strict=True):
            t0 = time.time()
            img = load_image(src)
            img_id = maybe_id or _derive_image_id(src, img)
            h, w = img.shape[:2]
            warnings: list[str] = []

            # --- Layer 1
            detections_batches = self.detector.predict([img])
            detections: list[Detection] = detections_batches[0]
            if self.cfg.max_parts and len(detections) > self.cfg.max_parts:
                warnings.append(
                    f"Truncated {len(detections)} detections to top-{self.cfg.max_parts} by confidence."
                )
                detections = sorted(detections, key=lambda d: -d.confidence)[: self.cfg.max_parts]

            if not detections:
                reports.append(
                    build_report(
                        image_id=img_id,
                        image_width=w,
                        image_height=h,
                        parts=[],
                        pretrained_baseline=self.pretrained_baseline,
                        model_versions=self.cfg.versions,
                        warnings=["No parts detected above confidence threshold."],
                    )
                )
                continue

            # --- Expand + crop
            crops_np: list[np.ndarray] = []
            for d in detections:
                ex = expand_bbox(d.bbox_xyxy_px, w, h, margin=self.cfg.crop_margin)
                crops_np.append(crop(img, ex))

            # --- Layer 2
            l2_input = batch_tensor_from_crops(
                crops_np,
                size=self.damage_model.input_size,
                mean=self.damage_model.mean,
                std=self.damage_model.std,
            ).to(self.device)

            l2_probs_list: list[np.ndarray] = []
            for chunk in chunked(l2_input, self.cfg.batch_size):
                p = self.damage_model.predict_proba(chunk).cpu().numpy()
                l2_probs_list.append(p)
            l2_probs = (
                np.concatenate(l2_probs_list, axis=0)
                if l2_probs_list
                else np.zeros((0, len(self.cfg.l2_classes)))
            )

            # --- Layer 3
            l3_input = batch_tensor_from_crops(
                crops_np,
                size=self.severity_model.input_size,
                mean=self.severity_model.mean,
                std=self.severity_model.std,
            ).to(self.device)

            severity_outputs = []
            for chunk in chunked(l3_input, self.cfg.batch_size):
                severity_outputs.extend(self.severity_model.predict(chunk))

            # --- Assemble report
            parts = []
            for det, dam_probs, sev in zip(detections, l2_probs, severity_outputs, strict=True):
                parts.append(
                    build_part_assessment(
                        detection=det.to_dict(),
                        damage_probs={cls: float(dam_probs[j]) for j, cls in enumerate(self.cfg.l2_classes)},
                        damage_threshold=self.cfg.l2_threshold,
                        severity={
                            "grade": sev.grade,
                            "grade_index": sev.grade_index,
                            "grade_confidence": sev.grade_confidence,
                            "severity_probs": sev.severity_probs,
                            "repair_probability": sev.repair_probability,
                            "replace_probability": sev.replace_probability,
                            "recommendation": sev.recommendation,
                        },
                        pretrained_baseline=self.pretrained_baseline,
                        use_rule_override=self.cfg.use_rule_override,
                    )
                )

            elapsed_ms = int((time.time() - t0) * 1000)
            report = build_report(
                image_id=img_id,
                image_width=w,
                image_height=h,
                parts=parts,
                pretrained_baseline=self.pretrained_baseline,
                model_versions=self.cfg.versions,
                warnings=warnings,
            )
            report["inference_ms"] = elapsed_ms
            reports.append(report)

        return reports


def _derive_image_id(src: Any, img: np.ndarray) -> str:
    if isinstance(src, str | Path):
        return Path(str(src)).name
    h = hashlib.sha1(img.tobytes()[:4096]).hexdigest()[:12]
    return f"image_{h}"
