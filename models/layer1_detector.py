"""Layer 1: Car part detection using Ultralytics YOLO.

The base model defaults to YOLOv8x. YOLOv11x is supported by pointing
`weights` to `yolo11x.pt` (config flag `arch: yolo11x`). The detector
outputs per-image lists of `Detection` dataclasses with normalized box
coordinates, class name, class index, and confidence score.

Out-of-the-box (no fine-tuning) the model runs with COCO-pretrained
weights, where the class vocabulary is the 80 COCO classes. For real use
the user fine-tunes on a car-part dataset with the 13 custom classes
defined in configs/layer1.yaml.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_PART_CLASSES: list[str] = [
    "bumper",
    "hood",
    "fender",
    "door",
    "windshield",
    "headlight",
    "taillight",
    "mirror",
    "trunk",
    "roof",
    "quarter_panel",
    "grille",
    "wheel",
]


@dataclass
class Detection:
    """A single bounding-box detection in normalized [0, 1] xyxy coords."""

    part: str  # class name
    class_id: int  # class index
    confidence: float  # [0, 1]
    bbox_xyxy_norm: tuple[float, float, float, float]
    bbox_xyxy_px: tuple[int, int, int, int]
    image_width: int
    image_height: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PartDetector:
    """Thin wrapper around ultralytics.YOLO.

    Parameters
    ----------
    weights:
        Path to a .pt weights file OR an ultralytics hub tag
        (e.g. "yolov8x.pt", "yolo11x.pt"). If the file does not exist
        locally, ultralytics will attempt to download it on first load.
    classes:
        Ordered class list. Must match the trained model. If None,
        `DEFAULT_PART_CLASSES` is used (for fine-tuned weights).
        With COCO-pretrained weights the detector reports its built-in
        class names, not this list.
    conf_threshold, iou_threshold:
        Inference thresholds.
    device:
        "cuda", "cpu", "cuda:0", or "mps". If None, ultralytics autopicks.
    """

    def __init__(
        self,
        weights: str | Path = "yolov8x.pt",
        classes: list[str] | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str | None = None,
        img_size: int = 640,
    ):
        # Lazy import so the module can be imported in lightweight test envs
        from ultralytics import YOLO

        self.weights = str(weights)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.img_size = img_size
        self.classes = classes or DEFAULT_PART_CLASSES
        self.model = YOLO(self.weights)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict(
        self,
        images: np.ndarray | list[np.ndarray] | str | Path | list[str | Path],
        conf: float | None = None,
        iou: float | None = None,
    ) -> list[list[Detection]]:
        """Run detection and return Detection lists (one list per image)."""
        results = self.model.predict(
            source=images,
            conf=conf if conf is not None else self.conf_threshold,
            iou=iou if iou is not None else self.iou_threshold,
            device=self.device,
            imgsz=self.img_size,
            verbose=False,
        )

        model_names = self.model.names  # dict[int, str]
        batch_out: list[list[Detection]] = []

        for r in results:
            h, w = int(r.orig_shape[0]), int(r.orig_shape[1])
            dets: list[Detection] = []
            if r.boxes is None or len(r.boxes) == 0:
                batch_out.append(dets)
                continue

            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)

            for box, conf_v, cid in zip(xyxy, confs, cls_ids, strict=True):
                x1, y1, x2, y2 = box.tolist()
                part_name = model_names.get(int(cid), f"class_{int(cid)}")
                dets.append(
                    Detection(
                        part=part_name,
                        class_id=int(cid),
                        confidence=float(conf_v),
                        bbox_xyxy_norm=(
                            max(0.0, x1 / w),
                            max(0.0, y1 / h),
                            min(1.0, x2 / w),
                            min(1.0, y2 / h),
                        ),
                        bbox_xyxy_px=(
                            int(round(x1)),
                            int(round(y1)),
                            int(round(x2)),
                            int(round(y2)),
                        ),
                        image_width=w,
                        image_height=h,
                    )
                )
            batch_out.append(dets)
        return batch_out

    # ------------------------------------------------------------------ #
    # Training (delegates to ultralytics CLI-equivalent API)
    # ------------------------------------------------------------------ #
    def train(self, **kwargs: Any) -> Any:
        """Proxy to ultralytics YOLO.train(). See ultralytics docs for args."""
        return self.model.train(**kwargs)

    def export(self, format: str = "onnx", **kwargs: Any) -> Any:
        """Proxy to ultralytics export (ONNX, TorchScript, TensorRT, ...)."""
        return self.model.export(format=format, **kwargs)

    @staticmethod
    def iterate_class_names(detections: Iterable[Detection]) -> list[str]:
        return [d.part for d in detections]
