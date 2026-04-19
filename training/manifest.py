"""Torch-free YOLO-format manifest helpers.

Split out of ``training.datasets`` so validation scripts can use them
without importing torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LabelLine:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float


class PartDetectionManifest:
    """Lightweight verifier for YOLO-format directory layouts.

    Expected layout::

        <root>/images/<id>.jpg
        <root>/labels/<id>.txt
        <root>/data.yaml   # classes list
    """

    def __init__(self, root: str | Path, classes: list[str] | None = None):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.classes = classes

    def list_images(self) -> list[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted([p for p in self.images_dir.glob("*") if p.suffix.lower() in exts])

    @staticmethod
    def parse_label_file(path: Path) -> list[LabelLine]:
        out: list[LabelLine] = []
        with path.open("r") as f:
            for i, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                parts = raw.split()
                if len(parts) != 5:
                    raise ValueError(f"{path}:{i} expected 5 values, got {len(parts)}: {raw!r}")
                cid = int(parts[0])
                cx, cy, w, h = (float(x) for x in parts[1:])
                out.append(LabelLine(cid, cx, cy, w, h))
        return out
