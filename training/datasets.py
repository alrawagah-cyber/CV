"""Dataset classes for each layer.

- PartDetectionManifest: validates YOLO-format detection data on disk.
  (YOLO training consumes files directly via ultralytics; this class is used
  by scripts/validate_dataset.py and tests.)
- DamageTypeDataset: multi-label classification on cropped images.
- SeverityDataset: ordinal severity + optional repair/replace labels on crops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Layer 1 — manifest / validation helper for YOLO-format directories
# ---------------------------------------------------------------------------
@dataclass
class LabelLine:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float


class PartDetectionManifest:
    """Lightweight verifier for YOLO-format dir layouts. No torch Dataset here:
    ultralytics handles detection training, so we only need validation helpers.

    Expected layout:
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


# ---------------------------------------------------------------------------
# Layer 2 — multi-label damage type classification
# ---------------------------------------------------------------------------
class DamageTypeDataset(Dataset):
    """Multi-label dataset.

    CSV schema (header required):
        image,dent,scratch,crack,shatter,tear,deformation,paint_loss,puncture,misalignment
    The "image" column is a path relative to `root/crops/`.
    """

    def __init__(
        self,
        root: str | Path,
        annotations_csv: str | Path,
        classes: list[str],
        transform: Any | None = None,
    ):
        self.root = Path(root)
        self.crops_dir = self.root / "crops"
        self.df = pd.read_csv(annotations_csv)
        self.classes = list(classes)
        self.transform = transform

        missing = [c for c in ["image", *self.classes] if c not in self.df.columns]
        if missing:
            raise ValueError(f"Layer-2 CSV missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = self.crops_dir / str(row["image"])
        img = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            img_t = self.transform(image=img)["image"]
        else:
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        labels = torch.tensor(
            [float(row[c]) for c in self.classes],
            dtype=torch.float32,
        )
        return {"image": img_t, "labels": labels, "path": str(img_path)}


# ---------------------------------------------------------------------------
# Layer 3 — severity (ordinal) + repair/replace
# ---------------------------------------------------------------------------
class SeverityDataset(Dataset):
    """Ordinal severity dataset.

    CSV schema:
        image,part,damage_type,severity,repair_or_replace
    Where:
        severity ∈ {0, 1, 2, 3}   # minor, moderate, severe, total_loss
        repair_or_replace ∈ {0, 1} # 0=repair, 1=replace
    """

    REQUIRED_COLS = ("image", "part", "damage_type", "severity", "repair_or_replace")

    def __init__(
        self,
        root: str | Path,
        annotations_csv: str | Path,
        grades: list[str],
        transform: Any | None = None,
    ):
        self.root = Path(root)
        self.crops_dir = self.root / "crops"
        self.df = pd.read_csv(annotations_csv)
        self.grades = list(grades)
        self.transform = transform

        missing = [c for c in self.REQUIRED_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Layer-3 CSV missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = self.crops_dir / str(row["image"])
        img = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            img_t = self.transform(image=img)["image"]
        else:
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        severity = int(row["severity"])
        if severity < 0 or severity >= len(self.grades):
            raise ValueError(f"severity {severity} out of range for grades {self.grades}")
        repair_or_replace = int(row["repair_or_replace"])

        return {
            "image": img_t,
            "severity": torch.tensor(severity, dtype=torch.long),
            "repair_or_replace": torch.tensor(repair_or_replace, dtype=torch.float32),
            "part": str(row["part"]),
            "damage_type": str(row["damage_type"]),
            "path": str(img_path),
        }
