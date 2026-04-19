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

from training.manifest import LabelLine, PartDetectionManifest  # re-export


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
