"""Tests for scripts/prepare_roboflow_l2_dataset.py."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from models.layer2_damage import DEFAULT_DAMAGE_CLASSES
from scripts.prepare_roboflow_l2_dataset import prepare

REPO_ROOT = Path(__file__).resolve().parent.parent
L2_MAPPING = REPO_ROOT / "configs" / "roboflow_mappings" / "damage_types.yaml"


def _write_yolo_export(
    root: Path, names: list[str], rows_per_split: dict[str, list[tuple[float, ...]]]
) -> None:
    """Create a minimal Roboflow YOLOv8 export at `root`."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names, "nc": len(names)}))
    for split, rows in rows_per_split.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        arr = np.zeros((200, 200, 3), dtype=np.uint8)
        arr[40:120, 60:140] = 128
        Image.fromarray(arr).save(root / split / "images" / "im1.jpg")
        label_lines = [" ".join(str(x) for x in r) for r in rows]
        (root / split / "labels" / "im1.txt").write_text("\n".join(label_lines) + "\n")


def test_prepare_l2_emits_onehot_rows(tmp_path: Path):
    src = tmp_path / "rf"
    _write_yolo_export(
        src,
        names=["dent", "scratch", "rust"],  # rust is in the skip list
        rows_per_split={
            "train": [
                (0, 0.5, 0.5, 0.3, 0.2),  # dent
                (1, 0.3, 0.3, 0.1, 0.1),  # scratch
                (2, 0.2, 0.2, 0.1, 0.1),  # rust -> dropped
            ],
        },
    )
    out = tmp_path / "layer2"
    stats = prepare(
        input_dir=src,
        mapping_path=L2_MAPPING,
        output_dir=out,
        splits=["train"],
        prefix="t",
        on_unknown="error",
        margin=0.05,
        dry_run=False,
        min_box_frac=0.0,
    )

    assert stats["boxes_total"] == 3
    assert stats["boxes_kept"] == 2
    assert stats["boxes_dropped"] == 1
    assert stats["crops_written"] == 2
    assert stats["per_target"]["dent"] == 1
    assert stats["per_target"]["scratch"] == 1

    csv_path = out / "t_train.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    # Confirm header + one-hot semantics
    for r in rows:
        ones = [c for c in DEFAULT_DAMAGE_CLASSES if r[c] == "1"]
        zeros = [c for c in DEFAULT_DAMAGE_CLASSES if r[c] == "0"]
        assert len(ones) == 1
        assert len(zeros) == len(DEFAULT_DAMAGE_CLASSES) - 1

    # Crops exist
    crops = list((out / "crops").iterdir())
    assert len(crops) == 2


def test_prepare_l2_appends_across_datasets(tmp_path: Path):
    # Running twice with different prefixes should stack rows in the same CSV.
    out = tmp_path / "layer2"
    for prefix, damage in [("a", "dent"), ("b", "scratch")]:
        src = tmp_path / f"rf_{prefix}"
        _write_yolo_export(src, names=[damage], rows_per_split={"train": [(0, 0.5, 0.5, 0.4, 0.3)]})
        prepare(
            input_dir=src,
            mapping_path=L2_MAPPING,
            output_dir=out,
            splits=["train"],
            prefix=prefix,
            on_unknown="error",
            margin=0.05,
            dry_run=False,
            min_box_frac=0.0,
        )

    # Note: CSV filename includes the prefix, so we end up with two distinct CSVs by design.
    # Check both exist.
    assert (out / "a_train.csv").exists()
    assert (out / "b_train.csv").exists()
    with open(out / "a_train.csv") as fa:
        rows_a = list(csv.DictReader(fa))
    with open(out / "b_train.csv") as fb:
        rows_b = list(csv.DictReader(fb))
    assert rows_a[0]["dent"] == "1"
    assert rows_b[0]["scratch"] == "1"


def test_prepare_l2_skip_mode_tolerates_unknown(tmp_path: Path):
    src = tmp_path / "rf"
    _write_yolo_export(
        src,
        names=["dent", "absurdly_novel_damage_type"],
        rows_per_split={"train": [(0, 0.5, 0.5, 0.2, 0.2), (1, 0.3, 0.3, 0.1, 0.1)]},
    )
    stats = prepare(
        input_dir=src,
        mapping_path=L2_MAPPING,
        output_dir=tmp_path / "out",
        splits=["train"],
        prefix="",
        on_unknown="skip",
        margin=0.05,
        dry_run=False,
        min_box_frac=0.0,
    )
    assert stats["boxes_kept"] == 1
    assert "absurdly_novel_damage_type" in stats["unknown_src_names"]


def test_prepare_l2_error_on_unknown(tmp_path: Path):
    src = tmp_path / "rf"
    _write_yolo_export(
        src,
        names=["dent", "brand_new_damage"],
        rows_per_split={"train": [(0, 0.5, 0.5, 0.2, 0.2)]},
    )
    with pytest.raises(SystemExit):
        prepare(
            input_dir=src,
            mapping_path=L2_MAPPING,
            output_dir=tmp_path / "out",
            splits=["train"],
            prefix="",
            on_unknown="error",
            margin=0.05,
            dry_run=True,
            min_box_frac=0.0,
        )


def test_prepare_l2_drops_tiny_boxes(tmp_path: Path):
    src = tmp_path / "rf"
    _write_yolo_export(
        src,
        names=["dent"],
        rows_per_split={
            "train": [
                (0, 0.5, 0.5, 0.01, 0.01),  # area=0.0001 — tiny, dropped with default 0.0008 threshold
                (0, 0.5, 0.5, 0.3, 0.3),  # area=0.09 — kept
            ]
        },
    )
    stats = prepare(
        input_dir=src,
        mapping_path=L2_MAPPING,
        output_dir=tmp_path / "out",
        splits=["train"],
        prefix="",
        on_unknown="error",
        margin=0.05,
        dry_run=False,
        min_box_frac=0.0008,
    )
    assert stats["boxes_kept"] == 1
    assert stats["boxes_dropped"] == 1
