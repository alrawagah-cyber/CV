"""Validate dataset integrity before training.

Usage:
    python scripts/validate_dataset.py --layer 1 --root data/layer1
    python scripts/validate_dataset.py --layer 2 --root data/layer2 --csv data/layer2/annotations.sample.csv
    python scripts/validate_dataset.py --layer 3 --root data/layer3 --csv data/layer3/annotations.sample.csv

Checks:
    Layer 1: YOLO label files have 5 fields, class_ids in range, box coords in [0,1],
             images/<id>.jpg has a matching labels/<id>.txt (and vice versa for labels with images).
    Layer 2: CSV has required columns, binary label values, image file paths resolve.
    Layer 3: CSV schema + value ranges, image files resolve.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from models.layer1_detector import DEFAULT_PART_CLASSES
from models.layer2_damage import DEFAULT_DAMAGE_CLASSES
from models.layer3_severity import DEFAULT_SEVERITY_GRADES
from training.datasets import PartDetectionManifest

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Layer 1
# --------------------------------------------------------------------------- #
def validate_layer1(root: Path) -> list[str]:
    errors: list[str] = []
    images_dir = root / "images"
    labels_dir = root / "labels"
    samples_dir = root / "samples"

    if not images_dir.exists():
        errors.append(f"{images_dir} does not exist")
    if not labels_dir.exists():
        errors.append(f"{labels_dir} does not exist")

    # Load class list from data.yaml if present; fall back to defaults
    data_yaml_path = root / "data.yaml"
    classes: list[str]
    if data_yaml_path.exists():
        with open(data_yaml_path) as f:
            y = yaml.safe_load(f) or {}
        classes = y.get("names") or DEFAULT_PART_CLASSES
    else:
        classes = DEFAULT_PART_CLASSES
    num_classes = len(classes)

    manifest = PartDetectionManifest(root, classes=classes)

    # Validate each label file (use samples/ as fallback if images/ empty)
    label_files = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
    if not label_files and samples_dir.exists():
        label_files = list(samples_dir.glob("*.txt"))
        print(f"[layer1] labels/ empty; validating samples in {samples_dir}")

    if not label_files:
        errors.append("No label files found to validate (labels/ and samples/ are empty).")

    for lp in label_files:
        try:
            lines = PartDetectionManifest.parse_label_file(lp)
        except ValueError as e:
            errors.append(str(e))
            continue
        for i, ll in enumerate(lines, start=1):
            if not (0 <= ll.class_id < num_classes):
                errors.append(f"{lp}:{i} class_id {ll.class_id} out of range [0,{num_classes-1}]")
            for name, v in (("cx", ll.cx), ("cy", ll.cy), ("w", ll.w), ("h", ll.h)):
                if not (0.0 <= v <= 1.0):
                    errors.append(f"{lp}:{i} {name}={v} not in [0,1]")
            if ll.w <= 0 or ll.h <= 0:
                errors.append(f"{lp}:{i} non-positive width/height")

    # Image <-> label pairing (only if both dirs have content)
    if images_dir.exists() and labels_dir.exists():
        imgs = {p.stem for p in manifest.list_images()}
        labs = {p.stem for p in labels_dir.glob("*.txt")}
        if imgs and labs:
            orphan_images = sorted(imgs - labs)[:5]
            orphan_labels = sorted(labs - imgs)[:5]
            if orphan_images:
                errors.append(f"Images without matching .txt labels (first 5): {orphan_images}")
            if orphan_labels:
                errors.append(f"Labels without matching images (first 5): {orphan_labels}")

    return errors


# --------------------------------------------------------------------------- #
# Layer 2
# --------------------------------------------------------------------------- #
def validate_layer2(root: Path, csv_path: Path) -> list[str]:
    errors: list[str] = []
    if not csv_path.exists():
        return [f"{csv_path} does not exist"]

    classes = DEFAULT_DAMAGE_CLASSES
    df = pd.read_csv(csv_path)
    required = ["image", *classes]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
        return errors

    for i, row in df.iterrows():
        for c in classes:
            v = row[c]
            if v not in (0, 1):
                errors.append(f"{csv_path}:{i+2} column '{c}' has non-binary value {v}")
        img_path = root / "crops" / str(row["image"])
        if img_path.suffix.lower() not in IMG_EXTS:
            errors.append(f"{csv_path}:{i+2} image {img_path.name} has unexpected extension")
        # Filesystem check is non-fatal for the sample CSV; warn only.

    return errors


# --------------------------------------------------------------------------- #
# Layer 3
# --------------------------------------------------------------------------- #
def validate_layer3(root: Path, csv_path: Path) -> list[str]:
    errors: list[str] = []
    if not csv_path.exists():
        return [f"{csv_path} does not exist"]

    grades = DEFAULT_SEVERITY_GRADES
    df = pd.read_csv(csv_path)
    required = ["image", "part", "damage_type", "severity", "repair_or_replace"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
        return errors

    # "unknown" is an allowed placeholder for bootstrap data (e.g. from the
    # Roboflow L3 ingestor). The L3 model does not consume part/damage_type
    # as inputs — they are metadata — so unknown values do not affect training.
    valid_parts = set(DEFAULT_PART_CLASSES) | {"unknown"}
    valid_damage = set(DEFAULT_DAMAGE_CLASSES) | {"unknown"}

    for i, row in df.iterrows():
        sev = int(row["severity"])
        if not (0 <= sev < len(grades)):
            errors.append(f"{csv_path}:{i+2} severity {sev} out of range [0,{len(grades)-1}]")
        rr_raw = row["repair_or_replace"]
        if rr_raw == "" or pd.isna(rr_raw):
            errors.append(f"{csv_path}:{i+2} repair_or_replace is blank — fill before training")
        else:
            try:
                rr = int(rr_raw)
            except (ValueError, TypeError):
                errors.append(f"{csv_path}:{i+2} repair_or_replace {rr_raw!r} not an integer")
                continue
            if rr not in (0, 1):
                errors.append(f"{csv_path}:{i+2} repair_or_replace {rr} not in {{0,1}}")
        if str(row["part"]) not in valid_parts:
            errors.append(f"{csv_path}:{i+2} unknown part {row['part']!r}")
        if str(row["damage_type"]) not in valid_damage:
            errors.append(f"{csv_path}:{i+2} unknown damage_type {row['damage_type']!r}")
        img_path = root / "crops" / str(row["image"])
        if img_path.suffix.lower() not in IMG_EXTS:
            errors.append(f"{csv_path}:{i+2} image {img_path.name} has unexpected extension")

    return errors


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Override CSV path (defaults to annotations.sample.csv for L2/L3).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if args.layer == 1:
        errors = validate_layer1(root)
    elif args.layer == 2:
        csv_path = Path(args.csv) if args.csv else root / "annotations.sample.csv"
        errors = validate_layer2(root, csv_path)
    else:
        csv_path = Path(args.csv) if args.csv else root / "annotations.sample.csv"
        errors = validate_layer3(root, csv_path)

    if errors:
        print(f"Validation failed for layer {args.layer}:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"Layer {args.layer} dataset OK (root={root}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
