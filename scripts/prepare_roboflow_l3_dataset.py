"""Ingest a Roboflow car-accident-severity export into Layer-3 format.

Supports two source layouts:

1. **Detection** (Roboflow YOLOv8 export with severity class names on bboxes)::

       <input>/
       ├── data.yaml                     # names: [minor, moderate, severe, ...]
       ├── train/
       │   ├── images/
       │   └── labels/
       └── valid/

   Each bbox becomes a crop tagged with that severity.

2. **Classification** (Roboflow folders-per-class export)::

       <input>/
       ├── train/
       │   ├── minor/    <- folder name is the severity label
       │   ├── moderate/
       │   └── severe/
       └── valid/

   Each image becomes a single sample (no cropping).

Output schema — matches `training.datasets.SeverityDataset`::

    image,part,damage_type,severity,repair_or_replace

Because these datasets rarely link severity to a specific part or damage
type, `part` and `damage_type` default to the placeholder ``"unknown"``.
`repair_or_replace` defaults to a rule over severity (severe/total_loss -> 1,
else 0), but you can override with --no-rule-repair to emit an empty string
the annotator must fill.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Allow running as `python scripts/prepare_roboflow_l3_dataset.py` without PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.layer3_severity import DEFAULT_SEVERITY_GRADES  # noqa: E402
from scripts._roboflow_common import (  # noqa: E402
    IMG_EXTS,
    _norm,
    iterate_split,
    load_mapping,
    load_source_classes,
    resolve_remap,
    resolve_splits,
    yolo_to_xyxy_px,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("prepare_roboflow_l3")


L3_HEADER = ["image", "part", "damage_type", "severity", "repair_or_replace"]


def _severity_index(name: str, target: list[str]) -> int:
    return target.index(name)


def _default_repair_replace(severity_idx: int, use_rule: bool) -> str:
    if not use_rule:
        return ""
    return "1" if severity_idx >= 2 else "0"  # severe/total_loss -> replace


def _expand_and_clip(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int, margin: float
) -> tuple[int, int, int, int]:
    bw, bh = x2 - x1, y2 - y1
    dx, dy = int(bw * margin), int(bh * margin)
    return (max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy))


def _csv_path(output: Path, split: str, prefix: str) -> Path:
    base = f"{prefix}_{split}" if prefix else split
    return output / f"{base}.csv"


# --------------------------------------------------------------------------- #
# Detection mode
# --------------------------------------------------------------------------- #
def _ingest_detection(
    input_dir: Path,
    output_dir: Path,
    splits: list[str],
    prefix: str,
    mapping_path: Path,
    on_unknown: str,
    margin: float,
    use_rule: bool,
    min_box_frac: float,
    dry_run: bool,
) -> dict[str, Any]:
    target, mapping, skip = load_mapping(mapping_path)
    target_set = set(target)
    expected = set(DEFAULT_SEVERITY_GRADES)
    if not target_set.issubset(expected):
        raise ValueError(f"Mapping targets {target_set - expected} not in L3 grades {expected}.")

    src_classes = load_source_classes(input_dir / "data.yaml")
    remap, unknown = resolve_remap(src_classes, mapping, skip, target, on_unknown)
    if on_unknown == "error" and unknown:
        raise SystemExit(
            "Unmapped source severity labels (edit mapping YAML or use "
            "--on-unknown {skip,keep}):\n  - " + "\n  - ".join(sorted(set(unknown)))
        )

    crops_dir = output_dir / "crops"
    if not dry_run:
        crops_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "mode": "detection",
        "boxes_total": 0,
        "boxes_kept": 0,
        "boxes_dropped": 0,
        "crops_written": 0,
        "per_severity": Counter(),
        "per_split_rows": Counter(),
        "unknown_src_names": sorted(set(unknown)) if on_unknown != "error" else [],
    }

    for split in splits:
        pair = iterate_split(input_dir, split)
        if pair is None:
            continue
        imgs_dir, lbls_dir = pair
        csv_path = _csv_path(output_dir, split, prefix)
        rows: list[list[str]] = []

        for img_path in sorted(imgs_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            label_path = lbls_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
            except Exception as exc:
                logger.warning("Skipping %s: %s", img_path, exc)
                continue
            h, w = img.shape[:2]

            for idx, raw in enumerate(label_path.read_text().splitlines()):
                parts = raw.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    src_id = int(parts[0])
                    cx, cy, bw, bh = (float(x) for x in parts[1:5])
                except ValueError:
                    continue
                stats["boxes_total"] += 1

                sev_name = remap.get(src_id)
                if sev_name is None:
                    stats["boxes_dropped"] += 1
                    continue
                if bw * bh < min_box_frac:
                    stats["boxes_dropped"] += 1
                    continue

                x1, y1, x2, y2 = yolo_to_xyxy_px(cx, cy, bw, bh, w, h)
                x1, y1, x2, y2 = _expand_and_clip(x1, y1, x2, y2, w, h, margin)
                crop = img[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    stats["boxes_dropped"] += 1
                    continue

                crop_name = (
                    f"{prefix}_{split}_{img_path.stem}_{idx:02d}.jpg"
                    if prefix
                    else f"{split}_{img_path.stem}_{idx:02d}.jpg"
                )
                if not dry_run:
                    Image.fromarray(crop).save(crops_dir / crop_name, quality=92)
                stats["crops_written"] += 1

                sev_idx = _severity_index(sev_name, DEFAULT_SEVERITY_GRADES)
                rows.append(
                    [
                        crop_name,
                        "unknown",
                        "unknown",
                        str(sev_idx),
                        _default_repair_replace(sev_idx, use_rule),
                    ]
                )
                stats["boxes_kept"] += 1
                stats["per_severity"][sev_name] += 1
                stats["per_split_rows"][split] += 1

        if rows and not dry_run:
            is_new = not csv_path.exists()
            with csv_path.open("a", newline="") as f:
                w_csv = csv.writer(f)
                if is_new:
                    w_csv.writerow(L3_HEADER)
                w_csv.writerows(rows)

    return stats


# --------------------------------------------------------------------------- #
# Classification mode
# --------------------------------------------------------------------------- #
def _ingest_classification(
    input_dir: Path,
    output_dir: Path,
    splits: list[str],
    prefix: str,
    mapping_path: Path,
    on_unknown: str,
    use_rule: bool,
    dry_run: bool,
) -> dict[str, Any]:
    target, mapping, skip = load_mapping(mapping_path)
    target_set = set(target)
    if not target_set.issubset(set(DEFAULT_SEVERITY_GRADES)):
        raise ValueError("Mapping targets contain non-L3 grades.")

    crops_dir = output_dir / "crops"
    if not dry_run:
        crops_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "mode": "classification",
        "images_total": 0,
        "images_kept": 0,
        "images_dropped": 0,
        "per_severity": Counter(),
        "per_split_rows": Counter(),
        "unknown_src_names": [],
    }
    unknown_seen: set[str] = set()

    for split in splits:
        split_dir = input_dir / (
            split if (input_dir / split).exists() else ("val" if split == "valid" else split)
        )
        if not split_dir.exists():
            continue
        csv_path = _csv_path(output_dir, split, prefix)
        rows: list[list[str]] = []

        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            key = _norm(cls_dir.name)
            if key in skip:
                continue
            if key in mapping:
                sev_name = mapping[key]
            elif key in target_set and on_unknown == "keep":
                sev_name = key
            else:
                if on_unknown == "error":
                    raise SystemExit(
                        f"Unmapped severity folder '{cls_dir.name}' "
                        f"(use --on-unknown {{skip,keep}} or extend the mapping)."
                    )
                unknown_seen.add(cls_dir.name)
                continue

            sev_idx = _severity_index(sev_name, DEFAULT_SEVERITY_GRADES)

            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue
                stats["images_total"] += 1
                out_name = (
                    f"{prefix}_{split}_{cls_dir.name}_{img_path.name}"
                    if prefix
                    else f"{split}_{cls_dir.name}_{img_path.name}"
                )
                if not dry_run:
                    shutil.copy2(img_path, crops_dir / out_name)
                rows.append(
                    [
                        out_name,
                        "unknown",
                        "unknown",
                        str(sev_idx),
                        _default_repair_replace(sev_idx, use_rule),
                    ]
                )
                stats["images_kept"] += 1
                stats["per_severity"][sev_name] += 1
                stats["per_split_rows"][split] += 1

        if rows and not dry_run:
            is_new = not csv_path.exists()
            with csv_path.open("a", newline="") as f:
                w_csv = csv.writer(f)
                if is_new:
                    w_csv.writerow(L3_HEADER)
                w_csv.writerows(rows)

    stats["unknown_src_names"] = sorted(unknown_seen)
    return stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/layer3"), type=Path)
    parser.add_argument("--split", default="all", choices=["train", "valid", "test", "all"])
    parser.add_argument("--prefix", default="")
    parser.add_argument("--format", default="detection", choices=["detection", "classification"])
    parser.add_argument("--on-unknown", default="error", choices=["error", "skip", "keep"])
    parser.add_argument(
        "--margin", default=0.08, type=float, help="Detection mode: fractional bbox expansion."
    )
    parser.add_argument("--min-box-frac", default=0.0008, type=float)
    parser.add_argument(
        "--no-rule-repair",
        action="store_true",
        help="Do not auto-fill repair_or_replace from severity; leave empty for annotators.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    use_rule = not args.no_rule_repair
    splits = resolve_splits(args.split)

    if args.format == "detection":
        if not (args.input / "data.yaml").exists():
            logger.error("Detection mode requires data.yaml in %s", args.input)
            return 1
        stats = _ingest_detection(
            input_dir=args.input,
            output_dir=args.output,
            splits=splits,
            prefix=args.prefix,
            mapping_path=args.mapping,
            on_unknown=args.on_unknown,
            margin=args.margin,
            use_rule=use_rule,
            min_box_frac=args.min_box_frac,
            dry_run=args.dry_run,
        )
    else:
        stats = _ingest_classification(
            input_dir=args.input,
            output_dir=args.output,
            splits=splits,
            prefix=args.prefix,
            mapping_path=args.mapping,
            on_unknown=args.on_unknown,
            use_rule=use_rule,
            dry_run=args.dry_run,
        )

    print("=" * 64)
    print(f"Mode: {stats['mode']}")
    if stats["mode"] == "detection":
        print(
            f"Boxes total/kept/dropped:  "
            f"{stats['boxes_total']}/{stats['boxes_kept']}/{stats['boxes_dropped']}"
        )
        print(f"Crops written:             {stats['crops_written']}")
    else:
        print(
            f"Images total/kept/dropped: "
            f"{stats['images_total']}/{stats['images_kept']}/{stats['images_dropped']}"
        )
    print("Per-severity counts:")
    for name, n in stats["per_severity"].most_common():
        print(f"  {name:<12}  {n}")
    print("Per-split row counts:")
    for split, n in stats["per_split_rows"].most_common():
        print(f"  {split:<8}  {n}")
    if stats["unknown_src_names"]:
        print("Unknown source classes (treated per --on-unknown):")
        for n in stats["unknown_src_names"]:
            print(f"  - {n}")
    print("\nNote: 'part' and 'damage_type' are set to 'unknown' — enrich before")
    print("production training, or live with them as metadata (the L3 model does")
    print("not consume them as inputs).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
