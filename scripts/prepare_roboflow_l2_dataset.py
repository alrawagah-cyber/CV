"""Ingest a Roboflow YOLOv8 car-damage-detection export into Layer-2 format.

Each damage bounding box in the Roboflow export becomes:
    - a cropped image (expanded by --margin) written to <output>/crops/
    - one row in <output>/<split>.csv with the mapped damage type set to 1
      and all other damage columns set to 0.

Multi-label training still works: single-label rows are a valid degenerate
case of multi-label. If you later discover that two Roboflow bboxes from
different datasets correspond to the *same physical region* (e.g. the same
dented bumper labeled 'dent' in one dataset and 'deformation' in another),
you can merge rows by filename in a downstream step — the ingestor does not
attempt this automatically.

Usage
-----

    python scripts/prepare_roboflow_l2_dataset.py \\
        --input ~/roboflow_raw/car-damage-zxk33 \\
        --mapping configs/roboflow_mappings/damage_types.yaml \\
        --output data/layer2 \\
        --prefix zxk33 \\
        --split all \\
        --on-unknown skip \\
        --margin 0.08
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

# Allow running as `python scripts/prepare_roboflow_l2_dataset.py` without PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.class_constants import DEFAULT_DAMAGE_CLASSES  # noqa: E402
from scripts._roboflow_common import (  # noqa: E402
    IMG_EXTS,
    iterate_split,
    load_mapping,
    load_source_classes,
    resolve_remap,
    resolve_splits,
    yolo_to_xyxy_px,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("prepare_roboflow_l2")


def _expand_and_clip(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int, margin: float
) -> tuple[int, int, int, int]:
    bw, bh = x2 - x1, y2 - y1
    dx, dy = int(bw * margin), int(bh * margin)
    return (
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(w, x2 + dx),
        min(h, y2 + dy),
    )


def _csv_path(output: Path, split: str, prefix: str) -> Path:
    base = f"{prefix}_{split}" if prefix else split
    return output / f"{base}.csv"


def prepare(
    input_dir: Path,
    mapping_path: Path,
    output_dir: Path,
    splits: list[str],
    prefix: str,
    on_unknown: str,
    margin: float,
    dry_run: bool,
    min_box_frac: float,
) -> dict[str, object]:
    target, mapping, skip = load_mapping(mapping_path)
    target_set = set(target)
    # Verify the mapping targets are a subset of our L2 vocabulary.
    if not target_set.issubset(set(DEFAULT_DAMAGE_CLASSES)):
        extras = target_set - set(DEFAULT_DAMAGE_CLASSES)
        raise ValueError(
            f"Mapping targets contain non-L2 classes: {extras}. "
            f"Expected subset of {DEFAULT_DAMAGE_CLASSES}."
        )

    src_classes = load_source_classes(input_dir / "data.yaml")
    logger.info("Source dataset classes (%d): %s", len(src_classes), src_classes)

    remap, unknown = resolve_remap(src_classes, mapping, skip, target, on_unknown)
    if on_unknown == "error" and unknown:
        raise SystemExit(
            "Unmapped source classes (edit mapping YAML or use "
            "--on-unknown {skip,keep}):\n  - " + "\n  - ".join(sorted(set(unknown)))
        )

    crops_dir = output_dir / "crops"
    if not dry_run:
        crops_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "boxes_total": 0,
        "boxes_kept": 0,
        "boxes_dropped": 0,
        "crops_written": 0,
        "per_split_rows": Counter(),
        "per_target": Counter(),
        "unknown_src_names": sorted(set(unknown)) if on_unknown != "error" else [],
    }

    for split in splits:
        pair = iterate_split(input_dir, split)
        if pair is None:
            logger.info("Split '%s' not present, skipping.", split)
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
                raw = raw.strip()
                if not raw:
                    continue
                parts = raw.split()
                if len(parts) < 5:
                    continue
                try:
                    src_id = int(parts[0])
                    cx, cy, bw, bh = (float(x) for x in parts[1:5])
                except ValueError:
                    continue
                stats["boxes_total"] += 1

                tgt_name = remap.get(src_id)
                if tgt_name is None:
                    stats["boxes_dropped"] += 1
                    continue

                # Drop tiny boxes — they crop down to noise
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

                row_vec = ["1" if c == tgt_name else "0" for c in DEFAULT_DAMAGE_CLASSES]
                rows.append([crop_name, *row_vec])
                stats["boxes_kept"] += 1
                stats["per_target"][tgt_name] += 1
                stats["per_split_rows"][split] += 1

        # Append rows to this split's CSV. Use append-then-sort so users can
        # incrementally stack datasets with repeated invocations.
        if rows and not dry_run:
            is_new = not csv_path.exists()
            with csv_path.open("a", newline="") as f:
                w_csv = csv.writer(f)
                if is_new:
                    w_csv.writerow(["image", *DEFAULT_DAMAGE_CLASSES])
                w_csv.writerows(rows)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/layer2"), type=Path)
    parser.add_argument("--split", default="all", choices=["train", "valid", "test", "all"])
    parser.add_argument(
        "--prefix", default="", help="Filename prefix (prevents collisions across stacked datasets)."
    )
    parser.add_argument("--on-unknown", default="error", choices=["error", "skip", "keep"])
    parser.add_argument(
        "--margin",
        default=0.08,
        type=float,
        help="Fractional bbox expansion before cropping (0.08 = 8 percent).",
    )
    parser.add_argument(
        "--min-box-frac",
        default=0.0008,
        type=float,
        help="Drop boxes whose normalized area is below this (default 0.0008 = 0.08 percent).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (args.input / "data.yaml").exists():
        logger.error("Missing data.yaml under %s (is this a Roboflow YOLOv8 export?)", args.input)
        return 1

    stats = prepare(
        input_dir=args.input,
        mapping_path=args.mapping,
        output_dir=args.output,
        splits=resolve_splits(args.split),
        prefix=args.prefix,
        on_unknown=args.on_unknown,
        margin=args.margin,
        dry_run=args.dry_run,
        min_box_frac=args.min_box_frac,
    )

    print("=" * 64)
    print(f"Boxes (total):    {stats['boxes_total']}")
    print(f"Boxes (kept):     {stats['boxes_kept']}")
    print(f"Boxes (dropped):  {stats['boxes_dropped']}")
    print(f"Crops written:    {stats['crops_written']}")
    print("Per-damage-type counts:")
    for name, n in stats["per_target"].most_common():
        print(f"  {name:<14}  {n}")
    print("Per-split row counts:")
    for split, n in stats["per_split_rows"].most_common():
        print(f"  {split:<8}  {n}")
    if stats["unknown_src_names"]:
        print("Unknown source classes (treated per --on-unknown):")
        for n in stats["unknown_src_names"]:
            print(f"  - {n}")
    print(
        "\nNext: run scripts/validate_dataset.py --layer 2 --root "
        f"{args.output} --csv {_csv_path(args.output, 'train', args.prefix)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
