"""Ingest the VehiDE (Vietnamese car-damage) Kaggle dataset into Layer-2 format.

VehiDE ships as:

    <input>/
        0Train_via_annos.json       # VIA polygon annotations (train split)
        0Val_via_annos.json         # VIA polygon annotations (val split)
        image/                      # 11,621 train images
        validation/                 # 2,324 val images

Each VIA JSON is a dict keyed by image filename. Each entry has:

    {
        "name": "<arbitrary>",
        "regions": [
            {
                "all_x": [x1, x2, x3, ...],  # polygon vertices (pixel coords)
                "all_y": [y1, y2, y3, ...],
                "class": "tray_son"          # Vietnamese class string
            },
            ...
        ]
    }

Class strings are Vietnamese (tray_son, mop_lom, rach, mat_bo_phan, be_den,
thung, vo_kinh). They translate to our English L2 vocabulary via the
existing configs/roboflow_mappings/damage_types.yaml mapping.

Usage
-----

    python scripts/prepare_vehide_dataset.py \\
        --input ~/kaggle_l2l3/vehide \\
        --mapping configs/roboflow_mappings/damage_types.yaml \\
        --output data/layer2 \\
        --prefix vehide \\
        --on-unknown skip \\
        --margin 0.08
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.class_constants import DEFAULT_DAMAGE_CLASSES  # noqa: E402
from scripts._roboflow_common import IMG_EXTS, _norm, load_mapping  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("prepare_vehide")

# Fixed split layout inside the VehiDE zip.
# The Kaggle archive nests images one level deeper ("image/image/", "validation/validation/")
# — each candidate in the list is tried in order, the first existing one wins.
SPLITS = {
    "train": {"json": "0Train_via_annos.json", "images": ["image/image", "image"]},
    "valid": {"json": "0Val_via_annos.json", "images": ["validation/validation", "validation"]},
}


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


def _polygon_bbox_px(
    all_x: list[float], all_y: list[float], img_w: int, img_h: int
) -> tuple[int, int, int, int] | None:
    """Min/max enclosing bbox for a polygon, in pixel coords."""
    if not all_x or not all_y or len(all_x) != len(all_y):
        return None
    x1 = max(0, int(round(min(all_x))))
    y1 = max(0, int(round(min(all_y))))
    x2 = min(img_w, int(round(max(all_x))))
    y2 = min(img_h, int(round(max(all_y))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _csv_path(output: Path, split: str, prefix: str) -> Path:
    base = f"{prefix}_{split}" if prefix else split
    return output / f"{base}.csv"


def _load_via_annos(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare(
    input_dir: Path,
    mapping_path: Path,
    output_dir: Path,
    prefix: str,
    on_unknown: str,
    margin: float,
    dry_run: bool,
    min_box_frac: float,
) -> dict[str, object]:
    target, mapping, skip = load_mapping(mapping_path)
    target_set = set(target)
    if not target_set.issubset(set(DEFAULT_DAMAGE_CLASSES)):
        extras = target_set - set(DEFAULT_DAMAGE_CLASSES)
        raise ValueError(
            f"Mapping targets contain non-L2 classes: {extras}. "
            f"Expected subset of {DEFAULT_DAMAGE_CLASSES}."
        )

    crops_dir = output_dir / "crops"
    if not dry_run:
        crops_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, object] = {
        "boxes_total": 0,
        "boxes_kept": 0,
        "boxes_dropped": 0,
        "crops_written": 0,
        "per_split_rows": Counter(),
        "per_target": Counter(),
        "unknown_src_names": Counter(),
    }

    for split, meta in SPLITS.items():
        json_path = input_dir / meta["json"]
        if not json_path.exists():
            logger.warning("Split '%s' missing (%s) — skipping.", split, json_path)
            continue
        imgs_dir = next((input_dir / c for c in meta["images"] if (input_dir / c).is_dir()), None)
        if imgs_dir is None:
            logger.warning(
                "Split '%s' image dir not found under any of %s — skipping.",
                split,
                [str(input_dir / c) for c in meta["images"]],
            )
            continue

        logger.info("Loading annotations: %s", json_path)
        annos = _load_via_annos(json_path)
        logger.info("Split '%s' has %d annotated images under %s", split, len(annos), imgs_dir)

        csv_path = _csv_path(output_dir, split, prefix)
        rows: list[list[str]] = []

        for idx, (fname, entry) in enumerate(annos.items()):
            img_path = imgs_dir / fname
            if not img_path.exists() or img_path.suffix.lower() not in IMG_EXTS:
                continue
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
            except Exception as exc:
                logger.warning("Skipping %s: %s", img_path, exc)
                continue
            h, w = img.shape[:2]
            total_area = float(h * w)

            regions = entry.get("regions") or []
            for r_idx, region in enumerate(regions):
                stats["boxes_total"] += 1
                src_name = region.get("class")
                key = _norm(src_name) if src_name else ""
                tgt_name: str | None
                if not key:
                    stats["boxes_dropped"] += 1
                    continue
                if key in skip:
                    stats["boxes_dropped"] += 1
                    continue
                if key in mapping:
                    tgt_name = mapping[key]
                else:
                    stats["unknown_src_names"][src_name] += 1
                    if on_unknown == "skip":
                        stats["boxes_dropped"] += 1
                        continue
                    elif on_unknown == "keep":
                        if key not in target_set:
                            raise ValueError(
                                f"--on-unknown=keep requires {src_name!r} (normalized "
                                f"{key!r}) to already be a target class."
                            )
                        tgt_name = key
                    else:  # "error"
                        raise SystemExit(
                            f"Unmapped class {src_name!r} (normalized {key!r}). "
                            f"Edit mapping YAML or pass --on-unknown {{skip,keep}}."
                        )

                bbox = _polygon_bbox_px(region.get("all_x") or [], region.get("all_y") or [], w, h)
                if bbox is None:
                    stats["boxes_dropped"] += 1
                    continue
                x1, y1, x2, y2 = bbox

                box_area_frac = ((x2 - x1) * (y2 - y1)) / total_area if total_area else 0.0
                if box_area_frac < min_box_frac:
                    stats["boxes_dropped"] += 1
                    continue

                x1, y1, x2, y2 = _expand_and_clip(x1, y1, x2, y2, w, h, margin)
                crop = img[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    stats["boxes_dropped"] += 1
                    continue

                stem = Path(fname).stem
                crop_name = (
                    f"{prefix}_{split}_{stem}_{r_idx:02d}.jpg"
                    if prefix
                    else f"{split}_{stem}_{r_idx:02d}.jpg"
                )
                if not dry_run:
                    Image.fromarray(crop).save(crops_dir / crop_name, quality=92)
                stats["crops_written"] += 1

                row_vec = ["1" if c == tgt_name else "0" for c in DEFAULT_DAMAGE_CLASSES]
                rows.append([crop_name, *row_vec])
                stats["boxes_kept"] += 1
                stats["per_target"][tgt_name] += 1
                stats["per_split_rows"][split] += 1

            if (idx + 1) % 500 == 0:
                logger.info("  … processed %d / %d images for split %s", idx + 1, len(annos), split)

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
    parser.add_argument("--prefix", default="vehide", help="Filename prefix (collision-free stacking).")
    parser.add_argument("--on-unknown", default="skip", choices=["error", "skip", "keep"])
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

    if not (args.input / "0Train_via_annos.json").exists():
        logger.error("Missing 0Train_via_annos.json under %s (is this the VehiDE export?)", args.input)
        return 1

    stats = prepare(
        input_dir=args.input,
        mapping_path=args.mapping,
        output_dir=args.output,
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
        for n, c in stats["unknown_src_names"].most_common():
            print(f"  {n!r:<20}  {c}")
    print(
        "\nNext: run scripts/validate_dataset.py --layer 2 --root "
        f"{args.output} --csv {_csv_path(args.output, 'train', args.prefix)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
