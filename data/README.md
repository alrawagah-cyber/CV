# Dataset Format Specification

This pipeline consumes three distinct dataset formats — one per layer. The sample files in this folder show the expected schema with a few dummy rows. After dropping your real data into the empty `images/`, `labels/`, and `crops/` directories, validate with:

```bash
python scripts/validate_dataset.py --layer 1 --root data/layer1
python scripts/validate_dataset.py --layer 2 --root data/layer2
python scripts/validate_dataset.py --layer 3 --root data/layer3
```

---

## Layer 1 — Part Detection (YOLO format)

Directory layout:

```
data/layer1/
├── data.yaml               # Ultralytics data file (points to train/val split)
├── images/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── labels/
│   ├── 000001.txt
│   ├── 000002.txt
│   └── ...
└── samples/                # example label files (format reference only)
```

Each `labels/<id>.txt` contains one bounding box per line:

```
<class_id> <cx> <cy> <w> <h>
```

Where `class_id` is an integer index into the class list (see `configs/layer1.yaml:model.classes`) and `cx, cy, w, h` are normalized to `[0, 1]` relative to the image dimensions. Example:

```
0 0.523 0.610 0.180 0.120
4 0.310 0.405 0.090 0.055
```

`data.yaml` must follow [Ultralytics' data file spec](https://docs.ultralytics.com/datasets/detect/):

```yaml
path: data/layer1
train: images
val:   images   # point to a disjoint val split in practice
nc: 13
names: [bumper, hood, fender, door, windshield, headlight, taillight,
        mirror, trunk, roof, quarter_panel, grille, wheel]
```

---

## Layer 2 — Damage Type (multi-label)

Directory layout:

```
data/layer2/
├── crops/
│   ├── crop_000001.jpg
│   ├── crop_000002.jpg
│   └── ...
├── train.csv
├── val.csv
└── annotations.sample.csv  # schema reference (this file)
```

CSV header (order not required, but all columns must be present):

```
image,dent,scratch,crack,shatter,tear,deformation,paint_loss,puncture,misalignment
```

- `image`: filename relative to `crops/`
- One binary (0/1) column per damage type. Multi-label: any number of damage types can be `1` for a single crop.

Example rows:

```csv
image,dent,scratch,crack,shatter,tear,deformation,paint_loss,puncture,misalignment
crop_000001.jpg,1,1,0,0,0,0,1,0,0
crop_000002.jpg,0,0,1,1,0,0,0,0,0
```

---

## Layer 3 — Severity (ordinal) + Repair/Replace

Directory layout:

```
data/layer3/
├── crops/
├── train.csv
├── val.csv
└── annotations.sample.csv
```

CSV schema:

```
image,part,damage_type,severity,repair_or_replace
```

- `image`: filename relative to `crops/`
- `part`: one of the 13 Layer-1 class names
- `damage_type`: one of the 9 Layer-2 class names
- `severity`: integer in `{0, 1, 2, 3}` corresponding to `{minor, moderate, severe, total_loss}`
- `repair_or_replace`: `0` = repair, `1` = replace

Example rows:

```csv
image,part,damage_type,severity,repair_or_replace
crop_000001.jpg,bumper,dent,1,0
crop_000002.jpg,windshield,shatter,3,1
```

---

## Relationship between layers

The Layer-2 and Layer-3 crops are typically produced by running Layer-1 detections and saving each expanded bounding box as its own image. A helper script in `scripts/` will be provided once your Layer-1 model is fine-tuned. For now you can prepare Layer 2 / Layer 3 datasets independently if you already have cropped images with annotations.
