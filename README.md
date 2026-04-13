# Car Damage Assessment Pipeline

An **enterprise-grade, three-layer deep-learning pipeline** for automated car damage assessment on insurance claims. Out of the box it runs end-to-end on a single image using pretrained backbones; drop in your own annotated datasets to fine-tune each layer independently.

## Architecture

```
 ┌────────────────────────────────────────────────────────────────────┐
 │                          Input Image(s)                            │
 └───────────────┬──────────────────────────────────────────┬─────────┘
                 ▼                                          ▼
 ┌──────────────────────────┐                 ┌──────────────────────────┐
 │  Layer 1: Part Detector  │                 │      FastAPI service     │
 │  YOLOv8x / YOLOv11x      │                 │   /assess (sync)         │
 │  bumper, hood, door, ... │                 │   /assess/batch (Celery) │
 └──────────┬───────────────┘                 │   /health /metrics /jobs │
            │ crops + bboxes                  └──────────────────────────┘
            ▼
 ┌──────────────────────────┐
 │ Layer 2: Damage Type     │  ConvNeXt-V2-L  (multi-label sigmoid)
 │ dent, scratch, crack,    │  → {dent, scratch, crack, shatter,
 │ shatter, tear, ...       │     tear, deformation, paint_loss, ...}
 └──────────┬───────────────┘
            │
            ▼
 ┌──────────────────────────┐
 │ Layer 3: Severity        │  Swin-V2-L + CORAL ordinal head
 │ + repair/replace head    │  → {minor, moderate, severe, total_loss}
 └──────────┬───────────────┘  → repair | replace
            ▼
 ┌──────────────────────────┐
 │    ClaimReport (JSON)    │  schema_version 1.0
 └──────────────────────────┘
```

The `ClaimAssessor` class (`inference/claim_assessor.py`) chains the three layers and emits a structured JSON report per image.

---

## Quickstart

```bash
# 1. Install
pip install -r requirements-dev.txt

# 2. Cache pretrained weights (YOLOv8x, ConvNeXt-V2-L, SwinV2-L)
python scripts/download_weights.py

# 3. Generate a synthetic test image and run the full pipeline
python scripts/generate_sample_image.py
python scripts/run_inference.py --image data/samples/test_car.jpg

# 4. Or launch the full stack (API + Celery worker + Redis + Prometheus)
docker-compose up --build
curl -F "file=@data/samples/test_car.jpg" http://localhost:8000/assess | jq
```

Until you fine-tune layers 2 and 3, the pipeline emits reports flagged with `"pretrained_baseline": true` so downstream consumers can distinguish baseline runs from production runs.

---

## Repository layout

```
models/          - architecture definitions per layer (+ CORAL heads, registry)
training/        - datasets, augmentations, generic AMP trainer, L1/L2/L3 entrypoints
inference/       - ClaimAssessor, pre/post-processing, batching
api/             - FastAPI app, Pydantic schemas, Celery worker, Prometheus metrics
configs/         - YAML configs for each layer + inference + API
data/            - dataset layout spec + sample annotation files (drop yours here)
tests/           - pytest (CPU-only, stubbed models)
scripts/         - download_weights, run_inference, export_onnx, validate_dataset,
                   prepare_roboflow_dataset, extract_crops
docker/          - Dockerfile.base / .api / .worker + prometheus.yml
```

---

## Preparing data

### Layer 1 — ingesting Roboflow datasets

Most public part-detection datasets ship with their own class vocabulary (e.g. `front_bumper`, `rear_bumper`, `Head Lamp`, ...). `scripts/prepare_roboflow_dataset.py` remaps them into our 13-class vocabulary and merges them into `data/layer1/`.

```bash
# 1. On Roboflow Universe, click "Download Dataset" → format "YOLOv8".
#    Unzip each into a scratch folder, e.g. ~/roboflow_raw/<name>/.

# 2. Dry-run to see what the mapping would produce:
python scripts/prepare_roboflow_dataset.py \
    --input ~/roboflow_raw/car-part-q8otu \
    --mapping configs/roboflow_mappings/default.yaml \
    --output data/layer1 \
    --prefix cpq8otu \
    --on-unknown skip \
    --dry-run

# 3. If it complains about unmapped classes, edit the mapping YAML and re-run.
#    --on-unknown error  stops on anything not in mapping/skip (recommended once mapping is final).
#    --on-unknown skip   drops unknowns silently (useful for exploration).

# 4. Merge the dataset in:
python scripts/prepare_roboflow_dataset.py \
    --input ~/roboflow_raw/car-part-q8otu \
    --mapping configs/roboflow_mappings/default.yaml \
    --output data/layer1 \
    --prefix cpq8otu

# 5. Repeat for each Roboflow export, changing --prefix so filenames don't collide:
python scripts/prepare_roboflow_dataset.py --input ~/roboflow_raw/car-part-detect-chrlc \
    --mapping configs/roboflow_mappings/default.yaml --output data/layer1 --prefix chrlc
python scripts/prepare_roboflow_dataset.py --input ~/roboflow_raw/part-autolabeld \
    --mapping configs/roboflow_mappings/default.yaml --output data/layer1 --prefix autolabeld
python scripts/prepare_roboflow_dataset.py --input ~/roboflow_raw/car-mz8m3 \
    --mapping configs/roboflow_mappings/default.yaml --output data/layer1 --prefix mz8m3

# 6. Validate:
python scripts/validate_dataset.py --layer 1 --root data/layer1
```

The mapping file (`configs/roboflow_mappings/default.yaml`) is case-insensitive and collapses separators, so `Front-Bumper`, `front_bumper`, and `FRONT BUMPER` all resolve to `bumper`. You can create per-dataset overrides next to the default if one project uses unusual names.

### Layers 2 & 3 — generating crops from Layer 1 detections

Once Layer 1 is fine-tuned (or even using the COCO baseline), extract candidate crops so you only have to label damage type (L2) and severity (L3), not bounding boxes:

```bash
# Layer 2 manifest (multi-label damage type, all cols zeroed, ready to flip bits):
python scripts/extract_crops.py \
    --source data/layer1/images \
    --weights runs/layer1/exp/weights/best.pt \
    --layer 2 \
    --out data/layer2/crops \
    --manifest data/layer2/crops_manifest.csv \
    --conf 0.3 --margin 0.1

# Layer 3 manifest (severity + repair/replace, placeholders for your labelers):
python scripts/extract_crops.py \
    --source data/layer1/images \
    --weights runs/layer1/exp/weights/best.pt \
    --layer 3 \
    --out data/layer3/crops \
    --manifest data/layer3/crops_manifest.csv
```

Open the manifest CSV in Label Studio or a spreadsheet, annotate, then split into `train.csv` / `val.csv`. Keep the metadata columns (`source_image`, `part`, `detection_confidence`, bbox coords) — they're handy for stratified splits and per-part evaluation later.

> **L2/L3 data caveat.** Roboflow part-detection datasets do **not** provide damage-type or severity labels. You'll need a damage-annotated source (e.g. CarDD, Roboflow "car damage" projects, or your own insurance claim photos) to label what comes out of `extract_crops.py`.

### Layer 2 — ingesting Roboflow damage-detection datasets

Roboflow damage datasets give you bounding boxes around each damaged region with a single damage-type label (e.g. `dent`, `scratch`, `broken_glass`). `scripts/prepare_roboflow_l2_dataset.py` turns every bbox into a cropped image and writes a single-label row in our multi-label CSV — which is a valid special case of multi-label training.

```bash
# Download each L2 Roboflow project as YOLOv8 zip and unzip to ~/roboflow_l2/<name>/.
for name in car-damage-zxk33:zxk33 car-damage-wpmh2:wpmh2 car-damage-detection-eyy6t:eyy6t automobile-damage-detection:auto; do
    dir=${name%:*}; prefix=${name#*:}
    python scripts/prepare_roboflow_l2_dataset.py \
        --input ~/roboflow_l2/$dir \
        --mapping configs/roboflow_mappings/damage_types.yaml \
        --output data/layer2 \
        --prefix $prefix \
        --on-unknown skip
done

# Validate each split CSV (one per prefix per split):
python scripts/validate_dataset.py --layer 2 --root data/layer2 --csv data/layer2/zxk33_train.csv
```

Merge the per-dataset CSVs however suits your splitting strategy (a simple `cat` of the `*_train.csv` files works; or shuffle + stratify with pandas). Keep `<prefix>_valid.csv` files distinct from train so you can hold out an entire source dataset for out-of-distribution validation.

### Layer 3 — ingesting severity datasets

L3 accepts two Roboflow export shapes:

- **Detection** (severity is the bbox class name): each bbox → one crop → one row with that severity.
- **Classification** (severity is the folder name): each image → one row, no cropping.

```bash
# Detection-format dataset:
python scripts/prepare_roboflow_l3_dataset.py \
    --input ~/roboflow_l3/car-damage-datasets \
    --mapping configs/roboflow_mappings/severity.yaml \
    --output data/layer3 \
    --format detection \
    --prefix cdd \
    --on-unknown skip

# Classification-format dataset (Roboflow folders-per-class export):
python scripts/prepare_roboflow_l3_dataset.py \
    --input ~/roboflow_l3/car-accident-severity \
    --mapping configs/roboflow_mappings/severity.yaml \
    --output data/layer3 \
    --format classification \
    --prefix cas \
    --on-unknown skip

python scripts/validate_dataset.py --layer 3 --root data/layer3 --csv data/layer3/cdd_train.csv
```

The ingestor fills `part` and `damage_type` with the placeholder `"unknown"`. The L3 model doesn't consume those columns as inputs (they're metadata), so training still works. The `repair_or_replace` column is auto-filled from a severity rule (`severe`/`total_loss` → 1, else 0); pass `--no-rule-repair` to leave it blank for human annotators.

> **L3 data caveat.** Most public "accident severity" datasets score severity at the whole-image (or whole-accident) level, not per damaged part. Our L3 model is designed for part-level crops, so training it on image-level data is a noisy bootstrap — good enough for a working MVP, not a final production model. Plan to either (a) re-annotate a small held-out set at part level for evaluation, or (b) feed proprietary insurance claim data once you have it.

---

## Fine-tuning guide

Each layer has its own config in `configs/` and its own entrypoint. See `data/README.md` for the expected annotation format.

### Layer 1 — Part detector (YOLO)

```bash
# After running prepare_roboflow_dataset.py (above), data/layer1/ is populated.
# Update data/layer1/data.yaml to point at disjoint train/val splits.
python scripts/validate_dataset.py --layer 1 --root data/layer1
python training/train_layer1.py --config configs/layer1.yaml
```

Ultralytics handles the training loop, AMP, mosaic/mixup augmentation, schedule, and best-weights selection. Outputs land under `runs/layer1/<run_name>/weights/best.pt`. Point `configs/inference.yaml:layer1.weights` at that file.

### Layer 2 — Damage type (multi-label)

```bash
# Populate data/layer2/crops/ + train.csv + val.csv (see data/layer2/annotations.sample.csv).
python scripts/validate_dataset.py --layer 2 --root data/layer2 --csv data/layer2/train.csv
python training/train_layer2.py --config configs/layer2.yaml
```

The trainer uses BCE-with-logits (optionally class-balanced via `pos_weight`), mixed precision, EMA, cosine LR with warmup, and early stopping. Best checkpoint → `checkpoints/layer2/layer2_best.pt`.

### Layer 3 — Severity + repair/replace

```bash
python scripts/validate_dataset.py --layer 3 --root data/layer3 --csv data/layer3/train.csv
python training/train_layer3.py --config configs/layer3.yaml
```

Uses the CORAL ordinal-regression loss on severity and BCE on repair/replace. Best checkpoint → `checkpoints/layer3/layer3_best.pt`.

After each layer is fine-tuned, update `configs/inference.yaml` with the checkpoint paths and restart the API; the report's `pretrained_baseline` flag will flip to `false` once all heads are fine-tuned.

---

## API

Auto-generated OpenAPI docs live at `http://localhost:8000/docs`. Key endpoints:

| Endpoint | Method | Description |
|-----------|--------|-------------|
| `/health` | GET | Liveness + model status |
| `/metrics` | GET | Prometheus metrics |
| `/assess` | POST | **Sync** assessment on a single image (`multipart/form-data`, field `file`) |
| `/assess/batch` | POST | **Async** batch via Celery; returns a `job_id` |
| `/jobs/{job_id}` | GET | Poll batch job status + result |

Per-endpoint rate limits are configurable in `configs/api.yaml` and enforced by `slowapi`.

### Response schema (excerpt)

```json
{
  "image_id": "claim_12345.jpg",
  "parts_detected": 3,
  "parts_damaged": 2,
  "parts_requiring_replacement": 1,
  "overall_assessment": "major_damage",
  "parts": [
    {
      "part": "bumper",
      "detection_confidence": 0.93,
      "bbox_xyxy_px": [128, 192, 384, 312],
      "damage_types": [
        {"type": "dent",    "probability": 0.82},
        {"type": "scratch", "probability": 0.61}
      ],
      "primary_damage_type": "dent",
      "severity": {"grade": "moderate", "grade_index": 1, "grade_confidence": 0.64, "probs": {...}},
      "recommendation": "repair",
      "repair_probability": 0.71,
      "replace_probability": 0.29,
      "pretrained_baseline": false
    }
  ],
  "model_versions": {"layer1": "yolov8x_v1", "layer2": "convnextv2_large_v1", "layer3": "swinv2_large_v1"},
  "pretrained_baseline": false,
  "schema_version": "1.0"
}
```

Full Pydantic definitions: `api/schemas.py`.

---

## Model versioning / A/B testing

`configs/inference.yaml` declares the active version tag per layer (`layer{1,2,3}.version`). Ship a new model by:

1. Saving its checkpoint.
2. Registering a new builder in `models/registry.py` if the architecture differs.
3. Editing `inference.yaml` to point at the new weights + version tag.

The report includes `model_versions` so downstream systems can bucket predictions by model generation for A/B analysis.

---

## Observability

- **structlog**-formatted JSON logs with a per-request `x-request-id` header (see `api/middleware.py`).
- **Prometheus** metrics at `/metrics`: request counters, inference latency histogram, parts-detected histogram, assessment-error counter.
- **Celery retries** with exponential backoff on task failure (`api/tasks.py`).

---

## Export for production

```bash
python scripts/export_onnx.py --layer all --out exports/
```

Produces ONNX + TorchScript artifacts for all three layers, ready for TensorRT / ONNXRuntime deployments.

---

## Testing & CI

```bash
pytest -q                 # fast unit + API tests (stubbed models, CPU-only)
pytest -m integration     # exercises the real models (requires weights)
```

GitHub Actions (`.github/workflows/ci.yml`) runs `ruff`, `black --check`, and the fast test suite on every push.

---

## License

Proprietary — © alrawagah-cyber.
