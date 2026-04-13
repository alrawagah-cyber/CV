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
scripts/         - download_weights, run_inference, export_onnx, validate_dataset
docker/          - Dockerfile.base / .api / .worker + prometheus.yml
```

---

## Fine-tuning guide

Each layer has its own config in `configs/` and its own entrypoint. See `data/README.md` for the expected annotation format.

### Layer 1 — Part detector (YOLO)

```bash
# Drop images into data/layer1/images/ and YOLO .txt labels into data/layer1/labels/.
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
