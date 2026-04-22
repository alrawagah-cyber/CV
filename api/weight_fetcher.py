"""Fetch model checkpoints from Google Cloud Storage at container startup.

The Docker image intentionally does NOT bake in the ~2GB model weights —
doing so would slow down builds, bloat the registry, and tie deploys to
data. Instead, the container pulls weights from a GCS bucket on first
startup and caches them to the local ``checkpoints/`` directory.

Configure via environment variables::

    CDP_WEIGHTS_BUCKET   — GCS bucket name (no "gs://" prefix).
    CDP_WEIGHTS_PREFIX   — Optional prefix inside the bucket (default "").

Expected layout under <bucket>/<prefix>/::

    layer1/best.pt
    layer2/best.pt
    layer3/best.pt

If ``CDP_WEIGHTS_BUCKET`` is unset, the fetcher is a no-op (useful for
local runs with pre-mounted weights).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Where the API configs expect weights to live on disk.
# Includes .onnx + .onnx.data alongside .pt so the assessor can pick
# ONNX Runtime when present (2-3x faster CPU inference). Missing files
# are skipped silently — only present blobs are downloaded.
DEFAULT_TARGETS: dict[str, str] = {
    # PyTorch checkpoints (always present)
    "layer1/best.pt": "checkpoints/layer1/best.pt",
    "layer2/best.pt": "checkpoints/layer2/best.pt",
    "layer3/best.pt": "checkpoints/layer3/best.pt",
    # ONNX exports (optional — fetcher tolerates missing blobs)
    "layer1/best.onnx": "checkpoints/layer1/best.onnx",
    "layer2/best.onnx": "checkpoints/layer2/best.onnx",
    "layer2/best.onnx.data": "checkpoints/layer2/layer2_damage.onnx.data",
    "layer3/best.onnx": "checkpoints/layer3/best.onnx",
    "layer3/best.onnx.data": "checkpoints/layer3/layer3_severity.onnx.data",
}


def fetch_weights_if_configured(
    repo_root: str | Path = ".",
    bucket: str | None = None,
    prefix: str | None = None,
    targets: dict[str, str] | None = None,
    force: bool = False,
) -> list[str]:
    """Download missing weights from GCS. Returns list of local paths refreshed.

    Idempotent — files already present are skipped unless ``force=True``.
    """
    bucket_name = bucket if bucket is not None else os.environ.get("CDP_WEIGHTS_BUCKET")
    if not bucket_name:
        logger.info("CDP_WEIGHTS_BUCKET not set; skipping weight fetch.")
        return []

    prefix_str = (prefix if prefix is not None else os.environ.get("CDP_WEIGHTS_PREFIX", "")).strip("/")
    mapping = targets if targets is not None else DEFAULT_TARGETS
    root = Path(repo_root)

    try:
        from google.cloud import storage
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-storage is required for CDP_WEIGHTS_BUCKET; install it "
            "or unset the environment variable."
        ) from exc

    client = storage.Client()
    gcs_bucket = client.bucket(bucket_name)

    refreshed: list[str] = []
    for src, dst in mapping.items():
        dst_path = root / dst
        if dst_path.exists() and not force:
            logger.info("Weight already present, skipping: %s", dst_path)
            continue

        key = f"{prefix_str}/{src}" if prefix_str else src
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        blob = gcs_bucket.blob(key)
        if not blob.exists(client=client):
            # Optional blobs (.onnx, .onnx.data) may not exist yet — log but don't fail.
            if src.endswith((".onnx", ".onnx.data")):
                logger.info("Optional ONNX blob not in bucket, skipping: gs://%s/%s", bucket_name, key)
                continue
            raise FileNotFoundError(
                f"Weight blob not found at gs://{bucket_name}/{key}. "
                f"Check CDP_WEIGHTS_BUCKET / CDP_WEIGHTS_PREFIX."
            )
        logger.info("Downloading gs://%s/%s -> %s", bucket_name, key, dst_path)
        blob.download_to_filename(str(dst_path))
        refreshed.append(str(dst_path))

    return refreshed
