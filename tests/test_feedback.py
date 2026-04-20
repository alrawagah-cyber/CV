"""Tests for the feedback capture (Stage 1 of the human-in-the-loop loop)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.feedback_store import (
    LocalFeedbackStore,
    _ext_for_content_type,
    _sanitize_segment,
    build_store_from_config,
)


# --------------------------------------------------------------------------- #
# FeedbackStore unit tests
# --------------------------------------------------------------------------- #
def test_local_store_writes_bundle(tmp_path: Path):
    store = LocalFeedbackStore(tmp_path / "fb")
    uri = store.put_bundle(
        claim_id="CLM-123",
        feedback_id="abcd1234",
        manifest={"feedback_id": "abcd1234", "claim_id": "CLM-123"},
        predicted={"parts_detected": 1},
        corrected={"corrected_parts": [{"part": "door"}]},
        image_bytes=b"\x89PNG\r\nfake",
        image_content_type="image/png",
    )

    bundle = tmp_path / "fb" / "CLM-123" / "abcd1234"
    assert uri.startswith("file://")
    assert (bundle / "manifest.json").exists()
    assert (bundle / "predicted.json").exists()
    assert (bundle / "corrected.json").exists()
    assert (bundle / "image.png").exists()
    assert json.loads((bundle / "corrected.json").read_text())["corrected_parts"][0]["part"] == "door"


def test_local_store_sanitizes_path_segments(tmp_path: Path):
    store = LocalFeedbackStore(tmp_path / "fb")
    # Claim IDs containing path traversal must not escape the root.
    store.put_bundle(
        claim_id="../../etc/passwd",
        feedback_id="f1",
        manifest={},
        predicted={},
        corrected={},
    )
    assert not (tmp_path.parent / "etc" / "passwd").exists()
    # Everything lives under fb/
    assert any((tmp_path / "fb").rglob("manifest.json"))


def test_build_store_from_config_local(tmp_path: Path):
    store = build_store_from_config({"backend": "local", "local_root": str(tmp_path / "x")})
    assert isinstance(store, LocalFeedbackStore)


def test_build_store_from_config_gcs_requires_bucket():
    with pytest.raises(ValueError, match="bucket"):
        build_store_from_config({"backend": "gcs"})


def test_build_store_unknown_backend():
    with pytest.raises(ValueError, match="Unknown feedback backend"):
        build_store_from_config({"backend": "s3"})


def test_sanitize_segment_handles_unsafe_chars():
    assert _sanitize_segment("a/b") == "a_b"
    assert _sanitize_segment("../x") == "x"
    assert _sanitize_segment("") == "unknown"
    assert _sanitize_segment("CLM-123_abc") == "CLM-123_abc"


def test_ext_for_content_type():
    assert _ext_for_content_type("image/jpeg") == ".jpg"
    assert _ext_for_content_type("image/png") == ".png"
    assert _ext_for_content_type("image/webp") == ".webp"
    assert _ext_for_content_type(None) == ".jpg"


# --------------------------------------------------------------------------- #
# /feedback endpoint integration tests
# --------------------------------------------------------------------------- #
def _png_bytes() -> bytes:
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8) + 100)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _sample_original_report() -> dict:
    return {
        "image_id": "test.jpg",
        "image_width": 640,
        "image_height": 480,
        "parts_detected": 1,
        "parts_damaged": 1,
        "parts_requiring_replacement": 0,
        "overall_assessment": "minor_damage",
        "parts": [
            {
                "part": "mirror",
                "class_id": 7,
                "detection_confidence": 0.9,
                "bbox_xyxy_px": [10, 20, 100, 200],
                "bbox_xyxy_norm": [0.01, 0.04, 0.15, 0.42],
                "damage_types": [{"type": "dent", "probability": 0.8}],
                "damage_probs_all": {"dent": 0.8},
                "primary_damage_type": "dent",
                "severity": {
                    "grade": "minor",
                    "grade_index": 0,
                    "grade_confidence": 0.7,
                    "probs": {"minor": 0.7, "moderate": 0.2, "severe": 0.1, "total_loss": 0.0},
                },
                "recommendation": "repair",
                "repair_probability": 0.8,
                "replace_probability": 0.2,
                "pretrained_baseline": False,
            }
        ],
        "pretrained_baseline": False,
        "model_versions": {"layer1": "v1", "layer2": "v1", "layer3": "v1"},
    }


@pytest.fixture
def feedback_client(tmp_path: Path, monkeypatch):
    """Run the real FastAPI app but with a local feedback store rooted in tmp_path."""
    monkeypatch.setenv("CDP_LOAD_MODELS", "0")
    # Write a minimal feedback.yaml pointing at tmp_path.
    fb_cfg = tmp_path / "feedback.yaml"
    fb_cfg.write_text(f"backend: local\nlocal_root: {tmp_path / 'feedback'}\nmax_image_bytes: 1048576\n")
    monkeypatch.setenv("CDP_FEEDBACK_CONFIG", str(fb_cfg))

    from api.main import create_app

    app = create_app()
    with TestClient(app) as client:
        yield client, tmp_path


def test_feedback_multipart_with_image(feedback_client):
    client, tmp_path = feedback_client
    payload = {
        "claim_id": "CLM-42",
        "adjuster_id": "adj_007",
        "original_report": _sample_original_report(),
        "corrected_parts": [
            {
                "part": "mirror",
                "damage_types": ["dent", "shatter"],
                "primary_damage_type": "shatter",
                "severity": "severe",
                "recommendation": "replace",
                "adjuster_notes": "Glass is fully shattered.",
            }
        ],
        "corrected_overall_assessment": "major_damage",
        "notes": "Model missed glass shatter on the mirror.",
    }
    r = client.post(
        "/feedback",
        data={"feedback": json.dumps(payload)},
        files={"image": ("mirror.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["claim_id"] == "CLM-42"
    assert body["status"] == "stored"
    assert body["stored_at"].startswith("file://")

    # Verify the bundle on disk contains the corrected damage types.
    bundles = list((tmp_path / "feedback").rglob("corrected.json"))
    assert len(bundles) == 1
    data = json.loads(bundles[0].read_text())
    assert data["corrected_parts"][0]["primary_damage_type"] == "shatter"
    assert "shatter" in data["corrected_parts"][0]["damage_types"]


def test_feedback_json_only(feedback_client):
    client, tmp_path = feedback_client
    payload = {
        "claim_id": "CLM-42",
        "adjuster_id": "adj_007",
        "original_report": _sample_original_report(),
        "corrected_parts": [],
        "notes": "Report looked correct; no changes.",
    }
    r = client.post("/feedback", json=payload)
    assert r.status_code == 201, r.text
    # No image saved.
    bundle_dir = list((tmp_path / "feedback").rglob("manifest.json"))[0].parent
    assert not list(bundle_dir.glob("image.*"))


def test_feedback_rejects_oversize_image(feedback_client):
    client, _ = feedback_client
    # 2 MiB of zeros; max_image_bytes in the fixture is 1 MiB.
    big = b"\x00" * (2 * 1024 * 1024)
    payload = {
        "claim_id": "CLM-99",
        "adjuster_id": "adj_007",
        "original_report": _sample_original_report(),
    }
    r = client.post(
        "/feedback",
        data={"feedback": json.dumps(payload)},
        files={"image": ("huge.png", big, "image/png")},
    )
    assert r.status_code == 413


def test_feedback_rejects_non_image_upload(feedback_client):
    client, _ = feedback_client
    payload = {
        "claim_id": "CLM-99",
        "adjuster_id": "adj_007",
        "original_report": _sample_original_report(),
    }
    r = client.post(
        "/feedback",
        data={"feedback": json.dumps(payload)},
        files={"image": ("not_an_image.txt", b"just text", "text/plain")},
    )
    assert r.status_code == 400


def test_feedback_rejects_bad_json(feedback_client):
    client, _ = feedback_client
    r = client.post(
        "/feedback",
        data={"feedback": "{not valid json"},
        files={"image": ("x.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 400


def test_feedback_rejects_missing_claim_id(feedback_client):
    client, _ = feedback_client
    r = client.post("/feedback", json={"adjuster_id": "adj"})
    assert r.status_code == 400
