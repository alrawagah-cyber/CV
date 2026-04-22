"""Tests for the admin feedback browser: list_bundles, get_bundle, and API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.feedback_store import LocalFeedbackStore


# --------------------------------------------------------------------------- #
# Unit tests for LocalFeedbackStore.list_bundles / get_bundle
# --------------------------------------------------------------------------- #
@pytest.fixture
def local_store(tmp_path):
    return LocalFeedbackStore(root=tmp_path / "feedback")


def _write_bundle(store: LocalFeedbackStore, claim_id: str, feedback_id: str, **manifest_overrides):
    manifest = {
        "feedback_id": feedback_id,
        "claim_id": claim_id,
        "adjuster_id": "adj-1",
        "captured_at": "2025-06-01T12:00:00+00:00",
        "has_image": False,
        "notes": "Test note",
        "schema_version": "1.0",
        "parts_delta": 2,
    }
    manifest.update(manifest_overrides)
    predicted = {"image_id": "test.png", "parts": []}
    corrected = {"corrected_parts": [], "notes": "ok"}
    store.put_bundle(
        claim_id=claim_id,
        feedback_id=feedback_id,
        manifest=manifest,
        predicted=predicted,
        corrected=corrected,
    )


def test_list_bundles_empty(local_store):
    assert local_store.list_bundles() == []


def test_list_bundles_returns_summaries(local_store):
    _write_bundle(local_store, "CLM-001", "fb-aaa")
    _write_bundle(local_store, "CLM-002", "fb-bbb", adjuster_id="adj-2", notes="Another note")

    bundles = local_store.list_bundles()
    assert len(bundles) == 2
    ids = {b["feedback_id"] for b in bundles}
    assert ids == {"fb-aaa", "fb-bbb"}

    for b in bundles:
        assert "claim_id" in b
        assert "adjuster_id" in b
        assert "captured_at" in b
        assert "has_image" in b
        assert "parts_delta" in b
        assert "notes" in b


def test_list_bundles_detects_image(local_store):
    _write_bundle(local_store, "CLM-003", "fb-ccc")
    # Manually place an image file in the bundle directory.
    bundle_dir = local_store.root / "CLM-003" / "fb-ccc"
    (bundle_dir / "image.jpg").write_bytes(b"\xff\xd8dummy")

    bundles = local_store.list_bundles()
    assert len(bundles) == 1
    assert bundles[0]["has_image"] is True


def test_get_bundle_found(local_store):
    _write_bundle(local_store, "CLM-010", "fb-xyz")
    bundle = local_store.get_bundle("fb-xyz")
    assert bundle is not None
    assert "manifest" in bundle
    assert "predicted" in bundle
    assert "corrected" in bundle
    assert bundle["manifest"]["feedback_id"] == "fb-xyz"


def test_get_bundle_not_found(local_store):
    assert local_store.get_bundle("nonexistent") is None


# --------------------------------------------------------------------------- #
# API endpoint tests
# --------------------------------------------------------------------------- #
@pytest.fixture
def admin_client(tmp_path, monkeypatch):
    monkeypatch.setenv("CDP_LOAD_MODELS", "0")
    from api.main import create_app

    app = create_app()
    store = LocalFeedbackStore(root=tmp_path / "feedback")
    with TestClient(app) as c:
        c.app.state.feedback_store = store
        c.app.state._test_store = store  # expose for helpers
        yield c


def test_admin_feedback_list_empty(admin_client):
    r = admin_client.get("/admin/feedback")
    assert r.status_code == 200
    assert r.json() == []


def test_admin_feedback_list_with_data(admin_client):
    store = admin_client.app.state._test_store
    _write_bundle(store, "CLM-100", "fb-111")
    _write_bundle(store, "CLM-200", "fb-222", captured_at="2025-07-01T00:00:00+00:00")

    r = admin_client.get("/admin/feedback")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 2
    # Most recent first.
    assert data[0]["feedback_id"] == "fb-222"


def test_admin_feedback_list_pagination(admin_client):
    store = admin_client.app.state._test_store
    for i in range(5):
        _write_bundle(store, f"CLM-{i}", f"fb-{i}", captured_at=f"2025-01-0{i + 1}T00:00:00+00:00")

    r = admin_client.get("/admin/feedback?limit=2&offset=0")
    assert r.status_code == 200
    assert len(r.json()) == 2

    r2 = admin_client.get("/admin/feedback?limit=2&offset=2")
    assert r2.status_code == 200
    assert len(r2.json()) == 2

    r3 = admin_client.get("/admin/feedback?limit=2&offset=4")
    assert r3.status_code == 200
    assert len(r3.json()) == 1


def test_admin_feedback_detail_found(admin_client):
    store = admin_client.app.state._test_store
    _write_bundle(store, "CLM-300", "fb-detail")

    r = admin_client.get("/admin/feedback/fb-detail")
    assert r.status_code == 200
    body = r.json()
    assert "manifest" in body
    assert "predicted" in body
    assert "corrected" in body
    assert body["manifest"]["claim_id"] == "CLM-300"


def test_admin_feedback_detail_not_found(admin_client):
    r = admin_client.get("/admin/feedback/does-not-exist")
    assert r.status_code == 404


def test_admin_page_served(admin_client):
    r = admin_client.get("/admin")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "Feedback Browser" in r.text


def test_admin_endpoints_open_without_api_key(tmp_path, monkeypatch):
    """Admin endpoints should be accessible even when CDP_API_KEYS is set."""
    monkeypatch.setenv("CDP_LOAD_MODELS", "0")
    monkeypatch.setenv("CDP_API_KEYS", "secret-key-123")
    from api.main import create_app

    app = create_app()
    store = LocalFeedbackStore(root=tmp_path / "feedback")
    with TestClient(app) as c:
        c.app.state.feedback_store = store
        # No X-API-Key header — should still succeed for admin paths.
        r = c.get("/admin/feedback")
        assert r.status_code == 200

        r2 = c.get("/admin")
        assert r2.status_code == 200
