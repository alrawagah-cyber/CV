"""API tests: /health returns 'starting' when models not loaded; /assess returns
the expected JSON schema when an assessor is injected into app.state.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client(stub_assessor, monkeypatch):
    # Prevent real model loading in lifespan.
    monkeypatch.setenv("CDP_LOAD_MODELS", "0")
    from api.main import create_app

    app = create_app()
    with TestClient(app) as c:
        # Inject stub assessor
        c.app.state.assessor = stub_assessor
        yield c


def _png_bytes() -> bytes:
    img = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8) + 200)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True
    assert "version" in body


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "cdp_api_requests_total" in r.text or "cdp_" in r.text


def test_assess_success(client):
    files = {"file": ("t.png", _png_bytes(), "image/png")}
    r = client.post("/assess", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["schema_version"] == "1.0"
    assert body["parts_detected"] >= 1
    assert isinstance(body["parts"], list)


def test_assess_wrong_content_type(client):
    files = {"file": ("t.txt", b"hello", "text/plain")}
    r = client.post("/assess", files=files)
    assert r.status_code == 400
