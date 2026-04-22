"""Tests for the GCS weight fetcher — uses a fake storage client so the
tests work offline and without google-cloud-storage credentials.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from api.weight_fetcher import DEFAULT_TARGETS, fetch_weights_if_configured


def test_no_op_when_bucket_not_set(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CDP_WEIGHTS_BUCKET", raising=False)
    result = fetch_weights_if_configured(repo_root=tmp_path)
    assert result == []


def test_skips_existing_weights(tmp_path: Path, monkeypatch):
    # Pre-create all 3 weights on disk.
    for dst in DEFAULT_TARGETS.values():
        p = tmp_path / dst
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake-weight")

    called = {"n": 0}

    class FakeBlob:
        def exists(self, **_):
            called["n"] += 1
            return True

        def download_to_filename(self, path):
            raise AssertionError("should not download when weight exists")

    class FakeBucket:
        def blob(self, key):
            return FakeBlob()

    class FakeClient:
        def bucket(self, name):
            return FakeBucket()

    # Patch google.cloud.storage.Client
    import api.weight_fetcher as wf

    monkeypatch.setattr(
        wf,
        "fetch_weights_if_configured",
        wf.fetch_weights_if_configured,  # don't actually re-patch; let original run
    )
    # Inject fake storage module
    import sys
    import types

    fake_storage = types.ModuleType("google.cloud.storage")
    fake_storage.Client = lambda: FakeClient()
    fake_cloud = types.ModuleType("google.cloud")
    fake_cloud.storage = fake_storage
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.cloud", fake_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", fake_storage)

    refreshed = fetch_weights_if_configured(repo_root=tmp_path, bucket="dummy")
    assert refreshed == []
    # blob.exists was never called because local files short-circuited the check
    assert called["n"] == 0


def test_downloads_missing_weights(tmp_path: Path, monkeypatch):
    # Do not pre-create any weights — all 3 should be downloaded.
    downloaded: list[str] = []

    class FakeBlob:
        def __init__(self, key: str):
            self.key = key

        def exists(self, **_):
            return True

        def download_to_filename(self, path):
            Path(path).write_bytes(b"downloaded:" + self.key.encode())
            downloaded.append(str(path))

    class FakeBucket:
        def blob(self, key: str) -> FakeBlob:
            return FakeBlob(key)

    class FakeClient:
        def bucket(self, name):
            return FakeBucket()

    import sys
    import types

    fake_storage = types.ModuleType("google.cloud.storage")
    fake_storage.Client = lambda: FakeClient()
    fake_cloud = types.ModuleType("google.cloud")
    fake_cloud.storage = fake_storage
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.cloud", fake_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", fake_storage)

    refreshed = fetch_weights_if_configured(repo_root=tmp_path, bucket="dummy", prefix="test/prefix")
    # All known targets (.pt + .onnx + .onnx.data) downloaded.
    assert len(refreshed) == len(DEFAULT_TARGETS)
    for dst in DEFAULT_TARGETS.values():
        assert (tmp_path / dst).exists()


def test_missing_blob_raises(tmp_path: Path, monkeypatch):
    class FakeBlob:
        def exists(self, **_):
            return False

        def download_to_filename(self, path):
            raise AssertionError("should not download when blob missing")

    class FakeBucket:
        def blob(self, key):
            return FakeBlob()

    class FakeClient:
        def bucket(self, name):
            return FakeBucket()

    import sys
    import types

    fake_storage = types.ModuleType("google.cloud.storage")
    fake_storage.Client = lambda: FakeClient()
    fake_cloud = types.ModuleType("google.cloud")
    fake_cloud.storage = fake_storage
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.cloud", fake_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", fake_storage)

    with pytest.raises(FileNotFoundError, match="Weight blob not found"):
        fetch_weights_if_configured(repo_root=tmp_path, bucket="dummy")
