"""Tests for the total-loss pre-screen and its integration into ClaimAssessor."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from inference.total_loss_screen import TotalLossScreener


# --------------------------------------------------------------------------- #
# Unit tests for TotalLossScreener
# --------------------------------------------------------------------------- #
class TestTotalLossScreenerDisabled:
    def test_no_weights_returns_disabled(self):
        screener = TotalLossScreener(weights=None)
        assert not screener.enabled
        assert screener.model is None

    def test_screen_returns_false_when_disabled(self, sample_rgb_image):
        screener = TotalLossScreener(weights=None)
        is_tl, conf = screener.screen(sample_rgb_image)
        assert is_tl is False
        assert conf == 0.0


class TestTotalLossScreenerEnabled:
    @pytest.fixture()
    def screener_checkpoint(self, tmp_path):
        """Save a tiny ResNet-18 checkpoint with a binary head."""
        from torchvision import models

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        path = tmp_path / "total_loss_screen.pt"
        torch.save({"state_dict": model.state_dict()}, path)
        return path

    def test_loads_checkpoint(self, screener_checkpoint):
        screener = TotalLossScreener(weights=screener_checkpoint, device="cpu")
        assert screener.enabled
        assert screener.model is not None

    def test_screen_returns_tuple(self, screener_checkpoint, sample_rgb_image):
        screener = TotalLossScreener(weights=screener_checkpoint, device="cpu")
        result = screener.screen(sample_rgb_image)
        assert isinstance(result, tuple)
        assert len(result) == 2
        is_tl, conf = result
        assert isinstance(is_tl, bool)
        assert 0.0 <= conf <= 1.0


# --------------------------------------------------------------------------- #
# Integration: screener wired into ClaimAssessor
# --------------------------------------------------------------------------- #
class TestTotalLossScreenInAssessor:
    def test_screened_total_loss_skips_pipeline(self, stub_assessor, sample_rgb_image):
        """When the screener fires, the report should carry screened_total_loss=True."""
        stub_assessor.screener = TotalLossScreener(weights=None)
        stub_assessor.screener.enabled = True
        stub_assessor.screener.model = True  # truthy sentinel
        stub_assessor.screener.screen = lambda _img: (True, 0.99)
        stub_assessor.cfg.total_loss_screen_threshold = 0.85

        report = stub_assessor.assess(sample_rgb_image, image_id="tl_test")
        assert report["screened_total_loss"] is True
        assert report["overall_assessment"] == "total_loss"
        assert report["total_loss_confidence"] == 0.99
        assert report["parts"] == []
        assert "inference_ms" in report

    def test_below_threshold_runs_normal_pipeline(self, stub_assessor, sample_rgb_image):
        """Confidence below threshold should fall through to the 3-layer pipeline."""
        stub_assessor.screener = TotalLossScreener(weights=None)
        stub_assessor.screener.enabled = True
        stub_assessor.screener.model = True
        stub_assessor.screener.screen = lambda _img: (True, 0.50)
        stub_assessor.cfg.total_loss_screen_threshold = 0.85

        report = stub_assessor.assess(sample_rgb_image, image_id="not_tl")
        assert report.get("screened_total_loss") is not True
        assert report["parts_detected"] == 1

    def test_disabled_screener_runs_normal_pipeline(self, stub_assessor, sample_rgb_image):
        report = stub_assessor.assess(sample_rgb_image, image_id="normal")
        assert report.get("screened_total_loss") is not True
        assert report["parts_detected"] == 1

    def test_batch_with_mix(self, stub_assessor, sample_rgb_image):
        """In a batch, only images that pass the threshold are screened."""
        call_count = {"n": 0}

        def _screen(img):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return True, 0.95
            return False, 0.1

        stub_assessor.screener = TotalLossScreener(weights=None)
        stub_assessor.screener.enabled = True
        stub_assessor.screener.model = True
        stub_assessor.screener.screen = _screen
        stub_assessor.cfg.total_loss_screen_threshold = 0.85

        reports = stub_assessor.assess_batch([sample_rgb_image, sample_rgb_image])
        assert len(reports) == 2
        assert reports[0]["screened_total_loss"] is True
        assert reports[0]["parts"] == []
        assert reports[1].get("screened_total_loss") is not True
        assert reports[1]["parts_detected"] == 1
