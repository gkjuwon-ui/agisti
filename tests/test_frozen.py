"""
Tests for agisti.frozen — discovery, mask, integrity.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest
import torch

from agisti.types import FreezeLevel
from agisti.config import FrozenDiscoveryConfig
from agisti.frozen.discovery import FrozenZoneDiscovery, AdaptiveFrozenDiscovery
from agisti.frozen.mask import FrozenMask
from agisti.frozen.integrity import IntegrityMonitor, QuickIntegrityCheck


# ─── FrozenZoneDiscovery Tests ─────────────────────

class TestFrozenZoneDiscovery:
    """Tests for frozen zone discovery and sensitivity analysis."""

    def _make_model_params(self) -> dict[str, torch.Tensor]:
        """Create mock model parameters."""
        return {
            "layers.0.weight": torch.randn(64, 64),
            "layers.1.weight": torch.randn(64, 64),
            "layers.2.weight": torch.randn(64, 64),
            "embed.weight": torch.randn(1000, 64),
            "lm_head.weight": torch.randn(1000, 64),
        }

    def test_creation(self):
        config = FrozenDiscoveryConfig()
        discovery = FrozenZoneDiscovery(config)
        assert discovery is not None

    def test_analyze_sensitivity(self):
        config = FrozenDiscoveryConfig()
        discovery = FrozenZoneDiscovery(config)
        params = self._make_model_params()
        result = discovery.analyze_sensitivity(params)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_discover_frozen_layers(self):
        config = FrozenDiscoveryConfig()
        discovery = FrozenZoneDiscovery(config)
        params = self._make_model_params()
        frozen = discovery.discover(params)
        assert isinstance(frozen, (set, list, dict))

    def test_sensitivity_scores_are_finite(self):
        config = FrozenDiscoveryConfig()
        discovery = FrozenZoneDiscovery(config)
        params = self._make_model_params()
        scores = discovery.analyze_sensitivity(params)
        for name, score in scores.items():
            assert not (hasattr(score, '__float__') and
                       (score != score)),  # NaN check
                       f"{name} has NaN sensitivity"


class TestAdaptiveFrozenDiscovery:
    """Tests for adaptive frozen zone discovery."""

    def _make_params(self) -> dict[str, torch.Tensor]:
        return {
            f"layers.{i}.weight": torch.randn(32, 32)
            for i in range(4)
        }

    def test_creation(self):
        discovery = AdaptiveFrozenDiscovery(
            FrozenDiscoveryConfig(),
            model_params=self._make_params(),
        )
        assert discovery is not None

    def test_update_with_iteration_result(self):
        from agisti.types import IterationResult
        discovery = AdaptiveFrozenDiscovery(
            FrozenDiscoveryConfig(),
            model_params=self._make_params(),
        )
        result = IterationResult(
            iteration_id=0,
            proposed_delta_norm=0.01,
            virtual_loss_before=2.0,
            virtual_loss_after=1.8,
            refined_delta_norm=0.01,
            quick_bench_scores={"math": 0.5},
            accepted=True,
            wall_time_seconds=1.0,
            gpu_memory_peak_gb=10.0,
        )
        # Should not raise
        discovery.update(result)


# ─── FrozenMask Tests ──────────────────────────────

class TestFrozenMask:
    """Tests for FrozenMask with SHA-256 integrity."""

    def _make_params(self) -> dict[str, torch.Tensor]:
        torch.manual_seed(42)
        return {
            "layer.0.weight": torch.randn(32, 32),
            "layer.1.weight": torch.randn(32, 32),
            "layer.2.weight": torch.randn(32, 32),
        }

    def test_create_mask(self):
        params = self._make_params()
        mask = FrozenMask.from_params(
            params,
            frozen_names={"layer.0.weight"},
        )
        assert mask is not None
        assert mask.is_frozen("layer.0.weight") is True
        assert mask.is_frozen("layer.1.weight") is False

    def test_frozen_count(self):
        params = self._make_params()
        mask = FrozenMask.from_params(
            params,
            frozen_names={"layer.0.weight", "layer.1.weight"},
        )
        assert mask.frozen_count == 2

    def test_integrity_check_passes(self):
        params = self._make_params()
        mask = FrozenMask.from_params(
            params,
            frozen_names={"layer.0.weight"},
        )
        assert mask.check_integrity(params) is True

    def test_integrity_check_fails_on_modification(self):
        params = self._make_params()
        mask = FrozenMask.from_params(
            params,
            frozen_names={"layer.0.weight"},
        )
        # Modify frozen parameter
        params["layer.0.weight"] += 1.0
        assert mask.check_integrity(params) is False

    def test_sha256_hash_deterministic(self):
        params = self._make_params()
        mask1 = FrozenMask.from_params(
            params, frozen_names={"layer.0.weight"}
        )
        mask2 = FrozenMask.from_params(
            params, frozen_names={"layer.0.weight"}
        )
        assert mask1.get_hash("layer.0.weight") == mask2.get_hash("layer.0.weight")

    def test_unfrozen_params(self):
        params = self._make_params()
        mask = FrozenMask.from_params(
            params,
            frozen_names={"layer.0.weight"},
        )
        unfrozen = mask.unfrozen_names
        assert "layer.1.weight" in unfrozen
        assert "layer.2.weight" in unfrozen
        assert "layer.0.weight" not in unfrozen

    def test_empty_frozen_set(self):
        params = self._make_params()
        mask = FrozenMask.from_params(params, frozen_names=set())
        assert mask.frozen_count == 0
        assert mask.check_integrity(params) is True

    def test_all_frozen(self):
        params = self._make_params()
        mask = FrozenMask.from_params(
            params,
            frozen_names=set(params.keys()),
        )
        assert mask.frozen_count == 3

    def test_save_and_load(self):
        params = self._make_params()
        mask = FrozenMask.from_params(
            params, frozen_names={"layer.0.weight"}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mask.json"
            mask.save(path)
            loaded = FrozenMask.load(path)
            assert loaded.frozen_count == 1
            assert loaded.is_frozen("layer.0.weight")

    def test_violation_list(self):
        params = self._make_params()
        mask = FrozenMask.from_params(
            params,
            frozen_names={"layer.0.weight", "layer.1.weight"},
        )
        # Modify one frozen param
        params["layer.0.weight"] += 0.001
        violations = mask.get_violations(params)
        assert "layer.0.weight" in violations
        assert "layer.1.weight" not in violations


# ─── IntegrityMonitor Tests ────────────────────────

class TestIntegrityMonitor:
    """Tests for IntegrityMonitor."""

    def _make_params(self) -> dict[str, torch.Tensor]:
        torch.manual_seed(42)
        return {
            f"layer.{i}.weight": torch.randn(16, 16)
            for i in range(3)
        }

    def test_creation(self):
        params = self._make_params()
        mask = FrozenMask.from_params(
            params, frozen_names={"layer.0.weight"}
        )
        monitor = IntegrityMonitor(mask)
        assert monitor is not None

    def test_verify_passes(self):
        params = self._make_params()
        mask = FrozenMask.from_params(
            params, frozen_names={"layer.0.weight"}
        )
        monitor = IntegrityMonitor(mask)
        assert monitor.verify(params) is True

    def test_verify_fails_on_modification(self):
        params = self._make_params()
        mask = FrozenMask.from_params(
            params, frozen_names={"layer.0.weight"}
        )
        monitor = IntegrityMonitor(mask)
        params["layer.0.weight"] += 1.0
        assert monitor.verify(params) is False

    def test_violation_count(self):
        params = self._make_params()
        mask = FrozenMask.from_params(
            params,
            frozen_names={"layer.0.weight", "layer.1.weight"},
        )
        monitor = IntegrityMonitor(mask)
        params["layer.0.weight"] += 1.0
        params["layer.1.weight"] += 1.0
        count = monitor.count_violations(params)
        assert count == 2


class TestQuickIntegrityCheck:
    """Tests for QuickIntegrityCheck (sampling-based)."""

    def _make_params(self) -> dict[str, torch.Tensor]:
        torch.manual_seed(42)
        return {
            f"layer.{i}.weight": torch.randn(16, 16)
            for i in range(5)
        }

    def test_quick_check_passes(self):
        params = self._make_params()
        mask = FrozenMask.from_params(
            params,
            frozen_names={"layer.0.weight", "layer.1.weight"},
        )
        check = QuickIntegrityCheck(mask, sample_fraction=0.5)
        assert check.verify(params) is True

    def test_quick_check_fails(self):
        params = self._make_params()
        mask = FrozenMask.from_params(
            params,
            frozen_names={"layer.0.weight", "layer.1.weight"},
        )
        check = QuickIntegrityCheck(mask, sample_fraction=1.0)
        params["layer.0.weight"] += 1.0
        assert check.verify(params) is False
