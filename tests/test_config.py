"""
Tests for agisti.config — configuration system.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agisti.config import (
    ArchSurgeryConfig,
    CatastropheConfig,
    CeilingBreakerConfig,
    CheckpointConfig,
    ConvergenceConfig,
    FullBenchConfig,
    FrozenDiscoveryConfig,
    GPUAllocation,
    GPUConfig,
    IterationConfig,
    MACRO_CONFIG,
    MetaStrategy,
    MICRO_CONFIG,
    MoEConfig,
    PHASE_0_CONFIG,
    PHASE_1_CONFIG,
    PhaseConfig,
    QuickBenchConfig,
    RunPodConfig,
    SurgeryConfig,
    FULL_BENCH_SUITES,
    PHASE0_STRATEGY,
    save_config,
    load_config,
)
from agisti.types import PhaseId


class TestIterationConfig:
    """Tests for IterationConfig."""

    def test_defaults(self):
        cfg = IterationConfig()
        assert cfg.problems_per_iteration > 0
        assert cfg.virtual_train_problems > 0
        assert cfg.checkpoint_every > 0

    def test_custom_values(self):
        cfg = IterationConfig(
            problems_per_iteration=200,
            virtual_train_problems=30,
            checkpoint_every=50,
        )
        assert cfg.problems_per_iteration == 200
        assert cfg.virtual_train_problems == 30


class TestSurgeryConfig:
    """Tests for SurgeryConfig."""

    def test_micro_config(self):
        assert MICRO_CONFIG.rank > 0
        assert MICRO_CONFIG.budget > 0
        assert MICRO_CONFIG.rank <= 16  # micro = small rank

    def test_macro_config(self):
        assert MACRO_CONFIG.rank > MICRO_CONFIG.rank
        assert MACRO_CONFIG.budget > MICRO_CONFIG.budget

    def test_custom(self):
        cfg = SurgeryConfig(rank=16, budget=0.02)
        assert cfg.rank == 16
        assert cfg.budget == 0.02


class TestArchSurgeryConfig:
    """Tests for ArchSurgeryConfig."""

    def test_defaults(self):
        cfg = ArchSurgeryConfig()
        assert isinstance(cfg.allowed_operations, list)
        assert len(cfg.allowed_operations) > 0


class TestQuickBenchConfig:
    """Tests for QuickBenchConfig."""

    def test_defaults(self):
        cfg = QuickBenchConfig()
        assert cfg.problems_per_domain > 0
        assert 0 < cfg.significance_level < 1

    def test_significance(self):
        cfg = QuickBenchConfig(significance_level=0.01)
        assert cfg.significance_level == 0.01


class TestFullBenchConfig:
    """Tests for FullBenchConfig."""

    def test_defaults(self):
        cfg = FullBenchConfig()
        assert isinstance(cfg.suites, list)
        assert len(cfg.suites) > 0

    def test_suites(self):
        assert isinstance(FULL_BENCH_SUITES, list)
        assert len(FULL_BENCH_SUITES) > 0


class TestFrozenDiscoveryConfig:
    """Tests for FrozenDiscoveryConfig."""

    def test_defaults(self):
        cfg = FrozenDiscoveryConfig()
        assert cfg.num_probes > 0
        assert 0 < cfg.sensitivity_threshold < 1


class TestMoEConfig:
    """Tests for MoEConfig."""

    def test_defaults(self):
        cfg = MoEConfig()
        assert cfg.num_experts > 0
        assert cfg.top_k > 0
        assert cfg.top_k <= cfg.num_experts


class TestCheckpointConfig:
    """Tests for CheckpointConfig."""

    def test_defaults(self):
        cfg = CheckpointConfig()
        assert cfg.max_checkpoints > 0
        assert cfg.keep_best > 0
        assert cfg.keep_best <= cfg.max_checkpoints


class TestConvergenceConfig:
    """Tests for ConvergenceConfig."""

    def test_defaults(self):
        cfg = ConvergenceConfig()
        assert cfg.window_size > 0
        assert cfg.plateau_threshold > 0


class TestCatastropheConfig:
    """Tests for CatastropheConfig."""

    def test_defaults(self):
        cfg = CatastropheConfig()
        assert cfg.sudden_collapse_threshold > 0
        assert cfg.plateau_window > 0
        assert cfg.crash_threshold > 0
        assert cfg.regression_threshold > 0
        assert cfg.divergence_norm_ratio > 0
        assert cfg.mode_collapse_threshold > 0
        assert cfg.loss_spike_ratio > 0
        assert cfg.stall_iterations > 0
        assert cfg.max_emergency_count > 0

    def test_thresholds_reasonable(self):
        cfg = CatastropheConfig()
        # Crash threshold should be a fraction
        assert 0 < cfg.crash_threshold < 1
        # Mode collapse threshold near 1
        assert cfg.mode_collapse_threshold > 0.5
        # Loss spike ratio should be > 1
        assert cfg.loss_spike_ratio > 1


class TestMetaStrategy:
    """Tests for MetaStrategy."""

    def test_defaults(self):
        s = MetaStrategy()
        assert s.surgery_type in ("micro", "macro", "architecture")
        assert s.lora_rank > 0
        assert s.surgery_budget > 0
        assert isinstance(s.target_layers, list)
        assert isinstance(s.focus_domains, list)
        assert s.emergency_stop is False

    def test_phase0_strategy(self):
        s = PHASE0_STRATEGY
        assert s.surgery_type == "micro"
        assert s.lora_rank <= 8


class TestCeilingBreakerConfig:
    """Tests for CeilingBreakerConfig."""

    def test_defaults(self):
        cfg = CeilingBreakerConfig()
        assert cfg.max_level >= 1
        assert isinstance(cfg.level_1_enabled, bool)


class TestGPUConfig:
    """Tests for GPU config."""

    def test_allocation(self):
        alloc = GPUAllocation(
            device_id=0,
            role="inference",
            memory_budget_gb=40.0,
        )
        assert alloc.device_id == 0
        assert alloc.role == "inference"

    def test_gpu_config(self):
        cfg = GPUConfig(
            allocations=[
                GPUAllocation(0, "inference", 40.0),
                GPUAllocation(1, "virtual_train", 60.0),
            ],
        )
        assert len(cfg.allocations) == 2


class TestRunPodConfig:
    """Tests for RunPodConfig."""

    def test_defaults(self):
        cfg = RunPodConfig()
        assert isinstance(cfg.endpoint_id, str)


class TestPhaseConfig:
    """Tests for PhaseConfig."""

    def test_phase_0(self):
        cfg = PHASE_0_CONFIG
        assert cfg.phase_id == PhaseId.PHASE_0
        assert "0.5" in cfg.model_name.lower() or "qwen" in cfg.model_name.lower()
        assert cfg.max_epochs > 0
        assert cfg.iterations_per_epoch > 0

    def test_phase_1(self):
        cfg = PHASE_1_CONFIG
        assert cfg.phase_id == PhaseId.PHASE_1
        assert cfg.max_epochs > 0

    def test_phase_0_is_smaller(self):
        # Phase 0 should be smaller/faster than phase 1
        assert PHASE_0_CONFIG.max_epochs <= PHASE_1_CONFIG.max_epochs


class TestSaveLoadConfig:
    """Tests for config serialization."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.json"
            strategy = MetaStrategy(
                surgery_type="macro",
                lora_rank=16,
                surgery_budget=0.02,
                difficulty_level=3,
            )
            save_config(strategy, path)
            loaded = load_config(path)
            assert loaded is not None

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "config.json"
            save_config(MetaStrategy(), path)
            assert path.exists()

    def test_load_nonexistent(self):
        result = load_config(Path("/nonexistent/config.json"))
        assert result is None
