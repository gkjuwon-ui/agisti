"""
Tests for agisti.orchestrator — gpu, runpod, orchestrator.
"""

from __future__ import annotations

import json
import pytest

from agisti.types import (
    IterationState,
    PhaseId,
    IterationResult,
)
from agisti.config import (
    GPUConfig,
    GPUAllocation,
    RunPodConfig,
    PhaseConfig,
    PHASE_0_CONFIG,
    PHASE_1_CONFIG,
)
from agisti.orchestrator.gpu import (
    GPUOrchestrator,
    GPURoleAssignment,
    RunPodCostEstimator,
    GPURole,
    GPUInfo,
    CostEstimate,
)
from agisti.orchestrator.runpod import (
    RunPodClient,
    RunPodOrchestrator,
    RunPodJob,
    RunPodJobStatus,
    IterationPayload,
    RunPodError,
)
from agisti.orchestrator.orchestrator import (
    AGISTIOrchestrator,
    PhaseState,
    OrchestratorStats,
)


# ─── GPUInfo Tests ────────────────────────────────

class TestGPUInfo:
    """Tests for GPU information dataclass."""

    def test_creation(self):
        info = GPUInfo(
            index=0,
            name="A100-SXM4-80GB",
            total_memory_gb=80.0,
            free_memory_gb=75.0,
        )
        assert info.index == 0
        assert info.name == "A100-SXM4-80GB"
        assert info.total_memory_gb == 80.0

    def test_used_memory(self):
        info = GPUInfo(
            index=0,
            name="A100",
            total_memory_gb=80.0,
            free_memory_gb=60.0,
        )
        assert info.used_memory_gb == pytest.approx(20.0)

    def test_utilization(self):
        info = GPUInfo(
            index=0,
            name="A100",
            total_memory_gb=80.0,
            free_memory_gb=40.0,
        )
        assert info.utilization == pytest.approx(0.5)


# ─── GPURoleAssignment Tests ─────────────────────

class TestGPURoleAssignment:
    """Tests for GPU role assignment."""

    def test_creation(self):
        assignment = GPURoleAssignment(
            inference_devices=[0],
            virtual_training_devices=[1],
            benchmark_devices=[0],
        )
        assert assignment.inference_devices == [0]
        assert assignment.virtual_training_devices == [1]

    def test_all_devices(self):
        assignment = GPURoleAssignment(
            inference_devices=[0, 1],
            virtual_training_devices=[2, 3],
            benchmark_devices=[0],
        )
        all_devs = assignment.all_devices
        assert 0 in all_devs
        assert 2 in all_devs
        assert 3 in all_devs


# ─── GPUOrchestrator Tests ───────────────────────

class TestGPUOrchestrator:
    """Tests for GPU orchestration."""

    def test_creation(self):
        config = GPUConfig(
            allocations=[
                GPUAllocation(
                    gpu_index=0,
                    role="inference",
                    memory_fraction=0.5,
                ),
            ],
        )
        orch = GPUOrchestrator(config)
        assert orch is not None

    def test_allocate_single_gpu(self):
        config = GPUConfig(
            allocations=[
                GPUAllocation(gpu_index=0, role="inference", memory_fraction=1.0),
            ],
        )
        orch = GPUOrchestrator(config)
        assignment = orch.allocate()
        assert isinstance(assignment, GPURoleAssignment)

    def test_memory_budget(self):
        config = GPUConfig(
            allocations=[
                GPUAllocation(gpu_index=0, role="inference", memory_fraction=0.5),
                GPUAllocation(gpu_index=0, role="virtual_training", memory_fraction=0.5),
            ],
        )
        orch = GPUOrchestrator(config)
        total_fraction = sum(a.memory_fraction for a in config.allocations)
        assert total_fraction <= 1.0


# ─── RunPodCostEstimator Tests ───────────────────

class TestRunPodCostEstimator:
    """Tests for RunPod cost estimation."""

    def test_creation(self):
        estimator = RunPodCostEstimator()
        assert estimator is not None

    def test_estimate_single_phase(self):
        estimator = RunPodCostEstimator()
        cost = estimator.estimate(
            gpu_type="A100",
            num_gpus=1,
            hours=10.0,
        )
        assert isinstance(cost, CostEstimate)
        assert cost.total_cost > 0.0
        assert cost.per_hour > 0.0

    def test_estimate_scales_with_gpus(self):
        estimator = RunPodCostEstimator()
        cost_1 = estimator.estimate("A100", num_gpus=1, hours=10.0)
        cost_4 = estimator.estimate("A100", num_gpus=4, hours=10.0)
        assert cost_4.total_cost > cost_1.total_cost

    def test_estimate_scales_with_hours(self):
        estimator = RunPodCostEstimator()
        cost_10 = estimator.estimate("A100", num_gpus=1, hours=10.0)
        cost_20 = estimator.estimate("A100", num_gpus=1, hours=20.0)
        assert cost_20.total_cost > cost_10.total_cost

    def test_format_cost_table(self):
        estimator = RunPodCostEstimator()
        table = estimator.format_cost_table()
        assert isinstance(table, str)
        assert "A100" in table or len(table) > 0

    def test_estimate_all_phases(self):
        estimator = RunPodCostEstimator()
        all_costs = estimator.estimate_all_phases()
        assert isinstance(all_costs, dict)
        assert len(all_costs) > 0


# ─── RunPodJob Tests ──────────────────────────────

class TestRunPodJob:
    """Tests for RunPod job tracking."""

    def test_creation(self):
        job = RunPodJob(
            job_id="job_123",
            status=RunPodJobStatus.QUEUED,
            endpoint_id="ep_456",
        )
        assert job.job_id == "job_123"
        assert job.status == RunPodJobStatus.QUEUED

    def test_is_terminal(self):
        completed = RunPodJob(
            job_id="j1",
            status=RunPodJobStatus.COMPLETED,
            endpoint_id="ep",
        )
        assert completed.is_terminal is True

        running = RunPodJob(
            job_id="j2",
            status=RunPodJobStatus.IN_PROGRESS,
            endpoint_id="ep",
        )
        assert running.is_terminal is False

        failed = RunPodJob(
            job_id="j3",
            status=RunPodJobStatus.FAILED,
            endpoint_id="ep",
        )
        assert failed.is_terminal is True

    def test_status_values(self):
        for status in RunPodJobStatus:
            assert hasattr(status, "value")


# ─── IterationPayload Tests ─────────────────────

class TestIterationPayload:
    """Tests for RunPod iteration payload serialization."""

    def test_creation(self):
        payload = IterationPayload(
            iteration=0,
            phase="phase_0",
            config={"key": "value"},
        )
        assert payload.iteration == 0

    def test_to_dict(self):
        payload = IterationPayload(
            iteration=5,
            phase="phase_1",
            config={"lr": 0.001},
        )
        d = payload.to_dict()
        assert d["iteration"] == 5
        assert d["phase"] == "phase_1"

    def test_serializable(self):
        payload = IterationPayload(
            iteration=1,
            phase="phase_0",
            config={"epochs": 10},
        )
        json_str = json.dumps(payload.to_dict())
        assert isinstance(json_str, str)
        loaded = json.loads(json_str)
        assert loaded["iteration"] == 1


# ─── RunPodClient Tests ──────────────────────────

class TestRunPodClient:
    """Tests for RunPod API client (mocked)."""

    def test_creation(self):
        client = RunPodClient(
            api_key="test_key",
            endpoint_id="ep_test",
        )
        assert client is not None

    def test_endpoint_url(self):
        client = RunPodClient(
            api_key="key",
            endpoint_id="ep_123",
        )
        assert "ep_123" in client.endpoint_url


# ─── PhaseState Tests ────────────────────────────

class TestPhaseState:
    """Tests for PhaseState tracking."""

    def test_creation(self):
        state = PhaseState(
            phase_id=PhaseId.PHASE_0,
            iteration=0,
            best_score=0.0,
        )
        assert state.phase_id == PhaseId.PHASE_0

    def test_update_score(self):
        state = PhaseState(
            phase_id=PhaseId.PHASE_0,
            iteration=0,
            best_score=0.0,
        )
        state.update(score=0.5)
        assert state.best_score == pytest.approx(0.5)
        state.update(score=0.3)
        assert state.best_score == pytest.approx(0.5)  # Best preserved

    def test_increment_iteration(self):
        state = PhaseState(
            phase_id=PhaseId.PHASE_0,
            iteration=0,
            best_score=0.0,
        )
        state.advance()
        assert state.iteration == 1
        state.advance()
        assert state.iteration == 2


# ─── OrchestratorStats Tests ─────────────────────

class TestOrchestratorStats:
    """Tests for orchestrator statistics tracking."""

    def test_creation(self):
        stats = OrchestratorStats()
        assert stats.total_iterations == 0

    def test_record(self):
        stats = OrchestratorStats()
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
        stats.record(result)
        assert stats.total_iterations == 1

    def test_acceptance_rate(self):
        stats = OrchestratorStats()
        for i in range(10):
            result = IterationResult(
                iteration_id=i,
                proposed_delta_norm=0.01,
                virtual_loss_before=2.0,
                virtual_loss_after=1.8,
                refined_delta_norm=0.01,
                quick_bench_scores={"math": 0.5},
                accepted=i % 2 == 0,  # 50% accepted
                wall_time_seconds=1.0,
                gpu_memory_peak_gb=10.0,
            )
            stats.record(result)
        assert stats.acceptance_rate == pytest.approx(0.5)

    def test_average_wall_time(self):
        stats = OrchestratorStats()
        for i in range(5):
            result = IterationResult(
                iteration_id=i,
                proposed_delta_norm=0.01,
                virtual_loss_before=2.0,
                virtual_loss_after=1.8,
                refined_delta_norm=0.01,
                quick_bench_scores={"math": 0.5},
                accepted=True,
                wall_time_seconds=float(i + 1),
                gpu_memory_peak_gb=10.0,
            )
            stats.record(result)
        assert stats.avg_wall_time == pytest.approx(3.0)  # (1+2+3+4+5)/5

    def test_summary(self):
        stats = OrchestratorStats()
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
        stats.record(result)
        summary = stats.summary()
        assert isinstance(summary, dict)
        assert "total_iterations" in summary
