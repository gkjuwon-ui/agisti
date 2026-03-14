"""
GPU Orchestrator — allocates GPU resources for AGISTI pipeline steps.

Manages GPU assignments for inference, virtual training, benchmarking,
and checkpointing. Supports both single-GPU and multi-GPU configurations.

Design: §9 — GPU Orchestration (Compute Orchestration).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

logger = logging.getLogger(__name__)


class GPURole(str, Enum):
    """Role assignment for a GPU."""
    INFERENCE = "inference"
    VIRTUAL_TRAIN = "virtual_train"
    BENCHMARK = "benchmark"
    CHECKPOINT_IO = "checkpoint_io"
    SHARED = "shared"


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    device_id: int
    name: str
    total_memory_gb: float
    free_memory_gb: float
    compute_capability: tuple[int, int] = (0, 0)

    @property
    def used_memory_gb(self) -> float:
        return self.total_memory_gb - self.free_memory_gb

    @property
    def utilization(self) -> float:
        if self.total_memory_gb == 0:
            return 0.0
        return self.used_memory_gb / self.total_memory_gb


@dataclass
class GPURoleAssignment:
    """Maps GPUs to roles for the iteration pipeline."""
    inference_gpus: list[int] = field(default_factory=list)
    virtual_train_gpus: list[int] = field(default_factory=list)
    benchmark_gpus: list[int] = field(default_factory=list)
    checkpoint_gpus: list[int] = field(default_factory=list)
    shared_gpus: list[int] = field(default_factory=list)

    @property
    def all_gpus(self) -> set[int]:
        return (
            set(self.inference_gpus)
            | set(self.virtual_train_gpus)
            | set(self.benchmark_gpus)
            | set(self.checkpoint_gpus)
            | set(self.shared_gpus)
        )

    @property
    def inference_devices(self) -> list[torch.device]:
        return [torch.device(f"cuda:{g}") for g in self.inference_gpus]

    @property
    def virtual_train_device(self) -> torch.device:
        if self.virtual_train_gpus:
            return torch.device(f"cuda:{self.virtual_train_gpus[0]}")
        if self.shared_gpus:
            return torch.device(f"cuda:{self.shared_gpus[0]}")
        return torch.device("cuda:0")

    @property
    def benchmark_device(self) -> torch.device:
        if self.benchmark_gpus:
            return torch.device(f"cuda:{self.benchmark_gpus[0]}")
        if self.shared_gpus:
            return torch.device(f"cuda:{self.shared_gpus[-1]}")
        return torch.device("cuda:0")


class GPUOrchestrator:
    """
    Manages GPU discovery, role assignment, and memory tracking.

    Automatically detects available GPUs and assigns roles
    based on the number of GPUs and phase requirements.

    Allocation strategies:
    - 1 GPU:  All roles on GPU 0 (shared)
    - 2 GPUs: Inference on 0, VT+Bench on 1
    - 4 GPUs: Inference on 0-1, VT on 2, Bench+Ckpt on 3
    - 8 GPUs: Inference on 0-3, VT on 4-5, Bench on 6, Ckpt on 7
    """

    def __init__(self) -> None:
        self._gpus: list[GPUInfo] = []
        self._assignment: GPURoleAssignment | None = None
        self._memory_budget_gb: dict[int, float] = {}

    def discover_gpus(self) -> list[GPUInfo]:
        """
        Detect all available CUDA GPUs.

        Returns:
            List of GPUInfo for each GPU.
        """
        self._gpus = []

        if not torch.cuda.is_available():
            logger.warning("No CUDA GPUs available")
            return self._gpus

        n_gpus = torch.cuda.device_count()
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_mem / (1024 ** 3)

            # Get free memory
            torch.cuda.set_device(i)
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            free_gb = free_mem / (1024 ** 3)

            gpu = GPUInfo(
                device_id=i,
                name=props.name,
                total_memory_gb=total_gb,
                free_memory_gb=free_gb,
                compute_capability=(props.major, props.minor),
            )
            self._gpus.append(gpu)

            logger.info(
                "GPU %d: %s (%.1f GB total, %.1f GB free)",
                i, props.name, total_gb, free_gb,
            )

        return self._gpus

    @property
    def gpu_count(self) -> int:
        return len(self._gpus)

    @property
    def total_memory_gb(self) -> float:
        return sum(g.total_memory_gb for g in self._gpus)

    @property
    def assignment(self) -> GPURoleAssignment | None:
        return self._assignment

    def allocate(
        self,
        n_inference: int | None = None,
        n_virtual_train: int | None = None,
    ) -> GPURoleAssignment:
        """
        Allocate GPUs to pipeline roles.

        If not specified, uses automatic allocation based on GPU count.

        Args:
            n_inference: Number of GPUs for inference.
            n_virtual_train: Number of GPUs for virtual training.

        Returns:
            GPURoleAssignment with all assignments.
        """
        if not self._gpus:
            self.discover_gpus()

        n = len(self._gpus)

        if n == 0:
            # CPU-only mode
            self._assignment = GPURoleAssignment()
            logger.warning("No GPUs found, running in CPU mode")
            return self._assignment

        if n == 1:
            self._assignment = self._allocate_single()
        elif n == 2:
            self._assignment = self._allocate_dual()
        elif n <= 4:
            self._assignment = self._allocate_quad(n_inference)
        else:
            self._assignment = self._allocate_multi(
                n, n_inference, n_virtual_train,
            )

        logger.info("GPU allocation: %s", self._assignment)
        return self._assignment

    def _allocate_single(self) -> GPURoleAssignment:
        """Single GPU: everything shared."""
        return GPURoleAssignment(shared_gpus=[0])

    def _allocate_dual(self) -> GPURoleAssignment:
        """Two GPUs: inference on 0, VT+bench on 1."""
        return GPURoleAssignment(
            inference_gpus=[0],
            virtual_train_gpus=[1],
            benchmark_gpus=[1],
            checkpoint_gpus=[1],
        )

    def _allocate_quad(
        self, n_inference: int | None = None,
    ) -> GPURoleAssignment:
        """3-4 GPUs."""
        n = len(self._gpus)
        n_inf = n_inference or (n - 2)
        n_inf = min(n_inf, n - 1)

        inf_gpus = list(range(n_inf))
        remaining = list(range(n_inf, n))

        return GPURoleAssignment(
            inference_gpus=inf_gpus,
            virtual_train_gpus=[remaining[0]] if remaining else inf_gpus,
            benchmark_gpus=(
                [remaining[-1]] if len(remaining) > 1
                else [remaining[0]] if remaining else inf_gpus[-1:]
            ),
            checkpoint_gpus=[remaining[-1]] if remaining else inf_gpus[-1:],
        )

    def _allocate_multi(
        self,
        n: int,
        n_inference: int | None = None,
        n_virtual_train: int | None = None,
    ) -> GPURoleAssignment:
        """5+ GPUs: full separation per §9.2."""
        n_inf = n_inference or (n // 2)
        n_vt = n_virtual_train or max(1, (n - n_inf) // 2)
        n_bench = 1
        n_ckpt = 1

        # Ensure we don't exceed total
        total_assigned = n_inf + n_vt + n_bench + n_ckpt
        if total_assigned > n:
            n_vt = max(1, n - n_inf - n_bench - n_ckpt)

        inf_gpus = list(range(n_inf))
        vt_start = n_inf
        vt_gpus = list(range(vt_start, vt_start + n_vt))
        bench_gpu = vt_start + n_vt
        ckpt_gpu = bench_gpu + 1

        return GPURoleAssignment(
            inference_gpus=inf_gpus,
            virtual_train_gpus=vt_gpus,
            benchmark_gpus=[min(bench_gpu, n - 1)],
            checkpoint_gpus=[min(ckpt_gpu, n - 1)],
        )

    def set_memory_budget(
        self,
        device_id: int,
        budget_gb: float,
    ) -> None:
        """
        Set memory budget for a GPU.

        Limits the fraction of GPU memory used by PyTorch.
        """
        self._memory_budget_gb[device_id] = budget_gb

        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            fraction = budget_gb / self._gpus[device_id].total_memory_gb
            fraction = min(fraction, 1.0)
            torch.cuda.set_per_process_memory_fraction(
                fraction, device_id,
            )
            logger.info(
                "GPU %d memory budget: %.1f GB (%.1f%%)",
                device_id, budget_gb, fraction * 100,
            )

    def get_memory_snapshot(self) -> list[dict[str, float]]:
        """
        Get current memory usage for all GPUs.

        Returns:
            List of dicts with allocated, reserved, free (GB).
        """
        snapshots = []
        for gpu in self._gpus:
            if torch.cuda.is_available():
                torch.cuda.set_device(gpu.device_id)
                allocated = (
                    torch.cuda.memory_allocated(gpu.device_id)
                    / (1024 ** 3)
                )
                reserved = (
                    torch.cuda.memory_reserved(gpu.device_id)
                    / (1024 ** 3)
                )
                free_mem, _ = torch.cuda.mem_get_info(gpu.device_id)
                free_gb = free_mem / (1024 ** 3)
            else:
                allocated = reserved = free_gb = 0.0

            snapshots.append({
                "device_id": gpu.device_id,
                "name": gpu.name,
                "allocated_gb": round(allocated, 3),
                "reserved_gb": round(reserved, 3),
                "free_gb": round(free_gb, 3),
                "total_gb": round(gpu.total_memory_gb, 3),
            })
        return snapshots

    def clear_cache(self, device_id: int | None = None) -> None:
        """Clear CUDA memory cache."""
        if not torch.cuda.is_available():
            return
        if device_id is not None:
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()
        else:
            for gpu in self._gpus:
                torch.cuda.set_device(gpu.device_id)
                torch.cuda.empty_cache()


# ─── Cost Estimator ──────────────────────────────────

@dataclass
class CostEstimate:
    """Estimated cost for a training run."""
    phase: str
    gpu_type: str
    n_gpus: int
    iterations: int
    estimated_hours: float
    cost_per_hour: float
    total_cost_usd: float
    iter_per_hour: float


class RunPodCostEstimator:
    """
    Estimates RunPod Serverless costs for AGISTI training.

    Per §9.4 — RunPod pricing and estimates.
    """

    # RunPod pricing (approximate, 2024-2025)
    COST_PER_HOUR: dict[str, float] = {
        "RTX_4090":      0.44,
        "A100_40G":      1.04,
        "A100_80G":      1.64,
        "A100_80G_2x":   3.28,
        "A100_80G_4x":   6.56,
        "H100_80G":      3.89,
        "H100_80G_4x":  15.56,
        "H100_80G_8x":  31.12,
    }

    # Phase configs (iterations per hour and GPU type)
    PHASE_CONFIGS: dict[str, dict[str, Any]] = {
        "phase_0": {
            "gpu": "A100_80G",
            "n_gpus": 1,
            "iter_per_hour": 120,
        },
        "phase_1": {
            "gpu": "A100_80G_4x",
            "n_gpus": 4,
            "iter_per_hour": 30,
        },
        "phase_2": {
            "gpu": "H100_80G_8x",
            "n_gpus": 8,
            "iter_per_hour": 8,
        },
        "phase_3": {
            "gpu": "H100_80G_8x",
            "n_gpus": 16,
            "iter_per_hour": 2,
        },
    }

    def estimate(
        self,
        phase: str,
        iterations: int,
        gpu_override: str | None = None,
    ) -> CostEstimate:
        """
        Estimate cost for a phase.

        Args:
            phase: Phase identifier (e.g., "phase_0").
            iterations: Number of iterations to run.
            gpu_override: Override GPU type.

        Returns:
            CostEstimate with breakdown.
        """
        config = self.PHASE_CONFIGS.get(phase)
        if config is None:
            raise ValueError(f"Unknown phase: {phase}")

        gpu_type = gpu_override or config["gpu"]
        cost_hr = self.COST_PER_HOUR.get(gpu_type, 0.0)
        iph = config["iter_per_hour"]

        hours = iterations / iph
        total = hours * cost_hr

        return CostEstimate(
            phase=phase,
            gpu_type=gpu_type,
            n_gpus=config["n_gpus"],
            iterations=iterations,
            estimated_hours=round(hours, 2),
            cost_per_hour=cost_hr,
            total_cost_usd=round(total, 2),
            iter_per_hour=iph,
        )

    def estimate_all_phases(
        self,
        iterations_per_phase: dict[str, int] | None = None,
    ) -> list[CostEstimate]:
        """
        Estimate costs for all phases.

        Args:
            iterations_per_phase: Dict mapping phase → iteration count.
                Defaults to typical values.

        Returns:
            List of CostEstimate per phase.
        """
        defaults = {
            "phase_0": 1000,
            "phase_1": 5000,
            "phase_2": 10000,
            "phase_3": 20000,
        }
        iters = iterations_per_phase or defaults

        estimates = []
        for phase in sorted(self.PHASE_CONFIGS.keys()):
            if phase in iters:
                estimates.append(self.estimate(phase, iters[phase]))
        return estimates

    def format_cost_table(
        self,
        estimates: list[CostEstimate] | None = None,
    ) -> str:
        """Pretty-print cost table."""
        if estimates is None:
            estimates = self.estimate_all_phases()

        lines = [
            f"{'Phase':<10} {'GPU':<15} {'Iters':<8} "
            f"{'Hours':<8} {'$/hr':<8} {'Total $':<10}",
            "-" * 65,
        ]

        total_cost = 0.0
        for e in estimates:
            lines.append(
                f"{e.phase:<10} {e.gpu_type:<15} {e.iterations:<8} "
                f"{e.estimated_hours:<8.1f} "
                f"${e.cost_per_hour:<7.2f} "
                f"${e.total_cost_usd:<9.2f}"
            )
            total_cost += e.total_cost_usd

        lines.append("-" * 65)
        lines.append(f"{'TOTAL':<43} ${total_cost:,.2f}")

        return "\n".join(lines)
