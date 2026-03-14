"""
Configuration system for AGISTI.

All configurable parameters are centralized here so humans
can review and override before each run.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from agisti.types import (
    AnswerType,
    PhaseId,
    SurgeryType,
    VERIFIABLE_TYPES,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Iteration Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class IterationConfig:
    """Configuration for a single AGISTI iteration."""

    iteration_id: int = 0
    surgery_budget: float = 0.01  # β_t: max allowed delta norm
    lora_rank: int = 4
    target_layers: list[str] = field(default_factory=list)
    difficulty_level: float = 0.3  # Start easy, adaptive engine adjusts
    quick_bench_threshold: float = -0.005
    virtual_train_steps: int = 10
    virtual_train_lr: float = 1e-5
    problems_per_iteration: int = 20
    min_verifiable_for_surgery: int = 5
    max_generation_tokens: int = 4096
    trace_activations: bool = True
    checkpoint_every: int = 50  # Save checkpoint every N iterations


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Surgery Configs (per strategy)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class SurgeryConfig:
    """Quantitative specification for a surgery strategy."""

    surgery_type: SurgeryType = SurgeryType.MICRO
    lora_rank: int = 4
    budget_ratio: float = 0.01
    max_layers_per_iter: int = 3
    virtual_train_steps: int = 10
    virtual_train_lr: float = 1e-5
    quick_bench_threshold: float = -0.005
    cooldown_on_fail: int = 1
    prerequisite: str | None = None

    def to_iteration_config(
        self,
        iteration_id: int,
        model_param_norm: float,
        target_layers: list[str],
    ) -> IterationConfig:
        budget = self.budget_ratio * model_param_norm
        return IterationConfig(
            iteration_id=iteration_id,
            surgery_budget=budget,
            lora_rank=self.lora_rank,
            target_layers=target_layers[: self.max_layers_per_iter],
            virtual_train_steps=self.virtual_train_steps,
            virtual_train_lr=self.virtual_train_lr,
            quick_bench_threshold=self.quick_bench_threshold,
        )


MICRO_CONFIG = SurgeryConfig(
    surgery_type=SurgeryType.MICRO,
    lora_rank=4,
    budget_ratio=0.01,
    max_layers_per_iter=3,
    virtual_train_steps=10,
    quick_bench_threshold=-0.005,
    cooldown_on_fail=1,
)

MACRO_CONFIG = SurgeryConfig(
    surgery_type=SurgeryType.MACRO,
    lora_rank=32,
    budget_ratio=0.05,
    max_layers_per_iter=8,
    virtual_train_steps=50,
    quick_bench_threshold=-0.002,
    cooldown_on_fail=5,
    prerequisite="10_consecutive_micro_success",
)


@dataclass
class ArchSurgeryConfig:
    """Architecture surgery config — Phase 3 only."""

    operations: list[str] = field(default_factory=lambda: [
        "add_attention_head",
        "expand_ffn_dim",
        "add_moe_expert",
        "add_layer",
    ])
    quick_bench_threshold: float = 0.0
    full_bench_required: bool = True
    external_bench_required: bool = True
    prerequisite: str = "50_consecutive_success"
    rollback_depth: int = 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Benchmark Configs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class BenchmarkSuiteSpec:
    """Specification for one benchmark in a suite."""

    name: str
    count: int
    weight: float
    source: str = "standard"  # "standard" | "self_generated" | "external_fresh"


@dataclass
class QuickBenchConfig:
    sample_per_domain: int = 10
    timeout_seconds: float = 30.0
    significance_level: float = 0.05

    @property
    def num_problems(self) -> int:
        return self.sample_per_domain * 5  # approximate across domains


@dataclass
class FullBenchConfig:
    phase: PhaseId = PhaseId.PHASE_0
    suites: list[BenchmarkSuiteSpec] = field(default_factory=list)
    timeout_seconds: float = 600.0  # 10 minutes

    def __post_init__(self):
        if not self.suites:
            self.suites = FULL_BENCH_SUITES.get(self.phase, [])


FULL_BENCH_SUITES: dict[PhaseId, list[BenchmarkSuiteSpec]] = {
    PhaseId.PHASE_0: [
        BenchmarkSuiteSpec("gsm8k_mini", 100, 0.3),
        BenchmarkSuiteSpec("hellaswag_mini", 100, 0.2),
        BenchmarkSuiteSpec("self_generated", 100, 0.2, "self_generated"),
        BenchmarkSuiteSpec("external_fresh", 50, 0.3, "external_fresh"),
    ],
    PhaseId.PHASE_1: [
        BenchmarkSuiteSpec("mmlu_full", 14042, 0.25),
        BenchmarkSuiteSpec("humaneval", 164, 0.15),
        BenchmarkSuiteSpec("gsm8k_full", 1319, 0.15),
        BenchmarkSuiteSpec("arc_challenge", 1172, 0.1),
        BenchmarkSuiteSpec("hellaswag", 10042, 0.05),
        BenchmarkSuiteSpec("self_generated", 200, 0.15, "self_generated"),
        BenchmarkSuiteSpec("external_fresh", 150, 0.15, "external_fresh"),
    ],
    PhaseId.PHASE_2: [
        BenchmarkSuiteSpec("mmlu_pro", 12032, 0.15),
        BenchmarkSuiteSpec("humaneval_plus", 164, 0.1),
        BenchmarkSuiteSpec("mbpp_plus", 399, 0.1),
        BenchmarkSuiteSpec("gsm8k_full", 1319, 0.1),
        BenchmarkSuiteSpec("math_500", 500, 0.1),
        BenchmarkSuiteSpec("gpqa_diamond", 198, 0.1),
        BenchmarkSuiteSpec("arc_challenge", 1172, 0.05),
        BenchmarkSuiteSpec("self_generated", 200, 0.15, "self_generated"),
        BenchmarkSuiteSpec("external_fresh", 200, 0.15, "external_fresh"),
    ],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Frozen Zone Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class FrozenDiscoveryConfig:
    """Config for automatic frozen zone discovery."""

    noise_scale: float = 0.1
    severe_drop_threshold: float = 0.30
    moderate_drop_threshold: float = 0.10
    max_frozen_ratio: float = 0.70  # error if >70% is frozen
    always_frozen_patterns: list[str] = field(default_factory=lambda: [
        "model.embed_tokens",
        "model.norm",
        "lm_head",
    ])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MoE Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class MoEConfig:
    """MoE-specific surgery configuration."""

    over_activation_ratio: float = 1.5
    forbidden_patterns: list[str] = field(default_factory=lambda: [
        "*.block_sparse_moe.gate",
        "*.block_sparse_moe.shared_expert",
    ])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Checkpoint Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class CheckpointConfig:
    """Policy for checkpoint management and garbage collection."""

    keep_last_n: int = 10
    keep_every_nth: int = 50
    max_branches: int = 3
    branch_survival_epochs: int = 20
    max_disk_gb: float = 500.0
    base_dir: str = "checkpoint_tree"
    format: str = "safetensors"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convergence Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ConvergenceConfig:
    """Convergence detection parameters."""

    window_size: int = 100
    delta_min: float = 0.001
    budget_increase_factor: float = 2.0
    difficulty_step: int = 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Catastrophe Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class CatastropheConfig:
    """Thresholds for catastrophe detection."""

    sudden_collapse_threshold: float = 0.20
    plateau_window: int = 50
    plateau_slope_threshold: float = 1e-4
    diversity_slope_threshold: float = 0.005
    self_reinforcement_window: int = 20
    acceptance_rate_high: float = 0.95
    acceptance_rate_low: float = 0.10
    crash_threshold: float = 0.15
    regression_threshold: float = 0.03
    divergence_norm_ratio: float = 5.0
    mode_collapse_threshold: float = 0.80
    loss_spike_ratio: float = 3.0
    stall_iterations: int = 30
    stall_threshold: float = 0.005
    max_emergency_count: int = 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MetaStrategy Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class MetaStrategy:
    """Strategy for next iteration(s). Updated by MetaStrategyEngine."""

    surgery_type: SurgeryType = SurgeryType.MICRO
    lora_rank: int = 4
    surgery_budget: float = 0.01
    target_layers: list[str] = field(default_factory=list)
    focus_domains: list[str] = field(default_factory=list)
    difficulty_level: float = 0.3
    exploration_rate: float = 0.15
    external_weight: float = 0.3
    strategy_score: float = 0.0
    iterations_with_this: int = 0
    consecutive_successes: int = 0
    emergency_stop: bool = False
    ceiling_level: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MetaStrategy:
        d = d.copy()
        if "surgery_type" in d and isinstance(d["surgery_type"], str):
            d["surgery_type"] = SurgeryType(d["surgery_type"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Default Phase 0 strategy — prevents "empty strategy" bugs
PHASE0_STRATEGY = MetaStrategy(
    surgery_type=SurgeryType.MICRO,
    lora_rank=4,
    surgery_budget=0.01,
    difficulty_level=0.3,
    exploration_rate=0.1,
    external_weight=0.0,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ceiling Breaker Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ExternalSourceConfig:
    """Config for an external problem source."""

    name: str
    fetcher_class: str  # fully qualified class name
    transform_type: str
    answer_type: AnswerType
    refresh_interval: str = "daily"
    max_age_days: int = 30


@dataclass
class CeilingBreakerConfig:
    """Configuration for the 4-level ceiling breaker system."""

    # Level 1: External Surgery Signal
    external_sources: list[ExternalSourceConfig] = field(default_factory=list)
    external_weight_min: float = 0.1
    external_weight_max: float = 0.8
    external_weight_adaptation_step: float = 0.05

    # Level 2: RAG Surgery
    rag_enabled: bool = False  # activated in Phase 2
    rag_top_k: int = 3
    rag_max_tokens_per_doc: int = 512
    rag_min_flip_count: int = 3
    retriever_sources: list[str] = field(default_factory=lambda: [
        "arxiv", "wikipedia", "textbooks",
    ])

    # Level 3: Inter-Model Cross-Pollination
    cross_model_enabled: bool = False  # activated in Phase 3
    cka_threshold: float = 0.5
    min_informative_problems: int = 3

    # Level 4: Compositional Discovery
    compositional_enabled: bool = False
    min_domain_competency: float = 0.7
    emergence_threshold: float = 1.2

    # Phase-specific signal weights
    signal_weights: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "phase_0": {"self": 1.0, "external": 0.0, "rag": 0.0, "cross": 0.0},
        "phase_1": {"self": 0.5, "external": 0.5, "rag": 0.0, "cross": 0.0},
        "phase_2": {"self": 0.2, "external": 0.2, "rag": 0.6, "cross": 0.0},
        "phase_3": {"self": 0.1, "external": 0.1, "rag": 0.3, "cross": 0.5},
    })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GPU Orchestration Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class GPUAllocation:
    """GPU role assignment."""

    role: str
    gpu_ids: list[int]
    memory_budget_gb: float
    note: str = ""


@dataclass
class GPUConfig:
    """Full GPU orchestration config."""

    allocations: list[GPUAllocation] = field(default_factory=list)
    total_gpus: int = 1

    @classmethod
    def single_gpu(cls) -> GPUConfig:
        return cls(
            total_gpus=1,
            allocations=[
                GPUAllocation("all", [0], 24.0, "Single GPU — everything shares"),
            ],
        )

    @classmethod
    def phase2_8gpu(cls) -> GPUConfig:
        return cls(
            total_gpus=8,
            allocations=[
                GPUAllocation("inference", [0, 1, 2, 3], 70.0, "Pipeline parallel"),
                GPUAllocation("virtual_train", [4, 5], 75.0, "Gradient storage"),
                GPUAllocation("benchmark", [6], 60.0, "Quick/Full bench"),
                GPUAllocation("checkpoint_io", [7], 40.0, "Async safetensors I/O"),
            ],
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RunPod Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class RunPodConfig:
    """RunPod serverless integration config."""

    enabled: bool = False
    api_key: str = ""  # set via env var RUNPOD_API_KEY
    gpu_type: str = "A100_80G"
    cost_per_hour: float = 1.64
    max_budget_usd: float = 100.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class PhaseConfig:
    """Top-level config for a specific phase."""

    phase: PhaseId
    model_name: str
    target_iterations: int
    epoch_size: int
    max_epochs: int = 100
    iterations_per_epoch: int = 50
    surgery: SurgeryConfig = field(default_factory=lambda: copy.deepcopy(MICRO_CONFIG))
    benchmark: FullBenchConfig = field(default_factory=FullBenchConfig)
    quick_bench: QuickBenchConfig = field(default_factory=QuickBenchConfig)
    frozen: FrozenDiscoveryConfig = field(default_factory=FrozenDiscoveryConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    catastrophe: CatastropheConfig = field(default_factory=CatastropheConfig)
    ceiling: CeilingBreakerConfig = field(default_factory=CeilingBreakerConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig.single_gpu)
    runpod: RunPodConfig = field(default_factory=RunPodConfig)
    moe: MoEConfig | None = None
    strategy: MetaStrategy = field(default_factory=lambda: copy.deepcopy(PHASE0_STRATEGY))
    output_dir: str = "agisti_output"
    log_level: str = "INFO"
    seed: int = 42

    # Alias for consistency
    @property
    def phase_id(self) -> PhaseId:
        return self.phase

    def __post_init__(self):
        self.benchmark.phase = self.phase


PHASE_0_CONFIG = PhaseConfig(
    phase=PhaseId.PHASE_0,
    model_name="Qwen/Qwen2.5-0.5B",
    target_iterations=1000,
    epoch_size=10,
)

PHASE_1_CONFIG = PhaseConfig(
    phase=PhaseId.PHASE_1,
    model_name="Qwen/Qwen2.5-7B",
    target_iterations=5000,
    epoch_size=20,
    surgery=SurgeryConfig(
        surgery_type=SurgeryType.MICRO,
        lora_rank=8,
        budget_ratio=0.01,
        max_layers_per_iter=5,
        virtual_train_steps=20,
    ),
    ceiling=CeilingBreakerConfig(
        external_weight_min=0.3,
        external_weight_max=0.8,
    ),
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Config I/O
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def save_config(config: PhaseConfig, path: str | Path) -> None:
    """Serialize config to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(config)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def load_config(path: str | Path) -> PhaseConfig:
    """Deserialize config from JSON."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    # Reconstruct enums
    if "phase" in data:
        data["phase"] = PhaseId(data["phase"])
    if "surgery" in data and "surgery_type" in data["surgery"]:
        data["surgery"]["surgery_type"] = SurgeryType(data["surgery"]["surgery_type"])
    if "strategy" in data:
        data["strategy"] = MetaStrategy.from_dict(data["strategy"])
    else:
        data["strategy"] = copy.deepcopy(PHASE0_STRATEGY)

    # Reconstruct nested dataclasses
    if "surgery" in data and isinstance(data["surgery"], dict):
        data["surgery"] = SurgeryConfig(**{
            k: v for k, v in data["surgery"].items()
            if k in SurgeryConfig.__dataclass_fields__
        })
    if "benchmark" in data and isinstance(data["benchmark"], dict):
        suites = data["benchmark"].pop("suites", [])
        data["benchmark"] = FullBenchConfig(
            phase=PhaseId(data["benchmark"].get("phase", "phase_0")),
            suites=[BenchmarkSuiteSpec(**s) for s in suites],
            timeout_seconds=data["benchmark"].get("timeout_seconds", 600.0),
        )
    if "quick_bench" in data and isinstance(data["quick_bench"], dict):
        data["quick_bench"] = QuickBenchConfig(**data["quick_bench"])
    if "frozen" in data and isinstance(data["frozen"], dict):
        data["frozen"] = FrozenDiscoveryConfig(**{
            k: v for k, v in data["frozen"].items()
            if k in FrozenDiscoveryConfig.__dataclass_fields__
        })
    if "checkpoint" in data and isinstance(data["checkpoint"], dict):
        data["checkpoint"] = CheckpointConfig(**data["checkpoint"])
    if "convergence" in data and isinstance(data["convergence"], dict):
        data["convergence"] = ConvergenceConfig(**data["convergence"])
    if "catastrophe" in data and isinstance(data["catastrophe"], dict):
        data["catastrophe"] = CatastropheConfig(**{
            k: v for k, v in data["catastrophe"].items()
            if k in CatastropheConfig.__dataclass_fields__
        })
    if "ceiling" in data and isinstance(data["ceiling"], dict):
        data["ceiling"] = CeilingBreakerConfig(**{
            k: v for k, v in data["ceiling"].items()
            if k in CeilingBreakerConfig.__dataclass_fields__
        })
    if "gpu" in data and isinstance(data["gpu"], dict):
        allocs = data["gpu"].pop("allocations", [])
        data["gpu"] = GPUConfig(
            total_gpus=data["gpu"].get("total_gpus", 1),
            allocations=[GPUAllocation(**a) for a in allocs],
        )
    if "runpod" in data and isinstance(data["runpod"], dict):
        data["runpod"] = RunPodConfig(**data["runpod"])

    valid_fields = set(PhaseConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return PhaseConfig(**filtered)
