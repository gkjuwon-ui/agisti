"""
Core type definitions for AGISTI.

Every dataclass, enum, and type alias used across the system
is defined here to avoid circular imports and ensure consistency.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import torch
from torch import Tensor


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enums
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class AnswerType(str, Enum):
    """How to verify correctness of a problem's answer."""

    EXACT_MATCH = "exact"
    REGEX = "regex"
    CODE_EXEC = "code_exec"
    PROOF_CHECK = "proof_check"
    NUMERIC_RANGE = "numeric"

    # Aliases for convenience
    @classmethod
    def _missing_(cls, value: object) -> AnswerType | None:
        aliases = {
            "code": cls.CODE_EXEC,
            "proof": cls.PROOF_CHECK,
            "multiple_choice": cls.EXACT_MATCH,
            "mcq": cls.EXACT_MATCH,
            "open_ended": cls.REGEX,
            "exact_match": cls.EXACT_MATCH,
            "numeric_range": cls.NUMERIC_RANGE,
        }
        if isinstance(value, str):
            return aliases.get(value.lower())
        return None


# Class-level aliases for backward compatibility
AnswerType.NUMERIC = AnswerType.NUMERIC_RANGE  # type: ignore[attr-defined]
AnswerType.CODE = AnswerType.CODE_EXEC  # type: ignore[attr-defined]
AnswerType.PROOF = AnswerType.PROOF_CHECK  # type: ignore[attr-defined]
AnswerType.MULTIPLE_CHOICE = AnswerType.EXACT_MATCH  # type: ignore[attr-defined]
AnswerType.MCQ = AnswerType.EXACT_MATCH  # type: ignore[attr-defined]
AnswerType.OPEN_ENDED = AnswerType.REGEX  # type: ignore[attr-defined]


# Phase 0-1: only these types are trusted for self-surgery signals
VERIFIABLE_TYPES = frozenset({
    AnswerType.CODE_EXEC,
    AnswerType.NUMERIC_RANGE,
    AnswerType.PROOF_CHECK,
})


class SurgeryType(str, Enum):
    MICRO = "micro"
    MACRO = "macro"
    ARCHITECTURE = "arch"


class AlertLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class AlertType(str, Enum):
    SUDDEN_COLLAPSE = "sudden_collapse"
    PLATEAU = "plateau"
    DIVERSITY_DIVERGENCE = "diversity_divergence"
    SELF_REINFORCEMENT_CONTAMINATION = "self_reinforcement_contamination"
    TOO_EASY = "too_easy"
    TOO_HARD = "too_hard"
    REGRESSION = "regression"
    DIVERGENCE = "divergence"
    OSCILLATION = "oscillation"
    FROZEN_VIOLATION = "frozen_violation"
    MODE_COLLAPSE = "mode_collapse"
    LOSS_SPIKE = "loss_spike"
    CONVERGENCE_STALL = "convergence_stall"


class IterationState(str, Enum):
    IDLE = "idle"
    PROBE = "probe"
    GENERATE = "generate"
    SOLVE = "solve"
    EVALUATE = "evaluate"
    PROPOSE = "propose"
    VIRTUAL_TRAIN = "virtual_train"
    APPLY_DELTA = "apply_delta"
    QUICK_BENCH = "quick_bench"
    SNAPSHOT = "snapshot"
    FULL_BENCH = "full_bench"
    CHECKPOINT = "checkpoint"
    DISCARD = "discard"
    ROLLBACK = "rollback"
    FEEDBACK = "feedback"
    CEILING_BREAK = "ceiling_break"
    PHASE_TRANSITION = "phase_transition"
    COMPLETE = "complete"


class FreezeLevel(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


class PhaseId(str, Enum):
    PHASE_0 = "phase_0"
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    PHASE_3 = "phase_3"


class ConvergenceAction(str, Enum):
    CONTINUE = "continue"
    STOP = "stop"
    ROLLBACK = "rollback"
    INCREASE_BUDGET = "increase_budget"
    INCREASE_DIFFICULTY = "increase_difficulty"
    MACRO_SURGERY = "macro_surgery"
    NEXT_PHASE = "next_phase"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Problem & Solution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class Problem:
    """Standard schema for all AGISTI problems."""

    domain: str
    question: str
    answer_type: AnswerType
    id: str = ""
    difficulty: int | float = 1  # 1-5
    answer: str = ""  # expected correct answer
    verification_code: str | None = None
    chain_of_thought: str = ""
    estimated_difficulty: float = 0.5
    tags: tuple[str, ...] = ()
    # External problems may carry a ground truth source
    ground_truth: str | None = None
    tolerance: float = 1e-6  # for NUMERIC_RANGE
    max_answer_tokens: int = 512
    source: str = "self"  # "self" | "external" | source name
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", self.generate_id())
        # Auto-populate answer from metadata if not set
        if not self.answer and self.metadata.get("expected_answer"):
            object.__setattr__(self, "answer", str(self.metadata["expected_answer"]))

    @staticmethod
    def generate_id() -> str:
        return uuid.uuid4().hex[:12]

    def content_hash(self) -> str:
        return hashlib.sha256(self.question.encode("utf-8")).hexdigest()

    def verify(self, answer: str) -> bool:
        """Verify an answer against this problem's expected answer."""
        from agisti.generation.verification import verify_answer
        return verify_answer(
            answer=answer,
            expected=self.answer,
            answer_type=self.answer_type,
            verification_code=self.verification_code,
            tolerance=self.tolerance,
        )


@dataclass
class Solution:
    """Model output for a given problem."""

    problem_id: str
    answer: str
    chain_of_thought: str
    tokens_generated: int
    generation_time_seconds: float
    logprobs: list[float] | None = None


@dataclass
class ErrorReport:
    """Analysis of why a solution was wrong."""

    problem_id: str
    domain: str
    expected: str
    actual: str
    answer_type: AnswerType
    error_category: str  # e.g. "arithmetic", "logic", "hallucination"
    reasoning_trace: str = ""


@dataclass
class FailedProblem:
    """A problem that the model answered incorrectly, with metadata."""

    problem: Problem
    original_solution: Solution
    domain: str
    ground_truth: str
    answer_type: AnswerType
    verification_code: str | None = None
    tolerance: float = 1e-6

    def verify(self, answer: str) -> bool:
        from agisti.generation.verification import verify_answer
        return verify_answer(
            answer=answer,
            expected=self.ground_truth,
            answer_type=self.answer_type,
            verification_code=self.verification_code,
            tolerance=self.tolerance,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Surgery Types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class LoRALayerDelta:
    """Low-rank delta for a single layer: ΔW ≈ A · Bᵀ"""

    A: Tensor  # shape: (d_out, rank)
    B: Tensor  # shape: (rank, d_in)

    @property
    def rank(self) -> int:
        return self.A.shape[1]

    def norm(self) -> float:
        """L2 norm of the full delta: ||A @ B||_F"""
        # Frobenius norm via SVD for efficiency on large layers
        return torch.linalg.norm(self.A @ self.B).item()

    def scale_to(self, target_norm: float) -> None:
        """Scale the delta so that ||A @ B||_F == target_norm."""
        current = self.norm()
        if current < 1e-12:
            return
        factor = (target_norm / current) ** 0.5
        self.A.mul_(factor)
        self.B.mul_(factor)

    def to_full(self) -> Tensor:
        """Materialize the full delta matrix."""
        return self.A @ self.B

    def clone(self) -> LoRALayerDelta:
        return LoRALayerDelta(A=self.A.clone(), B=self.B.clone())

    def detach(self) -> LoRALayerDelta:
        return LoRALayerDelta(
            A=self.A.detach().clone(),
            B=self.B.detach().clone(),
        )


@dataclass
class LoRADelta:
    """Collection of LoRA deltas across multiple layers."""

    rank: int
    layers: dict[str, LoRALayerDelta] = field(default_factory=dict)

    def add_layer(self, name: str, delta: LoRALayerDelta) -> None:
        self.layers[name] = delta

    def norm(self) -> float:
        """Total L2 norm across all layers."""
        if not self.layers:
            return 0.0
        total_sq = sum(d.norm() ** 2 for d in self.layers.values())
        return total_sq ** 0.5

    def scale_to(self, target_norm: float) -> None:
        current = self.norm()
        if current < 1e-12:
            return
        ratio = target_norm / current
        for layer_delta in self.layers.values():
            layer_delta.A.mul_(ratio ** 0.5)
            layer_delta.B.mul_(ratio ** 0.5)

    def clone_with_grad(self) -> LoRADelta:
        """Clone all A, B matrices with requires_grad=True for gradient refinement."""
        new_delta = LoRADelta(rank=self.rank)
        for name, ld in self.layers.items():
            a = ld.A.detach().clone().requires_grad_(True)
            b = ld.B.detach().clone().requires_grad_(True)
            new_delta.add_layer(name, LoRALayerDelta(A=a, B=b))
        return new_delta

    def detach_all(self) -> LoRADelta:
        new_delta = LoRADelta(rank=self.rank)
        for name, ld in self.layers.items():
            new_delta.add_layer(name, ld.detach())
        return new_delta

    @property
    def layer_names(self) -> list[str]:
        """List of layer names in this delta."""
        return list(self.layers.keys())

    def get_layer(self, name: str) -> LoRALayerDelta | None:
        """Get layer delta by name, returning None if not found."""
        return self.layers.get(name)

    def __contains__(self, item: str) -> bool:
        return item in self.layers

    def __getitem__(self, key: str) -> LoRALayerDelta:
        return self.layers[key]

    def __iter__(self):
        return iter(self.layers)

    def items(self):
        return self.layers.items()

    def values(self):
        return self.layers.values()

    def keys(self):
        return self.layers.keys()

    def __len__(self) -> int:
        return len(self.layers)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Iteration Results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class IterationResult:
    """Complete result from one iteration of the AGISTI loop."""

    iteration_id: int
    proposed_delta_norm: float
    virtual_loss_before: float
    virtual_loss_after: float
    refined_delta_norm: float
    quick_bench_scores: dict[str, float]
    accepted: bool
    rejection_reason: str | None
    wall_time_seconds: float
    gpu_memory_peak_gb: float
    target_layers: list[str] = field(default_factory=list)
    external_bench_score: float | None = None
    self_bench_score: float | None = None
    surgery_type: SurgeryType = SurgeryType.MICRO
    epoch: int = 0
    timestamp: float = field(default_factory=time.time)
    # Rich result fields used by feedback/orchestrator
    quick_bench: QuickBenchResult | None = None
    delta: LoRADelta | None = None
    solutions: list[Solution] = field(default_factory=list)
    loss: float | None = None
    frozen_violations: list[str] = field(default_factory=list)

    @property
    def iteration(self) -> int:
        """Alias for iteration_id."""
        return self.iteration_id


@dataclass
class VirtualTrainResult:
    """Result from virtual training (simulation before real surgery)."""

    loss_before: float
    loss_after: float
    loss_decreased: bool
    refined_delta: LoRADelta | None
    refinement_steps: int
    grad_flow_ok: bool = True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Signals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class SelfSignal:
    """Activation contrast from self-generated problem evaluation."""

    contrasts: dict[str, Tensor]
    correct_count: int
    wrong_count: int
    verifiable_count: int


@dataclass
class ExternalSignal:
    """Activation contrast from external problem evaluation."""

    usable: bool
    reason: str | None = None
    contrasts: dict[str, Tensor] | None = None
    correct_count: int = 0
    wrong_count: int = 0
    source: str = ""
    problems_used: int = 0


@dataclass
class RAGSignal:
    """Signal from retrieval-augmented surgery."""

    usable: bool
    reason: str | None = None
    contrasts: dict[str, Tensor] | None = None
    flip_count: int = 0
    total_attempted: int = 0
    flip_rate: float = 0.0


@dataclass
class CrossSignal:
    """Signal from inter-model cross-pollination."""

    usable: bool
    reason: str | None = None
    contrasts: dict[str, Tensor] | None = None
    informative_problems: dict[str, int] = field(default_factory=dict)


@dataclass
class BlendedSignal:
    """Final blended signal from all sources."""

    contrasts: dict[str, Tensor]
    weights: dict[str, float]  # source → weight used
    sources_used: list[str]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Benchmark Results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class QuickBenchResult:
    """Result from a quick benchmark run."""

    scores: dict[str, float]
    elapsed_seconds: float
    passed: bool
    regressions: dict[str, dict[str, Any]]
    raw_answers: dict[str, list[bool]] = field(default_factory=dict)
    accuracy: float = 0.0
    domain_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class FullBenchResult:
    """Result from a full benchmark run."""

    weighted_score: float
    domain_scores: dict[str, float]
    passed: bool
    is_pareto_improvement: bool
    elapsed_seconds: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Probing & Competency
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class Probe:
    """A fixed test question for measuring competency in a domain."""

    id: str
    domain: str
    question: str
    expected_answer: str
    answer_type: AnswerType
    verification_code: str | None = None
    max_answer_tokens: int = 256
    tolerance: float = 1e-6

    @property
    def problem(self) -> Problem:
        """Convert this Probe into a Problem for evaluation."""
        return Problem(
            id=self.id,
            domain=self.domain,
            difficulty=1,
            question=self.question,
            answer=self.expected_answer,
            answer_type=self.answer_type,
            verification_code=self.verification_code,
            tolerance=self.tolerance,
            max_answer_tokens=self.max_answer_tokens,
            source="probe",
        )

    def verify(self, answer: str) -> bool:
        from agisti.generation.verification import verify_answer
        return verify_answer(
            answer=answer,
            expected=self.expected_answer,
            answer_type=self.answer_type,
            verification_code=self.verification_code,
            tolerance=self.tolerance,
        )


@dataclass
class WeaknessReport:
    """Report on a model weakness in a specific domain."""

    domain: str
    weakness_score: float  # higher = weaker
    current_competency: float
    stagnation_score: float = 0.0
    absolute_weakness: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Alerts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class Alert:
    """An alert from the catastrophe detector."""

    level: AlertLevel
    type: AlertType
    message: str
    action: str = ""
    iteration: int = -1
    domain: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Checkpoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class CheckpointInfo:
    """Metadata about a saved checkpoint."""

    epoch: int
    iteration: int
    timestamp: float
    path: str
    weighted_score: float
    domain_scores: dict[str, float]
    frozen_checksums: dict[str, str]
    branch_name: str = "main"
    is_permanent: bool = False
    parent_epoch: int | None = None


@dataclass
class BranchInfo:
    """Metadata about an exploration branch."""

    name: str
    parent_epoch: int
    strategy_description: str
    created_at: float
    best_score: float = 0.0
    epochs_alive: int = 0
    promoted_to_main: bool = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Document (for RAG)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class Document:
    """A retrieved document for RAG surgery."""

    source: str
    doc_id: int
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Exceptions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class FrozenZoneViolation(RuntimeError):
    """Raised when a frozen zone is modified. Stops the system immediately."""
    pass


class SurgeryBudgetExceeded(ValueError):
    """Raised when a proposed delta exceeds the surgery budget."""
    pass


class InsufficientVerifiableProblems(ValueError):
    """Raised when there aren't enough verifiable problems for surgery."""
    pass


class CatastropheDetected(RuntimeError):
    """Raised on critical catastrophe detection (e.g. sudden collapse)."""
    pass


class ConvergenceReached(Exception):
    """Raised when the system has converged and needs escalation."""
    pass


class EmergencyRollbackRequired(RuntimeError):
    """Raised when emergency rollback to last checkpoint is needed."""
    pass
