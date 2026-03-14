"""
Tests for agisti.types — type definitions, enums, dataclasses.
"""

from __future__ import annotations

import hashlib
import math

import pytest

from agisti.types import (
    AlertLevel,
    AlertType,
    AnswerType,
    Alert,
    BranchInfo,
    CheckpointInfo,
    ConvergenceAction,
    Document,
    ErrorReport,
    ExternalSignal,
    FailedProblem,
    FreezeLevel,
    IterationResult,
    IterationState,
    LoRADelta,
    LoRALayerDelta,
    PhaseId,
    Problem,
    QuickBenchResult,
    FullBenchResult,
    RAGSignal,
    CrossSignal,
    BlendedSignal,
    SelfSignal,
    Solution,
    SurgeryType,
    VirtualTrainResult,
    Probe,
    WeaknessReport,
    VERIFIABLE_TYPES,
    FrozenZoneViolation,
    SurgeryBudgetExceeded,
    InsufficientVerifiableProblems,
    CatastropheDetected,
    ConvergenceReached,
    EmergencyRollbackRequired,
)


class TestAnswerType:
    """Tests for AnswerType enum."""

    def test_all_types_defined(self):
        expected = {
            "exact", "regex", "code_exec",
            "proof_check", "numeric",
        }
        actual = {t.value for t in AnswerType}
        assert expected == actual

    def test_verifiable_types(self):
        assert AnswerType.NUMERIC_RANGE in VERIFIABLE_TYPES
        assert AnswerType.CODE_EXEC in VERIFIABLE_TYPES
        assert AnswerType.PROOF_CHECK in VERIFIABLE_TYPES
        assert AnswerType.EXACT_MATCH not in VERIFIABLE_TYPES

    def test_string_conversion(self):
        assert AnswerType("numeric") == AnswerType.NUMERIC_RANGE
        assert AnswerType.CODE_EXEC.value == "code_exec"


class TestSurgeryType:
    """Tests for SurgeryType enum."""

    def test_values(self):
        assert SurgeryType.MICRO.value == "micro"
        assert SurgeryType.MACRO.value == "macro"
        assert SurgeryType.ARCHITECTURE.value == "architecture"

    def test_from_string(self):
        assert SurgeryType("micro") == SurgeryType.MICRO


class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_all_levels(self):
        levels = {l.value for l in AlertLevel}
        assert "info" in levels
        assert "warning" in levels
        assert "critical" in levels
        assert "emergency" in levels

    def test_ordering_semantics(self):
        # Verify that levels have increasing severity in value
        level_order = [
            AlertLevel.INFO,
            AlertLevel.WARNING,
            AlertLevel.CRITICAL,
            AlertLevel.EMERGENCY,
        ]
        for i in range(len(level_order) - 1):
            assert level_order[i].value != level_order[i + 1].value


class TestAlertType:
    """Tests for AlertType enum."""

    def test_core_types_exist(self):
        # Types from original design
        assert AlertType.SUDDEN_COLLAPSE
        assert AlertType.PLATEAU
        assert AlertType.DIVERSITY_DIVERGENCE
        assert AlertType.SELF_REINFORCEMENT_CONTAMINATION
        assert AlertType.TOO_EASY
        assert AlertType.TOO_HARD
        # Types added for catastrophe detector
        assert AlertType.REGRESSION
        assert AlertType.DIVERGENCE
        assert AlertType.OSCILLATION
        assert AlertType.FROZEN_VIOLATION
        assert AlertType.MODE_COLLAPSE
        assert AlertType.LOSS_SPIKE
        assert AlertType.CONVERGENCE_STALL


class TestConvergenceAction:
    """Tests for ConvergenceAction enum."""

    def test_all_actions(self):
        actions = {a.value for a in ConvergenceAction}
        expected = {
            "continue", "stop", "rollback",
            "increase_budget", "increase_difficulty",
            "macro_surgery", "next_phase",
        }
        assert expected == actions


class TestIterationState:
    """Tests for IterationState enum."""

    def test_key_states_exist(self):
        assert IterationState.IDLE
        assert IterationState.PROBE
        assert IterationState.GENERATE
        assert IterationState.SOLVE
        assert IterationState.EVALUATE
        assert IterationState.PROPOSE
        assert IterationState.VIRTUAL_TRAIN
        assert IterationState.APPLY_DELTA
        assert IterationState.QUICK_BENCH
        assert IterationState.FEEDBACK
        assert IterationState.ROLLBACK
        assert IterationState.SNAPSHOT
        assert IterationState.COMPLETE

    def test_total_count(self):
        assert len(IterationState) >= 14


class TestPhaseId:
    """Tests for PhaseId enum."""

    def test_phases(self):
        assert PhaseId.PHASE_0.value == "phase_0"
        assert PhaseId.PHASE_1.value == "phase_1"
        assert PhaseId.PHASE_2.value == "phase_2"
        assert PhaseId.PHASE_3.value == "phase_3"


class TestFreezeLevel:
    """Tests for FreezeLevel enum."""

    def test_levels(self):
        assert FreezeLevel.FULL.value == "full"
        assert FreezeLevel.PARTIAL.value == "partial"
        assert FreezeLevel.NONE.value == "none"


# ─── Dataclass Tests ─────────────────────────────────

class TestProblem:
    """Tests for Problem dataclass."""

    def test_creation(self):
        p = Problem(
            question="What is 2+2?",
            answer="4",
            answer_type=AnswerType.NUMERIC,
            domain="arithmetic",
            difficulty=1,
        )
        assert p.question == "What is 2+2?"
        assert p.answer == "4"
        assert p.domain == "arithmetic"
        assert p.difficulty == 1

    def test_verify_numeric(self):
        p = Problem(
            question="2+2?",
            answer="4",
            answer_type=AnswerType.NUMERIC,
            domain="math",
            difficulty=1,
        )
        assert p.verify("4") is True
        assert p.verify("4.0") is True
        assert p.verify("5") is False

    def test_verify_multiple_choice(self):
        p = Problem(
            question="Which?",
            answer="B",
            answer_type=AnswerType.MULTIPLE_CHOICE,
            domain="logic",
            difficulty=1,
        )
        assert p.verify("B") is True
        assert p.verify("b") is True
        assert p.verify("A") is False

    def test_verify_open_ended_always_false(self):
        p = Problem(
            question="Explain...",
            answer="Anything",
            answer_type=AnswerType.OPEN_ENDED,
            domain="essay",
            difficulty=1,
        )
        # Open-ended cannot be mechanically verified
        assert p.verify("Anything") is False

    def test_metadata(self):
        p = Problem(
            question="q",
            answer="a",
            answer_type=AnswerType.NUMERIC,
            domain="d",
            difficulty=1,
            metadata={"source": "test"},
        )
        assert p.metadata["source"] == "test"


class TestSolution:
    """Tests for Solution dataclass."""

    def test_creation(self):
        s = Solution(
            answer="42",
            reasoning="Because 6*7=42",
            confidence=0.95,
            tokens_used=100,
        )
        assert s.answer == "42"
        assert s.confidence == 0.95
        assert s.tokens_used == 100


class TestLoRADelta:
    """Tests for LoRALayerDelta and LoRADelta."""

    def test_layer_delta_creation(self):
        import torch
        A = torch.randn(4, 128)
        B = torch.randn(256, 4)
        delta = LoRALayerDelta(A=A, B=B, rank=4)
        assert delta.rank == 4
        assert delta.A.shape == (4, 128)
        assert delta.B.shape == (256, 4)

    def test_layer_delta_to_full(self):
        import torch
        A = torch.randn(4, 128)
        B = torch.randn(256, 4)
        delta = LoRALayerDelta(A=A, B=B, rank=4)
        full = delta.to_full((256, 128))
        assert full.shape == (256, 128)

    def test_lora_delta_norm(self):
        import torch
        A1 = torch.randn(4, 128)
        B1 = torch.randn(256, 4)
        d1 = LoRALayerDelta(A=A1, B=B1, rank=4)

        delta = LoRADelta(layers={"layer0": d1})
        assert delta.norm() > 0

    def test_lora_delta_layer_names(self):
        import torch
        d1 = LoRALayerDelta(
            A=torch.randn(4, 64),
            B=torch.randn(128, 4),
            rank=4,
        )
        d2 = LoRALayerDelta(
            A=torch.randn(4, 64),
            B=torch.randn(128, 4),
            rank=4,
        )
        delta = LoRADelta(layers={"layer.0": d1, "layer.1": d2})
        assert set(delta.layer_names) == {"layer.0", "layer.1"}

    def test_lora_delta_get_layer(self):
        import torch
        d1 = LoRALayerDelta(
            A=torch.randn(4, 64),
            B=torch.randn(128, 4),
            rank=4,
        )
        delta = LoRADelta(layers={"layer.0": d1})
        assert delta.get_layer("layer.0") is d1
        assert delta.get_layer("nonexistent") is None

    def test_lora_delta_len(self):
        import torch
        d1 = LoRALayerDelta(
            A=torch.randn(4, 64),
            B=torch.randn(128, 4),
            rank=4,
        )
        delta = LoRADelta(layers={"x": d1})
        assert len(delta) == 1


class TestIterationResult:
    """Tests for IterationResult dataclass."""

    def test_creation(self):
        r = IterationResult(
            iteration_id=1,
            proposed_delta_norm=0.05,
            virtual_loss_before=2.0,
            virtual_loss_after=1.8,
            refined_delta_norm=0.04,
            quick_bench_scores={"math": 0.8},
            accepted=True,
            wall_time_seconds=30.0,
            gpu_memory_peak_gb=12.0,
            target_layers=["layer.0"],
            surgery_type="micro",
            epoch=0,
        )
        assert r.iteration_id == 1
        assert r.accepted is True
        assert r.epoch == 0

    def test_defaults(self):
        r = IterationResult(
            iteration_id=0,
            proposed_delta_norm=0.0,
            virtual_loss_before=0.0,
            virtual_loss_after=0.0,
            refined_delta_norm=0.0,
            quick_bench_scores={},
            accepted=False,
            wall_time_seconds=0.0,
            gpu_memory_peak_gb=0.0,
        )
        assert r.rejection_reason is None
        assert r.target_layers == []
        assert r.surgery_type == "micro"
        assert r.epoch == 0


class TestVirtualTrainResult:
    """Tests for VirtualTrainResult."""

    def test_loss_decreased(self):
        r = VirtualTrainResult(
            loss_before=2.0,
            loss_after=1.5,
            gradient_norm=0.1,
            steps=10,
        )
        assert r.loss_decreased is True

    def test_loss_increased(self):
        r = VirtualTrainResult(
            loss_before=1.5,
            loss_after=2.0,
            gradient_norm=0.1,
            steps=10,
        )
        assert r.loss_decreased is False


class TestQuickBenchResult:
    """Tests for QuickBenchResult."""

    def test_creation(self):
        r = QuickBenchResult(
            scores={"math": 0.8, "logic": 0.7},
            passed=True,
            p_value=0.01,
            accuracy=0.75,
            domain_breakdown={"math": 0.8, "logic": 0.7},
        )
        assert r.passed is True
        assert r.accuracy == 0.75
        assert len(r.domain_breakdown) == 2


class TestSignalTypes:
    """Tests for signal dataclasses."""

    def test_self_signal(self):
        import torch
        s = SelfSignal(
            layer_name="layer.0",
            contrast_direction=torch.randn(128),
            magnitude=0.5,
            confidence=0.8,
        )
        assert s.layer_name == "layer.0"
        assert s.magnitude == 0.5

    def test_external_signal(self):
        s = ExternalSignal(
            source="mmlu",
            domain="math",
            flip_rate=0.3,
            layer_contrasts={"layer.0": 0.5},
        )
        assert s.source == "mmlu"
        assert s.flip_rate == 0.3

    def test_rag_signal(self):
        s = RAGSignal(
            query="test",
            retrieved_docs=["doc1"],
            flip_rate=0.2,
            layer_contrasts={"layer.0": 0.1},
        )
        assert s.flip_rate == 0.2

    def test_cross_signal(self):
        s = CrossSignal(
            source_model="model_a",
            target_model="model_b",
            layer_mapping={"layer.0": "layer.1"},
            cka_similarities={"layer.0": 0.9},
        )
        assert s.source_model == "model_a"

    def test_blended_signal(self):
        import torch
        s = BlendedSignal(
            layer_name="layer.0",
            direction=torch.randn(128),
            weights={"self": 0.6, "external": 0.4},
            confidence=0.7,
        )
        assert s.confidence == 0.7
        assert sum(s.weights.values()) == pytest.approx(1.0)


class TestAlert:
    """Tests for Alert dataclass."""

    def test_creation(self):
        a = Alert(
            level=AlertLevel.WARNING,
            type=AlertType.PLATEAU,
            message="Training plateaued",
        )
        assert a.level == AlertLevel.WARNING
        assert a.type == AlertType.PLATEAU
        assert a.action == ""

    def test_with_details(self):
        a = Alert(
            level=AlertLevel.CRITICAL,
            type=AlertType.REGRESSION,
            message="Score regressed",
            iteration=42,
            details={"drop": 0.05},
        )
        assert a.iteration == 42
        assert a.details["drop"] == 0.05

    def test_timestamp_set(self):
        a = Alert(
            level=AlertLevel.INFO,
            type=AlertType.TOO_EASY,
            message="test",
        )
        assert a.timestamp > 0


class TestCheckpointInfo:
    """Tests for CheckpointInfo."""

    def test_creation(self):
        info = CheckpointInfo(
            iteration=100,
            score=0.85,
            path="/checkpoints/ckpt_100.pt",
            timestamp=1000.0,
        )
        assert info.iteration == 100
        assert info.score == 0.85


class TestBranchInfo:
    """Tests for BranchInfo."""

    def test_creation(self):
        info = BranchInfo(
            branch_id="branch_001",
            name="exploration_1",
            parent_iteration=50,
            created_at=1000.0,
        )
        assert info.branch_id == "branch_001"
        assert info.parent_iteration == 50


class TestDocument:
    """Tests for Document."""

    def test_creation(self):
        d = Document(
            content="Some text",
            source="arxiv",
            metadata={"id": "2301.12345"},
        )
        assert d.content == "Some text"
        assert d.source == "arxiv"


class TestProbe:
    """Tests for Probe."""

    def test_creation(self):
        p = Probe(
            domain="math",
            problems=[],
            accuracy=0.8,
            timestamp=1000.0,
        )
        assert p.domain == "math"
        assert p.accuracy == 0.8


class TestWeaknessReport:
    """Tests for WeaknessReport."""

    def test_creation(self):
        wr = WeaknessReport(
            weak_domains=["spatial"],
            domain_scores={"spatial": 0.3, "math": 0.9},
            failed_problems=[],
        )
        assert "spatial" in wr.weak_domains
        assert wr.domain_scores["spatial"] == 0.3


# ─── Exception Tests ─────────────────────────────────

class TestExceptions:
    """Tests for custom exceptions."""

    def test_frozen_zone_violation(self):
        exc = FrozenZoneViolation("layer.0 is frozen")
        assert "layer.0" in str(exc)

    def test_surgery_budget_exceeded(self):
        exc = SurgeryBudgetExceeded("0.02 > 0.01")
        assert "0.02" in str(exc)

    def test_insufficient_verifiable(self):
        exc = InsufficientVerifiableProblems("need 10, got 3")
        assert "10" in str(exc)

    def test_catastrophe_detected(self):
        exc = CatastropheDetected("severe regression")
        assert "regression" in str(exc)

    def test_convergence_reached(self):
        exc = ConvergenceReached("no improvement for 50 iters")
        assert "50" in str(exc)

    def test_emergency_rollback(self):
        exc = EmergencyRollbackRequired("loss diverged")
        assert "diverged" in str(exc)
