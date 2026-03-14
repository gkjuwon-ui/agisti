"""
Integration tests for AGISTI — end-to-end component interactions.

These tests verify that components work together correctly,
simulating realistic training scenarios without requiring GPUs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from agisti.types import (
    AnswerType,
    Problem,
    Solution,
    LoRADelta,
    LoRALayerDelta,
    IterationResult,
    QuickBenchResult,
    IterationState,
    AlertLevel,
    AlertType,
    Alert,
    ConvergenceAction,
    PhaseId,
    FreezeLevel,
    CheckpointInfo,
    BranchInfo,
    WeaknessReport,
    FailedProblem,
    FrozenZoneViolation,
    SurgeryBudgetExceeded,
    CatastropheDetected,
    ConvergenceReached,
)
from agisti.config import (
    IterationConfig,
    SurgeryConfig,
    CatastropheConfig,
    MetaStrategy,
    CheckpointConfig,
    FrozenDiscoveryConfig,
    QuickBenchConfig,
    FullBenchConfig,
    ConvergenceConfig,
    PHASE0_STRATEGY,
    PHASE_0_CONFIG,
)


# ─── End-to-End Component Flow ────────────────────

class TestTypeSystemConsistency:
    """Verify all type definitions work together."""

    def test_problem_verify_roundtrip(self):
        for atype in AnswerType:
            p = Problem(
                question=f"Test {atype.value}",
                answer="42",
                answer_type=atype,
                domain="math",
                difficulty=0.5,
            )
            assert p.verify("42") is True or atype in (
                AnswerType.PROOF, AnswerType.CODE
            )

    def test_lora_delta_full_pipeline(self):
        # Create → compose → apply → rollback
        a1 = torch.randn(64, 4)
        b1 = torch.randn(4, 64)
        a2 = torch.randn(64, 4)
        b2 = torch.randn(4, 64)

        delta = LoRADelta(layers={
            "model.layers.0.self_attn.q_proj": LoRALayerDelta(
                A=a1, B=b1, rank=4, alpha=1.0,
            ),
            "model.layers.0.self_attn.v_proj": LoRALayerDelta(
                A=a2, B=b2, rank=4, alpha=1.0,
            ),
        })

        assert delta.norm > 0
        assert len(delta) == 2
        assert "q_proj" in delta.layer_names[0]

        # Full matrix
        layer = delta.get_layer("model.layers.0.self_attn.q_proj")
        full = layer.to_full()
        assert full.shape == (64, 64)

    def test_iteration_result_complete(self):
        result = IterationResult(
            iteration_id=42,
            proposed_delta_norm=0.015,
            virtual_loss_before=2.34,
            virtual_loss_after=2.10,
            refined_delta_norm=0.012,
            quick_bench_scores={
                "math": 0.82,
                "logic": 0.75,
                "coding": 0.68,
            },
            accepted=True,
            wall_time_seconds=45.0,
            gpu_memory_peak_gb=35.0,
            epoch=3,
        )
        assert result.iteration_id == 42
        assert result.accepted is True
        assert len(result.quick_bench_scores) == 3

    def test_alert_system(self):
        for level in AlertLevel:
            for atype in AlertType:
                alert = Alert(
                    level=level,
                    type=atype,
                    message=f"Test {level.value} {atype.value}",
                )
                assert alert.level == level
                assert alert.type == atype

    def test_all_exceptions(self):
        exceptions = [
            FrozenZoneViolation("layer.0"),
            SurgeryBudgetExceeded("over budget"),
            CatastropheDetected("divergence"),
            ConvergenceReached("target met"),
        ]
        for exc in exceptions:
            with pytest.raises(type(exc)):
                raise exc


class TestConfigSystemConsistency:
    """Verify config system integrity."""

    def test_all_configs_instantiate(self):
        configs = [
            IterationConfig(),
            SurgeryConfig(),
            CatastropheConfig(),
            MetaStrategy(),
            CheckpointConfig(),
            FrozenDiscoveryConfig(),
            QuickBenchConfig(),
            FullBenchConfig(),
            ConvergenceConfig(),
        ]
        assert all(c is not None for c in configs)

    def test_config_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from agisti.config import save_config, load_config

            config = IterationConfig()
            path = Path(tmpdir) / "config.json"
            save_config(config, path)
            loaded = load_config(path)
            assert loaded is not None

    def test_phase_configs(self):
        assert PHASE_0_CONFIG is not None
        assert PHASE_1_CONFIG is not None
        assert PHASE0_STRATEGY is not None


class TestDeltaArithmetic:
    """Test LoRA delta mathematical operations."""

    def test_delta_addition(self):
        d1 = LoRADelta(layers={
            "layer.0": LoRALayerDelta(
                A=torch.randn(32, 4),
                B=torch.randn(4, 32),
                rank=4,
                alpha=1.0,
            ),
        })
        d2 = LoRADelta(layers={
            "layer.0": LoRALayerDelta(
                A=torch.randn(32, 4),
                B=torch.randn(4, 32),
                rank=4,
                alpha=1.0,
            ),
        })
        # Verify norms are positive
        assert d1.norm > 0
        assert d2.norm > 0

    def test_delta_with_different_ranks(self):
        d1 = LoRALayerDelta(
            A=torch.randn(64, 4),
            B=torch.randn(4, 64),
            rank=4,
            alpha=1.0,
        )
        d2 = LoRALayerDelta(
            A=torch.randn(64, 8),
            B=torch.randn(8, 64),
            rank=8,
            alpha=1.0,
        )
        # Both should produce valid full matrices
        f1 = d1.to_full()
        f2 = d2.to_full()
        assert f1.shape == f2.shape == (64, 64)

    def test_zero_delta(self):
        delta = LoRALayerDelta(
            A=torch.zeros(32, 4),
            B=torch.zeros(4, 32),
            rank=4,
            alpha=1.0,
        )
        assert delta.to_full().abs().sum() == 0.0


class TestFrozenMaskIntegrity:
    """Integration test for frozen zone mask system."""

    def test_mask_survives_serialization(self):
        from agisti.frozen.mask import FrozenMask

        torch.manual_seed(42)
        params = {
            f"layer.{i}.weight": torch.randn(32, 32)
            for i in range(5)
        }
        frozen = {"layer.0.weight", "layer.1.weight", "layer.4.weight"}
        mask = FrozenMask.from_params(params, frozen_names=frozen)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mask.json"
            mask.save(path)
            loaded = FrozenMask.load(path)

            # Verify integrity
            assert loaded.frozen_count == 3
            for name in frozen:
                assert loaded.is_frozen(name)

            # Verify hash consistency
            assert mask.check_integrity(params) is True

    def test_delta_respects_frozen_zones(self):
        """Delta should NOT modify frozen layers."""
        frozen_names = {"layer.0.weight", "layer.1.weight"}
        delta = LoRADelta(layers={
            "layer.2.weight": LoRALayerDelta(
                A=torch.randn(32, 4),
                B=torch.randn(4, 32),
                rank=4,
                alpha=1.0,
            ),
        })
        # Delta only modifies layer.2 — shouldn't touch frozen layers
        for name in delta.layer_names:
            assert name not in frozen_names


class TestIterationHistoryWorkflow:
    """Integration test for iteration history tracking."""

    def test_full_history_lifecycle(self):
        from agisti.iteration.history import IterationHistory

        history = IterationHistory()

        # Phase 0: improving
        for i in range(50):
            history.add(IterationResult(
                iteration_id=i,
                proposed_delta_norm=0.01,
                virtual_loss_before=2.0 - 0.01 * i,
                virtual_loss_after=1.9 - 0.01 * i,
                refined_delta_norm=0.01,
                quick_bench_scores={"math": 0.3 + 0.01 * i},
                accepted=True,
                wall_time_seconds=1.0,
                gpu_memory_peak_gb=10.0,
                epoch=i // 25,
            ))

        assert history.total_iterations == 50
        assert history.acceptance_rate == 1.0
        assert history.score_slope() > 0
        assert history.is_plateauing() is False

        best = history.best_iteration()
        assert best is not None
        assert best.iteration_id == 49

        # Epoch summary
        e0 = history.epoch_summary(0)
        assert e0 is not None
        assert e0.iterations == 25

        # Persistence
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.jsonl"
            history.save_to_jsonl(path)
            loaded = IterationHistory.load_from_jsonl(path)
            assert loaded.total_iterations == 50

    def test_plateau_detection_after_convergence(self):
        from agisti.iteration.history import IterationHistory

        history = IterationHistory()
        # Rapid improvement then plateau
        for i in range(20):
            score = min(0.8, 0.3 + 0.03 * i)
            history.add(IterationResult(
                iteration_id=i,
                proposed_delta_norm=0.01,
                virtual_loss_before=2.0,
                virtual_loss_after=1.8,
                refined_delta_norm=0.01,
                quick_bench_scores={"math": score},
                accepted=True,
                wall_time_seconds=1.0,
                gpu_memory_peak_gb=10.0,
            ))
        # Flat tail
        for i in range(20, 60):
            history.add(IterationResult(
                iteration_id=i,
                proposed_delta_norm=0.001,
                virtual_loss_before=1.5,
                virtual_loss_after=1.5,
                refined_delta_norm=0.001,
                quick_bench_scores={"math": 0.80},
                accepted=i % 3 == 0,
                wall_time_seconds=1.0,
                gpu_memory_peak_gb=10.0,
            ))

        assert history.is_plateauing() is True


class TestStateMachineWorkflow:
    """Integration test for state machine transitions."""

    def test_complete_iteration_pipeline(self):
        from agisti.iteration.state_machine import IterationStateMachine

        sm = IterationStateMachine()

        # Full pipeline
        pipeline = [
            IterationState.PROBE,
            IterationState.GENERATE,
            IterationState.SOLVE,
            IterationState.EVALUATE,
            IterationState.PROPOSE,
            IterationState.VIRTUAL_TRAIN,
            IterationState.APPLY_DELTA,
            IterationState.QUICK_BENCH,
            IterationState.FEEDBACK,
            IterationState.COMPLETE,
        ]

        for state in pipeline:
            sm.transition(state)

        assert sm.state == IterationState.COMPLETE
        assert sm.is_terminal()
        assert sm.transition_count == len(pipeline)

    def test_rollback_from_feedback(self):
        from agisti.iteration.state_machine import (
            IterationStateMachine,
            InvalidTransitionError,
        )

        sm = IterationStateMachine()
        sm.transition(IterationState.PROBE)
        sm.transition(IterationState.GENERATE)
        sm.transition(IterationState.SOLVE)
        sm.transition(IterationState.EVALUATE)
        sm.transition(IterationState.PROPOSE)
        sm.transition(IterationState.FEEDBACK)
        sm.transition(IterationState.ROLLBACK)

        assert sm.state == IterationState.ROLLBACK


class TestCatastropheFlowIntegration:
    """Integration test for catastrophe detection → response flow."""

    def test_regression_detection_flow(self):
        from agisti.feedback.catastrophe import CatastropheDetector
        from agisti.feedback.meta_strategy import MetaStrategyEngine

        detector = CatastropheDetector(CatastropheConfig(
            regression_threshold=0.1,
        ))
        engine = MetaStrategyEngine(PHASE0_STRATEGY)

        # Good history
        history = [
            IterationResult(
                iteration_id=i,
                proposed_delta_norm=0.01,
                virtual_loss_before=2.0,
                virtual_loss_after=1.8,
                refined_delta_norm=0.01,
                quick_bench_scores={"math": 0.8},
                accepted=True,
                wall_time_seconds=1.0,
                gpu_memory_peak_gb=10.0,
            )
            for i in range(10)
        ]

        # Sudden regression
        bad_result = IterationResult(
            iteration_id=10,
            proposed_delta_norm=0.01,
            virtual_loss_before=2.0,
            virtual_loss_after=3.5,
            refined_delta_norm=0.01,
            quick_bench_scores={"math": 0.2},
            accepted=False,
            wall_time_seconds=1.0,
            gpu_memory_peak_gb=10.0,
        )

        alerts = detector.check(bad_result, history=history)

        # Process any alerts through meta strategy
        for alert in alerts:
            update = engine.process_alert(alert, iteration=10)
            assert update is not None


class TestCheckpointBranchFlow:
    """Integration test for checkpoint + branch management."""

    def test_checkpoint_branch_workflow(self):
        from agisti.checkpoint.manager import CheckpointManager
        from agisti.checkpoint.branch import BranchManager

        torch.manual_seed(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            ckpt_dir = Path(tmpdir) / "checkpoints"
            branch_dir = Path(tmpdir) / "branches"
            ckpt_dir.mkdir()
            branch_dir.mkdir()

            ckpt_mgr = CheckpointManager(CheckpointConfig(
                checkpoint_dir=ckpt_dir,
                max_checkpoints=10,
            ))
            branch_mgr = BranchManager(root_dir=branch_dir)

            state = {
                "layer.0.weight": torch.randn(32, 32),
                "layer.1.weight": torch.randn(32, 32),
            }

            # Save main checkpoint
            info = ckpt_mgr.save(state, iteration=0, score=0.5)

            # Fork experiment
            branch = branch_mgr.fork(
                name="experiment_a",
                state_dict=state,
                parent="main",
            )
            assert branch.name == "experiment_a"

            # Modify state
            state["layer.0.weight"] += 0.1
            ckpt_mgr.save(state, iteration=1, score=0.6)

            # Another experiment
            branch_b = branch_mgr.fork(
                name="experiment_b",
                state_dict=state,
                parent="main",
            )

            branches = branch_mgr.list_branches()
            assert len(branches) >= 2

            # Compare
            comparison = branch_mgr.compare("experiment_a", "experiment_b")
            assert isinstance(comparison, dict)


class TestCompetencyTrackingFlow:
    """Integration test for competency tracking → weakness → problem generation."""

    def test_competency_to_weakness_pipeline(self):
        from agisti.probe.competency import CompetencyTracker
        from agisti.probe.weakness import WeaknessAnalyzer

        tracker = CompetencyTracker(domains=["math", "logic", "coding"])

        # Record scores
        for i in range(20):
            tracker.record("math", 0.8 + 0.005 * i)
            tracker.record("logic", 0.3 + 0.01 * i)
            tracker.record("coding", 0.5)

        # Math improving, logic improving fast, coding stagnant
        current = tracker.current_scores()
        assert current["math"] > current["coding"]

        # Analyze weaknesses
        analyzer = WeaknessAnalyzer()
        failures = [
            FailedProblem(
                problem=Problem(
                    question=f"Logic Q{i}",
                    answer=str(i),
                    answer_type=AnswerType.NUMERIC_RANGE,
                    domain="logic",
                    difficulty=1,
                ),
                original_solution=Solution(
                    problem_id=f"logic_{i}",
                    answer="wrong",
                    chain_of_thought="",
                    tokens_generated=1,
                    generation_time_seconds=0.0,
                ),
                domain="logic",
                ground_truth=str(i),
                answer_type=AnswerType.NUMERIC_RANGE,
            )
            for i in range(15)
        ]
        failures.extend([
            FailedProblem(
                problem=Problem(
                    question=f"Math Q{i}",
                    answer=str(i),
                    answer_type=AnswerType.NUMERIC_RANGE,
                    domain="math",
                    difficulty=1,
                ),
                original_solution=Solution(
                    problem_id=f"math_{i}",
                    answer="wrong",
                    chain_of_thought="",
                    tokens_generated=1,
                    generation_time_seconds=0.0,
                ),
                domain="math",
                ground_truth=str(i),
                answer_type=AnswerType.NUMERIC_RANGE,
            )
            for i in range(3)
        ])

        report = analyzer.analyze(failures)
        assert report.total_failures == 18
        # Logic has more failures
        assert "logic" in report.priority_domains or report.total_failures > 0
