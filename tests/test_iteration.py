"""
Tests for agisti.iteration — runner, state machine, history.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from agisti.types import (
    IterationResult,
    IterationState,
    QuickBenchResult,
)
from agisti.iteration.state_machine import (
    IterationStateMachine,
    IterationBatchTracker,
    InvalidTransitionError,
    StateTransition,
    VALID_TRANSITIONS,
)
from agisti.iteration.history import (
    IterationHistory,
    EpochSummary,
)


# ─── State Machine Tests ─────────────────────────────

class TestIterationStateMachine:
    """Tests for IterationStateMachine."""

    def test_initial_state(self):
        sm = IterationStateMachine()
        assert sm.state == IterationState.IDLE

    def test_valid_transition(self):
        sm = IterationStateMachine()
        record = sm.transition(IterationState.PROBE)
        assert sm.state == IterationState.PROBE
        assert isinstance(record, StateTransition)
        assert record.from_state == IterationState.IDLE
        assert record.to_state == IterationState.PROBE

    def test_full_pipeline(self):
        """Complete iteration pipeline should succeed."""
        sm = IterationStateMachine()
        states = [
            IterationState.PROBE,
            IterationState.GENERATE,
            IterationState.SOLVE,
            IterationState.EVALUATE,
            IterationState.PROPOSE,
            IterationState.VIRTUAL_TRAIN,
            IterationState.APPLY_DELTA,
            IterationState.QUICK_BENCH,
            IterationState.FEEDBACK,
        ]
        for state in states:
            sm.transition(state)
        assert sm.state == IterationState.FEEDBACK

    def test_invalid_transition_raises(self):
        sm = IterationStateMachine()
        with pytest.raises(InvalidTransitionError):
            sm.transition(IterationState.SOLVE)

    def test_cannot_skip_states(self):
        sm = IterationStateMachine()
        sm.transition(IterationState.PROBE)
        with pytest.raises(InvalidTransitionError):
            # Cannot skip GENERATE → SOLVE
            sm.transition(IterationState.SOLVE)

    def test_rollback_path(self):
        """FEEDBACK → ROLLBACK → IDLE should work."""
        sm = IterationStateMachine()
        sm.transition(IterationState.PROBE)
        sm.transition(IterationState.GENERATE)
        sm.transition(IterationState.SOLVE)
        sm.transition(IterationState.EVALUATE)
        sm.transition(IterationState.PROPOSE)
        sm.transition(IterationState.FEEDBACK)
        sm.transition(IterationState.ROLLBACK)
        assert sm.state == IterationState.ROLLBACK

    def test_history(self):
        sm = IterationStateMachine()
        sm.transition(IterationState.PROBE)
        sm.transition(IterationState.GENERATE)
        assert len(sm.history) == 2
        assert sm.transition_count == 2

    def test_reset(self):
        sm = IterationStateMachine()
        sm.transition(IterationState.PROBE)
        sm.reset()
        assert sm.state == IterationState.IDLE
        assert sm.transition_count == 0
        assert len(sm.history) == 0

    def test_step_timing(self):
        sm = IterationStateMachine()
        sm.transition(IterationState.PROBE)
        time.sleep(0.01)
        sm.transition(IterationState.GENERATE)

        timing = sm.get_step_timing()
        assert IterationState.PROBE.value in timing
        assert timing[IterationState.PROBE.value] > 0

    def test_slowest_step(self):
        sm = IterationStateMachine()
        sm.transition(IterationState.PROBE)
        sm.transition(IterationState.GENERATE)
        result = sm.get_slowest_step()
        assert result is not None
        assert isinstance(result[0], str)
        assert result[1] >= 0

    def test_has_visited(self):
        sm = IterationStateMachine()
        sm.transition(IterationState.PROBE)
        assert sm.has_visited(IterationState.PROBE)
        assert sm.has_visited(IterationState.IDLE)
        assert not sm.has_visited(IterationState.GENERATE)

    def test_is_terminal(self):
        sm = IterationStateMachine()
        assert not sm.is_terminal()

    def test_repr(self):
        sm = IterationStateMachine()
        s = repr(sm)
        assert "IDLE" in s


class TestValidTransitions:
    """Tests for the transition graph."""

    def test_all_states_have_entries(self):
        for state in IterationState:
            assert state in VALID_TRANSITIONS, (
                f"State {state.value} missing from VALID_TRANSITIONS"
            )

    def test_complete_is_terminal(self):
        assert len(VALID_TRANSITIONS[IterationState.COMPLETE]) == 0

    def test_idle_starts_with_probe(self):
        assert IterationState.PROBE in VALID_TRANSITIONS[IterationState.IDLE]


class TestIterationBatchTracker:
    """Tests for IterationBatchTracker."""

    def test_empty(self):
        tracker = IterationBatchTracker()
        assert tracker.total_iterations == 0
        assert tracker.error_rate == 0.0

    def test_record_iteration(self):
        tracker = IterationBatchTracker()
        sm = IterationStateMachine()
        sm.transition(IterationState.PROBE)
        sm.transition(IterationState.GENERATE)
        tracker.record_iteration(sm, wall_time=1.0)
        assert tracker.total_iterations == 1

    def test_error_rate(self):
        tracker = IterationBatchTracker()
        sm = IterationStateMachine()
        sm.transition(IterationState.PROBE)
        tracker.record_iteration(sm, wall_time=1.0)
        tracker.record_error()
        assert tracker.error_rate == pytest.approx(0.5)

    def test_throughput(self):
        tracker = IterationBatchTracker()
        sm = IterationStateMachine()
        sm.transition(IterationState.PROBE)
        for _ in range(5):
            tracker.record_iteration(sm, wall_time=2.0)
        assert tracker.get_throughput() == pytest.approx(0.5)

    def test_summary(self):
        tracker = IterationBatchTracker()
        sm = IterationStateMachine()
        sm.transition(IterationState.PROBE)
        tracker.record_iteration(sm, wall_time=1.0)
        s = tracker.summary()
        assert s["total_iterations"] == 1
        assert "avg_wall_time_s" in s


# ─── History Tests ─────────────────────────────────

class TestIterationHistory:
    """Tests for IterationHistory."""

    def _make_result(
        self,
        iteration: int,
        accepted: bool = True,
        score: float = 0.5,
        epoch: int = 0,
        delta_norm: float = 0.01,
    ) -> IterationResult:
        return IterationResult(
            iteration_id=iteration,
            proposed_delta_norm=delta_norm,
            virtual_loss_before=2.0,
            virtual_loss_after=1.8,
            refined_delta_norm=delta_norm,
            quick_bench_scores={"math": score},
            accepted=accepted,
            wall_time_seconds=1.0,
            gpu_memory_peak_gb=10.0,
            epoch=epoch,
        )

    def test_empty_history(self):
        h = IterationHistory()
        assert h.total_iterations == 0
        assert h.acceptance_rate == 0.0

    def test_add_and_count(self):
        h = IterationHistory()
        h.add(self._make_result(0))
        h.add(self._make_result(1))
        assert h.total_iterations == 2

    def test_acceptance_rate(self):
        h = IterationHistory()
        h.add(self._make_result(0, accepted=True))
        h.add(self._make_result(1, accepted=True))
        h.add(self._make_result(2, accepted=False))
        assert h.acceptance_rate == pytest.approx(2 / 3)

    def test_get_latest(self):
        h = IterationHistory()
        for i in range(10):
            h.add(self._make_result(i))
        latest = h.get_latest(3)
        assert len(latest) == 3
        assert latest[-1].iteration_id == 9

    def test_epoch_results(self):
        h = IterationHistory()
        h.add(self._make_result(0, epoch=0))
        h.add(self._make_result(1, epoch=0))
        h.add(self._make_result(2, epoch=1))
        assert len(h.get_epoch_results(0)) == 2
        assert len(h.get_epoch_results(1)) == 1
        assert len(h.get_epoch_results(99)) == 0

    def test_moving_average(self):
        h = IterationHistory()
        for i in range(20):
            h.add(self._make_result(i, score=float(i) / 20))
        ma = h.moving_average_score(window=5)
        assert len(ma) == 20
        # MA should be smoother
        assert ma[-1] > 0  # last values should be positive

    def test_score_slope_positive(self):
        h = IterationHistory()
        for i in range(20):
            h.add(self._make_result(i, score=0.1 * i))
        slope = h.score_slope()
        assert slope > 0

    def test_score_slope_negative(self):
        h = IterationHistory()
        for i in range(20):
            h.add(self._make_result(i, score=1.0 - 0.05 * i))
        slope = h.score_slope()
        assert slope < 0

    def test_not_plateauing_when_improving(self):
        h = IterationHistory()
        for i in range(40):
            h.add(self._make_result(i, score=0.1 * i))
        assert h.is_plateauing() is False

    def test_plateauing_when_constant(self):
        h = IterationHistory()
        for i in range(40):
            h.add(self._make_result(i, score=0.5))
        assert h.is_plateauing() is True

    def test_best_iteration(self):
        h = IterationHistory()
        h.add(self._make_result(0, score=0.3))
        h.add(self._make_result(1, score=0.9))
        h.add(self._make_result(2, score=0.5))
        best = h.best_iteration()
        assert best is not None
        assert best.iteration_id == 1

    def test_epoch_summary(self):
        h = IterationHistory()
        h.add(self._make_result(0, epoch=0, accepted=True, score=0.6))
        h.add(self._make_result(1, epoch=0, accepted=False, score=0.4))
        summary = h.epoch_summary(0)
        assert summary is not None
        assert summary.iterations == 2
        assert summary.accepted == 1
        assert summary.rejection_rate == 0.5

    def test_all_domains(self):
        h = IterationHistory()
        r = IterationResult(
            iteration_id=0,
            proposed_delta_norm=0.01,
            virtual_loss_before=2.0,
            virtual_loss_after=1.8,
            refined_delta_norm=0.01,
            quick_bench_scores={"math": 0.8, "logic": 0.7},
            accepted=True,
            wall_time_seconds=1.0,
            gpu_memory_peak_gb=10.0,
        )
        h.add(r)
        domains = h.all_domains()
        assert "math" in domains
        assert "logic" in domains

    def test_domain_progress(self):
        h = IterationHistory()
        for i in range(10):
            r = IterationResult(
                iteration_id=i,
                proposed_delta_norm=0.01,
                virtual_loss_before=2.0,
                virtual_loss_after=1.8,
                refined_delta_norm=0.01,
                quick_bench_scores={"math": 0.5 + 0.03 * i},
                accepted=True,
                wall_time_seconds=1.0,
                gpu_memory_peak_gb=10.0,
            )
            h.add(r)
        progress = h.domain_progress()
        assert "math" in progress
        assert progress["math"]["first"] < progress["math"]["last"]

    def test_statistics(self):
        h = IterationHistory()
        for i in range(5):
            h.add(self._make_result(i, score=0.5 + 0.1 * i))
        stats = h.statistics()
        assert stats["total_iterations"] == 5
        assert "score" in stats
        assert "slope" in stats

    def test_format_report(self):
        h = IterationHistory()
        for i in range(5):
            h.add(self._make_result(i))
        report = h.format_report()
        assert "AGISTI" in report
        assert "iterations" in report.lower()


class TestIterationHistoryPersistence:
    """Tests for history persistence (JSONL)."""

    def test_save_and_load(self):
        import tempfile

        h = IterationHistory()
        for i in range(10):
            h.add(IterationResult(
                iteration_id=i,
                proposed_delta_norm=0.01,
                virtual_loss_before=2.0,
                virtual_loss_after=1.8,
                refined_delta_norm=0.01,
                quick_bench_scores={"math": 0.5 + 0.01 * i},
                accepted=i % 2 == 0,
                wall_time_seconds=1.0,
                gpu_memory_peak_gb=10.0,
                epoch=i // 5,
            ))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.jsonl"
            h.save_to_jsonl(path)

            loaded = IterationHistory.load_from_jsonl(path)
            assert loaded.total_iterations == 10
            assert loaded.accepted_count == h.accepted_count

    def test_load_nonexistent(self):
        from pathlib import Path as P
        h = IterationHistory.load_from_jsonl(P("/does/not/exist.jsonl"))
        assert h.total_iterations == 0


class TestEpochSummary:
    """Tests for EpochSummary dataclass."""

    def test_rejection_rate(self):
        s = EpochSummary(
            epoch=0,
            iterations=10,
            accepted=7,
            rejected=3,
            acceptance_rate=0.7,
            avg_delta_norm=0.01,
            avg_quick_bench=0.5,
            best_quick_bench=0.8,
            total_wall_time=60.0,
        )
        assert s.rejection_rate == pytest.approx(0.3)
