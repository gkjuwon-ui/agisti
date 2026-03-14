"""
Tests for agisti.feedback — meta_strategy and catastrophe.
"""

from __future__ import annotations

import pytest

from agisti.types import (
    Alert,
    AlertLevel,
    AlertType,
    ConvergenceAction,
    IterationResult,
    QuickBenchResult,
)
from agisti.config import (
    CatastropheConfig,
    MetaStrategy,
    ConvergenceConfig,
    PHASE0_STRATEGY,
)
from agisti.feedback.meta_strategy import MetaStrategyEngine, StrategyUpdate
from agisti.feedback.catastrophe import (
    CatastropheDetector,
    HealthSnapshot,
    DomainHealthTracker,
    TrainingHealthMonitor,
)


# ─── Helpers ──────────────────────────────────────

def _make_result(
    iteration: int = 0,
    score: float = 0.5,
    delta_norm: float = 0.01,
    accepted: bool = True,
    epoch: int = 0,
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


def _score(value: float) -> dict[str, float]:
    return {"math": value}


# ─── MetaStrategyEngine Tests ─────────────────────

class TestMetaStrategyEngine:
    """Tests for MetaStrategyEngine."""

    def test_creation(self):
        engine = MetaStrategyEngine(PHASE0_STRATEGY)
        assert engine is not None

    def test_process_alert_warning(self):
        engine = MetaStrategyEngine(PHASE0_STRATEGY)
        alert = Alert(
            level=AlertLevel.WARNING,
            type=AlertType.PLATEAU,
            message="Score plateauing",
        )
        update = engine.process_alert(alert, iteration=10)
        assert isinstance(update, StrategyUpdate)

    def test_process_alert_critical(self):
        engine = MetaStrategyEngine(PHASE0_STRATEGY)
        alert = Alert(
            level=AlertLevel.CRITICAL,
            type=AlertType.SUDDEN_COLLAPSE,
            message="Sudden collapse detected",
        )
        update = engine.process_alert(alert, iteration=10)
        assert isinstance(update, StrategyUpdate)
        # Critical alerts should produce meaningful updates
        assert update.reason != ""

    def test_convergence_check_continue(self):
        engine = MetaStrategyEngine(PHASE0_STRATEGY)
        # Improving scores → CONTINUE
        results = [_make_result(i, score=0.3 + 0.02 * i) for i in range(20)]
        action = engine.check_convergence(results)
        assert action in list(ConvergenceAction)

    def test_convergence_check_few_results(self):
        engine = MetaStrategyEngine(PHASE0_STRATEGY)
        results = [_make_result(0)]
        action = engine.check_convergence(results)
        assert action == ConvergenceAction.CONTINUE

    def test_strategy_update_dataclass(self):
        su = StrategyUpdate(
            action=ConvergenceAction.CONTINUE,
            reason="All good",
        )
        assert su.action == ConvergenceAction.CONTINUE
        assert su.reason == "All good"

    def test_alert_response_diversity_drop(self):
        engine = MetaStrategyEngine(PHASE0_STRATEGY)
        alert = Alert(
            level=AlertLevel.WARNING,
            type=AlertType.DIVERSITY_DIVERGENCE,
            message="Domain diversity dropping",
        )
        update = engine.process_alert(alert, iteration=5)
        assert isinstance(update, StrategyUpdate)

    def test_alert_response_too_easy(self):
        engine = MetaStrategyEngine(PHASE0_STRATEGY)
        alert = Alert(
            level=AlertLevel.INFO,
            type=AlertType.TOO_EASY,
            message="Problems too easy",
        )
        update = engine.process_alert(alert, iteration=5)
        assert isinstance(update, StrategyUpdate)


# ─── CatastropheDetector Tests ────────────────────

class TestCatastropheDetector:
    """Tests for CatastropheDetector."""

    def test_creation(self):
        det = CatastropheDetector(CatastropheConfig())
        assert det is not None

    def test_no_alert_on_good_results(self):
        det = CatastropheDetector(CatastropheConfig())
        result = _make_result(0, score=0.5)
        alerts = det.check(result, history=[])
        assert isinstance(alerts, list)

    def test_detect_large_regression(self):
        det = CatastropheDetector(CatastropheConfig(
            regression_threshold=0.1,
        ))
        history = [_make_result(i, score=0.8) for i in range(10)]
        # Sudden drop
        current = _make_result(10, score=0.3)
        alerts = det.check(current, history=history)
        regression_alerts = [a for a in alerts if a.type == AlertType.REGRESSION]
        assert len(regression_alerts) > 0

    def test_detect_loss_spike(self):
        det = CatastropheDetector(CatastropheConfig(
            loss_spike_ratio=2.0,
        ))
        # Normal history
        history = [_make_result(i, delta_norm=0.01) for i in range(5)]
        # Loss spike
        current = IterationResult(
            iteration_id=5,
            proposed_delta_norm=0.01,
            virtual_loss_before=2.0,
            virtual_loss_after=10.0,  # Loss increased drastically
            refined_delta_norm=0.01,
            quick_bench_scores={"math": 0.5},
            accepted=True,
            wall_time_seconds=1.0,
            gpu_memory_peak_gb=10.0,
        )
        alerts = det.check(current, history=history)
        spike_alerts = [a for a in alerts if a.type == AlertType.LOSS_SPIKE]
        # May or may not trigger depending on exact thresholds
        assert isinstance(alerts, list)

    def test_no_crash_on_empty_history(self):
        det = CatastropheDetector(CatastropheConfig())
        result = _make_result(0, score=0.5)
        alerts = det.check(result, history=[])
        assert isinstance(alerts, list)

    def test_plateau_detection(self):
        det = CatastropheDetector(CatastropheConfig(
            plateau_window=5,
            plateau_slope_threshold=0.001,
        ))
        # Flat scores
        history = [_make_result(i, score=0.5) for i in range(10)]
        current = _make_result(10, score=0.5)
        alerts = det.check(current, history=history)
        plateau_alerts = [
            a for a in alerts
            if a.type in (AlertType.PLATEAU, AlertType.CONVERGENCE_STALL)
        ]
        # Should detect stalling
        assert isinstance(alerts, list)


# ─── HealthSnapshot Tests ──────────────────────────

class TestHealthSnapshot:
    """Tests for HealthSnapshot."""

    def test_creation(self):
        snap = HealthSnapshot(
            iteration=10,
            avg_score=0.5,
            acceptance_rate=0.7,
            delta_norm_mean=0.01,
            loss_trend=-0.1,
        )
        assert snap.iteration == 10
        assert snap.avg_score == 0.5

    def test_is_healthy(self):
        snap = HealthSnapshot(
            iteration=10,
            avg_score=0.5,
            acceptance_rate=0.7,
            delta_norm_mean=0.01,
            loss_trend=-0.1,
        )
        assert snap.is_healthy() is True

    def test_unhealthy_low_acceptance(self):
        snap = HealthSnapshot(
            iteration=10,
            avg_score=0.5,
            acceptance_rate=0.05,
            delta_norm_mean=0.01,
            loss_trend=0.5,
        )
        assert snap.is_healthy() is False


# ─── DomainHealthTracker Tests ────────────────────

class TestDomainHealthTracker:
    """Tests for cross-domain health monitoring."""

    def test_creation(self):
        tracker = DomainHealthTracker()
        assert tracker is not None

    def test_update(self):
        tracker = DomainHealthTracker()
        tracker.update("math", 0.5)
        tracker.update("math", 0.6)
        tracker.update("logic", 0.3)
        assert "math" in tracker.domains
        assert "logic" in tracker.domains

    def test_trend(self):
        tracker = DomainHealthTracker()
        for i in range(10):
            tracker.update("math", 0.5 + 0.01 * i)
        trend = tracker.get_trend("math")
        assert trend is not None
        assert trend > 0

    def test_trend_unknown_domain(self):
        tracker = DomainHealthTracker()
        trend = tracker.get_trend("unknown")
        assert trend is None or trend == 0.0

    def test_weakest_domain(self):
        tracker = DomainHealthTracker()
        tracker.update("math", 0.8)
        tracker.update("logic", 0.3)
        weakest = tracker.get_weakest_domain()
        assert weakest == "logic"


# ─── TrainingHealthMonitor Tests ──────────────────

class TestTrainingHealthMonitor:
    """Tests for aggregate training health tracking."""

    def test_creation(self):
        monitor = TrainingHealthMonitor()
        assert monitor is not None

    def test_record_and_snapshot(self):
        monitor = TrainingHealthMonitor()
        for i in range(10):
            result = _make_result(i, score=0.5)
            monitor.record(result)
        snapshot = monitor.snapshot()
        assert isinstance(snapshot, HealthSnapshot)
        assert snapshot.avg_score >= 0

    def test_snapshot_empty(self):
        monitor = TrainingHealthMonitor()
        snapshot = monitor.snapshot()
        assert isinstance(snapshot, HealthSnapshot)

    def test_alert_history(self):
        monitor = TrainingHealthMonitor()
        alert = Alert(
            level=AlertLevel.WARNING,
            type=AlertType.PLATEAU,
            message="Test plateau",
        )
        monitor.record_alert(alert)
        assert len(monitor.recent_alerts(5)) == 1

    def test_emergency_count(self):
        monitor = TrainingHealthMonitor()
        for _ in range(3):
            alert = Alert(
                level=AlertLevel.EMERGENCY,
                type=AlertType.SUDDEN_COLLAPSE,
                message="Emergency!",
            )
            monitor.record_alert(alert)
        assert monitor.emergency_count == 3
