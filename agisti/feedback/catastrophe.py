"""
CatastropheDetector — monitors training health and fires alerts.

Implements 5 alert types (§8.1):
1. REGRESSION: Accuracy dropped significantly
2. DIVERGENCE: Loss or delta norm spiking
3. OSCILLATION: Accuracy swinging back and forth
4. FROZEN_VIOLATION: Frozen zone integrity breached
5. MODE_COLLAPSE: All answers becoming identical

Each alert has a severity level and triggers different responses
in the MetaStrategyEngine.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from agisti.types import (
    Alert,
    AlertLevel,
    AlertType,
    CatastropheDetected,
    EmergencyRollbackRequired,
    IterationResult,
)
from agisti.config import CatastropheConfig

logger = logging.getLogger(__name__)


@dataclass
class HealthSnapshot:
    """Training health at a point in time."""
    iteration: int
    accuracy: float
    loss: float | None = None
    delta_norm: float = 0.0
    domain_accuracies: dict[str, float] = field(default_factory=dict)
    answer_distribution: dict[str, int] = field(default_factory=dict)


class CatastropheDetector:
    """
    Monitors training for catastrophic events.

    Runs after each iteration to check if the model is behaving
    unhealthily. Fires alerts with appropriate severity levels.

    Alert escalation:
    - INFO: Logged but no action
    - WARNING: Strategy adjustment recommended
    - CRITICAL: Strategy adjustment required
    - EMERGENCY: Immediate rollback required
    """

    def __init__(
        self,
        config: CatastropheConfig | None = None,
        max_history: int = 500,
    ):
        self.config = config or CatastropheConfig()
        self._history: deque[HealthSnapshot] = deque(maxlen=max_history)
        self._alerts: list[Alert] = []
        self._emergency_count = 0

    def check(self, result: IterationResult) -> list[Alert]:
        """
        Check iteration result for catastrophic events.

        Args:
            result: Latest iteration result.

        Returns:
            List of alerts (may be empty for healthy iterations).

        Raises:
            EmergencyRollbackRequired: If EMERGENCY-level alert fires.
        """
        snapshot = self._extract_snapshot(result)
        self._history.append(snapshot)

        if len(self._history) < 2:
            return []

        alerts: list[Alert] = []

        # Check all detectors
        alerts.extend(self._check_regression(snapshot))
        alerts.extend(self._check_divergence(snapshot))
        alerts.extend(self._check_oscillation())
        alerts.extend(self._check_frozen_violation(result))
        alerts.extend(self._check_mode_collapse(snapshot))
        alerts.extend(self._check_loss_spike(snapshot))
        alerts.extend(self._check_convergence_stall())

        self._alerts.extend(alerts)

        # Emergency handling
        emergency_alerts = [
            a for a in alerts if a.level == AlertLevel.EMERGENCY
        ]
        if emergency_alerts:
            self._emergency_count += 1
            logger.critical(
                "EMERGENCY ALERT (count=%d): %s",
                self._emergency_count,
                "; ".join(a.message for a in emergency_alerts),
            )
            if self._emergency_count >= self.config.max_emergency_count:
                raise EmergencyRollbackRequired(
                    f"Too many emergencies ({self._emergency_count}): "
                    f"{emergency_alerts[0].message}"
                )

        return alerts

    def _extract_snapshot(
        self, result: IterationResult,
    ) -> HealthSnapshot:
        """Extract health snapshot from iteration result."""
        accuracy = 0.0
        domain_accs: dict[str, float] = {}

        if result.quick_bench:
            accuracy = result.quick_bench.accuracy
            domain_accs = dict(result.quick_bench.domain_breakdown)

        delta_norm = 0.0
        if result.delta:
            delta_norm = result.delta.norm()

        # Collect answer distribution
        answer_dist: dict[str, int] = {}
        if result.solutions:
            for sol in result.solutions:
                key = sol.answer[:50] if sol.answer else "<empty>"
                answer_dist[key] = answer_dist.get(key, 0) + 1

        return HealthSnapshot(
            iteration=result.iteration,
            accuracy=accuracy,
            loss=result.loss,
            delta_norm=delta_norm,
            domain_accuracies=domain_accs,
            answer_distribution=answer_dist,
        )

    def _check_regression(
        self, snapshot: HealthSnapshot,
    ) -> list[Alert]:
        """Check for accuracy regression."""
        if len(self._history) < 2:
            return []

        prev = self._history[-2]
        drop = prev.accuracy - snapshot.accuracy

        if drop <= 0:
            return []

        if drop >= self.config.crash_threshold:
            return [Alert(
                type=AlertType.REGRESSION,
                level=AlertLevel.EMERGENCY,
                message=(
                    f"Accuracy crash: {prev.accuracy:.3f} → "
                    f"{snapshot.accuracy:.3f} (Δ={-drop:.3f})"
                ),
                iteration=snapshot.iteration,
                details={
                    "previous": prev.accuracy,
                    "current": snapshot.accuracy,
                    "drop": drop,
                },
            )]

        if drop >= self.config.regression_threshold:
            level = (
                AlertLevel.CRITICAL
                if drop >= self.config.regression_threshold * 2
                else AlertLevel.WARNING
            )
            return [Alert(
                type=AlertType.REGRESSION,
                level=level,
                message=(
                    f"Accuracy regression: {prev.accuracy:.3f} → "
                    f"{snapshot.accuracy:.3f} (Δ={-drop:.3f})"
                ),
                iteration=snapshot.iteration,
                details={"drop": drop},
            )]

        return []

    def _check_divergence(
        self, snapshot: HealthSnapshot,
    ) -> list[Alert]:
        """Check for delta norm divergence."""
        if len(self._history) < 5:
            return []

        recent_norms = [
            h.delta_norm for h in list(self._history)[-5:]
            if h.delta_norm > 0
        ]
        if len(recent_norms) < 3:
            return []

        avg_norm = sum(recent_norms[:-1]) / len(recent_norms[:-1])
        if avg_norm <= 0:
            return []

        ratio = snapshot.delta_norm / avg_norm

        if ratio > self.config.divergence_norm_ratio:
            level = (
                AlertLevel.CRITICAL
                if ratio > self.config.divergence_norm_ratio * 2
                else AlertLevel.WARNING
            )
            return [Alert(
                type=AlertType.DIVERGENCE,
                level=level,
                message=(
                    f"Delta norm divergence: {snapshot.delta_norm:.4f} "
                    f"({ratio:.1f}x of avg {avg_norm:.4f})"
                ),
                iteration=snapshot.iteration,
                details={
                    "norm": snapshot.delta_norm,
                    "avg_norm": avg_norm,
                    "ratio": ratio,
                },
            )]

        return []

    def _check_oscillation(self) -> list[Alert]:
        """
        Check for accuracy oscillation.

        Oscillation = sign changes in improvement. If the last N
        improvements alternate between positive and negative,
        the model is oscillating.
        """
        if len(self._history) < 6:
            return []

        recent = list(self._history)[-6:]
        accs = [h.accuracy for h in recent]

        # Count sign changes
        diffs = [accs[i+1] - accs[i] for i in range(len(accs) - 1)]
        sign_changes = sum(
            1 for i in range(len(diffs) - 1)
            if (diffs[i] > 0) != (diffs[i+1] > 0)
        )

        # 4+ sign changes in 5 diffs = oscillation
        if sign_changes >= 4:
            amplitude = max(accs) - min(accs)
            level = (
                AlertLevel.CRITICAL
                if amplitude > 0.05
                else AlertLevel.WARNING
            )
            return [Alert(
                type=AlertType.OSCILLATION,
                level=level,
                message=(
                    f"Accuracy oscillation: {sign_changes} sign changes, "
                    f"amplitude={amplitude:.3f}"
                ),
                iteration=recent[-1].iteration,
                details={
                    "sign_changes": sign_changes,
                    "amplitude": amplitude,
                    "recent_accs": accs,
                },
            )]

        return []

    def _check_frozen_violation(
        self, result: IterationResult,
    ) -> list[Alert]:
        """Check for frozen zone violations."""
        if not result.frozen_violations:
            return []

        return [Alert(
            type=AlertType.FROZEN_VIOLATION,
            level=AlertLevel.EMERGENCY,
            message=(
                f"Frozen zone violation detected: "
                f"{len(result.frozen_violations)} layer(s)"
            ),
            iteration=result.iteration,
            details={"layers": result.frozen_violations},
        )]

    def _check_mode_collapse(
        self, snapshot: HealthSnapshot,
    ) -> list[Alert]:
        """
        Check for mode collapse — all answers becoming identical.

        If the most common answer accounts for >80% of all answers,
        the model has collapsed to a single mode.
        """
        if not snapshot.answer_distribution:
            return []

        total = sum(snapshot.answer_distribution.values())
        if total < 10:
            return []

        most_common_count = max(snapshot.answer_distribution.values())
        collapse_ratio = most_common_count / total

        if collapse_ratio > self.config.mode_collapse_threshold:
            level = (
                AlertLevel.CRITICAL
                if collapse_ratio > 0.95
                else AlertLevel.WARNING
            )
            return [Alert(
                type=AlertType.MODE_COLLAPSE,
                level=level,
                message=(
                    f"Mode collapse: {collapse_ratio:.1%} of answers "
                    f"are identical"
                ),
                iteration=snapshot.iteration,
                details={
                    "collapse_ratio": collapse_ratio,
                    "most_common_count": most_common_count,
                    "total": total,
                    "unique_answers": len(snapshot.answer_distribution),
                },
            )]

        return []

    def _check_loss_spike(
        self, snapshot: HealthSnapshot,
    ) -> list[Alert]:
        """Check for sudden loss spikes."""
        if snapshot.loss is None:
            return []

        if len(self._history) < 5:
            return []

        # Compute average loss
        recent_losses = [
            h.loss for h in list(self._history)[-5:]
            if h.loss is not None
        ]
        if len(recent_losses) < 3:
            return []

        avg_loss = sum(recent_losses[:-1]) / len(recent_losses[:-1])
        if avg_loss <= 0:
            return []

        ratio = snapshot.loss / avg_loss

        if ratio > self.config.loss_spike_ratio:
            return [Alert(
                type=AlertType.LOSS_SPIKE,
                level=AlertLevel.CRITICAL,
                message=(
                    f"Loss spike: {snapshot.loss:.4f} "
                    f"({ratio:.1f}x of avg {avg_loss:.4f})"
                ),
                iteration=snapshot.iteration,
                details={
                    "loss": snapshot.loss,
                    "avg_loss": avg_loss,
                    "ratio": ratio,
                },
            )]

        return []

    def _check_convergence_stall(self) -> list[Alert]:
        """
        Check for convergence stall — no improvement for many iterations.
        """
        stall_window = self.config.stall_iterations
        if len(self._history) < stall_window:
            return []

        recent = list(self._history)[-stall_window:]
        accs = [h.accuracy for h in recent]
        improvement = accs[-1] - accs[0]

        if abs(improvement) < self.config.stall_threshold:
            return [Alert(
                type=AlertType.CONVERGENCE_STALL,
                level=AlertLevel.WARNING,
                message=(
                    f"Convergence stall: Δaccuracy={improvement:.4f} "
                    f"over {stall_window} iterations"
                ),
                iteration=recent[-1].iteration,
                details={
                    "improvement": improvement,
                    "window": stall_window,
                    "start_acc": accs[0],
                    "end_acc": accs[-1],
                },
            )]

        return []

    @property
    def all_alerts(self) -> list[Alert]:
        """All alerts ever fired."""
        return list(self._alerts)

    @property
    def recent_alerts(self) -> list[Alert]:
        """Alerts from the last 5 iterations."""
        if not self._history:
            return []
        current = self._history[-1].iteration
        return [
            a for a in self._alerts
            if a.iteration >= current - 5
        ]

    @property
    def emergency_count(self) -> int:
        return self._emergency_count

    def get_health_summary(self) -> dict[str, Any]:
        """Get a summary of training health."""
        if not self._history:
            return {"status": "no_data"}

        recent = list(self._history)[-10:]
        accs = [h.accuracy for h in recent]
        norms = [h.delta_norm for h in recent if h.delta_norm > 0]

        return {
            "status": "healthy" if not self.recent_alerts else "alerting",
            "iterations_tracked": len(self._history),
            "current_accuracy": accs[-1] if accs else 0.0,
            "accuracy_trend": (accs[-1] - accs[0]) if len(accs) > 1 else 0.0,
            "avg_delta_norm": sum(norms) / len(norms) if norms else 0.0,
            "total_alerts": len(self._alerts),
            "emergency_count": self._emergency_count,
            "recent_alert_count": len(self.recent_alerts),
        }

    def reset(self) -> None:
        """Reset all tracking state."""
        self._history.clear()
        self._alerts.clear()
        self._emergency_count = 0


class DomainHealthTracker:
    """
    Track per-domain health to detect domain-specific problems.

    Monitors each domain independently and detects:
    - Domain-specific regression
    - One domain dragging overall performance
    - Domain diversity collapse
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._domain_history: dict[str, deque[float]] = {}

    def update(self, domain_accuracies: dict[str, float]) -> list[Alert]:
        """Update domain accuracies and check for issues."""
        alerts: list[Alert] = []

        for domain, acc in domain_accuracies.items():
            if domain not in self._domain_history:
                self._domain_history[domain] = deque(maxlen=self.window_size * 2)
            self._domain_history[domain].append(acc)

        # Check domain-specific regression
        alerts.extend(self._check_domain_regression())

        # Check domain diversity
        alerts.extend(self._check_domain_diversity(domain_accuracies))

        return alerts

    def _check_domain_regression(self) -> list[Alert]:
        """Detect regression in specific domains."""
        alerts: list[Alert] = []

        for domain, history in self._domain_history.items():
            if len(history) < 5:
                continue

            recent = list(history)[-5:]
            if len(recent) >= 2 and recent[-1] < recent[0] - 0.1:
                alerts.append(Alert(
                    type=AlertType.REGRESSION,
                    level=AlertLevel.WARNING,
                    message=(
                        f"Domain '{domain}' regression: "
                        f"{recent[0]:.3f} → {recent[-1]:.3f}"
                    ),
                    iteration=-1,
                    details={
                        "domain": domain,
                        "drop": recent[0] - recent[-1],
                    },
                ))

        return alerts

    def _check_domain_diversity(
        self, domain_accuracies: dict[str, float],
    ) -> list[Alert]:
        """Check if domain performance is too concentrated."""
        if len(domain_accuracies) < 3:
            return []

        accs = list(domain_accuracies.values())
        mean = sum(accs) / len(accs)
        var = sum((a - mean) ** 2 for a in accs) / len(accs)
        std = math.sqrt(var) if var > 0 else 0.0

        # High variance means uneven domain performance
        if std > 0.2:
            worst = min(domain_accuracies, key=domain_accuracies.get)
            best = max(domain_accuracies, key=domain_accuracies.get)
            return [Alert(
                type=AlertType.MODE_COLLAPSE,
                level=AlertLevel.INFO,
                message=(
                    f"Domain diversity issue: std={std:.3f}, "
                    f"best={best} ({domain_accuracies[best]:.3f}), "
                    f"worst={worst} ({domain_accuracies[worst]:.3f})"
                ),
                iteration=-1,
                details={
                    "std": std,
                    "best_domain": best,
                    "worst_domain": worst,
                },
            )]

        return []

    def get_domain_trends(self) -> dict[str, float]:
        """Get improvement trend per domain."""
        trends: dict[str, float] = {}
        for domain, history in self._domain_history.items():
            if len(history) < 3:
                trends[domain] = 0.0
                continue

            recent = list(history)[-self.window_size:]
            n = len(recent)
            x_mean = (n - 1) / 2
            y_mean = sum(recent) / n

            numerator = sum(
                (i - x_mean) * (y - y_mean)
                for i, y in enumerate(recent)
            )
            denominator = sum(
                (i - x_mean) ** 2 for i in range(n)
            )

            trends[domain] = (
                numerator / denominator if denominator > 0 else 0.0
            )

        return trends

    def get_weakest_domain(self) -> str | None:
        """Get the currently weakest domain."""
        if not self._domain_history:
            return None

        latest: dict[str, float] = {}
        for domain, history in self._domain_history.items():
            if history:
                latest[domain] = history[-1]

        if not latest:
            return None

        return min(latest, key=latest.get)


class TrainingHealthMonitor:
    """
    Composite health monitor combining catastrophe detection
    and domain-level tracking.

    This is the main class that should be used by the orchestrator.
    """

    def __init__(
        self,
        config: CatastropheConfig | None = None,
    ):
        self.detector = CatastropheDetector(config)
        self.domain_tracker = DomainHealthTracker()
        self._iteration = 0

    def check(self, result: IterationResult) -> list[Alert]:
        """
        Run all health checks on an iteration result.

        Returns combined alerts from catastrophe detection
        and domain health tracking.
        """
        self._iteration += 1

        alerts = self.detector.check(result)

        if result.quick_bench and result.quick_bench.domain_breakdown:
            domain_alerts = self.domain_tracker.update(
                dict(result.quick_bench.domain_breakdown),
            )
            alerts.extend(domain_alerts)

        return alerts

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive health summary."""
        summary = self.detector.get_health_summary()
        summary["domain_trends"] = self.domain_tracker.get_domain_trends()
        summary["weakest_domain"] = self.domain_tracker.get_weakest_domain()
        return summary

    def reset(self) -> None:
        """Reset all state."""
        self.detector.reset()
        self.domain_tracker = DomainHealthTracker()
        self._iteration = 0
