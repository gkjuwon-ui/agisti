"""
Difficulty adaptation — zone of proximal development targeting.

Adjusts problem difficulty so the model is always challenged
at the right level: not too easy (wasted training), not too hard
(model can't learn from total failures).

Target accuracy zone: 30-70% (configurable).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DifficultyProfile:
    """Snapshot of a domain's difficulty state."""
    domain: str
    current_difficulty: float
    target_accuracy_low: float
    target_accuracy_high: float
    recent_accuracy: float
    trend: float  # positive = accuracy improving
    pressure: float  # positive = need harder problems
    optimal: bool  # accuracy within target zone

    def __repr__(self) -> str:
        status = "✓" if self.optimal else "✗"
        return (
            f"DifficultyProfile({self.domain}: "
            f"diff={self.current_difficulty:.2f}, "
            f"acc={self.recent_accuracy:.1%}, "
            f"pressure={self.pressure:+.2f} {status})"
        )


class AdaptiveDifficultyEngine:
    """
    Manages difficulty adaptation across all domains.

    Uses a PID-like controller to smoothly converge
    on the target accuracy zone:
    - P (proportional): how far from target zone
    - I (integral): accumulated error over time
    - D (derivative): rate of change of accuracy

    This prevents oscillation between too-easy and too-hard.
    """

    def __init__(
        self,
        target_accuracy_low: float = 0.3,
        target_accuracy_high: float = 0.7,
        kp: float = 0.1,  # proportional gain
        ki: float = 0.01,  # integral gain
        kd: float = 0.05,  # derivative gain
        max_step: float = 0.15,  # max difficulty change per update
        min_difficulty: float = 0.05,
        max_difficulty: float = 0.95,
    ):
        self.target_low = target_accuracy_low
        self.target_high = target_accuracy_high
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_step = max_step
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty

        # Per-domain state
        self._difficulty: dict[str, float] = {}
        self._accuracy_history: dict[str, list[float]] = {}
        self._integral_error: dict[str, float] = {}
        self._prev_error: dict[str, float] = {}

    def get_difficulty(self, domain: str) -> float:
        """Get current target difficulty for a domain."""
        return self._difficulty.get(domain, 0.5)

    def update(self, domain: str, accuracy: float) -> float:
        """
        Update difficulty based on observed accuracy.

        Uses PID control to smoothly adjust difficulty toward
        the target accuracy zone.

        Returns:
            New target difficulty.
        """
        current = self._difficulty.get(domain, 0.5)
        target_center = (self.target_low + self.target_high) / 2

        # Error: how far accuracy is from target center
        # Positive error = accuracy too high = need harder problems
        error = accuracy - target_center

        # Dead zone: if within target range, zero error
        if self.target_low <= accuracy <= self.target_high:
            error = 0.0

        # Track history
        if domain not in self._accuracy_history:
            self._accuracy_history[domain] = []
            self._integral_error[domain] = 0.0
            self._prev_error[domain] = 0.0

        self._accuracy_history[domain].append(accuracy)

        # PID computation
        p_term = self.kp * error
        self._integral_error[domain] += error
        # Anti-windup: clamp integral
        self._integral_error[domain] = max(
            -5.0, min(5.0, self._integral_error[domain]),
        )
        i_term = self.ki * self._integral_error[domain]
        d_term = self.kd * (error - self._prev_error[domain])
        self._prev_error[domain] = error

        adjustment = p_term + i_term + d_term

        # Clamp adjustment
        adjustment = max(-self.max_step, min(self.max_step, adjustment))

        new_difficulty = current + adjustment
        new_difficulty = max(
            self.min_difficulty,
            min(self.max_difficulty, new_difficulty),
        )

        self._difficulty[domain] = new_difficulty

        if abs(adjustment) > 0.005:
            logger.debug(
                "Domain %s: difficulty %.3f → %.3f "
                "(accuracy=%.1f%%, P=%.3f, I=%.3f, D=%.3f)",
                domain,
                current,
                new_difficulty,
                accuracy * 100,
                p_term,
                i_term,
                d_term,
            )

        return new_difficulty

    def get_profile(self, domain: str) -> DifficultyProfile:
        """Get full difficulty profile for a domain."""
        history = self._accuracy_history.get(domain, [])
        recent_accuracy = history[-1] if history else 0.5
        trend = self._compute_trend(history)
        pressure = self._compute_pressure(domain)

        return DifficultyProfile(
            domain=domain,
            current_difficulty=self.get_difficulty(domain),
            target_accuracy_low=self.target_low,
            target_accuracy_high=self.target_high,
            recent_accuracy=recent_accuracy,
            trend=trend,
            pressure=pressure,
            optimal=self.target_low <= recent_accuracy <= self.target_high,
        )

    def get_all_profiles(self) -> list[DifficultyProfile]:
        """Get profiles for all tracked domains."""
        return [self.get_profile(d) for d in self._difficulty]

    def reset_domain(self, domain: str) -> None:
        """Reset a domain's difficulty (e.g., after architecture change)."""
        self._difficulty.pop(domain, None)
        self._accuracy_history.pop(domain, None)
        self._integral_error.pop(domain, None)
        self._prev_error.pop(domain, None)

    def bulk_update(
        self, domain_accuracies: dict[str, float],
    ) -> dict[str, float]:
        """Update multiple domains at once."""
        return {
            domain: self.update(domain, accuracy)
            for domain, accuracy in domain_accuracies.items()
        }

    def _compute_trend(
        self, history: list[float], window: int = 10,
    ) -> float:
        """Compute accuracy trend (slope)."""
        if len(history) < 3:
            return 0.0
        recent = history[-window:]
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(recent))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if abs(den) < 1e-12:
            return 0.0
        return num / den

    def _compute_pressure(self, domain: str) -> float:
        """
        Compute learning pressure.

        Positive = need harder problems, negative = need easier.
        """
        history = self._accuracy_history.get(domain, [])
        if not history:
            return 0.0
        recent = history[-5:] if len(history) >= 5 else history
        avg = sum(recent) / len(recent)
        center = (self.target_low + self.target_high) / 2
        return (avg - center) / max(center, 0.01)


class CurriculumScheduler:
    """
    Schedules which domains and difficulty levels to focus on.

    Implements a curriculum learning strategy that:
    1. Starts with easier problems in weak domains
    2. Gradually increases difficulty as mastery improves
    3. Maintains minimum exposure to all domains for coverage
    """

    def __init__(
        self,
        difficulty_engine: AdaptiveDifficultyEngine,
        min_domain_fraction: float = 0.1,
        weakness_multiplier: float = 3.0,
    ):
        self.engine = difficulty_engine
        self.min_fraction = min_domain_fraction
        self.weakness_multiplier = weakness_multiplier

    def compute_domain_weights(
        self,
        domain_scores: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute training weight per domain.

        Weak domains get more training time (inversely proportional
        to competency score), subject to minimum exposure constraint.
        """
        if not domain_scores:
            return {}

        # Inverse weighting: lower score = higher weight
        raw_weights = {}
        for domain, score in domain_scores.items():
            # Clamp score away from 0 to avoid division issues
            clamped = max(0.05, score)
            raw_weights[domain] = (1.0 / clamped) ** self.weakness_multiplier

        # Normalize weights
        total = sum(raw_weights.values())
        if total < 1e-12:
            # Equal weights fallback
            n = len(raw_weights)
            return {d: 1.0 / n for d in raw_weights}

        normalized = {d: w / total for d, w in raw_weights.items()}

        # Apply minimum fraction
        n_domains = len(normalized)
        for domain in normalized:
            normalized[domain] = max(
                self.min_fraction / n_domains,
                normalized[domain],
            )

        # Re-normalize after applying minimums
        total = sum(normalized.values())
        normalized = {d: w / total for d, w in normalized.items()}

        return normalized

    def compute_problem_counts(
        self,
        domain_scores: dict[str, float],
        total_problems: int,
    ) -> dict[str, int]:
        """
        Compute how many problems to generate per domain.

        Distributes total_problems budget across domains
        based on their weights.
        """
        weights = self.compute_domain_weights(domain_scores)
        counts: dict[str, int] = {}
        allocated = 0

        for domain in sorted(weights.keys(), key=lambda d: -weights[d]):
            count = max(1, round(weights[domain] * total_problems))
            counts[domain] = count
            allocated += count

        # Adjust for rounding errors
        if allocated > total_problems:
            excess = allocated - total_problems
            for domain in sorted(
                counts.keys(), key=lambda d: counts[d], reverse=True,
            ):
                if excess <= 0:
                    break
                reduce = min(counts[domain] - 1, excess)
                counts[domain] -= reduce
                excess -= reduce
        elif allocated < total_problems:
            deficit = total_problems - allocated
            weakest = sorted(weights.keys(), key=lambda d: -weights[d])
            for i, domain in enumerate(weakest):
                if deficit <= 0:
                    break
                counts[domain] += 1
                deficit -= 1

        return counts

    def plan_iteration(
        self,
        domain_scores: dict[str, float],
        total_problems: int,
    ) -> list[dict[str, Any]]:
        """
        Full iteration plan: what to generate, at what difficulty.

        Returns list of generation specs for the ProblemGenerator.
        """
        counts = self.compute_problem_counts(domain_scores, total_problems)
        plan = []

        for domain, count in counts.items():
            profile = self.engine.get_profile(domain)
            plan.append({
                "domain": domain,
                "count": count,
                "difficulty": profile.current_difficulty,
                "weight": counts[domain] / total_problems,
                "optimal_zone": profile.optimal,
                "pressure": profile.pressure,
            })

        # Sort plan: non-optimal domains first
        plan.sort(key=lambda x: (x["optimal_zone"], -x["count"]))

        return plan
