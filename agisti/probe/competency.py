"""
CompetencyVector — tracks model capability across domains.

Maintains an exponential moving average (EMA) of per-domain scores,
providing a stable view of model strengths and weaknesses.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CompetencyVector:
    """
    Per-domain competency tracker with EMA smoothing.

    Stores the model's current ability in each measured domain
    as a smoothed score between 0 and 1.
    """

    scores: dict[str, float] = field(default_factory=dict)
    ema_alpha: float = 0.3  # smoothing factor for EMA updates
    raw_history: dict[str, list[float]] = field(default_factory=dict)
    raw_answers: dict[str, list[bool]] = field(default_factory=dict)
    update_count: int = 0

    @property
    def domains(self) -> list[str]:
        return list(self.scores.keys())

    def __getitem__(self, domain: str) -> float:
        return self.scores.get(domain, 0.0)

    def __setitem__(self, domain: str, value: float) -> None:
        self.scores[domain] = max(0.0, min(1.0, value))

    def as_dict(self) -> dict[str, float]:
        return dict(self.scores)

    def update(self, new_scores: dict[str, float]) -> None:
        """
        Update competency with new measurement using EMA.

        EMA formula: score_new = α * measured + (1 - α) * score_old
        This prevents noisy measurements from causing wild swings.
        """
        for domain, measured in new_scores.items():
            measured = max(0.0, min(1.0, measured))

            if domain in self.scores:
                old = self.scores[domain]
                self.scores[domain] = (
                    self.ema_alpha * measured + (1 - self.ema_alpha) * old
                )
            else:
                self.scores[domain] = measured

            # Track raw history
            if domain not in self.raw_history:
                self.raw_history[domain] = []
            self.raw_history[domain].append(measured)

        self.update_count += 1

    def update_with_answers(
        self,
        domain: str,
        correct: int,
        total: int,
        per_problem_results: list[bool] | None = None,
    ) -> None:
        """Update with raw correct/total counts."""
        if total == 0:
            return
        score = correct / total
        self.update({domain: score})
        if per_problem_results is not None:
            self.raw_answers[domain] = per_problem_results

    def get_trend(self, domain: str, window: int = 10) -> float:
        """
        Get the improvement trend for a domain.

        Returns the linear regression slope over the last `window` measurements.
        Positive = improving, negative = declining, ~0 = stagnant.
        """
        history = self.raw_history.get(domain, [])
        if len(history) < 3:
            return 0.0

        recent = history[-window:]
        return _linear_slope(recent)

    def get_overall_score(self) -> float:
        """Weighted average of all domain scores."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def get_weakest_domains(self, top_k: int = 3) -> list[tuple[str, float]]:
        """Get the weakest domains, sorted by score ascending."""
        return sorted(self.scores.items(), key=lambda x: x[1])[:top_k]

    def get_strongest_domains(self, top_k: int = 3) -> list[tuple[str, float]]:
        """Get the strongest domains, sorted by score descending."""
        return sorted(self.scores.items(), key=lambda x: -x[1])[:top_k]

    def domain_variance(self) -> float:
        """Variance across domains (high = unbalanced model)."""
        if len(self.scores) < 2:
            return 0.0
        values = list(self.scores.values())
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    def check_regression(
        self,
        previous: CompetencyVector,
        threshold: float = -0.005,
    ) -> dict[str, dict[str, float]]:
        """
        Check for regressions compared to previous competency.

        Returns dict of domains that regressed more than threshold.
        """
        regressions: dict[str, dict[str, float]] = {}
        for domain, current_score in self.scores.items():
            prev_score = previous.scores.get(domain, 0.0)
            delta = current_score - prev_score
            if delta < threshold:
                regressions[domain] = {
                    "previous": prev_score,
                    "current": current_score,
                    "delta": delta,
                    "threshold": threshold,
                }
        return regressions

    def snapshot(self) -> CompetencyVector:
        """Create a deep copy."""
        return CompetencyVector(
            scores=dict(self.scores),
            ema_alpha=self.ema_alpha,
            raw_history={k: list(v) for k, v in self.raw_history.items()},
            raw_answers={k: list(v) for k, v in self.raw_answers.items()},
            update_count=self.update_count,
        )

    def save(self, path: str | Path) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "scores": self.scores,
            "ema_alpha": self.ema_alpha,
            "update_count": self.update_count,
            "raw_history": {
                k: v[-100:] for k, v in self.raw_history.items()
            },
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CompetencyVector:
        """Load from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            scores=data.get("scores", {}),
            ema_alpha=data.get("ema_alpha", 0.3),
            raw_history=data.get("raw_history", {}),
            update_count=data.get("update_count", 0),
        )

    def to_summary_string(self) -> str:
        """Human-readable summary."""
        if not self.scores:
            return "CompetencyVector: (empty)"
        lines = ["CompetencyVector:"]
        for domain, score in sorted(self.scores.items(), key=lambda x: -x[1]):
            trend = self.get_trend(domain)
            arrow = "↑" if trend > 0.005 else "↓" if trend < -0.005 else "→"
            lines.append(f"  {domain}: {score:.1%} {arrow}")
        lines.append(f"  Overall: {self.get_overall_score():.1%}")
        return "\n".join(lines)


class CompetencyTracker:
    """
    Tracks competency over time and provides analytics.

    Stores a history of CompetencyVectors and computes
    higher-order metrics like convergence rate and domain divergence.
    """

    def __init__(self, max_history: int = 1000):
        self.history: list[CompetencyVector] = []
        self.max_history = max_history

    def record(self, competency: CompetencyVector) -> None:
        """Record a new competency measurement."""
        self.history.append(competency.snapshot())
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    @property
    def current(self) -> CompetencyVector | None:
        return self.history[-1] if self.history else None

    @property
    def length(self) -> int:
        return len(self.history)

    def overall_trend(self, window: int = 20) -> float:
        """Overall improvement trend across all domains."""
        if len(self.history) < 3:
            return 0.0

        recent = self.history[-window:]
        overall_scores = [cv.get_overall_score() for cv in recent]
        return _linear_slope(overall_scores)

    def domain_trend(self, domain: str, window: int = 20) -> float:
        """Trend for a specific domain."""
        if len(self.history) < 3:
            return 0.0

        recent = self.history[-window:]
        scores = [cv[domain] for cv in recent if domain in cv.scores]
        if len(scores) < 3:
            return 0.0
        return _linear_slope(scores)

    def is_converging(
        self,
        window: int = 100,
        delta_min: float = 0.001,
    ) -> bool:
        """
        Check if the model has converged (improvement rate below threshold).

        CONVERGED ⟺ (1/W) Σ(S(θ_t) - S(θ_{t-1})) < δ_min for W = window
        """
        if len(self.history) < window:
            return False

        recent = self.history[-window:]
        deltas = []
        for i in range(1, len(recent)):
            prev_overall = recent[i - 1].get_overall_score()
            curr_overall = recent[i].get_overall_score()
            deltas.append(curr_overall - prev_overall)

        avg_delta = sum(deltas) / len(deltas)
        return avg_delta < delta_min

    def diversity_divergence(self, window: int = 10) -> tuple[list[str], list[str]]:
        """
        Detect domains improving while others decline.

        Returns:
            Tuple of (improving_domains, degrading_domains).
        """
        if len(self.history) < window:
            return [], []

        improving = []
        degrading = []

        recent = self.history[-window:]
        all_domains = recent[-1].domains if recent else []

        for domain in all_domains:
            scores = [cv[domain] for cv in recent if domain in cv.scores]
            if len(scores) < 3:
                continue
            slope = _linear_slope(scores)
            if slope > 0.005:
                improving.append(domain)
            elif slope < -0.005:
                degrading.append(domain)

        return improving, degrading

    def get_plateau_domains(
        self,
        window: int = 50,
        slope_threshold: float = 1e-4,
    ) -> list[str]:
        """Find domains that have plateaued."""
        if len(self.history) < window:
            return []

        plateaued = []
        recent = self.history[-window:]
        all_domains = recent[-1].domains if recent else []

        for domain in all_domains:
            scores = [cv[domain] for cv in recent if domain in cv.scores]
            if len(scores) < 10:
                continue
            slope = _linear_slope(scores)
            if abs(slope) < slope_threshold:
                plateaued.append(domain)

        return plateaued

    def save(self, path: str | Path) -> None:
        """Save tracker history to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "history_length": len(self.history),
            "recent_history": [
                cv.as_dict() for cv in self.history[-100:]
            ],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _linear_slope(values: list[float]) -> float:
    """Compute linear regression slope via least squares."""
    n = len(values)
    if n < 2:
        return 0.0

    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator
