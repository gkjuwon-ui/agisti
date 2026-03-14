"""
WeaknessReport — analyzes failure patterns to guide targeted surgery.

Clusters errors by domain, layer, type, and severity to produce
actionable weakness reports that drive problem generation and surgery focus.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agisti.types import (
    AnswerType,
    ErrorReport,
    FailedProblem,
    Problem,
    SurgeryType,
    WeaknessReport as WeaknessReportType,
)
from agisti.probe.competency import CompetencyVector

logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    """High-level error taxonomy."""
    FACTUAL = "factual"
    REASONING = "reasoning"
    INSTRUCTION_FOLLOWING = "instruction_following"
    FORMAT = "format"
    KNOWLEDGE_GAP = "knowledge_gap"
    HALLUCINATION = "hallucination"
    COMPUTATION = "computation"
    UNCLEAR = "unclear"


@dataclass
class DomainWeakness:
    """Weakness analysis for a single domain."""
    domain: str
    score: float
    trend: float  # negative = worsening
    failure_count: int
    total_evaluated: int
    failure_categories: dict[str, int] = field(default_factory=dict)
    example_failures: list[FailedProblem] = field(default_factory=list)
    recommended_surgery: SurgeryType = SurgeryType.MICRO

    @property
    def failure_rate(self) -> float:
        if self.total_evaluated == 0:
            return 0.0
        return self.failure_count / self.total_evaluated

    @property
    def severity(self) -> float:
        """Combined severity: low score + negative trend + high failure rate."""
        score_penalty = max(0.0, 0.5 - self.score) * 2.0  # 0-1 scale
        trend_penalty = max(0.0, -self.trend) * 10.0  # amplify
        rate_penalty = self.failure_rate
        return (score_penalty + trend_penalty + rate_penalty) / 3.0


@dataclass
class LayerWeakness:
    """Analysis of which layers are most associated with failures."""
    layer_name: str
    activation_variance: float
    hit_count: int  # times this layer appeared in failure activations
    mean_magnitude: float

    @property
    def severity(self) -> float:
        return self.activation_variance * self.hit_count


class WeaknessAnalyzer:
    """
    Analyzes failed problems and competency data to produce WeaknessReports.

    The WeaknessReport drives:
    1. Problem generation focus (generate more where we're weak)
    2. Surgery type selection (micro vs macro vs arch)
    3. Frozen zone discovery (don't freeze failing layers)
    """

    def __init__(
        self,
        max_examples_per_domain: int = 10,
        weakness_threshold: float = 0.7,
        regression_threshold: float = -0.005,
    ):
        self.max_examples = max_examples_per_domain
        self.weakness_threshold = weakness_threshold
        self.regression_threshold = regression_threshold

    def analyze(
        self,
        competency: CompetencyVector,
        failed_problems: list[FailedProblem],
        recent_errors: list[ErrorReport] | None = None,
    ) -> WeaknessReportType:
        """
        Produce a comprehensive weakness report.

        Args:
            competency: Current competency vector.
            failed_problems: List of problems the model got wrong.
            recent_errors: Optional catched errors from latest iteration.

        Returns:
            WeaknessReport with prioritized domains and recommendations.
        """
        # Group failures by domain
        domain_failures = self._group_by_domain(failed_problems)

        # Analyze each domain
        domain_weaknesses = []
        for domain, failures in domain_failures.items():
            dw = self._analyze_domain(domain, failures, competency)
            domain_weaknesses.append(dw)

        # Sort by severity (worst first)
        domain_weaknesses.sort(key=lambda d: -d.severity)

        # Build priority domain list
        weak_domains = [
            dw.domain for dw in domain_weaknesses
            if dw.score < self.weakness_threshold or dw.trend < self.regression_threshold
        ]

        # Determine recommended surgery type from worst domain
        recommended_type = self._recommend_surgery_type(domain_weaknesses)

        # Build target layers from failure analysis
        target_layers = self._identify_target_layers(failed_problems)

        return WeaknessReportType(
            weak_domains=weak_domains,
            target_layers=target_layers,
            recommended_type=recommended_type,
        )

    def analyze_detailed(
        self,
        competency: CompetencyVector,
        failed_problems: list[FailedProblem],
    ) -> list[DomainWeakness]:
        """Detailed per-domain analysis for reporting."""
        domain_failures = self._group_by_domain(failed_problems)
        weaknesses = []
        for domain, failures in domain_failures.items():
            dw = self._analyze_domain(domain, failures, competency)
            weaknesses.append(dw)
        weaknesses.sort(key=lambda d: -d.severity)
        return weaknesses

    def _group_by_domain(
        self, failed: list[FailedProblem],
    ) -> dict[str, list[FailedProblem]]:
        """Group failed problems by their domain."""
        grouped: dict[str, list[FailedProblem]] = defaultdict(list)
        for fp in failed:
            domain = fp.problem.domain
            grouped[domain].append(fp)
        return dict(grouped)

    def _analyze_domain(
        self,
        domain: str,
        failures: list[FailedProblem],
        competency: CompetencyVector,
    ) -> DomainWeakness:
        """Analyze a single domain's weaknesses."""
        score = competency[domain]
        trend = competency.get_trend(domain)

        # Categorize failures
        categories: Counter[str] = Counter()
        for fp in failures:
            cat = self._categorize_failure(fp)
            categories[cat.value] += 1

        # Select diverse examples
        examples = self._select_diverse_examples(failures)

        # Recommend surgery type
        recommended = self._domain_surgery_recommendation(
            score, trend, len(failures), categories,
        )

        return DomainWeakness(
            domain=domain,
            score=score,
            trend=trend,
            failure_count=len(failures),
            total_evaluated=len(failures),  # approximate
            failure_categories=dict(categories),
            example_failures=examples,
            recommended_surgery=recommended,
        )

    def _categorize_failure(self, fp: FailedProblem) -> FailureCategory:
        """Categorize a failure based on the error pattern."""
        error = fp.error_report
        if error is None:
            return FailureCategory.UNCLEAR

        msg = error.message.lower()

        # Pattern matching on error messages
        if any(w in msg for w in ["hallucin", "fabricat", "made up"]):
            return FailureCategory.HALLUCINATION
        if any(w in msg for w in ["wrong fact", "incorrect", "outdated"]):
            return FailureCategory.FACTUAL
        if any(w in msg for w in ["logic", "reason", "deduct", "step"]):
            return FailureCategory.REASONING
        if any(w in msg for w in ["format", "json", "parse", "syntax"]):
            return FailureCategory.FORMAT
        if any(w in msg for w in ["instruct", "follow", "asked for", "directive"]):
            return FailureCategory.INSTRUCTION_FOLLOWING
        if any(w in msg for w in ["calculat", "math", "comput", "arithmet"]):
            return FailureCategory.COMPUTATION
        if any(w in msg for w in ["know", "unfamiliar", "never seen"]):
            return FailureCategory.KNOWLEDGE_GAP

        return FailureCategory.UNCLEAR

    def _select_diverse_examples(
        self, failures: list[FailedProblem],
    ) -> list[FailedProblem]:
        """Select diverse failure examples covering different error types."""
        if len(failures) <= self.max_examples:
            return list(failures)

        # Categorize all
        by_category: dict[FailureCategory, list[FailedProblem]] = defaultdict(list)
        for fp in failures:
            cat = self._categorize_failure(fp)
            by_category[cat].append(fp)

        # Round-robin selection from each category
        selected: list[FailedProblem] = []
        categories = list(by_category.keys())
        idx_per_cat = {cat: 0 for cat in categories}

        while len(selected) < self.max_examples and categories:
            for cat in list(categories):
                if len(selected) >= self.max_examples:
                    break
                idx = idx_per_cat[cat]
                items = by_category[cat]
                if idx < len(items):
                    selected.append(items[idx])
                    idx_per_cat[cat] = idx + 1
                else:
                    categories.remove(cat)

        return selected

    def _domain_surgery_recommendation(
        self,
        score: float,
        trend: float,
        failure_count: int,
        categories: Counter[str],
    ) -> SurgeryType:
        """Recommend surgery type based on domain analysis."""
        # Very low score + many failures = MACRO surgery needed
        if score < 0.3 and failure_count > 20:
            return SurgeryType.MACRO

        # Sharp decline suggests macro intervention
        if trend < -0.02:
            return SurgeryType.MACRO

        # Moderate weakness with specific failure patterns = MICRO
        if score < 0.6:
            return SurgeryType.MICRO

        # Mostly, hallucination or knowledge gaps may indicate
        # more fundamental issues
        halluc_count = categories.get(FailureCategory.HALLUCINATION.value, 0)
        knowledge_count = categories.get(FailureCategory.KNOWLEDGE_GAP.value, 0)
        if (halluc_count + knowledge_count) > failure_count * 0.5:
            return SurgeryType.MACRO

        return SurgeryType.MICRO

    def _recommend_surgery_type(
        self, weaknesses: list[DomainWeakness],
    ) -> SurgeryType:
        """Recommend overall surgery type from domain analyses."""
        if not weaknesses:
            return SurgeryType.MICRO

        # Count recommendations
        macro_count = sum(
            1 for dw in weaknesses
            if dw.recommended_surgery == SurgeryType.MACRO
        )
        if macro_count > len(weaknesses) * 0.5:
            return SurgeryType.MACRO

        # Check if top weakness is very severe
        worst = weaknesses[0]
        if worst.severity > 0.6:
            return SurgeryType.MACRO

        return SurgeryType.MICRO

    def _identify_target_layers(
        self, failed: list[FailedProblem],
    ) -> list[str]:
        """
        Identify which layers should be targeted for surgery.

        Uses the error patterns and problem types to determine
        which layers are most likely responsible.

        In a real deployment, this would use activation tracing data.
        Here we use heuristics based on domain type.
        """
        domain_counter: Counter[str] = Counter()
        for fp in failed:
            domain_counter[fp.problem.domain] += 1

        target_layers = []
        for domain, count in domain_counter.most_common(5):
            # Heuristic: reasoning failures → later layers
            # factual failures → embedding + early layers
            # format failures → final layers
            if "reasoning" in domain.lower() or "math" in domain.lower():
                target_layers.extend([
                    f"model.layers.{i}" for i in range(-6, 0)
                ])
            elif "knowledge" in domain.lower() or "factual" in domain.lower():
                target_layers.extend([
                    f"model.layers.{i}" for i in range(0, 6)
                ])
            else:
                target_layers.extend([
                    f"model.layers.{i}" for i in range(-3, 0)
                ])

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for layer in target_layers:
            if layer not in seen:
                seen.add(layer)
                unique.append(layer)

        return unique


class FailurePatternDetector:
    """
    Detects recurring failure patterns across iterations.

    Looks for systematic failures that might indicate:
    - Catastrophic patterns (same problem failing repeatedly)
    - Regression patterns (previously solved problems failing again)
    - Mode collapse (all failures in one category)
    """

    def __init__(self, history_window: int = 50):
        self.history_window = history_window
        self.failure_history: list[list[FailedProblem]] = []
        self.problem_failure_counts: Counter[str] = Counter()

    def record_failures(self, failures: list[FailedProblem]) -> None:
        """Record failures from one iteration."""
        self.failure_history.append(failures)
        if len(self.failure_history) > self.history_window:
            self.failure_history = self.failure_history[-self.history_window:]

        for fp in failures:
            self.problem_failure_counts[fp.problem.id] += 1

    def detect_persistent_failures(
        self, min_repeats: int = 3,
    ) -> list[tuple[str, int]]:
        """Find problems that have failed multiple times."""
        persistent = [
            (pid, count)
            for pid, count in self.problem_failure_counts.most_common()
            if count >= min_repeats
        ]
        return persistent

    def detect_regression(
        self,
        current_failures: list[FailedProblem],
        previously_solved: set[str],
    ) -> list[FailedProblem]:
        """Find failures that were previously solved."""
        regressed = []
        for fp in current_failures:
            if fp.problem.id in previously_solved:
                regressed.append(fp)
        return regressed

    def detect_mode_collapse(
        self,
        failures: list[FailedProblem],
        dominance_threshold: float = 0.7,
    ) -> FailureCategory | None:
        """
        Detect if failures are concentrated in one category.

        If >70% of failures are in one category, something systemic
        is wrong with that capability.
        """
        if len(failures) < 5:
            return None

        analyzer = WeaknessAnalyzer()
        categories: Counter[str] = Counter()
        for fp in failures:
            cat = analyzer._categorize_failure(fp)
            categories[cat.value] += 1

        most_common_cat, most_common_count = categories.most_common(1)[0]
        if most_common_count / len(failures) > dominance_threshold:
            return FailureCategory(most_common_cat)

        return None

    def get_failure_velocity(self) -> float:
        """Rate of change of failure count across recent iterations."""
        if len(self.failure_history) < 3:
            return 0.0

        counts = [len(fs) for fs in self.failure_history[-20:]]
        # Simple linear slope
        n = len(counts)
        x_mean = (n - 1) / 2.0
        y_mean = sum(counts) / n
        num = sum((i - x_mean) * (c - y_mean) for i, c in enumerate(counts))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if abs(den) < 1e-12:
            return 0.0
        return num / den
