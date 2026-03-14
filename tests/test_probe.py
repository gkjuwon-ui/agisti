"""
Tests for agisti.probe — competency, weakness, active_prober.
"""

from __future__ import annotations

import time

import pytest

from agisti.types import (
    AnswerType,
    Problem,
    Solution,
    ErrorReport,
    FailedProblem,
    Probe,
    WeaknessReport,
)
from agisti.probe.competency import CompetencyVector, CompetencyTracker
from agisti.probe.weakness import (
    FailureCategory,
    DomainWeakness,
    WeaknessAnalyzer,
    FailurePatternDetector,
)
from agisti.probe.active_prober import (
    ProbeBank,
    ActiveProber,
    ProbeScheduler,
    ProbeBankBuilder,
)


# ─── CompetencyVector Tests ───────────────────────

class TestCompetencyVector:
    """Tests for CompetencyVector with EMA."""

    def test_creation(self):
        cv = CompetencyVector(domains=["math", "logic", "science"])
        assert len(cv.domains) == 3

    def test_update_score(self):
        cv = CompetencyVector(domains=["math"])
        # Update multiple times
        for _ in range(10):
            cv.update("math", 0.8)
        score = cv.get("math")
        assert 0.7 < score < 0.9

    def test_ema_decay(self):
        cv = CompetencyVector(domains=["math"], alpha=0.3)
        # Start with high scores
        for _ in range(10):
            cv.update("math", 1.0)
        high = cv.get("math")
        # Then low scores
        for _ in range(10):
            cv.update("math", 0.0)
        low = cv.get("math")
        assert low < high

    def test_unknown_domain_raises(self):
        cv = CompetencyVector(domains=["math"])
        with pytest.raises((KeyError, ValueError)):
            cv.get("unknown_domain")

    def test_all_scores(self):
        cv = CompetencyVector(domains=["math", "logic"])
        cv.update("math", 0.8)
        cv.update("logic", 0.5)
        scores = cv.all_scores()
        assert "math" in scores
        assert "logic" in scores

    def test_weakest_domain(self):
        cv = CompetencyVector(domains=["math", "logic", "coding"])
        cv.update("math", 0.9)
        cv.update("logic", 0.3)
        cv.update("coding", 0.7)
        weakest = cv.weakest()
        assert weakest == "logic"

    def test_strongest_domain(self):
        cv = CompetencyVector(domains=["math", "logic"])
        cv.update("math", 0.9)
        cv.update("logic", 0.3)
        strongest = cv.strongest()
        assert strongest == "math"


class TestCompetencyTracker:
    """Tests for CompetencyTracker."""

    def test_creation(self):
        tracker = CompetencyTracker(domains=["math", "logic"])
        assert tracker is not None

    def test_record_and_retrieve(self):
        tracker = CompetencyTracker(domains=["math"])
        for i in range(5):
            tracker.record("math", 0.5 + 0.05 * i)
        history = tracker.get_history("math")
        assert len(history) == 5

    def test_improvement_rate(self):
        tracker = CompetencyTracker(domains=["math"])
        for i in range(20):
            tracker.record("math", 0.3 + 0.02 * i)
        rate = tracker.improvement_rate("math")
        assert rate > 0

    def test_current_scores(self):
        tracker = CompetencyTracker(domains=["math", "logic"])
        tracker.record("math", 0.8)
        tracker.record("logic", 0.6)
        current = tracker.current_scores()
        assert "math" in current
        assert "logic" in current


# ─── WeaknessAnalyzer Tests ──────────────────────

class TestFailureCategory:
    """Tests for FailureCategory enum."""

    def test_enum_members(self):
        assert hasattr(FailureCategory, "CONCEPTUAL")
        assert hasattr(FailureCategory, "COMPUTATIONAL")


class TestDomainWeakness:
    """Tests for DomainWeakness tracking."""

    def test_creation(self):
        dw = DomainWeakness(
            domain="math",
            categories={FailureCategory.CONCEPTUAL: 5},
            total_failures=5,
            failure_rate=0.5,
        )
        assert dw.domain == "math"
        assert dw.failure_rate == 0.5

    def test_dominant_category(self):
        dw = DomainWeakness(
            domain="math",
            categories={
                FailureCategory.CONCEPTUAL: 10,
                FailureCategory.COMPUTATIONAL: 2,
            },
            total_failures=12,
            failure_rate=0.6,
        )
        dominant = dw.dominant_category
        assert dominant == FailureCategory.CONCEPTUAL


class TestWeaknessAnalyzer:
    """Tests for WeaknessAnalyzer."""

    def _make_failed_problem(
        self,
        domain: str = "math",
        difficulty: float = 0.5,
    ) -> FailedProblem:
        return FailedProblem(
            problem=Problem(
                question="What is 2+2?",
                answer="4",
                answer_type=AnswerType.NUMERIC_RANGE,
                domain=domain,
                difficulty=int(difficulty * 5) or 1,
            ),
            original_solution=Solution(
                problem_id="test",
                answer="5",
                chain_of_thought="",
                tokens_generated=1,
                generation_time_seconds=0.0,
            ),
            domain=domain,
            ground_truth="4",
            answer_type=AnswerType.NUMERIC_RANGE,
        )

    def test_creation(self):
        analyzer = WeaknessAnalyzer()
        assert analyzer is not None

    def test_analyze_failures(self):
        analyzer = WeaknessAnalyzer()
        failures = [self._make_failed_problem() for _ in range(10)]
        report = analyzer.analyze(failures)
        assert isinstance(report, WeaknessReport)

    def test_analyze_empty(self):
        analyzer = WeaknessAnalyzer()
        report = analyzer.analyze([])
        assert isinstance(report, WeaknessReport)
        assert report.total_failures == 0

    def test_multi_domain_analysis(self):
        analyzer = WeaknessAnalyzer()
        failures = [
            self._make_failed_problem(domain="math"),
            self._make_failed_problem(domain="math"),
            self._make_failed_problem(domain="logic"),
        ]
        report = analyzer.analyze(failures)
        assert report.total_failures == 3

    def test_difficulty_correlation(self):
        analyzer = WeaknessAnalyzer()
        # Easy problems failed → might indicate conceptual issues
        easy_failures = [
            self._make_failed_problem(difficulty=0.1)
            for _ in range(5)
        ]
        # Hard problems failed → might be normal
        hard_failures = [
            self._make_failed_problem(difficulty=0.9)
            for _ in range(5)
        ]
        report_easy = analyzer.analyze(easy_failures)
        report_hard = analyzer.analyze(hard_failures)
        # Both should be valid reports
        assert report_easy.total_failures == 5
        assert report_hard.total_failures == 5


class TestFailurePatternDetector:
    """Tests for FailurePatternDetector."""

    def test_creation(self):
        detector = FailurePatternDetector()
        assert detector is not None

    def test_detect_patterns(self):
        detector = FailurePatternDetector()
        failures = [
            FailedProblem(
                problem=Problem(
                    question=f"Q{i}",
                    answer=str(i),
                    answer_type=AnswerType.NUMERIC_RANGE,
                    domain="math",
                    difficulty=1,
                ),
                original_solution=Solution(
                    problem_id=f"test_{i}",
                    answer="wrong",
                    chain_of_thought="",
                    tokens_generated=1,
                    generation_time_seconds=0.0,
                ),
                domain="math",
                ground_truth=str(i),
                answer_type=AnswerType.NUMERIC_RANGE,
            )
            for i in range(20)
        ]
        patterns = detector.detect(failures)
        assert isinstance(patterns, list)


# ─── ProbeBank Tests ──────────────────────────────

class TestProbeBank:
    """Tests for ProbeBank."""

    def _make_probes(self, n: int = 10) -> list[Probe]:
        return [
            Probe(
                problem=Problem(
                    question=f"What is {i}+{i}?",
                    answer=str(2 * i),
                    answer_type=AnswerType.NUMERIC,
                    domain="math",
                    difficulty=0.5,
                ),
                expected_answer=str(2 * i),
                domain="math",
                difficulty=0.5,
            )
            for i in range(n)
        ]

    def test_creation(self):
        bank = ProbeBank(self._make_probes())
        assert len(bank) == 10

    def test_sample(self):
        bank = ProbeBank(self._make_probes(20))
        sample = bank.sample(5)
        assert len(sample) == 5

    def test_sample_by_domain(self):
        probes = self._make_probes(10)
        # Add different domain probes
        logic_probes = [
            Probe(
                problem=Problem(
                    question=f"Is {i} > 0?",
                    answer="yes",
                    answer_type=AnswerType.MCQ,
                    domain="logic",
                    difficulty=0.3,
                ),
                expected_answer="yes",
                domain="logic",
                difficulty=0.3,
            )
            for i in range(5)
        ]
        bank = ProbeBank(probes + logic_probes)
        math_probes = bank.sample_by_domain("math", 3)
        assert len(math_probes) == 3
        assert all(p.domain == "math" for p in math_probes)

    def test_domains(self):
        probes = self._make_probes(5)
        bank = ProbeBank(probes)
        domains = bank.domains
        assert "math" in domains

    def test_filter_by_difficulty(self):
        probes = [
            Probe(
                problem=Problem(
                    question=f"Q{i}",
                    answer=str(i),
                    answer_type=AnswerType.NUMERIC,
                    domain="math",
                    difficulty=0.1 * i,
                ),
                expected_answer=str(i),
                domain="math",
                difficulty=0.1 * i,
            )
            for i in range(10)
        ]
        bank = ProbeBank(probes)
        hard = bank.filter_by_difficulty(min_diff=0.5)
        assert all(p.difficulty >= 0.5 for p in hard)


# ─── ActiveProber Tests ───────────────────────────

class TestActiveProber:
    """Tests for ActiveProber."""

    def test_creation(self):
        probes = [
            Probe(
                problem=Problem(
                    question="What is 1+1?",
                    answer="2",
                    answer_type=AnswerType.NUMERIC,
                    domain="math",
                    difficulty=0.1,
                ),
                expected_answer="2",
                domain="math",
                difficulty=0.1,
            )
        ]
        bank = ProbeBank(probes)
        prober = ActiveProber(bank)
        assert prober is not None

    def test_select_probes(self):
        probes = [
            Probe(
                problem=Problem(
                    question=f"Q{i}",
                    answer=str(i),
                    answer_type=AnswerType.NUMERIC,
                    domain="math" if i % 2 == 0 else "logic",
                    difficulty=0.5,
                ),
                expected_answer=str(i),
                domain="math" if i % 2 == 0 else "logic",
                difficulty=0.5,
            )
            for i in range(20)
        ]
        bank = ProbeBank(probes)
        prober = ActiveProber(bank)
        selected = prober.select(n=5)
        assert len(selected) == 5


class TestProbeScheduler:
    """Tests for ProbeScheduler."""

    def test_creation(self):
        sched = ProbeScheduler(check_interval=10)
        assert sched is not None

    def test_should_probe(self):
        sched = ProbeScheduler(check_interval=5)
        assert sched.should_probe(0) is True
        assert sched.should_probe(1) is False
        assert sched.should_probe(5) is True
        assert sched.should_probe(10) is True

    def test_adaptive_interval(self):
        sched = ProbeScheduler(check_interval=10, adaptive=True)
        # Should adapt based on failure rate
        sched.update_failure_rate(0.8)  # High failure rate
        # Should probe more frequently
        assert sched.effective_interval <= 10


class TestProbeBankBuilder:
    """Tests for ProbeBankBuilder."""

    def test_build_from_problems(self):
        problems = [
            Problem(
                question=f"Q{i}",
                answer=str(i),
                answer_type=AnswerType.NUMERIC,
                domain="math",
                difficulty=0.5,
            )
            for i in range(10)
        ]
        builder = ProbeBankBuilder()
        bank = builder.from_problems(problems)
        assert isinstance(bank, ProbeBank)
        assert len(bank) == 10

    def test_build_balanced(self):
        problems = [
            Problem(
                question=f"Q{i}",
                answer=str(i),
                answer_type=AnswerType.NUMERIC,
                domain="math" if i < 8 else "logic",
                difficulty=0.5,
            )
            for i in range(10)
        ]
        builder = ProbeBankBuilder()
        bank = builder.from_problems(problems, balanced=True)
        assert isinstance(bank, ProbeBank)
