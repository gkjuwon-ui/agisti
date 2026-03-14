"""
Tests for agisti.benchmark — quick_bench, mcnemar, full_bench, external_validator.
"""

from __future__ import annotations

import math

import pytest

from agisti.types import (
    AnswerType,
    Problem,
    QuickBenchResult,
    FullBenchResult,
)
from agisti.config import QuickBenchConfig, FullBenchConfig
from agisti.benchmark.quick_bench import QuickBenchSuite, QuickBench
from agisti.benchmark.mcnemar import mcnemar_test
from agisti.benchmark.full_bench import FullBench, BenchmarkOrchestrator
from agisti.benchmark.external_validator import DynamicExternalValidator


# ─── QuickBenchSuite Tests ────────────────────────

class TestQuickBenchSuite:
    """Tests for individual quick benchmark suite."""

    def _make_problems(self, n: int = 20) -> list[Problem]:
        return [
            Problem(
                question=f"What is {i}*2?",
                answer=str(i * 2),
                answer_type=AnswerType.NUMERIC,
                domain="math",
                difficulty=0.5,
            )
            for i in range(n)
        ]

    def test_creation(self):
        suite = QuickBenchSuite(
            name="basic_math",
            domain="math",
            problems=self._make_problems(),
        )
        assert suite.name == "basic_math"
        assert len(suite.problems) == 20

    def test_sample(self):
        suite = QuickBenchSuite(
            name="basic_math",
            domain="math",
            problems=self._make_problems(50),
        )
        sample = suite.sample(10)
        assert len(sample) == 10

    def test_empty_suite(self):
        suite = QuickBenchSuite(
            name="empty",
            domain="test",
            problems=[],
        )
        assert len(suite.problems) == 0


class TestQuickBench:
    """Tests for QuickBench orchestration."""

    def test_creation(self):
        qb = QuickBench(QuickBenchConfig())
        assert qb is not None

    def test_add_suite(self):
        qb = QuickBench(QuickBenchConfig())
        problems = [
            Problem(
                question="What is 1+1?",
                answer="2",
                answer_type=AnswerType.NUMERIC,
                domain="math",
                difficulty=0.1,
            )
        ]
        qb.add_suite(QuickBenchSuite("test", "math", problems))
        assert len(qb.suites) == 1

    def test_result_type(self):
        result = QuickBenchResult(
            scores={"math": 0.8, "logic": 0.6},
            per_problem={},
            total_problems=20,
            wall_time_seconds=1.0,
        )
        assert result.accuracy == pytest.approx(0.7)

    def test_result_domain_breakdown(self):
        result = QuickBenchResult(
            scores={"math": 0.9, "logic": 0.5, "coding": 0.3},
            per_problem={},
            total_problems=30,
            wall_time_seconds=2.0,
        )
        breakdown = result.domain_breakdown
        assert "math" in breakdown
        assert breakdown["math"] == pytest.approx(0.9)


# ─── McNemar Test Tests ──────────────────────────

class TestMcNemarTest:
    """Tests for McNemar's statistical test."""

    def test_identical_models(self):
        # Both models get same answers → no significant difference
        baseline = [True] * 50 + [False] * 50
        candidate = [True] * 50 + [False] * 50
        stat, p_value = mcnemar_test(baseline, candidate)
        assert p_value > 0.05  # Not significant

    def test_clearly_different_models(self):
        # One model much better than other
        baseline = [False] * 100
        candidate = [True] * 100
        stat, p_value = mcnemar_test(baseline, candidate)
        assert p_value < 0.05  # Significant difference

    def test_symmetric(self):
        # McNemar should give same result regardless of order
        a = [True, False, True, True, False] * 20
        b = [True, True, False, True, True] * 20
        stat1, p1 = mcnemar_test(a, b)
        stat2, p2 = mcnemar_test(b, a)
        assert stat1 == pytest.approx(stat2)
        assert p1 == pytest.approx(p2)

    def test_equal_length_required(self):
        with pytest.raises((ValueError, AssertionError)):
            mcnemar_test([True], [True, False])

    def test_returns_finite_values(self):
        a = [True, False, True, False, True] * 10
        b = [False, True, True, False, False] * 10
        stat, p = mcnemar_test(a, b)
        assert math.isfinite(stat)
        assert 0.0 <= p <= 1.0

    def test_small_sample(self):
        a = [True, False]
        b = [False, True]
        stat, p = mcnemar_test(a, b)
        assert math.isfinite(stat)

    def test_all_agree(self):
        a = [True] * 20
        b = [True] * 20
        stat, p = mcnemar_test(a, b)
        # When all agree, test statistic should indicate no difference
        assert p >= 0.0


# ─── FullBench Tests ──────────────────────────────

class TestFullBench:
    """Tests for full benchmark suite."""

    def test_creation(self):
        fb = FullBench(FullBenchConfig())
        assert fb is not None

    def test_result_type(self):
        result = FullBenchResult(
            scores={"GSM8K": 0.78, "MATH": 0.56},
            metadata={
                "model": "qwen-0.5b",
                "total_time": 120.0,
            },
            wall_time_seconds=120.0,
        )
        assert result.scores["GSM8K"] == pytest.approx(0.78)
        assert "model" in result.metadata


class TestBenchmarkOrchestrator:
    """Tests for benchmark orchestration."""

    def test_creation(self):
        orch = BenchmarkOrchestrator()
        assert orch is not None

    def test_register_benchmark(self):
        orch = BenchmarkOrchestrator()
        config = FullBenchConfig()
        orch.register("gsm8k", FullBench(config))
        assert "gsm8k" in orch.registered

    def test_list_benchmarks(self):
        orch = BenchmarkOrchestrator()
        assert isinstance(orch.registered, (list, dict, set))


# ─── DynamicExternalValidator Tests ───────────────

class TestDynamicExternalValidator:
    """Tests for external validation with dynamic fetchers."""

    def test_creation(self):
        validator = DynamicExternalValidator()
        assert validator is not None

    def test_register_fetcher(self):
        validator = DynamicExternalValidator()

        def dummy_fetcher(model_path: str) -> dict:
            return {"score": 0.5}

        validator.register_fetcher("test_bench", dummy_fetcher)
        assert "test_bench" in validator.fetchers

    def test_validate_with_fetcher(self):
        validator = DynamicExternalValidator()

        def dummy_fetcher(model_path: str) -> dict:
            return {"score": 0.75}

        validator.register_fetcher("test_bench", dummy_fetcher)
        result = validator.validate("test_bench", "/fake/model/path")
        assert isinstance(result, dict)
        assert result["score"] == pytest.approx(0.75)

    def test_validate_unknown_benchmark(self):
        validator = DynamicExternalValidator()
        with pytest.raises((KeyError, ValueError)):
            validator.validate("nonexistent", "/fake/path")

    def test_supported_benchmarks(self):
        validator = DynamicExternalValidator()
        validator.register_fetcher("a", lambda m: {})
        validator.register_fetcher("b", lambda m: {})
        assert len(validator.supported_benchmarks) == 2

    def test_validate_all(self):
        validator = DynamicExternalValidator()
        validator.register_fetcher("bench1", lambda m: {"score": 0.8})
        validator.register_fetcher("bench2", lambda m: {"score": 0.6})
        results = validator.validate_all("/fake/model")
        assert len(results) == 2
        assert "bench1" in results
        assert "bench2" in results
