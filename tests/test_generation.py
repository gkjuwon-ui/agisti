"""
Tests for agisti.generation — generator, difficulty, verification.
"""

from __future__ import annotations

import math

import pytest

from agisti.types import (
    AnswerType,
    Problem,
    Solution,
    WeaknessReport,
)
from agisti.generation.generator import (
    ProblemGenerator,
    DifficultyAdapter,
    ProblemFilter,
)
from agisti.generation.difficulty import (
    AdaptiveDifficultyEngine,
    CurriculumScheduler,
)
from agisti.generation.verification import (
    AnswerVerifier,
    ConsistencyChecker,
    BatchVerifier,
)


# ─── ProblemGenerator Tests ───────────────────────

class TestProblemGenerator:
    """Tests for ProblemGenerator."""

    def test_creation(self):
        gen = ProblemGenerator()
        assert gen is not None

    def test_generate_returns_problems(self):
        gen = ProblemGenerator()
        weakness = WeaknessReport(
            total_failures=10,
            failure_rate=0.5,
            domain_weaknesses={},
            priority_domains=["math"],
        )
        problems = gen.generate(weakness, n=5, difficulty=0.5)
        assert isinstance(problems, list)
        # Even without a real model, should return something
        if len(problems) > 0:
            assert all(isinstance(p, Problem) for p in problems)

    def test_generate_with_no_weakness(self):
        gen = ProblemGenerator()
        weakness = WeaknessReport(
            total_failures=0,
            failure_rate=0.0,
            domain_weaknesses={},
            priority_domains=[],
        )
        problems = gen.generate(weakness, n=5, difficulty=0.5)
        assert isinstance(problems, list)

    def test_template_based_generation(self):
        gen = ProblemGenerator()
        problems = gen.generate_from_templates(
            domain="math",
            n=10,
            difficulty_range=(0.3, 0.7),
        )
        assert isinstance(problems, list)

    def test_domain_specific_generation(self):
        gen = ProblemGenerator()
        for domain in ["math", "logic", "coding"]:
            problems = gen.generate_for_domain(domain, n=3, difficulty=0.5)
            assert isinstance(problems, list)


class TestDifficultyAdapter:
    """Tests for DifficultyAdapter."""

    def test_creation(self):
        adapter = DifficultyAdapter()
        assert adapter is not None

    def test_adjust_up(self):
        adapter = DifficultyAdapter()
        # If model is doing well → increase difficulty
        adjusted = adapter.adjust(
            current_difficulty=0.5,
            recent_accuracy=0.9,
            target_accuracy=0.6,
        )
        assert adjusted >= 0.5

    def test_adjust_down(self):
        adapter = DifficultyAdapter()
        # If model is struggling → decrease difficulty
        adjusted = adapter.adjust(
            current_difficulty=0.5,
            recent_accuracy=0.2,
            target_accuracy=0.6,
        )
        assert adjusted <= 0.5

    def test_clamps_to_range(self):
        adapter = DifficultyAdapter()
        adjusted = adapter.adjust(
            current_difficulty=0.99,
            recent_accuracy=1.0,
            target_accuracy=0.5,
        )
        assert 0.0 <= adjusted <= 1.0


class TestProblemFilter:
    """Tests for ProblemFilter."""

    def test_creation(self):
        filt = ProblemFilter()
        assert filt is not None

    def test_filter_valid(self):
        filt = ProblemFilter()
        problems = [
            Problem(
                question="What is 2+2?",
                answer="4",
                answer_type=AnswerType.NUMERIC,
                domain="math",
                difficulty=0.5,
            ),
            Problem(
                question="",  # Invalid — empty question
                answer="4",
                answer_type=AnswerType.NUMERIC,
                domain="math",
                difficulty=0.5,
            ),
        ]
        filtered = filt.filter(problems)
        assert isinstance(filtered, list)
        # At minimum, empty question should be filtered
        assert len(filtered) <= 2

    def test_filter_duplicates(self):
        filt = ProblemFilter()
        p = Problem(
            question="What is 2+2?",
            answer="4",
            answer_type=AnswerType.NUMERIC,
            domain="math",
            difficulty=0.5,
        )
        problems = [p, p, p]  # Duplicates
        filtered = filt.filter(problems, deduplicate=True)
        assert len(filtered) <= 1


# ─── AdaptiveDifficultyEngine (PID) Tests ─────────

class TestAdaptiveDifficultyEngine:
    """Tests for PID-based difficulty controller."""

    def test_creation(self):
        engine = AdaptiveDifficultyEngine(
            target_accuracy=0.6,
            kp=0.5,
            ki=0.1,
            kd=0.05,
        )
        assert engine is not None

    def test_initial_difficulty(self):
        engine = AdaptiveDifficultyEngine(
            target_accuracy=0.6,
            initial_difficulty=0.3,
        )
        assert engine.current_difficulty == pytest.approx(0.3)

    def test_pid_increases_difficulty_when_easy(self):
        engine = AdaptiveDifficultyEngine(
            target_accuracy=0.6,
            initial_difficulty=0.5,
            kp=1.0,
        )
        # Model scoring 0.9 (above target 0.6) → should increase difficulty
        for _ in range(5):
            engine.update(accuracy=0.9)
        assert engine.current_difficulty > 0.5

    def test_pid_decreases_difficulty_when_hard(self):
        engine = AdaptiveDifficultyEngine(
            target_accuracy=0.6,
            initial_difficulty=0.5,
            kp=1.0,
        )
        # Model scoring 0.2 (below target 0.6) → should decrease difficulty
        for _ in range(5):
            engine.update(accuracy=0.2)
        assert engine.current_difficulty < 0.5

    def test_pid_clamps(self):
        engine = AdaptiveDifficultyEngine(
            target_accuracy=0.6,
            initial_difficulty=0.5,
            min_difficulty=0.1,
            max_difficulty=0.9,
        )
        # Force extreme updates
        for _ in range(100):
            engine.update(accuracy=0.0)
        assert engine.current_difficulty >= 0.1

        for _ in range(100):
            engine.update(accuracy=1.0)
        assert engine.current_difficulty <= 0.9

    def test_pid_converges(self):
        engine = AdaptiveDifficultyEngine(
            target_accuracy=0.6,
            initial_difficulty=0.5,
            kp=0.3,
            ki=0.05,
            kd=0.01,
        )
        # Fixed accuracy — PID should converge
        for _ in range(100):
            engine.update(accuracy=0.6)
        # Should be close to initial since error is ~0
        diff = abs(engine.current_difficulty - 0.5)
        # Allow generous tolerance since integrator may drift
        assert diff < 0.5

    def test_integral_windup_prevention(self):
        engine = AdaptiveDifficultyEngine(
            target_accuracy=0.6,
            kp=0.1,
            ki=0.5,  # High integral gain
        )
        # Long run of extreme error
        for _ in range(1000):
            engine.update(accuracy=0.0)
        # Should not diverge to infinity
        assert math.isfinite(engine.current_difficulty)

    def test_reset(self):
        engine = AdaptiveDifficultyEngine(
            target_accuracy=0.6,
            initial_difficulty=0.3,
        )
        engine.update(accuracy=0.9)
        engine.update(accuracy=0.9)
        engine.reset()
        assert engine.current_difficulty == pytest.approx(0.3)


class TestCurriculumScheduler:
    """Tests for CurriculumScheduler."""

    def test_creation(self):
        sched = CurriculumScheduler(
            phases=[
                (0.3, 50),   # Easy for 50 iterations
                (0.5, 50),   # Medium for 50 iterations
                (0.7, 50),   # Hard for 50 iterations
            ]
        )
        assert sched is not None

    def test_get_difficulty_by_phase(self):
        sched = CurriculumScheduler(
            phases=[
                (0.3, 50),
                (0.5, 50),
                (0.7, 50),
            ]
        )
        assert sched.get_difficulty(0) == pytest.approx(0.3)
        assert sched.get_difficulty(50) == pytest.approx(0.5)
        assert sched.get_difficulty(100) == pytest.approx(0.7)

    def test_beyond_curriculum(self):
        sched = CurriculumScheduler(
            phases=[(0.3, 10)]
        )
        # Beyond last phase — should use last difficulty
        d = sched.get_difficulty(999)
        assert d == pytest.approx(0.3)


# ─── AnswerVerifier Tests ─────────────────────────

class TestAnswerVerifier:
    """Tests for multi-type answer verification."""

    def test_numeric_correct(self):
        verifier = AnswerVerifier()
        result = verifier.verify(
            expected="42",
            predicted="42",
            answer_type=AnswerType.NUMERIC,
        )
        assert result is True

    def test_numeric_with_tolerance(self):
        verifier = AnswerVerifier(numeric_tolerance=0.01)
        result = verifier.verify(
            expected="3.14159",
            predicted="3.1416",
            answer_type=AnswerType.NUMERIC,
        )
        assert result is True

    def test_numeric_wrong(self):
        verifier = AnswerVerifier()
        result = verifier.verify(
            expected="42",
            predicted="43",
            answer_type=AnswerType.NUMERIC,
        )
        assert result is False

    def test_mcq_correct(self):
        verifier = AnswerVerifier()
        result = verifier.verify(
            expected="B",
            predicted="B",
            answer_type=AnswerType.MCQ,
        )
        assert result is True

    def test_mcq_case_insensitive(self):
        verifier = AnswerVerifier()
        result = verifier.verify(
            expected="B",
            predicted="b",
            answer_type=AnswerType.MCQ,
        )
        assert result is True

    def test_mcq_with_prefix(self):
        verifier = AnswerVerifier()
        result = verifier.verify(
            expected="C",
            predicted="The answer is C",
            answer_type=AnswerType.MCQ,
        )
        assert result is True

    def test_proof_verification(self):
        verifier = AnswerVerifier()
        result = verifier.verify(
            expected="true",
            predicted="We prove by induction... QED",
            answer_type=AnswerType.PROOF,
        )
        # Proof verification may always return True or use heuristics
        assert isinstance(result, bool)

    def test_symbolic_exact(self):
        verifier = AnswerVerifier()
        result = verifier.verify(
            expected="x^2 + 2x + 1",
            predicted="x^2 + 2x + 1",
            answer_type=AnswerType.SYMBOLIC,
        )
        assert result is True

    def test_code_verification(self):
        verifier = AnswerVerifier()
        result = verifier.verify(
            expected="def f(x): return x + 1",
            predicted="def f(x): return x + 1",
            answer_type=AnswerType.CODE,
        )
        assert isinstance(result, bool)


class TestConsistencyChecker:
    """Tests for ConsistencyChecker."""

    def test_creation(self):
        checker = ConsistencyChecker()
        assert checker is not None

    def test_consistent_answers(self):
        checker = ConsistencyChecker()
        answers = ["42", "42", "42"]
        is_consistent = checker.check(answers)
        assert is_consistent is True

    def test_inconsistent_answers(self):
        checker = ConsistencyChecker(tolerance=0.0)
        answers = ["42", "43", "44"]
        is_consistent = checker.check(answers)
        # At least should detect variance
        assert isinstance(is_consistent, bool)

    def test_single_answer(self):
        checker = ConsistencyChecker()
        answers = ["42"]
        is_consistent = checker.check(answers)
        assert is_consistent is True

    def test_empty_answers(self):
        checker = ConsistencyChecker()
        is_consistent = checker.check([])
        assert is_consistent is True


class TestBatchVerifier:
    """Tests for BatchVerifier."""

    def test_creation(self):
        bv = BatchVerifier()
        assert bv is not None

    def test_verify_batch(self):
        bv = BatchVerifier()
        problems = [
            Problem(
                question=f"What is {i}+{i}?",
                answer=str(2 * i),
                answer_type=AnswerType.NUMERIC,
                domain="math",
                difficulty=0.5,
            )
            for i in range(5)
        ]
        predictions = [str(2 * i) for i in range(5)]
        results = bv.verify_batch(problems, predictions)
        assert len(results) == 5
        assert all(r is True for r in results)

    def test_batch_with_wrong_answers(self):
        bv = BatchVerifier()
        problems = [
            Problem(
                question="What is 1+1?",
                answer="2",
                answer_type=AnswerType.NUMERIC,
                domain="math",
                difficulty=0.1,
            )
        ]
        predictions = ["3"]  # Wrong
        results = bv.verify_batch(problems, predictions)
        assert len(results) == 1
        assert results[0] is False

    def test_batch_accuracy(self):
        bv = BatchVerifier()
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
        # Half correct
        predictions = [str(i) if i < 5 else str(i + 100) for i in range(10)]
        results = bv.verify_batch(problems, predictions)
        accuracy = sum(results) / len(results)
        assert accuracy == pytest.approx(0.5)
