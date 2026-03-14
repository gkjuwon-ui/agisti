"""
FullBench — comprehensive phase-level benchmark suite.

Runs extensive evaluation across multiple benchmark suites
when transitioning between phases. More thorough than QuickBench
but runs less frequently.

Design: §6.2 — Phase-level comprehensive evaluation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from agisti.types import AnswerType, FullBenchResult, Problem
from agisti.config import FullBenchConfig, BenchmarkSuiteSpec, FULL_BENCH_SUITES
from agisti.generation.verification import AnswerVerifier
from agisti.benchmark.quick_bench import QuickBenchSuite, QuickBenchProblem
from agisti.benchmark.mcnemar import mcnemar_test, SignificanceResult

logger = logging.getLogger(__name__)


@dataclass
class SuiteResult:
    """Result for a single benchmark suite."""
    name: str
    accuracy: float
    total: int
    correct: int
    per_problem_results: list[bool]
    problem_ids: list[str]
    domain_breakdown: dict[str, dict[str, int]] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def domain_accuracies(self) -> dict[str, float]:
        return {
            domain: stats["correct"] / stats["total"]
            for domain, stats in self.domain_breakdown.items()
            if stats["total"] > 0
        }


class FullBench:
    """
    Comprehensive benchmark runner for phase transitions.

    Runs multiple benchmark suites with thoroughness appropriate
    for the current phase. Phase transitions only occur when
    FullBench confirms the model meets phase gate criteria.
    """

    def __init__(
        self,
        suites: dict[str, QuickBenchSuite],
        config: FullBenchConfig | None = None,
        verifier: AnswerVerifier | None = None,
        max_gen_tokens: int = 1024,
    ):
        self.suites = suites
        self.config = config or FullBenchConfig()
        self.verifier = verifier or AnswerVerifier()
        self.max_gen_tokens = max_gen_tokens
        self._previous_results: dict[str, SuiteResult] | None = None

    def run(
        self,
        model: Any,
        tokenizer: Any,
        suite_names: list[str] | None = None,
        phase: int = 0,
    ) -> FullBenchResult:
        """
        Run full benchmark evaluation.

        Args:
            model: Model to evaluate.
            tokenizer: Tokenizer.
            suite_names: Which suites to run (None = all).
            phase: Current phase (for phase-appropriate suites).

        Returns:
            FullBenchResult with comprehensive metrics.
        """
        start = time.monotonic()

        # Select suites
        if suite_names is None:
            suite_names = list(self.suites.keys())

        suite_results: dict[str, SuiteResult] = {}
        all_domain_scores: dict[str, list[float]] = {}

        for name in suite_names:
            suite = self.suites.get(name)
            if suite is None:
                logger.warning("Suite '%s' not found, skipping", name)
                continue

            result = self._run_suite(model, tokenizer, suite, name)
            suite_results[name] = result

            # Aggregate domain scores
            for domain, accuracy in result.domain_accuracies().items():
                if domain not in all_domain_scores:
                    all_domain_scores[domain] = []
                all_domain_scores[domain].append(accuracy)

        elapsed = time.monotonic() - start

        # Compute overall metrics
        total_correct = sum(r.correct for r in suite_results.values())
        total_problems = sum(r.total for r in suite_results.values())
        overall_accuracy = total_correct / total_problems if total_problems > 0 else 0.0

        # Average domain scores
        domain_scores = {
            d: sum(scores) / len(scores)
            for d, scores in all_domain_scores.items()
        }

        # Significance against previous run
        significance = self._compute_overall_significance(suite_results)

        # Check for regressions
        regressions = self._detect_regressions(suite_results)

        # Store for next comparison
        self._previous_results = suite_results

        logger.info(
            "FullBench: %.1f%% overall (%d/%d) across %d suites in %.1fs",
            overall_accuracy * 100,
            total_correct,
            total_problems,
            len(suite_results),
            elapsed,
        )

        if regressions:
            logger.warning(
                "Regressions detected in: %s",
                ", ".join(regressions.keys()),
            )

        return FullBenchResult(
            accuracy=overall_accuracy,
            suite_scores={n: r.accuracy for n, r in suite_results.items()},
            domain_scores=domain_scores,
            regressions=regressions,
        )

    def run_for_phase(
        self,
        model: Any,
        tokenizer: Any,
        phase: int,
    ) -> FullBenchResult:
        """Run phase-specific benchmark suites."""
        phase_key = f"phase_{phase}"
        suite_specs = FULL_BENCH_SUITES.get(phase_key, [])
        suite_names = [spec.name for spec in suite_specs]

        if not suite_names:
            suite_names = list(self.suites.keys())

        return self.run(model, tokenizer, suite_names, phase)

    def check_phase_gate(
        self,
        result: FullBenchResult,
        phase: int,
        min_accuracy: float = 0.7,
    ) -> tuple[bool, str]:
        """
        Check if FullBench results pass the phase gate.

        Phase gates:
        - Phase 0→1: >70% overall accuracy, no suite <50%
        - Phase 1→2: >75% overall, no suite <55%
        - Phase 2→3: >80% overall, no suite <60%
        """
        min_overall = min_accuracy + phase * 0.05
        min_suite = 0.5 + phase * 0.05

        if result.accuracy < min_overall:
            return False, (
                f"Overall accuracy {result.accuracy:.1%} < "
                f"required {min_overall:.1%}"
            )

        for suite_name, score in result.suite_scores.items():
            if score < min_suite:
                return False, (
                    f"Suite '{suite_name}' accuracy {score:.1%} < "
                    f"required {min_suite:.1%}"
                )

        if result.regressions:
            regression_severity = max(
                abs(v) for v in result.regressions.values()
            )
            if regression_severity > 0.1:
                return False, (
                    f"Significant regression detected "
                    f"(max: {regression_severity:.1%})"
                )

        return True, "Phase gate passed"

    def _run_suite(
        self,
        model: Any,
        tokenizer: Any,
        suite: QuickBenchSuite,
        name: str,
    ) -> SuiteResult:
        """Run a single benchmark suite."""
        start = time.monotonic()

        # Use all problems for full bench
        problems = suite.sample(suite.total, seed=42)

        results: list[bool] = []
        problem_ids: list[str] = []
        domain_correct: dict[str, int] = {}
        domain_total: dict[str, int] = {}

        for bench_p in problems:
            problem = bench_p.to_problem()
            problem_ids.append(bench_p.id)

            try:
                answer = self._generate_answer(model, tokenizer, problem)
                verification = self.verifier.verify(
                    problem, answer,
                    expected_answer=bench_p.expected_answer,
                    tolerance=bench_p.tolerance,
                )
                correct = verification.correct
            except Exception as e:
                logger.debug("FullBench error in %s: %s", name, e)
                correct = False

            results.append(correct)

            d = bench_p.domain
            domain_total[d] = domain_total.get(d, 0) + 1
            if correct:
                domain_correct[d] = domain_correct.get(d, 0) + 1

        elapsed = time.monotonic() - start
        total_correct = sum(results)

        breakdown = {
            d: {"correct": domain_correct.get(d, 0), "total": domain_total[d]}
            for d in domain_total
        }

        return SuiteResult(
            name=name,
            accuracy=total_correct / len(results) if results else 0.0,
            total=len(results),
            correct=total_correct,
            per_problem_results=results,
            problem_ids=problem_ids,
            domain_breakdown=breakdown,
            elapsed_seconds=elapsed,
        )

    def _compute_overall_significance(
        self,
        current_results: dict[str, SuiteResult],
    ) -> SignificanceResult | None:
        """Compute significance across all suites combined."""
        if self._previous_results is None:
            return None

        all_prev = []
        all_curr = []

        for name, current in current_results.items():
            previous = self._previous_results.get(name)
            if previous is None:
                continue

            # Match by problem ID
            prev_lookup = dict(
                zip(previous.problem_ids, previous.per_problem_results),
            )
            for pid, result in zip(current.problem_ids, current.per_problem_results):
                if pid in prev_lookup:
                    all_prev.append(prev_lookup[pid])
                    all_curr.append(result)

        if len(all_prev) < 20:
            return None

        return mcnemar_test(all_prev, all_curr)

    def _detect_regressions(
        self,
        current_results: dict[str, SuiteResult],
    ) -> dict[str, float]:
        """Detect suite-level regressions compared to previous run."""
        if self._previous_results is None:
            return {}

        regressions = {}
        for name, current in current_results.items():
            previous = self._previous_results.get(name)
            if previous is None:
                continue

            delta = current.accuracy - previous.accuracy
            if delta < -0.02:  # 2% regression threshold
                regressions[name] = delta

        return regressions

    def _generate_answer(
        self,
        model: Any,
        tokenizer: Any,
        problem: Problem,
    ) -> str:
        """Generate answer for benchmarking."""
        prompt = f"Question: {problem.question}\n\nAnswer:"
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_gen_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class BenchmarkOrchestrator:
    """
    Orchestrates when to run QuickBench vs FullBench.

    QuickBench: every iteration (fast, ~200 problems)
    FullBench: at phase transitions (thorough, all problems)
    """

    def __init__(
        self,
        quick_bench: Any,  # QuickBench
        full_bench: FullBench,
        full_bench_interval: int = 100,
    ):
        self.quick_bench = quick_bench
        self.full_bench = full_bench
        self.full_bench_interval = full_bench_interval

    def should_run_full(
        self,
        iteration: int,
        phase_transition: bool = False,
    ) -> bool:
        """Determine if full benchmark should run."""
        if phase_transition:
            return True
        return iteration > 0 and iteration % self.full_bench_interval == 0

    def run_appropriate(
        self,
        model: Any,
        tokenizer: Any,
        iteration: int,
        epoch: int,
        phase: int,
        phase_transition: bool = False,
    ) -> dict[str, Any]:
        """Run the appropriate benchmark based on context."""
        results: dict[str, Any] = {}

        # Always run QuickBench
        quick_result = self.quick_bench.run(model, tokenizer, epoch)
        results["quick"] = quick_result

        # Run FullBench when appropriate
        if self.should_run_full(iteration, phase_transition):
            full_result = self.full_bench.run_for_phase(
                model, tokenizer, phase,
            )
            results["full"] = full_result

        return results
