"""
QuickBench — fast benchmark for iteration-level evaluation.

Runs ~200 problems with epoch-rotated seeding to prevent
memorization gaming. Uses McNemar's test for significance.

Design: §6.1 — Iteration-level quick check.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from agisti.types import (
    AnswerType,
    Problem,
    QuickBenchResult,
    VERIFIABLE_TYPES,
)
from agisti.config import QuickBenchConfig, BenchmarkSuiteSpec
from agisti.generation.verification import AnswerVerifier
from agisti.benchmark.mcnemar import mcnemar_test, SignificanceResult

logger = logging.getLogger(__name__)


@dataclass
class QuickBenchProblem:
    """A benchmark problem with known answer."""
    id: str
    question: str
    domain: str
    answer_type: AnswerType
    expected_answer: str
    difficulty: float = 0.5
    tolerance: float = 0.0
    tags: list[str] = field(default_factory=list)

    def to_problem(self) -> Problem:
        return Problem(
            id=self.id,
            question=self.question,
            domain=self.domain,
            answer=self.expected_answer,
            answer_type=self.answer_type,
            difficulty=int(self.difficulty * 5) or 1,
            tolerance=self.tolerance,
            metadata={
                "expected_answer": self.expected_answer,
                "tolerance": self.tolerance,
                "bench_id": self.id,
            },
        )


class QuickBenchSuite:
    """
    A collection of benchmark problems organized by domain.

    Immutable once loaded — benchmark integrity is critical.
    """

    def __init__(self, name: str, problems: list[QuickBenchProblem]):
        self.name = name
        self._problems = list(problems)
        self._by_domain: dict[str, list[QuickBenchProblem]] = {}
        for p in self._problems:
            if p.domain not in self._by_domain:
                self._by_domain[p.domain] = []
            self._by_domain[p.domain].append(p)

    @property
    def total(self) -> int:
        return len(self._problems)

    @property
    def domains(self) -> list[str]:
        return list(self._by_domain.keys())

    def sample(self, count: int, seed: int) -> list[QuickBenchProblem]:
        """
        Sample problems with a deterministic seed.

        The seed is epoch-rotated: seed = epoch * 1337 + 42
        This ensures different problems each epoch to prevent
        the model from memorizing the benchmark order.
        """
        import random
        rng = random.Random(seed)
        if count >= len(self._problems):
            items = list(self._problems)
            rng.shuffle(items)
            return items
        return rng.sample(self._problems, count)

    def sample_by_domain(
        self,
        per_domain: int,
        seed: int,
    ) -> list[QuickBenchProblem]:
        """Sample equally from each domain."""
        import random
        rng = random.Random(seed)
        sampled = []
        for domain, problems in sorted(self._by_domain.items()):
            k = min(per_domain, len(problems))
            sampled.extend(rng.sample(problems, k))
        return sampled

    @classmethod
    def from_jsonl(cls, name: str, path: str | Path) -> QuickBenchSuite:
        """Load from JSONL file."""
        problems = []
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                problems.append(QuickBenchProblem(
                    id=data.get("id", hashlib.md5(line.encode()).hexdigest()[:12]),
                    question=data["question"],
                    domain=data["domain"],
                    answer_type=AnswerType(data.get("answer_type", "exact_match")),
                    expected_answer=data["expected_answer"],
                    difficulty=data.get("difficulty", 0.5),
                    tolerance=data.get("tolerance", 0.0),
                    tags=data.get("tags", []),
                ))
        logger.info("Loaded QuickBenchSuite '%s': %d problems", name, len(problems))
        return cls(name, problems)


class QuickBench:
    """
    Fast benchmark runner for per-iteration evaluation.

    Configuration from §6.1:
    - ~200 problems per evaluation
    - Epoch-rotated seed: seed = epoch * 1337 + 42
    - McNemar's test for significance
    - Runs in <30 seconds on target hardware
    """

    def __init__(
        self,
        suite: QuickBenchSuite,
        config: QuickBenchConfig | None = None,
        verifier: AnswerVerifier | None = None,
        max_gen_tokens: int = 512,
    ):
        self.suite = suite
        self.config = config or QuickBenchConfig()
        self.verifier = verifier or AnswerVerifier()
        self.max_gen_tokens = max_gen_tokens
        self._previous_results: list[bool] | None = None
        self._previous_problems: list[str] | None = None

    def run(
        self,
        model: Any,
        tokenizer: Any,
        epoch: int,
    ) -> QuickBenchResult:
        """
        Run quick benchmark.

        Args:
            model: The model to evaluate.
            tokenizer: Tokenizer for the model.
            epoch: Current epoch (for seed rotation).

        Returns:
            QuickBenchResult with accuracy, significance, and domain breakdown.
        """
        start = time.monotonic()

        # Epoch-rotated seed
        seed = epoch * 1337 + 42
        problems = self.suite.sample(self.config.num_problems, seed)

        results: list[bool] = []
        problem_ids: list[str] = []
        domain_correct: dict[str, int] = {}
        domain_total: dict[str, int] = {}

        for bench_problem in problems:
            problem = bench_problem.to_problem()
            problem_ids.append(bench_problem.id)

            try:
                answer = self._generate_answer(model, tokenizer, problem)
                verification = self.verifier.verify(
                    problem,
                    answer,
                    expected_answer=bench_problem.expected_answer,
                    tolerance=bench_problem.tolerance,
                )
                correct = verification.correct
            except Exception as e:
                logger.debug("QuickBench problem error: %s", e)
                correct = False

            results.append(correct)
            # Free KV cache between problems
            torch.cuda.empty_cache()

            domain = bench_problem.domain
            domain_total[domain] = domain_total.get(domain, 0) + 1
            if correct:
                domain_correct[domain] = domain_correct.get(domain, 0) + 1

        elapsed = time.monotonic() - start
        accuracy = sum(results) / len(results) if results else 0.0

        # McNemar significance test against previous run
        p_value = 1.0
        significant = False
        if self._previous_results is not None:
            # Align results by problem ID for McNemar
            sig_result = self._compute_significance(
                problem_ids, results,
            )
            if sig_result is not None:
                p_value = sig_result.p_value
                significant = sig_result.significant

        # Domain scores
        domain_scores = {
            d: domain_correct.get(d, 0) / domain_total[d]
            for d in domain_total
        }

        # Detect regressions (domains that got worse)
        regressions: dict[str, dict[str, Any]] = {}
        if self._previous_results is not None and self._previous_problems is not None:
            # Compare per-domain to detect regressions
            for domain, score in domain_scores.items():
                prev_domain_score = getattr(self, "_prev_domain_scores", {}).get(domain)
                if prev_domain_score is not None and score < prev_domain_score - 0.05:
                    regressions[domain] = {
                        "previous": prev_domain_score,
                        "current": score,
                        "delta": score - prev_domain_score,
                    }

        # Store for next comparison
        self._previous_results = results
        self._previous_problems = problem_ids
        self._prev_domain_scores = dict(domain_scores)

        passed = accuracy >= self.config.pass_threshold if hasattr(self.config, "pass_threshold") else True

        logger.info(
            "QuickBench: %.1f%% (%d/%d) in %.1fs%s",
            accuracy * 100,
            sum(results),
            len(results),
            elapsed,
            " (significant)" if significant else "",
        )

        return QuickBenchResult(
            scores=domain_scores,
            elapsed_seconds=elapsed,
            passed=passed,
            regressions=regressions,
            accuracy=accuracy,
            domain_breakdown=domain_scores,
        )

    def _compute_significance(
        self,
        current_ids: list[str],
        current_results: list[bool],
    ) -> SignificanceResult | None:
        """Compute McNemar significance against previous results."""
        if self._previous_results is None or self._previous_problems is None:
            return None

        # Build lookup for previous results
        prev_lookup = dict(zip(self._previous_problems, self._previous_results))

        # Find overlapping problems
        current_matched = []
        prev_matched = []
        for pid, result in zip(current_ids, current_results):
            if pid in prev_lookup:
                current_matched.append(result)
                prev_matched.append(prev_lookup[pid])

        if len(current_matched) < 20:
            return None

        return mcnemar_test(prev_matched, current_matched)

    def _generate_answer(
        self,
        model: Any,
        tokenizer: Any,
        problem: Problem,
    ) -> str:
        """Generate a single answer."""
        from agisti.generation.prompt_utils import format_for_model

        prompt = f"Question: {problem.question}\n\nAnswer:"
        inputs = format_for_model(prompt, tokenizer)

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

    def reset(self) -> None:
        """Reset comparison state."""
        self._previous_results = None
        self._previous_problems = None
