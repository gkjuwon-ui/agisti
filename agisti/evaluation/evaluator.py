"""
Evaluator — run model inference and evaluate solutions.

Handles the full pipeline:
1. Format problem as prompt
2. Generate model response
3. Parse response into solution
4. Verify against expected answer
5. Collect correct/incorrect pairs for surgery
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from agisti.types import (
    AnswerType,
    ErrorReport,
    FailedProblem,
    Problem,
    Solution,
)
from agisti.generation.verification import AnswerVerifier, VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a batch of problems."""
    problems: list[Problem]
    solutions: list[Solution]
    verifications: list[VerificationResult]
    correct_pairs: list[tuple[Problem, Solution]]
    incorrect_pairs: list[tuple[Problem, Solution]]
    failed_problems: list[FailedProblem]
    total_time_seconds: float

    @property
    def accuracy(self) -> float:
        total = len(self.problems)
        return len(self.correct_pairs) / total if total > 0 else 0.0

    @property
    def total(self) -> int:
        return len(self.problems)

    @property
    def correct_count(self) -> int:
        return len(self.correct_pairs)

    @property
    def incorrect_count(self) -> int:
        return len(self.incorrect_pairs)

    def domain_breakdown(self) -> dict[str, dict[str, int]]:
        """Break down results by domain."""
        breakdown: dict[str, dict[str, int]] = {}
        for p, s in self.correct_pairs:
            if p.domain not in breakdown:
                breakdown[p.domain] = {"correct": 0, "incorrect": 0}
            breakdown[p.domain]["correct"] += 1
        for p, s in self.incorrect_pairs:
            if p.domain not in breakdown:
                breakdown[p.domain] = {"correct": 0, "incorrect": 0}
            breakdown[p.domain]["incorrect"] += 1
        return breakdown


class ModelEvaluator:
    """
    Evaluates a model on a set of problems.

    Orchestrates generation → parsing → verification pipeline.
    """

    def __init__(
        self,
        verifier: AnswerVerifier | None = None,
        max_gen_tokens: int = 1024,
        temperature: float = 0.0,
        batch_size: int = 8,
    ):
        self.verifier = verifier or AnswerVerifier()
        self.max_gen_tokens = max_gen_tokens
        self.temperature = temperature
        self.batch_size = batch_size

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        problems: list[Problem],
    ) -> EvaluationResult:
        """
        Evaluate model on a list of problems.

        Processes in batches for GPU efficiency.
        """
        start_time = time.monotonic()

        all_solutions: list[Solution] = []
        all_verifications: list[VerificationResult] = []
        correct_pairs: list[tuple[Problem, Solution]] = []
        incorrect_pairs: list[tuple[Problem, Solution]] = []
        failed_problems: list[FailedProblem] = []

        for batch_start in range(0, len(problems), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(problems))
            batch = problems[batch_start:batch_end]

            solutions = self._generate_solutions_batch(
                model, tokenizer, batch,
            )

            for problem, solution in zip(batch, solutions):
                verification = self.verifier.verify(
                    problem,
                    solution.answer,
                    expected_answer=problem.answer,
                )

                all_solutions.append(solution)
                all_verifications.append(verification)

                if verification.correct:
                    correct_pairs.append((problem, solution))
                else:
                    incorrect_pairs.append((problem, solution))
                    fp = FailedProblem(
                        problem=problem,
                        original_solution=solution,
                        domain=problem.domain,
                        ground_truth=problem.answer,
                        answer_type=problem.answer_type,
                        verification_code=problem.verification_code,
                    )
                    failed_problems.append(fp)

        elapsed = time.monotonic() - start_time

        logger.info(
            "Evaluation: %d/%d correct (%.1f%%) in %.1fs",
            len(correct_pairs),
            len(problems),
            len(correct_pairs) / len(problems) * 100 if problems else 0,
            elapsed,
        )

        return EvaluationResult(
            problems=problems,
            solutions=all_solutions,
            verifications=all_verifications,
            correct_pairs=correct_pairs,
            incorrect_pairs=incorrect_pairs,
            failed_problems=failed_problems,
            total_time_seconds=elapsed,
        )

    def evaluate_single(
        self,
        model: Any,
        tokenizer: Any,
        problem: Problem,
    ) -> tuple[Solution, VerificationResult]:
        """Evaluate a single problem."""
        start = time.monotonic()
        answer = self._generate_answer(model, tokenizer, problem)
        gen_time = time.monotonic() - start

        solution = Solution(
            problem_id=problem.id,
            answer=answer,
            chain_of_thought="",
            tokens_generated=len(answer.split()),
            generation_time_seconds=gen_time,
        )

        verification = self.verifier.verify(
            problem,
            answer,
            expected_answer=problem.answer,
        )

        return solution, verification

    def _generate_solutions_batch(
        self,
        model: Any,
        tokenizer: Any,
        problems: list[Problem],
    ) -> list[Solution]:
        """Generate solutions for a batch of problems using true batched generation."""
        from agisti.generation.prompt_utils import format_for_model

        # Prepare all prompts
        prompts = [self._format_prompt(p) for p in problems]

        # Try true batched generation first
        try:
            return self._batched_generate(model, tokenizer, problems, prompts)
        except Exception as e:
            logger.debug("Batched generation failed (%s), falling back to sequential", e)

        # Fallback: sequential
        solutions = []
        for problem in problems:
            try:
                answer = self._generate_answer(model, tokenizer, problem)
            except Exception as e:
                logger.warning("Generation error for problem %s: %s", problem.id, e)
                answer = f"<generation_error: {e}>"

            solutions.append(Solution(
                problem_id=problem.id,
                answer=answer,
                chain_of_thought="",
                tokens_generated=len(answer.split()),
                generation_time_seconds=0.0,
            ))

        return solutions

    def _batched_generate(
        self,
        model: Any,
        tokenizer: Any,
        problems: list[Problem],
        prompts: list[str],
    ) -> list[Solution]:
        """True batched generation using left-padded inputs."""
        from agisti.generation.prompt_utils import format_for_model

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Tokenize all prompts with left-padding for generation
        original_side = getattr(tokenizer, 'padding_side', 'right')
        tokenizer.padding_side = 'left'
        try:
            batch_inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
        finally:
            tokenizer.padding_side = original_side

        device = next(model.parameters()).device
        input_ids = batch_inputs["input_ids"].to(device)
        attention_mask = batch_inputs["attention_mask"].to(device)
        input_lengths = attention_mask.sum(dim=1)

        start = time.monotonic()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_gen_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_time = time.monotonic() - start

        solutions = []
        for i, problem in enumerate(problems):
            new_tokens = outputs[i][input_ids.shape[1]:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            solutions.append(Solution(
                problem_id=problem.id,
                answer=answer,
                chain_of_thought="",
                tokens_generated=len(answer.split()),
                generation_time_seconds=gen_time / len(problems),
            ))

        return solutions

    def _generate_answer(
        self,
        model: Any,
        tokenizer: Any,
        problem: Problem,
    ) -> str:
        """Generate a single answer from the model."""
        from agisti.generation.prompt_utils import format_for_model

        prompt = self._format_prompt(problem)
        inputs = format_for_model(prompt, tokenizer)

        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_gen_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _format_prompt(self, problem: Problem) -> str:
        """Format a problem into a prompt."""
        type_hints = {
            AnswerType.EXACT_MATCH: "Think step by step, then give your final answer on a new line starting with 'Final Answer:'",
            AnswerType.NUMERIC_RANGE: "Think step by step, then give your numerical answer on a new line starting with 'Final Answer:'",
            AnswerType.CODE_EXEC: "Write Python code to solve this. Put the code in a ```python``` block.",
            AnswerType.PROOF_CHECK: "Prove this step by step.",
            AnswerType.OPEN_ENDED: "Answer thoroughly.",
        }
        hint = type_hints.get(problem.answer_type, "Answer the question.")
        return f"Question: {problem.question}\n\n{hint}\n\nAnswer:"


class SolutionCollector:
    """
    Collects and organizes solutions for surgery.

    Separates correct from incorrect solutions and prepares
    the problem pairs needed for activation contrast surgery.
    """

    def __init__(self):
        self.correct_solutions: list[tuple[Problem, Solution]] = []
        self.incorrect_solutions: list[tuple[Problem, Solution]] = []
        self._history_length = 0

    def add_result(self, result: EvaluationResult) -> None:
        """Add an evaluation result to the collection."""
        self.correct_solutions.extend(result.correct_pairs)
        self.incorrect_solutions.extend(result.incorrect_pairs)
        self._history_length += 1

    def get_contrast_pairs(
        self,
        max_pairs: int = 100,
    ) -> tuple[list[tuple[Problem, str]], list[tuple[Problem, str]]]:
        """
        Get matched pairs for activation contrast.

        Returns:
            (correct_pairs, incorrect_pairs) where each pair is
            (problem, answer_text).
        """
        correct = [
            (p, s.answer) for p, s in self.correct_solutions[-max_pairs:]
        ]
        incorrect = [
            (p, s.answer) for p, s in self.incorrect_solutions[-max_pairs:]
        ]
        return correct, incorrect

    def get_failed_problems(self) -> list[FailedProblem]:
        """Convert incorrect solutions to FailedProblems."""
        failed = []
        for problem, solution in self.incorrect_solutions:
            fp = FailedProblem(
                problem=problem,
                original_solution=solution,
                domain=problem.domain,
                ground_truth=problem.answer,
                answer_type=problem.answer_type,
                verification_code=problem.verification_code,
            )
            failed.append(fp)
        return failed

    def clear(self) -> None:
        """Clear collected solutions."""
        self.correct_solutions.clear()
        self.incorrect_solutions.clear()

    @property
    def total_correct(self) -> int:
        return len(self.correct_solutions)

    @property
    def total_incorrect(self) -> int:
        return len(self.incorrect_solutions)

    @property
    def accuracy(self) -> float:
        total = self.total_correct + self.total_incorrect
        return self.total_correct / total if total > 0 else 0.0
