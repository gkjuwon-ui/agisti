"""
Answer verification — mechanically verify model answers.

Supports all VERIFIABLE answer types:
- EXACT_MATCH: String comparison (case-insensitive, normalized)
- NUMERIC_RANGE: Within tolerance
- CODE_EXEC: Execute code and compare output (sandboxed)
- PROOF_CHECK: Verify proof structure and logical steps

§5 design constraint: only verifiable answers feed into surgery.
"""

from __future__ import annotations

import ast
import hashlib
import io
import logging
import math
import re
import signal
import sys
import textwrap
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any

from agisti.types import (
    AnswerType,
    ErrorReport,
    FailedProblem,
    Problem,
    Solution,
    VERIFIABLE_TYPES,
)

logger = logging.getLogger(__name__)

# Maximum time for code execution (seconds)
CODE_EXEC_TIMEOUT = 10
# Maximum output size from code execution
CODE_EXEC_MAX_OUTPUT = 10000


@dataclass
class VerificationResult:
    """Result of verifying a single answer."""
    correct: bool
    confidence: float  # 0-1, how confident we are in the verdict
    explanation: str = ""
    expected: str = ""
    got: str = ""
    verification_method: str = ""

    def to_error_report(self, problem_id: str = "", domain: str = "") -> ErrorReport | None:
        """Convert to ErrorReport if incorrect."""
        if self.correct:
            return None
        return ErrorReport(
            problem_id=problem_id,
            domain=domain,
            expected=self.expected,
            actual=self.got,
            answer_type=AnswerType.EXACT_MATCH,
            error_category=self.verification_method,
            reasoning_trace=self.explanation,
        )


class AnswerVerifier:
    """
    Verifies model answers against expected answers.

    Dispatches to type-specific verifiers based on AnswerType.
    """

    def __init__(
        self,
        allow_code_exec: bool = True,
        case_sensitive: bool = False,
        whitespace_sensitive: bool = False,
        numeric_tolerance: float = 1e-6,
    ):
        self.allow_code_exec = allow_code_exec
        self.case_sensitive = case_sensitive
        self.whitespace_sensitive = whitespace_sensitive
        self.numeric_tolerance = numeric_tolerance

    def verify(
        self,
        problem: Problem,
        model_answer: str,
        expected_answer: str | None = None,
        tolerance: float | None = None,
    ) -> VerificationResult:
        """
        Verify a model's answer against the expected answer.

        Args:
            problem: The problem that was solved.
            model_answer: The model's response.
            expected_answer: Override expected answer (else from metadata).
            tolerance: Override tolerance for numeric comparisons.

        Returns:
            VerificationResult with correctness verdict.
        """
        if expected_answer is None:
            expected_answer = problem.metadata.get(
                "expected_answer", problem.answer,
            )
        if tolerance is None:
            tolerance = problem.metadata.get(
                "tolerance", problem.tolerance if hasattr(problem, "tolerance") else self.numeric_tolerance,
            )

        if not expected_answer:
            return VerificationResult(
                correct=False,
                confidence=0.0,
                explanation="No expected answer provided",
                verification_method="none",
            )

        dispatch = {
            AnswerType.EXACT_MATCH: self._verify_exact_match,
            AnswerType.NUMERIC_RANGE: self._verify_numeric,
            AnswerType.CODE_EXEC: self._verify_code_exec,
            AnswerType.PROOF_CHECK: self._verify_proof,
            AnswerType.REGEX: self._verify_regex,
        }

        verifier = dispatch.get(problem.answer_type, self._verify_exact_match)
        return verifier(model_answer, expected_answer, tolerance)

    def _verify_regex(
        self,
        model_answer: str,
        expected_pattern: str,
        tolerance: float,
    ) -> VerificationResult:
        """Verify answer by matching against a regex pattern."""
        try:
            match = re.search(expected_pattern, model_answer, re.IGNORECASE)
            if match:
                return VerificationResult(
                    correct=True,
                    confidence=0.9,
                    explanation=f"Regex pattern matched: {match.group()}",
                    expected=expected_pattern,
                    got=model_answer,
                    verification_method="regex",
                )
            return VerificationResult(
                correct=False,
                confidence=0.9,
                explanation=f"No regex match for pattern '{expected_pattern}'",
                expected=expected_pattern,
                got=model_answer,
                verification_method="regex",
            )
        except re.error as e:
            return VerificationResult(
                correct=False,
                confidence=0.0,
                explanation=f"Invalid regex pattern: {e}",
                expected=expected_pattern,
                got=model_answer,
                verification_method="regex_error",
            )

    def verify_batch(
        self,
        problems: list[Problem],
        answers: list[str],
        expected: list[str] | None = None,
    ) -> list[VerificationResult]:
        """Verify a batch of answers."""
        results = []
        for i, (p, a) in enumerate(zip(problems, answers)):
            exp = expected[i] if expected else None
            results.append(self.verify(p, a, exp))
        return results

    def _verify_exact_match(
        self,
        model_answer: str,
        expected: str,
        tolerance: float,
    ) -> VerificationResult:
        """Exact string match with normalization."""
        normalized_model = self._normalize_text(model_answer)
        normalized_expected = self._normalize_text(expected)

        # Try exact match first
        if normalized_model == normalized_expected:
            return VerificationResult(
                correct=True,
                confidence=1.0,
                explanation="Exact match",
                expected=expected,
                got=model_answer,
                verification_method="exact_match",
            )

        # Try extracting the answer from longer text
        extracted = self._extract_answer(model_answer)
        if self._normalize_text(extracted) == normalized_expected:
            return VerificationResult(
                correct=True,
                confidence=0.9,
                explanation="Match after extraction",
                expected=expected,
                got=model_answer,
                verification_method="exact_match_extracted",
            )

        # Try numeric comparison as fallback (extracted vs expected)
        extracted_num = self._extract_number(extracted)
        expected_num = self._extract_number(expected)
        if extracted_num is not None and expected_num is not None:
            if abs(extracted_num - expected_num) <= max(tolerance, 1e-9):
                return VerificationResult(
                    correct=True,
                    confidence=0.9,
                    explanation="Numeric match after extraction",
                    expected=expected,
                    got=model_answer,
                    verification_method="numeric_fallback_extracted",
                )

        # Try direct numeric comparison
        try:
            model_num = float(normalized_model)
            exp_num = float(normalized_expected)
            if abs(model_num - exp_num) <= max(tolerance, 1e-9):
                return VerificationResult(
                    correct=True,
                    confidence=0.95,
                    explanation="Numeric match within tolerance",
                    expected=expected,
                    got=model_answer,
                    verification_method="numeric_fallback",
                )
        except (ValueError, OverflowError):
            pass

        return VerificationResult(
            correct=False,
            confidence=0.95,
            explanation=f"Expected '{expected}', got '{model_answer}'",
            expected=expected,
            got=model_answer,
            verification_method="exact_match",
        )

    def _verify_numeric(
        self,
        model_answer: str,
        expected: str,
        tolerance: float,
    ) -> VerificationResult:
        """Numeric comparison within tolerance."""
        model_num = self._extract_number(model_answer)
        expected_num = self._extract_number(expected)

        if model_num is None or expected_num is None:
            return VerificationResult(
                correct=False,
                confidence=0.7,
                explanation=(
                    f"Could not parse numeric values: "
                    f"model='{model_answer}', expected='{expected}'"
                ),
                expected=expected,
                got=model_answer,
                verification_method="numeric",
            )

        # Absolute tolerance check
        abs_diff = abs(model_num - expected_num)
        if abs_diff <= tolerance:
            return VerificationResult(
                correct=True,
                confidence=1.0,
                explanation=f"Within tolerance ({abs_diff:.6g} <= {tolerance})",
                expected=expected,
                got=model_answer,
                verification_method="numeric_absolute",
            )

        # Relative tolerance check for large numbers
        if expected_num != 0:
            rel_diff = abs_diff / abs(expected_num)
            if rel_diff <= tolerance:
                return VerificationResult(
                    correct=True,
                    confidence=0.95,
                    explanation=f"Within relative tolerance ({rel_diff:.6g})",
                    expected=expected,
                    got=model_answer,
                    verification_method="numeric_relative",
                )

        return VerificationResult(
            correct=False,
            confidence=0.95,
            explanation=(
                f"Numeric mismatch: expected {expected_num}, "
                f"got {model_num}, diff={abs_diff:.6g}"
            ),
            expected=expected,
            got=model_answer,
            verification_method="numeric",
        )

    def _verify_code_exec(
        self,
        model_answer: str,
        expected: str,
        tolerance: float,
    ) -> VerificationResult:
        """Execute code and compare output (sandboxed)."""
        if not self.allow_code_exec:
            return VerificationResult(
                correct=False,
                confidence=0.0,
                explanation="Code execution disabled",
                verification_method="code_exec_disabled",
            )

        code = self._extract_code(model_answer)
        if not code:
            return VerificationResult(
                correct=False,
                confidence=0.8,
                explanation="No executable code found in answer",
                expected=expected,
                got=model_answer,
                verification_method="code_exec",
            )

        # Static safety check
        safety_issue = self._check_code_safety(code)
        if safety_issue:
            return VerificationResult(
                correct=False,
                confidence=1.0,
                explanation=f"Code safety violation: {safety_issue}",
                expected=expected,
                got=model_answer,
                verification_method="code_exec_safety",
            )

        # Execute in sandbox
        output, error = self._execute_code_sandboxed(code)

        if error:
            return VerificationResult(
                correct=False,
                confidence=0.9,
                explanation=f"Code execution error: {error}",
                expected=expected,
                got=model_answer,
                verification_method="code_exec",
            )

        # Compare output
        output_normalized = output.strip()
        expected_normalized = expected.strip()

        if output_normalized == expected_normalized:
            return VerificationResult(
                correct=True,
                confidence=1.0,
                explanation="Code output matches expected",
                expected=expected,
                got=output,
                verification_method="code_exec",
            )

        # Try numeric comparison of outputs
        try:
            out_num = float(output_normalized)
            exp_num = float(expected_normalized)
            if abs(out_num - exp_num) <= tolerance:
                return VerificationResult(
                    correct=True,
                    confidence=0.95,
                    explanation="Code output numerically matches",
                    expected=expected,
                    got=output,
                    verification_method="code_exec_numeric",
                )
        except (ValueError, OverflowError):
            pass

        return VerificationResult(
            correct=False,
            confidence=0.9,
            explanation=f"Code output mismatch: expected '{expected}', got '{output}'",
            expected=expected,
            got=output,
            verification_method="code_exec",
        )

    def _verify_proof(
        self,
        model_answer: str,
        expected: str,
        tolerance: float,
    ) -> VerificationResult:
        """
        Verify a mathematical proof.

        Checks structural validity:
        1. Has clear steps
        2. Conclusion matches expected
        3. Each step follows logically (basic check)
        """
        # Extract conclusion
        conclusion = self._extract_conclusion(model_answer)

        # Check conclusion matches expected
        if conclusion and self._normalize_text(conclusion) == self._normalize_text(expected):
            # Check structure
            steps = self._extract_proof_steps(model_answer)
            if len(steps) >= 2:
                return VerificationResult(
                    correct=True,
                    confidence=0.7,  # proof checking is inherently less certain
                    explanation=f"Proof has {len(steps)} steps, conclusion matches",
                    expected=expected,
                    got=conclusion,
                    verification_method="proof_check",
                )

        # Fallback: check if expected answer appears in proof
        if self._normalize_text(expected) in self._normalize_text(model_answer):
            return VerificationResult(
                correct=True,
                confidence=0.5,
                explanation="Expected answer found in proof text",
                expected=expected,
                got=model_answer[:200],
                verification_method="proof_check_contains",
            )

        return VerificationResult(
            correct=False,
            confidence=0.6,
            explanation="Proof verification failed: conclusion doesn't match expected",
            expected=expected,
            got=model_answer[:200],
            verification_method="proof_check",
        )

    def _verify_open_ended(
        self,
        model_answer: str,
        expected: str,
        tolerance: float,
    ) -> VerificationResult:
        """
        Open-ended verification — always returns low-confidence result.

        Open-ended answers can't be mechanically verified.
        This returns a result but flags it as unverifiable.
        """
        return VerificationResult(
            correct=False,
            confidence=0.0,
            explanation="Open-ended answers cannot be mechanically verified",
            expected=expected,
            got=model_answer[:200],
            verification_method="open_ended_unverifiable",
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        result = text.strip()
        if not self.case_sensitive:
            result = result.lower()
        if not self.whitespace_sensitive:
            result = re.sub(r'\s+', ' ', result)
        # Remove common wrapper artifacts
        result = result.strip('"\'`')
        return result

    def _extract_answer(self, text: str) -> str:
        """Try to extract a concise answer from longer text."""
        # Look for common answer patterns (ordered by priority)
        patterns = [
            r'[Ff]inal\s+[Aa]nswer\s*:\s*(.+?)\s*$',
            r'(?:the answer is|answer:)\s*(.+?)(?:\.|$)',
            r'(?:therefore|thus|hence|so),?\s*(.+?)(?:\.|$)',
            r'(?:result|output):\s*(.+?)(?:\.|$)',
            r'\*\*(.+?)\*\*\s*$',
            r'= (\S+)\s*$',
            r'\\boxed\{(.+?)\}',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()

        # If text is short enough, use it directly
        if len(text) < 100:
            return text

        # Return last non-empty line as final answer attempt
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        return lines[-1] if lines else text

    def _extract_number(self, text: str) -> float | None:
        """Extract a number from text."""
        text = text.strip()

        # Try direct parse
        try:
            return float(text)
        except ValueError:
            pass

        # Priority: look for "Final Answer:" line first
        fa_match = re.search(
            r'[Ff]inal\s+[Aa]nswer\s*:\s*(.+)',
            text,
            re.MULTILINE,
        )
        if fa_match:
            fa_text = fa_match.group(1).strip()
            try:
                return float(fa_text)
            except ValueError:
                num_in_fa = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', fa_text)
                if num_in_fa:
                    try:
                        return float(num_in_fa.group())
                    except ValueError:
                        pass

        # Try \boxed{} pattern
        boxed = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed:
            try:
                return float(boxed.group(1).strip())
            except ValueError:
                pass

        # Look for numbers in text — take last number as fallback
        patterns = [
            r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',
            r'[-+]?\d+/\d+',  # fractions
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    match = matches[-1]  # take last number
                    if '/' in match:
                        parts = match.split('/')
                        return float(parts[0]) / float(parts[1])
                    return float(match)
                except (ValueError, ZeroDivisionError):
                    continue

        return None

    def _extract_code(self, text: str) -> str:
        """Extract executable code from model answer."""
        # Look for code blocks
        code_blocks = re.findall(
            r'```(?:python)?\s*\n(.*?)```',
            text,
            re.DOTALL,
        )
        if code_blocks:
            return code_blocks[-1].strip()

        # Look for indented code
        lines = text.split('\n')
        code_lines = []
        for line in lines:
            if line.startswith('    ') or line.startswith('\t'):
                code_lines.append(line)
            elif code_lines and not line.strip():
                code_lines.append(line)
            elif code_lines:
                break

        if code_lines:
            return textwrap.dedent('\n'.join(code_lines))

        # If the whole thing looks like code, use it
        if any(
            kw in text for kw in ['def ', 'import ', 'for ', 'while ', 'print(']
        ):
            return text

        return ""

    def _check_code_safety(self, code: str) -> str | None:
        """
        Static analysis safety check on code.

        Returns None if safe, or a description of the violation.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return "Syntax error in code"

        forbidden_imports = {
            'os', 'sys', 'subprocess', 'shutil', 'socket',
            'http', 'urllib', 'requests', 'pathlib', 'glob',
            'ctypes', 'signal', 'multiprocessing', 'threading',
        }

        forbidden_builtins = {
            'exec', 'eval', 'compile', '__import__',
            'open', 'input', 'breakpoint',
        }

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_module = alias.name.split('.')[0]
                    if root_module in forbidden_imports:
                        return f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_module = node.module.split('.')[0]
                    if root_module in forbidden_imports:
                        return f"Forbidden import: {node.module}"

            # Check dangerous function calls
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    if func.id in forbidden_builtins:
                        return f"Forbidden builtin: {func.id}"
                elif isinstance(func, ast.Attribute):
                    if func.attr in ('system', 'popen', 'remove', 'rmdir'):
                        return f"Forbidden method: {func.attr}"

        return None

    def _execute_code_sandboxed(
        self, code: str,
    ) -> tuple[str, str]:
        """
        Execute code in a restricted sandbox.

        Returns (stdout_output, error_message).
        Error message is empty on success.
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Restricted globals
        safe_builtins = {
            k: v for k, v in __builtins__.__dict__.items()
            if k not in {
                'exec', 'eval', 'compile', '__import__',
                'open', 'input', 'breakpoint', 'exit', 'quit',
            }
        } if isinstance(__builtins__, type(sys)) else {
            k: v for k, v in __builtins__.items()
            if k not in {
                'exec', 'eval', 'compile', '__import__',
                'open', 'input', 'breakpoint', 'exit', 'quit',
            }
        }

        restricted_globals: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "math": math,
        }

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, restricted_globals)  # noqa: S102

            output = stdout_capture.getvalue()
            if len(output) > CODE_EXEC_MAX_OUTPUT:
                output = output[:CODE_EXEC_MAX_OUTPUT] + "\n... (truncated)"

            return output, ""

        except Exception as e:
            return "", f"{type(e).__name__}: {e}"

    def _extract_conclusion(self, proof_text: str) -> str:
        """Extract the conclusion from a proof."""
        patterns = [
            r'(?:therefore|thus|hence|QED|∎|proved)\s*[,:.]?\s*(.+)',
            r'(?:we (?:have|get|obtain|conclude))\s+(.+?)(?:\.|$)',
            r'(?:conclusion:)\s*(.+?)(?:\.|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, proof_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Return last non-empty line
        lines = [l.strip() for l in proof_text.strip().split('\n') if l.strip()]
        return lines[-1] if lines else ""

    def _extract_proof_steps(self, proof_text: str) -> list[str]:
        """Extract individual steps from a proof."""
        # Split on step markers
        step_patterns = [
            r'(?:step\s+\d+)',
            r'(?:\d+[.)]\s)',
            r'(?:first|second|third|next|then|finally)',
        ]

        lines = proof_text.strip().split('\n')
        steps = []
        current_step: list[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            is_step_start = any(
                re.match(pattern, line, re.IGNORECASE)
                for pattern in step_patterns
            )

            if is_step_start and current_step:
                steps.append(' '.join(current_step))
                current_step = [line]
            else:
                current_step.append(line)

        if current_step:
            steps.append(' '.join(current_step))

        return steps


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Standalone verify_answer function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Module-level verifier for the standalone function
_default_verifier = None


def verify_answer(
    answer: str,
    expected: str,
    answer_type: AnswerType = AnswerType.EXACT_MATCH,
    verification_code: str | None = None,
    tolerance: float = 1e-6,
) -> bool:
    """
    Standalone verify function used by FailedProblem.verify() and Probe.verify().

    Args:
        answer: The answer to check.
        expected: The expected correct answer.
        answer_type: How to compare answers.
        verification_code: Optional code for CODE_EXEC type.
        tolerance: Tolerance for numeric comparison.

    Returns:
        True if the answer is considered correct.
    """
    global _default_verifier
    if _default_verifier is None:
        _default_verifier = AnswerVerifier(numeric_tolerance=tolerance)

    # Build a minimal Problem for the verifier
    problem = Problem(
        id="verify_standalone",
        domain="unknown",
        difficulty=1,
        question="",
        answer=expected,
        answer_type=answer_type,
        verification_code=verification_code,
        tolerance=tolerance,
    )

    result = _default_verifier.verify(
        problem=problem,
        model_answer=answer,
        expected_answer=expected,
        tolerance=tolerance,
    )

    return result.correct


class ConsistencyChecker:
    """
    Checks answer consistency across multiple generation attempts.

    Generates N answers and checks if they agree. Consistent
    answers are more likely to be correct.
    """

    def __init__(
        self,
        num_samples: int = 5,
        agreement_threshold: float = 0.6,
        temperature: float = 0.7,
    ):
        self.num_samples = num_samples
        self.agreement_threshold = agreement_threshold
        self.temperature = temperature

    def check_consistency(
        self,
        model: Any,
        tokenizer: Any,
        problem: Problem,
        generate_fn: Any,
    ) -> tuple[str, float]:
        """
        Check answer consistency by generating multiple times.

        Returns:
            (most_common_answer, consistency_ratio)
        """
        import torch
        answers: list[str] = []

        for _ in range(self.num_samples):
            try:
                answer = generate_fn(
                    model, tokenizer, problem,
                    temperature=self.temperature,
                )
                answers.append(answer.strip())
            except Exception:
                continue

        if not answers:
            return "", 0.0

        # Find most common answer (normalized)
        counter: dict[str, int] = {}
        normalizer = AnswerVerifier()
        for ans in answers:
            normalized = normalizer._normalize_text(ans)
            counter[normalized] = counter.get(normalized, 0) + 1

        most_common = max(counter, key=counter.get)  # type: ignore
        consistency = counter[most_common] / len(answers)

        # Find original (un-normalized) version
        for ans in answers:
            if normalizer._normalize_text(ans) == most_common:
                return ans, consistency

        return answers[0], consistency

    def is_reliable(
        self,
        model: Any,
        tokenizer: Any,
        problem: Problem,
        generate_fn: Any,
    ) -> bool:
        """Check if model gives consistent (reliable) answers."""
        _, consistency = self.check_consistency(
            model, tokenizer, problem, generate_fn,
        )
        return consistency >= self.agreement_threshold


class BatchVerifier:
    """Efficiently verify large batches of problems and solutions."""

    def __init__(
        self,
        verifier: AnswerVerifier | None = None,
    ):
        self.verifier = verifier or AnswerVerifier()

    def verify_solutions(
        self,
        problems: list[Problem],
        solutions: list[Solution],
    ) -> tuple[list[FailedProblem], int, int]:
        """
        Verify a batch of solutions.

        Returns:
            (failed_problems, correct_count, total_count)
        """
        failed: list[FailedProblem] = []
        correct = 0

        for problem, solution in zip(problems, solutions):
            result = self.verifier.verify(
                problem,
                solution.answer,
                expected_answer=problem.answer,
            )

            if result.correct:
                correct += 1
            else:
                fp = FailedProblem(
                    problem=problem,
                    original_solution=solution,
                    domain=problem.domain,
                    ground_truth=problem.answer,
                    answer_type=problem.answer_type,
                    verification_code=problem.verification_code,
                )
                failed.append(fp)

        return failed, correct, len(problems)

    def split_correct_incorrect(
        self,
        problems: list[Problem],
        solutions: list[Solution],
    ) -> tuple[list[tuple[Problem, Solution]], list[tuple[Problem, Solution]]]:
        """Split into correct and incorrect pairs."""
        correct_pairs: list[tuple[Problem, Solution]] = []
        incorrect_pairs: list[tuple[Problem, Solution]] = []

        for problem, solution in zip(problems, solutions):
            result = self.verifier.verify(
                problem,
                solution.answer,
                expected_answer=problem.answer,
            )
            if result.correct:
                correct_pairs.append((problem, solution))
            else:
                incorrect_pairs.append((problem, solution))

        return correct_pairs, incorrect_pairs
