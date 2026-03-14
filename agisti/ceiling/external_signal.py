"""
Ceiling Breaker Level 1 — External Surgery Signal.

Breaks the information ceiling by using external problems with
ground-truth answers to generate surgery direction signals.

Instead of the model evaluating itself (circular reasoning),
external ground truth determines correct/wrong classification,
allowing the model to learn genuinely new information.

Design: §11.1.1 — External Surgery Signal.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from agisti.types import (
    AnswerType,
    ExternalSignal,
    Problem,
    Solution,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# External Source Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ExternalSourceConfig:
    """Configuration for an external problem source."""
    name: str
    fetcher_class: type[ExternalFetcher]
    max_age_days: int = 30
    max_problems_per_batch: int = 200
    answer_types: list[AnswerType] = field(default_factory=lambda: [
        AnswerType.EXACT_MATCH,
        AnswerType.NUMERIC_RANGE,
        AnswerType.CODE_EXEC,
    ])


@dataclass
class ExternalProblem:
    """A problem with verified ground truth from external source."""
    question: str
    ground_truth: str
    answer_type: AnswerType
    domain: str = ""
    source: str = ""
    tolerance: float = 1e-6
    verification_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_problem(self) -> Problem:
        """Convert to standard Problem type."""
        return Problem(
            question=self.question,
            answer=self.ground_truth,
            answer_type=self.answer_type,
            domain=self.domain,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fetcher Interface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ExternalFetcher(ABC):
    """Base class for external problem fetchers."""

    @abstractmethod
    def fetch_with_ground_truth(
        self,
        count: int,
        max_age_days: int = 30,
    ) -> list[ExternalProblem]:
        """Fetch problems with verified ground truth answers."""
        ...


class FileFetcher(ExternalFetcher):
    """Fetch external problems from JSONL files."""

    def __init__(self, path: str):
        self.path = path
        self._cache: list[ExternalProblem] | None = None

    def fetch_with_ground_truth(
        self,
        count: int,
        max_age_days: int = 30,
    ) -> list[ExternalProblem]:
        if self._cache is None:
            self._cache = self._load()
        return self._cache[:count]

    def _load(self) -> list[ExternalProblem]:
        import json
        from pathlib import Path

        path = Path(self.path)
        if not path.exists():
            logger.warning("External problem file not found: %s", self.path)
            return []

        problems: list[ExternalProblem] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            problems.append(ExternalProblem(
                question=data["question"],
                ground_truth=data["answer"],
                answer_type=AnswerType(data.get("answer_type", "exact_match")),
                domain=data.get("domain", ""),
                source=data.get("source", "file"),
                tolerance=data.get("tolerance", 1e-6),
                verification_code=data.get("verification_code"),
            ))

        return problems


class HuggingFaceFetcher(ExternalFetcher):
    """Fetch problems from HuggingFace datasets."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "test",
        question_field: str = "question",
        answer_field: str = "answer",
        answer_type: AnswerType = AnswerType.EXACT_MATCH,
        config_name: str | None = None,
    ):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split
        self.question_field = question_field
        self.answer_field = answer_field
        self.answer_type = answer_type

    def fetch_with_ground_truth(
        self,
        count: int,
        max_age_days: int = 30,
    ) -> list[ExternalProblem]:
        try:
            from datasets import load_dataset
        except ImportError:
            logger.warning("datasets library not available for HuggingFace fetcher")
            return []

        try:
            ds = load_dataset(self.dataset_name, self.config_name, split=self.split)
        except Exception as e:
            logger.warning("Failed to load dataset %s: %s", self.dataset_name, e)
            return []

        problems: list[ExternalProblem] = []
        for item in ds:
            if len(problems) >= count:
                break
            q = item.get(self.question_field, "")
            a = item.get(self.answer_field, "")
            if q and a:
                problems.append(ExternalProblem(
                    question=str(q),
                    ground_truth=str(a),
                    answer_type=self.answer_type,
                    source=self.dataset_name,
                ))

        return problems


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# External Surgery Signal
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ExternalSurgerySignal:
    """
    Information Ceiling Breaker Level 1.

    Uses external problems with ground-truth answers to determine
    correct/wrong classification, eliminating circular self-evaluation.

    The surgery direction is derived from external information,
    enabling the model to learn genuinely new knowledge beyond
    what's in its weights (novel acquisition, not just latent extraction).
    """

    def __init__(
        self,
        sources: dict[str, ExternalSourceConfig] | None = None,
    ):
        self.sources = sources or {}
        self._seen_hashes: set[str] = set()
        self._fetcher_cache: dict[str, ExternalFetcher] = {}

    def register_source(
        self,
        name: str,
        config: ExternalSourceConfig,
    ) -> None:
        """Register an external problem source."""
        self.sources[name] = config

    def collect_signal(
        self,
        model: nn.Module,
        tokenizer: Any,
        source_name: str,
        count: int,
        trace_layers: list[str],
        max_length: int = 512,
    ) -> ExternalSignal:
        """
        Collect external surgery signal from a named source.

        1. Fetch external problems with ground truth
        2. Model solves them (with activation tracing)
        3. Ground truth classifies correct/wrong (NOT self-evaluation)
        4. Compute activation contrast from classification

        Args:
            model: The model to probe.
            tokenizer: Tokenizer for the model.
            source_name: Which source to fetch from.
            count: Number of problems to use.
            trace_layers: Layers to trace activations for.
            max_length: Max generation length.

        Returns:
            ExternalSignal with activation contrasts (or usable=False).
        """
        if source_name not in self.sources:
            return ExternalSignal(
                usable=False,
                reason=f"Unknown source: {source_name}",
            )

        config = self.sources[source_name]

        # Get or create fetcher
        if source_name not in self._fetcher_cache:
            self._fetcher_cache[source_name] = config.fetcher_class()

        fetcher = self._fetcher_cache[source_name]

        # 1. Fetch external problems
        raw_items = fetcher.fetch_with_ground_truth(
            count=count * 2,  # extra for dedup/filtering
            max_age_days=config.max_age_days,
        )

        # Deduplication by content hash
        problems: list[ExternalProblem] = []
        for item in raw_items:
            h = hashlib.sha256(item.question.encode()).hexdigest()
            if h not in self._seen_hashes:
                self._seen_hashes.add(h)
                problems.append(item)
            if len(problems) >= count:
                break

        if len(problems) < 6:
            return ExternalSignal(
                usable=False,
                reason=f"Too few problems from {source_name}: {len(problems)}",
            )

        # 2. Model solves with activation tracing
        solutions: list[str] = []
        activation_maps: list[dict[str, Tensor]] = []

        hooks: list[torch.utils.hooks.RemovableHook] = []

        for problem in problems:
            acts = self._solve_with_tracing(
                model, tokenizer, problem, trace_layers, max_length,
            )
            solutions.append(acts["answer"])
            activation_maps.append(acts["activations"])

        # 3. Ground truth classification (NOT self-evaluation)
        correct_indices: list[int] = []
        wrong_indices: list[int] = []

        for i, (problem, answer) in enumerate(zip(problems, solutions)):
            is_correct = self._verify_against_ground_truth(problem, answer)
            if is_correct:
                correct_indices.append(i)
            else:
                wrong_indices.append(i)

        if len(correct_indices) < 3 or len(wrong_indices) < 3:
            return ExternalSignal(
                usable=False,
                reason=(
                    f"Insufficient contrast pairs: "
                    f"{len(correct_indices)} correct, {len(wrong_indices)} wrong"
                ),
            )

        # 4. Activation contrast (ground-truth based)
        contrasts: dict[str, Tensor] = {}
        for layer_name in trace_layers:
            correct_acts = torch.stack([
                activation_maps[i][layer_name] for i in correct_indices
                if layer_name in activation_maps[i]
            ])
            wrong_acts = torch.stack([
                activation_maps[i][layer_name] for i in wrong_indices
                if layer_name in activation_maps[i]
            ])

            if correct_acts.shape[0] == 0 or wrong_acts.shape[0] == 0:
                continue

            contrasts[layer_name] = correct_acts.mean(0) - wrong_acts.mean(0)

        if not contrasts:
            return ExternalSignal(
                usable=False,
                reason="No valid activation contrasts computed",
            )

        return ExternalSignal(
            usable=True,
            contrasts=contrasts,
            correct_count=len(correct_indices),
            wrong_count=len(wrong_indices),
            source=source_name,
            problems_used=len(problems),
        )

    def _solve_with_tracing(
        self,
        model: nn.Module,
        tokenizer: Any,
        problem: ExternalProblem,
        trace_layers: list[str],
        max_length: int,
    ) -> dict[str, Any]:
        """
        Solve a problem while capturing layer activations.

        Returns dict with 'answer' and 'activations'.
        """
        activations: dict[str, Tensor] = {}
        hooks: list[torch.utils.hooks.RemovableHook] = []

        def make_hook(name: str):
            def hook_fn(module: nn.Module, input: Any, output: Any) -> None:
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                # Last token activation
                if out.dim() >= 2:
                    activations[name] = out[:, -1, :].detach().clone()
                else:
                    activations[name] = out.detach().clone()
            return hook_fn

        # Install hooks
        for name, module in model.named_modules():
            if name in trace_layers:
                h = module.register_forward_hook(make_hook(name))
                hooks.append(h)

        try:
            # Tokenize and generate
            prompt = f"Question: {problem.question}\nAnswer:"
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=False,
                )

            answer = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

        finally:
            for h in hooks:
                h.remove()

        return {
            "answer": answer,
            "activations": activations,
        }

    def _verify_against_ground_truth(
        self,
        problem: ExternalProblem,
        model_answer: str,
    ) -> bool:
        """
        Verify model's answer against external ground truth.

        This is the key difference from self-evaluation:
        we use the source's ground truth, not the model's own judgment.
        """
        if problem.answer_type == AnswerType.EXACT_MATCH:
            return self._normalize(model_answer) == self._normalize(
                problem.ground_truth,
            )

        elif problem.answer_type == AnswerType.NUMERIC_RANGE:
            try:
                model_val = self._extract_number(model_answer)
                truth_val = float(problem.ground_truth)
                return abs(model_val - truth_val) < problem.tolerance
            except (ValueError, TypeError):
                return False

        elif problem.answer_type == AnswerType.CODE_EXEC:
            if problem.verification_code:
                return self._run_verification_code(
                    problem.verification_code, model_answer,
                )
            return self._normalize(model_answer) == self._normalize(
                problem.ground_truth,
            )

        return False

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison."""
        import re
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        # Remove common prefixes
        for prefix in ["the answer is", "answer:", "result:"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        return text

    @staticmethod
    def _extract_number(text: str) -> float:
        """Extract first number from text."""
        import re
        match = re.search(r"-?\d+\.?\d*", text)
        if match:
            return float(match.group())
        raise ValueError(f"No number in: {text}")

    @staticmethod
    def _run_verification_code(code: str, answer: str) -> bool:
        """
        Run verification code safely.

        The verification code should define a function `verify(answer) -> bool`.
        """
        import ast

        # Safety check: only allow safe AST nodes
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False

        forbidden_names = {
            "exec", "eval", "__import__", "compile",
            "open", "os", "sys", "subprocess", "shutil",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                logger.warning("Forbidden name in verification code: %s", node.id)
                return False
            if isinstance(node, ast.Import | ast.ImportFrom):
                return False

        namespace: dict[str, Any] = {"answer": answer}
        try:
            exec(code, {"__builtins__": {}}, namespace)  # noqa: S102
            verify_fn = namespace.get("verify")
            if callable(verify_fn):
                return bool(verify_fn(answer))
        except Exception:
            pass

        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# External Weight Adapter (§11.1.1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ExternalWeightAdapter:
    """
    Adapts external signal weight based on performance trends.

    Rules:
    - self↑ external↓ → approaching ceiling, increase external weight
    - self↑ external↑ → working well, maintain current weight
    - external↓ → external signal may be noisy, decrease weight
    """

    def __init__(self, initial_weight: float = 0.3):
        self.weight = initial_weight
        self._ext_scores: list[float] = []
        self._self_scores: list[float] = []

    def record(
        self,
        self_score: float,
        external_score: float,
    ) -> None:
        """Record iteration scores."""
        self._self_scores.append(self_score)
        self._ext_scores.append(external_score)

    def adapt(self) -> float:
        """Compute adapted external weight."""
        if len(self._ext_scores) < 20 or len(self._self_scores) < 20:
            return self.weight

        ext_slope = self._compute_slope(self._ext_scores[-20:])
        self_slope = self._compute_slope(self._self_scores[-20:])

        if self_slope > 0.003 and ext_slope < 0.001:
            # Approaching ceiling: increase external weight
            self.weight = min(0.8, self.weight + 0.05)
        elif ext_slope < -0.003:
            # External noise: decrease weight
            self.weight = max(0.1, self.weight - 0.1)

        return self.weight

    @staticmethod
    def _compute_slope(values: list[float]) -> float:
        """Compute linear regression slope."""
        n = len(values)
        if n < 2:
            return 0.0

        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum(
            (i - x_mean) * (y - y_mean)
            for i, y in enumerate(values)
        )
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        return numerator / denominator if denominator > 0 else 0.0
