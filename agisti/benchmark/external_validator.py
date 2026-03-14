"""
DynamicExternalValidator — bridges external benchmark APIs.

Fetches problems from external benchmark services and provides
fresh evaluation data independent of internal benchmarks.
This prevents overfitting to internal benchmark distributions.

Design: §6.3 — External validation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agisti.types import AnswerType, Problem
from agisti.generation.verification import AnswerVerifier, VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class ExternalBenchmarkSpec:
    """Specification for an external benchmark."""
    name: str
    source: str  # "file", "api", "huggingface"
    path_or_url: str
    answer_type: AnswerType = AnswerType.EXACT_MATCH
    domain: str = "general"
    max_problems: int = 500
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExternalValidationResult:
    """Result from validating against an external benchmark."""
    source_name: str
    accuracy: float
    total: int
    correct: int
    domain_scores: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


class ExternalBenchmarkFetcher(ABC):
    """Abstract base for fetching external benchmark data."""

    @abstractmethod
    def fetch(self, spec: ExternalBenchmarkSpec) -> list[dict[str, Any]]:
        """Fetch problems from the external source."""
        ...

    @abstractmethod
    def can_handle(self, source: str) -> bool:
        """Check if this fetcher handles the given source type."""
        ...


class FileBenchmarkFetcher(ExternalBenchmarkFetcher):
    """Fetch benchmarks from local JSONL files."""

    def can_handle(self, source: str) -> bool:
        return source == "file"

    def fetch(self, spec: ExternalBenchmarkSpec) -> list[dict[str, Any]]:
        path = Path(spec.path_or_url)
        if not path.exists():
            logger.warning("Benchmark file not found: %s", path)
            return []

        problems = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    problems.append(data)
                except json.JSONDecodeError:
                    continue

        if spec.max_problems and len(problems) > spec.max_problems:
            problems = problems[:spec.max_problems]

        return problems


class HuggingFaceBenchmarkFetcher(ExternalBenchmarkFetcher):
    """Fetch benchmarks from HuggingFace datasets."""

    def can_handle(self, source: str) -> bool:
        return source == "huggingface"

    def fetch(self, spec: ExternalBenchmarkSpec) -> list[dict[str, Any]]:
        """Fetch from a HuggingFace dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            logger.warning(
                "datasets library not installed, "
                "cannot fetch HuggingFace benchmarks",
            )
            return []

        try:
            config_name = spec.metadata.get("config", None)
            dataset = load_dataset(
                spec.path_or_url,
                config_name,
                split=spec.metadata.get("split", "test"),
            )

            problems = []
            question_field = spec.metadata.get("question_field", "question")
            answer_field = spec.metadata.get("answer_field", "answer")

            for i, item in enumerate(dataset):
                if i >= spec.max_problems:
                    break
                problems.append({
                    "question": str(item[question_field]),
                    "expected_answer": str(item[answer_field]),
                    "domain": spec.domain,
                    "answer_type": spec.answer_type.value,
                })

            return problems

        except Exception as e:
            logger.warning("Failed to fetch HuggingFace dataset: %s", e)
            return []


class DynamicExternalValidator:
    """
    Validates model against external benchmark sources.

    Provides independent validation that prevents overfitting
    to internal benchmarks. Results are weighted separately
    in the signal blender.
    """

    def __init__(
        self,
        benchmark_specs: list[ExternalBenchmarkSpec] | None = None,
        verifier: AnswerVerifier | None = None,
        max_gen_tokens: int = 512,
        cache_dir: str | Path | None = None,
    ):
        self.specs = benchmark_specs or []
        self.verifier = verifier or AnswerVerifier()
        self.max_gen_tokens = max_gen_tokens
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Register fetchers
        self._fetchers: list[ExternalBenchmarkFetcher] = [
            FileBenchmarkFetcher(),
            HuggingFaceBenchmarkFetcher(),
        ]

        # Cache fetched problems
        self._cached_problems: dict[str, list[Problem]] = {}

    def add_spec(self, spec: ExternalBenchmarkSpec) -> None:
        """Add a new external benchmark spec."""
        self.specs.append(spec)

    def register_fetcher(self, fetcher: ExternalBenchmarkFetcher) -> None:
        """Register a custom benchmark fetcher."""
        self._fetchers.append(fetcher)

    def validate(
        self,
        model: Any,
        tokenizer: Any,
        spec_names: list[str] | None = None,
    ) -> dict[str, ExternalValidationResult]:
        """
        Run validation against external benchmarks.

        Args:
            model: Model to validate.
            tokenizer: Tokenizer.
            spec_names: Which specs to run (None = all enabled).

        Returns:
            Dict mapping spec name to validation result.
        """
        import torch

        results: dict[str, ExternalValidationResult] = {}

        for spec in self.specs:
            if not spec.enabled:
                continue
            if spec_names and spec.name not in spec_names:
                continue

            logger.info("Validating against: %s", spec.name)
            start = time.monotonic()

            try:
                problems = self._get_problems(spec)
                if not problems:
                    logger.warning("No problems loaded for %s", spec.name)
                    continue

                correct = 0
                total = 0
                domain_correct: dict[str, int] = {}
                domain_total: dict[str, int] = {}
                errors: list[str] = []

                for problem in problems:
                    total += 1
                    try:
                        answer = self._generate_answer(
                            model, tokenizer, problem,
                        )
                        verification = self.verifier.verify(
                            problem, answer,
                        )
                        if verification.correct:
                            correct += 1
                            d = problem.domain
                            domain_correct[d] = domain_correct.get(d, 0) + 1

                        d = problem.domain
                        domain_total[d] = domain_total.get(d, 0) + 1

                    except Exception as e:
                        logger.debug("Validation error: %s", e)
                        errors.append(str(e))

                elapsed = time.monotonic() - start
                accuracy = correct / total if total > 0 else 0.0

                domain_scores = {
                    d: domain_correct.get(d, 0) / domain_total[d]
                    for d in domain_total
                    if domain_total[d] > 0
                }

                results[spec.name] = ExternalValidationResult(
                    source_name=spec.name,
                    accuracy=accuracy,
                    total=total,
                    correct=correct,
                    domain_scores=domain_scores,
                    elapsed_seconds=elapsed,
                    errors=errors[:10],  # cap error log
                )

                logger.info(
                    "%s: %.1f%% (%d/%d) in %.1fs",
                    spec.name,
                    accuracy * 100,
                    correct,
                    total,
                    elapsed,
                )

            except Exception as e:
                logger.error("Failed to validate %s: %s", spec.name, e)
                results[spec.name] = ExternalValidationResult(
                    source_name=spec.name,
                    accuracy=0.0,
                    total=0,
                    correct=0,
                    errors=[str(e)],
                )

        return results

    def _get_problems(self, spec: ExternalBenchmarkSpec) -> list[Problem]:
        """Get problems for a benchmark spec (with caching)."""
        if spec.name in self._cached_problems:
            return self._cached_problems[spec.name]

        # Try cache_dir
        if self.cache_dir:
            cache_file = self.cache_dir / f"{spec.name}.jsonl"
            if cache_file.exists():
                problems = self._load_cached(cache_file)
                if problems:
                    self._cached_problems[spec.name] = problems
                    return problems

        # Fetch fresh
        fetcher = self._find_fetcher(spec.source)
        if fetcher is None:
            logger.warning("No fetcher for source type: %s", spec.source)
            return []

        raw_data = fetcher.fetch(spec)
        problems = self._convert_to_problems(raw_data, spec)

        # Cache in memory
        self._cached_problems[spec.name] = problems

        # Cache to disk
        if self.cache_dir and problems:
            self._save_cache(problems, spec)

        return problems

    def _find_fetcher(self, source: str) -> ExternalBenchmarkFetcher | None:
        """Find a fetcher that handles the given source type."""
        for fetcher in self._fetchers:
            if fetcher.can_handle(source):
                return fetcher
        return None

    def _convert_to_problems(
        self,
        raw_data: list[dict[str, Any]],
        spec: ExternalBenchmarkSpec,
    ) -> list[Problem]:
        """Convert raw fetched data to Problem objects."""
        problems = []
        for item in raw_data:
            try:
                answer_type = AnswerType(
                    item.get("answer_type", spec.answer_type.value),
                )
                problem = Problem(
                    question=str(item["question"]),
                    domain=item.get("domain", spec.domain),
                    answer_type=answer_type,
                    difficulty=float(item.get("difficulty", 0.5)),
                    metadata={
                        "expected_answer": str(
                            item.get("expected_answer", ""),
                        ),
                        "tolerance": float(item.get("tolerance", 0.0)),
                        "source": spec.name,
                    },
                )
                problems.append(problem)
            except (KeyError, ValueError) as e:
                logger.debug("Skipping malformed external problem: %s", e)

        return problems

    def _load_cached(self, path: Path) -> list[Problem]:
        """Load cached problems from JSONL."""
        problems = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    problems.append(Problem(
                        question=data["question"],
                        domain=data["domain"],
                        answer_type=AnswerType(data["answer_type"]),
                        difficulty=data.get("difficulty", 0.5),
                        metadata=data.get("metadata", {}),
                    ))
                except (KeyError, ValueError):
                    continue
        return problems

    def _save_cache(
        self,
        problems: list[Problem],
        spec: ExternalBenchmarkSpec,
    ) -> None:
        """Save problems to disk cache."""
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{spec.name}.jsonl"

        with cache_file.open("w", encoding="utf-8") as f:
            for p in problems:
                data = {
                    "question": p.question,
                    "domain": p.domain,
                    "answer_type": p.answer_type.value,
                    "difficulty": p.difficulty,
                    "metadata": p.metadata,
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _generate_answer(
        self,
        model: Any,
        tokenizer: Any,
        problem: Problem,
    ) -> str:
        """Generate answer for external validation."""
        import torch

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
