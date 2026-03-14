"""
ActiveProber — measures model competency through targeted probing.

The prober maintains a bank of calibrated problems and uses them
to measure per-domain accuracy. Probes are distinct from training
data to avoid circular evaluation (§5 design constraint).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from agisti.types import (
    AnswerType,
    FailedProblem,
    Problem,
    Probe,
    Solution,
    ErrorReport,
    VERIFIABLE_TYPES,
)
from agisti.probe.competency import CompetencyVector, CompetencyTracker
from agisti.probe.weakness import WeaknessAnalyzer, FailurePatternDetector

logger = logging.getLogger(__name__)


@dataclass
class ProbeBank:
    """
    A collection of calibrated probe problems organized by domain.

    Each probe has a known correct answer that can be verified mechanically.
    The bank is loaded from files and is immutable during a training run
    to prevent data contamination.
    """

    probes: dict[str, list[Probe]] = field(default_factory=dict)

    @property
    def domains(self) -> list[str]:
        return list(self.probes.keys())

    @property
    def total_probes(self) -> int:
        return sum(len(ps) for ps in self.probes.values())

    def get_probes(
        self,
        domain: str,
        count: int | None = None,
        seed: int | None = None,
    ) -> list[Probe]:
        """Get probes for a domain, optionally sampled."""
        available = self.probes.get(domain, [])
        if not available:
            return []
        if count is None or count >= len(available):
            return list(available)

        rng = random.Random(seed)
        return rng.sample(available, min(count, len(available)))

    def get_all_probes(self) -> list[Probe]:
        """Get all probes across all domains."""
        all_probes = []
        for domain_probes in self.probes.values():
            all_probes.extend(domain_probes)
        return all_probes

    def add_probe(self, probe: Probe) -> None:
        """Add a probe to the bank."""
        if probe.domain not in self.probes:
            self.probes[probe.domain] = []
        self.probes[probe.domain].append(probe)

    def domain_coverage(self) -> dict[str, int]:
        """Number of probes per domain."""
        return {d: len(ps) for d, ps in self.probes.items()}

    @classmethod
    def from_jsonl(cls, path: str | Path) -> ProbeBank:
        """Load probe bank from a JSONL file."""
        bank = cls()
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    probe = Probe(
                        id=data.get("id", hashlib.sha256(line.encode()).hexdigest()[:12]),
                        domain=data["domain"],
                        question=data["question"],
                        expected_answer=data["expected_answer"],
                        answer_type=AnswerType(data.get("answer_type", "exact_match")),
                        tolerance=data.get("tolerance", 0.0),
                    )
                    bank.add_probe(probe)
                except (KeyError, ValueError) as e:
                    logger.warning(
                        "Failed to parse probe at line %d: %s", line_num, e,
                    )

        logger.info(
            "Loaded probe bank: %d probes across %d domains",
            bank.total_probes,
            len(bank.domains),
        )
        return bank

    def save_jsonl(self, path: str | Path) -> None:
        """Save probe bank to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for domain, probes in sorted(self.probes.items()):
                for probe in probes:
                    data = {
                        "id": probe.id,
                        "question": probe.question,
                        "domain": probe.domain,
                        "answer_type": probe.answer_type.value,
                        "expected_answer": probe.expected_answer,
                        "tolerance": probe.tolerance,
                    }
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")


class ActiveProber:
    """
    Actively probes the model to measure competency.

    The probing process:
    1. Select probes for each domain from the probe bank
    2. Generate model answers for each probe
    3. Verify answers against known correct answers
    4. Update the competency vector with results
    5. Analyze failure patterns for surgery guidance

    Key design decision from §5: probes are separate from training data
    to avoid circular evaluation (training on what you test = meaningless).
    """

    def __init__(
        self,
        probe_bank: ProbeBank,
        probes_per_domain: int = 20,
        max_gen_tokens: int = 512,
        temperature: float = 0.0,  # greedy for evaluation
        ema_alpha: float = 0.3,
    ):
        self.probe_bank = probe_bank
        self.probes_per_domain = probes_per_domain
        self.max_gen_tokens = max_gen_tokens
        self.temperature = temperature

        self.competency = CompetencyVector(ema_alpha=ema_alpha)
        self.tracker = CompetencyTracker()
        self.weakness_analyzer = WeaknessAnalyzer()
        self.pattern_detector = FailurePatternDetector()

        self._iteration = 0

    def probe_all_domains(
        self,
        model: Any,
        tokenizer: Any,
        seed: int | None = None,
    ) -> tuple[CompetencyVector, list[FailedProblem]]:
        """
        Probe model across all domains and update competency.

        Args:
            model: The language model to probe.
            tokenizer: Tokenizer for the model.
            seed: Random seed for probe selection (epoch-rotated).

        Returns:
            Updated CompetencyVector and list of failed problems.
        """
        self._iteration += 1
        effective_seed = seed if seed is not None else self._iteration * 1337 + 42

        all_failed: list[FailedProblem] = []
        new_scores: dict[str, float] = {}

        for domain in self.probe_bank.domains:
            probes = self.probe_bank.get_probes(
                domain,
                count=self.probes_per_domain,
                seed=effective_seed,
            )
            if not probes:
                continue

            correct, total, failed = self._evaluate_probes(
                model, tokenizer, probes,
            )

            new_scores[domain] = correct / total if total > 0 else 0.0
            all_failed.extend(failed)

            logger.info(
                "Domain %s: %d/%d correct (%.1f%%)",
                domain,
                correct,
                total,
                100 * correct / total if total > 0 else 0,
            )

        # Update competency vector with EMA
        self.competency.update(new_scores)
        self.tracker.record(self.competency)

        # Record failure patterns
        self.pattern_detector.record_failures(all_failed)

        logger.info(
            "Probing complete: overall %.1f%%, %d failures",
            self.competency.get_overall_score() * 100,
            len(all_failed),
        )

        return self.competency, all_failed

    def probe_domains(
        self,
        model: Any,
        tokenizer: Any,
        domains: list[str],
        seed: int | None = None,
    ) -> tuple[dict[str, float], list[FailedProblem]]:
        """Probe specific domains only."""
        effective_seed = seed if seed is not None else self._iteration * 1337 + 42

        all_failed: list[FailedProblem] = []
        scores: dict[str, float] = {}

        for domain in domains:
            probes = self.probe_bank.get_probes(
                domain,
                count=self.probes_per_domain,
                seed=effective_seed,
            )
            if not probes:
                continue

            correct, total, failed = self._evaluate_probes(
                model, tokenizer, probes,
            )
            scores[domain] = correct / total if total > 0 else 0.0
            all_failed.extend(failed)

        self.competency.update(scores)
        return scores, all_failed

    def measure_targeted(
        self,
        model: Any,
        tokenizer: Any,
        probes: list[Probe],
    ) -> tuple[int, int, list[FailedProblem]]:
        """
        Evaluate a specific set of probes.
        Returns (correct, total, failed).
        """
        return self._evaluate_probes(model, tokenizer, probes)

    def get_weakness_report(
        self, failed: list[FailedProblem],
    ) -> Any:
        """Generate a weakness report from recent failures."""
        return self.weakness_analyzer.analyze(
            self.competency,
            failed,
        )

    def check_regression(
        self,
        previous_competency: CompetencyVector | None = None,
    ) -> dict[str, dict[str, float]]:
        """Check for regressions since previous measurement."""
        if previous_competency is None and len(self.tracker.history) >= 2:
            previous_competency = self.tracker.history[-2]
        if previous_competency is None:
            return {}
        return self.competency.check_regression(previous_competency)

    def is_converging(self) -> bool:
        """Check if model improvement has converged."""
        return self.tracker.is_converging()

    def _evaluate_probes(
        self,
        model: Any,
        tokenizer: Any,
        probes: list[Probe],
    ) -> tuple[int, int, list[FailedProblem]]:
        """
        Run probes through the model and verify answers.

        Returns:
            (correct_count, total_count, list_of_failures)
        """
        correct = 0
        total = 0
        failed: list[FailedProblem] = []

        for probe in probes:
            total += 1
            try:
                answer = self._generate_answer(
                    model, tokenizer, probe.problem,
                )
                is_correct = probe.verify(answer)

                if is_correct:
                    correct += 1
                else:
                    fp = FailedProblem(
                        problem=probe.problem,
                        original_solution=Solution(
                            problem_id=probe.id,
                            answer=str(answer),
                            chain_of_thought="",
                            tokens_generated=0,
                            generation_time_seconds=0.0,
                        ),
                        domain=probe.domain,
                        ground_truth=str(probe.expected_answer),
                        answer_type=probe.answer_type,
                        verification_code=probe.verification_code,
                    )
                    failed.append(fp)

            except Exception as e:
                logger.warning("Probe evaluation error: %s", e)
                fp = FailedProblem(
                    problem=probe.problem,
                    original_solution=Solution(
                        problem_id=probe.id,
                        answer="<error>",
                        chain_of_thought=str(e),
                        tokens_generated=0,
                        generation_time_seconds=0.0,
                    ),
                    domain=probe.domain,
                    ground_truth=str(probe.expected_answer),
                    answer_type=probe.answer_type,
                    verification_code=probe.verification_code,
                )
                failed.append(fp)

        return correct, total, failed

    def _generate_answer(
        self,
        model: Any,
        tokenizer: Any,
        problem: Problem,
    ) -> str:
        """Generate an answer from the model for a problem."""
        from agisti.generation.prompt_utils import format_for_model

        prompt = self._format_probe_prompt(problem)
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

        # Decode only the new tokens
        new_tokens = outputs[0][input_ids.shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return answer

    def _format_probe_prompt(self, problem: Problem) -> str:
        """Format a problem as a prompt for the model."""
        type_instruction = {
            AnswerType.EXACT_MATCH: "Give the exact answer.",
            AnswerType.NUMERIC_RANGE: "Give a numerical answer.",
            AnswerType.CODE_EXEC: "Write executable code that solves this.",
            AnswerType.PROOF_CHECK: "Provide a step-by-step proof.",
            AnswerType.OPEN_ENDED: "Answer the question.",
        }
        instruction = type_instruction.get(
            problem.answer_type, "Answer the question.",
        )
        return f"Question: {problem.question}\n\n{instruction}\n\nAnswer:"


class ProbeScheduler:
    """
    Schedules probing frequency and domain focus.

    Adaptively chooses which domains to probe more frequently
    based on their stability and performance.
    """

    def __init__(
        self,
        full_probe_interval: int = 10,
        weak_probe_interval: int = 3,
        max_domains_per_quick_probe: int = 5,
    ):
        self.full_probe_interval = full_probe_interval
        self.weak_probe_interval = weak_probe_interval
        self.max_quick_domains = max_domains_per_quick_probe

    def should_full_probe(self, iteration: int) -> bool:
        """Whether to probe ALL domains this iteration."""
        return iteration % self.full_probe_interval == 0

    def select_domains_for_quick_probe(
        self,
        competency: CompetencyVector,
        iteration: int,
    ) -> list[str]:
        """
        Select domains for a quick probe.

        Prioritizes:
        1. Weakest domains (most room for improvement)
        2. Recently declining domains (catching regressions)
        3. Rotating through other domains for coverage
        """
        if not competency.domains:
            return []

        # Get weakest domains
        weakest = competency.get_weakest_domains(
            top_k=self.max_quick_domains // 2,
        )
        selected = [d for d, _ in weakest]

        # Add declining domains
        for domain in competency.domains:
            if len(selected) >= self.max_quick_domains:
                break
            trend = competency.get_trend(domain)
            if trend < -0.005 and domain not in selected:
                selected.append(domain)

        # Fill remaining with rotation
        remaining = [d for d in competency.domains if d not in selected]
        if remaining:
            offset = iteration % len(remaining)
            rotated = remaining[offset:] + remaining[:offset]
            for d in rotated:
                if len(selected) >= self.max_quick_domains:
                    break
                selected.append(d)

        return selected


class ProbeBankBuilder:
    """
    Builds probe banks from various data sources.

    Ensures probes are:
    1. Verifiable (only VERIFIABLE_TYPES)
    2. Calibrated (difficulty is measured, not assumed)
    3. Diverse (covers multiple domains and difficulty levels)
    """

    def __init__(self, min_probes_per_domain: int = 20):
        self.min_per_domain = min_probes_per_domain
        self._probes: list[Probe] = []

    def add_verified_problem(
        self,
        question: str,
        expected_answer: str,
        domain: str,
        answer_type: AnswerType,
        difficulty: float = 0.5,
        tolerance: float = 0.0,
    ) -> None:
        """Add a single verified probe."""
        if answer_type not in VERIFIABLE_TYPES and answer_type != AnswerType.EXACT_MATCH:
            logger.warning(
                "Probe answer type %s is not mechanically verifiable, "
                "adding anyway with exact match semantics",
                answer_type,
            )

        probe = Probe(
            problem=Problem(
                question=question,
                domain=domain,
                answer_type=answer_type,
                difficulty=difficulty,
            ),
            expected_answer=expected_answer,
            tolerance=tolerance,
        )
        self._probes.append(probe)

    def build(self) -> ProbeBank:
        """Build a ProbeBank from accumulated probes."""
        bank = ProbeBank()
        for probe in self._probes:
            bank.add_probe(probe)

        # Validate coverage
        coverage = bank.domain_coverage()
        for domain, count in coverage.items():
            if count < self.min_per_domain:
                logger.warning(
                    "Domain %s has only %d probes (minimum: %d)",
                    domain,
                    count,
                    self.min_per_domain,
                )

        return bank

    def from_qa_pairs(
        self,
        pairs: list[dict[str, Any]],
        domain: str,
        answer_type: AnswerType = AnswerType.EXACT_MATCH,
    ) -> None:
        """Bulk add from Q&A pairs."""
        for pair in pairs:
            self.add_verified_problem(
                question=pair["question"],
                expected_answer=pair["answer"],
                domain=domain,
                answer_type=answer_type,
                difficulty=pair.get("difficulty", 0.5),
                tolerance=pair.get("tolerance", 0.0),
            )

    def calibrate_difficulty(
        self,
        model: Any,
        tokenizer: Any,
        num_runs: int = 3,
    ) -> None:
        """
        Calibrate probe difficulty by running them through a reference model.

        A probe's difficulty is set to (1 - accuracy_over_runs).
        """
        prober = ActiveProber(
            probe_bank=ProbeBank(),
            probes_per_domain=100,
        )

        for probe in self._probes:
            successes = 0
            for run in range(num_runs):
                try:
                    answer = prober._generate_answer(
                        model, tokenizer, probe.problem,
                    )
                    if probe.verify(answer):
                        successes += 1
                except Exception:
                    pass  # counts as failure

            probe.problem.difficulty = 1.0 - (successes / num_runs)
