"""
Ceiling Breaker Level 4 — Compositional Discovery.

The highest level of ceiling breaking: discovering novel compositional
patterns that no single source (self, external, RAG, reference models)
provides directly.

Key insight: when the model has a skill A and skill B separately,
but fails at problems requiring A+B together, the "compositional gap"
can be identified and surgically addressed.

This module:
1. Identifies existing competencies from probe results
2. Generates compositional problems requiring skill combinations
3. Finds which combinations fail (competency gap)
4. Targets surgery specifically at the combination pathway

Design: §11.1.4 (conceptual) — Compositional Discovery.
"""

from __future__ import annotations

import itertools
import logging
import random
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from agisti.types import AnswerType, Problem

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Competency Pair Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class CompetencyPair:
    """A pair of skills that may have a compositional gap."""
    skill_a: str
    skill_b: str
    individual_acc_a: float  # Accuracy on skill A alone
    individual_acc_b: float  # Accuracy on skill B alone
    combined_acc: float  # Accuracy on A+B together
    gap: float  # min(individual) - combined; >0 = gap exists

    @property
    def has_gap(self) -> bool:
        """True if combined is worse than individual skills predict."""
        return self.gap > 0.1  # Meaningful gap threshold

    @property
    def gap_severity(self) -> float:
        """Severity normalized to [0, 1]."""
        expected = min(self.individual_acc_a, self.individual_acc_b)
        if expected < 0.01:
            return 0.0
        return max(0.0, min(1.0, (expected - self.combined_acc) / expected))


@dataclass
class CompositionalGap:
    """Detailed analysis of a compositional gap."""
    pair: CompetencyPair
    sample_problems: list[Problem]
    failure_patterns: dict[str, int]
    activation_contrast: dict[str, Tensor] | None = None
    surgical_priority: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Compositional Problem Generator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# Templates for compositional problems
COMPOSITION_TEMPLATES: dict[tuple[str, str], list[str]] = {
    ("math", "logic"): [
        "If x² + y² = {a} and x > y > 0 and x + y is odd, "
        "what are the possible values of x + y?",
        "A sequence follows the rule: if the term is even, divide by 2; "
        "if odd, multiply by {a} and add 1. Starting from {b}, "
        "does the sequence eventually reach 1?",
    ],
    ("math", "coding"): [
        "Write a function that computes the {a}-th Fibonacci number "
        "modulo {b}. What is the output for n={c}?",
        "Given a matrix [[{a},{b}],[{c},{d}]], compute the determinant "
        "and verify with a single line of Python code.",
    ],
    ("logic", "knowledge"): [
        "If all mammals are warm-blooded, and whales are mammals, "
        "what can we conclude about the thermoregulation mechanism "
        "whales use in cold ocean water?",
        "Using the principle of excluded middle, determine whether "
        "the following statement about {topic} is necessarily true, "
        "necessarily false, or contingent.",
    ],
    ("coding", "knowledge"): [
        "Write a function to validate {format} format according to "
        "the official specification. What edge case does most implementations miss?",
        "Implement the {algorithm} algorithm and compute its time complexity "
        "for an input of size {n}.",
    ],
    ("math", "knowledge"): [
        "In {field}, the relationship between {concept_a} and {concept_b} "
        "follows a {relation} law. Express this mathematically and compute "
        "the value when {var} = {val}.",
    ],
}


class CompositionalProblemGenerator:
    """
    Generates problems that require combining two distinct skills.

    Uses templates and parameterization to create problems that
    test compositional reasoning — the ability to apply multiple
    learned skills simultaneously.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._templates = dict(COMPOSITION_TEMPLATES)

    def add_template(
        self,
        skill_pair: tuple[str, str],
        template: str,
    ) -> None:
        """Add a custom composition template."""
        key = tuple(sorted(skill_pair))
        if key not in self._templates:
            self._templates[key] = []
        self._templates[key].append(template)

    def generate(
        self,
        skill_a: str,
        skill_b: str,
        count: int = 10,
        difficulty: int = 1,
    ) -> list[Problem]:
        """
        Generate compositional problems combining two skills.

        Args:
            skill_a: First skill domain.
            skill_b: Second skill domain.
            count: Number of problems to generate.
            difficulty: 1-5 difficulty scale.

        Returns:
            List of Problems requiring both skills.
        """
        key = tuple(sorted((skill_a, skill_b)))
        templates = self._templates.get(key, [])

        if not templates:
            # Fallback: generate generic compositional prompt
            return self._generate_generic(skill_a, skill_b, count, difficulty)

        problems: list[Problem] = []
        for _ in range(count):
            template = self.rng.choice(templates)
            params = self._generate_params(difficulty)
            try:
                question = template.format(**params)
            except (KeyError, IndexError):
                question = template  # Use raw if params don't match

            problems.append(Problem(
                question=question,
                answer="",  # Compositional problems need model or external verification
                answer_type=AnswerType.EXACT_MATCH,
                domain=f"{skill_a}+{skill_b}",
                difficulty=difficulty,
            ))

        return problems

    def _generate_generic(
        self,
        skill_a: str,
        skill_b: str,
        count: int,
        difficulty: int,
    ) -> list[Problem]:
        """Generate generic compositional problems."""
        prompts = [
            (
                f"This problem requires both {skill_a} and {skill_b} skills. "
                f"Difficulty level {difficulty}/5.\n"
                f"Consider a scenario where {skill_a} concepts must be applied "
                f"using {skill_b} reasoning to arrive at the answer."
            ),
        ]

        problems: list[Problem] = []
        for i in range(count):
            prompt = self.rng.choice(prompts)
            problems.append(Problem(
                question=prompt,
                answer="",
                answer_type=AnswerType.EXACT_MATCH,
                domain=f"{skill_a}+{skill_b}",
                difficulty=difficulty,
            ))

        return problems

    def _generate_params(self, difficulty: int) -> dict[str, Any]:
        """Generate random parameters scaled by difficulty."""
        scale = 2 ** difficulty
        return {
            "a": self.rng.randint(2, 10 * scale),
            "b": self.rng.randint(2, 10 * scale),
            "c": self.rng.randint(1, 5 * scale),
            "d": self.rng.randint(1, 5 * scale),
            "n": self.rng.randint(5, 20 * scale),
            "val": round(self.rng.uniform(0.1, 10.0 * scale), 2),
            "var": self.rng.choice(["x", "t", "n", "k"]),
            "topic": self.rng.choice([
                "quantum mechanics", "graph theory",
                "thermodynamics", "number theory",
            ]),
            "format": self.rng.choice(["JSON", "XML", "CSV", "YAML"]),
            "algorithm": self.rng.choice([
                "Dijkstra's", "A*", "QuickSort", "KMP",
            ]),
            "field": self.rng.choice([
                "physics", "chemistry", "biology", "economics",
            ]),
            "concept_a": "energy",
            "concept_b": "frequency",
            "relation": self.rng.choice(["linear", "inverse-square", "exponential"]),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Compositional Discovery Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CompositionalDiscovery:
    """
    Discovers and addresses compositional capability gaps.

    Pipeline:
    1. Identify individual skill competencies from probe data
    2. Generate pairwise compositional tests
    3. Find pairs where combined < individual (gap)
    4. Generate activation contrasts for gaps
    5. Prioritize gaps by severity and surgibility

    This is the most speculative ceiling breaker level —
    it attempts to discover emergent compositional abilities
    that don't arise naturally from training.
    """

    def __init__(
        self,
        domains: list[str],
        generator: CompositionalProblemGenerator | None = None,
        min_gap: float = 0.1,
        problems_per_pair: int = 20,
    ):
        self.domains = domains
        self.generator = generator or CompositionalProblemGenerator()
        self.min_gap = min_gap
        self.problems_per_pair = problems_per_pair
        self._individual_accs: dict[str, float] = {}
        self._pair_accs: dict[tuple[str, str], float] = {}
        self._gaps: list[CompositionalGap] = []

    def set_individual_accuracies(
        self, accuracies: dict[str, float],
    ) -> None:
        """Set per-domain individual accuracies from probe results."""
        self._individual_accs = dict(accuracies)

    def discover_gaps(
        self,
        model: nn.Module,
        tokenizer: Any,
        trace_layers: list[str],
        max_pairs: int = 10,
    ) -> list[CompositionalGap]:
        """
        Discover compositional gaps by testing skill combinations.

        Tests pairwise combinations of domains where both individual
        skills are reasonably strong (>40% accuracy) to find combinations
        that fail unexpectedly.

        Args:
            model: The model to test.
            tokenizer: Tokenizer for the model.
            trace_layers: Layers to trace for surgery signal.
            max_pairs: Maximum number of pairs to test.

        Returns:
            List of CompositionalGaps, sorted by severity.
        """
        # Filter for domains with reasonable individual competency
        eligible = [
            d for d, acc in self._individual_accs.items()
            if acc >= 0.4
        ]

        if len(eligible) < 2:
            logger.info("Fewer than 2 eligible domains for compositional testing")
            return []

        # Generate all pairs
        pairs = list(itertools.combinations(eligible, 2))
        if len(pairs) > max_pairs:
            pairs = random.sample(pairs, max_pairs)

        gaps: list[CompositionalGap] = []

        for skill_a, skill_b in pairs:
            gap = self._test_pair(
                model, tokenizer, skill_a, skill_b, trace_layers,
            )
            if gap is not None:
                gaps.append(gap)

        # Sort by severity (highest first)
        gaps.sort(key=lambda g: -g.pair.gap_severity)
        self._gaps = gaps

        logger.info(
            "Compositional discovery: tested %d pairs, found %d gaps",
            len(pairs), len(gaps),
        )

        return gaps

    def _test_pair(
        self,
        model: nn.Module,
        tokenizer: Any,
        skill_a: str,
        skill_b: str,
        trace_layers: list[str],
    ) -> CompositionalGap | None:
        """Test a specific skill pair for compositional gap."""
        # Generate compositional problems
        problems = self.generator.generate(
            skill_a, skill_b,
            count=self.problems_per_pair,
            difficulty=2,
        )

        if not problems:
            return None

        # Solve and track correct/wrong + activations
        correct_count = 0
        correct_acts: list[dict[str, Tensor]] = []
        wrong_acts: list[dict[str, Tensor]] = []
        failure_patterns: dict[str, int] = {}

        for problem in problems:
            answer, acts = self._solve_traced(
                model, tokenizer, problem, trace_layers,
            )

            # Since these are compositional, we need to approximate correctness
            is_correct = self._approximate_correctness(problem, answer)

            if is_correct:
                correct_count += 1
                correct_acts.append(acts)
            else:
                wrong_acts.append(acts)
                pattern = self._classify_failure(answer)
                failure_patterns[pattern] = failure_patterns.get(pattern, 0) + 1

        combined_acc = correct_count / max(len(problems), 1)
        individual_a = self._individual_accs.get(skill_a, 0.5)
        individual_b = self._individual_accs.get(skill_b, 0.5)
        gap = min(individual_a, individual_b) - combined_acc

        pair_key = tuple(sorted((skill_a, skill_b)))
        self._pair_accs[pair_key] = combined_acc

        pair = CompetencyPair(
            skill_a=skill_a,
            skill_b=skill_b,
            individual_acc_a=individual_a,
            individual_acc_b=individual_b,
            combined_acc=combined_acc,
            gap=gap,
        )

        if not pair.has_gap:
            return None

        # Compute activation contrast if we have enough data
        contrast: dict[str, Tensor] | None = None
        if len(correct_acts) >= 2 and len(wrong_acts) >= 2:
            contrast = {}
            for layer in trace_layers:
                c_tensors = [
                    a[layer] for a in correct_acts if layer in a
                ]
                w_tensors = [
                    a[layer] for a in wrong_acts if layer in a
                ]
                if c_tensors and w_tensors:
                    contrast[layer] = (
                        torch.stack(c_tensors).mean(0)
                        - torch.stack(w_tensors).mean(0)
                    )

        return CompositionalGap(
            pair=pair,
            sample_problems=problems,
            failure_patterns=failure_patterns,
            activation_contrast=contrast,
            surgical_priority=pair.gap_severity,
        )

    def _solve_traced(
        self,
        model: nn.Module,
        tokenizer: Any,
        problem: Problem,
        trace_layers: list[str],
    ) -> tuple[str, dict[str, Tensor]]:
        """Solve with activation tracing."""
        activations: dict[str, Tensor] = {}
        hooks: list[torch.utils.hooks.RemovableHook] = []

        def make_hook(name: str):
            def hook_fn(module: nn.Module, input: Any, output: Any) -> None:
                out = output[0] if isinstance(output, tuple) else output
                if out.dim() >= 2:
                    activations[name] = out[:, -1, :].detach().clone()
                else:
                    activations[name] = out.detach().clone()
            return hook_fn

        for name, module in model.named_modules():
            if name in trace_layers:
                hooks.append(module.register_forward_hook(make_hook(name)))

        try:
            prompt = f"Question: {problem.question}\nAnswer:"
            inputs = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=2048,
            )
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=256, do_sample=False,
                )

            answer = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
        finally:
            for h in hooks:
                h.remove()

        return answer, activations

    def _approximate_correctness(
        self,
        problem: Problem,
        answer: str,
    ) -> bool:
        """
        Approximate whether a compositional answer is correct.

        For compositional problems without ground truth, use heuristics:
        - Non-empty answer
        - Contains relevant reasoning
        - Not a refusal/hedge
        """
        if not answer or len(answer) < 10:
            return False

        # Check for refusals
        refusal_patterns = [
            "i cannot", "i can't", "i'm not sure",
            "this is too", "i don't know",
        ]
        lower = answer.lower()
        if any(p in lower for p in refusal_patterns):
            return False

        # Check that answer actually attempts the problem
        if problem.answer and problem.verify(answer):
            return True

        # Heuristic: answer is substantive
        return len(answer) > 50

    @staticmethod
    def _classify_failure(answer: str) -> str:
        """Classify failure mode."""
        if not answer:
            return "empty"
        lower = answer.lower()
        if any(r in lower for r in ["i cannot", "i can't", "unable"]):
            return "refusal"
        if len(answer) < 20:
            return "too_short"
        if "?" in answer and answer.count("?") > answer.count("."):
            return "confused"
        return "wrong_answer"

    def get_priority_gaps(self, top_k: int = 5) -> list[CompositionalGap]:
        """Get the highest-priority gaps for surgery targeting."""
        return self._gaps[:top_k]

    def get_gap_report(self) -> dict[str, Any]:
        """Generate a summary report of discovered gaps."""
        return {
            "total_gaps": len(self._gaps),
            "domains_tested": len(self._individual_accs),
            "pairs_tested": len(self._pair_accs),
            "gaps": [
                {
                    "skills": f"{g.pair.skill_a}+{g.pair.skill_b}",
                    "individual_a": g.pair.individual_acc_a,
                    "individual_b": g.pair.individual_acc_b,
                    "combined": g.pair.combined_acc,
                    "gap": g.pair.gap,
                    "severity": g.pair.gap_severity,
                    "failure_patterns": g.failure_patterns,
                    "has_contrast": g.activation_contrast is not None,
                }
                for g in self._gaps
            ],
        }
