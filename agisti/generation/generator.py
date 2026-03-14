"""
Problem generator — creates training problems for targeted surgery.

Generates problems focused on the model's weak domains, using
structured prompts and difficulty adaptation. Only generates
problems with VERIFIABLE answer types (§5 design constraint).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any

import torch

from agisti.types import (
    AnswerType,
    Problem,
    VERIFIABLE_TYPES,
    InsufficientVerifiableProblems,
    WeaknessReport,
)
from agisti.config import IterationConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates for structured problem generation
# ---------------------------------------------------------------------------

PROBLEM_GEN_SYSTEM = """You are a problem generator for AI training.
Generate problems that test specific capabilities.

CRITICAL RULES:
1. Every problem MUST have a single, deterministic correct answer.
2. The answer must be verifiable without subjective judgment.
3. Problems should target the specified domain and difficulty level.
4. Output MUST be valid JSON.

Answer types you can use:
- exact_match: The answer is a specific string or value.
- numeric_range: The answer is a number within a tolerance.
- code_exec: The answer is code that produces a specific output.
- proof_check: The answer is a mathematical proof with checkable steps.
DO NOT use: open_ended (not verifiable)."""

PROBLEM_GEN_USER = """Generate {count} problems for domain: {domain}
Target difficulty: {difficulty:.2f} (0=trivial, 1=expert)
Focus areas: {focus}

Output format (JSON array):
[
  {{
    "question": "...",
    "answer_type": "exact_match|numeric_range|code_exec|proof_check",
    "expected_answer": "...",
    "tolerance": 0.0,
    "difficulty": 0.5,
    "tags": ["tag1", "tag2"]
  }}
]

Generate exactly {count} problems. Be creative and challenging.
Output ONLY the JSON array — no explanations, no markdown fences, no comments."""

FOLLOW_UP_GEN = """The model consistently fails at these types of problems:
{failure_examples}

Generate {count} NEW problems that test the same capability from
different angles. Make them progressively harder. The problems must
be DIFFERENT from the examples above."""


@dataclass
class GenerationRequest:
    """Specification for what to generate."""
    domain: str
    count: int
    difficulty: float
    focus_areas: list[str] = field(default_factory=list)
    failure_examples: list[str] = field(default_factory=list)
    answer_types: list[AnswerType] = field(default_factory=lambda: [
        AnswerType.EXACT_MATCH,
        AnswerType.NUMERIC_RANGE,
        AnswerType.CODE_EXEC,
    ])


class ProblemGenerator:
    """
    Generates targeted training problems using a teacher model.

    Design constraints:
    - Only generates VERIFIABLE problems (no open-ended)
    - Focuses on weak domains identified by ActiveProber
    - Uses failure examples to generate harder variants
    - Each problem gets a content hash to prevent duplicates
    """

    def __init__(
        self,
        teacher_model: Any | None = None,
        teacher_tokenizer: Any | None = None,
        max_gen_tokens: int = 2048,
        temperature: float = 0.7,
        dedup_history_size: int = 10000,
    ):
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.max_gen_tokens = max_gen_tokens
        self.temperature = temperature

        # Track generated problem hashes to avoid duplicates
        self._generated_hashes: set[str] = set()
        self._dedup_limit = dedup_history_size

    def generate_from_weaknesses(
        self,
        weakness_report: WeaknessReport,
        problems_per_domain: int = 50,
        min_verifiable_ratio: float = 0.8,
    ) -> list[Problem]:
        """
        Generate problems targeting identified weaknesses.

        Args:
            weakness_report: Report from WeaknessAnalyzer.
            problems_per_domain: How many problems to generate per weak domain.
            min_verifiable_ratio: Minimum fraction that must be verifiable.

        Returns:
            List of generated and filtered problems.
        """
        all_problems: list[Problem] = []

        for domain in weakness_report.weak_domains:
            request = GenerationRequest(
                domain=domain,
                count=problems_per_domain,
                difficulty=0.5,
                focus_areas=[],
            )

            problems = self.generate(request)
            all_problems.extend(problems)

        # Filter to verifiable only
        verifiable = [
            p for p in all_problems if p.answer_type in VERIFIABLE_TYPES
        ]

        verifiable_ratio = len(verifiable) / len(all_problems) if all_problems else 0
        if verifiable_ratio < min_verifiable_ratio:
            logger.warning(
                "Only %.1f%% of generated problems are verifiable "
                "(threshold: %.1f%%)",
                verifiable_ratio * 100,
                min_verifiable_ratio * 100,
            )
            if len(verifiable) < 10:
                raise InsufficientVerifiableProblems(
                    f"Only {len(verifiable)} verifiable problems generated, "
                    f"need at least 10",
                )

        logger.info(
            "Generated %d problems (%d verifiable) across %d domains",
            len(all_problems),
            len(verifiable),
            len(weakness_report.weak_domains),
        )
        return verifiable

    def generate(self, request: GenerationRequest) -> list[Problem]:
        """
        Generate problems for a single request.

        Uses the teacher model if available, otherwise falls back
        to template-based generation.
        """
        if self.teacher_model is not None:
            problems = self._generate_with_model(request)
            if problems:
                logger.info(
                    "Model self-generated %d problems (domain=%s, diff=%.2f)",
                    len(problems), request.domain, request.difficulty,
                )
                return problems
            logger.warning("Model-based generation returned 0 problems, falling back to templates")
        return self._generate_from_templates(request)

    def generate_follow_up(
        self,
        domain: str,
        failure_examples: list[Problem],
        count: int = 20,
    ) -> list[Problem]:
        """
        Generate follow-up problems based on failure patterns.

        Creates harder variants of problems the model got wrong.
        """
        examples_text = "\n".join(
            f"- {p.question}" for p in failure_examples[:10]
        )

        request = GenerationRequest(
            domain=domain,
            count=count,
            difficulty=min(
                1.0,
                max(p.difficulty for p in failure_examples) + 0.1,
            ),
            failure_examples=[examples_text],
        )
        return self.generate(request)

    def _generate_with_model(
        self, request: GenerationRequest,
    ) -> list[Problem]:
        """Generate using the teacher model."""
        model = self.teacher_model
        tokenizer = self.teacher_tokenizer

        focus = ", ".join(request.focus_areas) if request.focus_areas else "general"

        user_prompt = PROBLEM_GEN_USER.format(
            count=request.count,
            domain=request.domain,
            difficulty=request.difficulty,
            focus=focus,
        )

        # If we have failure examples, use follow-up template
        if request.failure_examples:
            user_prompt += "\n\n" + FOLLOW_UP_GEN.format(
                failure_examples="\n".join(request.failure_examples),
                count=request.count,
            )

        messages = [
            {"role": "system", "content": PROBLEM_GEN_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        # Format as chat
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"{PROBLEM_GEN_SYSTEM}\n\n{user_prompt}\n\nAnswer:"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_gen_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                top_p=0.95,
            )

        new_tokens = outputs[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        problems = self._parse_generated_problems(
            response, request.domain, request.difficulty,
        )
        return self._deduplicate(problems)

    @staticmethod
    def _repair_json(text: str) -> str:
        """Best-effort repair of common LLM JSON mistakes."""
        # Strip markdown code fences
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
        # Remove single-line comments  (// ...)
        text = re.sub(r'//[^\n]*', '', text)
        # Remove trailing commas before ] or }
        text = re.sub(r',\s*([}\]])', r'\1', text)
        # Replace single quotes used as JSON delimiters with double quotes
        # (only when they look like JSON keys/values)
        text = re.sub(r"(?<=[\[{,:\s])'([^']*)'(?=\s*[,:\]}])", r'"\1"', text)
        return text

    def _parse_generated_problems(
        self,
        response: str,
        default_domain: str,
        default_difficulty: float,
    ) -> list[Problem]:
        """Parse model-generated JSON into Problem objects.

        Robust: repairs common LLM JSON errors, falls back to
        per-object extraction if full-array parse fails.
        """
        # Extract JSON array from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            logger.warning("No JSON array found in generated response")
            logger.debug("Raw generation (first 500 chars): %s", response[:500])
            return []

        raw_json = json_match.group()
        repaired = self._repair_json(raw_json)

        items: list[dict] = []
        try:
            items = json.loads(repaired)
        except json.JSONDecodeError:
            # Last resort: extract individual {...} objects one by one
            logger.info("Full JSON parse failed, extracting individual objects")
            depth = 0
            start = -1
            for idx, ch in enumerate(repaired):
                if ch == '{':
                    if depth == 0:
                        start = idx
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and start >= 0:
                        fragment = repaired[start:idx + 1]
                        try:
                            obj = json.loads(fragment)
                            if isinstance(obj, dict) and "question" in obj:
                                items.append(obj)
                        except json.JSONDecodeError:
                            pass
                        start = -1
            if items:
                logger.info("Recovered %d problems from individual extraction", len(items))
            else:
                logger.warning("JSON repair failed completely")
                logger.debug("Repaired text (first 500): %s", repaired[:500])

        problems = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                answer_type_str = item.get("answer_type", "exact_match")
                answer_type = AnswerType(answer_type_str)

                # Skip non-verifiable types
                if answer_type not in VERIFIABLE_TYPES:
                    if answer_type != AnswerType.EXACT_MATCH:
                        continue

                problem = Problem(
                    question=str(item["question"]),
                    domain=default_domain,
                    answer_type=answer_type,
                    difficulty=float(item.get("difficulty", default_difficulty)),
                    metadata={
                        "expected_answer": str(item.get("expected_answer", "")),
                        "tolerance": float(item.get("tolerance", 0.0)),
                        "tags": item.get("tags", []),
                        "generated": True,
                    },
                )
                problems.append(problem)
            except (KeyError, ValueError) as e:
                logger.debug("Skipping malformed problem: %s", e)

        return problems

    def _generate_from_templates(
        self, request: GenerationRequest,
    ) -> list[Problem]:
        """
        Template-based fallback generation when no teacher model is available.

        Not as good as model-based generation, but provides
        structured problems with verifiable answers.
        """
        generators = {
            "math": self._gen_math_problems,
            "coding": self._gen_coding_problems,
            "logic": self._gen_logic_problems,
            "knowledge": self._gen_knowledge_problems,
        }

        # Find best matching generator
        domain_lower = request.domain.lower()
        gen_func = None
        for key, func in generators.items():
            if key in domain_lower:
                gen_func = func
                break

        if gen_func is None:
            gen_func = self._gen_math_problems  # default fallback

        problems = gen_func(request.count, request.difficulty)
        return self._deduplicate(problems)

    def _gen_math_problems(
        self, count: int, difficulty: float,
    ) -> list[Problem]:
        """Generate math problems with numeric answers."""
        problems = []
        rng = random.Random(len(self._generated_hashes))

        for i in range(count):
            level = int(difficulty * 5) + 1  # 1-6

            if level <= 2:
                a, b = rng.randint(1, 100), rng.randint(1, 100)
                op = rng.choice(["+", "-", "*"])
                if op == "+":
                    answer = a + b
                elif op == "-":
                    answer = a - b
                else:
                    answer = a * b
                question = f"What is {a} {op} {b}?"
            elif level <= 4:
                a = rng.randint(2, 20)
                b = rng.randint(2, 6)
                answer = a ** b
                question = f"What is {a}^{b}?"
            else:
                # Harder: modular arithmetic, combinatorics
                n = rng.randint(5, 15)
                k = rng.randint(2, min(n, 6))
                # C(n, k)
                answer = 1
                for j in range(k):
                    answer = answer * (n - j) // (j + 1)
                question = f"What is C({n},{k})? (combinations)"

            problems.append(Problem(
                question=question,
                domain="math",
                answer_type=AnswerType.NUMERIC_RANGE,
                difficulty=difficulty,
                metadata={
                    "expected_answer": str(answer),
                    "tolerance": 0.0,
                    "generated": True,
                },
            ))

        return problems

    def _gen_coding_problems(
        self, count: int, difficulty: float,
    ) -> list[Problem]:
        """Generate coding problems with executable answers."""
        problems = []
        rng = random.Random(len(self._generated_hashes) + 42)

        templates = [
            {
                "question": "Write a Python function `f(n)` that returns the sum "
                            "of digits of n. What is f({n})?",
                "gen": lambda r: (
                    n := r.randint(100, 99999),
                    sum(int(d) for d in str(n)),
                ),
            },
            {
                "question": "Write a Python function `f(s)` that returns s reversed. "
                            "What is f('{s}')?",
                "gen": lambda r: (
                    s := ''.join(r.choices("abcdefghij", k=r.randint(5, 10))),
                    s[::-1],
                ),
            },
            {
                "question": "Write a Python function `f(n)` that returns True if n "
                            "is prime, False otherwise. What is f({n})?",
                "gen": lambda r: (
                    n := r.choice([
                        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                        4, 6, 8, 9, 10, 12, 14, 15, 16, 18,
                    ]),
                    str(all(n % i != 0 for i in range(2, int(n**0.5) + 1)) and n > 1),
                ),
            },
            {
                "question": "Write a Python function `f(lst)` that returns the "
                            "second largest element. What is f({lst})?",
                "gen": lambda r: (
                    lst := sorted(r.sample(range(1, 100), k=r.randint(4, 8))),
                    lst[-2],
                ),
            },
        ]

        for i in range(count):
            template = rng.choice(templates)
            result = template["gen"](rng)
            n_or_s, answer = result

            question = template["question"].format(
                n=n_or_s, s=n_or_s, lst=n_or_s,
            )

            problems.append(Problem(
                question=question,
                domain="coding",
                answer_type=AnswerType.EXACT_MATCH,
                difficulty=difficulty,
                metadata={
                    "expected_answer": str(answer),
                    "tolerance": 0.0,
                    "generated": True,
                },
            ))

        return problems

    def _gen_logic_problems(
        self, count: int, difficulty: float,
    ) -> list[Problem]:
        """Generate logic problems."""
        problems = []
        rng = random.Random(len(self._generated_hashes) + 99)

        for i in range(count):
            # Syllogism-style
            subjects = ["cats", "dogs", "birds", "fish", "insects"]
            properties = ["can fly", "have legs", "live in water",
                         "are mammals", "have wings"]

            s1 = rng.choice(subjects)
            s2 = rng.choice([s for s in subjects if s != s1])
            p1 = rng.choice(properties)
            p2 = rng.choice([p for p in properties if p != p1])

            # Simple valid syllogism
            valid = rng.choice([True, False])
            if valid:
                question = (
                    f"All {s1} {p1}. All things that {p1} {p2}. "
                    f"Therefore, all {s1} {p2}. Is this logically valid? "
                    f"Answer 'valid' or 'invalid'."
                )
                answer = "valid"
            else:
                question = (
                    f"Some {s1} {p1}. All {s2} {p2}. "
                    f"Therefore, all {s1} {p2}. Is this logically valid? "
                    f"Answer 'valid' or 'invalid'."
                )
                answer = "invalid"

            problems.append(Problem(
                question=question,
                domain="logic",
                answer_type=AnswerType.EXACT_MATCH,
                difficulty=difficulty,
                metadata={
                    "expected_answer": answer,
                    "tolerance": 0.0,
                    "generated": True,
                },
            ))

        return problems

    def _gen_knowledge_problems(
        self, count: int, difficulty: float,
    ) -> list[Problem]:
        """Generate factual knowledge problems."""
        problems = []
        # With templates, these are necessarily pattern-based
        rng = random.Random(len(self._generated_hashes) + 777)

        number_facts = [
            ("How many planets are in the solar system?", "8"),
            ("How many continents are there?", "7"),
            ("How many elements are in the periodic table?", "118"),
            ("What year did World War II end?", "1945"),
            ("What is the atomic number of carbon?", "6"),
            ("How many bones does an adult human body have?", "206"),
            ("What is the speed of light in m/s (approximate)?", "299792458"),
            ("How many chromosomes do humans have?", "46"),
            ("In what year was the Declaration of Independence signed?", "1776"),
            ("What is the boiling point of water in Celsius?", "100"),
        ]

        rng.shuffle(number_facts)
        for i in range(min(count, len(number_facts))):
            q, a = number_facts[i]
            problems.append(Problem(
                question=q,
                domain="knowledge",
                answer_type=AnswerType.EXACT_MATCH,
                difficulty=difficulty,
                metadata={
                    "expected_answer": a,
                    "tolerance": 0.0,
                    "generated": True,
                },
            ))

        return problems

    def _deduplicate(self, problems: list[Problem]) -> list[Problem]:
        """Remove duplicate problems based on content hash."""
        unique = []
        for p in problems:
            h = p.content_hash
            if h not in self._generated_hashes:
                self._generated_hashes.add(h)
                unique.append(p)

        # Trim history if needed
        if len(self._generated_hashes) > self._dedup_limit:
            excess = len(self._generated_hashes) - self._dedup_limit
            # Can't efficiently remove oldest from a set, just reset
            self._generated_hashes = set(
                list(self._generated_hashes)[-self._dedup_limit:]
            )

        return unique


class DifficultyAdapter:
    """
    Adapts problem difficulty based on model performance.

    Implements the "zone of proximal development" concept:
    problems should be hard enough to teach something new,
    but not so hard that the model can't learn from them.

    Target difficulty zone: 30-70% accuracy.
    """

    def __init__(
        self,
        min_accuracy: float = 0.3,
        max_accuracy: float = 0.7,
        step_size: float = 0.05,
    ):
        self.min_accuracy = min_accuracy
        self.max_accuracy = max_accuracy
        self.step_size = step_size
        self.domain_difficulty: dict[str, float] = {}
        self.domain_accuracy_history: dict[str, list[float]] = {}

    def get_difficulty(self, domain: str) -> float:
        """Get current target difficulty for a domain."""
        return self.domain_difficulty.get(domain, 0.5)

    def update(self, domain: str, accuracy: float) -> float:
        """
        Update difficulty based on observed accuracy.

        If accuracy is too high (>0.7): increase difficulty.
        If accuracy is too low (<0.3): decrease difficulty.
        Otherwise: keep current difficulty.
        """
        current = self.domain_difficulty.get(domain, 0.5)

        if domain not in self.domain_accuracy_history:
            self.domain_accuracy_history[domain] = []
        self.domain_accuracy_history[domain].append(accuracy)

        if accuracy > self.max_accuracy:
            new_difficulty = min(1.0, current + self.step_size)
        elif accuracy < self.min_accuracy:
            new_difficulty = max(0.0, current - self.step_size)
        else:
            new_difficulty = current

        self.domain_difficulty[domain] = new_difficulty

        if abs(new_difficulty - current) > 0.001:
            logger.info(
                "Domain %s difficulty: %.2f → %.2f (accuracy: %.1f%%)",
                domain,
                current,
                new_difficulty,
                accuracy * 100,
            )

        return new_difficulty

    def get_pressure(self, domain: str) -> float:
        """
        Get learning pressure score: how much harder should problems be?

        Range: -1.0 (much easier needed) to +1.0 (much harder needed).
        """
        history = self.domain_accuracy_history.get(domain, [])
        if len(history) < 3:
            return 0.0

        recent = history[-10:]
        avg = sum(recent) / len(recent)

        # Map to [-1, 1] range centered on the target zone
        target_center = (self.min_accuracy + self.max_accuracy) / 2
        return (avg - target_center) / target_center


class ProblemFilter:
    """
    Filters generated problems for quality and appropriateness.

    Checks:
    1. Problem has sufficient content (not too short/long)
    2. Answer is present in metadata
    3. Answer type is verifiable
    4. Problem is not a duplicate
    5. Difficulty is within acceptable range
    """

    def __init__(
        self,
        min_question_length: int = 20,
        max_question_length: int = 5000,
        require_metadata_answer: bool = True,
    ):
        self.min_length = min_question_length
        self.max_length = max_question_length
        self.require_answer = require_metadata_answer

    def filter(self, problems: list[Problem]) -> list[Problem]:
        """Filter problems, returning only those passing all checks."""
        return [p for p in problems if self._passes(p)]

    def _passes(self, problem: Problem) -> bool:
        """Check if a single problem passes all filters."""
        # Length check
        q_len = len(problem.question)
        if q_len < self.min_length or q_len > self.max_length:
            return False

        # Verifiable type check
        if problem.answer_type not in VERIFIABLE_TYPES:
            if problem.answer_type != AnswerType.EXACT_MATCH:
                return False

        # Answer metadata check
        if self.require_answer:
            if "expected_answer" not in problem.metadata:
                return False
            if not problem.metadata["expected_answer"]:
                return False

        # Difficulty range sanity
        if problem.difficulty < 0.0 or problem.difficulty > 1.0:
            return False

        return True

    def filter_with_report(
        self, problems: list[Problem],
    ) -> tuple[list[Problem], dict[str, int]]:
        """Filter and report reasons for rejection."""
        passed = []
        rejected_reasons: dict[str, int] = {
            "too_short": 0,
            "too_long": 0,
            "not_verifiable": 0,
            "no_answer": 0,
            "bad_difficulty": 0,
        }

        for p in problems:
            q_len = len(p.question)

            if q_len < self.min_length:
                rejected_reasons["too_short"] += 1
                continue
            if q_len > self.max_length:
                rejected_reasons["too_long"] += 1
                continue
            if p.answer_type not in VERIFIABLE_TYPES and p.answer_type != AnswerType.EXACT_MATCH:
                rejected_reasons["not_verifiable"] += 1
                continue
            if self.require_answer and (
                "expected_answer" not in p.metadata or not p.metadata["expected_answer"]
            ):
                rejected_reasons["no_answer"] += 1
                continue
            if p.difficulty < 0.0 or p.difficulty > 1.0:
                rejected_reasons["bad_difficulty"] += 1
                continue

            passed.append(p)

        return passed, rejected_reasons
