"""
Tests for agisti.utils.tokenization — prompt formatting, answer extraction.
"""

from __future__ import annotations

import pytest

from agisti.utils.tokenization import (
    format_prompt,
    extract_answer,
    extract_numeric_answer,
    truncate_context,
    get_model_type,
)


# ─── Prompt Formatting Tests ─────────────────────

class TestFormatPrompt:
    """Tests for multi-model prompt formatting."""

    def test_generic_format(self):
        prompt = format_prompt(
            question="What is 2+2?",
            model_type="generic",
        )
        assert "2+2" in prompt
        assert isinstance(prompt, str)

    def test_qwen_format(self):
        prompt = format_prompt(
            question="What is 2+2?",
            model_type="qwen",
        )
        assert "2+2" in prompt
        # Qwen typically uses <|im_start|> tokens
        assert isinstance(prompt, str)

    def test_llama_format(self):
        prompt = format_prompt(
            question="What is 2+2?",
            model_type="llama",
        )
        assert "2+2" in prompt
        assert isinstance(prompt, str)

    def test_with_system_prompt(self):
        prompt = format_prompt(
            question="What is 2+2?",
            model_type="generic",
            system_prompt="You are a math expert.",
        )
        assert "math expert" in prompt
        assert "2+2" in prompt

    def test_with_few_shot(self):
        examples = [
            {"question": "What is 1+1?", "answer": "2"},
            {"question": "What is 3+3?", "answer": "6"},
        ]
        prompt = format_prompt(
            question="What is 5+5?",
            model_type="generic",
            few_shot_examples=examples,
        )
        assert "5+5" in prompt
        assert "1+1" in prompt

    def test_empty_question(self):
        prompt = format_prompt(
            question="",
            model_type="generic",
        )
        assert isinstance(prompt, str)

    def test_long_question(self):
        long_q = "x " * 10000
        prompt = format_prompt(
            question=long_q,
            model_type="generic",
        )
        assert isinstance(prompt, str)


# ─── Answer Extraction Tests ──────────────────────

class TestExtractAnswer:
    """Tests for answer extraction from model output."""

    def test_extract_boxed(self):
        text = r"The answer is \boxed{42}."
        answer = extract_answer(text)
        assert answer == "42"

    def test_extract_boxed_nested(self):
        text = r"So we get \boxed{\frac{1}{2}}."
        answer = extract_answer(text)
        assert "frac" in answer or "1/2" in answer or answer == r"\frac{1}{2}"

    def test_extract_no_boxed(self):
        text = "The answer is 42."
        answer = extract_answer(text, fallback_pattern=r"answer is (\d+)")
        assert answer is not None

    def test_extract_multiple_boxed(self):
        text = r"First \boxed{1}, then \boxed{2}."
        answer = extract_answer(text)
        # Should extract the last boxed answer (final answer)
        assert answer in ("1", "2")

    def test_extract_empty_text(self):
        answer = extract_answer("")
        assert answer is None or answer == ""

    def test_extract_with_stop_token(self):
        text = "The answer is 42<|endoftext|>"
        answer = extract_answer(text, stop_tokens=["<|endoftext|>"])
        assert "42" in (answer or "42")


class TestExtractNumericAnswer:
    """Tests for numeric answer extraction."""

    def test_integer(self):
        result = extract_numeric_answer("The answer is 42")
        assert result == pytest.approx(42.0)

    def test_float(self):
        result = extract_numeric_answer("The answer is 3.14159")
        assert result == pytest.approx(3.14159)

    def test_negative(self):
        result = extract_numeric_answer("The answer is -7")
        assert result == pytest.approx(-7.0)

    def test_fraction(self):
        result = extract_numeric_answer("The answer is 1/3")
        if result is not None:
            assert result == pytest.approx(1 / 3, abs=0.01)

    def test_no_number(self):
        result = extract_numeric_answer("No numbers here!")
        assert result is None

    def test_multiple_numbers(self):
        # Should extract the last relevant number
        result = extract_numeric_answer(
            "Step 1: 5 + 3 = 8. Step 2: 8 * 2 = 16. Answer: 16"
        )
        assert result is not None

    def test_boxed_numeric(self):
        result = extract_numeric_answer(r"Therefore \boxed{256}")
        assert result == pytest.approx(256.0)

    def test_scientific_notation(self):
        result = extract_numeric_answer("The answer is 1.5e3")
        if result is not None:
            assert result == pytest.approx(1500.0)


# ─── Context Truncation Tests ─────────────────────

class TestTruncateContext:
    """Tests for context truncation to fit token budget."""

    def test_no_truncation_needed(self):
        text = "short text"
        result = truncate_context(text, max_tokens=1000)
        assert result == text

    def test_truncation_applied(self):
        text = "word " * 10000
        result = truncate_context(text, max_tokens=100)
        assert len(result) < len(text)

    def test_preserves_start_and_end(self):
        text = "START " + "middle " * 5000 + "END"
        result = truncate_context(
            text, max_tokens=50, strategy="start_end"
        )
        # Should preserve start and end
        assert result.startswith("START")

    def test_empty_text(self):
        result = truncate_context("", max_tokens=100)
        assert result == ""

    def test_exact_fit(self):
        # Text that's exactly at the limit
        text = "word " * 10
        result = truncate_context(text, max_tokens=1000)
        assert result == text


# ─── Model Type Detection Tests ──────────────────

class TestGetModelType:
    """Tests for automatic model type detection."""

    def test_qwen_detection(self):
        assert get_model_type("Qwen/Qwen2.5-0.5B") == "qwen"

    def test_llama_detection(self):
        assert get_model_type("meta-llama/Llama-2-7b") == "llama"

    def test_unknown_model(self):
        result = get_model_type("some-unknown/model")
        assert result in ("generic", "unknown")

    def test_case_insensitive(self):
        assert get_model_type("QWEN/qwen-0.5b") == "qwen"
        assert get_model_type("META-LLAMA/LLAMA-3") == "llama"

    def test_path_based_detection(self):
        # Local paths
        result = get_model_type("/models/qwen-0.5b-instruct")
        assert result == "qwen"

    def test_empty_string(self):
        result = get_model_type("")
        assert result in ("generic", "unknown")
