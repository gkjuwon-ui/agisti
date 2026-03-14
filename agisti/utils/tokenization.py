"""
Tokenization Utilities — helpers for tokenizer operations
used across the AGISTI pipeline.

Handles:
- Prompt formatting (system/user/assistant template)
- Token counting and truncation
- Batch tokenization with padding
- Answer extraction from generated text
- Special token management

Works with HuggingFace tokenizers (PreTrainedTokenizer).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TokenizedPrompt:
    """Tokenized prompt with metadata."""
    input_ids: list[int]
    attention_mask: list[int]
    token_count: int
    truncated: bool = False


def format_prompt(
    question: str,
    system_prompt: str | None = None,
    few_shot: list[tuple[str, str]] | None = None,
    model_type: str = "qwen",
) -> str:
    """
    Format a prompt for the model.

    Supports different model chat template formats.

    Args:
        question: The main question/problem.
        system_prompt: Optional system message.
        few_shot: Optional list of (question, answer) tuples.
        model_type: Model type for template selection.

    Returns:
        Formatted prompt string.
    """
    if model_type in ("qwen", "qwen2", "qwen2.5"):
        return _format_qwen(question, system_prompt, few_shot)
    elif model_type in ("llama", "llama3"):
        return _format_llama(question, system_prompt, few_shot)
    else:
        return _format_generic(question, system_prompt, few_shot)


def _format_qwen(
    question: str,
    system_prompt: str | None = None,
    few_shot: list[tuple[str, str]] | None = None,
) -> str:
    """Qwen-style chat template."""
    parts = []

    if system_prompt:
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

    if few_shot:
        for q, a in few_shot:
            parts.append(f"<|im_start|>user\n{q}<|im_end|>")
            parts.append(f"<|im_start|>assistant\n{a}<|im_end|>")

    parts.append(f"<|im_start|>user\n{question}<|im_end|>")
    parts.append("<|im_start|>assistant\n")

    return "\n".join(parts)


def _format_llama(
    question: str,
    system_prompt: str | None = None,
    few_shot: list[tuple[str, str]] | None = None,
) -> str:
    """Llama 3-style chat template."""
    parts = []
    parts.append("<|begin_of_text|>")

    if system_prompt:
        parts.append(
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
        )

    if few_shot:
        for q, a in few_shot:
            parts.append(
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{q}<|eot_id|>"
            )
            parts.append(
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{a}<|eot_id|>"
            )

    parts.append(
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}<|eot_id|>"
    )
    parts.append(
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    return "".join(parts)


def _format_generic(
    question: str,
    system_prompt: str | None = None,
    few_shot: list[tuple[str, str]] | None = None,
) -> str:
    """Generic prompt format."""
    parts = []

    if system_prompt:
        parts.append(f"### System:\n{system_prompt}\n")

    if few_shot:
        for q, a in few_shot:
            parts.append(f"### User:\n{q}\n")
            parts.append(f"### Assistant:\n{a}\n")

    parts.append(f"### User:\n{question}\n")
    parts.append("### Assistant:\n")

    return "\n".join(parts)


def tokenize(
    text: str,
    tokenizer: Any,
    max_length: int = 2048,
    truncation: bool = True,
    add_special_tokens: bool = True,
) -> TokenizedPrompt:
    """
    Tokenize text with metadata.

    Args:
        text: Input text to tokenize.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum token count.
        truncation: Whether to truncate.
        add_special_tokens: Whether to add BOS/EOS.

    Returns:
        TokenizedPrompt with ids and metadata.
    """
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=truncation,
        add_special_tokens=add_special_tokens,
        return_attention_mask=True,
        padding=False,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_count = len(input_ids)

    # Detect truncation
    full_count = len(
        tokenizer.encode(text, add_special_tokens=add_special_tokens)
    )
    truncated = full_count > max_length

    return TokenizedPrompt(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_count=token_count,
        truncated=truncated,
    )


def count_tokens(
    text: str,
    tokenizer: Any,
) -> int:
    """Count tokens in text."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def batch_tokenize(
    texts: list[str],
    tokenizer: Any,
    max_length: int = 2048,
    padding: bool = True,
    return_tensors: str = "pt",
) -> dict[str, Any]:
    """
    Batch tokenize multiple texts with padding.

    Args:
        texts: List of texts.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum token count.
        padding: Whether to pad to max length in batch.
        return_tensors: Output tensor format.

    Returns:
        Dict with input_ids, attention_mask tensors.
    """
    return tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=padding,
        return_tensors=return_tensors,
        return_attention_mask=True,
    )


def extract_answer(
    generated_text: str,
    stop_tokens: list[str] | None = None,
) -> str:
    """
    Extract the answer portion from generated text.

    Looks for common answer delimiters and extracts
    the final answer.

    Args:
        generated_text: Full generated text.
        stop_tokens: Additional stop token strings.

    Returns:
        Extracted answer string.
    """
    text = generated_text.strip()

    # Try to find boxed answer (LaTeX style)
    boxed_match = re.search(
        r"\\boxed\{([^}]+)\}", text,
    )
    if boxed_match:
        return boxed_match.group(1).strip()

    # Try "The answer is" pattern
    answer_match = re.search(
        r"(?:the answer is|answer:|final answer:)\s*(.+?)(?:\.|$)",
        text,
        re.IGNORECASE,
    )
    if answer_match:
        return answer_match.group(1).strip()

    # Apply stop tokens
    default_stops = [
        "<|im_end|>", "<|eot_id|>", "</s>",
        "\n\n", "###",
    ]
    all_stops = default_stops + (stop_tokens or [])

    for stop in all_stops:
        idx = text.find(stop)
        if idx > 0:
            text = text[:idx]

    # Return last line as answer fallback
    lines = text.strip().split("\n")
    return lines[-1].strip()


def extract_numeric_answer(text: str) -> float | None:
    """
    Extract a numeric answer from text.

    Handles integers, decimals, fractions, negatives.
    Returns None if no number found.
    """
    # Try boxed
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        text = boxed.group(1)

    # Try fraction
    frac = re.search(r"(-?\d+)\s*/\s*(\d+)", text)
    if frac:
        numer = int(frac.group(1))
        denom = int(frac.group(2))
        if denom != 0:
            return numer / denom

    # Try decimal/integer (last number in text)
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass

    return None


def truncate_context(
    context: str,
    max_tokens: int,
    tokenizer: Any,
    keep_start: int = 100,
    keep_end: int = 100,
) -> str:
    """
    Truncate context to fit within token budget.

    Keeps the start and end of the context, removing the middle.

    Args:
        context: Text to truncate.
        max_tokens: Target token count.
        tokenizer: For counting tokens.
        keep_start: Tokens to keep from the start.
        keep_end: Tokens to keep from the end.

    Returns:
        Truncated text.
    """
    tokens = tokenizer.encode(context, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return context

    start_toks = tokens[:keep_start]
    end_toks = tokens[-keep_end:]

    # Fill middle budget
    middle_budget = max_tokens - keep_start - keep_end
    if middle_budget > 0:
        mid_start = keep_start
        mid_end = len(tokens) - keep_end
        mid_toks = tokens[mid_start:mid_start + middle_budget]
        final_toks = start_toks + mid_toks + end_toks
    else:
        final_toks = start_toks + end_toks

    return tokenizer.decode(final_toks, skip_special_tokens=True)


def get_model_type(tokenizer: Any) -> str:
    """
    Detect model type from tokenizer.

    Returns:
        Model type string: "qwen", "llama", or "generic".
    """
    name = getattr(tokenizer, "name_or_path", "").lower()

    if "qwen" in name:
        return "qwen"
    elif "llama" in name:
        return "llama"
    elif "mistral" in name:
        return "llama"  # Mistral uses similar template
    else:
        return "generic"
