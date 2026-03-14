"""
Prompt formatting utilities — handles both base and instruct models.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def format_for_model(prompt: str, tokenizer: Any) -> dict[str, Any]:
    """
    Format a prompt string into tokenizer inputs, using chat template
    for instruct-tuned models and raw text for base models.

    Returns dict with 'input_ids' and 'attention_mask' tensors.
    """
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return tokenizer(
            formatted, return_tensors="pt",
            truncation=True, max_length=2048,
        )
    else:
        return tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=2048,
        )
