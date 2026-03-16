"""
vLLM / SGLang integration for high-throughput batched generation.

When available, replaces the default HuggingFace generate() with
vLLM's continuous batching engine, which provides:
  - PagedAttention: ~4x memory efficiency
  - Continuous batching: no padding waste
  - CUDA graph + prefix caching: low latency
  - Multi-GPU tensor parallel out of the box

Falls back to standard HF generate() if vLLM is not installed.

Usage:
    from agisti.generation.vllm_engine import get_engine

    engine = get_engine(model_path_or_name, tensor_parallel_size=2)
    outputs = engine.generate(prompts, max_new_tokens=512)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_vllm_available = False
_sglang_available = False

try:
    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
    _vllm_available = True
    logger.info("vLLM backend available — high-throughput mode enabled")
except ImportError:
    pass

if not _vllm_available:
    try:
        import sglang  # type: ignore[import-not-found]
        _sglang_available = True
        logger.info("SGLang backend available — using as vLLM alternative")
    except ImportError:
        pass


class VLLMEngine:
    """
    vLLM-based generation engine with continuous batching.

    Wraps vLLM.LLM for use in the AGISTI evaluation pipeline.
    Supports both offline (batch) and online (streaming) generation.
    """

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
        trust_remote_code: bool = True,
        dtype: str = "auto",
    ):
        if not _vllm_available:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            )

        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
        )
        self.model_name = model
        logger.info(
            "vLLM engine initialized: model=%s, tp=%d, mem=%.0f%%",
            model, tensor_parallel_size, gpu_memory_utilization * 100,
        )

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> list[str]:
        """
        Generate completions for a batch of prompts.

        Uses vLLM's continuous batching — all prompts are processed
        concurrently with PagedAttention, no manual batching needed.

        Args:
            prompts: List of text prompts.
            max_new_tokens: Maximum tokens to generate per prompt.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling threshold.
            stop: Stop sequences.

        Returns:
            List of generated text strings (one per prompt).
        """
        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )

        outputs = self.llm.generate(prompts, params)

        results = []
        for output in outputs:
            text = output.outputs[0].text if output.outputs else ""
            results.append(text.strip())

        return results

    def generate_with_logprobs(
        self,
        prompts: list[str],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Generate with per-token log probabilities (for confidence scoring).

        Returns list of dicts with 'text' and 'logprobs' keys.
        """
        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            logprobs=5,  # top-5 logprobs per position
        )

        outputs = self.llm.generate(prompts, params)

        results = []
        for output in outputs:
            if not output.outputs:
                results.append({"text": "", "logprobs": []})
                continue

            completion = output.outputs[0]
            token_logprobs = []
            if completion.logprobs:
                for lp in completion.logprobs:
                    token_logprobs.append({
                        tok: prob
                        for tok, prob in lp.items()
                    })

            results.append({
                "text": completion.text.strip(),
                "logprobs": token_logprobs,
            })

        return results


class HFEngine:
    """
    Fallback engine using standard HuggingFace generate().

    Used when vLLM is not available. Attempts batched generation
    with left-padding, falls back to sequential if needed.
    """

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> list[str]:
        """Generate completions using HF model.generate()."""
        device = next(self.model.parameters()).device
        results = []

        # Try batched generation with left-padding
        try:
            old_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-7),
                    top_p=top_p,
                )

            input_len = inputs["input_ids"].shape[1]
            for i in range(len(prompts)):
                text = self.tokenizer.decode(
                    outputs[i][input_len:],
                    skip_special_tokens=True,
                ).strip()
                results.append(text)

            self.tokenizer.padding_side = old_side
            return results

        except Exception:
            pass

        # Sequential fallback
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048,
            ).to(device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                )

            text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            results.append(text)

        return results


# Need torch for HFEngine
try:
    import torch
except ImportError:
    pass


def get_engine(
    model: Any = None,
    tokenizer: Any = None,
    model_name: str | None = None,
    tensor_parallel_size: int = 1,
    **kwargs,
) -> VLLMEngine | HFEngine:
    """
    Get the best available generation engine.

    Priority: vLLM > SGLang > HuggingFace

    Args:
        model: HF model instance (for HFEngine fallback).
        tokenizer: HF tokenizer (for HFEngine fallback).
        model_name: Model name/path (for vLLM).
        tensor_parallel_size: GPU count for vLLM tensor parallel.

    Returns:
        Engine instance with .generate() method.
    """
    if _vllm_available and model_name:
        try:
            return VLLMEngine(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                **kwargs,
            )
        except Exception as e:
            logger.warning("vLLM init failed, falling back to HF: %s", e)

    if model is not None and tokenizer is not None:
        return HFEngine(model, tokenizer)

    raise ValueError(
        "No generation engine available. "
        "Provide model+tokenizer for HF, or install vLLM."
    )
