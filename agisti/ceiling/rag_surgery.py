"""
Ceiling Breaker Level 2 — Retrieval-Augmented Surgery.

The model's failures reveal gaps in its knowledge. This module:
1. Takes problems the model failed
2. Searches for relevant documents (ArXiv, Wikipedia, textbooks)
3. Model re-solves with context → now correct
4. Contrast: context_present vs context_absent activations
5. Surgery delta from this contrast → bakes knowledge into weights

Analogy: "Study with notes → memorize → pass exam without notes"

Design: §11.1.2 — Retrieval-Augmented Self-Surgery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from agisti.types import (
    AnswerType,
    FailedProblem,
    Problem,
    RAGSignal,
    Solution,
)
from agisti.ceiling.retriever import DocumentRetriever

logger = logging.getLogger(__name__)


@dataclass
class RAGFlipResult:
    """Result of a RAG-assisted re-solve attempt."""
    problem: FailedProblem
    acts_without: dict[str, Tensor]
    acts_with: dict[str, Tensor]
    context: str
    original_answer: str
    rag_answer: str
    flipped: bool  # True if wrong→correct with context


class RetrievalAugmentedSurgery:
    """
    Information Ceiling Breaker Level 2.

    When the model fails a problem but succeeds with retrieved context,
    the activation difference (with_context - without_context) represents
    the "knowledge gap" that needs to be baked into the weights.

    This effectively converts retrieval-time knowledge into inference-time
    capability — the model learns to produce the right activations even
    without the external context present.

    Information gain:
        Info(θ_{t+1}) ≤ Info(θ_t) + Info(ext_problems) + Info(retrieved_docs)

    The retrieval corpus size directly scales the information ceiling.
    """

    def __init__(
        self,
        retriever: DocumentRetriever,
        min_flips: int = 3,
        max_context_tokens: int = 1536,
    ):
        self.retriever = retriever
        self.min_flips = min_flips
        self.max_context_tokens = max_context_tokens

    def generate_rag_signal(
        self,
        model: nn.Module,
        tokenizer: Any,
        failed_problems: list[FailedProblem],
        trace_layers: list[str],
        max_length: int = 512,
    ) -> RAGSignal:
        """
        Generate RAG-based surgery signal from failed problems.

        For each failed problem:
        1. Search for relevant documents
        2. Re-solve with retrieved context
        3. If flip (wrong→correct): record activation contrast

        The contrast between "with context" and "without context" activations
        captures the knowledge that needs to be encoded in weights.

        Args:
            model: The model to probe.
            tokenizer: Tokenizer for the model.
            failed_problems: Problems the model got wrong.
            trace_layers: Layers to trace activations for.
            max_length: Maximum generation tokens.

        Returns:
            RAGSignal with contrasts (or usable=False).
        """
        effective_flips: list[RAGFlipResult] = []

        for problem in failed_problems:
            flip = self._attempt_rag_flip(
                model, tokenizer, problem, trace_layers, max_length,
            )
            if flip is not None and flip.flipped:
                effective_flips.append(flip)

        if len(effective_flips) < self.min_flips:
            return RAGSignal(
                usable=False,
                reason=(
                    f"Insufficient RAG flips: {len(effective_flips)} "
                    f"(need {self.min_flips})"
                ),
            )

        # Compute activation contrast across all flipped problems
        contrasts: dict[str, Tensor] = {}
        for layer_name in trace_layers:
            without_acts: list[Tensor] = []
            with_acts: list[Tensor] = []

            for flip in effective_flips:
                if layer_name in flip.acts_without and layer_name in flip.acts_with:
                    without_acts.append(flip.acts_without[layer_name])
                    with_acts.append(flip.acts_with[layer_name])

            if len(without_acts) < 2:
                continue

            without_stacked = torch.stack(without_acts)
            with_stacked = torch.stack(with_acts)

            # The "knowledge gap" activation pattern
            # Positive direction = what context adds = what needs baking
            contrasts[layer_name] = (
                with_stacked.mean(0) - without_stacked.mean(0)
            )

        if not contrasts:
            return RAGSignal(
                usable=False,
                reason="No valid layer contrasts computed",
            )

        return RAGSignal(
            usable=True,
            contrasts=contrasts,
            flip_count=len(effective_flips),
            total_attempted=len(failed_problems),
            flip_rate=len(effective_flips) / max(len(failed_problems), 1),
        )

    def _attempt_rag_flip(
        self,
        model: nn.Module,
        tokenizer: Any,
        problem: FailedProblem,
        trace_layers: list[str],
        max_length: int,
    ) -> RAGFlipResult | None:
        """
        Attempt to flip a failed problem by providing retrieved context.

        Returns RAGFlipResult if attempt was valid, None if error.
        """
        # 1. Retrieve relevant documents
        query = f"{problem.domain}: {problem.question}"
        documents = self.retriever.search(
            query=query,
            top_k=3,
            max_tokens_per_doc=512,
        )

        if not documents:
            return None

        context = self.retriever.get_context_string(
            documents,
            max_total_tokens=self.max_context_tokens,
        )

        # 2. Solve without context (capture activations)
        original_prompt = f"Question: {problem.question}\nAnswer:"
        acts_without = self._forward_with_tracing(
            model, tokenizer, original_prompt, trace_layers,
        )

        # 3. Solve with context
        augmented_prompt = (
            f"Reference information:\n{context}\n\n"
            f"Based on the above, solve:\n{problem.question}\nAnswer:"
        )
        acts_with = self._forward_with_tracing(
            model, tokenizer, augmented_prompt, trace_layers,
        )

        # 4. Generate answers for both
        original_answer = self._generate_answer(
            model, tokenizer, original_prompt, max_length,
        )
        rag_answer = self._generate_answer(
            model, tokenizer, augmented_prompt, max_length,
        )

        # 5. Check if flip occurred
        was_wrong = not problem.verify(original_answer)
        now_correct = problem.verify(rag_answer)
        flipped = was_wrong and now_correct

        return RAGFlipResult(
            problem=problem,
            acts_without=acts_without,
            acts_with=acts_with,
            context=context,
            original_answer=original_answer,
            rag_answer=rag_answer,
            flipped=flipped,
        )

    def _forward_with_tracing(
        self,
        model: nn.Module,
        tokenizer: Any,
        prompt: str,
        trace_layers: list[str],
    ) -> dict[str, Tensor]:
        """Do a forward pass capturing activations at specified layers."""
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
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        return activations

    def _generate_answer(
        self,
        model: nn.Module,
        tokenizer: Any,
        prompt: str,
        max_length: int,
    ) -> str:
        """Generate an answer from the model."""
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
            )

        return tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RAG Signal Analyzer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RAGSignalAnalyzer:
    """
    Analyzes RAG signals to determine which domains and layer types
    benefit most from retrieval-augmented surgery.

    Tracks flip rates across domains and iterations to optimize
    the retrieval strategy over time.
    """

    def __init__(self):
        self._domain_flip_rates: dict[str, list[float]] = {}
        self._layer_contrast_norms: dict[str, list[float]] = {}

    def record(self, signal: RAGSignal, domain: str = "") -> None:
        """Record a RAG signal for analysis."""
        if domain:
            if domain not in self._domain_flip_rates:
                self._domain_flip_rates[domain] = []
            self._domain_flip_rates[domain].append(signal.flip_rate)

        if signal.contrasts:
            for layer, contrast in signal.contrasts.items():
                if layer not in self._layer_contrast_norms:
                    self._layer_contrast_norms[layer] = []
                self._layer_contrast_norms[layer].append(
                    float(contrast.norm()),
                )

    def get_most_responsive_domains(
        self,
        min_samples: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Get domains ranked by RAG flip responsiveness.

        Higher flip rate = more knowledge gap = more benefit from RAG.
        """
        results: list[tuple[str, float]] = []
        for domain, rates in self._domain_flip_rates.items():
            if len(rates) < min_samples:
                continue
            avg_rate = sum(rates) / len(rates)
            results.append((domain, avg_rate))

        results.sort(key=lambda x: -x[1])
        return results

    def get_most_active_layers(
        self,
        min_samples: int = 3,
    ) -> list[tuple[str, float]]:
        """
        Get layers ranked by contrast magnitude.

        Higher contrast norm = layer is more affected by context = more
        surgery potential.
        """
        results: list[tuple[str, float]] = []
        for layer, norms in self._layer_contrast_norms.items():
            if len(norms) < min_samples:
                continue
            avg_norm = sum(norms) / len(norms)
            results.append((layer, avg_norm))

        results.sort(key=lambda x: -x[1])
        return results

    def should_use_rag(self, domain: str) -> bool:
        """
        Determine if RAG surgery is likely to help for a domain.

        Returns True if the domain has had meaningful flip rates.
        """
        rates = self._domain_flip_rates.get(domain, [])
        if len(rates) < 3:
            return True  # Default: try it

        avg = sum(rates) / len(rates)
        return avg > 0.05  # At least 5% flip rate

    def get_summary(self) -> dict[str, Any]:
        """Get analysis summary."""
        return {
            "domains_tracked": len(self._domain_flip_rates),
            "layers_tracked": len(self._layer_contrast_norms),
            "responsive_domains": self.get_most_responsive_domains(),
            "active_layers": self.get_most_active_layers()[:10],
        }
