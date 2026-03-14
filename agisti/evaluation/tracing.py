"""
Activation tracing — capture per-layer activations during inference.

Used by the surgery proposer to compute activation contrasts
between correct and incorrect answer generation.
Records activations at each layer to identify where the model
diverges between correct and incorrect behavior.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

import torch
import torch.nn as nn

from agisti.types import Problem, Solution

logger = logging.getLogger(__name__)


@dataclass
class LayerActivation:
    """Activation tensor at a specific layer."""
    layer_name: str
    activation: torch.Tensor  # [seq_len, hidden_dim] or [batch, seq_len, hidden_dim]
    position: str = "output"  # "output" or "input"

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.activation.shape)

    @property
    def norm(self) -> float:
        return self.activation.float().norm().item()

    @property
    def mean(self) -> float:
        return self.activation.float().mean().item()

    @property
    def std(self) -> float:
        return self.activation.float().std().item()

    def to_device(self, device: str | torch.device) -> LayerActivation:
        return LayerActivation(
            layer_name=self.layer_name,
            activation=self.activation.to(device),
            position=self.position,
        )

    def detach_cpu(self) -> LayerActivation:
        return LayerActivation(
            layer_name=self.layer_name,
            activation=self.activation.detach().cpu(),
            position=self.position,
        )


@dataclass
class ActivationTrace:
    """Complete activation trace for one generation."""
    problem: Problem
    answer: str
    is_correct: bool
    layer_activations: dict[str, LayerActivation] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def layers(self) -> list[str]:
        return list(self.layer_activations.keys())

    def get_activation(self, layer_name: str) -> torch.Tensor | None:
        la = self.layer_activations.get(layer_name)
        return la.activation if la is not None else None

    def detach_cpu(self) -> ActivationTrace:
        """Move all activations to CPU and detach."""
        return ActivationTrace(
            problem=self.problem,
            answer=self.answer,
            is_correct=self.is_correct,
            layer_activations={
                k: v.detach_cpu() for k, v in self.layer_activations.items()
            },
            metadata=dict(self.metadata),
        )


class ActivationTracer:
    """
    Traces activations through a model during inference.

    Installs forward hooks on specified layers to capture
    their output tensors. This data feeds into the surgery
    proposer's activation contrast computation.
    """

    def __init__(
        self,
        target_layers: list[str] | None = None,
        capture_input: bool = False,
        detach_to_cpu: bool = True,
        max_seq_length: int | None = None,
    ):
        self.target_layers = target_layers
        self.capture_input = capture_input
        self.detach_to_cpu = detach_to_cpu
        self.max_seq_length = max_seq_length

        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._activations: dict[str, LayerActivation] = {}

    def trace(
        self,
        model: nn.Module,
        tokenizer: Any,
        problem: Problem,
        answer: str,
        is_correct: bool,
    ) -> ActivationTrace:
        """
        Trace activations for a single problem+answer pair.

        Feeds the question+answer through the model and captures
        activations at each target layer.
        """
        prompt = self._format_for_tracing(problem, answer)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length or 2048,
        )

        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Install hooks
        layers = self._resolve_layers(model)
        self._activations.clear()
        self._install_hooks(model, layers)

        try:
            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        finally:
            self._remove_hooks()

        # Build trace
        trace_activations = {}
        for name, la in self._activations.items():
            if self.detach_to_cpu:
                trace_activations[name] = la.detach_cpu()
            else:
                trace_activations[name] = la

        self._activations.clear()

        return ActivationTrace(
            problem=problem,
            answer=answer,
            is_correct=is_correct,
            layer_activations=trace_activations,
            metadata={
                "prompt_length": input_ids.shape[1],
                "device": str(device),
            },
        )

    def trace_batch(
        self,
        model: nn.Module,
        tokenizer: Any,
        problems: list[Problem],
        answers: list[str],
        correctness: list[bool],
    ) -> list[ActivationTrace]:
        """Trace activations for a batch of problems."""
        traces = []
        for problem, answer, is_correct in zip(problems, answers, correctness):
            trace = self.trace(
                model, tokenizer, problem, answer, is_correct,
            )
            traces.append(trace)
        return traces

    def trace_contrast_pairs(
        self,
        model: nn.Module,
        tokenizer: Any,
        correct_pairs: list[tuple[Problem, str]],
        incorrect_pairs: list[tuple[Problem, str]],
    ) -> tuple[list[ActivationTrace], list[ActivationTrace]]:
        """
        Trace activation contrasts between correct and incorrect answers.

        This is the primary input for the surgery proposer.
        """
        correct_traces = []
        for problem, answer in correct_pairs:
            trace = self.trace(
                model, tokenizer, problem, answer, is_correct=True,
            )
            correct_traces.append(trace)

        incorrect_traces = []
        for problem, answer in incorrect_pairs:
            trace = self.trace(
                model, tokenizer, problem, answer, is_correct=False,
            )
            incorrect_traces.append(trace)

        return correct_traces, incorrect_traces

    def _resolve_layers(self, model: nn.Module) -> list[tuple[str, nn.Module]]:
        """Resolve target layer names to actual modules."""
        if self.target_layers is not None:
            resolved = []
            for name, module in model.named_modules():
                if name in self.target_layers or any(
                    tl in name for tl in self.target_layers
                ):
                    resolved.append((name, module))
            return resolved

        # Auto-detect transformer layers
        return self._auto_detect_layers(model)

    def _auto_detect_layers(
        self, model: nn.Module,
    ) -> list[tuple[str, nn.Module]]:
        """Auto-detect layer modules in a transformer model."""
        layers = []
        for name, module in model.named_modules():
            # Common transformer layer patterns
            if any(pattern in name for pattern in [
                ".layers.", ".h.", ".blocks.",
                ".transformer.layer.",
            ]):
                # Only capture the top-level layer module
                parts = name.split(".")
                is_sublayer = any(
                    sub in parts[-1]
                    for sub in [
                        "self_attn", "mlp", "norm", "ln",
                        "attention", "feed", "dropout",
                    ]
                )
                if not is_sublayer:
                    layers.append((name, module))

        if not layers:
            # Fallback: capture all linear layers
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    layers.append((name, module))

        return layers

    def _install_hooks(
        self,
        model: nn.Module,
        layers: list[tuple[str, nn.Module]],
    ) -> None:
        """Install forward hooks on target layers."""
        for name, module in layers:

            def make_hook(layer_name: str):
                def hook_fn(mod, inp, out):
                    # Handle tuple outputs (common in transformers)
                    activation = out
                    if isinstance(activation, tuple):
                        activation = activation[0]

                    if not isinstance(activation, torch.Tensor):
                        return

                    self._activations[layer_name] = LayerActivation(
                        layer_name=layer_name,
                        activation=activation,
                        position="output",
                    )

                    if self.capture_input and inp is not None:
                        inp_tensor = inp[0] if isinstance(inp, tuple) else inp
                        if isinstance(inp_tensor, torch.Tensor):
                            self._activations[f"{layer_name}_input"] = LayerActivation(
                                layer_name=f"{layer_name}_input",
                                activation=inp_tensor,
                                position="input",
                            )

                return hook_fn

            hook = module.register_forward_hook(make_hook(name))
            self._hooks.append(hook)

    def _remove_hooks(self) -> None:
        """Remove all installed hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _format_for_tracing(self, problem: Problem, answer: str) -> str:
        """Format problem+answer for tracing."""
        return f"Question: {problem.question}\n\nAnswer: {answer}"


class ActivationAnalyzer:
    """
    Analyzes activation traces to find divergence patterns.

    Computes statistics across correct vs incorrect traces
    to identify layers where behavior diverges most.
    """

    def compute_layer_divergence(
        self,
        correct_traces: list[ActivationTrace],
        incorrect_traces: list[ActivationTrace],
    ) -> dict[str, float]:
        """
        Compute divergence between correct and incorrect activations per layer.

        Higher divergence = this layer behaves very differently
        when the model gets it right vs wrong.
        """
        divergence: dict[str, float] = {}

        # Get common layers
        if not correct_traces or not incorrect_traces:
            return divergence

        common_layers = set(correct_traces[0].layers) & set(incorrect_traces[0].layers)

        for layer in common_layers:
            correct_acts = [
                t.get_activation(layer).float()
                for t in correct_traces
                if t.get_activation(layer) is not None
            ]
            incorrect_acts = [
                t.get_activation(layer).float()
                for t in incorrect_traces
                if t.get_activation(layer) is not None
            ]

            if not correct_acts or not incorrect_acts:
                continue

            # Compute mean activations (using last token position)
            correct_mean = torch.stack([
                a[:, -1, :] if a.dim() == 3 else a[-1, :]
                for a in correct_acts
            ]).mean(dim=0)

            incorrect_mean = torch.stack([
                a[:, -1, :] if a.dim() == 3 else a[-1, :]
                for a in incorrect_acts
            ]).mean(dim=0)

            # L2 divergence
            diff = correct_mean - incorrect_mean
            divergence[layer] = diff.norm().item()

        return divergence

    def find_critical_layers(
        self,
        correct_traces: list[ActivationTrace],
        incorrect_traces: list[ActivationTrace],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Find the layers with highest divergence."""
        divergence = self.compute_layer_divergence(
            correct_traces, incorrect_traces,
        )
        sorted_layers = sorted(
            divergence.items(), key=lambda x: -x[1],
        )
        return sorted_layers[:top_k]

    def compute_cosine_divergence(
        self,
        correct_traces: list[ActivationTrace],
        incorrect_traces: list[ActivationTrace],
    ) -> dict[str, float]:
        """
        Cosine-based divergence (direction change, not magnitude).

        1 - cosine_similarity: 0 = same direction, 2 = opposite direction.
        """
        divergence: dict[str, float] = {}

        if not correct_traces or not incorrect_traces:
            return divergence

        common_layers = set(correct_traces[0].layers) & set(incorrect_traces[0].layers)

        for layer in common_layers:
            correct_acts = [
                t.get_activation(layer).float()
                for t in correct_traces
                if t.get_activation(layer) is not None
            ]
            incorrect_acts = [
                t.get_activation(layer).float()
                for t in incorrect_traces
                if t.get_activation(layer) is not None
            ]

            if not correct_acts or not incorrect_acts:
                continue

            correct_mean = torch.stack([
                a[:, -1, :].flatten() if a.dim() == 3 else a[-1, :].flatten()
                for a in correct_acts
            ]).mean(dim=0)

            incorrect_mean = torch.stack([
                a[:, -1, :].flatten() if a.dim() == 3 else a[-1, :].flatten()
                for a in incorrect_acts
            ]).mean(dim=0)

            cos_sim = torch.nn.functional.cosine_similarity(
                correct_mean.unsqueeze(0),
                incorrect_mean.unsqueeze(0),
            ).item()

            divergence[layer] = 1.0 - cos_sim

        return divergence

    def activation_statistics(
        self,
        traces: list[ActivationTrace],
    ) -> dict[str, dict[str, float]]:
        """Compute per-layer statistics across traces."""
        stats: dict[str, dict[str, float]] = {}

        if not traces:
            return stats

        for layer in traces[0].layers:
            norms = []
            means = []
            stds = []

            for trace in traces:
                la = trace.layer_activations.get(layer)
                if la is not None:
                    norms.append(la.norm)
                    means.append(la.mean)
                    stds.append(la.std)

            if norms:
                stats[layer] = {
                    "mean_norm": sum(norms) / len(norms),
                    "mean_value": sum(means) / len(means),
                    "mean_std": sum(stds) / len(stds),
                    "norm_variance": _variance(norms),
                    "num_traces": float(len(norms)),
                }

        return stats


@contextmanager
def trace_context(
    model: nn.Module,
    target_layers: list[str] | None = None,
) -> Generator[ActivationTracer, None, None]:
    """Context manager for clean activation tracing."""
    tracer = ActivationTracer(target_layers=target_layers)
    try:
        yield tracer
    finally:
        tracer._remove_hooks()
        tracer._activations.clear()


def _variance(values: list[float]) -> float:
    """Compute variance of a list of values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)
