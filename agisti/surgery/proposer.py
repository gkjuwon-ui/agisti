"""
Surgery proposer — generates surgery delta via activation contrast.

Heuristic Direction Oracle: uses mean activation differences between
correct and wrong solutions to propose a surgery direction.
VirtualTrainer's gradient refinement corrects the magnitude.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from agisti.types import (
    LoRADelta,
    LoRALayerDelta,
    SelfSignal,
    ErrorReport,
    FrozenZoneViolation,
)
from agisti.surgery.delta import DeltaFactory, BudgetEnforcer

logger = logging.getLogger(__name__)


class SurgeryProposer:
    """
    Heuristic Direction Oracle — activation contrast based.

    Computes Δa_l = E[a_correct] - E[a_wrong] per layer,
    decomposes via SVD into low-rank A, B matrices,
    and enforces surgery budget constraints.

    This is a HEURISTIC: the activation contrast provides a direction
    hint, not an exact weight delta. VirtualTrainer refines the magnitude.
    """

    def __init__(
        self,
        lora_rank: int = 4,
        budget: float = 0.01,
        min_wrong_samples: int = 2,
    ):
        self.lora_rank = lora_rank
        self.budget = budget
        self.min_wrong_samples = min_wrong_samples
        self._enforcer = BudgetEnforcer(max_budget=budget)

    def propose(
        self,
        activation_maps: list[dict[str, Tensor]],
        scores: list[bool],
        target_layers: list[str],
        frozen_layer_names: set[str],
    ) -> tuple[LoRADelta, SelfSignal]:
        """
        Propose a surgery delta from activation contrast.

        Args:
            activation_maps: Per-problem, per-layer activation tensors.
            scores: Whether each problem was answered correctly.
            target_layers: Layers eligible for surgery.
            frozen_layer_names: Layers that must not be modified.

        Returns:
            Tuple of (proposed LoRA delta, self signal for blending).
        """
        correct_indices = [i for i, s in enumerate(scores) if s]
        wrong_indices = [i for i, s in enumerate(scores) if not s]

        if len(wrong_indices) < self.min_wrong_samples:
            logger.info(
                "Only %d wrong samples (need %d). Returning zero delta.",
                len(wrong_indices), self.min_wrong_samples,
            )
            return self._empty_delta(), SelfSignal(
                contrasts={},
                correct_count=len(correct_indices),
                wrong_count=len(wrong_indices),
                verifiable_count=len(scores),
            )

        contrasts: dict[str, Tensor] = {}
        delta = LoRADelta(rank=self.lora_rank)
        modifiable_layers = [
            l for l in target_layers if l not in frozen_layer_names
        ]

        if not modifiable_layers:
            logger.warning("No modifiable layers among target layers.")
            return self._empty_delta(), SelfSignal(
                contrasts={},
                correct_count=len(correct_indices),
                wrong_count=len(wrong_indices),
                verifiable_count=len(scores),
            )

        budget_per_layer = self.budget / len(modifiable_layers)

        for layer_name in modifiable_layers:
            correct_acts = self._collect_activations(
                activation_maps, layer_name, correct_indices
            )
            wrong_acts = self._collect_activations(
                activation_maps, layer_name, wrong_indices
            )

            if correct_acts is None or wrong_acts is None:
                continue

            # Activation contrast: direction from wrong to correct
            contrast = correct_acts.mean(0) - wrong_acts.mean(0)
            contrasts[layer_name] = contrast

            # SVD decomposition to low-rank delta
            layer_delta = DeltaFactory.from_contrast(
                contrast, self.lora_rank, budget_per_layer
            )
            delta.add_layer(layer_name, layer_delta)

        # Global budget enforcement
        delta = self._enforcer.enforce(delta)

        signal = SelfSignal(
            contrasts=contrasts,
            correct_count=len(correct_indices),
            wrong_count=len(wrong_indices),
            verifiable_count=len(scores),
        )

        logger.info(
            "Proposed delta: norm=%.6f, layers=%d, correct=%d, wrong=%d",
            delta.norm(), len(delta), len(correct_indices), len(wrong_indices),
        )
        return delta, signal

    def _collect_activations(
        self,
        activation_maps: list[dict[str, Tensor]],
        layer_name: str,
        indices: list[int],
    ) -> Tensor | None:
        """Collect and stack activations for given indices."""
        acts = []
        for i in indices:
            if i < len(activation_maps) and layer_name in activation_maps[i]:
                act = activation_maps[i][layer_name]
                # Flatten to 2D if needed (batch, features)
                if act.dim() > 2:
                    act = act.view(act.shape[0], -1)
                elif act.dim() == 1:
                    act = act.unsqueeze(0)
                acts.append(act.mean(0))  # Average over sequence positions

        if not acts:
            return None

        return torch.stack(acts)

    def _empty_delta(self) -> LoRADelta:
        return LoRADelta(rank=self.lora_rank)


class DirectionalAnalyzer:
    """
    Analyze the direction of proposed surgery relative to other signals.

    Used by propose_surgery_with_external() to determine how to
    blend self and external signals based on cosine similarity.
    """

    @staticmethod
    def cosine_similarity(a: Tensor, b: Tensor) -> float:
        """Compute cosine similarity between two contrast vectors."""
        a_flat = a.flatten().unsqueeze(0).float()
        b_flat = b.flatten().unsqueeze(0).float()

        a_norm = torch.linalg.norm(a_flat)
        b_norm = torch.linalg.norm(b_flat)

        if a_norm < 1e-12 or b_norm < 1e-12:
            return 0.0

        return F.cosine_similarity(a_flat, b_flat).item()

    @staticmethod
    def blend_with_direction_check(
        self_contrast: Tensor,
        ext_contrast: Tensor,
        external_weight: float,
        budget_scale: float = 1.0,
    ) -> tuple[Tensor, float]:
        """
        Blend self and external contrasts with direction-aware logic.

        Returns:
            Tuple of (blended contrast, adjusted budget_scale).

        Direction rules:
        - cos_sim > 0.3:  directions agree → weighted sum (reinforce)
        - cos_sim < -0.3: directions oppose → use external (self may be wrong)
        - else:           orthogonal → sum both independently
        """
        cos_sim = DirectionalAnalyzer.cosine_similarity(self_contrast, ext_contrast)

        if cos_sim > 0.3:
            # Directions agree: weighted blend
            blended = (1 - external_weight) * self_contrast + external_weight * ext_contrast
        elif cos_sim < -0.3:
            # Directions oppose: trust external, reduce budget (cautious)
            blended = ext_contrast
            budget_scale *= 0.5
        else:
            # Orthogonal: independent contributions
            blended = self_contrast + ext_contrast

        return blended, budget_scale

    @staticmethod
    def compute_contrast_stats(contrasts: dict[str, Tensor]) -> dict[str, dict[str, float]]:
        """Compute statistics on contrast vectors for logging."""
        stats: dict[str, dict[str, float]] = {}
        for layer_name, contrast in contrasts.items():
            flat = contrast.flatten().float()
            stats[layer_name] = {
                "norm": torch.linalg.norm(flat).item(),
                "mean": flat.mean().item(),
                "std": flat.std().item(),
                "max_abs": flat.abs().max().item(),
                "sparsity": (flat.abs() < 1e-6).float().mean().item(),
            }
        return stats


class ActivationCollector:
    """
    Collect activation maps during model inference.

    Installs forward hooks on target layers to capture
    intermediate activations for contrast computation.
    """

    def __init__(self, model: nn.Module, target_layers: list[str]):
        self.model = model
        self.target_layers = target_layers
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._activations: dict[str, Tensor] = {}

    def install_hooks(self) -> None:
        """Install forward hooks on target layers."""
        self.clear()
        named_modules = dict(self.model.named_modules())

        for layer_name in self.target_layers:
            if layer_name not in named_modules:
                logger.warning("Layer %s not found in model. Skipping.", layer_name)
                continue

            module = named_modules[layer_name]

            def make_hook(name: str):
                def hook_fn(mod: nn.Module, inp: Any, output: Any) -> None:
                    if isinstance(output, tuple):
                        # Many transformer layers return (hidden_states, ...)
                        out_tensor = output[0]
                    elif isinstance(output, Tensor):
                        out_tensor = output
                    else:
                        return
                    self._activations[name] = out_tensor.detach()
                return hook_fn

            hook = module.register_forward_hook(make_hook(layer_name))
            self._hooks.append(hook)

    def get_activations(self) -> dict[str, Tensor]:
        """Get captured activations (call after a forward pass)."""
        return dict(self._activations)

    def clear_activations(self) -> None:
        """Clear stored activations (call between forward passes)."""
        self._activations.clear()

    def clear(self) -> None:
        """Remove all hooks and clear state."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._activations.clear()

    def __enter__(self):
        self.install_hooks()
        return self

    def __exit__(self, *args):
        self.clear()


class ContrastComputer:
    """
    Compute activation contrasts between correct and wrong solutions.

    Separates the concern of contrast computation from the proposer,
    allowing different contrast methods (mean, max-margin, etc.).
    """

    @staticmethod
    def mean_contrast(
        correct_activations: list[dict[str, Tensor]],
        wrong_activations: list[dict[str, Tensor]],
        layer_names: list[str],
    ) -> dict[str, Tensor]:
        """
        Standard activation contrast: E[correct] - E[wrong].

        Args:
            correct_activations: Activations from correctly solved problems.
            wrong_activations: Activations from incorrectly solved problems.
            layer_names: Which layers to compute contrast for.

        Returns:
            Per-layer contrast tensors.
        """
        contrasts: dict[str, Tensor] = {}

        for layer_name in layer_names:
            correct_acts = _stack_layer_acts(correct_activations, layer_name)
            wrong_acts = _stack_layer_acts(wrong_activations, layer_name)

            if correct_acts is None or wrong_acts is None:
                continue

            contrasts[layer_name] = correct_acts.mean(0) - wrong_acts.mean(0)

        return contrasts

    @staticmethod
    def margin_contrast(
        correct_activations: list[dict[str, Tensor]],
        wrong_activations: list[dict[str, Tensor]],
        layer_names: list[str],
        margin: float = 0.1,
    ) -> dict[str, Tensor]:
        """
        Margin-based contrast: emphasizes dimensions with large differences.

        Only keeps contrast dimensions where |diff| > margin * std(diff).
        """
        contrasts: dict[str, Tensor] = {}

        for layer_name in layer_names:
            correct_acts = _stack_layer_acts(correct_activations, layer_name)
            wrong_acts = _stack_layer_acts(wrong_activations, layer_name)

            if correct_acts is None or wrong_acts is None:
                continue

            diff = correct_acts.mean(0) - wrong_acts.mean(0)
            threshold = margin * diff.abs().std()
            mask = diff.abs() > threshold
            contrasts[layer_name] = diff * mask.float()

        return contrasts


def _stack_layer_acts(
    activation_maps: list[dict[str, Tensor]],
    layer_name: str,
) -> Tensor | None:
    """Stack activations for a specific layer across multiple forward passes."""
    acts = []
    for act_map in activation_maps:
        if layer_name in act_map:
            a = act_map[layer_name]
            if a.dim() > 2:
                a = a.view(a.shape[0], -1)
            elif a.dim() == 1:
                a = a.unsqueeze(0)
            acts.append(a.mean(0))  # Average over sequence positions

    if not acts:
        return None
    return torch.stack(acts)
