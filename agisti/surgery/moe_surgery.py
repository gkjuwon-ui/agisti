"""
MoE (Mixture-of-Experts) specific surgery strategies.

For models like Llama 4 Behemoth or DeepSeek:
- Identify responsible experts via router logit analysis
- Selective surgery on culprit experts only
- MoE-specific forbidden zones (router, shared expert)
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from agisti.types import LoRADelta, LoRALayerDelta
from agisti.surgery.delta import DeltaFactory, BudgetEnforcer

logger = logging.getLogger(__name__)


# MoE Forbidden zone patterns
MOE_FORBIDDEN_PATTERNS = [
    "*.block_sparse_moe.gate",
    "*.block_sparse_moe.shared_expert",
    "*.moe.gate",
    "*.moe.shared_expert",
]


class MoESurgery:
    """
    Surgery engine specialized for Mixture-of-Experts models.

    Instead of modifying all model parameters, it:
    1. Analyzes router logits to find which experts handle wrong answers
    2. Surgically modifies only those "culprit" experts
    3. Leaves router and shared experts completely frozen

    VRAM saving: Only loads 1/N of expert parameters at a time.
    """

    def __init__(
        self,
        over_activation_ratio: float = 1.5,
        max_culprit_experts: int = 4,
    ):
        self.over_activation_ratio = over_activation_ratio
        self.max_culprit_experts = max_culprit_experts

    def identify_responsible_experts(
        self,
        model: nn.Module,
        wrong_inputs: list[Tensor],
        layer_idx: int,
    ) -> list[int]:
        """
        Identify which experts are responsible for wrong answers.

        Analyzes router logit distributions on inputs that produced
        wrong answers. Experts activated significantly more than average
        are deemed "responsible" — they're the ones making mistakes.

        Args:
            model: The MoE model.
            wrong_inputs: Input tensors for incorrectly answered problems.
            layer_idx: Which transformer layer to analyze.

        Returns:
            List of expert indices that are over-activated on wrong inputs.
        """
        expert_hit_counts: Counter = Counter()
        num_experts = self._get_num_experts(model, layer_idx)
        experts_per_token = self._get_experts_per_token(model, layer_idx)

        if num_experts == 0:
            logger.warning("Layer %d does not appear to be a MoE layer.", layer_idx)
            return []

        gate_module = self._get_gate_module(model, layer_idx)
        if gate_module is None:
            logger.warning("Could not find gate module for layer %d.", layer_idx)
            return []

        model.eval()
        with torch.no_grad():
            for x in wrong_inputs:
                # Get router logits for this input
                hidden_states = self._get_hidden_at_layer(model, x, layer_idx)
                if hidden_states is None:
                    continue

                router_logits = gate_module(hidden_states)

                # Get top-k experts selected by router
                k = min(experts_per_token, router_logits.shape[-1])
                top_experts = router_logits.topk(k, dim=-1).indices

                for expert_idx in top_experts.flatten().tolist():
                    expert_hit_counts[int(expert_idx)] += 1

        if not expert_hit_counts:
            return []

        total_hits = sum(expert_hit_counts.values())
        avg_hits = total_hits / num_experts

        # Over-activated = hits > avg * over_activation_ratio
        threshold = avg_hits * self.over_activation_ratio
        responsible = [
            eid for eid, count in expert_hit_counts.most_common()
            if count > threshold
        ]

        # Cap the number of culprit experts
        responsible = responsible[: self.max_culprit_experts]

        logger.info(
            "Layer %d: %d/%d experts identified as responsible. "
            "Threshold: %.1f hits (avg: %.1f, ratio: %.1f)",
            layer_idx, len(responsible), num_experts,
            threshold, avg_hits, self.over_activation_ratio,
        )

        return responsible

    def selective_surgery(
        self,
        model: nn.Module,
        delta: LoRADelta,
        responsible_experts: dict[int, list[int]],  # layer_idx → expert indices
        frozen_mask_names: set[str],
    ) -> None:
        """
        Apply surgery only to responsible experts.

        Args:
            model: The MoE model.
            delta: Full proposed delta.
            responsible_experts: Map of layer_idx → culprit expert indices.
            frozen_mask_names: Frozen layer names.
        """
        applied_count = 0

        for layer_idx, expert_list in responsible_experts.items():
            for expert_idx in expert_list:
                expert_key = f"model.layers.{layer_idx}.experts.{expert_idx}"

                # Check frozen
                is_frozen = any(
                    expert_key == f or expert_key.startswith(f + ".")
                    for f in frozen_mask_names
                )
                if is_frozen:
                    continue

                # Find matching delta
                for delta_key in delta.keys():
                    if expert_key in delta_key:
                        layer_delta = delta[delta_key]
                        self._apply_to_expert(
                            model, layer_idx, expert_idx, layer_delta
                        )
                        applied_count += 1

        logger.info(
            "MoE selective surgery: applied %d expert modifications "
            "across %d layers.",
            applied_count, len(responsible_experts),
        )

    def generate_expert_delta(
        self,
        model: nn.Module,
        correct_inputs: list[Tensor],
        wrong_inputs: list[Tensor],
        layer_idx: int,
        expert_idx: int,
        lora_rank: int,
        budget: float,
    ) -> LoRALayerDelta | None:
        """
        Generate a LoRA delta for a specific expert's FFN.

        Uses activation contrast within the expert's FFN:
        contrast = E[activation_correct] - E[activation_wrong]
        where activations are from the expert's up_proj output.
        """
        expert_module = self._get_expert_module(model, layer_idx, expert_idx)
        if expert_module is None:
            return None

        correct_acts = self._collect_expert_activations(
            model, expert_module, correct_inputs, layer_idx
        )
        wrong_acts = self._collect_expert_activations(
            model, expert_module, wrong_inputs, layer_idx
        )

        if correct_acts is None or wrong_acts is None:
            return None

        contrast = correct_acts.mean(0) - wrong_acts.mean(0)
        return DeltaFactory.from_contrast(contrast, lora_rank, budget)

    def _collect_expert_activations(
        self,
        model: nn.Module,
        expert_module: nn.Module,
        inputs: list[Tensor],
        layer_idx: int,
    ) -> Tensor | None:
        """Collect activations from a specific expert's FFN."""
        acts = []

        # Hook onto the expert's output
        captured: list[Tensor] = []

        def hook_fn(module: nn.Module, inp: Any, output: Any) -> None:
            if isinstance(output, Tensor):
                captured.append(output.detach().mean(dim=tuple(range(output.dim() - 1))))

        hook = expert_module.register_forward_hook(hook_fn)

        try:
            model.eval()
            with torch.no_grad():
                for x in inputs:
                    captured.clear()
                    model(x)
                    if captured:
                        acts.append(captured[0])
        finally:
            hook.remove()

        if not acts:
            return None
        return torch.stack(acts)

    def _apply_to_expert(
        self,
        model: nn.Module,
        layer_idx: int,
        expert_idx: int,
        layer_delta: LoRALayerDelta,
    ) -> None:
        """Apply a LoRA delta to a specific expert's weight."""
        expert = self._get_expert_module(model, layer_idx, expert_idx)
        if expert is None:
            return

        full_delta = layer_delta.to_full()

        # Apply to the expert's down_proj (output projection) by default
        target = None
        for name in ("down_proj", "w2", "fc2"):
            if hasattr(expert, name):
                target = getattr(expert, name)
                break

        if target is not None and hasattr(target, "weight"):
            with torch.no_grad():
                w = target.weight
                if full_delta.shape == w.shape:
                    w.add_(full_delta.to(w.device, w.dtype))
                elif full_delta.numel() == w.numel():
                    w.add_(full_delta.reshape(w.shape).to(w.device, w.dtype))

    def _get_num_experts(self, model: nn.Module, layer_idx: int) -> int:
        """Get the number of experts in a MoE layer."""
        layer = self._get_layer(model, layer_idx)
        if layer is None:
            return 0

        for attr in ("block_sparse_moe", "moe"):
            if hasattr(layer, attr):
                moe = getattr(layer, attr)
                if hasattr(moe, "num_experts"):
                    return moe.num_experts
                if hasattr(moe, "experts"):
                    return len(moe.experts)
        return 0

    def _get_experts_per_token(self, model: nn.Module, layer_idx: int) -> int:
        """Get the number of experts activated per token."""
        layer = self._get_layer(model, layer_idx)
        if layer is None:
            return 2

        for attr in ("block_sparse_moe", "moe"):
            if hasattr(layer, attr):
                moe = getattr(layer, attr)
                for k_attr in ("num_experts_per_tok", "top_k"):
                    if hasattr(moe, k_attr):
                        return getattr(moe, k_attr)
        return 2

    def _get_gate_module(self, model: nn.Module, layer_idx: int) -> nn.Module | None:
        """Get the router/gate module for a MoE layer."""
        layer = self._get_layer(model, layer_idx)
        if layer is None:
            return None

        for attr in ("block_sparse_moe", "moe"):
            if hasattr(layer, attr):
                moe = getattr(layer, attr)
                if hasattr(moe, "gate"):
                    return moe.gate
        return None

    def _get_expert_module(
        self, model: nn.Module, layer_idx: int, expert_idx: int
    ) -> nn.Module | None:
        """Get a specific expert module."""
        layer = self._get_layer(model, layer_idx)
        if layer is None:
            return None

        for attr in ("block_sparse_moe", "moe"):
            if hasattr(layer, attr):
                moe = getattr(layer, attr)
                if hasattr(moe, "experts"):
                    experts = moe.experts
                    if isinstance(experts, nn.ModuleList) and expert_idx < len(experts):
                        return experts[expert_idx]
        return None

    def _get_layer(self, model: nn.Module, layer_idx: int) -> nn.Module | None:
        """Get transformer layer by index."""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h
        else:
            return None

        if isinstance(layers, nn.ModuleList) and layer_idx < len(layers):
            return layers[layer_idx]
        return None

    def _get_hidden_at_layer(
        self,
        model: nn.Module,
        input_ids: Tensor,
        target_layer_idx: int,
    ) -> Tensor | None:
        """Get hidden states at a specific layer via forward hooks."""
        captured: list[Tensor] = []
        layer = self._get_layer(model, target_layer_idx)
        if layer is None:
            return None

        def hook_fn(module: nn.Module, inp: Any, output: Any) -> None:
            if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], Tensor):
                captured.append(inp[0].detach())

        hook = layer.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            hook.remove()

        return captured[0] if captured else None


def build_moe_forbidden_set(model: nn.Module) -> set[str]:
    """
    Auto-discover MoE forbidden zones in a model.

    Forbidden:
    - All router/gate modules
    - All shared expert modules
    """
    forbidden: set[str] = set()

    for name, module in model.named_modules():
        # Router/gate patterns
        if name.endswith(".gate") and any(
            p in name for p in ("block_sparse_moe", "moe")
        ):
            forbidden.add(name)

        # Shared expert patterns
        if "shared_expert" in name:
            forbidden.add(name)

    logger.info("MoE forbidden zones: %d modules", len(forbidden))
    return forbidden
