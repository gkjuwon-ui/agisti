"""
Surgery delta management — LoRA-style low-rank weight updates.

Provides the complete lifecycle of surgery deltas:
creation, composition, budget enforcement, serialization, and application.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor
from safetensors.torch import save_file, load_file

from agisti.types import (
    LoRADelta,
    LoRALayerDelta,
    SurgeryBudgetExceeded,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Delta Factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DeltaFactory:
    """Create LoRA deltas via SVD decomposition of activation contrasts."""

    @staticmethod
    def from_contrast(
        contrast_vector: Tensor,
        rank: int,
        budget_per_layer: float,
    ) -> LoRALayerDelta:
        """
        Decompose a contrast vector into a low-rank delta A @ B.

        The contrast vector is the difference between mean activations
        of correct vs wrong solutions. SVD extracts the top-r
        singular components as the surgery direction.

        Args:
            contrast_vector: Mean activation difference (correct - wrong).
            rank: Target LoRA rank.
            budget_per_layer: Maximum allowed norm for this layer's delta.

        Returns:
            LoRALayerDelta with A, B matrices.
        """
        if contrast_vector.dim() == 1:
            contrast_vector = contrast_vector.unsqueeze(0)

        # SVD requires float32 (not supported for bf16 on CPU)
        orig_dtype = contrast_vector.dtype
        contrast_vector = contrast_vector.float()

        # SVD decomposition — rank-r approximation
        U, S, Vt = torch.linalg.svd(contrast_vector, full_matrices=False)

        effective_rank = min(rank, len(S), U.shape[1], Vt.shape[0])
        if effective_rank == 0:
            d_out = contrast_vector.shape[0]
            d_in = contrast_vector.shape[1] if contrast_vector.dim() > 1 else 1
            return LoRALayerDelta(
                A=torch.zeros(d_out, 1, device=contrast_vector.device),
                B=torch.zeros(1, d_in, device=contrast_vector.device),
            )

        sqrt_s = S[:effective_rank].sqrt()
        A = U[:, :effective_rank] * sqrt_s.unsqueeze(0)
        B = Vt[:effective_rank, :] * sqrt_s.unsqueeze(1)

        delta = LoRALayerDelta(A=A, B=B)

        # Budget enforcement
        if delta.norm() > budget_per_layer:
            delta.scale_to(budget_per_layer)

        return delta

    @staticmethod
    def zeros(d_out: int, d_in: int, rank: int, device: torch.device | None = None) -> LoRALayerDelta:
        """Create a zero-valued delta."""
        return LoRALayerDelta(
            A=torch.zeros(d_out, rank, device=device),
            B=torch.zeros(rank, d_in, device=device),
        )

    @staticmethod
    def random(
        d_out: int,
        d_in: int,
        rank: int,
        scale: float = 0.01,
        device: torch.device | None = None,
    ) -> LoRALayerDelta:
        """Create a random delta (for testing)."""
        return LoRALayerDelta(
            A=torch.randn(d_out, rank, device=device) * scale,
            B=torch.randn(rank, d_in, device=device) * scale,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Delta Composition
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DeltaComposer:
    """Compose multiple delta sources into a single blended delta."""

    @staticmethod
    def blend_contrasts(
        contrasts: list[dict[str, Tensor]],
        weights: list[float],
    ) -> dict[str, Tensor]:
        """
        Weighted average of multiple contrast dictionaries.

        Args:
            contrasts: List of {layer_name → contrast_tensor} dicts.
            weights: Weight for each contrast source (must sum to 1).

        Returns:
            Blended contrast per layer.
        """
        if not contrasts:
            return {}

        total_weight = sum(weights)
        if total_weight < 1e-12:
            return {}

        normalized_weights = [w / total_weight for w in weights]

        all_layers: set[str] = set()
        for c in contrasts:
            all_layers.update(c.keys())

        blended: dict[str, Tensor] = {}
        for layer_name in all_layers:
            components = []
            component_weights = []
            for c, w in zip(contrasts, normalized_weights):
                if layer_name in c:
                    components.append(c[layer_name])
                    component_weights.append(w)

            if not components:
                continue

            # Re-normalize weights for available components
            cw_sum = sum(component_weights)
            result = torch.zeros_like(components[0])
            for comp, cw in zip(components, component_weights):
                result += (cw / cw_sum) * comp
            blended[layer_name] = result

        return blended

    @staticmethod
    def contrasts_to_delta(
        blended_contrasts: dict[str, Tensor],
        rank: int,
        total_budget: float,
        frozen_layer_names: set[str],
    ) -> LoRADelta:
        """Convert blended contrasts into a LoRA delta."""
        delta = LoRADelta(rank=rank)
        modifiable = {
            name: c for name, c in blended_contrasts.items()
            if name not in frozen_layer_names
        }

        if not modifiable:
            return delta

        budget_per_layer = total_budget / len(modifiable)

        for layer_name, contrast in modifiable.items():
            layer_delta = DeltaFactory.from_contrast(contrast, rank, budget_per_layer)
            delta.add_layer(layer_name, layer_delta)

        # Final budget check on total
        if delta.norm() > total_budget:
            delta.scale_to(total_budget)

        return delta


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Delta Serialization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DeltaSerializer:
    """Save and load LoRA deltas to/from disk."""

    @staticmethod
    def save(delta: LoRADelta, path: str | Path) -> None:
        """Save delta as safetensors with metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tensors: dict[str, Tensor] = {}
        metadata: dict[str, str] = {"rank": str(delta.rank)}

        for layer_name, ld in delta.items():
            safe_name = layer_name.replace(".", "__")
            tensors[f"{safe_name}__A"] = ld.A.contiguous().cpu()
            tensors[f"{safe_name}__B"] = ld.B.contiguous().cpu()

        metadata["layers"] = json.dumps(list(delta.keys()))
        save_file(tensors, str(path), metadata=metadata)
        logger.debug("Saved delta to %s (%d layers)", path, len(delta))

    @staticmethod
    def load(path: str | Path, device: torch.device | str = "cpu") -> LoRADelta:
        """Load delta from safetensors."""
        path = Path(path)
        tensors = load_file(str(path), device=str(device))

        # Reconstruct from naming convention
        layers_dict: dict[str, dict[str, Tensor]] = {}
        for key, tensor in tensors.items():
            parts = key.rsplit("__", 1)
            if len(parts) != 2:
                continue
            layer_safe_name, component = parts
            layer_name = layer_safe_name.replace("__", ".")
            if layer_name not in layers_dict:
                layers_dict[layer_name] = {}
            layers_dict[layer_name][component] = tensor

        # Determine rank from first layer
        rank = 1
        for layer_data in layers_dict.values():
            if "A" in layer_data:
                rank = layer_data["A"].shape[1]
                break

        delta = LoRADelta(rank=rank)
        for layer_name, data in layers_dict.items():
            if "A" in data and "B" in data:
                delta.add_layer(layer_name, LoRALayerDelta(A=data["A"], B=data["B"]))

        logger.debug("Loaded delta from %s (%d layers, rank=%d)", path, len(delta), rank)
        return delta


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Delta Budget Enforcement
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class BudgetEnforcer:
    """Enforce surgery budgets on deltas."""

    def __init__(self, max_budget: float, max_per_layer: float | None = None):
        self.max_budget = max_budget
        self.max_per_layer = max_per_layer

    def enforce(self, delta: LoRADelta) -> LoRADelta:
        """
        Enforce budget constraints. Returns a new delta if scaling was needed.

        Raises SurgeryBudgetExceeded if the delta is pathologically large
        (> 10x budget), suggesting a bug rather than a scaling issue.
        """
        total_norm = delta.norm()

        if total_norm > self.max_budget * 10:
            raise SurgeryBudgetExceeded(
                f"Delta norm {total_norm:.4f} is >10x budget {self.max_budget:.4f}. "
                f"This suggests a computation error, not a scaling issue."
            )

        # Per-layer enforcement
        if self.max_per_layer is not None:
            for name, ld in delta.items():
                if ld.norm() > self.max_per_layer:
                    ld.scale_to(self.max_per_layer)
                    logger.debug(
                        "Scaled layer %s delta from %.4f to %.4f",
                        name, ld.norm(), self.max_per_layer,
                    )

        # Total budget enforcement
        total_norm = delta.norm()
        if total_norm > self.max_budget:
            logger.info(
                "Scaling total delta from %.4f to %.4f (budget)",
                total_norm, self.max_budget,
            )
            delta.scale_to(self.max_budget)

        return delta

    def compute_per_layer_budget(self, n_layers: int) -> float:
        """Equal-share budget per layer."""
        if n_layers <= 0:
            return self.max_budget
        return self.max_budget / n_layers


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Delta Statistics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class DeltaStats:
    """Statistics about a delta for logging and analysis."""

    total_norm: float
    num_layers: int
    rank: int
    layer_norms: dict[str, float]
    total_params_modified: int
    max_layer_norm: float
    min_layer_norm: float
    mean_layer_norm: float

    @classmethod
    def from_delta(cls, delta: LoRADelta) -> DeltaStats:
        layer_norms = {name: ld.norm() for name, ld in delta.items()}
        norms_list = list(layer_norms.values()) or [0.0]
        total_params = sum(
            ld.A.numel() + ld.B.numel() for ld in delta.values()
        )
        return cls(
            total_norm=delta.norm(),
            num_layers=len(delta),
            rank=delta.rank,
            layer_norms=layer_norms,
            total_params_modified=total_params,
            max_layer_norm=max(norms_list),
            min_layer_norm=min(norms_list),
            mean_layer_norm=sum(norms_list) / len(norms_list),
        )

    def to_dict(self) -> dict:
        return {
            "total_norm": self.total_norm,
            "num_layers": self.num_layers,
            "rank": self.rank,
            "total_params_modified": self.total_params_modified,
            "max_layer_norm": self.max_layer_norm,
            "min_layer_norm": self.min_layer_norm,
            "mean_layer_norm": self.mean_layer_norm,
        }
