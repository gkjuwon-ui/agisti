"""
Tests for agisti.surgery — delta, proposer, applicator, moe, signal blender.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from agisti.types import LoRADelta, LoRALayerDelta, SurgeryType


class TestLoRALayerDelta:
    """Tests for LoRA layer delta operations."""

    def test_to_full_shape(self):
        A = torch.randn(4, 128)
        B = torch.randn(256, 4)
        delta = LoRALayerDelta(A=A, B=B, rank=4)
        full = delta.to_full((256, 128))
        assert full.shape == (256, 128)

    def test_to_full_reconstruction(self):
        """B @ A should equal to_full."""
        A = torch.randn(4, 64)
        B = torch.randn(32, 4)
        delta = LoRALayerDelta(A=A, B=B, rank=4)
        full = delta.to_full((32, 64))
        expected = B @ A
        assert torch.allclose(full, expected, atol=1e-6)

    def test_rank_matches(self):
        A = torch.randn(8, 100)
        B = torch.randn(200, 8)
        delta = LoRALayerDelta(A=A, B=B, rank=8)
        assert delta.rank == 8
        assert delta.A.shape[0] == 8
        assert delta.B.shape[1] == 8

    def test_small_rank(self):
        A = torch.randn(1, 50)
        B = torch.randn(50, 1)
        delta = LoRALayerDelta(A=A, B=B, rank=1)
        full = delta.to_full((50, 50))
        # Rank 1 → outer product
        assert full.shape == (50, 50)
        assert torch.linalg.matrix_rank(full).item() <= 1


class TestLoRADelta:
    """Tests for multi-layer LoRA delta."""

    def _make_delta(self, n_layers: int = 3, rank: int = 4) -> LoRADelta:
        layers = {}
        for i in range(n_layers):
            A = torch.randn(rank, 64)
            B = torch.randn(128, rank)
            layers[f"layer.{i}"] = LoRALayerDelta(A=A, B=B, rank=rank)
        return LoRADelta(layers=layers)

    def test_len(self):
        delta = self._make_delta(5)
        assert len(delta) == 5

    def test_layer_names(self):
        delta = self._make_delta(3)
        assert set(delta.layer_names) == {
            "layer.0", "layer.1", "layer.2",
        }

    def test_get_layer(self):
        delta = self._make_delta(2)
        assert delta.get_layer("layer.0") is not None
        assert delta.get_layer("layer.999") is None

    def test_norm_positive(self):
        delta = self._make_delta(3)
        assert delta.norm() > 0

    def test_empty_delta(self):
        delta = LoRADelta(layers={})
        assert len(delta) == 0
        assert delta.norm() == 0.0
        assert delta.layer_names == []

    def test_norm_consistency(self):
        """Norm should be deterministic."""
        delta = self._make_delta(3)
        n1 = delta.norm()
        n2 = delta.norm()
        assert n1 == n2


class TestDeltaApplication:
    """Tests for applying deltas to model weights."""

    def test_apply_changes_weights(self):
        """Applying a delta should change the model weights."""
        model = nn.Linear(64, 128, bias=False)
        original = model.weight.data.clone()

        A = torch.randn(4, 64)
        B = torch.randn(128, 4)
        delta_weight = B @ A

        model.weight.data += delta_weight

        assert not torch.allclose(model.weight.data, original)

    def test_apply_and_rollback(self):
        """Applying then removing a delta should restore original."""
        model = nn.Linear(64, 128, bias=False)
        original = model.weight.data.clone()

        A = torch.randn(4, 64)
        B = torch.randn(128, 4)
        delta_weight = B @ A

        model.weight.data += delta_weight
        model.weight.data -= delta_weight

        assert torch.allclose(
            model.weight.data, original, atol=1e-5,
        )


class TestDeltaNormBudget:
    """Tests for budget enforcement."""

    def test_norm_within_budget(self):
        """Create delta, check if norm can be computed and compared."""
        A = torch.randn(4, 64) * 0.01
        B = torch.randn(128, 4) * 0.01
        layer = LoRALayerDelta(A=A, B=B, rank=4)
        delta = LoRADelta(layers={"test": layer})

        budget = 1.0
        assert delta.norm() < budget

    def test_large_delta_exceeds_budget(self):
        """Large delta should exceed small budget."""
        A = torch.randn(4, 64) * 10.0
        B = torch.randn(128, 4) * 10.0
        layer = LoRALayerDelta(A=A, B=B, rank=4)
        delta = LoRADelta(layers={"test": layer})

        small_budget = 0.001
        assert delta.norm() > small_budget


class TestMultiLayerDelta:
    """Tests for applying deltas across multiple layers."""

    def test_multi_layer_model(self):
        """Apply deltas to a multi-layer model."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_0 = nn.Linear(64, 128, bias=False)
                self.layer_1 = nn.Linear(128, 64, bias=False)

        model = SimpleModel()
        original_0 = model.layer_0.weight.data.clone()
        original_1 = model.layer_1.weight.data.clone()

        # Create deltas
        d0 = LoRALayerDelta(
            A=torch.randn(4, 64) * 0.01,
            B=torch.randn(128, 4) * 0.01,
            rank=4,
        )
        d1 = LoRALayerDelta(
            A=torch.randn(4, 128) * 0.01,
            B=torch.randn(64, 4) * 0.01,
            rank=4,
        )

        # Apply
        model.layer_0.weight.data += d0.to_full((128, 64))
        model.layer_1.weight.data += d1.to_full((64, 128))

        assert not torch.allclose(model.layer_0.weight.data, original_0)
        assert not torch.allclose(model.layer_1.weight.data, original_1)
