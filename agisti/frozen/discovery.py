"""
FrozenZoneDiscovery — discovers which layers should be frozen.

Uses noise injection at the LAYER level (not parameter level)
to determine which layers are critical to preserve.

Design: §7 — Frozen zone discovery.
Key fix: Layer-level grouping not parameter-level (§7 v0.3 fix).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from agisti.types import FreezeLevel
from agisti.config import FrozenDiscoveryConfig
from agisti.probe.competency import CompetencyVector

logger = logging.getLogger(__name__)


@dataclass
class LayerSensitivity:
    """Sensitivity measurement for a single layer."""
    layer_name: str
    sensitivity_score: float  # how much noise hurts performance
    baseline_accuracy: float
    noisy_accuracy: float
    noise_scale: float
    num_parameters: int
    freeze_recommendation: FreezeLevel

    @property
    def accuracy_drop(self) -> float:
        return self.baseline_accuracy - self.noisy_accuracy

    @property
    def relative_drop(self) -> float:
        if self.baseline_accuracy < 1e-9:
            return 0.0
        return self.accuracy_drop / self.baseline_accuracy


@dataclass
class FrozenZoneReport:
    """Complete report from frozen zone discovery."""
    layers_tested: int
    frozen_layers: list[str]
    trainable_layers: list[str]
    partially_frozen_layers: list[str]
    sensitivities: list[LayerSensitivity]
    baseline_accuracy: float
    total_parameters: int
    frozen_parameters: int
    frozen_ratio: float
    elapsed_seconds: float

    def summary(self) -> str:
        lines = [
            f"FrozenZoneReport:",
            f"  Tested: {self.layers_tested} layers",
            f"  Frozen: {len(self.frozen_layers)} layers "
            f"({self.frozen_ratio:.1%} of params)",
            f"  Trainable: {len(self.trainable_layers)} layers",
            f"  Partially frozen: {len(self.partially_frozen_layers)} layers",
            f"  Baseline accuracy: {self.baseline_accuracy:.1%}",
            f"  Time: {self.elapsed_seconds:.1f}s",
        ]
        return "\n".join(lines)


class FrozenZoneDiscovery:
    """
    Discovers layers that should be frozen during surgery.

    Protocol (§7):
    1. Measure baseline accuracy on probe set
    2. For each LAYER (not individual params):
       a. Save layer state
       b. Add calibrated Gaussian noise
       c. Re-measure accuracy
       d. Restore layer state
    3. Compute sensitivity: (baseline - noisy) / baseline
    4. Freeze layers where sensitivity > threshold

    Layer-level grouping ensures all parameters in a layer
    are frozen/unfrozen together, preventing inconsistency.
    """

    def __init__(
        self,
        config: FrozenDiscoveryConfig | None = None,
        noise_scale: float = 0.1,
        sensitivity_threshold: float = 0.05,
        partial_freeze_threshold: float = 0.02,
    ):
        self.config = config or FrozenDiscoveryConfig()
        self.noise_scale = noise_scale
        self.sensitivity_threshold = sensitivity_threshold
        self.partial_threshold = partial_freeze_threshold

    def discover(
        self,
        model: nn.Module,
        eval_fn: Any,
        target_layers: list[str] | None = None,
    ) -> FrozenZoneReport:
        """
        Run frozen zone discovery on the model.

        Args:
            model: The model to analyze.
            eval_fn: Callable that takes model and returns accuracy (float).
                    This should run on a held-out probe set.
            target_layers: Specific layers to test. None = auto-detect.

        Returns:
            FrozenZoneReport with freeze recommendations.
        """
        start = time.monotonic()

        # Step 1: Baseline accuracy
        logger.info("Measuring baseline accuracy...")
        baseline_accuracy = eval_fn(model)
        logger.info("Baseline: %.1f%%", baseline_accuracy * 100)

        # Step 2: Identify layers to test
        layers = self._identify_layers(model, target_layers)
        logger.info("Testing %d layers for sensitivity", len(layers))

        # Step 3: Test each layer
        sensitivities: list[LayerSensitivity] = []
        for layer_name, layer_module in layers:
            sensitivity = self._test_layer_sensitivity(
                model, layer_name, layer_module, eval_fn, baseline_accuracy,
            )
            sensitivities.append(sensitivity)

        # Sort by sensitivity (highest first)
        sensitivities.sort(key=lambda s: -s.sensitivity_score)

        # Step 4: Classify layers
        frozen = []
        trainable = []
        partial = []

        total_params = 0
        frozen_params = 0

        for sens in sensitivities:
            total_params += sens.num_parameters
            if sens.freeze_recommendation == FreezeLevel.FULL:
                frozen.append(sens.layer_name)
                frozen_params += sens.num_parameters
            elif sens.freeze_recommendation == FreezeLevel.PARTIAL:
                partial.append(sens.layer_name)
                frozen_params += sens.num_parameters // 2  # estimate
            else:
                trainable.append(sens.layer_name)

        frozen_ratio = frozen_params / total_params if total_params > 0 else 0.0
        elapsed = time.monotonic() - start

        report = FrozenZoneReport(
            layers_tested=len(layers),
            frozen_layers=frozen,
            trainable_layers=trainable,
            partially_frozen_layers=partial,
            sensitivities=sensitivities,
            baseline_accuracy=baseline_accuracy,
            total_parameters=total_params,
            frozen_parameters=frozen_params,
            frozen_ratio=frozen_ratio,
            elapsed_seconds=elapsed,
        )

        logger.info(report.summary())
        return report

    def _test_layer_sensitivity(
        self,
        model: nn.Module,
        layer_name: str,
        layer_module: nn.Module,
        eval_fn: Any,
        baseline_accuracy: float,
    ) -> LayerSensitivity:
        """
        Test sensitivity of a single layer to noise.

        All parameters in the layer are perturbed together (layer-level grouping).
        """
        # Count parameters
        num_params = sum(p.numel() for p in layer_module.parameters())
        if num_params == 0:
            return LayerSensitivity(
                layer_name=layer_name,
                sensitivity_score=0.0,
                baseline_accuracy=baseline_accuracy,
                noisy_accuracy=baseline_accuracy,
                noise_scale=self.noise_scale,
                num_parameters=0,
                freeze_recommendation=FreezeLevel.NONE,
            )

        # Save original state
        original_state = {
            name: param.data.clone()
            for name, param in layer_module.named_parameters()
        }

        try:
            # Add calibrated noise to ALL parameters in the layer
            with torch.no_grad():
                for name, param in layer_module.named_parameters():
                    noise = torch.randn_like(param) * self.noise_scale * param.data.std()
                    param.data.add_(noise)

            # Measure noisy accuracy
            noisy_accuracy = eval_fn(model)

        finally:
            # Restore original parameters (critical!)
            with torch.no_grad():
                for name, param in layer_module.named_parameters():
                    if name in original_state:
                        param.data.copy_(original_state[name])

        # Compute sensitivity
        accuracy_drop = baseline_accuracy - noisy_accuracy
        sensitivity = accuracy_drop / max(baseline_accuracy, 1e-9)

        # Classify
        if sensitivity > self.sensitivity_threshold:
            recommendation = FreezeLevel.FULL
        elif sensitivity > self.partial_threshold:
            recommendation = FreezeLevel.PARTIAL
        else:
            recommendation = FreezeLevel.NONE

        logger.debug(
            "Layer %s: sensitivity=%.4f, drop=%.3f%%, → %s",
            layer_name,
            sensitivity,
            accuracy_drop * 100,
            recommendation.name,
        )

        return LayerSensitivity(
            layer_name=layer_name,
            sensitivity_score=sensitivity,
            baseline_accuracy=baseline_accuracy,
            noisy_accuracy=noisy_accuracy,
            noise_scale=self.noise_scale,
            num_parameters=num_params,
            freeze_recommendation=recommendation,
        )

    def _identify_layers(
        self,
        model: nn.Module,
        target_layers: list[str] | None,
    ) -> list[tuple[str, nn.Module]]:
        """Identify layer-level modules to test."""
        if target_layers is not None:
            layers = []
            for name, module in model.named_modules():
                if name in target_layers:
                    layers.append((name, module))
            return layers

        return self._auto_detect_transformer_layers(model)

    def _auto_detect_transformer_layers(
        self, model: nn.Module,
    ) -> list[tuple[str, nn.Module]]:
        """Auto-detect transformer layer modules."""
        layers = []
        for name, module in model.named_modules():
            # Match common transformer layer patterns
            # e.g. model.layers.0, model.transformer.h.0
            if any(
                pattern in name
                for pattern in [".layers.", ".h.", ".blocks."]
            ):
                # Only top-level layer modules (e.g., "model.layers.5"
                # not "model.layers.5.self_attn")
                parts = name.split(".")
                if parts[-1].isdigit():
                    layers.append((name, module))

        if not layers:
            logger.warning(
                "Could not auto-detect transformer layers. "
                "Falling back to all named modules with parameters.",
            )
            for name, module in model.named_modules():
                params = list(module.parameters(recurse=False))
                if params:
                    layers.append((name, module))

        return layers


class AdaptiveFrozenDiscovery(FrozenZoneDiscovery):
    """
    Extended discovery that adapts freeze boundaries over time.

    Re-runs discovery periodically to adjust as the model evolves.
    Layers that were previously trainable might become critical
    to freeze as the model improves in those areas.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._discovery_history: list[FrozenZoneReport] = []
        self._stable_frozen: set[str] = set()

    def discover_adaptive(
        self,
        model: nn.Module,
        eval_fn: Any,
        target_layers: list[str] | None = None,
        min_consistency: int = 3,
    ) -> FrozenZoneReport:
        """
        Run discovery and track consistency across runs.

        Layers that are consistently frozen across min_consistency
        runs become "stably frozen" and won't be unfrozen.
        """
        report = self.discover(model, eval_fn, target_layers)
        self._discovery_history.append(report)

        if len(self._discovery_history) >= min_consistency:
            recent = self._discovery_history[-min_consistency:]
            # Find layers frozen in ALL recent runs
            frozen_sets = [set(r.frozen_layers) for r in recent]
            consistently_frozen = frozen_sets[0]
            for s in frozen_sets[1:]:
                consistently_frozen &= s
            self._stable_frozen = consistently_frozen

        return report

    @property
    def stable_frozen_layers(self) -> set[str]:
        """Layers that are consistently frozen across runs."""
        return set(self._stable_frozen)

    def get_freeze_stability(self) -> dict[str, float]:
        """
        Get how stable each layer's freeze recommendation is.

        Returns fraction of runs where each layer was frozen.
        """
        if not self._discovery_history:
            return {}

        layer_frozen_count: dict[str, int] = defaultdict(int)
        all_layers: set[str] = set()

        for report in self._discovery_history:
            for layer in report.frozen_layers:
                layer_frozen_count[layer] += 1
            all_layers.update(report.frozen_layers)
            all_layers.update(report.trainable_layers)

        n = len(self._discovery_history)
        return {
            layer: layer_frozen_count.get(layer, 0) / n
            for layer in all_layers
        }
