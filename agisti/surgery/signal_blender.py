"""
Signal Blender — combines surgery signals from all sources.

Implements the multi-source signal blending from AGISTI design §11.1.5:
Δa_final = α·Δa_self + β·Δa_ext + γ·Δa_RAG + δ·Δa_cross

Phase-specific weights ensure graceful transition from self-only
to external-dominant as the system matures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from agisti.types import (
    SelfSignal,
    ExternalSignal,
    RAGSignal,
    CrossSignal,
    BlendedSignal,
    PhaseId,
    LoRADelta,
)
from agisti.surgery.delta import DeltaComposer

logger = logging.getLogger(__name__)


# Phase-specific default weights
PHASE_WEIGHTS: dict[str, dict[str, float]] = {
    "phase_0": {"self": 1.0, "external": 0.0, "rag": 0.0, "cross": 0.0},
    "phase_1_early": {"self": 0.7, "external": 0.3, "rag": 0.0, "cross": 0.0},
    "phase_1_late": {"self": 0.5, "external": 0.5, "rag": 0.0, "cross": 0.0},
    "phase_2_early": {"self": 0.3, "external": 0.3, "rag": 0.4, "cross": 0.0},
    "phase_2_late": {"self": 0.2, "external": 0.2, "rag": 0.6, "cross": 0.0},
    "phase_3": {"self": 0.1, "external": 0.1, "rag": 0.3, "cross": 0.5},
}


@dataclass
class SignalCollection:
    """All collected signals for one iteration."""

    self_signal: SelfSignal | None = None
    external_signal: ExternalSignal | None = None
    rag_signal: RAGSignal | None = None
    cross_signal: CrossSignal | None = None

    def available_sources(self) -> list[str]:
        sources = []
        if self.self_signal and self.self_signal.contrasts:
            sources.append("self")
        if self.external_signal and self.external_signal.usable:
            sources.append("external")
        if self.rag_signal and self.rag_signal.usable:
            sources.append("rag")
        if self.cross_signal and self.cross_signal.usable:
            sources.append("cross")
        return sources


class SignalBlender:
    """
    Blends multiple surgery signal sources into a single direction.

    The blender:
    1. Collects available signals
    2. Applies phase-appropriate weights
    3. Normalizes for available sources
    4. Produces a single blended contrast per layer
    """

    def __init__(
        self,
        phase_key: str = "phase_0",
        custom_weights: dict[str, float] | None = None,
    ):
        if custom_weights:
            self.weights = custom_weights
        elif phase_key in PHASE_WEIGHTS:
            self.weights = PHASE_WEIGHTS[phase_key].copy()
        else:
            self.weights = PHASE_WEIGHTS["phase_0"].copy()

    def blend(self, signals: SignalCollection) -> BlendedSignal:
        """
        Blend all available signals into a single set of contrasts.

        Missing signals are handled gracefully: their weight is
        redistributed proportionally to available sources.
        """
        available = signals.available_sources()

        if not available:
            return BlendedSignal(
                contrasts={},
                weights={},
                sources_used=[],
            )

        # Collect contrasts and weights for available sources
        contrasts_list: list[dict[str, Tensor]] = []
        weights_list: list[float] = []
        sources_used: list[str] = []

        source_contrasts = {
            "self": signals.self_signal.contrasts if signals.self_signal else None,
            "external": (
                signals.external_signal.contrasts
                if signals.external_signal and signals.external_signal.usable
                else None
            ),
            "rag": (
                signals.rag_signal.contrasts
                if signals.rag_signal and signals.rag_signal.usable
                else None
            ),
            "cross": (
                signals.cross_signal.contrasts
                if signals.cross_signal and signals.cross_signal.usable
                else None
            ),
        }

        for source_name in available:
            c = source_contrasts.get(source_name)
            if c is not None:
                contrasts_list.append(c)
                weights_list.append(self.weights.get(source_name, 0.0))
                sources_used.append(source_name)

        if not contrasts_list:
            return BlendedSignal(contrasts={}, weights={}, sources_used=[])

        # Normalize weights to sum to 1
        total_w = sum(weights_list)
        if total_w < 1e-12:
            # All available sources have zero weight — equal distribution
            weights_list = [1.0 / len(weights_list)] * len(weights_list)
        else:
            weights_list = [w / total_w for w in weights_list]

        # Blend
        blended = DeltaComposer.blend_contrasts(contrasts_list, weights_list)

        actual_weights = {s: w for s, w in zip(sources_used, weights_list)}

        logger.info(
            "Signal blend: sources=%s, weights=%s, layers=%d",
            sources_used,
            {s: f"{w:.3f}" for s, w in actual_weights.items()},
            len(blended),
        )

        return BlendedSignal(
            contrasts=blended,
            weights=actual_weights,
            sources_used=sources_used,
        )

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Manually update blending weights."""
        self.weights.update(new_weights)

    @classmethod
    def for_phase(cls, phase: PhaseId, iteration_in_phase: int = 0) -> SignalBlender:
        """Create a blender with phase-appropriate weights."""
        if phase == PhaseId.PHASE_0:
            key = "phase_0"
        elif phase == PhaseId.PHASE_1:
            key = "phase_1_early" if iteration_in_phase < 2500 else "phase_1_late"
        elif phase == PhaseId.PHASE_2:
            key = "phase_2_early" if iteration_in_phase < 5000 else "phase_2_late"
        else:
            key = "phase_3"
        return cls(phase_key=key)


class AdaptiveSignalBlender(SignalBlender):
    """
    Extended blender that adapts weights based on performance trends.

    Monitors which signal sources correlate with successful surgeries
    and adjusts weights accordingly.
    """

    def __init__(
        self,
        phase_key: str = "phase_0",
        adaptation_rate: float = 0.02,
        min_weight: float = 0.05,
        max_weight: float = 0.9,
        history_window: int = 50,
    ):
        super().__init__(phase_key=phase_key)
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.history_window = history_window
        self._success_history: list[dict[str, float]] = []
        self._source_contributions: dict[str, list[bool]] = {
            "self": [],
            "external": [],
            "rag": [],
            "cross": [],
        }

    def record_outcome(
        self,
        sources_used: list[str],
        surgery_accepted: bool,
        score_delta: float,
    ) -> None:
        """
        Record the outcome of a surgery to adapt future weights.

        Args:
            sources_used: Which signal sources contributed.
            surgery_accepted: Whether the surgery was accepted.
            score_delta: Score change (positive = improvement).
        """
        for source in sources_used:
            if source in self._source_contributions:
                self._source_contributions[source].append(
                    surgery_accepted and score_delta > 0
                )
                # Keep only recent history
                if len(self._source_contributions[source]) > self.history_window:
                    self._source_contributions[source] = (
                        self._source_contributions[source][-self.history_window:]
                    )

        self._adapt_weights()

    def _adapt_weights(self) -> None:
        """Adjust weights based on source success rates."""
        source_scores: dict[str, float] = {}

        for source, outcomes in self._source_contributions.items():
            if len(outcomes) < 5:
                continue
            success_rate = sum(1 for o in outcomes if o) / len(outcomes)
            source_scores[source] = success_rate

        if not source_scores:
            return

        # Adjust weights proportionally to success rates
        for source, score in source_scores.items():
            current = self.weights.get(source, 0.0)
            if score > 0.6:
                # High success rate — increase weight
                new = min(self.max_weight, current + self.adaptation_rate)
            elif score < 0.3:
                # Low success rate — decrease weight
                new = max(self.min_weight, current - self.adaptation_rate)
            else:
                new = current

            self.weights[source] = new

        # Re-normalize
        total = sum(self.weights.values())
        if total > 0:
            for k in self.weights:
                self.weights[k] /= total

    def get_effectiveness_report(self) -> dict[str, Any]:
        """Generate a report on signal source effectiveness."""
        report: dict[str, Any] = {}
        for source, outcomes in self._source_contributions.items():
            if not outcomes:
                report[source] = {"samples": 0, "success_rate": None}
                continue
            report[source] = {
                "samples": len(outcomes),
                "success_rate": sum(1 for o in outcomes if o) / len(outcomes),
                "current_weight": self.weights.get(source, 0.0),
            }
        return report
