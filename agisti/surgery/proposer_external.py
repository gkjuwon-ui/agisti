"""
External surgery proposer — blends self and external signals.

Implements propose_surgery_with_external() from AGISTI design §11.1.1.
The key distinction from the basic proposer is that external signals
are based on ground truth from external sources, not self-evaluation.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor

from agisti.types import (
    LoRADelta,
    LoRALayerDelta,
    SelfSignal,
    ExternalSignal,
    RAGSignal,
    CrossSignal,
    BlendedSignal,
)
from agisti.surgery.delta import DeltaFactory, BudgetEnforcer, DeltaComposer
from agisti.surgery.proposer import DirectionalAnalyzer

logger = logging.getLogger(__name__)


class ExternalSurgeryProposer:
    """
    Heuristic Direction Oracle v2 — Self + External Blended.

    Blends multiple signal sources:
    1. Self signal: activation contrast from self-generated problems
    2. External signal: contrast from external problems with ground truth
    3. RAG signal: contrast from retrieval-augmented re-solving (Phase 2)
    4. Cross signal: contrast from inter-model comparison (Phase 3)

    The external_weight adapts based on performance trends.
    """

    def __init__(
        self,
        lora_rank: int = 4,
        budget: float = 0.01,
        external_weight: float = 0.5,
    ):
        self.lora_rank = lora_rank
        self.budget = budget
        self.external_weight = external_weight
        self._enforcer = BudgetEnforcer(max_budget=budget)
        self._analyzer = DirectionalAnalyzer()

    def propose_with_external(
        self,
        self_signal: SelfSignal,
        external_signal: ExternalSignal,
        target_layers: list[str],
        frozen_layer_names: set[str],
        rag_signal: RAGSignal | None = None,
        cross_signal: CrossSignal | None = None,
        signal_weights: dict[str, float] | None = None,
    ) -> tuple[LoRADelta, BlendedSignal]:
        """
        Propose surgery delta by blending multiple signal sources.

        Args:
            self_signal: From self-generated problem evaluation.
            external_signal: From external problem evaluation (ground truth based).
            target_layers: Layers eligible for surgery.
            frozen_layer_names: Layers that must not be modified.
            rag_signal: Optional RAG-based signal (Phase 2+).
            cross_signal: Optional cross-model signal (Phase 3+).
            signal_weights: Override weights {source → weight}.

        Returns:
            Tuple of (proposed delta, blended signal info).
        """
        if signal_weights is None:
            signal_weights = self._default_weights(
                external_signal, rag_signal, cross_signal
            )

        modifiable_layers = [
            l for l in target_layers if l not in frozen_layer_names
        ]
        if not modifiable_layers:
            return LoRADelta(rank=self.lora_rank), BlendedSignal(
                contrasts={}, weights=signal_weights, sources_used=[],
            )

        # Collect available contrasts
        available_contrasts: list[dict[str, Tensor]] = []
        available_weights: list[float] = []
        sources_used: list[str] = []

        if self_signal.contrasts:
            available_contrasts.append(self_signal.contrasts)
            available_weights.append(signal_weights.get("self", 0.0))
            sources_used.append("self")

        if external_signal.usable and external_signal.contrasts:
            available_contrasts.append(external_signal.contrasts)
            available_weights.append(signal_weights.get("external", 0.0))
            sources_used.append("external")

        if rag_signal and rag_signal.usable and rag_signal.contrasts:
            available_contrasts.append(rag_signal.contrasts)
            available_weights.append(signal_weights.get("rag", 0.0))
            sources_used.append("rag")

        if cross_signal and cross_signal.usable and cross_signal.contrasts:
            available_contrasts.append(cross_signal.contrasts)
            available_weights.append(signal_weights.get("cross", 0.0))
            sources_used.append("cross")

        if not available_contrasts:
            logger.warning("No usable signal sources available.")
            return LoRADelta(rank=self.lora_rank), BlendedSignal(
                contrasts={}, weights=signal_weights, sources_used=[],
            )

        # Direction-aware blending for self + external
        budget_scale = 1.0
        if len(available_contrasts) >= 2 and "self" in sources_used and "external" in sources_used:
            budget_scale = self._directional_budget_adjustment(
                self_signal.contrasts,
                external_signal.contrasts,
                modifiable_layers,
            )

        # Blend all contrasts
        blended_contrasts = DeltaComposer.blend_contrasts(
            available_contrasts, available_weights
        )

        # Convert to delta
        adjusted_budget = self.budget * budget_scale
        delta = DeltaComposer.contrasts_to_delta(
            blended_contrasts=blended_contrasts,
            rank=self.lora_rank,
            total_budget=adjusted_budget,
            frozen_layer_names=frozen_layer_names,
        )

        blended = BlendedSignal(
            contrasts=blended_contrasts,
            weights={s: w for s, w in zip(sources_used, available_weights)},
            sources_used=sources_used,
        )

        logger.info(
            "External proposer: sources=%s, budget_scale=%.2f, delta_norm=%.6f",
            sources_used, budget_scale, delta.norm(),
        )
        return delta, blended

    def _default_weights(
        self,
        external_signal: ExternalSignal,
        rag_signal: RAGSignal | None,
        cross_signal: CrossSignal | None,
    ) -> dict[str, float]:
        """Compute default weights based on available signals."""
        weights: dict[str, float] = {"self": 1.0 - self.external_weight}

        if external_signal.usable:
            weights["external"] = self.external_weight

        if rag_signal and rag_signal.usable:
            # Redistribute: take from self and external equally
            rag_share = 0.3
            total_existing = sum(weights.values())
            for k in weights:
                weights[k] *= (1 - rag_share) / total_existing
            weights["rag"] = rag_share

        if cross_signal and cross_signal.usable:
            cross_share = 0.2
            total_existing = sum(weights.values())
            for k in weights:
                weights[k] *= (1 - cross_share) / total_existing
            weights["cross"] = cross_share

        return weights

    def _directional_budget_adjustment(
        self,
        self_contrasts: dict[str, Tensor],
        ext_contrasts: dict[str, Tensor],
        layers: list[str],
    ) -> float:
        """
        Check direction agreement across layers and adjust budget.

        If self and external signals consistently disagree, reduce budget.
        """
        sims = []
        for layer in layers:
            if layer in self_contrasts and layer in ext_contrasts:
                sim = self._analyzer.cosine_similarity(
                    self_contrasts[layer], ext_contrasts[layer]
                )
                sims.append(sim)

        if not sims:
            return 1.0

        avg_sim = sum(sims) / len(sims)

        if avg_sim < -0.3:
            # Consistent disagreement — be cautious
            logger.warning(
                "Self and external signals disagree (avg cos_sim=%.3f). "
                "Reducing budget to 50%%.",
                avg_sim,
            )
            return 0.5
        elif avg_sim > 0.3:
            # Good agreement — full budget
            return 1.0
        else:
            # Orthogonal — slightly cautious
            return 0.8


class ExternalWeightAdapter:
    """
    Adapts external signal weight based on performance trends.

    Rules:
    - Self improving but external stagnating → increase external weight (ceiling approaching)
    - Both improving → maintain current weight
    - External declining → decrease weight (external signal may be noisy)
    """

    def __init__(
        self,
        min_weight: float = 0.1,
        max_weight: float = 0.8,
        adaptation_step: float = 0.05,
        trend_window: int = 20,
        min_data_points: int = 5,
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.adaptation_step = adaptation_step
        self.trend_window = trend_window
        self.min_data_points = min_data_points

    def adapt(
        self,
        current_weight: float,
        self_scores: list[float],
        external_scores: list[float],
    ) -> float:
        """
        Compute adapted external weight based on score trends.

        Args:
            current_weight: Current external signal weight.
            self_scores: Recent self-benchmark scores (chronological).
            external_scores: Recent external benchmark scores.

        Returns:
            New external weight.
        """
        if (len(self_scores) < self.min_data_points
                or len(external_scores) < self.min_data_points):
            return current_weight

        # Use most recent window
        self_recent = self_scores[-self.trend_window:]
        ext_recent = external_scores[-self.trend_window:]

        self_slope = self._compute_slope(self_recent)
        ext_slope = self._compute_slope(ext_recent)

        # Decision logic
        if self_slope > 0.003 and ext_slope < 0.001:
            # Ceiling approaching: self improving but external stagnating
            new_weight = min(self.max_weight, current_weight + self.adaptation_step)
            logger.info(
                "Ceiling signal: self_slope=%.4f, ext_slope=%.4f → "
                "increasing external weight %.3f → %.3f",
                self_slope, ext_slope, current_weight, new_weight,
            )
            return new_weight

        elif ext_slope < -0.003:
            # External signal declining: reduce weight
            new_weight = max(self.min_weight, current_weight - self.adaptation_step * 2)
            logger.info(
                "External noise signal: ext_slope=%.4f → "
                "decreasing external weight %.3f → %.3f",
                ext_slope, current_weight, new_weight,
            )
            return new_weight

        return current_weight

    @staticmethod
    def _compute_slope(values: list[float]) -> float:
        """Linear regression slope."""
        import numpy as np
        if len(values) < 2:
            return 0.0
        x = list(range(len(values)))
        coeffs = np.polyfit(x, values, 1)
        return float(coeffs[0])
