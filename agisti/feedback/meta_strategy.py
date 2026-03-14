"""
MetaStrategyEngine — adapts training strategy based on performance signals.

Implements the 4 adaptation rules from §8:
1. Alert-based actions (respond to catastrophe alerts)
2. LoRA rank adaptation (adjust rank based on performance)
3. Target layer optimization (focus surgery on responsive layers)
4. Macro surgery promotion (when micro surgery plateaus)

Design: §8 — Meta-strategy adaptation.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from agisti.types import (
    Alert,
    AlertLevel,
    AlertType,
    ConvergenceAction,
    IterationResult,
    SurgeryType,
)
from agisti.config import MetaStrategy, PHASE0_STRATEGY, ConvergenceConfig

logger = logging.getLogger(__name__)


@dataclass
class StrategyUpdate:
    """A proposed change to the training strategy."""
    field: str  # which field of MetaStrategy to change
    old_value: Any
    new_value: Any
    reason: str
    confidence: float  # 0-1, how confident in this change
    rule: str  # which rule triggered this

    def __repr__(self) -> str:
        return (
            f"StrategyUpdate({self.field}: {self.old_value} → "
            f"{self.new_value}, reason='{self.reason}')"
        )


class MetaStrategyEngine:
    """
    Adapts the training strategy based on iteration results.

    Watches performance signals across iterations and adjusts
    the strategy to maximize learning efficiency. Implements
    a feedback loop: performance → strategy → action → performance.

    The 4 rules:
    1. ALERT RESPONSE: When catastrophe detector fires, adjust strategy
    2. RANK ADAPTATION: If improvement stalls, increase LoRA rank
    3. LAYER TARGETING: Focus on layers that respond best to surgery
    4. SURGERY PROMOTION: Escalate micro → macro when micro plateaus
    """

    def __init__(
        self,
        initial_strategy: MetaStrategy | None = None,
        convergence_config: ConvergenceConfig | None = None,
        window_size: int = 20,
    ):
        self.strategy = initial_strategy or PHASE0_STRATEGY
        self.convergence = convergence_config or ConvergenceConfig()
        self.window_size = window_size

        # History tracking
        self._results: deque[IterationResult] = deque(maxlen=1000)
        self._strategy_history: list[tuple[int, MetaStrategy, list[StrategyUpdate]]] = []
        self._alerts: deque[Alert] = deque(maxlen=500)
        self._improvements: deque[float] = deque(maxlen=200)
        self._iteration = 0

    @property
    def current_strategy(self) -> MetaStrategy:
        return self.strategy

    def update(
        self,
        result: IterationResult,
        alerts: list[Alert] | None = None,
    ) -> list[StrategyUpdate]:
        """
        Process an iteration result and adapt strategy.

        Args:
            result: The iteration result to process.
            alerts: Any alerts from catastrophe detection.

        Returns:
            List of strategy updates applied.
        """
        self._iteration += 1
        self._results.append(result)
        if alerts:
            self._alerts.extend(alerts)

        # Track improvement rate
        if len(self._results) >= 2:
            prev = self._results[-2]
            improvement = (
                result.quick_bench.accuracy - prev.quick_bench.accuracy
                if result.quick_bench and prev.quick_bench
                else 0.0
            )
            self._improvements.append(improvement)

        # Apply rules
        updates: list[StrategyUpdate] = []

        # Rule 1: Alert response
        if alerts:
            updates.extend(self._rule_alert_response(alerts))

        # Rule 2: Rank adaptation
        rank_updates = self._rule_rank_adaptation()
        updates.extend(rank_updates)

        # Rule 3: Layer targeting
        layer_updates = self._rule_layer_targeting(result)
        updates.extend(layer_updates)

        # Rule 4: Surgery promotion
        promo_updates = self._rule_surgery_promotion()
        updates.extend(promo_updates)

        # Apply all updates
        for update in updates:
            self._apply_update(update)

        if updates:
            self._strategy_history.append(
                (self._iteration, self.strategy, updates),
            )
            logger.info(
                "MetaStrategy updated (%d changes): %s",
                len(updates),
                "; ".join(u.reason for u in updates),
            )

        return updates

    def check_convergence(self) -> ConvergenceAction:
        """
        Check if training has converged and what action to take.

        Uses the convergence config thresholds.
        """
        if len(self._improvements) < self.convergence.window_size:
            return ConvergenceAction.CONTINUE

        recent = list(self._improvements)[-self.convergence.window_size:]
        avg_improvement = sum(recent) / len(recent)

        # Standard convergence: improvement below threshold
        if avg_improvement < self.convergence.delta_min:
            if avg_improvement < 0:
                return ConvergenceAction.ROLLBACK
            return ConvergenceAction.STOP

        return ConvergenceAction.CONTINUE

    def _rule_alert_response(
        self, alerts: list[Alert],
    ) -> list[StrategyUpdate]:
        """
        Rule 1: Respond to catastrophe alerts.

        - REGRESSION: Reduce learning rate (smaller deltas)
        - DIVERGENCE: Roll back rank increase
        - OSCILLATION: Increase LoRA rank for stability
        - FROZEN_VIOLATION: Emergency stop
        """
        updates: list[StrategyUpdate] = []

        for alert in alerts:
            if alert.level not in (AlertLevel.CRITICAL, AlertLevel.WARNING):
                continue

            if alert.type == AlertType.REGRESSION:
                # Reduce surgery aggressiveness
                old_rank = self.strategy.lora_rank
                new_rank = max(2, old_rank // 2)
                if new_rank != old_rank:
                    updates.append(StrategyUpdate(
                        field="lora_rank",
                        old_value=old_rank,
                        new_value=new_rank,
                        reason=f"Regression detected: reducing rank {old_rank}→{new_rank}",
                        confidence=0.8,
                        rule="alert_response",
                    ))

            elif alert.type == AlertType.DIVERGENCE:
                # Switch to smaller surgeries
                if self.strategy.surgery_type != SurgeryType.MICRO:
                    updates.append(StrategyUpdate(
                        field="surgery_type",
                        old_value=self.strategy.surgery_type,
                        new_value=SurgeryType.MICRO,
                        reason="Divergence: switching to micro surgery",
                        confidence=0.9,
                        rule="alert_response",
                    ))

            elif alert.type == AlertType.OSCILLATION:
                # Increase rank for more parameters / stable gradients
                old_rank = self.strategy.lora_rank
                new_rank = min(64, old_rank * 2)
                if new_rank != old_rank:
                    updates.append(StrategyUpdate(
                        field="lora_rank",
                        old_value=old_rank,
                        new_value=new_rank,
                        reason=f"Oscillation: increasing rank {old_rank}→{new_rank}",
                        confidence=0.7,
                        rule="alert_response",
                    ))

            elif alert.type == AlertType.FROZEN_VIOLATION:
                updates.append(StrategyUpdate(
                    field="emergency_stop",
                    old_value=False,
                    new_value=True,
                    reason="FROZEN ZONE VIOLATION: Emergency stop",
                    confidence=1.0,
                    rule="alert_response",
                ))

        return updates

    def _rule_rank_adaptation(self) -> list[StrategyUpdate]:
        """
        Rule 2: Adapt LoRA rank based on improvement rate.

        If improvement is slowing but not converged,
        increase rank to give the optimizer more capacity.
        """
        updates: list[StrategyUpdate] = []

        if len(self._improvements) < self.window_size:
            return updates

        recent = list(self._improvements)[-self.window_size:]
        avg = sum(recent) / len(recent)

        # Compute acceleration (change in improvement rate)
        if len(self._improvements) >= self.window_size * 2:
            older = list(self._improvements)[
                -self.window_size * 2:-self.window_size
            ]
            older_avg = sum(older) / len(older)
            acceleration = avg - older_avg
        else:
            acceleration = 0.0

        current_rank = self.strategy.lora_rank

        # Improvement slowing (deceleration) but still positive
        if 0 < avg < 0.005 and acceleration < -0.001:
            new_rank = min(64, current_rank + 4)
            if new_rank != current_rank:
                updates.append(StrategyUpdate(
                    field="lora_rank",
                    old_value=current_rank,
                    new_value=new_rank,
                    reason=(
                        f"Improvement decelerating (avg={avg:.4f}, "
                        f"accel={acceleration:.4f}): rank {current_rank}→{new_rank}"
                    ),
                    confidence=0.6,
                    rule="rank_adaptation",
                ))

        # Improvement strong: can reduce rank for efficiency
        elif avg > 0.02 and current_rank > 8:
            new_rank = max(4, current_rank - 2)
            updates.append(StrategyUpdate(
                field="lora_rank",
                old_value=current_rank,
                new_value=new_rank,
                reason=f"Strong improvement: reducing rank for efficiency",
                confidence=0.5,
                rule="rank_adaptation",
            ))

        return updates

    def _rule_layer_targeting(
        self, result: IterationResult,
    ) -> list[StrategyUpdate]:
        """
        Rule 3: Focus surgery on the most responsive layers.

        Tracks which layers produced the most improvement
        and shifts surgery focus toward them.
        """
        updates: list[StrategyUpdate] = []

        if not result.delta:
            return updates

        # Check if current target layers are producing results
        current_layers = list(self.strategy.target_layers)
        if not current_layers:
            return updates

        # Only update every N iterations
        if self._iteration % 10 != 0:
            return updates

        # Look at improvement patterns
        if len(self._results) < 10:
            return updates

        # Analyze which iterations had the best improvement
        recent = list(self._results)[-10:]
        best_iter = max(recent, key=lambda r: (
            r.quick_bench.accuracy if r.quick_bench else 0.0
        ))

        if best_iter.delta and best_iter.delta.layer_names:
            best_layers = best_iter.delta.layer_names[:10]
            if set(best_layers) != set(current_layers):
                updates.append(StrategyUpdate(
                    field="target_layers",
                    old_value=current_layers,
                    new_value=best_layers,
                    reason="Targeting most responsive layers",
                    confidence=0.5,
                    rule="layer_targeting",
                ))

        return updates

    def _rule_surgery_promotion(self) -> list[StrategyUpdate]:
        """
        Rule 4: Promote micro → macro when micro surgery plateaus.

        If micro surgery hasn't produced improvement for N iterations,
        escalate to macro surgery for more transformative changes.
        """
        updates: list[StrategyUpdate] = []

        if self.strategy.surgery_type != SurgeryType.MICRO:
            return updates

        if len(self._improvements) < 30:
            return updates

        recent = list(self._improvements)[-30:]
        avg = sum(recent) / len(recent)

        # Micro surgery plateau: promote to macro
        if avg < 0.001:
            updates.append(StrategyUpdate(
                field="surgery_type",
                old_value=SurgeryType.MICRO,
                new_value=SurgeryType.MACRO,
                reason=(
                    f"Micro surgery plateau (avg improvement={avg:.5f}): "
                    f"promoting to macro"
                ),
                confidence=0.7,
                rule="surgery_promotion",
            ))

            # Also increase rank for macro
            old_rank = self.strategy.lora_rank
            new_rank = max(old_rank, 16)
            if new_rank != old_rank:
                updates.append(StrategyUpdate(
                    field="lora_rank",
                    old_value=old_rank,
                    new_value=new_rank,
                    reason="Macro surgery needs higher rank",
                    confidence=0.7,
                    rule="surgery_promotion",
                ))

        return updates

    def _apply_update(self, update: StrategyUpdate) -> None:
        """Apply a strategy update."""
        if update.field == "lora_rank":
            self.strategy.lora_rank = update.new_value
        elif update.field == "surgery_type":
            self.strategy.surgery_type = update.new_value
        elif update.field == "target_layers":
            self.strategy.target_layers = update.new_value
        elif update.field == "emergency_stop":
            self.strategy.emergency_stop = update.new_value
        else:
            logger.warning("Unknown strategy field: %s", update.field)

    def get_history(
        self, last_n: int | None = None,
    ) -> list[tuple[int, MetaStrategy, list[StrategyUpdate]]]:
        """Get strategy update history."""
        if last_n is not None:
            return self._strategy_history[-last_n:]
        return list(self._strategy_history)

    def reset(self, strategy: MetaStrategy | None = None) -> None:
        """Reset strategy to initial state."""
        self.strategy = strategy or PHASE0_STRATEGY
        self._results.clear()
        self._improvements.clear()
        self._alerts.clear()
        self._strategy_history.clear()
        self._iteration = 0
