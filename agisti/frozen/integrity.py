"""
Frozen integrity — continuous integrity monitoring for frozen zones.

Provides runtime monitoring that verifies frozen parameters
haven't been modified between operations. This is a safety net
against bugs in the surgery pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from agisti.types import FrozenZoneViolation
from agisti.frozen.mask import FrozenMask

logger = logging.getLogger(__name__)


@dataclass
class IntegrityCheck:
    """Record of a single integrity check."""
    timestamp: float
    passed: bool
    violations: list[str]
    check_duration_ms: float
    context: str = ""


class IntegrityMonitor:
    """
    Continuous integrity monitoring for frozen zones.

    Runs checksums before and after every critical operation
    (surgery application, benchmark run, etc.) to ensure
    frozen parameters are never accidentally modified.
    """

    def __init__(
        self,
        frozen_mask: FrozenMask,
        check_interval: int = 1,  # check every N operations
        max_history: int = 1000,
    ):
        self.mask = frozen_mask
        self.check_interval = check_interval
        self.max_history = max_history
        self._history: list[IntegrityCheck] = []
        self._operation_count = 0
        self._violations_total = 0

    def check(
        self,
        model: nn.Module,
        context: str = "",
        raise_on_violation: bool = True,
    ) -> bool:
        """
        Run an integrity check.

        Args:
            model: Model to check.
            context: Description of what was happening when check ran.
            raise_on_violation: Whether to raise on failure.

        Returns:
            True if all frozen zones are intact.
        """
        self._operation_count += 1

        # Skip check based on interval
        if self._operation_count % self.check_interval != 0:
            return True

        start = time.monotonic()

        try:
            violations = self.mask.verify_integrity(
                model, raise_on_violation=False,
            )
        except Exception as e:
            logger.error("Integrity check failed with error: %s", e)
            violations = ["<check_error>"]

        duration_ms = (time.monotonic() - start) * 1000
        passed = len(violations) == 0

        check = IntegrityCheck(
            timestamp=time.time(),
            passed=passed,
            violations=violations,
            check_duration_ms=duration_ms,
            context=context,
        )
        self._record(check)

        if not passed:
            self._violations_total += len(violations)
            logger.error(
                "INTEGRITY CHECK FAILED [%s]: %d violations",
                context,
                len(violations),
            )
            if raise_on_violation:
                raise FrozenZoneViolation(
                    f"Integrity check failed ({context}): "
                    f"{', '.join(violations[:5])}"
                )
        else:
            logger.debug(
                "Integrity check OK [%s]: %.1fms",
                context,
                duration_ms,
            )

        return passed

    def pre_surgery_check(self, model: nn.Module) -> bool:
        """Run before applying surgery."""
        return self.check(model, "pre_surgery", raise_on_violation=True)

    def post_surgery_check(self, model: nn.Module) -> bool:
        """Run after applying surgery."""
        return self.check(model, "post_surgery", raise_on_violation=True)

    def pre_benchmark_check(self, model: nn.Module) -> bool:
        """Run before benchmark (should be read-only, but verify)."""
        return self.check(model, "pre_benchmark", raise_on_violation=True)

    @property
    def total_checks(self) -> int:
        return len(self._history)

    @property
    def total_violations(self) -> int:
        return self._violations_total

    @property
    def violation_rate(self) -> float:
        if not self._history:
            return 0.0
        return sum(
            1 for c in self._history if not c.passed
        ) / len(self._history)

    def recent_checks(self, n: int = 10) -> list[IntegrityCheck]:
        return self._history[-n:]

    def _record(self, check: IntegrityCheck) -> None:
        self._history.append(check)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]


class QuickIntegrityCheck:
    """
    Fast integrity check using parameter norms instead of full checksums.

    Much faster than full SHA-256 checksums but less precise.
    Good for frequent checks between surgery steps.
    """

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self._baseline_norms: dict[str, float] = {}

    def capture_baseline(self, model: nn.Module, frozen_layers: list[str]) -> None:
        """Capture norm baselines for frozen layers."""
        self._baseline_norms.clear()
        for name, module in model.named_modules():
            if name in frozen_layers:
                total_norm = sum(
                    p.data.float().norm().item()
                    for p in module.parameters()
                )
                self._baseline_norms[name] = total_norm

    def quick_check(self, model: nn.Module) -> list[str]:
        """
        Quick check: compare parameter norms.

        Returns list of layers with norm changes.
        """
        violations = []
        for name, module in model.named_modules():
            if name not in self._baseline_norms:
                continue
            current_norm = sum(
                p.data.float().norm().item()
                for p in module.parameters()
            )
            baseline = self._baseline_norms[name]
            if abs(current_norm - baseline) > self.tolerance:
                violations.append(name)
                logger.warning(
                    "Quick integrity: norm change in %s: "
                    "%.6f → %.6f (Δ=%.6f)",
                    name,
                    baseline,
                    current_norm,
                    abs(current_norm - baseline),
                )
        return violations

    def is_intact(self, model: nn.Module) -> bool:
        """Quick boolean check."""
        return len(self.quick_check(model)) == 0


def validate_frozen_before_surgery(
    model: nn.Module,
    frozen_mask: FrozenMask,
    delta_layers: list[str],
) -> None:
    """
    Validate that surgery delta doesn't target frozen layers.

    Raises FrozenZoneViolation if any delta layer is frozen.
    """
    for layer in delta_layers:
        if frozen_mask.is_frozen(layer):
            raise FrozenZoneViolation(
                f"Surgery delta targets frozen layer: {layer}. "
                f"Freeze level: {frozen_mask.get_level(layer).name}"
            )
