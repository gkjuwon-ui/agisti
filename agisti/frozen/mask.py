"""
FrozenMask — manages per-layer freeze state with integrity verification.

Maintains SHA-256 checksums for all frozen parameters to detect
any unauthorized modifications. This is a safety system: if frozen
parameters change, something has gone very wrong.

Design: §7 — Frozen integrity enforcement.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from agisti.types import FreezeLevel, FrozenZoneViolation

logger = logging.getLogger(__name__)


@dataclass
class LayerFreezeState:
    """Freeze state for a single layer."""
    layer_name: str
    level: FreezeLevel
    checksum: str  # SHA-256 of frozen parameters
    num_parameters: int
    param_names: list[str] = field(default_factory=list)


class FrozenMask:
    """
    Per-layer frozen mask with checksum integrity verification.

    Once a layer is frozen, its parameters are:
    1. Set to requires_grad=False
    2. Checksummed with SHA-256
    3. Verified before and after every surgery operation

    Any checksum mismatch triggers FrozenZoneViolation.
    """

    def __init__(self):
        self._freeze_states: dict[str, LayerFreezeState] = {}
        self._checksums: dict[str, str] = {}

    @property
    def frozen_layers(self) -> list[str]:
        return [
            name for name, state in self._freeze_states.items()
            if state.level in (FreezeLevel.FULL, FreezeLevel.PARTIAL)
        ]

    @property
    def trainable_layers(self) -> list[str]:
        return [
            name for name, state in self._freeze_states.items()
            if state.level == FreezeLevel.NONE
        ]

    @property
    def total_frozen(self) -> int:
        return sum(
            s.num_parameters for s in self._freeze_states.values()
            if s.level == FreezeLevel.FULL
        )

    def is_frozen(self, layer_name: str) -> bool:
        state = self._freeze_states.get(layer_name)
        return state is not None and state.level != FreezeLevel.NONE

    def get_level(self, layer_name: str) -> FreezeLevel:
        state = self._freeze_states.get(layer_name)
        return state.level if state else FreezeLevel.NONE

    def freeze_layer(
        self,
        model: nn.Module,
        layer_name: str,
        level: FreezeLevel = FreezeLevel.FULL,
    ) -> None:
        """
        Freeze a layer and compute its integrity checksum.

        Args:
            model: The model containing the layer.
            layer_name: Name of the layer to freeze.
            level: Freeze level (FULL or PARTIAL).
        """
        layer_module = self._get_layer_module(model, layer_name)
        if layer_module is None:
            logger.warning("Layer %s not found in model", layer_name)
            return

        param_names = []
        num_params = 0

        with torch.no_grad():
            for name, param in layer_module.named_parameters():
                if level == FreezeLevel.FULL:
                    param.requires_grad_(False)
                elif level == FreezeLevel.PARTIAL:
                    # Partial: freeze attention, keep MLP trainable
                    if "attn" in name or "attention" in name:
                        param.requires_grad_(False)
                param_names.append(name)
                num_params += param.numel()

        # Compute checksum
        checksum = self._compute_layer_checksum(layer_module)

        self._freeze_states[layer_name] = LayerFreezeState(
            layer_name=layer_name,
            level=level,
            checksum=checksum,
            num_parameters=num_params,
            param_names=param_names,
        )
        self._checksums[layer_name] = checksum

        logger.info(
            "Frozen layer %s (%s): %d params, checksum=%s...",
            layer_name,
            level.name,
            num_params,
            checksum[:16],
        )

    def freeze_from_report(
        self,
        model: nn.Module,
        frozen_layers: list[str],
        partial_layers: list[str] | None = None,
    ) -> None:
        """Apply freeze recommendations from a FrozenZoneReport."""
        for layer_name in frozen_layers:
            self.freeze_layer(model, layer_name, FreezeLevel.FULL)

        if partial_layers:
            for layer_name in partial_layers:
                self.freeze_layer(model, layer_name, FreezeLevel.PARTIAL)

    def unfreeze_layer(
        self,
        model: nn.Module,
        layer_name: str,
    ) -> None:
        """Unfreeze a layer (for surgical exception)."""
        layer_module = self._get_layer_module(model, layer_name)
        if layer_module is None:
            return

        with torch.no_grad():
            for param in layer_module.parameters():
                param.requires_grad_(True)

        if layer_name in self._freeze_states:
            del self._freeze_states[layer_name]
        if layer_name in self._checksums:
            del self._checksums[layer_name]

        logger.info("Unfroze layer %s", layer_name)

    def verify_integrity(
        self,
        model: nn.Module,
        raise_on_violation: bool = True,
    ) -> list[str]:
        """
        Verify all frozen layer checksums.

        Returns list of layers with integrity violations.
        Raises FrozenZoneViolation if any violations found.
        """
        violations = []

        for layer_name, state in self._freeze_states.items():
            if state.level == FreezeLevel.NONE:
                continue

            layer_module = self._get_layer_module(model, layer_name)
            if layer_module is None:
                logger.warning(
                    "Frozen layer %s no longer exists in model",
                    layer_name,
                )
                violations.append(layer_name)
                continue

            current_checksum = self._compute_layer_checksum(layer_module)
            if current_checksum != state.checksum:
                violations.append(layer_name)
                logger.error(
                    "FROZEN ZONE VIOLATION: Layer %s checksum mismatch! "
                    "Expected %s..., got %s...",
                    layer_name,
                    state.checksum[:16],
                    current_checksum[:16],
                )

        if violations and raise_on_violation:
            raise FrozenZoneViolation(
                f"Integrity violation in {len(violations)} frozen layers: "
                f"{', '.join(violations[:5])}"
            )

        if not violations:
            logger.debug("Frozen zone integrity verified: %d layers OK",
                        len(self._freeze_states))

        return violations

    def update_checksums(self, model: nn.Module) -> None:
        """
        Re-compute checksums after an authorized modification.

        This should ONLY be called after a deliberate, authorized
        change to frozen layers (e.g., during architecture surgery).
        """
        for layer_name, state in self._freeze_states.items():
            if state.level == FreezeLevel.NONE:
                continue

            layer_module = self._get_layer_module(model, layer_name)
            if layer_module is None:
                continue

            new_checksum = self._compute_layer_checksum(layer_module)
            state.checksum = new_checksum
            self._checksums[layer_name] = new_checksum

        logger.info("Updated checksums for %d frozen layers",
                    len(self._freeze_states))

    def get_trainable_params(
        self, model: nn.Module,
    ) -> list[tuple[str, nn.Parameter]]:
        """Get all parameters that are NOT frozen."""
        trainable = []
        frozen_prefixes = set(self.frozen_layers)

        for name, param in model.named_parameters():
            is_frozen = any(
                name.startswith(prefix) for prefix in frozen_prefixes
            )
            if not is_frozen and param.requires_grad:
                trainable.append((name, param))

        return trainable

    def freeze_ratio(self, model: nn.Module) -> float:
        """Compute the fraction of model parameters that are frozen."""
        total = sum(p.numel() for p in model.parameters())
        frozen = self.total_frozen
        return frozen / total if total > 0 else 0.0

    def save(self, path: str | Path) -> None:
        """Save freeze mask to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "freeze_states": {
                name: {
                    "level": state.level.value,
                    "checksum": state.checksum,
                    "num_parameters": state.num_parameters,
                    "param_names": state.param_names,
                }
                for name, state in self._freeze_states.items()
            },
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> FrozenMask:
        """Load freeze mask from JSON."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        mask = cls()
        for name, state_data in data.get("freeze_states", {}).items():
            mask._freeze_states[name] = LayerFreezeState(
                layer_name=name,
                level=FreezeLevel(state_data["level"]),
                checksum=state_data["checksum"],
                num_parameters=state_data["num_parameters"],
                param_names=state_data.get("param_names", []),
            )
            mask._checksums[name] = state_data["checksum"]

        return mask

    def _get_layer_module(
        self, model: nn.Module, layer_name: str,
    ) -> nn.Module | None:
        """Get a named module from the model."""
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        return None

    def _compute_layer_checksum(self, module: nn.Module) -> str:
        """
        Compute SHA-256 checksum of all parameters in a module.

        Concatenates all parameter bytes and hashes them.
        This is deterministic for the same parameter values.
        """
        hasher = hashlib.sha256()
        for name, param in sorted(module.named_parameters()):
            # Convert to bytes deterministically
            param_bytes = param.detach().cpu().float().numpy().tobytes()
            hasher.update(name.encode("utf-8"))
            hasher.update(param_bytes)
        return hasher.hexdigest()

    def summary(self) -> str:
        """Human-readable summary of freeze state."""
        full = sum(
            1 for s in self._freeze_states.values()
            if s.level == FreezeLevel.FULL
        )
        partial = sum(
            1 for s in self._freeze_states.values()
            if s.level == FreezeLevel.PARTIAL
        )
        none_ = sum(
            1 for s in self._freeze_states.values()
            if s.level == FreezeLevel.NONE
        )
        total_params = sum(
            s.num_parameters for s in self._freeze_states.values()
        )
        frozen_params = sum(
            s.num_parameters for s in self._freeze_states.values()
            if s.level in (FreezeLevel.FULL, FreezeLevel.PARTIAL)
        )

        return (
            f"FrozenMask: {full} full + {partial} partial + {none_} trainable, "
            f"{frozen_params:,}/{total_params:,} params frozen "
            f"({frozen_params/total_params:.1%})" if total_params > 0
            else "FrozenMask: (empty)"
        )
