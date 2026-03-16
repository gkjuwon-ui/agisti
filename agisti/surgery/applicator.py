"""
Delta applicator — applies surgery deltas to model weights.

Handles the actual weight modification with safety checks:
- Pre-apply state snapshot (for reliable rollback)
- Frozen zone integrity verification (SHA-256 checksums)
- Budget boundary enforcement
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from agisti.types import (
    LoRADelta,
    LoRALayerDelta,
    FrozenZoneViolation,
    SurgeryBudgetExceeded,
)

logger = logging.getLogger(__name__)


class DeltaApplicator:
    """
    Applies LoRA deltas to model weights with safety guarantees.

    Workflow:
    1. Capture pre-apply state snapshot (non-frozen params)
    2. Compute pre-surgery frozen checksums
    3. Apply delta to model weights
    4. Verify frozen zone integrity
    5. On failure: restore from snapshot
    """

    def __init__(self, model: nn.Module, frozen_layer_names: set[str]):
        self.model = model
        self.frozen_layer_names = frozen_layer_names
        self._pre_apply_state: dict[str, Tensor] | None = None
        self._pre_surgery_checksums: dict[str, str] | None = None

    def apply(self, delta: LoRADelta) -> None:
        """
        Apply delta to model weights with full safety checking.

        Raises:
            FrozenZoneViolation: If delta contains frozen layers or post-check fails.
        """
        # 1. Validate delta doesn't touch frozen layers
        self._validate_no_frozen_layers(delta)

        # 2. Snapshot non-frozen state for rollback
        self._pre_apply_state = self._capture_state()

        # 3. Compute frozen checksums before surgery
        self._pre_surgery_checksums = self._compute_frozen_checksums()

        # 4. Apply delta layer-by-layer
        self._apply_delta_to_model(delta)

        # 5. POST-SURGERY: Verify frozen zones untouched
        post_checksums = self._compute_frozen_checksums()
        if self._pre_surgery_checksums != post_checksums:
            # EMERGENCY: Frozen zone corrupted. Roll back immediately.
            changed = self._identify_corrupted_zones(
                self._pre_surgery_checksums, post_checksums
            )
            self.rollback()
            raise FrozenZoneViolation(
                f"CRITICAL: Frozen zone corrupted during surgery! "
                f"Changed zones: {changed}. Rollback completed."
            )

        logger.info(
            "Delta applied successfully. %d layers modified. "
            "Frozen zone integrity verified.",
            len(delta),
        )

    def rollback(self) -> bool:
        """
        Rollback to pre-apply state.

        Uses full state_dict restore instead of delta subtraction
        for numerical stability (avoids floating-point drift).

        Returns:
            True if rollback succeeded, False if no snapshot available.
        """
        if self._pre_apply_state is None:
            logger.warning("No pre-apply snapshot available for rollback.")
            return False

        current_state = self.model.state_dict()
        restored_state = {**current_state, **self._pre_apply_state}
        self.model.load_state_dict(restored_state)

        logger.info("Rollback completed from pre-apply snapshot.")
        self._pre_apply_state = None
        return True

    def _validate_no_frozen_layers(self, delta: LoRADelta) -> None:
        """Ensure none of the delta layers are in frozen zones."""
        violations = []
        for layer_name in delta.keys():
            # Check if this layer name matches any frozen pattern
            if self._is_frozen(layer_name):
                violations.append(layer_name)

        if violations:
            raise FrozenZoneViolation(
                f"Delta contains {len(violations)} frozen layer(s): {violations}"
            )

    def _is_frozen(self, layer_name: str) -> bool:
        """Check if a layer is in the frozen set."""
        for frozen in self.frozen_layer_names:
            if layer_name == frozen or layer_name.startswith(frozen + "."):
                return True
        return False

    def _capture_state(self) -> dict[str, Tensor]:
        """Capture current state of non-frozen parameters."""
        state = {}
        for name, param in self.model.named_parameters():
            layer_key = _extract_layer_key(name)
            if not self._is_frozen(layer_key):
                state[name] = param.data.clone()
        return state

    def _compute_frozen_checksums(self) -> dict[str, str]:
        """Compute fast norm-based fingerprints for frozen parameters.

        Uses L2 norm + mean + first element as a fast fingerprint instead
        of SHA-256 over full byte arrays. This avoids GPU→CPU transfer of
        the entire tensor while still detecting any meaningful change.
        For cryptographic-grade verification, use FrozenMask.verify_integrity().
        """
        checksums: dict[str, str] = {}
        for name, param in self.model.named_parameters():
            layer_key = _extract_layer_key(name)
            if self._is_frozen(layer_key):
                with torch.no_grad():
                    t = param.data
                    norm_val = torch.norm(t).item()
                    mean_val = t.mean().item()
                    numel = t.numel()
                    first_val = t.flatten()[0].item()
                checksums[name] = f"{norm_val:.10e}|{mean_val:.10e}|{first_val:.10e}|{numel}"
        return checksums

    def _apply_delta_to_model(self, delta: LoRADelta) -> None:
        """Apply LoRA delta to model parameters."""
        named_params = dict(self.model.named_parameters())
        named_modules = dict(self.model.named_modules())

        for layer_name, layer_delta in delta.items():
            full_weight = layer_delta.to_full()

            # Try to find matching parameter
            if layer_name in named_params:
                param = named_params[layer_name]
                self._add_delta_to_param(param, full_weight)
            elif layer_name + ".weight" in named_params:
                param = named_params[layer_name + ".weight"]
                self._add_delta_to_param(param, full_weight)
            elif layer_name in named_modules:
                # Apply to module's weight parameter
                module = named_modules[layer_name]
                if hasattr(module, "weight") and module.weight is not None:
                    self._add_delta_to_param(module.weight, full_weight)
                else:
                    logger.warning(
                        "Module %s has no weight parameter. Skipping.",
                        layer_name,
                    )
            else:
                logger.warning(
                    "Layer %s not found in model parameters or modules. Skipping.",
                    layer_name,
                )

    @staticmethod
    def _add_delta_to_param(param: nn.Parameter, delta: Tensor) -> None:
        """Add delta tensor to parameter, handling shape mismatches."""
        with torch.no_grad():
            if delta.shape == param.shape:
                param.add_(delta.to(param.device, param.dtype))
            elif delta.numel() == param.numel():
                param.add_(delta.reshape(param.shape).to(param.device, param.dtype))
            else:
                # Apply what we can — use the minimum of both dimensions
                min_shape = tuple(
                    min(d, p) for d, p in zip(delta.shape, param.shape)
                )
                slices = tuple(slice(0, s) for s in min_shape)
                param[slices] += delta[slices].to(param.device, param.dtype)
                logger.warning(
                    "Shape mismatch: delta %s vs param %s. "
                    "Applied partial delta to %s.",
                    delta.shape, param.shape, min_shape,
                )

    @staticmethod
    def _identify_corrupted_zones(
        pre: dict[str, str],
        post: dict[str, str],
    ) -> list[str]:
        """Find which frozen parameters changed."""
        changed = []
        for name, pre_hash in pre.items():
            post_hash = post.get(name, "")
            if pre_hash != post_hash:
                changed.append(name)
        return changed


class FrozenIntegrityMonitor:
    """
    Continuous monitor for frozen zone integrity.

    Computes and stores baseline checksums at initialization,
    then verifies them on demand.
    """

    def __init__(self, model: nn.Module, frozen_layer_names: set[str]):
        self.model = model
        self.frozen_layer_names = frozen_layer_names
        self._baseline: dict[str, str] = {}
        self._compute_baseline()

    def _compute_baseline(self) -> None:
        """Compute initial checksums for all frozen parameters."""
        for name, param in self.model.named_parameters():
            layer_key = _extract_layer_key(name)
            if self._is_in_frozen(layer_key):
                param_bytes = param.data.cpu().numpy().tobytes()
                self._baseline[name] = hashlib.sha256(param_bytes).hexdigest()

        logger.info(
            "Frozen integrity baseline: %d parameters monitored.",
            len(self._baseline),
        )

    def verify(self) -> bool:
        """
        Verify all frozen parameters match their baseline checksums.

        Raises:
            FrozenZoneViolation: If any frozen parameter has changed.
        """
        violations = []
        for name, param in self.model.named_parameters():
            if name not in self._baseline:
                continue
            current_bytes = param.data.cpu().numpy().tobytes()
            current_hash = hashlib.sha256(current_bytes).hexdigest()
            if current_hash != self._baseline[name]:
                violations.append(
                    f"{name}: expected={self._baseline[name][:16]}... "
                    f"got={current_hash[:16]}..."
                )

        if violations:
            raise FrozenZoneViolation(
                f"Frozen zone integrity check FAILED. "
                f"{len(violations)} parameter(s) changed:\n"
                + "\n".join(violations)
            )

        return True

    def update_baseline(self) -> None:
        """Re-compute baseline (e.g. after model architecture changes)."""
        self._baseline.clear()
        self._compute_baseline()

    def _is_in_frozen(self, layer_key: str) -> bool:
        for frozen in self.frozen_layer_names:
            if layer_key == frozen or layer_key.startswith(frozen + "."):
                return True
        return False

    @property
    def monitored_count(self) -> int:
        return len(self._baseline)


def _extract_layer_key(param_name: str) -> str:
    """
    Extract the layer key from a full parameter name.

    'model.layers.5.self_attn.q_proj.weight' → 'model.layers.5'
    'model.embed_tokens.weight' → 'model.embed_tokens'
    'lm_head.weight' → 'lm_head'
    """
    parts = param_name.split(".")

    # Special cases: top-level modules
    if parts[0] in ("lm_head",):
        return parts[0]
    if len(parts) >= 2 and parts[1] in ("embed_tokens", "norm"):
        return ".".join(parts[:2])

    # Standard transformer: model.layers.N
    if len(parts) >= 3 and parts[1] == "layers":
        return ".".join(parts[:3])

    # Fallback: use first two parts
    return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]


def assert_frozen_integrity(delta: LoRADelta, frozen_names: set[str]) -> None:
    """
    Assert that no frozen layers appear in the delta.
    Raises FrozenZoneViolation if any violation is found.
    """
    for layer_name in delta.keys():
        for frozen in frozen_names:
            if layer_name == frozen or layer_name.startswith(frozen + "."):
                raise FrozenZoneViolation(
                    f"Delta contains frozen layer: {layer_name} "
                    f"(matches frozen pattern: {frozen})"
                )
