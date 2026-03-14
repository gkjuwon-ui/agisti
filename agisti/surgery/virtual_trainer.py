"""
Virtual Trainer — simulates surgery before real application.

Implements the VirtualTrainer from AGISTI design §4.5.
Uses forward hooks to virtually apply deltas and gradient-based
refinement to correct the heuristic direction oracle's magnitude.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from agisti.types import (
    LoRADelta,
    LoRALayerDelta,
    VirtualTrainResult,
    Problem,
)

logger = logging.getLogger(__name__)


class VirtualTrainer:
    """
    Simulates surgery effects before committing to real weight changes.

    1. Clones delta with requires_grad=True
    2. Installs grad-aware forward hooks (output + A @ B in computation graph)
    3. Computes loss before/after (answer-only cross-entropy)
    4. If loss decreases, refines delta via gradient descent
    5. Checks gradient flow on first step

    This is the CT scan before surgery.
    """

    def __init__(
        self,
        base_lr: float = 1e-5,
        max_steps: int = 10,
        budget_divergence_factor: float = 1.5,
        tokenizer: object | None = None,
    ):
        self.base_lr = base_lr
        self.max_steps = max_steps
        self.budget_divergence_factor = budget_divergence_factor
        self.tokenizer = tokenizer

    def run(
        self,
        model: nn.Module,
        proposed_delta: LoRADelta,
        validation_problems: list[Problem],
        frozen_mask_names: set[str],
    ) -> VirtualTrainResult:
        """
        Run virtual training simulation.

        Args:
            model: The model to simulate surgery on.
            proposed_delta: Proposed delta from the surgery proposer.
            validation_problems: Problems to compute loss on.
            frozen_mask_names: Frozen layer names (for safety check).

        Returns:
            VirtualTrainResult with loss comparison and refined delta.
        """
        # Safety: ensure no frozen layers are in the delta
        for layer_name in proposed_delta.keys():
            if layer_name in frozen_mask_names:
                raise ValueError(
                    f"Proposed delta contains frozen layer: {layer_name}"
                )

        if not validation_problems:
            return VirtualTrainResult(
                loss_before=0.0,
                loss_after=0.0,
                loss_decreased=False,
                refined_delta=None,
                refinement_steps=0,
                grad_flow_ok=False,
            )

        # 1. Clone delta with gradient tracking
        refined_delta = proposed_delta.clone_with_grad()

        # 2. Install forward hooks
        hooks = []
        hook_controls: list[_HookControl] = []
        for name, layer_delta in refined_delta.items():
            hook, control = self._install_grad_aware_hook(model, name, layer_delta)
            hooks.append(hook)
            hook_controls.append(control)

        try:
            # 3. Compute loss without hooks (baseline) — no_grad saves ~10GB VRAM
            for ctrl in hook_controls:
                ctrl.active = False
            with torch.no_grad():
                loss_before = self._compute_loss(model, validation_problems).item()

            # 4. Compute loss with hooks (virtual surgery) — no_grad measurement
            for ctrl in hook_controls:
                ctrl.active = True
            with torch.no_grad():
                loss_after = self._compute_loss(model, validation_problems).item()
            torch.cuda.empty_cache()

            logger.info(
                "Virtual train: loss_before=%.6f, loss_after=%.6f (delta=%.6f)",
                loss_before, loss_after,
                loss_after - loss_before,
            )

            # 5. Gradient refinement (only if loss decreased)
            grad_flow_ok = True
            refinement_steps = 0

            if loss_after < loss_before:
                # Enable gradient checkpointing to trade compute for VRAM
                # This prevents storing all intermediate activations during backward
                _gc_was_enabled = getattr(model, 'is_gradient_checkpointing', False)
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                    logger.debug("Gradient checkpointing enabled for refinement")

                try:
                    refinement_steps, grad_flow_ok = self._refine_delta(
                        model=model,
                        refined_delta=refined_delta,
                        validation_problems=validation_problems,
                        hook_controls=hook_controls,
                        original_norm=proposed_delta.norm(),
                    )
                finally:
                    if hasattr(model, 'gradient_checkpointing_disable') and not _gc_was_enabled:
                        model.gradient_checkpointing_disable()
            else:
                logger.info("Virtual train: loss did not decrease. No refinement.")

        finally:
            # 6. Clean up hooks
            for hook in hooks:
                hook.remove()
            # Aggressive GPU cleanup after refinement
            for ld in refined_delta.values():
                if hasattr(ld.A, 'grad') and ld.A.grad is not None:
                    ld.A.grad = None
                if hasattr(ld.B, 'grad') and ld.B.grad is not None:
                    ld.B.grad = None
            gc.collect()
            torch.cuda.empty_cache()

        loss_decreased = loss_after < loss_before
        return VirtualTrainResult(
            loss_before=loss_before,
            loss_after=loss_after,
            loss_decreased=loss_decreased,
            refined_delta=refined_delta.detach_all() if loss_decreased else None,
            refinement_steps=refinement_steps,
            grad_flow_ok=grad_flow_ok,
        )

    def _refine_delta(
        self,
        model: nn.Module,
        refined_delta: LoRADelta,
        validation_problems: list[Problem],
        hook_controls: list[_HookControl],
        original_norm: float,
    ) -> tuple[int, bool]:
        """
        Gradient-based delta refinement.

        δ^(k+1) = δ^(k) - η · ∇_δ L(θ + δ^(k), V)

        Returns:
            Tuple of (steps completed, gradient flow ok).
        """
        grad_flow_ok = True
        steps_completed = 0

        for step in range(self.max_steps):
            # Ensure hooks are active
            for ctrl in hook_controls:
                ctrl.active = True

            # Per-problem backward to avoid accumulating all computation graphs
            n_probs = max(1, len(validation_problems))
            for prob in validation_problems:
                ploss = self._single_problem_loss(model, prob)
                if ploss is not None:
                    (ploss / n_probs).backward()
                    del ploss
            gc.collect()
            torch.cuda.empty_cache()

            # Check gradient flow on first step
            if step == 0:
                grad_flow_ok = all(
                    ld.A.grad is not None and ld.B.grad is not None
                    for ld in refined_delta.values()
                )
                if not grad_flow_ok:
                    logger.warning(
                        "Gradient flow failed: A.grad or B.grad is None. "
                        "Returning unrefined delta."
                    )
                    break

            # Update delta parameters
            with torch.no_grad():
                for layer_delta in refined_delta.values():
                    if layer_delta.A.grad is not None:
                        layer_delta.A -= self.base_lr * layer_delta.A.grad
                        layer_delta.A.grad.zero_()
                    if layer_delta.B.grad is not None:
                        layer_delta.B -= self.base_lr * layer_delta.B.grad
                        layer_delta.B.grad.zero_()

            steps_completed = step + 1

            # Budget divergence check
            current_norm = refined_delta.norm()
            max_allowed = original_norm * self.budget_divergence_factor
            if current_norm > max_allowed:
                logger.warning(
                    "Refinement diverging: norm=%.6f > %.6f (%.1fx original). "
                    "Scaling back and stopping.",
                    current_norm, max_allowed, self.budget_divergence_factor,
                )
                refined_delta.scale_to(original_norm)
                break

        logger.info(
            "Refinement: %d steps, final_norm=%.6f, grad_ok=%s",
            steps_completed, refined_delta.norm(), grad_flow_ok,
        )
        return steps_completed, grad_flow_ok

    def _install_grad_aware_hook(
        self,
        model: nn.Module,
        layer_name: str,
        layer_delta: LoRALayerDelta,
    ) -> tuple[torch.utils.hooks.RemovableHook, _HookControl]:
        """
        Install a forward hook that applies A @ B within the computation graph.

        The hook adds (A @ B) to the output, ensuring that when
        loss.backward() is called, gradients flow to A.grad and B.grad.

        Key: output + (layer_delta.A @ layer_delta.B) keeps A, B in the graph
        because they have requires_grad=True.
        """
        named_modules = dict(model.named_modules())
        if layer_name not in named_modules:
            raise ValueError(f"Layer {layer_name} not found in model")

        target_module = named_modules[layer_name]
        control = _HookControl(active=True)

        def hook_fn(
            module: nn.Module,
            inp: tuple,
            output: Tensor | tuple,
        ) -> Tensor | tuple:
            if not control.active:
                return output

            # Handle tuple outputs (common in transformer layers)
            is_tuple = isinstance(output, tuple)
            out_tensor = output[0] if is_tuple else output

            # Apply delta in computation graph
            delta_full = layer_delta.A @ layer_delta.B  # (d_out, d_in)

            # Match dimensions: delta_full is (d_out, d_in), output is (..., d_out)
            if out_tensor.dim() >= 2 and delta_full.shape[0] == out_tensor.shape[-1]:
                # output + bias-like addition
                modified = out_tensor + delta_full.sum(dim=-1)
            elif out_tensor.shape == delta_full.shape:
                modified = out_tensor + delta_full
            else:
                # Fallback: add flattened and reshape
                flat_delta = delta_full.flatten()
                flat_out = out_tensor.reshape(-1)
                min_len = min(len(flat_delta), len(flat_out))
                modified = out_tensor.clone()
                modified.reshape(-1)[:min_len] += flat_delta[:min_len]

            if is_tuple:
                return (modified,) + output[1:]
            return modified

        hook = target_module.register_forward_hook(hook_fn)
        return hook, control

    def _compute_loss(
        self,
        model: nn.Module,
        problems: list[Problem],
    ) -> Tensor:
        """
        Compute cross-entropy loss on answer tokens only.

        Question tokens are masked to prevent surgery direction contamination
        from long questions with short answers.
        """
        total_loss = 0.0
        count = 0

        for problem in problems:
            loss = self._single_problem_loss(model, problem)
            if loss is not None:
                total_loss += loss.item()
                count += 1

        if count == 0:
            return torch.tensor(0.0)

        return torch.tensor(total_loss / count)

    def _single_problem_loss(
        self,
        model: nn.Module,
        problem: Problem,
    ) -> Tensor | None:
        """Compute answer-only loss for a single problem."""
        if self.tokenizer is None:
            return None

        device = next(model.parameters()).device

        question_ids = self.tokenizer.encode(
            problem.question, return_tensors="pt"
        ).to(device)
        full_text = problem.question + problem.answer
        full_ids = self.tokenizer.encode(
            full_text, return_tensors="pt"
        ).to(device)

        if full_ids.shape[1] <= question_ids.shape[1]:
            return None

        q_len = question_ids.shape[1]

        # Forward pass
        outputs = model(full_ids, use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # Answer-only loss: mask question tokens
        # logits[..., q_len-1:-1, :] predicts tokens at positions q_len onwards
        answer_logits = logits[:, q_len - 1:-1, :].contiguous()
        answer_targets = full_ids[:, q_len:].contiguous()

        if answer_logits.shape[1] == 0 or answer_targets.shape[1] == 0:
            return None

        loss = F.cross_entropy(
            answer_logits.reshape(-1, answer_logits.shape[-1]),
            answer_targets.reshape(-1),
        )
        return loss


class _HookControl:
    """Simple mutable flag to toggle hook activity."""

    def __init__(self, active: bool = True):
        self.active = active


class VirtualTrainerFactory:
    """Factory to create VirtualTrainer with appropriate config."""

    @staticmethod
    def for_micro_surgery(tokenizer: object | None = None) -> VirtualTrainer:
        return VirtualTrainer(
            base_lr=1e-5,
            max_steps=10,
            budget_divergence_factor=1.5,
            tokenizer=tokenizer,
        )

    @staticmethod
    def for_macro_surgery(tokenizer: object | None = None) -> VirtualTrainer:
        return VirtualTrainer(
            base_lr=5e-6,
            max_steps=50,
            budget_divergence_factor=1.3,
            tokenizer=tokenizer,
        )
