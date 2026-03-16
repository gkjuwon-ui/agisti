"""
Ceiling Breaker Level 3 — Inter-Model Cross-Pollination.

When a single model exhausts its own architecture's representational
patterns, different models solving the same problems can reveal
alternative activation patterns that the target model is missing.

This is NOT distillation (copying outputs). It's extracting "directional
hints" from reference models' internal representations to guide surgery
on the target model.

Key operations:
1. CKA (Centered Kernel Alignment) for mapping layers across architectures
2. Procrustes alignment for projecting between representation spaces
3. Cross-contrast: ref_correct - target_wrong activations

Design: §11.1.3 — Inter-Model Cross-Pollination.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from agisti.types import CrossSignal, Problem

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CKA Implementation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def compute_cka(
    X: Tensor,
    Y: Tensor,
    debiased: bool = True,
) -> float:
    """
    Compute Centered Kernel Alignment between two activation matrices.

    CKA measures the similarity of learned representations between
    two layers (potentially from different models).

    Args:
        X: Activations from layer A, shape (n_samples, dim_x).
        Y: Activations from layer B, shape (n_samples, dim_y).
        debiased: Use debiased estimator (recommended).

    Returns:
        CKA similarity in [0, 1]. Higher = more similar representations.
    """
    # Handle potential shape mismatches by flattening
    if X.dim() > 2:
        X = X.flatten(start_dim=1)
    if Y.dim() > 2:
        Y = Y.flatten(start_dim=1)

    n = X.shape[0]
    if n < 3:
        return 0.0

    # Linear kernel CKA
    XX = X @ X.T  # (n, n)
    YY = Y @ Y.T  # (n, n)

    if debiased:
        return float(_debiased_cka(XX, YY, n))
    else:
        # Center the kernel matrices
        H = torch.eye(n, device=X.device) - torch.ones(n, n, device=X.device) / n
        XX_c = H @ XX @ H
        YY_c = H @ YY @ H

        hsic_xy = (XX_c * YY_c).sum() / ((n - 1) ** 2)
        hsic_xx = (XX_c * XX_c).sum() / ((n - 1) ** 2)
        hsic_yy = (YY_c * YY_c).sum() / ((n - 1) ** 2)

        product = float(hsic_xx * hsic_yy)
        if product <= 0:
            return 0.0

        denom = math.sqrt(product)
        if denom < 1e-10:
            return 0.0

        return float(hsic_xy / denom)


def _debiased_cka(K: Tensor, L: Tensor, n: int) -> float:
    """
    Debiased HSIC estimator for CKA.

    From Song et al. (2012) — avoids bias from finite sample size.
    """
    # Zero out diagonal (debiasing step)
    K_tilde = K.clone()
    L_tilde = L.clone()
    K_tilde.fill_diagonal_(0)
    L_tilde.fill_diagonal_(0)

    # HSIC terms
    term1 = (K_tilde * L_tilde).sum()
    term2 = K_tilde.sum() * L_tilde.sum() / ((n - 1) * (n - 2))
    term3 = 2.0 * (K_tilde.sum(dim=0) * L_tilde.sum(dim=0)).sum() / (n - 2)

    hsic = (term1 + term2 - term3) / (n * (n - 3))

    # Self-HSIC for normalization
    hsic_kk = _self_hsic(K_tilde, n)
    hsic_ll = _self_hsic(L_tilde, n)

    # Clamp: debiased HSIC can go negative due to finite samples
    product = hsic_kk * hsic_ll
    if product <= 0:
        return 0.0

    denom = math.sqrt(product)
    if denom < 1e-10:
        return 0.0

    return max(0.0, min(1.0, float(hsic / denom)))


def _self_hsic(K_tilde: Tensor, n: int) -> float:
    """Self-HSIC for debiased normalization."""
    term1 = (K_tilde * K_tilde).sum()
    term2 = K_tilde.sum() ** 2 / ((n - 1) * (n - 2))
    term3 = 2.0 * (K_tilde.sum(dim=0) ** 2).sum() / (n - 2)
    return float((term1 + term2 - term3) / (n * (n - 3)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Procrustes Alignment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ProcrustesAlignment:
    """Result of orthogonal Procrustes alignment."""
    rotation: Tensor  # (d_ref, d_target) rotation matrix
    scale: float
    error: float  # Frobenius norm of residual

    def transform(self, x: Tensor) -> Tensor:
        """Apply alignment transformation."""
        return self.scale * (x @ self.rotation)


def compute_procrustes(
    source: Tensor,
    target: Tensor,
) -> ProcrustesAlignment:
    """
    Compute orthogonal Procrustes alignment.

    Finds the optimal rotation matrix R such that:
        ||target - scale * source @ R||_F is minimized

    This maps activations from one model's representation space
    into another's, preserving relative structure.

    Args:
        source: Activations from reference model, (n, d_source).
        target: Activations from target model, (n, d_target).

    Returns:
        ProcrustesAlignment with rotation matrix and scale.
    """
    # Pad dimensions if needed
    d_source = source.shape[1]
    d_target = target.shape[1]

    if d_source != d_target:
        # Pad the smaller to match the larger
        max_d = max(d_source, d_target)
        if d_source < max_d:
            source = F.pad(source, (0, max_d - d_source))
        if d_target < max_d:
            target = F.pad(target, (0, max_d - d_target))

    # Center both
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)

    # SVD of cross-covariance
    M = source_centered.T @ target_centered  # (d, d)
    U, S, Vt = torch.linalg.svd(M)

    # Optimal rotation (handle reflection)
    det = torch.det(U @ Vt)
    sign_correction = torch.ones(U.shape[1], device=U.device)
    if det < 0:
        sign_correction[-1] = -1.0

    R = U @ torch.diag(sign_correction) @ Vt

    # Optimal scale
    source_norm = (source_centered ** 2).sum()
    if source_norm > 1e-10:
        scale = float((S * sign_correction).sum() / source_norm)
    else:
        scale = 1.0

    # Compute error
    aligned = scale * source_centered @ R
    error = float(torch.norm(target_centered - aligned, p="fro"))

    # Trim rotation to original dimensions
    R_trimmed = R[:d_source, :d_target]

    return ProcrustesAlignment(
        rotation=R_trimmed,
        scale=scale,
        error=error,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer Mapping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class LayerMapping:
    """Mapping between layers of two different model architectures."""
    target_layer: str
    ref_layer: str
    cka_score: float
    alignment: ProcrustesAlignment | None = None


class LayerMapper:
    """
    Maps layers between different model architectures using CKA.

    Runs the same inputs through both models, captures per-layer
    activations, then finds which layers have the most similar
    representations (highest CKA).
    """

    def __init__(self, min_cka: float = 0.5):
        self.min_cka = min_cka
        self._cache: dict[str, dict[str, LayerMapping]] = {}

    def build_mapping(
        self,
        target_model: nn.Module,
        ref_model: nn.Module,
        ref_name: str,
        tokenizer_target: Any,
        tokenizer_ref: Any,
        probe_texts: list[str],
        target_layers: list[str] | None = None,
        ref_layers: list[str] | None = None,
    ) -> dict[str, LayerMapping]:
        """
        Build a layer mapping between target and reference model.

        Args:
            target_model: The model being trained.
            ref_model: A reference model for cross-pollination.
            ref_name: Name for caching.
            tokenizer_target: Tokenizer for target model.
            tokenizer_ref: Tokenizer for reference model.
            probe_texts: Texts to use for activation collection.
            target_layers: Specific layers to map (None = auto-detect).
            ref_layers: Specific ref layers (None = auto-detect).

        Returns:
            Dict mapping target_layer → LayerMapping.
        """
        if ref_name in self._cache:
            return self._cache[ref_name]

        # Auto-detect layers if not specified
        if target_layers is None:
            target_layers = self._detect_layers(target_model)
        if ref_layers is None:
            ref_layers = self._detect_layers(ref_model)

        # Collect activations from both models
        target_acts = self._collect_activations(
            target_model, tokenizer_target, probe_texts, target_layers,
        )
        ref_acts = self._collect_activations(
            ref_model, tokenizer_ref, probe_texts, ref_layers,
        )

        # Compute CKA between all layer pairs
        # Use Rust-accelerated all-pairs when available (rayon parallel)
        mapping: dict[str, LayerMapping] = {}

        # Filter to layers that have activations
        valid_t = [l for l in target_layers if l in target_acts]
        valid_r = [l for l in ref_layers if l in ref_acts]

        if valid_t and valid_r:
            from agisti.accel import fast_cka_all_pairs

            t_act_list = [target_acts[l] for l in valid_t]
            r_act_list = [ref_acts[l] for l in valid_r]
            cka_matrix = fast_cka_all_pairs(t_act_list, r_act_list)

            for i, t_layer in enumerate(valid_t):
                best_j = int(cka_matrix[i].argmax())
                best_cka = float(cka_matrix[i, best_j])

                if best_cka < self.min_cka:
                    continue

                best_ref_layer = valid_r[best_j]

                # Compute Procrustes alignment for this pair
                alignment = compute_procrustes(
                    ref_acts[best_ref_layer],
                    target_acts[t_layer],
                )

                mapping[t_layer] = LayerMapping(
                    target_layer=t_layer,
                    ref_layer=best_ref_layer,
                    cka_score=best_cka,
                    alignment=alignment,
                )

        self._cache[ref_name] = mapping
        logger.info(
            "Built layer mapping for %s: %d/%d layers mapped (min CKA=%.3f)",
            ref_name, len(mapping), len(target_layers), self.min_cka,
        )

        return mapping

    def _detect_layers(self, model: nn.Module) -> list[str]:
        """Auto-detect transformer layers by naming convention."""
        import re
        layers: list[str] = []
        pattern = re.compile(
            r"(layers|blocks|h)\.\d+\.(self_attn|mlp|attention|feed_forward|"
            r"attn|ffn)"
        )
        for name, _ in model.named_modules():
            if pattern.search(name):
                layers.append(name)
        return layers

    def _collect_activations(
        self,
        model: nn.Module,
        tokenizer: Any,
        texts: list[str],
        layers: list[str],
    ) -> dict[str, Tensor]:
        """Collect per-layer activations for probe texts."""
        per_layer_acts: dict[str, list[Tensor]] = {l: [] for l in layers}
        hooks: list[torch.utils.hooks.RemovableHook] = []

        def make_hook(name: str, storage: list[Tensor]):
            def hook_fn(module: nn.Module, input: Any, output: Any) -> None:
                out = output[0] if isinstance(output, tuple) else output
                if out.dim() >= 2:
                    storage.append(out[:, -1, :].detach().cpu())
                else:
                    storage.append(out.detach().cpu())
            return hook_fn

        for name, module in model.named_modules():
            if name in per_layer_acts:
                h = module.register_forward_hook(
                    make_hook(name, per_layer_acts[name]),
                )
                hooks.append(h)

        try:
            device = next(model.parameters()).device
            for text in texts:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        # Stack per-layer activations: (n_texts, dim)
        result: dict[str, Tensor] = {}
        for layer, acts in per_layer_acts.items():
            if acts:
                result[layer] = torch.cat(acts, dim=0)

        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Inter-Model Surgery
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class InterModelSurgery:
    """
    Information Ceiling Breaker Level 3.

    Uses reference models' activation patterns to guide surgery
    on the target model. When the target fails but a reference succeeds:
    - Reference's "correct" activations show an alternative pattern
    - Procrustes-aligned into target's space = "what target is missing"
    - This becomes a cross-pollination surgery signal

    This is NOT distillation (output copying). It extracts directional
    hints from alternate representations, enabling the target to
    potentially exceed the reference's performance in specific areas.
    """

    def __init__(
        self,
        target_model: nn.Module,
        tokenizer_target: Any,
    ):
        self.target_model = target_model
        self.tokenizer_target = tokenizer_target
        self._references: dict[str, tuple[nn.Module, Any]] = {}
        self._layer_mapper = LayerMapper(min_cka=0.5)
        self._mappings: dict[str, dict[str, LayerMapping]] = {}

    def register_reference(
        self,
        name: str,
        model: nn.Module,
        tokenizer: Any,
    ) -> None:
        """Register a reference model for cross-pollination."""
        self._references[name] = (model, tokenizer)

    def build_mappings(
        self,
        probe_texts: list[str],
        target_layers: list[str] | None = None,
    ) -> None:
        """
        Build layer mappings for all registered reference models.

        Should be called once at the start, or periodically refreshed.
        """
        for name, (ref_model, ref_tokenizer) in self._references.items():
            mapping = self._layer_mapper.build_mapping(
                target_model=self.target_model,
                ref_model=ref_model,
                ref_name=name,
                tokenizer_target=self.tokenizer_target,
                tokenizer_ref=ref_tokenizer,
                probe_texts=probe_texts,
                target_layers=target_layers,
            )
            self._mappings[name] = mapping

    def generate_cross_signal(
        self,
        problems: list[Problem],
        trace_layers: list[str],
        max_length: int = 512,
    ) -> CrossSignal:
        """
        Generate cross-pollination surgery signal.

        For each problem:
        1. Target model solves (with tracing)
        2. Each reference model solves (with tracing)
        3. Find: target wrong + reference correct
        4. Align reference activations → target space
        5. Contrast: aligned_ref_correct - target_wrong

        Args:
            problems: Problems to solve across models.
            trace_layers: Target layers to trace.
            max_length: Max generation tokens.

        Returns:
            CrossSignal with aligned activation contrasts.
        """
        if not self._references:
            return CrossSignal(
                usable=False,
                reason="No reference models registered",
            )

        if not self._mappings:
            return CrossSignal(
                usable=False,
                reason="Layer mappings not built. Call build_mappings() first.",
            )

        # Target model solves all problems
        target_correct: set[int] = set()
        target_acts: dict[int, dict[str, Tensor]] = {}

        for i, problem in enumerate(problems):
            answer, acts = self._solve_traced(
                self.target_model,
                self.tokenizer_target,
                problem,
                trace_layers,
                max_length,
            )
            target_acts[i] = acts
            if problem.verify(answer):
                target_correct.add(i)

        # Reference models solve all problems
        ref_correct: dict[str, set[int]] = {}
        ref_acts: dict[str, dict[int, dict[str, Tensor]]] = {}

        for ref_name, (ref_model, ref_tokenizer) in self._references.items():
            ref_correct[ref_name] = set()
            ref_acts[ref_name] = {}

            mapping = self._mappings.get(ref_name, {})
            ref_trace_layers = [
                mapping[l].ref_layer
                for l in trace_layers if l in mapping
            ]

            for i, problem in enumerate(problems):
                answer, acts = self._solve_traced(
                    ref_model,
                    ref_tokenizer,
                    problem,
                    ref_trace_layers,
                    max_length,
                )
                ref_acts[ref_name][i] = acts
                if problem.verify(answer):
                    ref_correct[ref_name].add(i)

        # Find informative problems: target wrong + reference correct
        all_contrasts: dict[str, list[Tensor]] = {}

        for ref_name, mapping in self._mappings.items():
            informative = ref_correct.get(ref_name, set()) - target_correct
            if len(informative) < 3:
                continue

            for t_layer in trace_layers:
                if t_layer not in mapping:
                    continue

                lm = mapping[t_layer]
                r_layer = lm.ref_layer

                # Target activations on failed problems
                t_wrong_acts: list[Tensor] = []
                r_correct_acts: list[Tensor] = []

                for idx in informative:
                    t_act = target_acts.get(idx, {}).get(t_layer)
                    r_act = ref_acts.get(ref_name, {}).get(idx, {}).get(r_layer)

                    if t_act is not None and r_act is not None:
                        t_wrong_acts.append(t_act)

                        # Align reference activation to target space
                        if lm.alignment is not None:
                            r_aligned = lm.alignment.transform(r_act)
                        else:
                            # Fallback: linear projection
                            r_aligned = r_act[:, :t_act.shape[-1]] if r_act.shape[-1] > t_act.shape[-1] else r_act
                            if r_aligned.shape[-1] < t_act.shape[-1]:
                                r_aligned = F.pad(
                                    r_aligned,
                                    (0, t_act.shape[-1] - r_aligned.shape[-1]),
                                )

                        r_correct_acts.append(r_aligned)

                if len(t_wrong_acts) < 2:
                    continue

                # Cross-contrast: "what reference does when correct" -
                # "what target does when wrong"
                t_wrong_mean = torch.stack(t_wrong_acts).mean(0)
                r_correct_mean = torch.stack(r_correct_acts).mean(0)
                contrast = r_correct_mean - t_wrong_mean

                if t_layer not in all_contrasts:
                    all_contrasts[t_layer] = []
                all_contrasts[t_layer].append(contrast)

        # Average contrasts across reference models
        averaged: dict[str, Tensor] = {}
        for layer, signals in all_contrasts.items():
            if signals:
                averaged[layer] = torch.stack(signals).mean(0)

        informative_counts = {
            ref: len(ref_correct.get(ref, set()) - target_correct)
            for ref in self._references
        }

        return CrossSignal(
            usable=len(averaged) > 0,
            contrasts=averaged if averaged else None,
            informative_problems=informative_counts,
        )

    def _solve_traced(
        self,
        model: nn.Module,
        tokenizer: Any,
        problem: Problem,
        trace_layers: list[str],
        max_length: int,
    ) -> tuple[str, dict[str, Tensor]]:
        """Solve a problem with activation tracing."""
        activations: dict[str, Tensor] = {}
        hooks: list[torch.utils.hooks.RemovableHook] = []

        def make_hook(name: str):
            def hook_fn(module: nn.Module, input: Any, output: Any) -> None:
                out = output[0] if isinstance(output, tuple) else output
                if out.dim() >= 2:
                    activations[name] = out[:, -1, :].detach().clone()
                else:
                    activations[name] = out.detach().clone()
            return hook_fn

        for name, module in model.named_modules():
            if name in trace_layers:
                hooks.append(module.register_forward_hook(make_hook(name)))

        try:
            prompt = f"Question: {problem.question}\nAnswer:"
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=False,
                )

            answer = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

        finally:
            for h in hooks:
                h.remove()

        return answer, activations

    def get_cross_report(self) -> dict[str, Any]:
        """Get a report on cross-model relationships."""
        report: dict[str, Any] = {
            "reference_models": list(self._references.keys()),
            "mappings": {},
        }

        for ref_name, mapping in self._mappings.items():
            layers_info: list[dict[str, Any]] = []
            for t_layer, lm in mapping.items():
                layers_info.append({
                    "target_layer": lm.target_layer,
                    "ref_layer": lm.ref_layer,
                    "cka_score": lm.cka_score,
                    "alignment_error": (
                        lm.alignment.error if lm.alignment else None
                    ),
                })
            report["mappings"][ref_name] = {
                "mapped_layers": len(layers_info),
                "avg_cka": (
                    sum(l["cka_score"] for l in layers_info) / len(layers_info)
                    if layers_info else 0.0
                ),
                "layers": layers_info,
            }

        return report
