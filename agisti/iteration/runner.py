"""
Iteration Runner — the core AGISTI loop (§4.1, 10-step pipeline).

Each iteration of self-improvement follows a rigorous pipeline:
1. Active Probe → measure competency
2. Generate problems from weaknesses
3. Model solves problems
4. Verify & evaluate (mechanical verification only)
5. Compute activation contrasts (correct vs wrong)
6. Propose surgery (SVD-based LoRA delta)
7. Virtual training (simulate before applying)
8. Apply delta (with frozen zone guard)
9. Quick bench (with McNemar significance test)
10. Accept/reject + feedback

Design: §4.1 — One Iteration of Self-Improvement.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from agisti.types import (
    AnswerType,
    IterationResult,
    IterationState,
    LoRADelta,
    Problem,
    Solution,
    SurgeryType,
    VERIFIABLE_TYPES,
)
from agisti.config import IterationConfig, MetaStrategy
from agisti.generation.difficulty import AdaptiveDifficultyEngine
from agisti.iteration.state_machine import IterationStateMachine, StateTransition

logger = logging.getLogger(__name__)


@dataclass
class IterationContext:
    """
    Everything needed for an iteration, passed between steps.

    This is the scratch pad that accumulates state as the
    iteration progresses through the 10-step pipeline.
    """
    epoch: int
    iteration: int
    model: nn.Module
    tokenizer: Any
    config: IterationConfig
    strategy: MetaStrategy

    # Accumulated during iteration
    probe_scores: dict[str, float] = field(default_factory=dict)
    problems: list[Problem] = field(default_factory=list)
    solutions: list[Solution] = field(default_factory=list)
    correct_indices: list[int] = field(default_factory=list)
    wrong_indices: list[int] = field(default_factory=list)
    proposed_delta: LoRADelta | None = None
    refined_delta: LoRADelta | None = None
    virtual_loss_before: float = 0.0
    virtual_loss_after: float = 0.0
    quick_bench_score: float = 0.0
    quick_bench_scores: dict[str, float] = field(default_factory=dict)
    accepted: bool = False
    rejection_reason: str | None = None
    start_time: float = field(default_factory=time.time)
    pre_surgery_state: dict[str, Any] | None = None


class IterationRunner:
    """
    Runs a single iteration of the AGISTI self-improvement loop.

    Orchestrates the 10-step pipeline using components injected
    at construction time. Each step transitions the state machine
    and updates the IterationContext.

    The runner delegates to specialized components:
    - ActiveProber: step 1 (probe)
    - ProblemGenerator: step 2 (generate)
    - ModelEvaluator: step 3-4 (solve + verify)
    - SurgeryProposer: step 5-6 (contrast + propose)
    - VirtualTrainer: step 7 (simulate)
    - DeltaApplicator: step 8 (apply)
    - QuickBench: step 9 (benchmark)
    - Feedback: step 10 (accept/reject)
    """

    def __init__(
        self,
        prober: Any = None,
        generator: Any = None,
        evaluator: Any = None,
        verifier: Any = None,
        proposer: Any = None,
        virtual_trainer: Any = None,
        applicator: Any = None,
        quick_bench: Any = None,
        frozen_mask: Any = None,
        tracer: Any = None,
    ):
        self.prober = prober
        self.generator = generator
        self.evaluator = evaluator
        self.verifier = verifier
        self.proposer = proposer
        self.virtual_trainer = virtual_trainer
        self.applicator = applicator
        self.quick_bench = quick_bench
        self.frozen_mask = frozen_mask
        self.tracer = tracer

        self._state_machine = IterationStateMachine()
        self._difficulty_engine = AdaptiveDifficultyEngine(
            target_accuracy_low=0.2,
            target_accuracy_high=0.7,
        )

    def run(self, ctx: IterationContext) -> IterationResult:
        """
        Execute a complete iteration.

        Args:
            ctx: IterationContext with model, config, strategy.

        Returns:
            IterationResult with all metrics.
        """
        logger.info(
            "=== Iteration %d (epoch %d) ===",
            ctx.iteration, ctx.epoch,
        )

        self._state_machine.reset()

        try:
            # Step 1: Active Probe
            self._step_probe(ctx)

            # Step 2: Problem Generation
            self._step_generate(ctx)

            # Step 3-4: Solve & Verify
            self._step_solve_and_evaluate(ctx)
            torch.cuda.empty_cache()

            # Step 5-6: Activation Contrast & Propose Surgery
            self._step_propose_surgery(ctx)
            # Move delta to each layer's device (for multi-GPU device_map)
            if ctx.proposed_delta is not None:
                named_params = dict(ctx.model.named_parameters())
                named_modules = dict(ctx.model.named_modules())
                fallback_device = next(ctx.model.parameters()).device
                for name, ld in ctx.proposed_delta.layers.items():
                    target_device = fallback_device
                    if name in named_params:
                        target_device = named_params[name].device
                    elif name + ".weight" in named_params:
                        target_device = named_params[name + ".weight"].device
                    elif name in named_modules and hasattr(named_modules[name], 'weight') and named_modules[name].weight is not None:
                        target_device = named_modules[name].weight.device
                    ld.A = ld.A.to(target_device)
                    ld.B = ld.B.to(target_device)
            torch.cuda.empty_cache()

            # Step 7: Virtual Training
            self._step_virtual_train(ctx)
            import gc; gc.collect()
            torch.cuda.empty_cache()

            # Step 8: Apply Delta
            self._step_apply_delta(ctx)

            # Step 9: Quick Bench
            self._step_quick_bench(ctx)

            # Step 10: Accept/Reject
            self._step_accept_reject(ctx)

            # Aggressive memory cleanup after iteration
            ctx.pre_surgery_state = None
            ctx.solutions = []
            import gc; gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error("Iteration %d failed: %s", ctx.iteration, e)
            # Rollback if we applied a delta
            if ctx.pre_surgery_state is not None:
                logger.warning("Rolling back delta due to error")
                named_params = dict(ctx.model.named_parameters())
                with torch.no_grad():
                    for k, v in ctx.pre_surgery_state.items():
                        if k in named_params:
                            named_params[k].copy_(v.to(named_params[k].device))
            raise

        wall_time = time.time() - ctx.start_time

        result = IterationResult(
            iteration_id=ctx.iteration,
            proposed_delta_norm=(
                ctx.proposed_delta.norm() if ctx.proposed_delta else 0.0
            ),
            virtual_loss_before=ctx.virtual_loss_before,
            virtual_loss_after=ctx.virtual_loss_after,
            refined_delta_norm=(
                ctx.refined_delta.norm() if ctx.refined_delta else 0.0
            ),
            quick_bench_scores=ctx.quick_bench_scores,
            accepted=ctx.accepted,
            rejection_reason=ctx.rejection_reason,
            wall_time_seconds=wall_time,
            gpu_memory_peak_gb=self._get_gpu_memory_peak(),
            target_layers=list(ctx.strategy.target_layers),
            surgery_type=ctx.strategy.surgery_type,
            epoch=ctx.epoch,
            delta=ctx.refined_delta,
            solutions=ctx.solutions,
            loss=ctx.virtual_loss_after,
            quick_bench=getattr(ctx, 'quick_bench_result', None),
        )

        logger.info(
            "Iteration %d complete: accepted=%s, "
            "delta_norm=%.4f, bench=%.4f, time=%.1fs",
            ctx.iteration, ctx.accepted,
            result.refined_delta_norm,
            ctx.quick_bench_score,
            wall_time,
        )

        return result

    def _step_probe(self, ctx: IterationContext) -> None:
        """Step 1: Active probe to measure current competency."""
        self._state_machine.transition(IterationState.PROBE)

        if self.prober is None:
            logger.debug("No prober configured, skipping probe step")
            return

        probe_result = self.prober.probe_all_domains(
            model=ctx.model,
            tokenizer=ctx.tokenizer,
            seed=ctx.epoch,
        )

        # Handle various prober return types
        if isinstance(probe_result, dict):
            ctx.probe_scores = probe_result
        elif isinstance(probe_result, tuple):
            # (CompetencyVector, list[FailedProblem])
            competency_vec, _ = probe_result
            ctx.probe_scores = competency_vec.as_dict()
        else:
            # Assume iterable of probe results with domain/accuracy
            ctx.probe_scores = {
                p.domain: p.accuracy for p in probe_result
            }

        logger.info(
            "Probe: %d domains, avg=%.3f",
            len(ctx.probe_scores),
            (
                sum(ctx.probe_scores.values()) / len(ctx.probe_scores)
                if ctx.probe_scores else 0.0
            ),
        )

    def _step_generate(self, ctx: IterationContext) -> None:
        """Step 2: Generate problems targeting weaknesses with adaptive difficulty."""
        self._state_machine.transition(IterationState.GENERATE)

        if self.generator is None:
            logger.debug("No generator configured, skipping generate step")
            return

        from agisti.generation.generator import GenerationRequest
        focus = ctx.strategy.focus_domains if hasattr(ctx.strategy, 'focus_domains') else ["math"]
        primary_domain = focus[0] if focus else "math"

        # Use adaptive difficulty engine for each domain
        difficulty = self._difficulty_engine.get_difficulty(primary_domain)
        logger.info(
            "Generating problems: domain=%s, adaptive_difficulty=%.2f",
            primary_domain, difficulty,
        )

        request = GenerationRequest(
            domain=primary_domain,
            count=ctx.config.problems_per_iteration,
            difficulty=difficulty,
            focus_areas=focus,
        )
        ctx.problems = self.generator.generate(request)

        # Filter to verifiable types (including EXACT_MATCH for MCQ)
        original_count = len(ctx.problems)
        allowed_types = VERIFIABLE_TYPES | {AnswerType.EXACT_MATCH}
        ctx.problems = [
            p for p in ctx.problems
            if p.answer_type in allowed_types
        ]

        logger.info(
            "Generated %d problems (%d verifiable) at difficulty=%.2f",
            original_count, len(ctx.problems), difficulty,
        )

    def _step_solve_and_evaluate(self, ctx: IterationContext) -> None:
        """Step 3-4: Model solves problems, verify answers."""
        self._state_machine.transition(IterationState.SOLVE)

        if not ctx.problems:
            logger.warning("No problems to solve")
            self._state_machine.transition(IterationState.EVALUATE)
            return

        # Solve
        if self.evaluator is not None:
            eval_result = self.evaluator.evaluate(
                model=ctx.model,
                tokenizer=ctx.tokenizer,
                problems=ctx.problems,
            )
            ctx.solutions = eval_result.solutions
        else:
            ctx.solutions = []

        # Verify & classify
        self._state_machine.transition(IterationState.EVALUATE)

        if self.verifier is not None:
            for i, (prob, sol) in enumerate(
                zip(ctx.problems, ctx.solutions)
            ):
                verification = self.verifier.verify(
                    problem=prob,
                    model_answer=sol.answer,
                )
                if verification.correct:
                    ctx.correct_indices.append(i)
                else:
                    ctx.wrong_indices.append(i)
        else:
            # Fallback: use problem.verify
            for i, (prob, sol) in enumerate(
                zip(ctx.problems, ctx.solutions)
            ):
                if prob.verify(sol.answer):
                    ctx.correct_indices.append(i)
                else:
                    ctx.wrong_indices.append(i)

        accuracy = len(ctx.correct_indices) / max(len(ctx.problems), 1)
        logger.info(
            "Evaluation: %d correct, %d wrong (%.1f%% accuracy)",
            len(ctx.correct_indices),
            len(ctx.wrong_indices),
            accuracy * 100,
        )

        # Update adaptive difficulty engine
        focus = ctx.strategy.focus_domains if hasattr(ctx.strategy, 'focus_domains') else ["math"]
        primary_domain = focus[0] if focus else "math"
        new_diff = self._difficulty_engine.update(primary_domain, accuracy)
        logger.info(
            "Adaptive difficulty: domain=%s, accuracy=%.1f%%, next_difficulty=%.2f",
            primary_domain, accuracy * 100, new_diff,
        )

        # If 0% or 100% accuracy, retry generation at adjusted difficulty
        if (len(ctx.correct_indices) == 0 or len(ctx.wrong_indices) == 0) and self.generator is not None:
            max_retries = 2
            for retry in range(max_retries):
                retry_diff = self._difficulty_engine.get_difficulty(primary_domain)
                if len(ctx.correct_indices) == 0:
                    # Too hard → force lower
                    retry_diff = max(0.05, retry_diff - 0.15)
                else:
                    # Too easy → force higher
                    retry_diff = min(0.95, retry_diff + 0.15)
                self._difficulty_engine._difficulty[primary_domain] = retry_diff

                logger.info(
                    "Retry %d: regenerating at difficulty=%.2f (was %s correct)",
                    retry + 1, retry_diff,
                    "0%" if not ctx.correct_indices else "100%",
                )
                from agisti.generation.generator import GenerationRequest
                retry_req = GenerationRequest(
                    domain=primary_domain,
                    count=ctx.config.problems_per_iteration,
                    difficulty=retry_diff,
                    focus_areas=focus,
                )
                retry_problems = self.generator.generate(retry_req)
                allowed_types = VERIFIABLE_TYPES | {AnswerType.EXACT_MATCH}
                retry_problems = [p for p in retry_problems if p.answer_type in allowed_types]
                if not retry_problems:
                    continue

                # Re-solve retry problems
                ctx.problems = retry_problems
                ctx.solutions = []
                ctx.correct_indices = []
                ctx.wrong_indices = []
                if self.evaluator is not None:
                    eval_result = self.evaluator.evaluate(
                        model=ctx.model, tokenizer=ctx.tokenizer,
                        problems=ctx.problems,
                    )
                    ctx.solutions = eval_result.solutions
                if self.verifier is not None:
                    for i, (prob, sol) in enumerate(zip(ctx.problems, ctx.solutions)):
                        verification = self.verifier.verify(problem=prob, model_answer=sol.answer)
                        if verification.correct:
                            ctx.correct_indices.append(i)
                        else:
                            ctx.wrong_indices.append(i)

                retry_acc = len(ctx.correct_indices) / max(len(ctx.problems), 1)
                self._difficulty_engine.update(primary_domain, retry_acc)
                logger.info(
                    "Retry %d result: %d correct, %d wrong (%.1f%%)",
                    retry + 1, len(ctx.correct_indices),
                    len(ctx.wrong_indices), retry_acc * 100,
                )
                if ctx.correct_indices and ctx.wrong_indices:
                    break  # Got mixed results - perfect for contrast!

    def _step_propose_surgery(self, ctx: IterationContext) -> None:
        """Step 5-6: Compute activation contrasts, propose surgery."""
        self._state_machine.transition(IterationState.PROPOSE)

        if self.proposer is None or not ctx.correct_indices or not ctx.wrong_indices:
            logger.debug("Skipping surgery proposal (insufficient data)")
            return

        # Free VRAM before activation collection
        torch.cuda.empty_cache()

        # Build scores from eval results
        scores: list[bool] = [
            i in ctx.correct_indices for i in range(len(ctx.problems))
        ]

        # Auto-detect target layers if strategy doesn't specify any
        target_layers = list(ctx.strategy.target_layers)
        if not target_layers:
            target_layers = self._detect_target_layers(ctx.model)
            logger.info("Auto-detected %d target layers", len(target_layers))

        # Collect lightweight activation maps via hooks (on CPU to save VRAM)
        activation_maps: list[dict[str, Any]] = []
        if self.tracer is not None:
            for i, prob in enumerate(ctx.problems):
                with self.tracer.trace():
                    pass
                activation_maps.append(self.tracer.get_activations())
                self.tracer.clear()
        else:
            # Lightweight activation collection: hook target layers, run
            # forward pass per problem, collect activations on CPU
            activation_maps = self._collect_activations_lightweight(
                ctx.model, ctx.tokenizer, ctx.problems, target_layers,
            )

        # Get frozen layer names
        frozen_names: set[str] = set()
        if self.frozen_mask is not None:
            frozen_names = getattr(
                self.frozen_mask, 'frozen_names',
                getattr(self.frozen_mask, '_frozen_names', set()),
            )

        result = self.proposer.propose(
            activation_maps=activation_maps,
            scores=scores,
            target_layers=target_layers,
            frozen_layer_names=frozen_names,
        )

        # propose() returns (LoRADelta, SelfSignal)
        if isinstance(result, tuple):
            ctx.proposed_delta, _ = result
        else:
            ctx.proposed_delta = result

        if ctx.proposed_delta:
            logger.info(
                "Proposed delta: norm=%.4f, layers=%d",
                ctx.proposed_delta.norm(),
                len(ctx.proposed_delta),
            )

    @staticmethod
    def _detect_target_layers(model: nn.Module, max_layers: int = 6) -> list[str]:
        """Auto-detect MLP/attention projection layers as surgery targets."""
        candidates = []
        for name, mod in model.named_modules():
            # Target attention output projections and MLP layers
            if isinstance(mod, nn.Linear):
                if any(k in name for k in ('o_proj', 'down_proj', 'gate_proj', 'up_proj')):
                    candidates.append(name)
        # Spread across the model: pick evenly spaced layers
        if len(candidates) > max_layers:
            step = len(candidates) // max_layers
            candidates = candidates[::step][:max_layers]
        return candidates

    def _collect_activations_lightweight(
        self,
        model: nn.Module,
        tokenizer: Any,
        problems: list[Problem],
        target_layers: list[str],
    ) -> list[dict[str, Any]]:
        """Collect activations for surgery without a dedicated tracer.

        Uses temporary forward hooks, immediately moves activations
        to CPU to avoid VRAM pressure.
        """
        # Resolve target layer modules
        layer_modules: dict[str, nn.Module] = {}
        for name, mod in model.named_modules():
            if name in target_layers:
                layer_modules[name] = mod

        if not layer_modules:
            logger.warning("No target layer modules found in model")
            return [{} for _ in problems]

        activation_maps: list[dict[str, Any]] = []
        device = next(model.parameters()).device

        for prob in problems:
            collected: dict[str, Any] = {}
            hooks = []

            def make_hook(layer_name: str):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    # Mean over sequence dim, move to CPU
                    collected[layer_name] = output.detach().mean(dim=1).cpu()
                return hook_fn

            # Install hooks
            for name, mod in layer_modules.items():
                h = mod.register_forward_hook(make_hook(name))
                hooks.append(h)

            try:
                from agisti.generation.prompt_utils import format_for_model
                prompt = f"Question: {prob.question}\nAnswer:"
                inputs = format_for_model(prompt, tokenizer)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

            except Exception as e:
                logger.debug("Activation collection failed for problem: %s", e)
            finally:
                for h in hooks:
                    h.remove()

            activation_maps.append(collected)
            del input_ids, attention_mask
            torch.cuda.empty_cache()

        return activation_maps

    def _step_virtual_train(self, ctx: IterationContext) -> None:
        """Step 7: Virtual training — simulate surgery before applying."""
        self._state_machine.transition(IterationState.VIRTUAL_TRAIN)

        # GPU 캐시 정리 (72B OOM 방지)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.virtual_trainer is None or ctx.proposed_delta is None:
            ctx.refined_delta = ctx.proposed_delta
            return

        # Get frozen layer names for safety check
        frozen_names: set[str] = set()
        if self.frozen_mask is not None:
            frozen_names = getattr(
                self.frozen_mask, 'frozen_names',
                getattr(self.frozen_mask, '_frozen_names', set()),
            )

        vt_problems = ctx.problems[:min(
            len(ctx.problems),
            max(ctx.config.virtual_train_steps, 3),  # 72B: 최소 3개로 제한
        )]

        vt_result = self.virtual_trainer.run(
            model=ctx.model,
            proposed_delta=ctx.proposed_delta,
            validation_problems=vt_problems,
            frozen_mask_names=frozen_names,
        )

        ctx.virtual_loss_before = vt_result.loss_before
        ctx.virtual_loss_after = vt_result.loss_after
        ctx.refined_delta = vt_result.refined_delta or ctx.proposed_delta

        if not vt_result.loss_decreased:
            logger.warning(
                "Virtual training: loss INCREASED (%.4f → %.4f)",
                vt_result.loss_before, vt_result.loss_after,
            )

        logger.info(
            "Virtual training: %.4f → %.4f (%s), refined norm=%.4f",
            vt_result.loss_before,
            vt_result.loss_after,
            "↓" if vt_result.loss_decreased else "↑",
            ctx.refined_delta.norm() if ctx.refined_delta else 0.0,
        )

    def _step_apply_delta(self, ctx: IterationContext) -> None:
        """Step 8: Apply delta to model weights."""
        self._state_machine.transition(IterationState.APPLY_DELTA)

        if ctx.refined_delta is None:
            logger.debug("No delta to apply")
            return

        # GPU 캐시 정리 후 스냅샷 (72B OOM 방지)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Snapshot for rollback (only modified layers, on CPU to avoid OOM)
        self._state_machine.transition(IterationState.SNAPSHOT)
        modified_layers = set(ctx.refined_delta.layer_names) if ctx.refined_delta else set()
        ctx.pre_surgery_state = {}
        for k, v in ctx.model.named_parameters():
            if any(ln in k for ln in modified_layers):
                ctx.pre_surgery_state[k] = v.detach().cpu().clone()

        self._state_machine.transition(IterationState.APPLY_DELTA)

        if self.applicator is not None:
            self.applicator.apply(
                delta=ctx.refined_delta,
            )
        else:
            # Fallback: manual apply
            self._manual_apply_delta(ctx.model, ctx.refined_delta)

        logger.info("Delta applied to model")

        # Delta 적용 후 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _step_quick_bench(self, ctx: IterationContext) -> None:
        """Step 9: Quick benchmark with McNemar test."""
        self._state_machine.transition(IterationState.QUICK_BENCH)

        if self.quick_bench is None:
            logger.debug("No quick bench configured")
            return

        # GPU 캐시 정리 후 벤치 시작 (72B OOM 방지)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        bench_result = self.quick_bench.run(
            model=ctx.model,
            tokenizer=ctx.tokenizer,
            epoch=ctx.epoch,
        )

        ctx.quick_bench_score = bench_result.accuracy
        ctx.quick_bench_scores = bench_result.scores
        ctx.quick_bench_result = bench_result

        # 벤치 끝나면 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            "Quick bench: %.4f (passed=%s)",
            bench_result.accuracy,
            bench_result.passed,
        )

    def _step_accept_reject(self, ctx: IterationContext) -> None:
        """Step 10: Accept or reject the surgery."""
        self._state_machine.transition(IterationState.FEEDBACK)

        if ctx.refined_delta is None:
            ctx.accepted = False
            ctx.rejection_reason = "no_delta_proposed"
            return

        # Accept if quick bench passed and no regressions
        if ctx.quick_bench_score > 0:
            has_regressions = (
                hasattr(ctx, 'quick_bench_result')
                and ctx.quick_bench_result is not None
                and bool(ctx.quick_bench_result.regressions)
            )
            if has_regressions:
                ctx.accepted = False
                ctx.rejection_reason = "domain_regression_detected"
            else:
                ctx.accepted = True
        else:
            ctx.accepted = False
            ctx.rejection_reason = "quick_bench_failed"

        if not ctx.accepted and ctx.pre_surgery_state is not None:
            # Rollback — directly restore parameters without full state_dict copy
            self._state_machine.transition(IterationState.ROLLBACK)
            named_params = dict(ctx.model.named_parameters())
            with torch.no_grad():
                for k, v in ctx.pre_surgery_state.items():
                    if k in named_params:
                        named_params[k].copy_(v.to(named_params[k].device))
            logger.info("Surgery rejected, rolled back")
        elif ctx.accepted:
            logger.info("Surgery accepted ✓")

    @staticmethod
    def _manual_apply_delta(
        model: nn.Module, delta: LoRADelta,
    ) -> None:
        """Fallback: manually apply delta to model parameters."""
        state = model.state_dict()
        for layer_name, layer_delta in delta.items():
            # Find matching parameter
            for param_name, param in state.items():
                if layer_name in param_name and "weight" in param_name:
                    full_delta = layer_delta.to_full().to(param.device)
                    if full_delta.shape == param.shape:
                        state[param_name] = param + full_delta
                    break

        model.load_state_dict(state)

    @staticmethod
    def _get_gpu_memory_peak() -> float:
        """Get peak GPU memory usage in GB (summed across all GPUs)."""
        if torch.cuda.is_available():
            return sum(
                torch.cuda.max_memory_allocated(i)
                for i in range(torch.cuda.device_count())
            ) / (1024 ** 3)
        return 0.0
