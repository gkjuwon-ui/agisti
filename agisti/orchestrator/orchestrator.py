"""
AGISTI Orchestrator — the main training loop that ties all modules together.

Manages the full lifecycle:
1. Model & tokenizer loading
2. Frozen zone discovery
3. Component initialization (surgery, probe, generation, evaluation, benchmark)
4. Phase management (PHASE_0 → PHASE_3)
5. Epoch/iteration loop with MetaStrategy feedback
6. Checkpoint management & branching
7. Ceiling breaker activation
8. Catastrophe detection & emergency rollback

Design: §10 — Execution Phases, §4.1 — Iteration Loop, §8 — MetaStrategy.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from agisti.types import (
    Alert,
    AlertLevel,
    AlertType,
    ConvergenceAction,
    IterationResult,
    IterationState,
    LoRADelta,
    PhaseId,
)
from agisti.config import (
    CatastropheConfig,
    CheckpointConfig,
    ConvergenceConfig,
    FullBenchConfig,
    FrozenDiscoveryConfig,
    IterationConfig,
    MetaStrategy,
    PhaseConfig,
    QuickBenchConfig,
    PHASE_0_CONFIG,
    PHASE_1_CONFIG,
)
from agisti.iteration.runner import IterationContext, IterationRunner
from agisti.iteration.history import IterationHistory
from agisti.iteration.state_machine import IterationBatchTracker
from agisti.feedback.meta_strategy import MetaStrategyEngine
from agisti.feedback.catastrophe import (
    TrainingHealthMonitor,
    CatastropheDetector,
)
from agisti.checkpoint.manager import CheckpointManager
from agisti.checkpoint.branch import BranchManager
from agisti.checkpoint.gc import CheckpointGC, GCPolicy

logger = logging.getLogger(__name__)


@dataclass
class PhaseState:
    """
    Tracks state for the current phase.
    """
    phase_id: PhaseId
    config: PhaseConfig
    epoch: int = 0
    iteration: int = 0
    best_score: float = 0.0
    best_checkpoint: str | None = None
    start_time: float = field(default_factory=time.time)
    completed: bool = False
    abort_reason: str | None = None


@dataclass
class OrchestratorStats:
    """Aggregate statistics across all phases."""
    total_iterations: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    total_wall_time_s: float = 0.0
    phases_completed: int = 0
    emergency_rollbacks: int = 0
    ceiling_breaks: int = 0
    branches_explored: int = 0


class AGISTIOrchestrator:
    """
    Main orchestrator for the AGISTI self-improvement system.

    Manages the full training pipeline across phases.
    Each phase uses a different model scale and surgery strategy.

    Usage:
        orch = AGISTIOrchestrator(
            model=model,
            tokenizer=tokenizer,
            output_dir=Path("./agisti_output"),
            phase_config=PHASE_0_CONFIG,
        )
        orch.run()

    The orchestrator:
    - Runs epochs (each containing multiple iterations)
    - Uses MetaStrategy to adapt surgery parameters
    - Monitors for catastrophes and triggers rollbacks
    - Manages checkpoints and branching
    - Activates ceiling breakers when plateauing
    - Logs detailed progress and diagnostics
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        output_dir: Path,
        phase_config: PhaseConfig | None = None,
        iteration_config: IterationConfig | None = None,
        checkpoint_config: CheckpointConfig | None = None,
        convergence_config: ConvergenceConfig | None = None,
        catastrophe_config: CatastropheConfig | None = None,
        quick_bench_config: QuickBenchConfig | None = None,
        frozen_config: FrozenDiscoveryConfig | None = None,
        initial_strategy: MetaStrategy | None = None,
        # Pluggable components (optional)
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
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configs
        self.phase_config = phase_config or PHASE_0_CONFIG
        self.iter_config = iteration_config or IterationConfig()
        ckpt_cfg = checkpoint_config or CheckpointConfig()
        self.convergence_config = convergence_config or ConvergenceConfig()
        self.catastrophe_config = catastrophe_config or CatastropheConfig()
        self.frozen_config = frozen_config or FrozenDiscoveryConfig()
        self.qb_config = quick_bench_config or QuickBenchConfig()

        # Strategy
        self.strategy = initial_strategy or MetaStrategy()

        # Device
        self.device = device or (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Core components
        self._frozen_mask = frozen_mask
        self._runner = IterationRunner(
            prober=prober,
            generator=generator,
            evaluator=evaluator,
            verifier=verifier,
            proposer=proposer,
            virtual_trainer=virtual_trainer,
            applicator=applicator,
            quick_bench=quick_bench,
            frozen_mask=frozen_mask,
            tracer=tracer,
        )

        # History & feedback
        self.history = IterationHistory(
            window_size=self.convergence_config.window_size,
        )
        self._meta_engine = MetaStrategyEngine(
            convergence_config=self.convergence_config,
        )
        self._health_monitor = TrainingHealthMonitor(
            config=self.catastrophe_config,
        )
        self._catastrophe = CatastropheDetector(
            config=self.catastrophe_config,
        )

        # Checkpoint management
        self._ckpt_manager = CheckpointManager(
            base_dir=self.output_dir / "checkpoints",
            config=ckpt_cfg,
        )
        self._branch_manager = BranchManager(
            base_dir=self.output_dir / "branches",
        )
        self._gc = CheckpointGC(
            checkpoints_dir=self.output_dir / "checkpoints",
            policy=GCPolicy(
                max_checkpoints_per_branch=ckpt_cfg.keep_last_n,
                max_total_checkpoints=ckpt_cfg.keep_last_n * ckpt_cfg.max_branches,
            ),
        )

        # State tracking
        self._batch_tracker = IterationBatchTracker()
        self._stats = OrchestratorStats()
        self._phase_state: PhaseState | None = None
        self._alerts: list[Alert] = []
        self._best_model_state: dict[str, Any] | None = None
        self._best_score: float = 0.0

    # ─── Main Run Loop ────────────────────────────────

    def run(
        self,
        max_epochs: int | None = None,
        max_iterations: int | None = None,
        resume_from: str | None = None,
    ) -> OrchestratorStats:
        """
        Execute the full training pipeline.

        Args:
            max_epochs: Override max epochs (defaults to phase config).
            max_iterations: Hard cap on total iterations.
            resume_from: Checkpoint path to resume from.

        Returns:
            OrchestratorStats with aggregate metrics.
        """
        logger.info(
            "=" * 60 + "\n"
            "AGISTI Orchestrator starting\n"
            "Phase: %s\n"
            "Model: %s\n"
            "Output: %s\n"
            "Device: %s\n"
            + "=" * 60,
            self.phase_config.phase_id.value,
            self.phase_config.model_name,
            self.output_dir,
            self.device,
        )

        if resume_from:
            self._resume(resume_from)

        epochs = max_epochs or self.phase_config.max_epochs
        total_start = time.time()

        self._phase_state = PhaseState(
            phase_id=self.phase_config.phase_id,
            config=self.phase_config,
        )

        try:
            for epoch in range(epochs):
                self._phase_state.epoch = epoch

                should_stop = self._run_epoch(
                    epoch=epoch,
                    max_total_iterations=max_iterations,
                )

                if should_stop:
                    logger.info("Training stopped at epoch %d", epoch)
                    break

                # Checkpoint at epoch boundary
                self._checkpoint_epoch(epoch)

                # GC old checkpoints
                all_ckpts = self._ckpt_manager.list_checkpoints()
                best = self._ckpt_manager.get_best()
                best_path = best.path if best else None
                self._gc.collect(
                    all_checkpoints=all_ckpts,
                    best_path=best_path,
                )

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
        except Exception:
            logger.exception("Training failed with error")
            raise
        finally:
            self._stats.total_wall_time_s = time.time() - total_start
            self._phase_state.completed = True

            # Final checkpoint
            self._checkpoint_epoch(
                self._phase_state.epoch,
                label="final",
            )

        # Print final report
        logger.info("\n%s", self.history.format_report())

        return self._stats

    def _run_epoch(
        self,
        epoch: int,
        max_total_iterations: int | None = None,
    ) -> bool:
        """
        Run a single epoch (multiple iterations).

        Returns:
            True if training should stop.
        """
        iters_per_epoch = self.phase_config.iterations_per_epoch
        logger.info(
            "--- Epoch %d (%d iterations) ---",
            epoch, iters_per_epoch,
        )

        for i in range(iters_per_epoch):
            # Check total iteration limit
            if (
                max_total_iterations
                and self._stats.total_iterations >= max_total_iterations
            ):
                logger.info("Reached max total iterations")
                return True

            iteration_num = (
                epoch * iters_per_epoch + i
            )

            if self._phase_state is not None:
                self._phase_state.iteration = iteration_num

            # Run one iteration
            should_stop = self._run_single_iteration(
                epoch=epoch,
                iteration=iteration_num,
            )

            if should_stop:
                return True

        return False

    def _run_single_iteration(
        self,
        epoch: int,
        iteration: int,
    ) -> bool:
        """
        Execute a single iteration with full error handling.

        Returns:
            True if training should stop.
        """
        # GPU 메모리 정리 (72B OOM 방지)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        ctx = IterationContext(
            epoch=epoch,
            iteration=iteration,
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.iter_config,
            strategy=self.strategy,
        )

        try:
            result = self._runner.run(ctx)
        except Exception as e:
            logger.error("Iteration %d error: %s", iteration, e)
            self._batch_tracker.record_error()
            self._stats.total_iterations += 1

            # Check if too many errors
            if self._batch_tracker.error_rate > 0.5:
                logger.critical(
                    "Error rate %.1f%% — aborting",
                    self._batch_tracker.error_rate * 100,
                )
                return True
            return False

        # Record result
        self.history.add(result)
        self._stats.total_iterations += 1
        if result.accepted:
            self._stats.total_accepted += 1
        else:
            self._stats.total_rejected += 1

        # Track best
        score = self._score_from_result(result)
        if score > self._best_score:
            self._best_score = score
            # Skip full model snapshot for 72B — too expensive (~135GB)
            # Only save the modified layer names for reference
            if result.delta is not None:
                self._best_model_layers = list(result.delta.layer_names)
            if self._phase_state:
                self._phase_state.best_score = score

        # Catastrophe detection
        alerts = self._catastrophe.check(result)
        self._alerts.extend(alerts)
        for alert in alerts:
            if alert.level == AlertLevel.EMERGENCY:
                return self._handle_emergency(alert)

        # Health monitoring
        health_alerts = self._health_monitor.check(result)
        self._alerts.extend(health_alerts)

        # MetaStrategy feedback
        self._meta_engine.update(
            result=result,
            alerts=alerts + health_alerts,
        )

        # Check convergence
        action = self._meta_engine.check_convergence()

        if action == ConvergenceAction.STOP:
            logger.info("MetaStrategy: converged → stop")
            return True

        if action == ConvergenceAction.ROLLBACK:
            self._emergency_rollback("MetaStrategy requested rollback")
            return False

        # Strategy is updated internally by meta_engine.update()
        self.strategy = self._meta_engine.current_strategy

        # ── Per-iteration checkpoint (crash-safe) ──
        # Always save: lightweight log + delta first, then full model
        self._save_iteration_log(iteration, result, score)
        if result.accepted and result.delta is not None:
            self._save_delta(iteration, result.delta)

        # Full model checkpoint only at configured interval
        # (expensive for 72B: ~135GB per save, so don't do every iteration)
        if (
            iteration > 0
            and self.iter_config.checkpoint_every > 0
            and iteration % self.iter_config.checkpoint_every == 0
        ):
            self._save_checkpoint(iteration, score)

        # Aggressive VRAM cleanup between iterations
        import gc; gc.collect()
        torch.cuda.empty_cache()

        return False

    # ─── Emergency Handling ───────────────────────────

    def _handle_emergency(self, alert: Alert) -> bool:
        """
        Handle an emergency alert.

        Returns:
            True if training should stop.
        """
        logger.critical(
            "EMERGENCY: %s — %s",
            alert.type, alert.message,
        )
        self._stats.emergency_rollbacks += 1

        # Rollback to best known state
        self._emergency_rollback(alert.message)

        # If too many emergencies, stop
        max_emergencies = self.catastrophe_config.max_emergency_count
        if self._stats.emergency_rollbacks >= max_emergencies:
            logger.critical(
                "Too many emergencies (%d) — aborting training",
                self._stats.emergency_rollbacks,
            )
            return True

        return False

    def _emergency_rollback(self, reason: str) -> None:
        """
        Roll back model to best known state.

        Args:
            reason: Why the rollback is happening.
        """
        if self._best_model_state is not None:
            named_params = dict(self.model.named_parameters())
            with torch.no_grad():
                for k, v in self._best_model_state.items():
                    if k in named_params:
                        named_params[k].copy_(v.to(named_params[k].device))
            logger.warning(
                "Emergency rollback to best model state. Reason: %s",
                reason,
            )
        else:
            # Try checkpoint manager
            best_ckpt = self._ckpt_manager.get_best()
            if best_ckpt is not None:
                self._ckpt_manager.load(
                    self.model, checkpoint_path=best_ckpt.path,
                )
                logger.warning(
                    "Emergency rollback to checkpoint %s. Reason: %s",
                    best_ckpt.path, reason,
                )
            else:
                logger.error(
                    "No rollback target available! Reason: %s",
                    reason,
                )

    # ─── Checkpoint ───────────────────────────────────

    def _checkpoint_epoch(
        self,
        epoch: int,
        label: str | None = None,
    ) -> None:
        """Save checkpoint at epoch boundary."""
        tag = label or f"epoch_{epoch}"
        score = self._best_score
        self._save_checkpoint(epoch, score, tag=tag)

    def _save_checkpoint(
        self,
        iteration: int,
        score: float,
        tag: str | None = None,
        epoch: int = 0,
    ) -> None:
        """Save a checkpoint (non-fatal on failure)."""
        try:
            # Gather domain scores from history
            domain_scores: dict[str, float] = {}
            recent = self.history.get_latest(n=1)
            if recent and hasattr(recent[0], "quick_bench") and recent[0].quick_bench:
                domain_scores = recent[0].quick_bench.domain_breakdown

            # Gather frozen layer checksums
            frozen_checksums: dict[str, str] = {}
            if hasattr(self, "_frozen_mask") and self._frozen_mask is not None:
                self._frozen_mask.update_checksums(self.model)
                frozen_checksums = dict(self._frozen_mask._checksums)

            self._ckpt_manager.save(
                model=self.model,
                epoch=epoch,
                iteration=iteration,
                score=score,
                domain_scores=domain_scores,
                frozen_checksums=frozen_checksums,
                strategy=self.strategy,
            )
        except Exception as e:
            logger.warning("Checkpoint save failed (non-fatal): %s", e)

    def _save_iteration_log(
        self,
        iteration: int,
        result: IterationResult,
        score: float,
    ) -> None:
        """
        Append a lightweight JSON line to the iteration log.

        This is crash-safe: even if the process dies mid-iteration,
        all completed iterations are logged. Cost: ~1KB per iteration.
        """
        import json
        log_path = self.output_dir / "iteration_log.jsonl"
        try:
            entry = {
                "iteration": iteration,
                "epoch": result.epoch,
                "accepted": result.accepted,
                "rejection_reason": result.rejection_reason,
                "delta_norm": result.proposed_delta_norm,
                "quick_bench_score": score,
                "quick_bench_scores": result.quick_bench_scores,
                "virtual_loss_before": result.virtual_loss_before,
                "virtual_loss_after": result.virtual_loss_after,
                "wall_time_s": result.wall_time_seconds,
                "total_accepted": self._stats.total_accepted,
                "total_rejected": self._stats.total_rejected,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
                f.flush()
            logger.info("Iteration %d logged to %s", iteration, log_path)
        except Exception as e:
            logger.warning("Iteration log write failed (non-fatal): %s", e)

    def _save_delta(self, iteration: int, delta: LoRADelta) -> None:
        """
        Save accepted surgery delta to disk (lightweight, LoRA format).

        Deltas are tiny (~100KB for rank-16 LoRA on a few layers)
        compared to full model checkpoints (~135GB for 72B).
        """
        from agisti.surgery.delta import DeltaSerializer
        delta_dir = self.output_dir / "deltas"
        delta_dir.mkdir(parents=True, exist_ok=True)
        delta_path = delta_dir / f"delta_iter{iteration:04d}.safetensors"
        try:
            DeltaSerializer.save(delta, delta_path)
            logger.info(
                "Saved accepted delta for iteration %d (%d layers, norm=%.6f)",
                iteration, len(delta), delta.norm(),
            )
        except Exception as e:
            logger.warning("Delta save failed (non-fatal): %s", e)

    def _resume(self, checkpoint_path: str) -> None:
        """Resume from a checkpoint."""
        info = self._ckpt_manager.load(
            model=self.model,
            checkpoint_path=checkpoint_path,
        )
        if info and hasattr(info, "strategy") and info.strategy:
            self.strategy = info.strategy
            logger.info(
                "Resumed from checkpoint %s (iteration=%d)",
                checkpoint_path, info.iteration,
            )

    # ─── Ceiling Breaker ─────────────────────────────

    def activate_ceiling_breaker(
        self,
        level: int = 1,
    ) -> bool:
        """
        Activate ceiling breaker at the specified level.

        Levels:
        1. External surgery signal (ground-truth problems)
        2. RAG-augmented surgery (document retrieval)
        3. Inter-model surgery (CKA alignment)
        4. Compositional discovery (pairwise skills)

        Args:
            level: Ceiling breaker level (1-4).

        Returns:
            True if the ceiling break was successful.
        """
        logger.info("Activating ceiling breaker level %d", level)
        self._stats.ceiling_breaks += 1
        # Ceiling breaker components are activated via the config,
        # and the iteration runner picks them up. The orchestrator
        # just flags the strategy change.
        strategy = copy.deepcopy(self.strategy)
        strategy.ceiling_level = level
        self.strategy = strategy
        return True

    # ─── Branching ────────────────────────────────────

    def fork_branch(self, name: str) -> str:
        """
        Fork a new exploration branch.

        Args:
            name: Branch name.

        Returns:
            Branch ID.
        """
        branch_id = self._branch_manager.fork(
            name=name,
            model=self.model,
            strategy=self.strategy,
            iteration=self._stats.total_iterations,
        )
        self._stats.branches_explored += 1
        logger.info("Forked branch: %s (id=%s)", name, branch_id)
        return branch_id

    # ─── Phase Transition ────────────────────────────

    def transition_to_phase(
        self,
        new_config: PhaseConfig,
        model: nn.Module | None = None,
        tokenizer: Any = None,
    ) -> None:
        """
        Transition to a new phase.

        Args:
            new_config: Configuration for the new phase.
            model: New model (if upgrading scale).
            tokenizer: New tokenizer (if changed).
        """
        logger.info(
            "Phase transition: %s → %s",
            self.phase_config.phase_id.value,
            new_config.phase_id.value,
        )

        if model is not None:
            self.model = model
        if tokenizer is not None:
            self.tokenizer = tokenizer

        self.phase_config = new_config
        self._stats.phases_completed += 1

        # Reset strategy for new phase
        self.strategy = MetaStrategy()

        # Save transition checkpoint
        self._save_checkpoint(
            iteration=self._stats.total_iterations,
            score=self._best_score,
            tag=f"phase_transition_{new_config.phase_id.value}",
        )

    # ─── Utility ──────────────────────────────────────

    @staticmethod
    def _score_from_result(result: IterationResult) -> float:
        """Extract a single score from result."""
        if result.quick_bench_scores:
            values = list(result.quick_bench_scores.values())
            return sum(values) / len(values) if values else 0.0
        return 0.0

    @property
    def stats(self) -> OrchestratorStats:
        return self._stats

    @property
    def current_phase(self) -> PhaseState | None:
        return self._phase_state

    @property
    def alerts(self) -> list[Alert]:
        return list(self._alerts)

    def get_diagnostics(self) -> dict[str, Any]:
        """
        Get comprehensive diagnostics.

        Returns:
            Dict with all metrics and state.
        """
        return {
            "stats": {
                "total_iterations": self._stats.total_iterations,
                "total_accepted": self._stats.total_accepted,
                "total_rejected": self._stats.total_rejected,
                "acceptance_rate": (
                    self._stats.total_accepted
                    / max(self._stats.total_iterations, 1)
                ),
                "emergency_rollbacks": self._stats.emergency_rollbacks,
                "ceiling_breaks": self._stats.ceiling_breaks,
                "branches_explored": self._stats.branches_explored,
                "wall_time_s": self._stats.total_wall_time_s,
            },
            "phase": {
                "id": (
                    self._phase_state.phase_id.value
                    if self._phase_state else None
                ),
                "epoch": (
                    self._phase_state.epoch
                    if self._phase_state else 0
                ),
                "iteration": (
                    self._phase_state.iteration
                    if self._phase_state else 0
                ),
                "best_score": (
                    self._phase_state.best_score
                    if self._phase_state else 0.0
                ),
            },
            "strategy": {
                "surgery_type": self.strategy.surgery_type,
                "lora_rank": self.strategy.lora_rank,
                "surgery_budget": self.strategy.surgery_budget,
                "difficulty_level": self.strategy.difficulty_level,
                "exploration_rate": self.strategy.exploration_rate,
            },
            "history": self.history.statistics(),
            "health": self._health_monitor.summary(),
            "alerts_count": len(self._alerts),
            "batch_performance": self._batch_tracker.summary(),
        }

    def format_status(self) -> str:
        """
        One-line status string.
        """
        s = self._stats
        ar = (
            s.total_accepted / max(s.total_iterations, 1)
        )
        return (
            f"Phase={self.phase_config.phase_id.value} "
            f"Iter={s.total_iterations} "
            f"Accept={s.total_accepted}/{s.total_iterations} "
            f"({ar:.1%}) "
            f"Best={self._best_score:.4f} "
            f"EmRollback={s.emergency_rollbacks}"
        )
