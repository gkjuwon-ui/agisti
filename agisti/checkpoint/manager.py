"""
Checkpoint Manager — saves, loads, and manages model checkpoints.

Handles the full checkpoint lifecycle:
- Saving model state, optimizer state, frozen mask, strategy
- Loading checkpoints for resumption or rollback
- Maintaining the "best" checkpoint for emergency rollback
- Integration with the BranchManager for multi-branch experiments

Design: §9 — Checkpoint Management.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from agisti.types import CheckpointInfo
from agisti.config import CheckpointConfig, MetaStrategy

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Extended metadata for a checkpoint."""
    info: CheckpointInfo
    strategy: MetaStrategy | None = None
    frozen_layers: list[str] = field(default_factory=list)
    competency_snapshot: dict[str, float] = field(default_factory=dict)
    iteration_history_length: int = 0
    file_size_bytes: int = 0


class CheckpointManager:
    """
    Manages model checkpoints for the AGISTI training loop.

    Responsibilities:
    - Save checkpoints at configured intervals
    - Track the "best" checkpoint for emergency rollback
    - Load checkpoints for resumption
    - Support multi-branch checkpointing via BranchManager
    - Garbage collection of old checkpoints
    """

    def __init__(
        self,
        base_dir: str | Path,
        config: CheckpointConfig | None = None,
    ):
        self.base_dir = Path(base_dir)
        self.config = config or CheckpointConfig()

        # Create directory structure
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self._best_score: float = -float("inf")
        self._best_path: Path | None = None
        self._all_checkpoints: list[CheckpointMetadata] = []
        self._load_index()

    def save(
        self,
        model: nn.Module,
        epoch: int,
        iteration: int,
        score: float,
        domain_scores: dict[str, float],
        frozen_checksums: dict[str, str],
        strategy: MetaStrategy | None = None,
        branch: str = "main",
        optimizer_state: dict[str, Any] | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> CheckpointInfo:
        """
        Save a model checkpoint.

        Args:
            model: The model to checkpoint.
            epoch: Current epoch number.
            iteration: Current iteration number.
            score: Weighted benchmark score.
            domain_scores: Per-domain scores.
            frozen_checksums: SHA-256 checksums of frozen layers.
            strategy: Current meta-strategy.
            branch: Branch name (default "main").
            optimizer_state: Optional optimizer state dict.
            extra_state: Any additional state to save.

        Returns:
            CheckpointInfo describing the saved checkpoint.
        """
        timestamp = time.time()
        ckpt_name = (
            f"checkpoint_e{epoch}_i{iteration}_{branch}"
        )
        ckpt_dir = self.checkpoints_dir / branch / ckpt_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state — async via thread to avoid blocking GPU
        model_path = ckpt_dir / "model.pt"
        state_dict_cpu = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
        }

        import concurrent.futures

        save_futures: list[concurrent.futures.Future] = []

        def _save_model():
            torch.save(state_dict_cpu, model_path)

        executor = getattr(self, "_save_executor", None)
        if executor is None:
            self._save_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="ckpt-save",
            )
            executor = self._save_executor

        # Wait for any previous async save to finish before starting new one
        prev_future = getattr(self, "_prev_save_future", None)
        if prev_future is not None:
            prev_future.result()  # block until previous save done

        save_futures.append(executor.submit(_save_model))

        # Save optimizer state if provided (also async)
        if optimizer_state:
            opt_path = ckpt_dir / "optimizer.pt"
            opt_copy = {
                k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in optimizer_state.items()
            }

            def _save_opt():
                torch.save(opt_copy, opt_path)

            save_futures.append(executor.submit(_save_opt))

        # Build checkpoint info
        info = CheckpointInfo(
            epoch=epoch,
            iteration=iteration,
            timestamp=timestamp,
            path=str(ckpt_dir),
            weighted_score=score,
            domain_scores=dict(domain_scores),
            frozen_checksums=dict(frozen_checksums),
            branch_name=branch,
        )

        # Save metadata
        metadata = {
            "info": {
                "epoch": info.epoch,
                "iteration": info.iteration,
                "timestamp": info.timestamp,
                "path": info.path,
                "weighted_score": info.weighted_score,
                "domain_scores": info.domain_scores,
                "frozen_checksums": info.frozen_checksums,
                "branch_name": info.branch_name,
            },
            "strategy": strategy.to_dict() if strategy else None,
            "extra": extra_state or {},
        }

        meta_path = ckpt_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding="utf-8",
        )

        # Track metadata (file size estimated, actual computed after save)
        ckpt_meta = CheckpointMetadata(
            info=info,
            strategy=strategy,
            file_size_bytes=0,  # updated after async save completes
        )
        self._all_checkpoints.append(ckpt_meta)

        # Store futures so next save waits for completion
        def _on_save_done(meta=ckpt_meta, d=ckpt_dir):
            meta.file_size_bytes = sum(
                f.stat().st_size for f in d.rglob("*") if f.is_file()
            )

        import concurrent.futures

        combined = concurrent.futures.Future()

        def _wait_all(futures=save_futures, cb=_on_save_done):
            for f in futures:
                f.result()
            cb()

        self._prev_save_future = executor.submit(_wait_all)

        # Update best checkpoint
        if score > self._best_score:
            self._best_score = score
            self._best_path = ckpt_dir
            logger.info(
                "New best checkpoint: score=%.4f at epoch=%d iter=%d",
                score, epoch, iteration,
            )

        # Save index
        self._save_index()

        logger.info(
            "Saved checkpoint: %s (score=%.4f, branch=%s)",
            ckpt_name, score, branch,
        )

        return info

    def load(
        self,
        model: nn.Module,
        checkpoint_path: str | Path | None = None,
        load_best: bool = False,
        branch: str = "main",
    ) -> CheckpointInfo | None:
        """
        Load a model checkpoint.

        Args:
            model: Model to load weights into.
            checkpoint_path: Specific checkpoint to load.
            load_best: Load the best checkpoint by score.
            branch: Branch to search for latest checkpoint.

        Returns:
            CheckpointInfo if loaded, None if no checkpoint found.
        """
        ckpt_dir: Path | None = None

        if checkpoint_path:
            ckpt_dir = Path(checkpoint_path)
        elif load_best and self._best_path:
            ckpt_dir = self._best_path
        else:
            # Find latest checkpoint on branch
            ckpt_dir = self._find_latest(branch)

        if ckpt_dir is None or not ckpt_dir.exists():
            logger.warning("No checkpoint found to load")
            return None

        # Load model state
        model_path = ckpt_dir / "model.pt"
        if not model_path.exists():
            logger.error("Model file not found in checkpoint: %s", ckpt_dir)
            return None

        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        # Load metadata
        meta_path = ckpt_dir / "metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            info_data = metadata.get("info", {})
            info = CheckpointInfo(
                epoch=info_data.get("epoch", 0),
                iteration=info_data.get("iteration", 0),
                timestamp=info_data.get("timestamp", 0.0),
                path=str(ckpt_dir),
                weighted_score=info_data.get("weighted_score", 0.0),
                domain_scores=info_data.get("domain_scores", {}),
                frozen_checksums=info_data.get("frozen_checksums", {}),
                branch_name=info_data.get("branch_name", "main"),
            )
        else:
            info = CheckpointInfo(
                epoch=0, iteration=0, timestamp=0.0,
                path=str(ckpt_dir), weighted_score=0.0,
                domain_scores={}, frozen_checksums={},
            )

        logger.info(
            "Loaded checkpoint from %s (score=%.4f, epoch=%d)",
            ckpt_dir.name, info.weighted_score, info.epoch,
        )

        return info

    def load_optimizer(
        self,
        checkpoint_path: str | Path,
    ) -> dict[str, Any] | None:
        """Load optimizer state from a checkpoint."""
        opt_path = Path(checkpoint_path) / "optimizer.pt"
        if not opt_path.exists():
            return None
        return torch.load(opt_path, map_location="cpu", weights_only=True)

    def get_best(self) -> CheckpointInfo | None:
        """Get info about the best checkpoint."""
        if self._best_path is None:
            return None

        for meta in reversed(self._all_checkpoints):
            if meta.info.path == str(self._best_path):
                return meta.info

        return None

    def get_latest(
        self, branch: str = "main",
    ) -> CheckpointInfo | None:
        """Get info about the latest checkpoint on a branch."""
        branch_ckpts = [
            m for m in self._all_checkpoints
            if m.info.branch_name == branch
        ]
        if not branch_ckpts:
            return None
        return max(
            branch_ckpts, key=lambda m: m.info.iteration,
        ).info

    def rollback_to_best(self, model: nn.Module) -> CheckpointInfo | None:
        """Emergency rollback to the best checkpoint."""
        logger.warning("EMERGENCY ROLLBACK to best checkpoint")
        return self.load(model, load_best=True)

    def list_checkpoints(
        self, branch: str | None = None,
    ) -> list[CheckpointInfo]:
        """List all checkpoints, optionally filtered by branch."""
        results = [m.info for m in self._all_checkpoints]
        if branch:
            results = [c for c in results if c.branch_name == branch]
        return sorted(results, key=lambda c: c.iteration)

    def _find_latest(self, branch: str) -> Path | None:
        """Find the latest checkpoint directory for a branch."""
        branch_dir = self.checkpoints_dir / branch
        if not branch_dir.exists():
            return None

        candidates = sorted(
            [d for d in branch_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )

        return candidates[0] if candidates else None

    def _save_index(self) -> None:
        """Save the checkpoint index to disk."""
        index_path = self.base_dir / "checkpoint_index.json"
        data = {
            "best_score": self._best_score,
            "best_path": str(self._best_path) if self._best_path else None,
            "checkpoints": [
                {
                    "epoch": m.info.epoch,
                    "iteration": m.info.iteration,
                    "path": m.info.path,
                    "score": m.info.weighted_score,
                    "branch": m.info.branch_name,
                    "timestamp": m.info.timestamp,
                }
                for m in self._all_checkpoints
            ],
        }
        index_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8",
        )

    def _load_index(self) -> None:
        """Load checkpoint index from disk."""
        index_path = self.base_dir / "checkpoint_index.json"
        if not index_path.exists():
            return

        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            self._best_score = data.get("best_score", -float("inf"))
            bp = data.get("best_path")
            self._best_path = Path(bp) if bp else None

            for entry in data.get("checkpoints", []):
                info = CheckpointInfo(
                    epoch=entry.get("epoch", 0),
                    iteration=entry.get("iteration", 0),
                    timestamp=entry.get("timestamp", 0.0),
                    path=entry.get("path", ""),
                    weighted_score=entry.get("score", 0.0),
                    domain_scores={},
                    frozen_checksums={},
                    branch_name=entry.get("branch", "main"),
                )
                self._all_checkpoints.append(
                    CheckpointMetadata(info=info),
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load checkpoint index: %s", e)
