"""
Checkpoint Garbage Collection — manages disk space.

Implements a retention policy to keep disk usage bounded
while preserving critical checkpoints (best, branch points,
recent checkpoints).

Policies:
- Keep the best checkpoint always
- Keep the most recent N checkpoints per branch
- Keep checkpoints at branch fork points
- Remove stale branch checkpoints on prune
- Age-based expiry for old checkpoints

Design: §9 — Checkpoint lifecycle management.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agisti.types import CheckpointInfo
from agisti.config import CheckpointConfig

logger = logging.getLogger(__name__)


@dataclass
class GCPolicy:
    """Garbage collection policy configuration."""
    max_checkpoints_per_branch: int = 10
    max_total_checkpoints: int = 50
    max_age_hours: float = 168.0  # 7 days
    keep_best: bool = True
    keep_branch_points: bool = True
    min_free_space_gb: float = 10.0


@dataclass
class GCResult:
    """Result of a garbage collection run."""
    deleted_count: int
    freed_bytes: int
    kept_count: int
    skipped_reasons: dict[str, int]

    @property
    def freed_gb(self) -> float:
        return self.freed_bytes / (1024 ** 3)


class CheckpointGC:
    """
    Garbage collection for model checkpoints.

    Ensures disk usage stays bounded while preserving:
    1. The best checkpoint (emergency rollback)
    2. Recent checkpoints per branch
    3. Branch fork points (for diffing)
    """

    def __init__(
        self,
        checkpoints_dir: str | Path,
        policy: GCPolicy | None = None,
    ):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.policy = policy or GCPolicy()

    def collect(
        self,
        all_checkpoints: list[CheckpointInfo],
        best_path: str | None = None,
        branch_points: set[str] | None = None,
    ) -> GCResult:
        """
        Run garbage collection.

        Args:
            all_checkpoints: List of all known checkpoints.
            best_path: Path to the best checkpoint (never deleted).
            branch_points: Set of checkpoint paths that are branch fork points.

        Returns:
            GCResult with deletion stats.
        """
        if not all_checkpoints:
            return GCResult(0, 0, 0, {})

        protect = branch_points or set()
        if best_path:
            protect.add(best_path)

        # Determine which checkpoints to keep vs delete
        to_keep: set[str] = set()
        to_delete: list[CheckpointInfo] = []
        skipped: dict[str, int] = {}
        now = time.time()

        # Group by branch
        by_branch: dict[str, list[CheckpointInfo]] = {}
        for ckpt in all_checkpoints:
            branch = ckpt.branch_name
            if branch not in by_branch:
                by_branch[branch] = []
            by_branch[branch].append(ckpt)

        # For each branch, keep the N most recent
        for branch, ckpts in by_branch.items():
            sorted_ckpts = sorted(ckpts, key=lambda c: c.iteration, reverse=True)

            for i, ckpt in enumerate(sorted_ckpts):
                path = ckpt.path

                # Always keep protected checkpoints
                if path in protect:
                    to_keep.add(path)
                    skipped["protected"] = skipped.get("protected", 0) + 1
                    continue

                # Keep best
                if self.policy.keep_best and path == best_path:
                    to_keep.add(path)
                    skipped["best"] = skipped.get("best", 0) + 1
                    continue

                # Keep branch fork points
                if self.policy.keep_branch_points and path in protect:
                    to_keep.add(path)
                    skipped["branch_point"] = skipped.get("branch_point", 0) + 1
                    continue

                # Keep recent N per branch
                if i < self.policy.max_checkpoints_per_branch:
                    to_keep.add(path)
                    continue

                # Age check
                age_hours = (now - ckpt.timestamp) / 3600
                if age_hours < 1.0:
                    # Don't delete checkpoints less than 1 hour old
                    to_keep.add(path)
                    skipped["recent"] = skipped.get("recent", 0) + 1
                    continue

                # This checkpoint can be deleted
                to_delete.append(ckpt)

        # Enforce total limit
        if len(to_keep) > self.policy.max_total_checkpoints:
            # Sort kept checkpoints by score and remove lowest
            keep_list = []
            for ckpt in all_checkpoints:
                if ckpt.path in to_keep:
                    keep_list.append(ckpt)

            keep_list.sort(key=lambda c: c.weighted_score, reverse=True)

            for ckpt in keep_list[self.policy.max_total_checkpoints:]:
                if ckpt.path not in protect:
                    to_keep.discard(ckpt.path)
                    to_delete.append(ckpt)

        # Execute deletion
        freed = 0
        deleted = 0

        for ckpt in to_delete:
            ckpt_path = Path(ckpt.path)
            if ckpt_path.exists():
                try:
                    size = sum(
                        f.stat().st_size
                        for f in ckpt_path.rglob("*")
                        if f.is_file()
                    )
                    shutil.rmtree(ckpt_path)
                    freed += size
                    deleted += 1
                    logger.debug(
                        "Deleted checkpoint: %s (%.1f MB)",
                        ckpt_path.name, size / (1024 * 1024),
                    )
                except OSError as e:
                    logger.warning(
                        "Failed to delete %s: %s", ckpt_path, e,
                    )

        result = GCResult(
            deleted_count=deleted,
            freed_bytes=freed,
            kept_count=len(to_keep),
            skipped_reasons=skipped,
        )

        if deleted > 0:
            logger.info(
                "GC complete: deleted %d checkpoints, freed %.1f GB, kept %d",
                deleted, result.freed_gb, len(to_keep),
            )

        return result

    def check_disk_space(self) -> dict[str, float]:
        """
        Check disk space for the checkpoints directory.

        Returns dict with total_gb, used_gb, free_gb, usage_percent.
        """
        import shutil as sh

        try:
            usage = sh.disk_usage(self.checkpoints_dir)
            total_gb = usage.total / (1024 ** 3)
            used_gb = usage.used / (1024 ** 3)
            free_gb = usage.free / (1024 ** 3)

            return {
                "total_gb": total_gb,
                "used_gb": used_gb,
                "free_gb": free_gb,
                "usage_percent": (used_gb / total_gb * 100) if total_gb > 0 else 0,
            }
        except OSError:
            return {
                "total_gb": 0, "used_gb": 0,
                "free_gb": 0, "usage_percent": 0,
            }

    def needs_gc(
        self,
        checkpoint_count: int,
    ) -> bool:
        """
        Check if garbage collection should run.

        Returns True if:
        - Too many checkpoints
        - Disk space is low
        """
        if checkpoint_count > self.policy.max_total_checkpoints:
            return True

        disk = self.check_disk_space()
        if disk["free_gb"] < self.policy.min_free_space_gb:
            return True

        return False

    def estimate_checkpoint_size(self, model: Any) -> float:
        """
        Estimate the size of a checkpoint in GB.

        Uses parameter count as a proxy.
        """
        total_params = sum(
            p.numel() for p in model.parameters()
        )
        # Each parameter uses 4 bytes (float32) or 2 bytes (float16)
        # Plus overhead, so estimate 5 bytes per parameter
        estimated_bytes = total_params * 5
        return estimated_bytes / (1024 ** 3)
