"""
Branch Manager — manages multiple experimental branches.

Supports parallel exploration of different training strategies
by maintaining separate checkpoint branches that can be compared,
merged, or abandoned.

Each branch shares the same frozen layers but has its own:
- Checkpoint history
- Meta-strategy
- Delta accumulation

Design: §9 — Branch management.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from agisti.types import BranchInfo, CheckpointInfo

logger = logging.getLogger(__name__)


@dataclass
class BranchState:
    """Detailed state of a branch."""
    info: BranchInfo
    latest_checkpoint: CheckpointInfo | None = None
    score_history: list[float] = field(default_factory=list)
    iterations_run: int = 0
    is_stale: bool = False
    parent_branch: str | None = None
    fork_iteration: int | None = None


class BranchManager:
    """
    Manages experimental branches for multi-hypothesis exploration.

    In Phase 1: max 2 branches
    In Phase 2: max 3 branches

    Each branch tests a different strategy/configuration while sharing
    the same frozen layers. The best-performing branch gets promoted
    to main, others are pruned.

    Branch lifecycle:
    1. Fork from main (or another branch)
    2. Run iterations with branch-specific strategy
    3. Compare against main periodically
    4. Promote if better, prune if worse
    """

    def __init__(
        self,
        base_dir: str | Path,
        max_branches: int = 3,
    ):
        self.base_dir = Path(base_dir)
        self.max_branches = max_branches
        self._branches: dict[str, BranchState] = {}
        self._active_branch: str = "main"

        # Initialize main branch
        self._branches["main"] = BranchState(
            info=BranchInfo(
                name="main",
                parent_epoch=0,
                strategy_description="Primary training branch",
                created_at=time.time(),
            ),
        )

        self._load_state()

    @property
    def active_branch(self) -> str:
        return self._active_branch

    @property
    def branch_count(self) -> int:
        return len(self._branches)

    def fork(
        self,
        branch_name: str,
        description: str = "",
        parent: str = "main",
        model: nn.Module | None = None,
    ) -> BranchState:
        """
        Create a new branch forked from an existing one.

        Args:
            branch_name: Name for the new branch.
            description: Description of the experiment.
            parent: Branch to fork from.
            model: Model to save as branch starting point.

        Returns:
            The new BranchState.

        Raises:
            ValueError: If max branches exceeded or name exists.
        """
        if len(self._branches) >= self.max_branches:
            raise ValueError(
                f"Maximum branches ({self.max_branches}) reached. "
                f"Prune an existing branch first."
            )

        if branch_name in self._branches:
            raise ValueError(f"Branch '{branch_name}' already exists")

        if parent not in self._branches:
            raise ValueError(f"Parent branch '{parent}' not found")

        parent_state = self._branches[parent]

        info = BranchInfo(
            name=branch_name,
            parent_epoch=parent_state.iterations_run if parent_state else 0,
            strategy_description=description or f"Fork of {parent}",
            created_at=time.time(),
        )

        state = BranchState(
            info=info,
            parent_branch=parent,
            fork_iteration=(
                parent_state.iterations_run
                if parent_state else 0
            ),
        )

        # Copy parent's latest score as starting point
        if parent_state.score_history:
            state.score_history.append(parent_state.score_history[-1])

        self._branches[branch_name] = state

        # Save branch state
        branch_dir = self.base_dir / "branches" / branch_name
        branch_dir.mkdir(parents=True, exist_ok=True)

        # Save model snapshot if provided
        if model is not None:
            model_path = branch_dir / "fork_model.pt"
            torch.save(model.state_dict(), model_path)

        self._save_state()

        logger.info(
            "Created branch '%s' from '%s' (total branches: %d)",
            branch_name, parent, len(self._branches),
        )

        return state

    def switch(self, branch_name: str) -> None:
        """Switch the active branch."""
        if branch_name not in self._branches:
            raise ValueError(f"Branch '{branch_name}' not found")
        self._active_branch = branch_name
        logger.info("Switched to branch '%s'", branch_name)

    def record_score(
        self,
        branch_name: str,
        score: float,
        iteration: int | None = None,
    ) -> None:
        """Record a score for a branch."""
        if branch_name not in self._branches:
            return

        state = self._branches[branch_name]
        state.score_history.append(score)
        if iteration is not None:
            state.iterations_run = iteration

    def compare(
        self,
        branch_a: str = "main",
        branch_b: str | None = None,
        window: int = 20,
    ) -> dict[str, Any]:
        """
        Compare two branches' performance.

        Args:
            branch_a: First branch (default: main).
            branch_b: Second branch (default: active if != main).
            window: Number of recent scores to compare.

        Returns:
            Comparison dict with scores, trends, and recommendation.
        """
        if branch_b is None:
            branch_b = (
                self._active_branch
                if self._active_branch != branch_a
                else None
            )
            if branch_b is None:
                return {"error": "Cannot compare: only one branch"}

        state_a = self._branches.get(branch_a)
        state_b = self._branches.get(branch_b)

        if not state_a or not state_b:
            return {"error": "Branch not found"}

        scores_a = state_a.score_history[-window:]
        scores_b = state_b.score_history[-window:]

        avg_a = sum(scores_a) / len(scores_a) if scores_a else 0.0
        avg_b = sum(scores_b) / len(scores_b) if scores_b else 0.0

        trend_a = self._compute_trend(scores_a)
        trend_b = self._compute_trend(scores_b)

        # Recommendation
        if avg_b > avg_a + 0.01 and trend_b > 0:
            recommendation = f"promote_{branch_b}"
        elif avg_a > avg_b + 0.01 and trend_a > 0:
            recommendation = f"promote_{branch_a}"
        elif trend_b < -0.005:
            recommendation = f"prune_{branch_b}"
        elif trend_a < -0.005:
            recommendation = f"prune_{branch_a}"
        else:
            recommendation = "continue"

        return {
            "branch_a": branch_a,
            "branch_b": branch_b,
            "avg_score_a": avg_a,
            "avg_score_b": avg_b,
            "trend_a": trend_a,
            "trend_b": trend_b,
            "latest_a": scores_a[-1] if scores_a else None,
            "latest_b": scores_b[-1] if scores_b else None,
            "recommendation": recommendation,
        }

    def promote(
        self,
        branch_name: str,
        model: nn.Module | None = None,
    ) -> None:
        """
        Promote a branch to become the new main.

        The current main becomes an archived branch.

        Args:
            branch_name: Branch to promote.
            model: Optional model to save.
        """
        if branch_name not in self._branches:
            raise ValueError(f"Branch '{branch_name}' not found")

        if branch_name == "main":
            return  # Already main

        # Archive current main
        old_main = self._branches.get("main")
        if old_main:
            archive_name = f"main_archived_{int(time.time())}"
            self._branches[archive_name] = old_main
            self._branches[archive_name].info.name = archive_name

        # Promote
        promoted = self._branches.pop(branch_name)
        promoted.info.name = "main"
        self._branches["main"] = promoted
        self._active_branch = "main"

        self._save_state()

        logger.info(
            "Promoted branch '%s' to main (old main archived as '%s')",
            branch_name, archive_name if old_main else "none",
        )

    def prune(self, branch_name: str) -> None:
        """
        Remove a branch.

        Cannot prune the main branch.
        """
        if branch_name == "main":
            raise ValueError("Cannot prune main branch")

        if branch_name not in self._branches:
            raise ValueError(f"Branch '{branch_name}' not found")

        del self._branches[branch_name]

        if self._active_branch == branch_name:
            self._active_branch = "main"

        self._save_state()
        logger.info("Pruned branch '%s'", branch_name)

    def prune_stale(self, min_improvement: float = 0.005) -> list[str]:
        """
        Prune branches that have stalled.

        Returns names of pruned branches.
        """
        pruned: list[str] = []
        for name, state in list(self._branches.items()):
            if name == "main":
                continue

            if len(state.score_history) < 20:
                continue

            trend = self._compute_trend(state.score_history[-20:])
            if trend < min_improvement:
                state.is_stale = True
                logger.info(
                    "Branch '%s' is stale (trend=%.5f), pruning",
                    name, trend,
                )
                self.prune(name)
                pruned.append(name)

        return pruned

    def get_branch(self, name: str) -> BranchState | None:
        """Get branch state by name."""
        return self._branches.get(name)

    def list_branches(self) -> list[BranchInfo]:
        """List all branches."""
        return [s.info for s in self._branches.values()]

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all branches."""
        return {
            "active": self._active_branch,
            "count": len(self._branches),
            "max": self.max_branches,
            "branches": {
                name: {
                    "iterations": state.iterations_run,
                    "latest_score": (
                        state.score_history[-1]
                        if state.score_history else None
                    ),
                    "score_count": len(state.score_history),
                    "parent": state.parent_branch,
                    "stale": state.is_stale,
                }
                for name, state in self._branches.items()
            },
        }

    @staticmethod
    def _compute_trend(scores: list[float]) -> float:
        """Compute linear trend from scores."""
        n = len(scores)
        if n < 3:
            return 0.0

        x_mean = (n - 1) / 2
        y_mean = sum(scores) / n

        numerator = sum(
            (i - x_mean) * (y - y_mean)
            for i, y in enumerate(scores)
        )
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        return numerator / denominator if denominator > 0 else 0.0

    def _save_state(self) -> None:
        """Save branch manager state to disk."""
        state_path = self.base_dir / "branches" / "branch_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "active": self._active_branch,
            "branches": {},
        }
        for name, state in self._branches.items():
            data["branches"][name] = {
                "name": state.info.name,
                "created_at": state.info.created_at,
                "description": state.info.strategy_description,
                "score_history": state.score_history[-100:],  # Keep last 100
                "iterations_run": state.iterations_run,
                "parent": state.parent_branch,
                "fork_iteration": state.fork_iteration,
            }

        state_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8",
        )

    def _load_state(self) -> None:
        """Load branch manager state from disk."""
        state_path = self.base_dir / "branches" / "branch_state.json"
        if not state_path.exists():
            return

        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            self._active_branch = data.get("active", "main")

            for name, bdata in data.get("branches", {}).items():
                info = BranchInfo(
                    name=bdata.get("name", name),
                    parent_epoch=bdata.get("parent_epoch", 0),
                    strategy_description=bdata.get("strategy_description", ""),
                    created_at=bdata.get("created_at", 0.0),
                )
                self._branches[name] = BranchState(
                    info=info,
                    score_history=bdata.get("score_history", []),
                    iterations_run=bdata.get("iterations_run", 0),
                    parent_branch=bdata.get("parent"),
                    fork_iteration=bdata.get("fork_iteration"),
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load branch state: %s", e)
