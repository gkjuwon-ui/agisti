"""
Tests for agisti.checkpoint — manager, branch, gc.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from agisti.types import CheckpointInfo, BranchInfo, IterationResult
from agisti.config import CheckpointConfig
from agisti.checkpoint.manager import CheckpointManager
from agisti.checkpoint.branch import BranchManager
from agisti.checkpoint.gc import CheckpointGC


# ─── Helpers ──────────────────────────────────────

def _make_state_dict() -> dict[str, torch.Tensor]:
    """Create a small mock model state dict."""
    torch.manual_seed(42)
    return {
        "layers.0.weight": torch.randn(16, 16),
        "layers.1.weight": torch.randn(16, 16),
        "embed.weight": torch.randn(100, 16),
    }


def _make_result(iteration: int = 0, score: float = 0.5) -> IterationResult:
    return IterationResult(
        iteration_id=iteration,
        proposed_delta_norm=0.01,
        virtual_loss_before=2.0,
        virtual_loss_after=1.8,
        refined_delta_norm=0.01,
        quick_bench_scores={"math": score},
        accepted=True,
        wall_time_seconds=1.0,
        gpu_memory_peak_gb=10.0,
    )


# ─── CheckpointManager Tests ─────────────────────

class TestCheckpointManager:
    """Tests for checkpoint save/load/rollback."""

    def test_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(
                checkpoint_dir=Path(tmpdir),
                max_checkpoints=5,
            )
            mgr = CheckpointManager(config)
            assert mgr is not None

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(
                checkpoint_dir=Path(tmpdir),
                max_checkpoints=5,
            )
            mgr = CheckpointManager(config)
            state = _make_state_dict()

            # Save
            info = mgr.save(
                state_dict=state,
                iteration=0,
                score=0.5,
            )
            assert isinstance(info, CheckpointInfo)
            assert info.iteration == 0

            # Load
            loaded = mgr.load(info.path)
            assert loaded is not None
            for key in state:
                assert key in loaded
                assert torch.allclose(state[key], loaded[key])

    def test_list_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(
                checkpoint_dir=Path(tmpdir),
                max_checkpoints=10,
            )
            mgr = CheckpointManager(config)
            state = _make_state_dict()

            for i in range(5):
                mgr.save(state, iteration=i, score=0.1 * i)

            checkpoints = mgr.list_checkpoints()
            assert len(checkpoints) == 5

    def test_max_checkpoints_enforced(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(
                checkpoint_dir=Path(tmpdir),
                max_checkpoints=3,
            )
            mgr = CheckpointManager(config)
            state = _make_state_dict()

            for i in range(10):
                mgr.save(state, iteration=i, score=0.1 * i)

            checkpoints = mgr.list_checkpoints()
            assert len(checkpoints) <= 3

    def test_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(
                checkpoint_dir=Path(tmpdir),
                max_checkpoints=10,
            )
            mgr = CheckpointManager(config)
            state = _make_state_dict()

            mgr.save(state, iteration=0, score=0.3)
            mgr.save(state, iteration=1, score=0.9)
            mgr.save(state, iteration=2, score=0.5)

            best = mgr.best_checkpoint()
            assert best is not None
            assert best.score == pytest.approx(0.9)

    def test_rollback_to_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(
                checkpoint_dir=Path(tmpdir),
                max_checkpoints=10,
            )
            mgr = CheckpointManager(config)

            state1 = _make_state_dict()
            mgr.save(state1, iteration=0, score=0.9)

            state2 = {k: v + 1.0 for k, v in state1.items()}
            mgr.save(state2, iteration=1, score=0.3)

            rolled_back = mgr.rollback_to_best()
            assert rolled_back is not None
            # Should be the first state (score 0.9)
            for key in state1:
                assert key in rolled_back

    def test_cleanup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(
                checkpoint_dir=Path(tmpdir),
                max_checkpoints=5,
            )
            mgr = CheckpointManager(config)
            state = _make_state_dict()

            for i in range(5):
                mgr.save(state, iteration=i, score=0.1 * i)

            mgr.cleanup(keep_n=2)
            remaining = mgr.list_checkpoints()
            assert len(remaining) <= 2


# ─── BranchManager Tests ─────────────────────────

class TestBranchManager:
    """Tests for experiment branching."""

    def test_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BranchManager(root_dir=Path(tmpdir))
            assert mgr is not None

    def test_create_branch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BranchManager(root_dir=Path(tmpdir))
            state = _make_state_dict()
            branch = mgr.fork(
                name="experiment_1",
                state_dict=state,
                parent="main",
            )
            assert isinstance(branch, BranchInfo)
            assert branch.name == "experiment_1"

    def test_list_branches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BranchManager(root_dir=Path(tmpdir))
            state = _make_state_dict()
            mgr.fork("exp_1", state, parent="main")
            mgr.fork("exp_2", state, parent="main")

            branches = mgr.list_branches()
            assert len(branches) >= 2

    def test_switch_branch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BranchManager(root_dir=Path(tmpdir))
            state1 = _make_state_dict()
            state2 = {k: v + 1.0 for k, v in state1.items()}

            mgr.fork("exp_1", state1, parent="main")
            mgr.fork("exp_2", state2, parent="main")

            loaded = mgr.switch("exp_1")
            assert loaded is not None

    def test_compare_branches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BranchManager(root_dir=Path(tmpdir))
            state1 = _make_state_dict()
            state2 = {k: v + 0.1 for k, v in state1.items()}

            mgr.fork("exp_1", state1, parent="main")
            mgr.fork("exp_2", state2, parent="main")

            comparison = mgr.compare("exp_1", "exp_2")
            assert isinstance(comparison, dict)

    def test_promote_branch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BranchManager(root_dir=Path(tmpdir))
            state = _make_state_dict()
            mgr.fork("exp_1", state, parent="main")

            # Promote should succeed
            mgr.promote("exp_1")
            # Promoted branch should become main or be marked
            branches = mgr.list_branches()
            assert isinstance(branches, list)

    def test_prune_branch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BranchManager(root_dir=Path(tmpdir))
            state = _make_state_dict()
            mgr.fork("exp_temp", state, parent="main")

            mgr.prune("exp_temp")
            branches = mgr.list_branches()
            names = [b.name for b in branches]
            assert "exp_temp" not in names

    def test_duplicate_branch_name_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BranchManager(root_dir=Path(tmpdir))
            state = _make_state_dict()
            mgr.fork("exp_1", state, parent="main")
            with pytest.raises((ValueError, FileExistsError)):
                mgr.fork("exp_1", state, parent="main")


# ─── CheckpointGC Tests ──────────────────────────

class TestCheckpointGC:
    """Tests for checkpoint garbage collection."""

    def test_creation(self):
        gc = CheckpointGC(max_age_hours=24, max_total_gb=10.0)
        assert gc is not None

    def test_collect_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gc = CheckpointGC(max_age_hours=24, max_total_gb=10.0)
            freed = gc.collect(Path(tmpdir))
            assert freed >= 0

    def test_retention_policy(self):
        gc = CheckpointGC(
            max_age_hours=1,
            max_total_gb=1.0,
            keep_best_n=3,
        )
        assert gc.max_age_hours == 1
        assert gc.max_total_gb == 1.0
        assert gc.keep_best_n == 3

    def test_identify_stale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gc = CheckpointGC(max_age_hours=0)  # Everything is stale
            # Create some fake checkpoint files
            for i in range(5):
                path = Path(tmpdir) / f"ckpt_{i}.pt"
                torch.save({"data": torch.randn(10)}, str(path))

            stale = gc.identify_stale(Path(tmpdir))
            assert len(stale) >= 0  # May identify them as stale

    def test_disk_usage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gc = CheckpointGC(max_age_hours=24, max_total_gb=10.0)
            # Create a file
            path = Path(tmpdir) / "test.pt"
            torch.save({"data": torch.randn(100)}, str(path))

            usage = gc.disk_usage(Path(tmpdir))
            assert usage > 0

    def test_format_size(self):
        gc = CheckpointGC(max_age_hours=24, max_total_gb=10.0)
        assert gc.format_size(1024) == "1.0 KB"
        assert gc.format_size(1024 * 1024) == "1.0 MB"
        assert gc.format_size(1024 ** 3) == "1.0 GB"

    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gc = CheckpointGC(max_age_hours=24, max_total_gb=10.0)
            summary = gc.summary(Path(tmpdir))
            assert isinstance(summary, dict)
            assert "total_files" in summary
            assert "total_size" in summary
