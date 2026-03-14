"""
Tests for agisti.utils.logging — TrainingLogger, MetricsLogger, ProgressBar.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest

from agisti.utils.logging import (
    setup_logging,
    TrainingLogger,
    MetricsLogger,
    ProgressBar,
    AGISTIFormatter,
    log_section,
)


# ─── Setup Logging Tests ─────────────────────────

class TestSetupLogging:
    """Tests for logging setup."""

    def test_setup_returns_logger(self):
        logger = setup_logging("test_agisti")
        assert isinstance(logger, logging.Logger)

    def test_setup_with_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = setup_logging("test_file", log_file=log_path)
            logger.info("test message")
            assert log_path.exists()

    def test_setup_with_level(self):
        logger = setup_logging("test_level", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_formatter(self):
        fmt = AGISTIFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )
        formatted = fmt.format(record)
        assert "test message" in formatted


# ─── TrainingLogger Tests ─────────────────────────

class TestTrainingLogger:
    """Tests for structured training logger."""

    def test_creation(self):
        logger = TrainingLogger("test_training")
        assert logger is not None

    def test_log_iteration(self):
        logger = TrainingLogger("test_iter")
        # Should not raise
        logger.log_iteration(
            epoch=0,
            iteration=5,
            score=0.75,
            delta_norm=0.01,
            accepted=True,
        )

    def test_log_with_context(self):
        logger = TrainingLogger("test_ctx")
        logger.set_context(epoch=1, phase="phase_0")
        logger.log_iteration(
            epoch=1,
            iteration=10,
            score=0.8,
        )

    def test_timer_context(self):
        logger = TrainingLogger("test_timer")
        with logger.timer("test_operation") as t:
            # Do some work
            total = sum(range(100))
        assert t.elapsed >= 0

    def test_log_alert(self):
        logger = TrainingLogger("test_alert")
        logger.log_alert(
            level="WARNING",
            alert_type="PLATEAU",
            message="Score plateauing for 20 iterations",
        )

    def test_log_phase_transition(self):
        logger = TrainingLogger("test_phase")
        logger.log_phase_transition(
            from_phase="PHASE_0",
            to_phase="PHASE_1",
            reason="Convergence reached",
        )


# ─── MetricsLogger Tests ─────────────────────────

class TestMetricsLogger:
    """Tests for buffered metrics logger."""

    def test_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            logger = MetricsLogger(path)
            assert logger is not None

    def test_log_and_flush(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            logger = MetricsLogger(path, buffer_size=5)

            for i in range(10):
                logger.log({
                    "iteration": i,
                    "score": 0.5 + 0.01 * i,
                })
            logger.flush()

            assert path.exists()
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 10

    def test_auto_flush_on_buffer_full(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            logger = MetricsLogger(path, buffer_size=3)

            for i in range(5):
                logger.log({"i": i})
            logger.flush()

            lines = path.read_text().strip().split("\n")
            assert len(lines) == 5

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            with MetricsLogger(path) as logger:
                logger.log({"test": 1})
            # Should auto-flush on exit
            assert path.exists()


# ─── ProgressBar Tests ────────────────────────────

class TestProgressBar:
    """Tests for terminal progress bar (no tqdm)."""

    def test_creation(self):
        pb = ProgressBar(total=100, desc="Training")
        assert pb is not None

    def test_update(self):
        pb = ProgressBar(total=10, desc="Test")
        for i in range(10):
            pb.update(1)
        assert pb.current == 10

    def test_percentage(self):
        pb = ProgressBar(total=200, desc="Test")
        pb.update(100)
        assert pb.percentage == pytest.approx(50.0)

    def test_format(self):
        pb = ProgressBar(total=100, desc="Epochs")
        pb.update(50)
        text = pb.format()
        assert "50" in text
        assert "Epochs" in text

    def test_zero_total(self):
        pb = ProgressBar(total=0, desc="Empty")
        assert pb.percentage == 100.0

    def test_complete(self):
        pb = ProgressBar(total=5, desc="Test")
        for _ in range(5):
            pb.update(1)
        assert pb.is_complete


# ─── Log Section Context Manager Tests ────────────

class TestLogSection:
    """Tests for log_section context manager."""

    def test_section(self):
        logger = logging.getLogger("test_section")
        with log_section(logger, "Test Section"):
            logger.info("Inside section")

    def test_nested_sections(self):
        logger = logging.getLogger("test_nested")
        with log_section(logger, "Outer"):
            with log_section(logger, "Inner"):
                logger.info("Nested")
