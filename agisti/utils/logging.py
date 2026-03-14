"""
Structured Logging — AGISTI-specific logging with structured fields.

Provides:
- TrainingLogger: structured logger with iteration context
- ProgressBar: lightweight progress display
- MetricsLogger: writes metric records to JSONL
- setup_logging(): configures root logging

All logging uses stdlib `logging` with custom formatters.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator


def setup_logging(
    level: int = logging.INFO,
    log_dir: Path | None = None,
    name: str = "agisti",
) -> logging.Logger:
    """
    Configure AGISTI logging.

    Args:
        level: Logging level.
        log_dir: Directory for log files. If None, logs to stderr only.
        name: Root logger name.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(AGISTIFormatter())
    logger.addHandler(console)

    # File handler (if log_dir specified)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(
            log_dir / f"agisti_{timestamp}.log",
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(AGISTIFormatter(include_time=True))
        logger.addHandler(fh)

    return logger


class AGISTIFormatter(logging.Formatter):
    """
    Custom log formatter with color support and structured output.
    """

    COLORS = {
        logging.DEBUG: "\033[36m",     # cyan
        logging.INFO: "\033[32m",      # green
        logging.WARNING: "\033[33m",   # yellow
        logging.ERROR: "\033[31m",     # red
        logging.CRITICAL: "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def __init__(self, include_time: bool = False) -> None:
        super().__init__()
        self._include_time = include_time

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelno
        color = self.COLORS.get(level, "")

        parts = []
        if self._include_time:
            parts.append(
                time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(record.created),
                )
            )

        parts.append(f"{color}{record.levelname:<8}{self.RESET}")
        parts.append(f"[{record.name}]")
        parts.append(record.getMessage())

        if record.exc_info and record.exc_info[0] is not None:
            parts.append(self.formatException(record.exc_info))

        return " ".join(parts)


class TrainingLogger:
    """
    Structured logger for training iterations.

    Adds iteration context to all log messages.
    """

    def __init__(
        self,
        name: str = "agisti.train",
        epoch: int = 0,
        iteration: int = 0,
    ) -> None:
        self._logger = logging.getLogger(name)
        self._epoch = epoch
        self._iteration = iteration
        self._extra: dict[str, Any] = {}

    def set_context(
        self,
        epoch: int | None = None,
        iteration: int | None = None,
    ) -> None:
        """Update the logging context."""
        if epoch is not None:
            self._epoch = epoch
        if iteration is not None:
            self._iteration = iteration

    def add_extra(self, key: str, value: Any) -> None:
        """Add extra context field."""
        self._extra[key] = value

    def _prefix(self) -> str:
        return f"[E{self._epoch}:I{self._iteration}]"

    def debug(self, msg: str, *args: Any) -> None:
        self._logger.debug(f"{self._prefix()} {msg}", *args)

    def info(self, msg: str, *args: Any) -> None:
        self._logger.info(f"{self._prefix()} {msg}", *args)

    def warning(self, msg: str, *args: Any) -> None:
        self._logger.warning(f"{self._prefix()} {msg}", *args)

    def error(self, msg: str, *args: Any) -> None:
        self._logger.error(f"{self._prefix()} {msg}", *args)

    def critical(self, msg: str, *args: Any) -> None:
        self._logger.critical(f"{self._prefix()} {msg}", *args)

    def metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
    ) -> None:
        """Log a metric value."""
        s = step if step is not None else self._iteration
        self._logger.info(
            "%s metric %s=%.6f (step=%d)",
            self._prefix(), name, value, s,
        )

    @contextmanager
    def timer(self, label: str) -> Generator[None, None, None]:
        """Context manager that logs elapsed time."""
        start = time.time()
        self.debug("Starting: %s", label)
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.info(
                "Completed: %s (%.2fs)",
                label, elapsed,
            )


class MetricsLogger:
    """
    Writes structured metric records to a JSONL file.

    Each record includes timestamp, epoch, iteration, and metrics.
    Used for post-hoc analysis and visualization.
    """

    def __init__(
        self,
        output_path: Path,
        buffer_size: int = 10,
    ) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict[str, Any]] = []
        self._buffer_size = buffer_size

    def log(
        self,
        epoch: int,
        iteration: int,
        metrics: dict[str, Any],
    ) -> None:
        """
        Log a set of metrics.

        Args:
            epoch: Current epoch.
            iteration: Current iteration.
            metrics: Dict of metric name → value.
        """
        record = {
            "timestamp": time.time(),
            "epoch": epoch,
            "iteration": iteration,
            **metrics,
        }
        self._buffer.append(record)

        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def log_iteration_result(
        self,
        epoch: int,
        iteration: int,
        accepted: bool,
        delta_norm: float,
        score: float,
        loss: float = 0.0,
        wall_time: float = 0.0,
    ) -> None:
        """Convenience: log iteration result metrics."""
        self.log(epoch, iteration, {
            "accepted": accepted,
            "delta_norm": delta_norm,
            "score": score,
            "loss": loss,
            "wall_time_s": wall_time,
        })

    def flush(self) -> None:
        """Write buffered records to file."""
        if not self._buffer:
            return

        with open(self._path, "a", encoding="utf-8") as f:
            for record in self._buffer:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._buffer.clear()

    def close(self) -> None:
        """Flush and close."""
        self.flush()


class ProgressBar:
    """
    Lightweight progress bar for terminal output.

    Does not depend on external libraries (no tqdm required).
    """

    def __init__(
        self,
        total: int,
        prefix: str = "",
        width: int = 40,
    ) -> None:
        self._total = total
        self._prefix = prefix
        self._width = width
        self._current = 0
        self._start_time = time.time()

    def update(self, n: int = 1) -> None:
        """Advance progress by n."""
        self._current = min(self._current + n, self._total)
        self._render()

    def _render(self) -> None:
        """Render the progress bar to stderr."""
        if self._total == 0:
            return

        frac = self._current / self._total
        filled = int(self._width * frac)
        bar = "█" * filled + "░" * (self._width - filled)

        elapsed = time.time() - self._start_time
        if self._current > 0:
            eta = elapsed * (self._total - self._current) / self._current
            eta_str = f"ETA: {eta:.0f}s"
        else:
            eta_str = "ETA: ?"

        line = (
            f"\r{self._prefix} |{bar}| "
            f"{self._current}/{self._total} "
            f"({frac:.0%}) {eta_str}  "
        )
        sys.stderr.write(line)
        sys.stderr.flush()

        if self._current >= self._total:
            sys.stderr.write("\n")

    def finish(self) -> None:
        """Complete the progress bar."""
        self._current = self._total
        self._render()


@contextmanager
def log_section(
    logger: logging.Logger,
    title: str,
    level: int = logging.INFO,
) -> Generator[None, None, None]:
    """
    Context manager that logs section start/end with timing.
    """
    logger.log(level, "┌─ %s", title)
    start = time.time()
    try:
        yield
    except Exception:
        logger.log(level, "└─ %s FAILED (%.2fs)", title, time.time() - start)
        raise
    else:
        logger.log(
            level,
            "└─ %s complete (%.2fs)",
            title, time.time() - start,
        )
