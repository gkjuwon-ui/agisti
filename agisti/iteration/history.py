"""
Iteration History — tracks all iteration results across the training run.

Provides trend analysis, regression detection, and progress reporting.
Used by MetaStrategyEngine and CatastropheDetector for feedback.

Design: §4.1 — iteration tracking and metrics aggregation.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from agisti.types import IterationResult, QuickBenchResult

logger = logging.getLogger(__name__)


@dataclass
class EpochSummary:
    """Summary statistics for one epoch."""
    epoch: int
    iterations: int
    accepted: int
    rejected: int
    acceptance_rate: float
    avg_delta_norm: float
    avg_quick_bench: float
    best_quick_bench: float
    total_wall_time: float
    avg_loss: float = 0.0
    domains: dict[str, float] = field(default_factory=dict)

    @property
    def rejection_rate(self) -> float:
        return 1.0 - self.acceptance_rate


class IterationHistory:
    """
    Full history of iteration results.

    Stores all IterationResults and computes trend analysis,
    regression detection, moving averages, etc.

    Thread-safe: uses in-memory list, no external state.
    """

    def __init__(self, window_size: int = 20) -> None:
        self._results: list[IterationResult] = []
        self._epoch_results: dict[int, list[IterationResult]] = {}
        self._window_size = window_size

    @property
    def total_iterations(self) -> int:
        return len(self._results)

    @property
    def accepted_count(self) -> int:
        return sum(1 for r in self._results if r.accepted)

    @property
    def rejected_count(self) -> int:
        return sum(1 for r in self._results if not r.accepted)

    @property
    def acceptance_rate(self) -> float:
        if not self._results:
            return 0.0
        return self.accepted_count / len(self._results)

    def add(self, result: IterationResult) -> None:
        """Record an iteration result."""
        self._results.append(result)
        epoch = result.epoch
        self._epoch_results.setdefault(epoch, []).append(result)
        logger.debug(
            "Recorded iteration %d (epoch %d, accepted=%s)",
            result.iteration_id, epoch, result.accepted,
        )

    def get_latest(self, n: int = 1) -> list[IterationResult]:
        """Get the n most recent results."""
        return self._results[-n:]

    def get_epoch_results(self, epoch: int) -> list[IterationResult]:
        """Get all results for a specific epoch."""
        return self._epoch_results.get(epoch, [])

    def iter_results(self) -> Iterator[IterationResult]:
        """Iterate over all results chronologically."""
        yield from self._results

    # ─── Trend Analysis ──────────────────────────────

    def moving_average_score(
        self,
        window: int | None = None,
    ) -> list[float]:
        """
        Compute moving average of quick bench scores.

        Args:
            window: Window size. Defaults to self._window_size.

        Returns:
            List of moving average values.
        """
        w = window or self._window_size
        scores = self._extract_scores()
        if len(scores) < w:
            return scores

        result = []
        for i in range(len(scores)):
            start = max(0, i - w + 1)
            window_vals = scores[start:i + 1]
            result.append(sum(window_vals) / len(window_vals))
        return result

    def delta_norm_trend(
        self,
        window: int | None = None,
    ) -> list[float]:
        """
        Moving average of delta norms.

        Useful for detecting surgery decay (norms decreasing → convergence).
        """
        w = window or self._window_size
        norms = [r.refined_delta_norm for r in self._results]
        if len(norms) < w:
            return norms

        result = []
        for i in range(len(norms)):
            start = max(0, i - w + 1)
            window_vals = norms[start:i + 1]
            result.append(sum(window_vals) / len(window_vals))
        return result

    def acceptance_rate_trend(
        self,
        window: int | None = None,
    ) -> list[float]:
        """Windowed acceptance rate over time."""
        w = window or self._window_size
        accepted = [1.0 if r.accepted else 0.0 for r in self._results]
        if len(accepted) < w:
            # Not enough data for windowed analysis
            return [
                sum(accepted[:i + 1]) / (i + 1)
                for i in range(len(accepted))
            ]

        result = []
        for i in range(len(accepted)):
            start = max(0, i - w + 1)
            window_vals = accepted[start:i + 1]
            result.append(sum(window_vals) / len(window_vals))
        return result

    def score_slope(self, window: int | None = None) -> float:
        """
        Linear regression slope of recent scores.

        Positive = improving, negative = regressing.

        Uses ordinary least squares on the last `window` scores.
        """
        w = window or self._window_size
        scores = self._extract_scores()
        if len(scores) < 3:
            return 0.0

        recent = scores[-w:]
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n

        numerator = sum(
            (i - x_mean) * (y - y_mean)
            for i, y in enumerate(recent)
        )
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if abs(denominator) < 1e-10:
            return 0.0

        return numerator / denominator

    def is_plateauing(
        self,
        window: int = 30,
        threshold: float = 0.001,
    ) -> bool:
        """
        Detect if progress has plateaued.

        Args:
            window: Number of recent iterations to check.
            threshold: Minimum slope to NOT be plateauing.

        Returns:
            True if the score slope is below threshold.
        """
        if len(self._results) < window:
            return False
        return abs(self.score_slope(window)) < threshold

    def detect_regression(
        self,
        lookback: int = 5,
        threshold: float = 0.02,
    ) -> bool:
        """
        Detect regression (recent scores significantly worse).

        Compares the average of the last `lookback` results to
        the overall average.
        """
        if len(self._results) < lookback * 2:
            return False

        scores = self._extract_scores()
        overall_avg = sum(scores) / len(scores)
        recent_avg = sum(scores[-lookback:]) / lookback

        return (overall_avg - recent_avg) > threshold

    def best_iteration(self) -> IterationResult | None:
        """Get the iteration with the highest quick bench score."""
        if not self._results:
            return None

        scores = self._extract_scores()
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return self._results[best_idx]

    # ─── Epoch Summaries ─────────────────────────────

    def epoch_summary(self, epoch: int) -> EpochSummary | None:
        """Compute summary for a specific epoch."""
        results = self._epoch_results.get(epoch)
        if not results:
            return None

        accepted = sum(1 for r in results if r.accepted)
        scores = []
        for r in results:
            avg = self._score_from_result(r)
            scores.append(avg)

        return EpochSummary(
            epoch=epoch,
            iterations=len(results),
            accepted=accepted,
            rejected=len(results) - accepted,
            acceptance_rate=accepted / len(results),
            avg_delta_norm=statistics.mean(
                r.refined_delta_norm for r in results
            ),
            avg_quick_bench=statistics.mean(scores) if scores else 0.0,
            best_quick_bench=max(scores) if scores else 0.0,
            total_wall_time=sum(r.wall_time_seconds for r in results),
        )

    def all_epoch_summaries(self) -> list[EpochSummary]:
        """Get summaries for all epochs."""
        summaries = []
        for epoch in sorted(self._epoch_results.keys()):
            s = self.epoch_summary(epoch)
            if s is not None:
                summaries.append(s)
        return summaries

    # ─── Domain Breakdown ────────────────────────────

    def domain_scores_over_time(
        self,
        domain: str,
    ) -> list[float]:
        """Extract scores for a specific domain over iterations."""
        values = []
        for r in self._results:
            if r.quick_bench_scores and domain in r.quick_bench_scores:
                values.append(r.quick_bench_scores[domain])
        return values

    def all_domains(self) -> set[str]:
        """Get set of all domains seen."""
        domains: set[str] = set()
        for r in self._results:
            if r.quick_bench_scores:
                domains.update(r.quick_bench_scores.keys())
        return domains

    def domain_progress(self) -> dict[str, dict[str, float]]:
        """
        Per-domain progress report.

        Returns:
            Dict mapping domain → {first, last, best, slope}.
        """
        report: dict[str, dict[str, float]] = {}
        for domain in self.all_domains():
            scores = self.domain_scores_over_time(domain)
            if not scores:
                continue

            # Compute slope
            n = len(scores)
            if n >= 3:
                x_mean = (n - 1) / 2.0
                y_mean = sum(scores) / n
                num = sum(
                    (i - x_mean) * (y - y_mean)
                    for i, y in enumerate(scores)
                )
                den = sum((i - x_mean) ** 2 for i in range(n))
                slope = num / den if abs(den) > 1e-10 else 0.0
            else:
                slope = 0.0

            report[domain] = {
                "first": scores[0],
                "last": scores[-1],
                "best": max(scores),
                "worst": min(scores),
                "mean": sum(scores) / len(scores),
                "slope": slope,
                "count": float(len(scores)),
            }

        return report

    # ─── Persistence ─────────────────────────────────

    def save_to_jsonl(self, path: Path) -> None:
        """
        Save iteration history to JSONL file.

        Each line is one IterationResult serialized to JSON.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for result in self._results:
                record = {
                    "iteration_id": result.iteration_id,
                    "epoch": result.epoch,
                    "accepted": result.accepted,
                    "rejection_reason": result.rejection_reason,
                    "proposed_delta_norm": result.proposed_delta_norm,
                    "refined_delta_norm": result.refined_delta_norm,
                    "virtual_loss_before": result.virtual_loss_before,
                    "virtual_loss_after": result.virtual_loss_after,
                    "quick_bench_scores": result.quick_bench_scores,
                    "wall_time_seconds": result.wall_time_seconds,
                    "gpu_memory_peak_gb": result.gpu_memory_peak_gb,
                    "target_layers": result.target_layers,
                    "surgery_type": result.surgery_type,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            "Saved %d iterations to %s",
            len(self._results), path,
        )

    @classmethod
    def load_from_jsonl(cls, path: Path) -> IterationHistory:
        """Load iteration history from JSONL file."""
        history = cls()

        if not path.exists():
            return history

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                result = IterationResult(
                    iteration_id=record["iteration_id"],
                    proposed_delta_norm=record.get(
                        "proposed_delta_norm", 0.0
                    ),
                    virtual_loss_before=record.get(
                        "virtual_loss_before", 0.0
                    ),
                    virtual_loss_after=record.get(
                        "virtual_loss_after", 0.0
                    ),
                    refined_delta_norm=record.get(
                        "refined_delta_norm", 0.0
                    ),
                    quick_bench_scores=record.get(
                        "quick_bench_scores", {}
                    ),
                    accepted=record.get("accepted", False),
                    rejection_reason=record.get("rejection_reason"),
                    wall_time_seconds=record.get(
                        "wall_time_seconds", 0.0
                    ),
                    gpu_memory_peak_gb=record.get(
                        "gpu_memory_peak_gb", 0.0
                    ),
                    target_layers=record.get("target_layers", []),
                    surgery_type=record.get("surgery_type", "micro"),
                    epoch=record.get("epoch", 0),
                )
                history.add(result)

        logger.info(
            "Loaded %d iterations from %s",
            history.total_iterations, path,
        )
        return history

    # ─── Statistics ───────────────────────────────────

    def statistics(self) -> dict[str, Any]:
        """
        Comprehensive statistics.

        Returns:
            Dict with metrics, trends, domain breakdown.
        """
        scores = self._extract_scores()

        return {
            "total_iterations": self.total_iterations,
            "accepted": self.accepted_count,
            "rejected": self.rejected_count,
            "acceptance_rate": self.acceptance_rate,
            "score": {
                "mean": (
                    statistics.mean(scores)
                    if scores else 0.0
                ),
                "std": (
                    statistics.stdev(scores)
                    if len(scores) >= 2 else 0.0
                ),
                "min": min(scores) if scores else 0.0,
                "max": max(scores) if scores else 0.0,
                "latest": scores[-1] if scores else 0.0,
            },
            "delta_norm": {
                "mean": (
                    statistics.mean(
                        r.refined_delta_norm
                        for r in self._results
                    )
                    if self._results else 0.0
                ),
            },
            "wall_time": {
                "total_s": sum(
                    r.wall_time_seconds for r in self._results
                ),
                "avg_s": (
                    statistics.mean(
                        r.wall_time_seconds
                        for r in self._results
                    )
                    if self._results else 0.0
                ),
            },
            "slope": self.score_slope(),
            "plateauing": self.is_plateauing(),
            "regressing": self.detect_regression(),
            "domains": self.domain_progress(),
            "epochs": len(self._epoch_results),
        }

    def format_report(self) -> str:
        """
        Human-readable progress report.
        """
        stats = self.statistics()
        lines = [
            "=" * 60,
            "AGISTI Iteration History Report",
            "=" * 60,
            f"Total iterations: {stats['total_iterations']}",
            f"Accepted: {stats['accepted']} "
            f"({stats['acceptance_rate']:.1%})",
            f"Rejected: {stats['rejected']}",
            f"Score: mean={stats['score']['mean']:.4f}, "
            f"max={stats['score']['max']:.4f}, "
            f"latest={stats['score']['latest']:.4f}",
            f"Slope: {stats['slope']:.6f} "
            f"({'improving' if stats['slope'] > 0 else 'declining'})",
            f"Plateauing: {stats['plateauing']}",
            f"Regressing: {stats['regressing']}",
            f"Wall time: {stats['wall_time']['total_s']:.1f}s total, "
            f"{stats['wall_time']['avg_s']:.1f}s avg",
        ]

        if stats['domains']:
            lines.append("")
            lines.append("Domain Progress:")
            for domain, info in sorted(stats['domains'].items()):
                lines.append(
                    f"  {domain}: {info['first']:.3f} → "
                    f"{info['last']:.3f} "
                    f"(best={info['best']:.3f}, "
                    f"slope={info['slope']:.4f})"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    # ─── Internal ─────────────────────────────────────

    def _extract_scores(self) -> list[float]:
        """Extract a single representative score per iteration."""
        return [self._score_from_result(r) for r in self._results]

    @staticmethod
    def _score_from_result(result: IterationResult) -> float:
        """
        Extract a single numeric score from an IterationResult.

        Priority: quick_bench.accuracy > mean of quick_bench_scores.
        """
        if hasattr(result, 'quick_bench') and result.quick_bench is not None:
            return result.quick_bench.accuracy

        if result.quick_bench_scores:
            values = list(result.quick_bench_scores.values())
            return sum(values) / len(values) if values else 0.0

        return 0.0
