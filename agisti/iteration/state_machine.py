"""
Iteration State Machine — manages the state transitions
for each iteration of the AGISTI loop.

The state machine enforces valid transitions and provides
introspection: current state, timing, transition history.

Design: §4.1 — references IterationState enum from types.py.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from agisti.types import IterationState

logger = logging.getLogger(__name__)


# Valid transitions: from_state → set(to_states)
VALID_TRANSITIONS: dict[IterationState, set[IterationState]] = {
    IterationState.IDLE: {IterationState.PROBE},
    IterationState.PROBE: {IterationState.GENERATE},
    IterationState.GENERATE: {IterationState.SOLVE},
    IterationState.SOLVE: {IterationState.EVALUATE},
    IterationState.EVALUATE: {IterationState.PROPOSE},
    IterationState.PROPOSE: {
        IterationState.VIRTUAL_TRAIN,
        IterationState.FEEDBACK,  # skip VT if no delta
    },
    IterationState.VIRTUAL_TRAIN: {
        IterationState.APPLY_DELTA,
        IterationState.FEEDBACK,  # reject before apply
    },
    IterationState.SNAPSHOT: {IterationState.APPLY_DELTA},
    IterationState.APPLY_DELTA: {
        IterationState.QUICK_BENCH,
        IterationState.SNAPSHOT,  # snapshot before apply
    },
    IterationState.QUICK_BENCH: {IterationState.FEEDBACK},
    IterationState.FEEDBACK: {
        IterationState.ROLLBACK,
        IterationState.IDLE,
    },
    IterationState.ROLLBACK: {IterationState.IDLE},
    IterationState.CEILING_BREAK: {IterationState.PROPOSE},
    IterationState.PHASE_TRANSITION: {IterationState.IDLE},
    IterationState.COMPLETE: set(),  # terminal
}


@dataclass
class StateTransition:
    """Record of a state transition."""
    from_state: IterationState
    to_state: IterationState
    timestamp: float = field(default_factory=time.time)
    elapsed_ms: float = 0.0

    def __repr__(self) -> str:
        return (
            f"StateTransition({self.from_state.value}"
            f" → {self.to_state.value}, {self.elapsed_ms:.1f}ms)"
        )


class IterationStateMachine:
    """
    Manages state transitions for a single iteration.

    Enforces the valid transition graph and records timing
    information for each state.

    Usage:
        sm = IterationStateMachine()
        sm.transition(IterationState.PROBE)
        sm.transition(IterationState.GENERATE)
        ...
        sm.transition(IterationState.FEEDBACK)

        # Inspect timing
        for t in sm.history:
            print(t)
    """

    def __init__(self) -> None:
        self._state: IterationState = IterationState.IDLE
        self._history: list[StateTransition] = []
        self._state_enter_time: float = time.time()
        self._transition_count: int = 0

    @property
    def state(self) -> IterationState:
        """Current state."""
        return self._state

    @property
    def history(self) -> list[StateTransition]:
        """List of all transitions."""
        return list(self._history)

    @property
    def transition_count(self) -> int:
        """Total number of transitions made."""
        return self._transition_count

    def transition(self, to_state: IterationState) -> StateTransition:
        """
        Transition to a new state.

        Args:
            to_state: The target state.

        Returns:
            StateTransition record.

        Raises:
            InvalidTransitionError: If the transition is invalid.
        """
        from_state = self._state

        # Validate transition
        valid_targets = VALID_TRANSITIONS.get(from_state, set())
        if to_state not in valid_targets:
            raise InvalidTransitionError(
                f"Cannot transition from {from_state.value} "
                f"to {to_state.value}. "
                f"Valid targets: {[s.value for s in valid_targets]}"
            )

        now = time.time()
        elapsed_ms = (now - self._state_enter_time) * 1000

        record = StateTransition(
            from_state=from_state,
            to_state=to_state,
            timestamp=now,
            elapsed_ms=elapsed_ms,
        )

        self._history.append(record)
        self._state = to_state
        self._state_enter_time = now
        self._transition_count += 1

        logger.debug(
            "State: %s → %s (%.1fms)",
            from_state.value, to_state.value, elapsed_ms,
        )

        return record

    def reset(self) -> None:
        """Reset state machine to IDLE."""
        self._state = IterationState.IDLE
        self._history.clear()
        self._state_enter_time = time.time()
        self._transition_count = 0

    def get_step_timing(self) -> dict[str, float]:
        """
        Get total time spent in each state (ms).

        Returns:
            Dict mapping state name → total milliseconds.
        """
        timing: dict[str, float] = {}
        for t in self._history:
            key = t.from_state.value
            timing[key] = timing.get(key, 0.0) + t.elapsed_ms
        return timing

    def get_slowest_step(self) -> tuple[str, float] | None:
        """
        Find the step that took the most time.

        Returns:
            (state_name, elapsed_ms) or None if no history.
        """
        timing = self.get_step_timing()
        if not timing:
            return None
        slowest = max(timing.items(), key=lambda kv: kv[1])
        return slowest

    def get_average_step_time(self) -> float:
        """Average time per transition in ms."""
        if not self._history:
            return 0.0
        total = sum(t.elapsed_ms for t in self._history)
        return total / len(self._history)

    def is_terminal(self) -> bool:
        """Whether current state is terminal (COMPLETE)."""
        return self._state == IterationState.COMPLETE

    def has_visited(self, state: IterationState) -> bool:
        """Check if a state was visited during this iteration."""
        if self._state == state:
            return True
        return any(
            t.from_state == state or t.to_state == state
            for t in self._history
        )

    def __repr__(self) -> str:
        return (
            f"IterationStateMachine(state={self._state.value}, "
            f"transitions={self._transition_count})"
        )


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


# ─── Batch State Machine for Multi-Iteration Tracking ─────────

class IterationBatchTracker:
    """
    Tracks state machine statistics across many iterations.

    Aggregates timing data and transition patterns to
    identify bottlenecks in the iteration pipeline.
    """

    def __init__(self) -> None:
        self._iteration_timings: list[dict[str, float]] = []
        self._iteration_times: list[float] = []
        self._error_count: int = 0

    def record_iteration(
        self,
        sm: IterationStateMachine,
        wall_time: float,
    ) -> None:
        """Record timing from a completed iteration."""
        self._iteration_timings.append(sm.get_step_timing())
        self._iteration_times.append(wall_time)

    def record_error(self) -> None:
        """Record an iteration that errored."""
        self._error_count += 1

    @property
    def total_iterations(self) -> int:
        return len(self._iteration_timings)

    @property
    def error_rate(self) -> float:
        total = self.total_iterations + self._error_count
        if total == 0:
            return 0.0
        return self._error_count / total

    def get_average_step_timing(self) -> dict[str, float]:
        """
        Get average time per step across all iterations.

        Returns:
            Dict mapping state name → average milliseconds.
        """
        if not self._iteration_timings:
            return {}

        aggregated: dict[str, list[float]] = {}
        for timing in self._iteration_timings:
            for step, ms in timing.items():
                aggregated.setdefault(step, []).append(ms)

        return {
            step: sum(vals) / len(vals)
            for step, vals in aggregated.items()
        }

    def get_bottleneck(self) -> tuple[str, float] | None:
        """Identify the average slowest step across iterations."""
        avg = self.get_average_step_timing()
        if not avg:
            return None
        return max(avg.items(), key=lambda kv: kv[1])

    def get_throughput(self) -> float:
        """
        Iterations per second.
        """
        if not self._iteration_times:
            return 0.0
        total_time = sum(self._iteration_times)
        if total_time == 0:
            return 0.0
        return len(self._iteration_times) / total_time

    def summary(self) -> dict[str, Any]:
        """
        Summary report of batch iteration performance.

        Returns:
            Dict with average timings, bottleneck, throughput.
        """
        bottleneck = self.get_bottleneck()
        return {
            "total_iterations": self.total_iterations,
            "error_count": self._error_count,
            "error_rate": self.error_rate,
            "avg_wall_time_s": (
                sum(self._iteration_times) / len(self._iteration_times)
                if self._iteration_times else 0.0
            ),
            "throughput_per_s": self.get_throughput(),
            "bottleneck_step": bottleneck[0] if bottleneck else None,
            "bottleneck_avg_ms": bottleneck[1] if bottleneck else None,
            "avg_step_timing_ms": self.get_average_step_timing(),
        }
