"""
RunPod Integration — launches and manages AGISTI iterations
on RunPod Serverless infrastructure.

Supports both synchronous (single iteration) and batch
(multi-iteration) execution modes. Integrates with the
checkpoint system for remote state management.

Design: §9.4 — RunPod Serverless 통합.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class RunPodJobStatus(str, Enum):
    """Status of a RunPod job."""
    QUEUED = "IN_QUEUE"
    RUNNING = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"


@dataclass
class RunPodEndpoint:
    """RunPod Serverless endpoint configuration."""
    endpoint_id: str
    api_key: str
    base_url: str = "https://api.runpod.ai/v2"
    timeout_seconds: int = 300
    max_retries: int = 3

    @property
    def run_url(self) -> str:
        return f"{self.base_url}/{self.endpoint_id}/run"

    @property
    def runsync_url(self) -> str:
        return f"{self.base_url}/{self.endpoint_id}/runsync"

    @property
    def status_url(self) -> str:
        return f"{self.base_url}/{self.endpoint_id}/status"

    @property
    def cancel_url(self) -> str:
        return f"{self.base_url}/{self.endpoint_id}/cancel"

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


@dataclass
class RunPodJob:
    """Tracks a submitted RunPod job."""
    job_id: str
    endpoint_id: str
    submitted_at: float = field(default_factory=time.time)
    status: RunPodJobStatus = RunPodJobStatus.QUEUED
    result: dict[str, Any] | None = None
    error: str | None = None
    execution_time_ms: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            RunPodJobStatus.COMPLETED,
            RunPodJobStatus.FAILED,
            RunPodJobStatus.CANCELLED,
            RunPodJobStatus.TIMED_OUT,
        }

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.submitted_at


@dataclass
class IterationPayload:
    """
    Payload sent to RunPod for a single iteration.

    Contains the iteration configuration, model reference,
    and checkpoint location.
    """
    iteration_id: int
    epoch: int
    model_name: str
    checkpoint_path: str | None
    strategy: dict[str, Any]
    config: dict[str, Any]
    problems: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": {
                "action": "run_iteration",
                "iteration_id": self.iteration_id,
                "epoch": self.epoch,
                "model_name": self.model_name,
                "checkpoint_path": self.checkpoint_path,
                "strategy": self.strategy,
                "config": self.config,
                "problems": self.problems,
            }
        }


class RunPodClient:
    """
    HTTP client for RunPod API.

    Uses urllib to avoid adding requests dependency.
    Handles authentication, retries, and error parsing.
    """

    def __init__(self, endpoint: RunPodEndpoint) -> None:
        self._endpoint = endpoint

    def submit_async(
        self,
        payload: dict[str, Any],
    ) -> RunPodJob:
        """
        Submit an async job to RunPod.

        Args:
            payload: Job payload dict.

        Returns:
            RunPodJob with job_id.

        Raises:
            RunPodError: If submission fails.
        """
        import urllib.request
        import urllib.error

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint.run_url,
            data=data,
            headers=self._endpoint.headers,
            method="POST",
        )

        for attempt in range(self._endpoint.max_retries):
            try:
                with urllib.request.urlopen(
                    req, timeout=30,
                ) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    job_id = body.get("id", "")
                    return RunPodJob(
                        job_id=job_id,
                        endpoint_id=self._endpoint.endpoint_id,
                    )
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8", errors="replace")
                if attempt < self._endpoint.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "RunPod submit error (attempt %d/%d): %s %s. "
                        "Retrying in %ds...",
                        attempt + 1, self._endpoint.max_retries,
                        e.code, error_body, wait,
                    )
                    time.sleep(wait)
                else:
                    raise RunPodError(
                        f"Failed to submit job after "
                        f"{self._endpoint.max_retries} attempts: "
                        f"{e.code} {error_body}"
                    ) from e
            except urllib.error.URLError as e:
                if attempt < self._endpoint.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "RunPod connection error (attempt %d/%d): %s. "
                        "Retrying in %ds...",
                        attempt + 1, self._endpoint.max_retries,
                        e.reason, wait,
                    )
                    time.sleep(wait)
                else:
                    raise RunPodError(
                        f"RunPod connection failed: {e.reason}"
                    ) from e

        raise RunPodError("Exhausted retries")

    def submit_sync(
        self,
        payload: dict[str, Any],
    ) -> RunPodJob:
        """
        Submit a synchronous job (waits for result).

        Args:
            payload: Job payload dict.

        Returns:
            RunPodJob with result populated.
        """
        import urllib.request
        import urllib.error

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint.runsync_url,
            data=data,
            headers=self._endpoint.headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                req, timeout=self._endpoint.timeout_seconds,
            ) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                job = RunPodJob(
                    job_id=body.get("id", ""),
                    endpoint_id=self._endpoint.endpoint_id,
                    status=RunPodJobStatus(body.get("status", "COMPLETED")),
                    result=body.get("output"),
                    execution_time_ms=body.get("executionTime", 0),
                )
                return job
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RunPodError(
                f"Sync job failed: {e.code} {error_body}"
            ) from e

    def check_status(self, job_id: str) -> RunPodJob:
        """
        Check status of an async job.

        Args:
            job_id: The RunPod job ID.

        Returns:
            Updated RunPodJob.
        """
        import urllib.request
        import urllib.error

        url = f"{self._endpoint.status_url}/{job_id}"
        req = urllib.request.Request(
            url,
            headers=self._endpoint.headers,
            method="GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return RunPodJob(
                    job_id=job_id,
                    endpoint_id=self._endpoint.endpoint_id,
                    status=RunPodJobStatus(body.get("status", "IN_QUEUE")),
                    result=body.get("output"),
                    error=body.get("error"),
                    execution_time_ms=body.get("executionTime", 0),
                )
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RunPodError(
                f"Status check failed: {e.code} {error_body}"
            ) from e

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        import urllib.request
        import urllib.error

        url = f"{self._endpoint.cancel_url}/{job_id}"
        req = urllib.request.Request(
            url,
            headers=self._endpoint.headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("status") == "CANCELLED"
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            logger.error("Failed to cancel job %s: %s", job_id, e)
            return False

    def wait_for_completion(
        self,
        job: RunPodJob,
        poll_interval: float = 2.0,
        timeout: float | None = None,
    ) -> RunPodJob:
        """
        Poll until job is complete.

        Args:
            job: The RunPodJob to wait for.
            poll_interval: Seconds between status checks.
            timeout: Maximum wait time in seconds.

        Returns:
            Updated RunPodJob with result.

        Raises:
            RunPodError: If timeout or error occurs.
        """
        timeout = timeout or self._endpoint.timeout_seconds
        start = time.time()

        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                raise RunPodError(
                    f"Job {job.job_id} timed out after {elapsed:.0f}s"
                )

            updated = self.check_status(job.job_id)

            if updated.is_terminal:
                return updated

            logger.debug(
                "Job %s: %s (%.0fs elapsed)",
                job.job_id, updated.status.value, elapsed,
            )
            time.sleep(poll_interval)


class RunPodOrchestrator:
    """
    High-level orchestrator for running AGISTI on RunPod.

    Manages job submission, batching, result collection,
    and checkpoint synchronization with remote workers.
    """

    def __init__(
        self,
        endpoint: RunPodEndpoint | None = None,
        api_key: str | None = None,
        endpoint_id: str | None = None,
    ) -> None:
        if endpoint is not None:
            self._endpoint = endpoint
        else:
            key = api_key or os.environ.get("RUNPOD_API_KEY", "")
            eid = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID", "")
            if not key or not eid:
                raise RunPodError(
                    "RunPod API key and endpoint ID required. "
                    "Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID "
                    "environment variables."
                )
            self._endpoint = RunPodEndpoint(
                endpoint_id=eid,
                api_key=key,
            )

        self._client = RunPodClient(self._endpoint)
        self._active_jobs: dict[str, RunPodJob] = {}
        self._completed_jobs: list[RunPodJob] = []

    def run_iteration(
        self,
        payload: IterationPayload,
        sync: bool = True,
    ) -> RunPodJob:
        """
        Run a single iteration on RunPod.

        Args:
            payload: IterationPayload with config.
            sync: Whether to wait for completion.

        Returns:
            RunPodJob with result (if sync=True).
        """
        data = payload.to_dict()

        if sync:
            job = self._client.submit_sync(data)
        else:
            job = self._client.submit_async(data)
            self._active_jobs[job.job_id] = job

        if job.is_terminal:
            self._completed_jobs.append(job)

        logger.info(
            "RunPod job %s: %s (iter=%d)",
            job.job_id, job.status.value,
            payload.iteration_id,
        )

        return job

    def run_batch(
        self,
        payloads: list[IterationPayload],
        max_concurrent: int = 4,
        poll_interval: float = 5.0,
    ) -> list[RunPodJob]:
        """
        Run multiple iterations in parallel.

        Submits up to max_concurrent jobs at a time,
        waits for completion, then submits more.

        Args:
            payloads: List of payloads to execute.
            max_concurrent: Max concurrent RunPod jobs.
            poll_interval: Seconds between status polls.

        Returns:
            List of completed RunPodJobs.
        """
        results: list[RunPodJob] = []
        pending: list[IterationPayload] = list(payloads)

        while pending or self._active_jobs:
            # Submit new jobs up to max_concurrent
            while pending and len(self._active_jobs) < max_concurrent:
                payload = pending.pop(0)
                job = self.run_iteration(payload, sync=False)
                logger.info(
                    "Submitted batch job %s (iter=%d), "
                    "%d remaining",
                    job.job_id, payload.iteration_id,
                    len(pending),
                )

            # Poll active jobs
            completed_ids: list[str] = []
            for job_id, job in self._active_jobs.items():
                updated = self._client.check_status(job_id)
                if updated.is_terminal:
                    completed_ids.append(job_id)
                    results.append(updated)
                    self._completed_jobs.append(updated)
                    logger.info(
                        "Batch job %s completed: %s",
                        job_id, updated.status.value,
                    )

            for jid in completed_ids:
                del self._active_jobs[jid]

            if self._active_jobs:
                time.sleep(poll_interval)

        return results

    def cancel_all(self) -> int:
        """Cancel all active jobs."""
        cancelled = 0
        for job_id in list(self._active_jobs.keys()):
            if self._client.cancel_job(job_id):
                cancelled += 1
                del self._active_jobs[job_id]
        return cancelled

    @property
    def active_job_count(self) -> int:
        return len(self._active_jobs)

    @property
    def completed_job_count(self) -> int:
        return len(self._completed_jobs)

    def summary(self) -> dict[str, Any]:
        """Summary of all jobs."""
        successes = sum(
            1 for j in self._completed_jobs
            if j.status == RunPodJobStatus.COMPLETED
        )
        failures = sum(
            1 for j in self._completed_jobs
            if j.status == RunPodJobStatus.FAILED
        )
        return {
            "active": len(self._active_jobs),
            "completed": len(self._completed_jobs),
            "successes": successes,
            "failures": failures,
            "total_execution_time_ms": sum(
                j.execution_time_ms for j in self._completed_jobs
            ),
        }


class RunPodError(Exception):
    """Raised when RunPod API operations fail."""
    pass
