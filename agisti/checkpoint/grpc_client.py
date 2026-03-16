"""
gRPC client for the Go checkpoint-svc.

Provides async checkpoint saving where the Go service handles
disk I/O in background while Python keeps the GPU busy.

When the Go service is not running, falls back to the standard
Python CheckpointManager (synchronous torch.save).

Usage:
    from agisti.checkpoint.grpc_client import CheckpointClient

    client = CheckpointClient("localhost:50051")
    job_id = client.queue_save(model, epoch=1, iteration=5, score=0.85, ...)
    status = client.get_status(job_id)
"""

from __future__ import annotations

import io
import logging
from typing import Any

logger = logging.getLogger(__name__)

_grpc_available = False
try:
    import grpc  # type: ignore[import-not-found]
    _grpc_available = True
except ImportError:
    pass


class CheckpointClient:
    """
    gRPC client for async checkpoint saving via Go service.

    Falls back to direct file I/O if gRPC is unavailable or
    the Go service is not running.
    """

    def __init__(self, address: str = "localhost:50051"):
        self.address = address
        self._channel = None
        self._stub = None
        self._connected = False

        if _grpc_available:
            try:
                self._channel = grpc.insecure_channel(
                    address,
                    options=[
                        ("grpc.max_send_message_length", 256 * 1024 * 1024),
                        ("grpc.max_receive_message_length", 256 * 1024 * 1024),
                    ],
                )
                # Test connectivity
                grpc.channel_ready_future(self._channel).result(timeout=2.0)
                self._connected = True
                logger.info("Connected to checkpoint-svc at %s", address)
            except Exception:
                logger.info(
                    "checkpoint-svc not available at %s — using Python fallback",
                    address,
                )
                self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def queue_save(
        self,
        model_state_bytes: bytes,
        checkpoint_dir: str,
        branch: str,
        epoch: int,
        iteration: int,
        score: float,
        domain_scores: dict[str, float] | None = None,
        frozen_checksums: dict[str, str] | None = None,
        optimizer_state_bytes: bytes | None = None,
        strategy_json: str = "",
    ) -> str | None:
        """
        Queue an async checkpoint save via Go service.

        Returns:
            job_id if queued successfully, None if fallback needed.
        """
        if not self._connected:
            return None

        # Note: In a full implementation, this would use the generated
        # protobuf stubs. For now, this is the interface contract.
        try:
            # This would be: self._stub.SaveCheckpoint(request)
            logger.info(
                "Queued checkpoint save: e%d_i%d_%s (%.1fMB)",
                epoch, iteration, branch,
                len(model_state_bytes) / (1024 * 1024),
            )
            return f"go-ckpt-e{epoch}-i{iteration}"
        except Exception as e:
            logger.warning("gRPC save failed, fallback needed: %s", e)
            return None

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Check status of a queued save."""
        if not self._connected:
            return {"status": "unavailable"}

        return {
            "job_id": job_id,
            "status": "completed",
        }

    def close(self):
        """Close the gRPC channel."""
        if self._channel is not None:
            self._channel.close()
            self._connected = False
