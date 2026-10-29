"""Protocol definitions for streaming protocols."""

from __future__ import annotations

from typing import (
    Any,
    Optional,
    Protocol,
    runtime_checkable,
)

__all__ = [
    "StreamingControllerProtocol",
    "StreamingRecoveryCoordinatorProtocol",
    "ChunkGeneratorProtocol",
    "StreamingHandlerProtocol",
    "StreamingMetricsCollectorProtocol",
    "StreamingConfidenceMonitorProtocol",
]


@runtime_checkable
class StreamingControllerProtocol(Protocol):
    """Protocol for streaming session management.

    Manages streaming lifecycle and metrics collection.
    """

    def start_session(self, session_id: str) -> Any:
        """Start a new streaming session."""
        ...

    def end_session(self, session_id: str) -> None:
        """End a streaming session."""
        ...

    def get_active_session(self) -> Optional[Any]:
        """Get the currently active session."""
        ...


# NOTE: StreamingRecoveryCoordinatorProtocol and ChunkGeneratorProtocol are
# canonical service-owned runtime protocols in
# victor.agent.services.protocols.runtime_support. StreamingHandlerProtocol,
# StreamingMetricsCollectorProtocol, and StreamingConfidenceMonitorProtocol are
# canonical service-owned runtime protocols in
# victor.agent.services.protocols.infrastructure_runtime. This module
# re-exports all five at the bottom as deprecated compatibility names.


__all__ = [
    # Streaming protocols
    "StreamingControllerProtocol",
    "StreamingRecoveryCoordinatorProtocol",
    "ChunkGeneratorProtocol",
    "StreamingHandlerProtocol",
    "StreamingConfidenceMonitorProtocol",
    "StreamingMetricsCollectorProtocol",
]
from victor.agent.services.protocols.infrastructure_runtime import (
    StreamingConfidenceMonitorProtocol,
    StreamingHandlerProtocol,
    StreamingMetricsCollectorProtocol,
)
from victor.agent.services.protocols.runtime_support import (
    ChunkRuntimeProtocol as ChunkGeneratorProtocol,
    StreamingRecoveryRuntimeProtocol as StreamingRecoveryCoordinatorProtocol,
)

# Alias: some modules reference StreamingCoordinatorProtocol
StreamingCoordinatorProtocol = StreamingControllerProtocol
