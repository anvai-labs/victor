# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Service-owned structural protocols for streaming runtime helpers.

These protocols host the narrow structural contracts used by the streaming
helper modules under ``victor.agent.streaming`` so the active runtime path
does not need to define protocol surfaces in coordinator-era modules.

Where a canonical protocol already exists, this module re-exports a
service-owned alias. Where the streaming helpers only need a small subset of a
larger dependency, this module defines a focused structural protocol.
"""

from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    runtime_checkable,
)

from victor.agent.services.protocols.infrastructure_runtime import (
    IntentClassifierProtocol as StreamingIntentClassifierRuntimeProtocol,
    ReminderManagerProtocol as StreamingReminderRuntimeProtocol,
    ResponseSanitizerProtocol as StreamingSanitizerRuntimeProtocol,
    StreamingConfidenceMonitorProtocol,
    StreamingHandlerProtocol,
    StreamingMetricsCollectorProtocol,
)
from victor.agent.services.protocols.runtime_support import (
    ChunkRuntimeProtocol as StreamingChunkRuntimeProtocol,
    RLLearningRuntimeProtocol as StreamingRLRuntimeProtocol,
)
from victor.core.protocols import ProviderProtocol as StreamingProviderRuntimeProtocol
from victor.providers.base import StreamChunk

__all__ = [
    "StreamingChunkRuntimeProtocol",
    "StreamingConversationStateProtocol",
    "StreamingExecutionRuntimeProtocol",
    "StreamingIntentClassifierRuntimeProtocol",
    "StreamingMessageAdderProtocol",
    "StreamingProviderRuntimeProtocol",
    "StreamingReminderRuntimeProtocol",
    "StreamingRLRuntimeProtocol",
    "StreamingSanitizerRuntimeProtocol",
    "StreamingTrackerRuntimeProtocol",
    "ToolExecutionRecoveryRuntimeProtocol",
]


@runtime_checkable
class StreamingConversationStateProtocol(Protocol):
    """Narrow conversation-state view needed by streaming intent handling."""

    def get_state_summary(self) -> Dict[str, Any]:
        """Return a summary of the active conversation state."""
        ...


@runtime_checkable
class StreamingMessageAdderProtocol(Protocol):
    """Minimal message-writing contract for streaming helper modules."""

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """Add a message to the active conversation history."""
        ...


@runtime_checkable
class StreamingTrackerRuntimeProtocol(Protocol):
    """Shared unified-tracker contract for streaming helper modules."""

    def check_response_loop(self, content: str) -> bool:
        """Return whether the provided content indicates a repeated loop."""
        ...

    @property
    def config(self) -> Dict[str, Any]:
        """Tracker configuration used by streaming loop logic."""
        ...

    @property
    def unique_resources(self) -> Set[str]:
        """Resources observed during the active streaming turn."""
        ...


@runtime_checkable
class ToolExecutionRecoveryRuntimeProtocol(Protocol):
    """Narrow recovery runtime contract used by streaming tool execution."""

    def check_tool_budget(
        self,
        recovery_ctx: Any,
        warning_threshold: int = 250,
        warning_pct: float = 0.8,
        warning_remaining: int = 5,
    ) -> Optional[StreamChunk]:
        """Return a warning chunk when tool budget limits are reached."""
        ...

    def truncate_tool_calls(
        self,
        recovery_ctx: Any,
        tool_calls: List[Dict[str, Any]],
        remaining: int,
    ) -> Any:
        """Truncate tool calls to the remaining budget."""
        ...

    def filter_blocked_tool_calls(
        self,
        recovery_ctx: Any,
        tool_calls: List[Dict[str, Any]],
    ) -> Any:
        """Filter blocked tool calls and return recovery metadata."""
        ...

    def check_blocked_threshold(
        self,
        recovery_ctx: Any,
        all_blocked: bool,
    ) -> Optional[tuple[StreamChunk, bool]]:
        """Check whether blocked-tool thresholds require early handling."""
        ...


@runtime_checkable
class StreamingExecutionRuntimeProtocol(Protocol):
    """Structural runtime contract consumed by streaming chat executors."""

    _orchestrator: Any
    _intent_classification_handler: Any
    _continuation_handler: Any
    _tool_execution_handler: Any

    async def _create_stream_context(self, user_message: str, **kwargs: Any) -> Any:
        """Create a streaming context for the current turn."""
        ...

    def _run_iteration_pre_checks(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Run pre-checks for the next streaming iteration."""
        ...

    async def _stream_provider_response(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Stream the next provider response."""
        ...

    async def _handle_empty_response_recovery(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Handle recovery when the provider returns an empty response."""
        ...
