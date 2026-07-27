# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Service-owned host for active runtime infrastructure protocols.

These are the canonical runtime-facing protocol surfaces for the service-first
agent runtime. Legacy names under ``victor.agent.protocols`` remain as
compatibility aliases only.
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

__all__ = [
    "IntentClassifierProtocol",
    "ReminderManagerProtocol",
    "ResponseSanitizerProtocol",
    "StreamingConfidenceMonitorProtocol",
    "StreamingHandlerProtocol",
    "StreamingMetricsCollectorProtocol",
]


@runtime_checkable
class IntentClassifierProtocol(Protocol):
    """Protocol for intent classification service."""

    def classify(self, text: str) -> Any:
        """Classify user intent."""
        ...

    def get_confidence(self, text: str, intent: Any) -> float:
        """Get confidence score for a specific intent."""
        ...


@runtime_checkable
class ResponseSanitizerProtocol(Protocol):
    """Protocol for response sanitization."""

    def sanitize(self, response: str) -> str:
        """Sanitize model response."""
        ...


@runtime_checkable
class ReminderManagerProtocol(Protocol):
    """Protocol for context reminder management."""

    def reset(self) -> None:
        """Reset state for a new conversation turn."""
        ...

    def update_state(
        self,
        observed_files: Optional[Set[str]] = None,
        executed_tool: Optional[str] = None,
        tool_calls: Optional[int] = None,
        tool_budget: Optional[int] = None,
        task_complexity: Optional[str] = None,
        task_hint: Optional[str] = None,
        executed_tools: Optional[List[str]] = None,
    ) -> None:
        """Update the current context state."""
        ...

    def add_observed_file(self, file_path: str) -> None:
        """Add a file to the observed files set."""
        ...

    def get_consolidated_reminder(self, force: bool = False) -> Optional[str]:
        """Get a consolidated reminder combining all active reminders."""
        ...


@runtime_checkable
class StreamingHandlerProtocol(Protocol):
    """Protocol for streaming chat handler service."""

    async def handle_stream(
        self,
        stream: AsyncIterator[Any],
        context: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Handle streaming chat response."""
        ...


@runtime_checkable
class StreamingMetricsCollectorProtocol(Protocol):
    """Protocol for streaming metrics collection."""

    def record_chunk(
        self,
        chunk_size: int,
        timestamp: float,
        **metadata: Any,
    ) -> None:
        """Record a streaming chunk."""
        ...

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        ...

    def reset(self) -> None:
        """Reset metrics for new session."""
        ...


@runtime_checkable
class StreamingConfidenceMonitorProtocol(Protocol):
    """Protocol for streaming confidence monitor."""

    def record(self, content: str, completion_tokens: int = 0) -> None:
        """Update internal state from a chunk or turn output."""
        ...

    def should_stop(self) -> bool:
        """Return True if generation should stop now."""
        ...

    def reset(self) -> None:
        """Reset state for a new turn."""
        ...

    async def wrap_stream(self, stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """Wrap an async stream, stopping early when confident."""
        ...
