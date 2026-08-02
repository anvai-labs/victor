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

"""Service-owned runtime for streaming chunk generation.

This module hosts the canonical ChunkGenerator implementation on the
service-first side of the architecture.

The legacy ``victor.agent.chunk_generator`` module now re-exports this
implementation for compatibility.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.core.events import ObservabilityBus
from victor.providers.base import StreamChunk

if TYPE_CHECKING:
    from victor.agent.streaming import StreamingChatContext, StreamingChatHandler
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


class ChunkGenerator:
    """Generates streaming chunks for various purposes.

    This component provides a semantic interface for chunk generation operations,
    consolidating all chunk-related methods that were previously scattered across
    AgentOrchestrator.

    Architecture:
    - ChunkGenerator: High-level semantic interface for chunk operations
    - StreamingChatHandler: Low-level chunk creation and formatting
    - AgentOrchestrator: Uses ChunkGenerator for all streaming output

    Responsibilities:
    - Generate tool-related chunks (start, result)
    - Generate status chunks (thinking, budget errors, force response)
    - Generate content chunks (metrics, content, final markers)
    - Provide clean, semantic API for streaming operations

    Design Pattern:
    - Coordinator/Facade: Simplifies chunk generation interface
    - Delegation: Delegates to StreamingChatHandler for implementation

    Extracted from CRITICAL-001 Phase 2B.
    """

    def __init__(
        self,
        streaming_handler: "StreamingChatHandler",
        settings: "Settings",
        event_bus: Optional[ObservabilityBus] = None,
    ):
        """Initialize ChunkGenerator.

        Args:
            streaming_handler: Handler for streaming chunk generation
            settings: Application settings
            event_bus: Optional ObservabilityBus instance. If None, uses DI container.
        """
        self.streaming_handler = streaming_handler
        self.settings = settings
        self._event_bus = event_bus or self._get_default_bus()

    def _get_default_bus(self) -> Optional[ObservabilityBus]:
        """Get default ObservabilityBus from DI container.

        Returns:
            ObservabilityBus instance or None if unavailable
        """
        try:
            from victor.core.events import get_observability_bus

            return get_observability_bus()
        except Exception:
            return None

    # =====================================================================
    # Tool-Related Chunks
    # =====================================================================

    def generate_tool_start_chunk(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        status_msg: str,
        tool_call_id: Optional[str] = None,
    ) -> StreamChunk:
        """Generate chunk indicating tool execution start.

        Args:
            tool_name: Name of the tool being executed
            tool_args: Tool arguments
            status_msg: Status message to display
            tool_call_id: Provider tool_calls[].id — forwarded so renderers can
                track N concurrent tools independently (parallel batches)

        Returns:
            StreamChunk with tool start metadata
        """
        if self._event_bus:
            try:
                from victor.core.events.emit_helper import emit_event_sync

                emit_event_sync(
                    self._event_bus,
                    topic="lifecycle.chunk.tool_start",
                    data={
                        "tool_name": tool_name,
                        "status_msg": status_msg,
                        "category": "lifecycle",
                    },
                    source="ChunkGenerator",
                )
            except Exception as e:
                logger.debug(f"Failed to emit chunk tool start event: {e}")
        return self.streaming_handler.generate_tool_start_chunk(
            tool_name, tool_args, status_msg, tool_call_id=tool_call_id
        )

    def generate_tool_result_chunks(
        self,
        result: Dict[str, Any],
    ) -> List[StreamChunk]:
        """Generate chunks for tool execution result."""
        return self.streaming_handler.generate_tool_result_chunks(result)

    # =====================================================================
    # Status Chunks
    # =====================================================================

    def generate_thinking_status_chunk(self) -> StreamChunk:
        """Generate chunk indicating thinking/processing status."""
        return self.streaming_handler.generate_thinking_status_chunk()

    def generate_budget_error_chunk(self) -> StreamChunk:
        """Generate chunk for budget limit error."""
        return self.streaming_handler.generate_budget_error_chunk()

    def generate_force_response_error_chunk(self) -> StreamChunk:
        """Generate chunk for forced response error."""
        return self.streaming_handler.generate_force_response_error_chunk()

    def generate_final_marker_chunk(self) -> StreamChunk:
        """Generate final marker chunk to signal stream completion."""
        if self._event_bus:
            try:
                from victor.core.events.emit_helper import emit_event_sync

                emit_event_sync(
                    self._event_bus,
                    topic="lifecycle.chunk.stream_complete",
                    data={"category": "lifecycle"},
                    source="ChunkGenerator",
                )
            except Exception as e:
                logger.debug(f"Failed to emit stream complete event: {e}")
        return self.streaming_handler.generate_final_marker_chunk()

    # =====================================================================
    # Content Chunks
    # =====================================================================

    def generate_metrics_chunk(
        self,
        metrics_line: str,
        is_final: bool = False,
        prefix: str = "\n\n",
    ) -> StreamChunk:
        """Generate chunk for metrics display."""
        if self._event_bus:
            try:
                from victor.core.events.emit_helper import emit_event_sync

                emit_event_sync(
                    self._event_bus,
                    topic="metric.chunk.metrics_generated",
                    data={
                        "metrics_line": metrics_line[:200],
                        "is_final": is_final,
                        "category": "metric",
                    },
                    source="ChunkGenerator",
                )
            except Exception as e:
                logger.debug(f"Failed to emit metrics generated event: {e}")
        return self.streaming_handler.generate_metrics_chunk(
            metrics_line, is_final=is_final, prefix=prefix
        )

    def generate_content_chunk(
        self,
        content: str,
        is_final: bool = False,
        suffix: str = "",
    ) -> StreamChunk:
        """Generate chunk for content display."""
        return self.streaming_handler.generate_content_chunk(
            content, is_final=is_final, suffix=suffix
        )

    # =====================================================================
    # Budget Chunks
    # =====================================================================

    def get_budget_exhausted_chunks(
        self,
        stream_ctx: "StreamingChatContext",
    ) -> List[StreamChunk]:
        """Get chunks for budget exhaustion warning."""
        return self.streaming_handler.get_budget_exhausted_chunks(stream_ctx)

    # =====================================================================
    # Metrics Formatting
    # =====================================================================

    def format_completion_metrics(
        self,
        stream_ctx: "StreamingChatContext",
        elapsed_time: float,
        cost_str: Optional[str] = None,
    ) -> str:
        """Format performance metrics for normal completion."""
        return self.streaming_handler.format_completion_metrics(stream_ctx, elapsed_time, cost_str)

    def format_budget_exhausted_metrics(
        self,
        stream_ctx: "StreamingChatContext",
        elapsed_time: float,
        time_to_first_token: Optional[float] = None,
        cost_str: Optional[str] = None,
    ) -> str:
        """Format performance metrics for budget exhausted completion."""
        return self.streaming_handler.format_budget_exhausted_metrics(
            stream_ctx, elapsed_time, time_to_first_token, cost_str
        )


__all__ = ["ChunkGenerator"]
