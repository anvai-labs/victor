# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Required internal collaborators for the canonical tool execution runtime.

The compatibility adapter implements these boundaries. Payload dictionaries retain
the existing tool wire shape; they do not grant arbitrary access to runtime state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from victor.agent.services.tool_service import ToolResultContext
    from victor.agent.tool_pipeline import PipelineExecutionResult, ToolCallResult

ToolOutputFormatCallback = Callable[[str, dict[str, Any], Any], str]


class ToolBatchExecution(Protocol):
    """Execute once through the existing pipeline and expose consumed usage."""

    async def execute_tool_pipeline(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        interrupted_results: list[ToolCallResult],
    ) -> PipelineExecutionResult: ...

    def sync_tool_usage(self) -> None: ...


class ToolResultPublication(Protocol):
    """Publish through ToolService with the current session's mutable context."""

    def create_tool_result_context(
        self, *, format_tool_output: ToolOutputFormatCallback, require_persistence: bool
    ) -> ToolResultContext: ...

    def publish_tool_results(
        self, result: PipelineExecutionResult, context: ToolResultContext
    ) -> list[dict[str, Any]]: ...

    def sync_tool_result_state(self, context: ToolResultContext) -> None: ...
