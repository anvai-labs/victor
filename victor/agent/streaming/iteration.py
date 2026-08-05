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

"""Iteration result types for streaming chat.

Defines the possible outcomes of each iteration in the streaming loop,
enabling clean control flow and testable iteration logic.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor.providers.base import StreamChunk

if TYPE_CHECKING:
    from victor.agent.presentation import PresentationProtocol


class IterationAction(Enum):
    """Actions that can be taken after an iteration."""

    CONTINUE = auto()  # Continue to next iteration
    BREAK = auto()  # Exit the streaming loop
    YIELD_AND_CONTINUE = auto()  # Yield chunks and continue
    YIELD_AND_BREAK = auto()  # Yield chunks and exit
    FORCE_COMPLETION = auto()  # Force completion on next iteration


@dataclass
class IterationResult:
    """Result of a single iteration in the streaming loop.

    Encapsulates all outputs from processing one iteration,
    making the control flow explicit and testable.
    """

    action: IterationAction
    chunks: List[StreamChunk] = field(default_factory=list)
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    quality_score: float = 0.5
    tokens_used: float = 0.0
    clear_tool_calls: bool = False  # Signal to clear pending tool calls

    @property
    def should_break(self) -> bool:
        """Check if loop should break after this iteration."""
        return self.action in (IterationAction.BREAK, IterationAction.YIELD_AND_BREAK)

    @property
    def should_yield(self) -> bool:
        """Check if there are chunks to yield."""
        return self.action in (
            IterationAction.YIELD_AND_CONTINUE,
            IterationAction.YIELD_AND_BREAK,
        ) or bool(self.chunks)

    @property
    def has_tool_calls(self) -> bool:
        """Check if this iteration produced tool calls."""
        return bool(self.tool_calls)

    @property
    def has_content(self) -> bool:
        """Check if this iteration produced content."""
        return bool(self.content)

    def add_chunk(self, chunk: StreamChunk) -> None:
        """Add a chunk to the result."""
        self.chunks.append(chunk)

    def set_break(self) -> None:
        """Mark that the loop should break."""
        if self.chunks:
            self.action = IterationAction.YIELD_AND_BREAK
        else:
            self.action = IterationAction.BREAK

    def set_continue(self) -> None:
        """Mark that the loop should continue."""
        if self.chunks:
            self.action = IterationAction.YIELD_AND_CONTINUE
        else:
            self.action = IterationAction.CONTINUE


@dataclass
class ProviderResponseResult:
    """Result of streaming from the provider.

    Encapsulates the raw output from the LLM provider streaming.
    """

    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tokens_used: float = 0.0
    garbage_detected: bool = False
    error: Optional[str] = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return bool(self.tool_calls)

    @property
    def has_content(self) -> bool:
        """Check if response contains content."""
        return bool(self.content)


@dataclass
class ToolExecutionResult:
    """Result of executing tool calls.

    Encapsulates the outcomes of tool execution phase.
    """

    results: List[Dict[str, Any]] = field(default_factory=list)
    chunks: List[StreamChunk] = field(default_factory=list)
    all_succeeded: bool = True
    blocked_count: int = 0

    @property
    def has_results(self) -> bool:
        """Check if there are any results."""
        return bool(self.results)

    def add_result(
        self,
        name: str,
        success: bool,
        result: Any = None,
        error: Optional[str] = None,
        elapsed: float = 0.0,
        args: Optional[Dict[str, Any]] = None,
        follow_up_suggestions: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add a tool result."""
        payload = {
            "name": name,
            "success": success,
            "result": result,
            "error": error,
            "elapsed": elapsed,
            "args": args or {},
        }
        if follow_up_suggestions:
            payload["follow_up_suggestions"] = follow_up_suggestions
        self.results.append(payload)
        if not success:
            self.all_succeeded = False


def create_break_result(content: str = "", error: Optional[str] = None) -> IterationResult:
    """Create an iteration result that breaks the loop."""
    result = IterationResult(action=IterationAction.BREAK, content=content, error=error)
    if content:
        result.add_chunk(StreamChunk(content=content))
    return result


def create_continue_result(
    content: str = "",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    tokens: float = 0.0,
) -> IterationResult:
    """Create an iteration result that continues the loop."""
    return IterationResult(
        action=IterationAction.CONTINUE,
        content=content,
        tool_calls=tool_calls or [],
        tokens_used=tokens,
    )


def create_force_completion_result(
    reason: str = "",
    presentation: Optional["PresentationProtocol"] = None,
) -> IterationResult:
    """Create an iteration result that forces completion.

    Args:
        reason: Optional reason for forcing completion.
        presentation: Optional presentation adapter for icons.
            If None, creates default adapter.

    Returns:
        IterationResult configured for forced completion.
    """
    result = IterationResult(action=IterationAction.FORCE_COMPLETION)
    if reason:
        if presentation is None:
            from victor.agent.presentation import create_presentation_adapter

            presentation = create_presentation_adapter()

        warning_icon = presentation.icon("warning", with_color=False)
        result.add_chunk(StreamChunk(content=f"\n\n{warning_icon} {reason}\n"))
    return result
