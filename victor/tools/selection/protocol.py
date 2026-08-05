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

"""Tool Selection Strategy Protocol.

This module defines the protocol interface for tool selection strategies,
enabling pluggable selection algorithms across verticals.

The tool selection system supports three strategies:
- **Keyword**: Fast registry-based matching (<1ms)
- **Semantic**: ML-based embedding similarity (10-50ms)
- **Hybrid**: Blends both approaches (best of both worlds)

Example:
    from victor.tools.selection import (
        ToolSelectionStrategy,
        ToolSelectionContext,
        PerformanceProfile,
    )

    class CustomSelector(ToolSelectionStrategy):
        def get_strategy_name(self) -> str:
            return "custom"

        def get_performance_profile(self) -> PerformanceProfile:
            return PerformanceProfile(
                avg_latency_ms=5.0,
                requires_embeddings=False,
                requires_model_inference=False,
                memory_usage_mb=10.0,
            )

        async def select_tools(
            self,
            context: ToolSelectionContext,
            max_tools: int = 10,
        ) -> List[str]:
            # Custom selection logic
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.agent.conversation.state_machine import ConversationStage
    from victor.agent.task_classifier import ClassificationResult


@dataclass
class PerformanceProfile:
    """Performance characteristics of a selection strategy.

    Used for strategy selection based on latency/resource requirements.

    Attributes:
        avg_latency_ms: Average selection latency in milliseconds
        requires_embeddings: Whether strategy needs embedding service
        requires_model_inference: Whether strategy needs LLM inference
        memory_usage_mb: Approximate memory usage in megabytes
    """

    avg_latency_ms: float
    requires_embeddings: bool
    requires_model_inference: bool
    memory_usage_mb: float


@dataclass
class CrossVerticalToolSelectionContext:
    """Extended context for cross-vertical tool selection decisions.

    Renamed from ToolSelectionContext to be semantically distinct:
    - CrossVerticalToolSelectionContext (here): Extended cross-vertical selection
    - AgentToolSelectionContext (victor.agent.protocols): Basic agent-level context
    - VerticalToolSelectionContext (victor.core.verticals.protocols.tool_provider): Vertical-specific

    Provides comprehensive context for intelligent tool filtering,
    including conversation history, stage info, and task metadata.

    Attributes:
        prompt: Current user prompt/message
        conversation_history: List of prior conversation messages
        current_stage: Conversation stage name (PLANNING, EXECUTING, etc.)
        task_type: Type of task (analysis, implementation, etc.)
        provider_name: LLM provider name (anthropic, openai, etc.)
        model_name: Model name (claude-3, gpt-4, etc.)
        cost_budget: Remaining cost budget for tools
        enabled_tools: Specific tools to enable (None = all)
        disabled_tools: Specific tools to disable
        vertical: Vertical context (coding, devops, research, etc.)
        recent_tools: Recently used tool names
        turn_number: Current conversation turn
        classification_result: Task classification metadata
        max_tools: Maximum number of tools to select
        use_cost_aware: Whether to optimize for cost
    """

    prompt: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_stage: Optional[str] = None
    task_type: Optional[str] = None
    provider_name: str = ""
    model_name: str = ""
    cost_budget: float = 1.0
    enabled_tools: Optional[List[str]] = None
    disabled_tools: Optional[List[str]] = None
    vertical: Optional[str] = None
    recent_tools: List[str] = field(default_factory=list)
    turn_number: int = 0
    classification_result: Optional[Any] = None  # ClassificationResult
    max_tools: int = 10
    use_cost_aware: bool = False

    @classmethod
    def from_agent_context(
        cls,
        prompt: str,
        agent_context: Dict[str, Any],
    ) -> "CrossVerticalToolSelectionContext":
        """Create from agent's internal context dictionary.

        Args:
            prompt: User prompt
            agent_context: Agent's _exec_ctx or similar context dict

        Returns:
            ToolSelectionContext populated from agent context
        """
        return cls(
            prompt=prompt,
            conversation_history=agent_context.get("conversation_history", []),
            current_stage=agent_context.get("stage"),
            task_type=agent_context.get("task_type"),
            provider_name=agent_context.get("provider_name", ""),
            model_name=agent_context.get("model_name", ""),
            cost_budget=agent_context.get("cost_budget", 1.0),
            enabled_tools=agent_context.get("enabled_tools"),
            disabled_tools=agent_context.get("disabled_tools"),
            vertical=agent_context.get("vertical"),
            recent_tools=agent_context.get("recent_tools", []),
            turn_number=agent_context.get("turn_number", 0),
            classification_result=agent_context.get("classification_result"),
            max_tools=agent_context.get("max_tools", 10),
            use_cost_aware=agent_context.get("use_cost_aware", False),
        )


# Backward compatibility alias
ToolSelectionContext = CrossVerticalToolSelectionContext


@dataclass
class ToolSelectorFeatures:
    """Features supported by a tool selector implementation.

    Used to advertise capabilities and for feature detection.

    Attributes:
        supports_semantic_matching: ML-based semantic matching
        supports_context_awareness: Context-aware selection
        supports_cost_optimization: Cost-based tool optimization
        supports_usage_learning: Learning from usage patterns
        supports_workflow_patterns: Workflow pattern detection
        requires_embeddings: Whether embeddings are required
    """

    supports_semantic_matching: bool = False
    supports_context_awareness: bool = False
    supports_cost_optimization: bool = False
    supports_usage_learning: bool = False
    supports_workflow_patterns: bool = False
    requires_embeddings: bool = False


@runtime_checkable
class ToolSelectionStrategy(Protocol):
    """Protocol for tool selection strategies.

    All tool selection implementations should implement this protocol
    to enable pluggable selection algorithms.

    The protocol defines:
    - Strategy identification (name, performance profile)
    - Tool selection method
    - Context compatibility check
    - Feature advertisement
    - Lifecycle management (close)

    Example implementation:
        class MyCustomSelector(ToolSelectionStrategy):
            def get_strategy_name(self) -> str:
                return "custom"

            def get_performance_profile(self) -> PerformanceProfile:
                return PerformanceProfile(
                    avg_latency_ms=5.0,
                    requires_embeddings=False,
                    requires_model_inference=False,
                    memory_usage_mb=10.0,
                )

            async def select_tools(
                self,
                context: ToolSelectionContext,
                max_tools: int = 10,
            ) -> List[str]:
                # Selection logic
                return ["read", "write", "shell"]

            def supports_context(self, context: ToolSelectionContext) -> bool:
                return True

            def get_supported_features(self) -> ToolSelectorFeatures:
                return ToolSelectorFeatures()

            async def close(self) -> None:
                pass
    """

    def get_strategy_name(self) -> str:
        """Get the name of this strategy.

        Returns:
            Strategy name (e.g., "keyword", "semantic", "hybrid")
        """
        ...

    def get_performance_profile(self) -> PerformanceProfile:
        """Get performance characteristics.

        Returns:
            PerformanceProfile with latency/resource information
        """
        ...

    async def select_tools(
        self,
        context: ToolSelectionContext,
        max_tools: int = 10,
    ) -> List[str]:
        """Select relevant tools for the context.

        Args:
            context: Selection context with prompt, history, etc.
            max_tools: Maximum number of tools to return

        Returns:
            List of tool names, ordered by relevance
        """
        ...

    def supports_context(self, context: ToolSelectionContext) -> bool:
        """Check if this strategy can handle the context.

        Some strategies may require embeddings or specific providers.

        Args:
            context: Selection context to check

        Returns:
            True if strategy can handle this context
        """
        ...

    def get_supported_features(self) -> ToolSelectorFeatures:
        """Return features supported by this selector.

        Returns:
            ToolSelectorFeatures with capability flags
        """
        ...

    def record_tool_execution(
        self,
        tool_name: str,
        success: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record tool execution for learning (optional).

        Args:
            tool_name: Name of the executed tool
            success: Whether execution succeeded
            context: Additional context about execution
        """
        ...

    async def close(self) -> None:
        """Cleanup resources (e.g., embedding service)."""
        ...


class BaseToolSelectionStrategy(ABC):
    """Abstract base class for tool selection strategies.

    Provides default implementations for optional methods and
    common functionality shared across strategies.

    Subclasses must implement:
    - get_strategy_name()
    - get_performance_profile()
    - select_tools()

    Example:
        class MySelector(BaseToolSelectionStrategy):
            def get_strategy_name(self) -> str:
                return "my_strategy"

            def get_performance_profile(self) -> PerformanceProfile:
                return PerformanceProfile(...)

            async def select_tools(
                self,
                context: ToolSelectionContext,
                max_tools: int = 10,
            ) -> List[str]:
                # Implementation
                ...
    """

    def __init__(self) -> None:
        """Initialize base strategy."""
        self._closed = False

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        ...

    @abstractmethod
    def get_performance_profile(self) -> PerformanceProfile:
        """Get performance characteristics."""
        ...

    @abstractmethod
    async def select_tools(
        self,
        context: ToolSelectionContext,
        max_tools: int = 10,
    ) -> List[str]:
        """Select relevant tools for the context."""
        ...

    def supports_context(self, context: ToolSelectionContext) -> bool:
        """Default: supports all contexts.

        Override in subclass if strategy has requirements.
        """
        return True

    def get_supported_features(self) -> ToolSelectorFeatures:
        """Default: no special features.

        Override in subclass to advertise capabilities.
        """
        return ToolSelectorFeatures()

    def record_tool_execution(
        self,
        tool_name: str,
        success: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Default: no-op for learning.

        Override in subclass to implement usage learning.
        """
        _ = (tool_name, success, context)  # Intentional no-op, mark params as used

    async def close(self) -> None:
        """Default: mark as closed.

        Override in subclass if resources need cleanup.
        """
        self._closed = True


__all__ = [
    "PerformanceProfile",
    "ToolSelectionContext",
    "ToolSelectorFeatures",
    "ToolSelectionStrategy",
    "BaseToolSelectionStrategy",
]
