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

"""Stage Manager - Framework-level conversation stage management.

This module provides a high-level API for managing conversation stages,
promoting the ConversationStateMachine from agent/ to the framework layer.

Design Pattern: Facade + Protocol
================================
StageManager provides a simplified facade over ConversationStateMachine
while exposing key functionality through the StageManagerProtocol for
testability and abstraction.

Key Features:
- Stage detection based on tool usage and content patterns
- Stage-based tool prioritization with configurable boost values
- Custom stage definitions for vertical-specific workflows
- Event emission on stage transitions
- Transition history tracking

Usage:
    from victor.framework.stage_manager import StageManager, StageDefinition

    # Create with default stages
    manager = StageManager()

    # Record tool execution (triggers automatic stage detection)
    manager.record_tool("read", {"file_path": "/src/main.py"})

    # Get current stage and recommended tools
    stage = manager.get_stage()
    tools = manager.get_stage_tools()

    # Custom stage definitions (for verticals)
    custom_stages = {
        "data_loading": StageDefinition(
            name="data_loading",
            display_name="Loading Data",
            keywords=["load", "read", "import", "csv", "json"],
            tools={"read", "pandas_read"},
            order=1,
        ),
    }
    manager = StageManager(custom_stages=custom_stages)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    runtime_checkable,
)

from victor.core.shared_types import ConversationStage
from victor.agent.conversation.state_machine import (
    ConversationStateMachine,
    STAGE_ORDER,
)
from victor_contracts import StageDefinition as SdkStageDefinition
from victor_contracts import (
    normalize_stage_definition as normalize_sdk_stage_definition,
)

if TYPE_CHECKING:
    from victor.observability.hooks import StateHookManager
    from victor.core.events import ObservabilityBus as EventBus

logger = logging.getLogger(__name__)


# =============================================================================
# Stage Definition
# =============================================================================


@dataclass
class StageDefinition:
    """Definition of a conversation stage for vertical customization.

    Allows verticals to define custom stages with specific characteristics
    for their domain. This enables data analysis verticals to have stages
    like "data_loading", "cleaning", "analysis" while coding verticals
    use "reading", "execution", "verification".

    Attributes:
        name: Unique stage identifier (snake_case)
        display_name: Human-readable name for UI
        description: Detailed description of what happens in this stage
        keywords: Keywords that trigger detection of this stage
        tools: Tools recommended for this stage
        order: Ordering for stage progression (lower = earlier)
        can_transition_to: Stages this stage can transition to (None = any)
        min_confidence: Minimum confidence required to transition to this stage
    """

    name: str
    display_name: str = ""
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    tools: Set[str] = field(default_factory=set)
    order: int = 0
    can_transition_to: Optional[Set[str]] = None
    min_confidence: float = 0.5

    def __post_init__(self) -> None:
        """Set defaults after initialization."""
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()

    def to_sdk_definition(self) -> SdkStageDefinition:
        """Convert runtime stage metadata to the SDK definition contract."""
        return SdkStageDefinition(
            name=self.name,
            description=self.description,
            tools=set(self.tools),
            keywords=list(self.keywords),
            next_stages=set(self.can_transition_to or set()),
            min_confidence=self.min_confidence,
        )

    @classmethod
    def from_sdk_definition(
        cls,
        stage: SdkStageDefinition,
        *,
        display_name: str | None = None,
        order: int = 0,
    ) -> "StageDefinition":
        """Create a runtime stage definition from an SDK stage definition."""
        return cls(
            name=stage.name,
            display_name=display_name or stage.name.replace("_", " ").title(),
            description=stage.description,
            keywords=list(stage.keywords),
            tools=set(stage.tools),
            order=order,
            can_transition_to=set(stage.next_stages) or None,
            min_confidence=stage.min_confidence,
        )


StageDefinitionInput = StageDefinition | SdkStageDefinition | Dict[str, Any]


def to_runtime_stage_definition(
    stage_name: str,
    stage: StageDefinitionInput,
) -> StageDefinition:
    """Normalize SDK/runtime stage declarations to the runtime stage model."""
    if isinstance(stage, StageDefinition):
        return stage

    if isinstance(stage, dict):
        return StageDefinition(
            name=stage.get("name", stage_name),
            display_name=stage.get("display_name", ""),
            description=stage.get("description", ""),
            keywords=list(stage.get("keywords", [])),
            tools=set(stage.get("tools", [])),
            order=int(stage.get("order", 0)),
            can_transition_to=(
                set(stage["can_transition_to"])
                if stage.get("can_transition_to") is not None
                else None
            ),
            min_confidence=float(stage.get("min_confidence", 0.5)),
        )

    normalized = normalize_sdk_stage_definition(stage_name, stage)
    return StageDefinition.from_sdk_definition(
        normalized,
        display_name=getattr(stage, "display_name", None),
        order=int(getattr(stage, "order", 0)),
    )


def to_sdk_stage_definition(
    stage_name: str,
    stage: StageDefinitionInput,
) -> SdkStageDefinition:
    """Normalize runtime/SDK stage declarations to the SDK definition model."""
    if isinstance(stage, StageDefinition):
        return stage.to_sdk_definition()
    return normalize_sdk_stage_definition(stage_name, stage)


def normalize_runtime_stage_definitions(
    stages: Dict[str, StageDefinitionInput],
) -> Dict[str, StageDefinition]:
    """Normalize a stage-definition mapping to runtime stage definitions."""
    return {
        stage_name: to_runtime_stage_definition(stage_name, stage)
        for stage_name, stage in stages.items()
    }


@dataclass
class StageTransition:
    """Record of a stage transition.

    Attributes:
        from_stage: Previous stage name
        to_stage: New stage name
        confidence: Confidence score for the transition
        trigger: What triggered the transition (tool, content, manual)
        timestamp: Unix timestamp of transition
        context: Additional context about the transition
    """

    from_stage: str
    to_stage: str
    confidence: float
    trigger: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Stage Manager Protocol
# =============================================================================


@runtime_checkable
class StageManagerProtocol(Protocol):
    """Protocol for stage management operations.

    Defines the contract for stage management that can be satisfied by
    StageManager or mock implementations for testing.
    """

    def get_stage(self) -> ConversationStage:
        """Get current conversation stage.

        Returns:
            Current ConversationStage enum value
        """
        ...

    def get_stage_name(self) -> str:
        """Get current stage name as string.

        Returns:
            Stage name (e.g., "reading", "execution")
        """
        ...

    def get_stage_tools(self) -> Set[str]:
        """Get tools recommended for current stage.

        Returns:
            Set of tool names relevant to current stage
        """
        ...

    def record_tool(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Record tool execution and trigger stage detection.

        Args:
            tool_name: Name of executed tool
            args: Tool arguments
        """
        ...

    def record_message(self, content: str, is_user: bool = True) -> None:
        """Record message and trigger stage detection.

        Args:
            content: Message content
            is_user: Whether message is from user
        """
        ...

    def get_tool_priority_boost(self, tool_name: str) -> float:
        """Get priority boost for tool based on current stage.

        Args:
            tool_name: Name of tool to check

        Returns:
            Boost value (0.0 to 0.2) to add to similarity score
        """
        ...

    def should_include_tool(self, tool_name: str) -> bool:
        """Check if tool is recommended for current stage.

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool is relevant to current or adjacent stages
        """
        ...

    def reset(self) -> None:
        """Reset stage manager to initial state."""
        ...


# =============================================================================
# Stage Manager Configuration
# =============================================================================


@dataclass
class StageManagerConfig:
    """Configuration for StageManager.

    Attributes:
        track_history: Whether to track transition history
        max_history_size: Maximum transitions to keep in history
        transition_cooldown: Minimum seconds between transitions
        stage_tool_boost: Boost value for stage-relevant tools
        adjacent_tool_boost: Boost value for adjacent-stage tools
        backward_confidence_threshold: Min confidence for backward transitions
    """

    track_history: bool = True
    max_history_size: int = 100
    transition_cooldown: float = 2.0
    stage_tool_boost: float = 0.15
    adjacent_tool_boost: float = 0.08
    backward_confidence_threshold: float = 0.85


# =============================================================================
# Stage Manager
# =============================================================================


class StageManager:
    """Framework-level stage management facade.

    Provides a high-level API for managing conversation stages, wrapping
    the ConversationStateMachine with additional features:

    - Custom stage definitions for vertical customization
    - Configurable tool priority boosts
    - Extended transition history
    - Event bus integration

    This class implements StageManagerProtocol for testability.
    """

    def __init__(
        self,
        config: Optional[StageManagerConfig] = None,
        custom_stages: Optional[Dict[str, StageDefinitionInput]] = None,
        hooks: Optional["StateHookManager"] = None,
        event_bus: Optional["EventBus"] = None,
    ) -> None:
        """Initialize StageManager.

        Args:
            config: Configuration for stage management
            custom_stages: Optional custom stage definitions
            hooks: Optional StateHookManager for transition callbacks
            event_bus: Optional EventBus for event emission
        """
        self._config = config or StageManagerConfig()
        self._custom_stages = normalize_runtime_stage_definitions(custom_stages or {})

        # Create underlying state machine
        self._machine = ConversationStateMachine(
            hooks=hooks,
            track_history=self._config.track_history,
            max_history_size=self._config.max_history_size,
            event_bus=event_bus,
        )

        # Apply config to machine
        self._machine.TRANSITION_COOLDOWN_SECONDS = self._config.transition_cooldown
        self._machine.BACKWARD_TRANSITION_THRESHOLD = self._config.backward_confidence_threshold

        # Build custom stage tool mapping
        self._custom_stage_tools: Dict[str, Set[str]] = {
            name: defn.tools for name, defn in self._custom_stages.items()
        }

    @property
    def config(self) -> StageManagerConfig:
        """Get the configuration."""
        return self._config

    @property
    def machine(self) -> ConversationStateMachine:
        """Get the underlying state machine (escape hatch)."""
        return self._machine

    # =========================================================================
    # StageManagerProtocol Implementation
    # =========================================================================

    def get_stage(self) -> ConversationStage:
        """Get current conversation stage.

        Returns:
            Current ConversationStage enum value
        """
        return self._machine.get_stage()

    def get_stage_name(self) -> str:
        """Get current stage name as string.

        Returns:
            Stage name in lowercase (e.g., "reading", "execution")
        """
        return self._machine.get_stage().name.lower()

    def get_stage_tools(self) -> Set[str]:
        """Get tools recommended for current stage.

        First checks custom stage definitions, then falls back to
        the standard stage-to-tool mapping from metadata registry.

        Returns:
            Set of tool names relevant to current stage
        """
        stage_name = self.get_stage_name()

        # Check custom stages first
        if stage_name in self._custom_stage_tools:
            return self._custom_stage_tools[stage_name]

        # Fall back to standard mapping
        return self._machine.get_stage_tools()

    def record_tool(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Record tool execution and trigger stage detection.

        Args:
            tool_name: Name of executed tool
            args: Tool arguments
        """
        self._machine.record_tool_execution(tool_name, args)

    def record_message(self, content: str, is_user: bool = True) -> None:
        """Record message and trigger stage detection.

        Args:
            content: Message content
            is_user: Whether message is from user
        """
        self._machine.record_message(content, is_user)

    def get_tool_priority_boost(self, tool_name: str) -> float:
        """Get priority boost for tool based on current stage.

        Uses configured boost values instead of hardcoded defaults.

        Args:
            tool_name: Name of tool to check

        Returns:
            Boost value (0.0 to configured max) to add to similarity score
        """
        # Check current stage tools
        if tool_name in self.get_stage_tools():
            return self._config.stage_tool_boost

        # Check adjacent stage tools
        current_stage = self.get_stage()
        current_idx = STAGE_ORDER[current_stage]

        for stage in ConversationStage:
            if abs(STAGE_ORDER[stage] - current_idx) == 1:
                # Check custom stages first
                stage_name = stage.name.lower()
                stage_tools = self._custom_stage_tools.get(
                    stage_name, self._machine._get_tools_for_stage(stage)
                )
                if tool_name in stage_tools:
                    return self._config.adjacent_tool_boost

        return 0.0

    def should_include_tool(self, tool_name: str) -> bool:
        """Check if tool is recommended for current stage.

        Includes tools from current stage and adjacent stages for
        flexibility.

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool is relevant to current or adjacent stages
        """
        return self._machine.should_include_tool(tool_name)

    def reset(self) -> None:
        """Reset stage manager to initial state."""
        self._machine.reset()

    # =========================================================================
    # Extended API
    # =========================================================================

    def force_stage(self, stage: ConversationStage, confidence: float = 0.9) -> None:
        """Force transition to a specific stage.

        Use sparingly - prefer automatic detection. Useful for:
        - Testing specific stage behaviors
        - Recovering from incorrect stage detection
        - Vertical-specific stage forcing

        Args:
            stage: Stage to transition to
            confidence: Confidence level for the transition
        """
        self._machine._transition_to(stage, confidence)

    def get_state_summary(self) -> Dict[str, Any]:
        """Get detailed summary of current state.

        Returns:
            Dictionary with stage, confidence, history, and recommended tools
        """
        return self._machine.get_state_summary()

    def get_transition_history(self) -> List[Dict[str, Any]]:
        """Get the transition history.

        Returns:
            List of transition records
        """
        return self._machine.transition_history

    def get_transitions_summary(self) -> Dict[str, Any]:
        """Get summary statistics of transitions.

        Returns:
            Dictionary with transition counts, paths, and average confidence
        """
        return self._machine.get_transitions_summary()

    def set_hooks(self, hooks: "StateHookManager") -> None:
        """Set or replace the hook manager.

        Args:
            hooks: StateHookManager instance
        """
        self._machine.set_hooks(hooks)

    def clear_hooks(self) -> None:
        """Remove all hooks."""
        self._machine.clear_hooks()

    # =========================================================================
    # Custom Stage Management
    # =========================================================================

    def register_stage(self, definition: StageDefinitionInput) -> None:
        """Register a custom stage definition.

        Args:
            definition: Runtime or SDK stage definition to register
        """
        definition = to_runtime_stage_definition(getattr(definition, "name", ""), definition)
        self._custom_stages[definition.name] = definition
        self._custom_stage_tools[definition.name] = definition.tools

    def get_stage_definition(self, stage_name: str) -> Optional[StageDefinition]:
        """Get custom stage definition by name.

        Args:
            stage_name: Name of stage to look up

        Returns:
            StageDefinition if found, None otherwise
        """
        return self._custom_stages.get(stage_name)

    def get_all_stage_definitions(self) -> Dict[str, StageDefinition]:
        """Get all registered custom stage definitions.

        Returns:
            Dictionary mapping stage names to definitions
        """
        return dict(self._custom_stages)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize stage manager state.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "machine": self._machine.to_dict(),
            "config": {
                "track_history": self._config.track_history,
                "max_history_size": self._config.max_history_size,
                "transition_cooldown": self._config.transition_cooldown,
                "stage_tool_boost": self._config.stage_tool_boost,
                "adjacent_tool_boost": self._config.adjacent_tool_boost,
                "backward_confidence_threshold": self._config.backward_confidence_threshold,
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        custom_stages: Optional[Dict[str, StageDefinitionInput]] = None,
    ) -> "StageManager":
        """Restore stage manager from dictionary.

        Args:
            data: Dictionary from to_dict()
            custom_stages: Optional custom stage definitions

        Returns:
            Restored StageManager instance
        """
        config_data = data.get("config", {})
        config = StageManagerConfig(
            track_history=config_data.get("track_history", True),
            max_history_size=config_data.get("max_history_size", 100),
            transition_cooldown=config_data.get("transition_cooldown", 2.0),
            stage_tool_boost=config_data.get("stage_tool_boost", 0.15),
            adjacent_tool_boost=config_data.get("adjacent_tool_boost", 0.08),
            backward_confidence_threshold=config_data.get("backward_confidence_threshold", 0.85),
        )

        manager = cls(config=config, custom_stages=custom_stages)

        # Restore machine state
        if "machine" in data:
            manager._machine = ConversationStateMachine.from_dict(data["machine"])
            # Re-apply config
            manager._machine.TRANSITION_COOLDOWN_SECONDS = config.transition_cooldown
            manager._machine.BACKWARD_TRANSITION_THRESHOLD = config.backward_confidence_threshold

        return manager


# =============================================================================
# Factory Functions
# =============================================================================


def create_stage_manager(
    config: Optional[StageManagerConfig] = None,
    custom_stages: Optional[Dict[str, StageDefinitionInput]] = None,
    hooks: Optional["StateHookManager"] = None,
    event_bus: Optional["EventBus"] = None,
) -> StageManager:
    """Factory function to create a StageManager.

    Args:
        config: Optional configuration
        custom_stages: Optional custom stage definitions
        hooks: Optional StateHookManager
        event_bus: Optional EventBus

    Returns:
        Configured StageManager instance
    """
    return StageManager(
        config=config,
        custom_stages=custom_stages,
        hooks=hooks,
        event_bus=event_bus,
    )


# =============================================================================
# Standard Stage Definitions
# =============================================================================


def get_coding_stages() -> Dict[str, StageDefinition]:
    """Get standard stage definitions for coding vertical.

    Returns:
        Dictionary of stage definitions for coding tasks
    """
    return {
        "initial": StageDefinition(
            name="initial",
            display_name="Initial",
            description="First interaction, understanding the request",
            keywords=["what", "how", "where", "explain", "help"],
            tools={"overview", "ls", "search"},
            order=0,
        ),
        "planning": StageDefinition(
            name="planning",
            display_name="Planning",
            description="Understanding scope, planning approach",
            keywords=["plan", "approach", "strategy", "design"],
            tools={"search", "overview", "ls"},
            order=1,
        ),
        "reading": StageDefinition(
            name="reading",
            display_name="Reading",
            description="Reading files, gathering context",
            keywords=["show", "read", "look", "check", "find"],
            tools={"read", "search", "ls", "overview"},
            order=2,
        ),
        "analysis": StageDefinition(
            name="analysis",
            display_name="Analysis",
            description="Analyzing code, understanding structure",
            keywords=["analyze", "review", "examine", "understand"],
            tools={"read", "search", "overview"},
            order=3,
        ),
        "execution": StageDefinition(
            name="execution",
            display_name="Execution",
            description="Making changes, running commands",
            keywords=["change", "modify", "create", "fix", "implement"],
            tools={"write", "edit", "shell", "execute_code"},
            order=4,
        ),
        "verification": StageDefinition(
            name="verification",
            display_name="Verification",
            description="Testing, validating changes",
            keywords=["test", "verify", "check", "validate", "run"],
            tools={"shell", "read", "execute_code"},
            order=5,
        ),
        "completion": StageDefinition(
            name="completion",
            display_name="Completion",
            description="Summarizing, wrapping up",
            keywords=["done", "finish", "complete", "summarize"],
            tools={"shell"},
            order=6,
        ),
    }


def get_data_analysis_stages() -> Dict[str, StageDefinition]:
    """Get standard stage definitions for data analysis vertical.

    Returns:
        Dictionary of stage definitions for data analysis tasks
    """
    return {
        "initial": StageDefinition(
            name="initial",
            display_name="Initial",
            description="Understanding the analysis request",
            keywords=["analyze", "what", "show", "help"],
            tools={"read", "ls"},
            order=0,
        ),
        "data_loading": StageDefinition(
            name="data_loading",
            display_name="Loading Data",
            description="Loading and reading data files",
            keywords=["load", "read", "import", "csv", "json", "excel"],
            tools={"read", "execute_code"},
            order=1,
        ),
        "data_cleaning": StageDefinition(
            name="data_cleaning",
            display_name="Cleaning Data",
            description="Cleaning and preprocessing data",
            keywords=["clean", "preprocess", "missing", "null", "transform"],
            tools={"execute_code"},
            order=2,
        ),
        "analysis": StageDefinition(
            name="analysis",
            display_name="Analysis",
            description="Performing statistical analysis",
            keywords=["analyze", "correlate", "statistics", "mean", "median"],
            tools={"execute_code"},
            order=3,
        ),
        "visualization": StageDefinition(
            name="visualization",
            display_name="Visualization",
            description="Creating charts and visualizations",
            keywords=["plot", "chart", "visualize", "graph", "show"],
            tools={"execute_code"},
            order=4,
        ),
        "completion": StageDefinition(
            name="completion",
            display_name="Completion",
            description="Summarizing findings",
            keywords=["summary", "conclusion", "findings", "done"],
            tools={"write"},
            order=5,
        ),
    }


def get_research_stages() -> Dict[str, StageDefinition]:
    """Get standard stage definitions for research vertical.

    Returns:
        Dictionary of stage definitions for research tasks
    """
    return {
        "initial": StageDefinition(
            name="initial",
            display_name="Initial",
            description="Understanding the research question",
            keywords=["research", "find", "what", "how"],
            tools={"web_search"},
            order=0,
        ),
        "gathering": StageDefinition(
            name="gathering",
            display_name="Gathering Sources",
            description="Collecting relevant sources and information",
            keywords=["search", "find", "sources", "papers", "articles"],
            tools={"web_search", "web_fetch"},
            order=1,
        ),
        "reading": StageDefinition(
            name="reading",
            display_name="Reading Sources",
            description="Reading and extracting information from sources",
            keywords=["read", "extract", "summarize"],
            tools={"web_fetch", "read"},
            order=2,
        ),
        "synthesis": StageDefinition(
            name="synthesis",
            display_name="Synthesis",
            description="Synthesizing information into coherent analysis",
            keywords=["synthesize", "combine", "analyze", "compare"],
            tools={"write"},
            order=3,
        ),
        "completion": StageDefinition(
            name="completion",
            display_name="Completion",
            description="Final report and conclusions",
            keywords=["conclude", "summary", "final", "done"],
            tools={"write"},
            order=4,
        ),
    }


def get_default_stages() -> Dict[str, StageDefinition]:
    """Get generic default stage definitions for any vertical.

    Provides a domain-neutral 7-stage pattern that any vertical can inherit
    and customize. Uses generic tool names (no domain-specific tools like
    ``execute_code`` or ``shell``).

    Returns:
        Dictionary of stage definitions suitable for any vertical
    """
    return {
        "initial": StageDefinition(
            name="initial",
            display_name="Initial",
            description="Understanding the request",
            keywords=["what", "how", "where", "explain", "help"],
            tools={"read", "ls", "search"},
            order=0,
        ),
        "planning": StageDefinition(
            name="planning",
            display_name="Planning",
            description="Planning the approach",
            keywords=["plan", "approach", "strategy", "design"],
            tools={"search", "read", "ls"},
            order=1,
        ),
        "reading": StageDefinition(
            name="reading",
            display_name="Reading",
            description="Gathering context",
            keywords=["show", "read", "look", "check", "find"],
            tools={"read", "search", "ls", "grep"},
            order=2,
        ),
        "analysis": StageDefinition(
            name="analysis",
            display_name="Analysis",
            description="Analyzing information",
            keywords=["analyze", "review", "examine", "understand"],
            tools={"read", "search", "grep"},
            order=3,
        ),
        "execution": StageDefinition(
            name="execution",
            display_name="Execution",
            description="Making changes",
            keywords=["change", "modify", "create", "fix", "implement"],
            tools={"write", "edit", "shell"},
            order=4,
        ),
        "verification": StageDefinition(
            name="verification",
            display_name="Verification",
            description="Validating results",
            keywords=["test", "verify", "check", "validate", "run"],
            tools={"shell", "read"},
            order=5,
        ),
        "completion": StageDefinition(
            name="completion",
            display_name="Completion",
            description="Wrapping up",
            keywords=["done", "finish", "complete", "summarize"],
            tools={"write"},
            order=6,
        ),
    }
