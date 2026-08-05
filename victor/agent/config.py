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

"""Unified agent configuration system.

This module provides UnifiedAgentConfig, a single configuration class
that consolidates all agent configuration options:

- AgentConfig (victor.framework.config) - Foreground agent configuration
- SubAgentConfig (victor.agent.subagents.base) - Sub-agent configuration
- Background agent parameters - Async task execution configuration

This unified configuration eliminates code proliferation and ensures
consistent configuration across all agent types (Phase 4).

Usage:
    from victor.agent.config import UnifiedAgentConfig

    # Foreground agent
    config = UnifiedAgentConfig(mode="foreground")
    agent = await factory.create_agent(config=config)

    # Background agent
    config = UnifiedAgentConfig(
        mode="background",
        task="Implement feature X",
        tool_budget=100
    )
    agent = await factory.create_agent(config=config)

    # Team member
    config = UnifiedAgentConfig(
        mode="team_member",
        role="researcher",
        capabilities=["search", "analyze"]
    )
    agent = await factory.create_agent(config=config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.framework.config import AgentConfig
    from victor.agent.subagents.base import SubAgentConfig


FrameworkCompatibleAgentConfig = Union["AgentConfig", "UnifiedAgentConfig"]


class AgentLifecycleMode(str, Enum):
    """Agent execution mode.

    Defines how the agent will be created and used:
    - FOREGROUND: Interactive agent for direct user interaction
    - BACKGROUND: Async agent for background task execution
    - TEAM_MEMBER: Agent as part of a multi-agent team
    """

    FOREGROUND = "foreground"
    BACKGROUND = "background"
    TEAM_MEMBER = "team_member"


class ExecutionModeType(str, Enum):
    """Execution mode type for background agents.

    Defines the execution strategy for background tasks:
    - BUILD: Direct code generation and modification
    - PLAN: Create implementation plans without executing
    - EXPLORE: Understand and analyze codebase
    """

    BUILD = "build"  # Direct code generation and modification
    PLAN = "plan"  # Create implementation plans without executing
    EXPLORE = "explore"  # Understand and analyze codebase


@dataclass
class UnifiedAgentConfig:
    """Unified configuration for ALL agent types.

    This single configuration class consolidates:
    - AgentConfig (victor.framework.config)
    - SubAgentConfig (victor.agent.subagents.base)
    - Background agent parameters

    SOLID Principles:
    - SRP: Configuration only, no creation logic
    - OCP: Extensible via mode parameter
    - LSP: Config works with any IAgent implementation
    - ISP: Focused on configuration needs
    - DIP: High-level modules depend on this abstraction

    Attributes:
        # Mode-specific
        mode: Agent execution mode (foreground/background/team_member)

        # Common configuration (all modes)
        provider: LLM provider name
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        tool_budget: Maximum tool calls allowed
        max_iterations: Maximum loop iterations
        enable_observability: Enable event tracking

        # Foreground-specific
        enable_parallel_tools: Allow concurrent tool execution
        max_concurrent_tools: Max concurrent tool executions
        enable_context_compaction: Auto-compact context when needed
        enable_semantic_search: Use embedding-based tool selection
        enable_analytics: Collect usage analytics
        enable_tool_cache: Cache idempotent tool results

        # Background-specific
        task: Task description (required for background agents)
        mode_type: Execution mode (build/plan/explore)
        websocket: Enable websocket support
        timeout_seconds: Execution timeout

        # Team member-specific
        role: Team member role (e.g., "researcher", "executor")
        capabilities: List of agent capabilities
        description: Agent description for team coordination
        allowed_tools: Tools this agent can use
        can_spawn_subagents: Whether agent can create sub-agents

        # Advanced
        extra: Additional provider-specific settings
    """

    # ==========================================================================
    # Mode Selection
    # ==========================================================================

    mode: AgentLifecycleMode = AgentLifecycleMode.FOREGROUND

    # ==========================================================================
    # Common Configuration (All Modes)
    # ==========================================================================

    # Provider configuration
    provider: str = "anthropic"
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096

    # Resource limits
    tool_budget: int = 50
    max_iterations: int = 25

    # Features
    enable_observability: bool = True

    # ==========================================================================
    # Foreground-Specific Configuration
    # ==========================================================================

    # Tool execution
    enable_parallel_tools: bool = True
    max_concurrent_tools: int = 5

    # Context management
    enable_context_compaction: bool = True
    max_context_chars: Optional[int] = None

    # Features
    enable_semantic_search: bool = False
    enable_code_correction: bool = True
    enable_analytics: bool = True

    # Caching
    enable_tool_cache: bool = True
    tool_cache_ttl: int = 600

    # Recovery
    max_recovery_attempts: int = 3

    # Streaming
    streaming_timeout: float = 300.0

    # ==========================================================================
    # Background-Specific Configuration
    # ==========================================================================

    task: Optional[str] = None
    mode_type: ExecutionModeType = ExecutionModeType.BUILD
    websocket: bool = False
    timeout_seconds: int = 300

    # ==========================================================================
    # Team Member-Specific Configuration
    # ==========================================================================

    role: str = "team_member"
    capabilities: List[str] = field(default_factory=list)
    description: str = ""
    allowed_tools: List[str] = field(default_factory=list)
    can_spawn_subagents: bool = False
    context_limit: int = 128000
    working_directory: Optional[str] = None
    system_prompt_override: Optional[str] = None

    # ==========================================================================
    # Advanced Configuration
    # ==========================================================================

    # Additional settings (pass-through to Settings/extra config)
    extra: Dict[str, Any] = field(default_factory=dict)

    # ==========================================================================
    # Migration Helpers
    # ==========================================================================

    @classmethod
    def from_agent_config(cls, config: "AgentConfig") -> "UnifiedAgentConfig":
        """Convert from AgentConfig (foreground agent).

        Args:
            config: AgentConfig instance

        Returns:
            UnifiedAgentConfig with equivalent settings
        """
        return cls(
            mode="foreground",
            tool_budget=config.tool_budget,
            max_iterations=config.max_iterations,
            enable_parallel_tools=config.enable_parallel_tools,
            max_concurrent_tools=config.max_concurrent_tools,
            max_context_chars=config.max_context_chars,
            enable_context_compaction=config.enable_context_compaction,
            streaming_timeout=config.streaming_timeout,
            enable_semantic_search=config.enable_semantic_search,
            enable_code_correction=config.enable_code_correction,
            enable_analytics=config.enable_analytics,
            enable_tool_cache=config.enable_tool_cache,
            tool_cache_ttl=config.tool_cache_ttl,
            max_recovery_attempts=config.max_recovery_attempts,
            extra=config.extra,
        )

    @classmethod
    def from_subagent_config(cls, config: "SubAgentConfig") -> "UnifiedAgentConfig":
        """Convert from SubAgentConfig.

        Args:
            config: SubAgentConfig instance

        Returns:
            UnifiedAgentConfig with equivalent settings
        """
        # Map SubAgentRole to string
        role_str = config.role.value if hasattr(config.role, "value") else str(config.role)

        return cls(
            mode="team_member",
            role=role_str,
            task=config.task,
            allowed_tools=config.allowed_tools,
            tool_budget=config.tool_budget,
            context_limit=config.context_limit,
            can_spawn_subagents=config.can_spawn_subagents,
            working_directory=config.working_directory,
            timeout_seconds=config.timeout_seconds,
            system_prompt_override=config.system_prompt_override,
        )

    def to_agent_config(self) -> "AgentConfig":
        """Convert to AgentConfig (for backward compatibility).

        Returns:
            AgentConfig instance with equivalent settings
        """
        # Import here to avoid circular dependency
        from victor.framework.config import AgentConfig

        return AgentConfig(
            tool_budget=self.tool_budget,
            max_iterations=self.max_iterations,
            enable_parallel_tools=self.enable_parallel_tools,
            max_concurrent_tools=self.max_concurrent_tools,
            max_context_chars=self.max_context_chars,
            enable_context_compaction=self.enable_context_compaction,
            streaming_timeout=self.streaming_timeout,
            enable_semantic_search=self.enable_semantic_search,
            enable_code_correction=self.enable_code_correction,
            enable_analytics=self.enable_analytics,
            enable_tool_cache=self.enable_tool_cache,
            tool_cache_ttl=self.tool_cache_ttl,
            max_recovery_attempts=self.max_recovery_attempts,
            extra=self.extra,
        )

    def to_settings_dict(self) -> Dict[str, Any]:
        """Convert to Settings-compatible dictionary.

        Returns:
            Dictionary that can be used to override Settings attributes
        """
        settings = {
            "tool_call_budget": self.tool_budget,
            "max_iterations": self.max_iterations,
            "parallel_tool_execution": self.enable_parallel_tools,
            "max_concurrent_tools": self.max_concurrent_tools,
            "context_proactive_compaction": self.enable_context_compaction,
            "streaming_timeout": self.streaming_timeout,
            "use_semantic_tool_selection": self.enable_semantic_search,
            "code_correction_enabled": self.enable_code_correction,
            "analytics_enabled": self.enable_analytics,
            "tool_cache_enabled": self.enable_tool_cache,
            "tool_cache_ttl": self.tool_cache_ttl,
            "recovery_max_attempts": self.max_recovery_attempts,
        }
        if self.max_context_chars:
            settings["max_context_chars"] = self.max_context_chars
        settings.update(self.extra)
        return settings

    # ==========================================================================
    # Factory Methods
    # ==========================================================================

    @classmethod
    def foreground(
        cls,
        provider: str = "anthropic",
        model: Optional[str] = None,
        tool_budget: int = 50,
        enable_semantic_search: bool = False,
        **kwargs: Any,
    ) -> "UnifiedAgentConfig":
        """Create configuration for foreground agent.

        Args:
            provider: LLM provider name
            model: Model identifier
            tool_budget: Maximum tool calls
            enable_semantic_search: Use semantic tool selection
            **kwargs: Additional foreground-specific settings

        Returns:
            UnifiedAgentConfig configured for foreground mode
        """
        return cls(
            mode="foreground",
            provider=provider,
            model=model,
            tool_budget=tool_budget,
            enable_semantic_search=enable_semantic_search,
            **kwargs,
        )

    @classmethod
    def background(
        cls,
        task: str,
        mode_type: ExecutionModeType = ExecutionModeType.BUILD,
        tool_budget: int = 100,
        websocket: bool = False,
        **kwargs: Any,
    ) -> "UnifiedAgentConfig":
        """Create configuration for background agent.

        Args:
            task: Task description
            mode_type: Execution mode (build/plan/explore)
            tool_budget: Maximum tool calls
            websocket: Enable websocket support
            **kwargs: Additional background-specific settings

        Returns:
            UnifiedAgentConfig configured for background mode
        """
        return cls(
            mode="background",
            task=task,
            mode_type=mode_type,
            tool_budget=tool_budget,
            websocket=websocket,
            **kwargs,
        )

    @classmethod
    def team_member(
        cls,
        role: str,
        capabilities: List[str],
        description: str = "",
        allowed_tools: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "UnifiedAgentConfig":
        """Create configuration for team member agent.

        Args:
            role: Team member role
            capabilities: List of agent capabilities
            description: Agent description
            allowed_tools: Tools this agent can use
            **kwargs: Additional team member settings

        Returns:
            UnifiedAgentConfig configured for team member mode
        """
        return cls(
            mode="team_member",
            role=role,
            capabilities=capabilities,
            description=description,
            allowed_tools=allowed_tools or [],
            **kwargs,
        )

    # ==========================================================================
    # Presets
    # ==========================================================================

    @classmethod
    def minimal(cls) -> "UnifiedAgentConfig":
        """Create minimal configuration for simple tasks.

        Disables semantic search, analytics, and reduces budgets
        for quick, lightweight operations.
        """
        return cls(
            mode="foreground",
            tool_budget=15,
            max_iterations=10,
            enable_semantic_search=False,
            enable_analytics=False,
            enable_tool_cache=False,
        )

    @classmethod
    def high_budget(cls) -> "UnifiedAgentConfig":
        """Create high-budget configuration for complex tasks.

        Increases tool budget and iterations for tasks that
        require extensive exploration or many operations.
        """
        return cls(
            mode="foreground",
            tool_budget=200,
            max_iterations=100,
            max_concurrent_tools=10,
            enable_semantic_search=True,
        )

    @classmethod
    def airgapped(cls) -> "UnifiedAgentConfig":
        """Create configuration for air-gapped environments.

        Disables features that require network access.
        """
        return cls(
            mode="foreground",
            enable_semantic_search=False,
            enable_analytics=False,
            enable_tool_cache=False,
        )


def normalize_agent_config(
    config: Optional[FrameworkCompatibleAgentConfig],
) -> Optional[FrameworkCompatibleAgentConfig]:
    """Convert legacy AgentConfig instances to UnifiedAgentConfig.

    Framework entry points still accept the deprecated AgentConfig surface for
    backward compatibility. Internally, prefer UnifiedAgentConfig so presets and
    settings overlays flow through one canonical config representation.
    """
    if config is None or isinstance(config, UnifiedAgentConfig):
        return config

    from victor.framework.config import AgentConfig

    if isinstance(config, AgentConfig):
        return UnifiedAgentConfig.from_agent_config(config)

    return config


__all__ = [
    "UnifiedAgentConfig",
    "AgentLifecycleMode",
    "FrameworkCompatibleAgentConfig",
    "normalize_agent_config",
]
