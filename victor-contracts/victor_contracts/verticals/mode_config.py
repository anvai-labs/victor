"""Mode configuration types for vertical plugins.

Data types for defining operational modes (quick, standard, comprehensive)
and vertical-specific mode configurations. These are pure data types with
no runtime dependencies, safe for use in external vertical packages.

The ModeConfigRegistry (singleton, runtime-dependent) stays in
victor.core.mode_config and is accessible to verticals via
victor.framework.extensions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ModeLevel(Enum):
    """Predefined mode levels with standard configurations."""

    QUICK = "quick"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    EXTENDED = "extended"


# Budget validation upper bound 5x'd (500 -> 2500) 2026-10-31 so the 5x'd default
# mode budgets (e.g. devops migration max_iterations=600) remain valid.
@dataclass
class ModeConfig:
    """Simplified mode configuration with core operational parameters.

    Attributes:
        tool_budget: Maximum number of tool calls allowed (1-2500)
        max_iterations: Maximum conversation iterations (1-2500)
        exploration_multiplier: Factor for exploration iterations (0.1-10.0)
        allowed_tools: Optional set of allowed tool names
    """

    tool_budget: int
    max_iterations: int
    exploration_multiplier: float = 1.0
    allowed_tools: Optional[Set[str]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool_budget, int):
            try:
                self.tool_budget = int(self.tool_budget)
            except (TypeError, ValueError) as e:
                raise ValueError(f"tool_budget must be an integer: {e}") from e
        if not (1 <= self.tool_budget <= 2500):
            raise ValueError(f"tool_budget must be between 1 and 2500, got {self.tool_budget}")
        if not isinstance(self.max_iterations, int):
            try:
                self.max_iterations = int(self.max_iterations)
            except (TypeError, ValueError) as e:
                raise ValueError(f"max_iterations must be an integer: {e}") from e
        if not (1 <= self.max_iterations <= 2500):
            raise ValueError(
                f"max_iterations must be between 1 and 2500, " f"got {self.max_iterations}"
            )
        if not isinstance(self.exploration_multiplier, (int, float)):
            try:
                self.exploration_multiplier = float(self.exploration_multiplier)
            except (TypeError, ValueError) as e:
                raise ValueError(f"exploration_multiplier must be numeric: {e}") from e
        if not (0.1 <= self.exploration_multiplier <= 10.0):
            raise ValueError(
                f"exploration_multiplier must be between 0.1 and 10.0, "
                f"got {self.exploration_multiplier}"
            )
        if self.allowed_tools is not None and isinstance(self.allowed_tools, list):
            self.allowed_tools = set(self.allowed_tools)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModeConfig:
        return cls(
            tool_budget=data.get("tool_budget", 75),
            max_iterations=data.get("max_iterations", 50),
            exploration_multiplier=data.get("exploration_multiplier", 1.0),
            allowed_tools=data.get("allowed_tools"),
        )


@dataclass
class ModeDefinition:
    """Full mode definition with metadata for vertical-specific configs.

    Attributes:
        name: Mode identifier (1-50 chars)
        tool_budget: Maximum tool calls allowed (1-2500)
        max_iterations: Maximum conversation iterations (1-2500)
        temperature: LLM temperature setting (0.0-2.0)
        description: Human-readable description (max 500 chars)
        exploration_multiplier: Factor for exploration iterations (0.1-10.0)
        allowed_stages: Optional list of allowed workflow stages
        priority_tools: Optional list of tools to prioritize
        allowed_tools: Optional set of allowed tool names
        metadata: Additional mode-specific configuration
    """

    name: str
    tool_budget: int
    max_iterations: int
    temperature: float = 0.7
    description: str = ""
    exploration_multiplier: float = 1.0
    allowed_stages: List[str] = field(default_factory=list)
    priority_tools: List[str] = field(default_factory=list)
    allowed_tools: Optional[Set[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError("name must be a string")
        if not (1 <= len(self.name) <= 50):
            raise ValueError(f"name must be between 1 and 50 characters, " f"got {len(self.name)}")
        if not isinstance(self.tool_budget, int):
            try:
                self.tool_budget = int(self.tool_budget)
            except (TypeError, ValueError) as e:
                raise ValueError(f"tool_budget must be an integer: {e}") from e
        if not (1 <= self.tool_budget <= 2500):
            raise ValueError(f"tool_budget must be between 1 and 2500, " f"got {self.tool_budget}")
        if not isinstance(self.max_iterations, int):
            try:
                self.max_iterations = int(self.max_iterations)
            except (TypeError, ValueError) as e:
                raise ValueError(f"max_iterations must be an integer: {e}") from e
        if not (1 <= self.max_iterations <= 2500):
            raise ValueError(
                f"max_iterations must be between 1 and 2500, " f"got {self.max_iterations}"
            )
        if not isinstance(self.temperature, (int, float)):
            try:
                self.temperature = float(self.temperature)
            except (TypeError, ValueError) as e:
                raise ValueError(f"temperature must be numeric: {e}") from e
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"temperature must be between 0.0 and 2.0, " f"got {self.temperature}")
        if not isinstance(self.description, str):
            self.description = str(self.description)
        if len(self.description) > 500:
            logger.warning(
                "description exceeds 500 chars (%d), truncating",
                len(self.description),
            )
            self.description = self.description[:500]
        if not isinstance(self.exploration_multiplier, (int, float)):
            try:
                self.exploration_multiplier = float(self.exploration_multiplier)
            except (TypeError, ValueError) as e:
                raise ValueError(f"exploration_multiplier must be numeric: {e}") from e
        if not (0.1 <= self.exploration_multiplier <= 10.0):
            raise ValueError(
                f"exploration_multiplier must be between 0.1 and 10.0, "
                f"got {self.exploration_multiplier}"
            )
        if self.allowed_tools is not None and isinstance(self.allowed_tools, list):
            self.allowed_tools = set(self.allowed_tools)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModeDefinition:
        return cls(
            name=data.get("name", ""),
            tool_budget=data.get("tool_budget", 75),
            max_iterations=data.get("max_iterations", 50),
            temperature=data.get("temperature", 0.7),
            description=data.get("description", ""),
            exploration_multiplier=data.get("exploration_multiplier", 1.0),
            allowed_stages=data.get("allowed_stages", []),
            priority_tools=data.get("priority_tools", []),
            allowed_tools=data.get("allowed_tools"),
            metadata=data.get("metadata", {}),
        )

    def to_mode_config(self) -> ModeConfig:
        return ModeConfig(
            tool_budget=self.tool_budget,
            max_iterations=self.max_iterations,
            exploration_multiplier=self.exploration_multiplier,
            allowed_tools=self.allowed_tools,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tool_budget": self.tool_budget,
            "max_iterations": self.max_iterations,
            "temperature": self.temperature,
            "description": self.description,
            "exploration_multiplier": self.exploration_multiplier,
            "allowed_stages": self.allowed_stages,
            "priority_tools": self.priority_tools,
            "allowed_tools": (list(self.allowed_tools) if self.allowed_tools else None),
            "metadata": self.metadata,
        }


@dataclass
class VerticalModeConfig:
    """Mode configuration for a vertical.

    Attributes:
        vertical_name: Name of the vertical
        modes: Vertical-specific mode overrides
        task_budgets: Task type to tool budget mapping
        default_mode: Name of the default mode
        default_budget: Default tool budget when no mode specified
    """

    vertical_name: str
    modes: Dict[str, ModeDefinition] = field(default_factory=dict)
    task_budgets: Dict[str, int] = field(default_factory=dict)
    default_mode: str = "standard"
    default_budget: int = 50  # 5x'd 2026-10-31: agents run longer autonomous sessions


class StaticModeConfigProvider:
    """Pure SDK mode-config provider backed by static vertical definitions.

    This replaces the need for external verticals to import the runtime
    ``ModeConfigRegistry`` or ``RegistryBasedModeConfigProvider`` helpers.
    The core runtime already consumes ``ModeConfigProviderProtocol`` via
    duck typing and normalizes the returned objects centrally.
    """

    def __init__(self, config: VerticalModeConfig):
        """Initialize the provider with a vertical's mode configuration."""

        self._config = config

    def get_mode_configs(self) -> Dict[str, ModeConfig]:
        """Return mode configurations for this vertical."""

        return {
            name: definition.to_mode_config() for name, definition in self._config.modes.items()
        }

    def get_default_mode(self) -> str:
        """Return the configured default mode name."""

        return self._config.default_mode

    def get_default_tool_budget(self, task_type: Optional[str] = None) -> int:
        """Return the configured default tool budget."""

        if task_type:
            return self._config.task_budgets.get(task_type, self._config.default_budget)
        return self._config.default_budget

    def get_tool_budget_for_task(self, task_type: str) -> int:
        """Return the recommended budget for a task type."""

        return self.get_default_tool_budget(task_type)


__all__ = [
    "ModeConfig",
    "ModeDefinition",
    "ModeLevel",
    "StaticModeConfigProvider",
    "VerticalModeConfig",
]
