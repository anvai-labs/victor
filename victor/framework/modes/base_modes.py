"""Base mode configuration shared across all verticals.

Provides standard modes: normal, careful, fast, analysis.
Verticals override get_vertical_modes() to add domain-specific modes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModeDefinition:
    """Definition of an operational mode."""

    name: str
    tool_budget: int = 10
    max_iterations: int = 50
    temperature: float = 0.7
    description: str = ""
    exploration_multiplier: float = 1.0


# Standard modes available to all verticals
BASE_MODES: dict[str, ModeDefinition] = {
    "default": ModeDefinition(
        name="default",
        tool_budget=10,
        max_iterations=50,
        temperature=0.7,
        description="Balanced mode for general tasks",
    ),
    "careful": ModeDefinition(
        name="careful",
        tool_budget=20,
        max_iterations=100,
        temperature=0.5,
        description="Thorough mode with more exploration",
        exploration_multiplier=2.0,
    ),
    "fast": ModeDefinition(
        name="fast",
        tool_budget=5,
        max_iterations=20,
        temperature=0.3,
        description="Quick mode for simple tasks",
        exploration_multiplier=0.5,
    ),
    "analysis": ModeDefinition(
        name="analysis",
        tool_budget=30,
        max_iterations=100,
        temperature=0.8,
        description="Deep analysis mode",
        exploration_multiplier=2.5,
    ),
}

BASE_TASK_BUDGETS: dict[str, int] = {
    "code_generation": 3,
    "edit": 5,
    "search": 3,
    "analysis_deep": 25,
    "analyze": 10,
    "debug": 12,
    "test": 8,
    "refactor": 15,
}


class BaseModeConfig:
    """Base mode configuration.

    Verticals should subclass and override get_vertical_modes()
    and get_vertical_task_budgets() to add domain-specific modes.
    """

    @classmethod
    def get_modes(cls) -> dict[str, ModeDefinition]:
        """Get all modes: base + vertical-specific."""
        return {**BASE_MODES, **cls.get_vertical_modes()}

    @classmethod
    def get_vertical_modes(cls) -> dict[str, ModeDefinition]:
        """Override in vertical subclass to add domain-specific modes."""
        return {}

    @classmethod
    def get_task_budgets(cls) -> dict[str, int]:
        """Get all task budgets: base + vertical-specific."""
        return {**BASE_TASK_BUDGETS, **cls.get_vertical_task_budgets()}

    @classmethod
    def get_vertical_task_budgets(cls) -> dict[str, int]:
        """Override in vertical subclass."""
        return {}

    @classmethod
    def get_mode_for_complexity(cls, complexity: str) -> str:
        """Map complexity to mode. Override for vertical-specific mapping."""
        mapping = {
            "trivial": "fast",
            "simple": "default",
            "moderate": "default",
            "complex": "careful",
            "very_complex": "analysis",
        }
        return mapping.get(complexity, "default")

    @classmethod
    def register(cls, vertical_name: str) -> None:
        """Push modes into the central ModeConfigRegistry."""
        from victor.core.mode_config import ModeConfigRegistry

        registry = ModeConfigRegistry.get_instance()
        # Convert local ModeDefinition to core ModeDefinition for registry
        from victor.core.mode_config import ModeDefinition as CoreModeDefinition

        core_modes = {}
        for name, mode in cls.get_modes().items():
            core_modes[name] = CoreModeDefinition(
                name=mode.name,
                tool_budget=mode.tool_budget,
                max_iterations=mode.max_iterations,
                temperature=mode.temperature,
                description=mode.description,
            )
        registry.register_vertical(
            name=vertical_name,
            modes=core_modes,
            task_budgets=cls.get_task_budgets(),
        )
