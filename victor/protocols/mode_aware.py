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

"""Mode-aware component pattern.

This module provides the ModeAwareMixin for consistent mode controller access
throughout the codebase, eliminating scattered ad-hoc imports.

Design Pattern:
- Mixin Pattern: Components inherit ModeAwareMixin for mode awareness
- Lazy Loading: Mode controller loaded on first access
- Fail-Safe: Returns safe defaults if mode controller unavailable

Usage:
    class ToolSelector(ModeAwareMixin):
        def select_tools(self, tools: List[str]) -> List[str]:
            if self.is_build_mode:
                return tools  # All tools allowed in BUILD mode
            return self._filter_for_mode(tools)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Optional, Protocol, Set, runtime_checkable

if TYPE_CHECKING:
    from victor.agent.mode_controller import AgentModeController, OperationalModeConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class IModeController(Protocol):
    """Protocol for mode controller access.

    This protocol defines the minimum interface needed by ModeAwareMixin,
    allowing for easy testing and dependency injection.
    """

    @property
    def current_mode(self) -> Any:
        """Get the current agent mode."""
        ...

    @property
    def config(self) -> Any:
        """Get the current mode configuration."""
        ...

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed in the current mode."""
        ...

    def get_tool_priority(self, tool_name: str) -> float:
        """Get priority adjustment for a tool in current mode."""
        ...


@dataclass
class ModeInfo:
    """Information about the current mode.

    Provides a snapshot of mode state for components that need
    to make decisions based on mode without repeated controller access.

    Attributes:
        name: Mode name (BUILD, PLAN, EXPLORE)
        allow_all_tools: Whether all tools are allowed
        exploration_multiplier: Multiplier for exploration limits
        sandbox_dir: Sandbox directory for limited edits (if any)
        allowed_tools: Set of explicitly allowed tools
        disallowed_tools: Set of explicitly disallowed tools
    """

    name: str = "BUILD"
    allow_all_tools: bool = True
    exploration_multiplier: float = 1.0
    sandbox_dir: Optional[str] = None
    allowed_tools: Set[str] = None  # type: ignore
    disallowed_tools: Set[str] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.allowed_tools is None:
            self.allowed_tools = set()
        if self.disallowed_tools is None:
            self.disallowed_tools = set()

    @classmethod
    def default(cls) -> "ModeInfo":
        """Create default mode info (BUILD mode behavior)."""
        return cls(
            name="BUILD",
            allow_all_tools=True,
            exploration_multiplier=2.0,
            sandbox_dir=None,
            allowed_tools=set(),
            disallowed_tools=set(),
        )


class ModeAwareMixin:
    """Mixin for mode-aware components.

    Provides consistent access to mode controller and mode-related
    properties throughout the codebase.

    Features:
    - Lazy loading of mode controller
    - Cached property access for performance
    - Safe defaults when mode controller unavailable
    - Snapshot capability for batch operations

    Usage:
        class MyComponent(ModeAwareMixin):
            def do_something(self) -> None:
                if self.is_build_mode:
                    # Full capabilities
                    pass
                elif self.is_plan_mode:
                    # Read-only with sandbox
                    pass

    Attributes:
        _mode_controller: Optional cached mode controller reference
    """

    _mode_controller: Optional["AgentModeController"] = None
    _mode_info_cache: Optional[ModeInfo] = None

    @cached_property
    def mode_controller(self) -> Optional["AgentModeController"]:
        """Get the mode controller instance.

        Uses lazy loading with caching. Returns None if mode
        controller is unavailable (e.g., during testing).

        Returns:
            AgentModeController instance or None
        """
        if self._mode_controller is None:
            try:
                from victor.agent.mode_controller import get_mode_controller

                self._mode_controller = get_mode_controller()
            except Exception as e:
                logger.debug(f"Mode controller not available: {e}")
                return None
        return self._mode_controller

    def set_mode_controller(self, controller: "AgentModeController") -> None:
        """Set the mode controller explicitly.

        Useful for testing or dependency injection.

        Args:
            controller: Mode controller instance
        """
        self._mode_controller = controller
        # Invalidate cached_property
        if "mode_controller" in self.__dict__:
            del self.__dict__["mode_controller"]
        # Invalidate mode info cache
        self._mode_info_cache = None

    @property
    def current_mode_name(self) -> str:
        """Get the current mode name.

        Returns:
            Mode name string (BUILD, PLAN, EXPLORE)
        """
        mc = self.mode_controller
        if mc is None:
            return "BUILD"  # Default to BUILD
        return mc.current_mode.value.upper()

    @property
    def is_build_mode(self) -> bool:
        """Check if currently in BUILD mode.

        BUILD mode allows all tools with no restrictions.

        Note: When mode controller is unavailable, returns False (conservative)
        to ensure stage filtering and other mode-dependent behaviors continue
        to work as expected. This maintains backward compatibility with the
        original try/except pattern that would continue with filtering on error.

        Returns:
            True if in BUILD mode (allow_all_tools=True)
        """
        mc = self.mode_controller
        if mc is None:
            # Conservative default: don't assume build mode when controller unavailable
            # This ensures stage filtering and other protections remain active
            return False
        return mc.config.allow_all_tools

    @property
    def is_plan_mode(self) -> bool:
        """Check if currently in PLAN mode.

        PLAN mode allows read operations and sandbox edits.

        Returns:
            True if in PLAN mode
        """
        return self.current_mode_name == "PLAN"

    @property
    def is_explore_mode(self) -> bool:
        """Check if currently in EXPLORE mode.

        EXPLORE mode is read-only with sandbox notes.

        Returns:
            True if in EXPLORE mode
        """
        return self.current_mode_name == "EXPLORE"

    @property
    def exploration_multiplier(self) -> float:
        """Get the exploration limit multiplier for current mode.

        Higher multipliers allow more exploration iterations.
        - BUILD: 2.0x (reading before writing)
        - PLAN: 2.5x (thorough analysis)
        - EXPLORE: 3.0x (exploration is primary goal)

        Returns:
            Multiplier value (default 1.0)
        """
        mc = self.mode_controller
        if mc is None:
            return 1.0
        return getattr(mc.config, "exploration_multiplier", 1.0)

    @property
    def sandbox_dir(self) -> Optional[str]:
        """Get the sandbox directory for limited edits.

        In PLAN and EXPLORE modes, edits are restricted to sandbox.

        Returns:
            Sandbox directory path or None
        """
        mc = self.mode_controller
        if mc is None:
            return None
        return getattr(mc.config, "sandbox_dir", None)

    @property
    def allow_sandbox_edits(self) -> bool:
        """Check if sandbox edits are allowed.

        Returns:
            True if edits allowed in sandbox directory
        """
        mc = self.mode_controller
        if mc is None:
            return False
        return getattr(mc.config, "allow_sandbox_edits", False)

    def is_tool_allowed_by_mode(self, tool_name: str) -> bool:
        """Check if a tool is allowed by the current mode.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool is allowed
        """
        mc = self.mode_controller
        if mc is None:
            return True  # Default to allowing all
        return mc.is_tool_allowed(tool_name)

    def get_tool_mode_priority(self, tool_name: str) -> float:
        """Get mode-specific priority for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Priority multiplier (1.0 = no adjustment)
        """
        mc = self.mode_controller
        if mc is None:
            return 1.0
        return mc.get_tool_priority(tool_name)

    def get_mode_info(self, refresh: bool = False) -> ModeInfo:
        """Get a snapshot of current mode information.

        Useful for batch operations where repeated controller
        access would be inefficient.

        Args:
            refresh: Force refresh of cached info

        Returns:
            ModeInfo snapshot
        """
        if self._mode_info_cache is not None and not refresh:
            return self._mode_info_cache

        mc = self.mode_controller
        if mc is None:
            self._mode_info_cache = ModeInfo.default()
            return self._mode_info_cache

        config = mc.config
        self._mode_info_cache = ModeInfo(
            name=mc.current_mode.value.upper(),
            allow_all_tools=config.allow_all_tools,
            exploration_multiplier=getattr(config, "exploration_multiplier", 1.0),
            sandbox_dir=getattr(config, "sandbox_dir", None),
            allowed_tools=set(config.allowed_tools) if config.allowed_tools else set(),
            disallowed_tools=(set(config.disallowed_tools) if config.disallowed_tools else set()),
        )
        return self._mode_info_cache

    def get_mode_system_prompt(self) -> str:
        """Get the system prompt addition for current mode.

        Returns:
            System prompt text for current mode
        """
        mc = self.mode_controller
        if mc is None:
            return ""
        return mc.get_system_prompt_addition()


# Factory function for DI registration
def create_mode_aware_mixin() -> ModeAwareMixin:
    """Create a ModeAwareMixin instance for DI registration.

    Returns:
        New ModeAwareMixin instance
    """
    return ModeAwareMixin()
