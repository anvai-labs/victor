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

"""Backward-compatible shim bridging legacy CLI to Framework API.

.. deprecated::
    This module is deprecated and remains supported only through
    ``v1.0.0`` (``2027-06-30``). Migrate to ``Agent.create()`` from the
    Framework API or, for internal composition, ``AgentFactory`` /
    ``AgentCreationFactory``.

This module provides a transition layer that allows existing code using
AgentOrchestrator.from_settings() to benefit from framework features
(observability, verticals, unified events) without modification.

Migration Path:
    **OLD (deprecated):**
        shim = FrameworkShim(settings, profile)
        orchestrator = await shim.create_orchestrator()

    **NEW (Framework API):**
        from victor.framework import Agent

        orchestrator = await Agent.create(
            settings=settings,
            profile="default",
            vertical="coding",
        )

Design Pattern: Adapter
- Wraps legacy creation with framework wiring
- Adds framework features without breaking existing callers

For migration examples, see: ``docs/architecture/migration.md``
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Optional, Type, Union

from victor.framework.vertical_service import apply_vertical_configuration
from victor.core.verticals.adapters import ensure_runtime_vertical

if TYPE_CHECKING:
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator
    from victor.config.settings import Settings
    from victor.observability.integration import ObservabilityIntegration
    from victor.core.verticals.base import VerticalBase, VerticalConfig
    from victor.observability.debouncing.debouncer import SessionStartDebouncer

logger = logging.getLogger(__name__)


class FrameworkShim:
    """Shim layer for framework-enhanced orchestrator creation.

    .. deprecated::
        ``FrameworkShim`` is deprecated and remains supported only through
        ``v1.0.0`` (``2027-06-30``). Use ``Agent.create()`` from the
        Framework API instead.

    This class bridges the gap between the legacy CLI path (which uses
    AgentOrchestrator.from_settings()) and the new Framework API (which
    uses Agent.create()). It allows the CLI to gain framework features
    like observability and vertical configuration without breaking changes.

    Migration Path:
        **OLD (deprecated):**
            shim = FrameworkShim(settings, "default")
            orchestrator = await shim.create_orchestrator()

        **NEW (Framework API):**
            from victor.framework import Agent

            orchestrator = await Agent.create(
                settings=settings,
                profile="default",
            )

        **With vertical (deprecated):**
            shim = FrameworkShim(settings, "default", vertical="coding")

        **With vertical (Framework API):**
            from victor.framework import Agent

            orchestrator = await Agent.create(
                settings=settings,
                profile="default",
                vertical="coding",
            )

    Features:
        - Wraps AgentOrchestrator.from_settings() transparently
        - Wires ObservabilityIntegration automatically
        - Applies vertical configuration (tools, stages, system prompt)
        - Provides session ID generation and propagation
        - Maintains 100% backward compatibility
        - Includes event debouncing to prevent log bloat

    Attributes:
        orchestrator: Created orchestrator instance (after create_orchestrator())
        observability: ObservabilityIntegration instance (if enabled)
        session_id: Session ID for event correlation
    """

    def __init__(
        self,
        settings: "Settings",
        profile_name: str = "default",
        thinking: bool = False,
        vertical: Optional[Union[Type["VerticalBase"], str]] = None,
        enable_observability: bool = True,
        session_id: Optional[str] = None,
    ) -> None:
        """Initialize the FrameworkShim.

        .. deprecated::
            ``FrameworkShim`` is deprecated. Use ``Agent.create()`` instead.

        Args:
            settings: Victor settings object.
            profile_name: Profile name from profiles.yaml.
            thinking: Enable extended thinking mode.
            vertical: Vertical class or name to apply. If a string,
                will be looked up from VerticalRegistry.
            enable_observability: Whether to wire ObservabilityIntegration.
            session_id: Optional session ID for event correlation.
                If not provided, a new UUID will be generated.
        """
        import warnings

        warnings.warn(
            "FrameworkShim is deprecated. Use Agent.create() from the Framework API instead. "
            "Internal surface layers should compose AgentFactory / AgentCreationFactory. "
            "This warning-backed compatibility shim remains supported through "
            "v1.0.0 (2027-06-30). "
            "See docs/architecture/migration.md for migration guidance.",
            DeprecationWarning,
            stacklevel=2,
        )

        self._settings = settings
        self._profile_name = profile_name
        self._thinking = thinking

        # Store original vertical name for error reporting
        self._vertical_name = vertical if isinstance(vertical, str) else None
        self._vertical = self._resolve_vertical(vertical)

        self._enable_observability = enable_observability
        self._session_id = session_id or str(uuid.uuid4())
        # Set after create_orchestrator()
        self._orchestrator: Optional["AgentOrchestrator"] = None
        self._observability: Optional["ObservabilityIntegration"] = None
        self._vertical_config: Optional["VerticalConfig"] = None
        # Instance-level debouncer (loads from settings)
        self._debouncer: Optional["SessionStartDebouncer"] = None

    def _get_debouncer(self) -> "SessionStartDebouncer":
        """Get or create the session start debouncer for this instance.

        The debouncer is created on first access and configured from settings.

        Returns:
            SessionStartDebouncer instance configured from settings.
        """
        if self._debouncer is None:
            from victor.observability.debouncing import (
                SessionStartDebouncer,
                DebounceConfig,
            )

            # Load config from settings (with fallback to defaults)
            config = DebounceConfig.from_settings(self._settings)
            self._debouncer = SessionStartDebouncer(config)
        return self._debouncer

    def _resolve_vertical(
        self, vertical: Optional[Union[Type["VerticalBase"], str]]
    ) -> Optional[Type["VerticalBase"]]:
        """Resolve vertical from name or class.

        Args:
            vertical: Vertical class or string name.

        Returns:
            Vertical class or None.
        """
        if vertical is None:
            return None

        if isinstance(vertical, str):
            resolved = get_vertical(vertical)
            if resolved is None:
                from victor.core.verticals.base import VerticalRegistry

                logger.warning(
                    f"Vertical '{vertical}' not found in registry. "
                    f"Available: {VerticalRegistry.list_names()}"
                )
            return resolved

        return vertical

    async def create_orchestrator(self) -> "AgentOrchestrator":
        """Create orchestrator with framework features wired.

        This is the main entry point. It:
        1. Bootstraps the DI container with the correct vertical
        2. Creates the base orchestrator via from_settings()
        3. Applies vertical configuration if specified
        4. Wires observability if enabled
        5. Returns the enhanced orchestrator

        Returns:
            Configured AgentOrchestrator instance.

        Raises:
            RuntimeError: If orchestrator creation fails.
        """
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.core.bootstrap import ensure_bootstrapped

        logger.debug(
            f"FrameworkShim creating orchestrator: profile={self._profile_name}, "
            f"thinking={self._thinking}, vertical={self._vertical}, "
            f"observability={self._enable_observability}"
        )

        # Step 0: Ensure bootstrap with correct vertical BEFORE orchestrator creation
        # This ensures vertical services are registered with the correct vertical name
        # Use stored vertical name (preserves original name even if vertical not found)
        vertical_name = self._vertical_name or (self._vertical.name if self._vertical else None)
        ensure_bootstrapped(self._settings, vertical=vertical_name)

        # Step 1: Create base orchestrator
        self._orchestrator = await AgentOrchestrator.from_settings(
            self._settings,
            profile_name=self._profile_name,
            thinking=self._thinking,
        )

        # Step 2: Apply vertical configuration
        if self._vertical:
            self._apply_vertical(self._vertical)

        # Step 3: Wire observability
        if self._enable_observability:
            self._wire_observability()

        # Step 4: Initialize skill auto-selection
        await self._initialize_skill_matcher()

        logger.debug(f"FrameworkShim created orchestrator: session_id={self._session_id}")

        return self._orchestrator

    async def _initialize_skill_matcher(self) -> None:
        """Initialize embedding-based skill auto-selection on the orchestrator.

        Discovers all skills from verticals + entry points + user YAML,
        pre-embeds them, and attaches the matcher to the orchestrator.
        Silently skips if disabled or on any error.
        """
        try:
            # Check if auto-selection is enabled via settings
            if not getattr(self._settings, "skill_auto_select_enabled", True):
                return

            from victor.framework.skill_matcher import SkillMatcher
            from victor.framework.skills import SkillRegistry

            registry = SkillRegistry()

            # Load from discovered verticals
            try:
                from victor.core.verticals.vertical_loader import VerticalLoader

                loader = VerticalLoader()
                loader.discover_verticals()
                for _name, vertical_cls in loader._discovered_verticals.items():
                    if hasattr(vertical_cls, "get_skills"):
                        try:
                            registry.from_vertical(vertical_cls)
                        except Exception:
                            pass
            except Exception:
                pass

            # Load from entry points + user YAML
            try:
                registry.from_entry_points()
            except Exception:
                pass
            try:
                registry.from_user_skills()
            except Exception:
                pass

            if not registry.list_all():
                return

            # Build thresholds from settings
            high_t = getattr(self._settings, "skill_auto_select_high_threshold", 0.65)
            low_t = getattr(self._settings, "skill_auto_select_low_threshold", 0.45)
            use_edge = getattr(self._settings, "skill_auto_select_use_edge_fallback", True)

            matcher = SkillMatcher(
                high_threshold=high_t,
                low_threshold=low_t,
                use_edge_fallback=use_edge,
            )
            await matcher.initialize(registry)
            self._orchestrator._skill_matcher = matcher

            # Attach analytics tracker
            from victor.framework.skill_analytics import SkillAnalytics

            self._orchestrator._skill_analytics = SkillAnalytics()

            logger.info(
                "Skill auto-selection initialized: %d skills indexed",
                len(registry.list_all()),
            )
        except Exception:
            logger.debug("Skill auto-selection initialization skipped", exc_info=True)

    def _apply_vertical(self, vertical: Type["VerticalBase"]) -> None:
        """Apply vertical configuration to orchestrator.

        This delegates to VerticalIntegrationPipeline for unified vertical
        application across both CLI (FrameworkShim) and SDK (Agent.create()) paths.

        The pipeline applies:
        - System prompt (via prompt_builder)
        - Tool filter (via set_enabled_tools protocol)
        - Stage configuration (via vertical context)
        - Middleware chain (via apply_vertical_middleware protocol)
        - Safety extensions (via apply_vertical_safety_patterns protocol)
        - Prompt contributions (for task type hints)
        - Mode configuration and tool dependencies

        Args:
            vertical: Vertical class to apply.
        """
        logger.debug(f"Applying vertical via pipeline: {vertical.name}")

        # Use shared framework service for vertical application
        result = apply_vertical_configuration(self._orchestrator, vertical, source="cli")

        # Store result for access
        self._vertical_config = result.context.config if result.context else None
        self._integration_result = result

        if result.success:
            logger.info(
                f"Applied vertical '{result.vertical_name}' via CLI path: "
                f"tools={len(result.tools_applied)}, "
                f"middleware={result.middleware_count}, "
                f"safety={result.safety_patterns_count}, "
                f"hints={result.prompt_hints_count}"
            )
        else:
            for error in result.errors:
                logger.error(f"Vertical integration error: {error}")
            for warning in result.warnings:
                logger.warning(f"Vertical integration warning: {warning}")

    def _wire_observability(self) -> None:
        """Wire ObservabilityIntegration to orchestrator.

        Delegates to the shared setup_observability_integration() helper
        used by both SDK and CLI paths.
        """
        from victor.framework._internal import setup_observability_integration

        self._observability = setup_observability_integration(
            self._orchestrator,
            session_id=self._session_id,
        )

        logger.debug(f"Wired observability: session_id={self._session_id}")

    # =========================================================================
    # Public Properties
    # =========================================================================

    @property
    def orchestrator(self) -> Optional["AgentOrchestrator"]:
        """Get the created orchestrator.

        Returns:
            Orchestrator instance, or None if not yet created.
        """
        return self._orchestrator

    @property
    def observability(self) -> Optional["ObservabilityIntegration"]:
        """Get the ObservabilityIntegration if enabled.

        Returns:
            ObservabilityIntegration instance, or None if disabled.
        """
        return self._observability

    @property
    def session_id(self) -> str:
        """Get the session ID for event correlation.

        Returns:
            Session ID string.
        """
        return self._session_id

    @property
    def vertical(self) -> Optional[Type["VerticalBase"]]:
        """Get the applied vertical class.

        Returns:
            Vertical class, or None if no vertical applied.
        """
        return self._vertical

    @property
    def vertical_config(self) -> Optional["VerticalConfig"]:
        """Get the applied vertical configuration.

        Returns:
            VerticalConfig, or None if no vertical applied.
        """
        return self._vertical_config

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    def emit_session_start(self, metadata: Optional[dict] = None) -> None:
        """Emit a session start event if observability is enabled.

        This method includes adaptive debouncing to prevent log bloat from
        duplicate session_start events that occur within a short time window.

        Args:
            metadata: Optional session metadata.
        """
        if not self._observability:
            return

        # Prepare metadata with session_id (create a copy to avoid modifying caller's dict)
        event_metadata = dict(metadata or {})  # Create a shallow copy
        event_metadata.setdefault("session_id", self._session_id)

        # Apply debouncing to prevent log bloat
        debouncer = self._get_debouncer()

        if not debouncer.should_emit(self._session_id, event_metadata):
            logger.debug(f"Debounced duplicate session_start: {self._session_id}")
            return

        # Emit the event
        self._observability.on_session_start(event_metadata)
        debouncer.record(self._session_id, event_metadata)

    def emit_session_end(
        self,
        tool_calls: int = 0,
        duration_seconds: Optional[float] = None,
        success: bool = True,
    ) -> None:
        """Emit a session end event if observability is enabled.

        Args:
            tool_calls: Total tool calls in session.
            duration_seconds: Session duration.
            success: Whether session completed successfully.
        """
        if self._observability:
            self._observability.on_session_end(
                tool_calls=tool_calls,
                duration_seconds=duration_seconds,
                success=success,
            )


def get_vertical(name: Optional[str]) -> Optional[Type["VerticalBase"]]:
    """Convenience function to look up a vertical by name.

    This is a thin wrapper around VerticalRegistry.get() for CLI usage.

    Args:
        name: Vertical name (case-insensitive).

    Returns:
        Vertical class or None if not found.

    Example:
        vertical = get_vertical("coding")
        if vertical:
            shim = FrameworkShim(settings, vertical=vertical)
    """
    if name is None:
        return None

    # Import package root and re-run built-in registration once module init settles.
    # This self-heals early-import circular ordering where a built-in (notably coding)
    # can be skipped during the package's first import.
    import victor.core.verticals as core_verticals

    core_verticals._register_builtin_verticals()
    from victor.core.verticals.base import VerticalRegistry

    # Try exact match first
    result = VerticalRegistry.get(name)
    if result:
        return ensure_runtime_vertical(result)

    # Try case-insensitive match
    name_lower = name.lower()
    for registered_name in VerticalRegistry.list_names():
        if registered_name.lower() == name_lower:
            result = VerticalRegistry.get(registered_name)
            return ensure_runtime_vertical(result) if result is not None else None

    return None


def list_verticals() -> list[str]:
    """List all available vertical names.

    Returns:
        List of registered vertical names.

    Example:
        print(f"Available verticals: {list_verticals()}")
    """
    # Import package root and re-run built-in registration once module init settles.
    import victor.core.verticals as core_verticals

    core_verticals._register_builtin_verticals()
    from victor.core.verticals.base import VerticalRegistry

    return VerticalRegistry.list_names()
