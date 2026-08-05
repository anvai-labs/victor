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

"""Unified Tool Access Controller.

This module provides a single point of control for all tool access decisions,
replacing 6 scattered systems with a layered, composable architecture.

Design:
- Each access check layer has a precedence level
- Layers are checked in order: Safety > Mode > Session > Vertical > Stage > Intent
- First layer to deny a tool becomes the authoritative source
- All layers are checked even after denial (for audit/explain purposes)

Usage:
    controller = ToolAccessController(registry=registry)
    decision = controller.check_access("shell", context)

    if decision.allowed:
        # Execute tool
        pass
    else:
        logger.info(f"Tool denied: {decision.explain()}")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from victor.agent.protocols import (
    AccessPrecedence,
    IToolAccessController,
    ToolAccessContext,
    ToolAccessDecision,
)
from victor.agent.action_authorizer import (
    get_write_tools_for_policy,
    is_tool_blocked_for_intent,
    normalize_tool_name_for_policy,
    should_allow_shell_for_read_only_intent,
)
from victor.protocols.mode_aware import ModeAwareMixin
from victor.tools.core_tool_aliases import canonicalize_core_tool_name
from victor.tools.tool_names import ToolNames

if TYPE_CHECKING:
    from victor.agent.conversation.state_machine import ConversationStage
    from victor.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _core_tool_name(tool_name: str) -> str:
    """Canonicalize compact tool aliases for shared access-policy decisions."""
    return normalize_tool_name_for_policy(tool_name)


def _tool_matches(tool_name: str, names: Set[str]) -> bool:
    """Return True when a tool matches a set after core alias normalization."""
    canonical_names = {_core_tool_name(name) for name in names}
    return _core_tool_name(tool_name) in canonical_names


def _filter_matching_tools(all_tools: Set[str], names: Set[str]) -> Set[str]:
    """Filter original tool names using normalized membership checks."""
    canonical_names = {_core_tool_name(name) for name in names}
    return {tool for tool in all_tools if _core_tool_name(tool) in canonical_names}


# =============================================================================
# Access Layer Base Class
# =============================================================================


class AccessLayer(ABC):
    """Base class for tool access control layers.

    Each layer checks a specific aspect of tool access (safety, mode, etc.)
    and returns a decision. Layers are composable and checked in precedence order.

    Attributes:
        PRECEDENCE: Numeric precedence (lower = higher priority)
        NAME: Human-readable layer name
    """

    PRECEDENCE: int = 100  # Default low priority
    NAME: str = "base"

    @abstractmethod
    def check(self, tool_name: str, context: Optional[ToolAccessContext]) -> Tuple[bool, str]:
        """Check if a tool is allowed by this layer.

        Args:
            tool_name: Name of the tool to check
            context: Access context with mode, stage, intent, etc.

        Returns:
            Tuple of (is_allowed, reason_string)
        """
        ...

    def get_allowed_tools(
        self, all_tools: Set[str], context: Optional[ToolAccessContext]
    ) -> Set[str]:
        """Get all tools allowed by this layer.

        Default implementation checks each tool individually.
        Override for more efficient bulk filtering.

        Args:
            all_tools: Set of all available tool names
            context: Access context

        Returns:
            Set of allowed tool names
        """
        allowed = set()
        for tool in all_tools:
            is_allowed, _ = self.check(tool, context)
            if is_allowed:
                allowed.add(tool)
        return allowed


# =============================================================================
# Safety Layer (L0) - Highest Precedence
# =============================================================================


class SafetyLayer(AccessLayer):
    """Safety-based tool access control.

    Checks DangerLevel and blocks tools that are too dangerous for
    the current context (e.g., system-level commands in sandbox mode).

    Precedence: 0 (highest - safety first)
    """

    PRECEDENCE = AccessPrecedence.SAFETY.value
    NAME = "safety"

    # Tools that are always blocked for safety
    BLOCKED_TOOLS: Set[str] = frozenset(
        {
            # No tools are unconditionally blocked for now
            # Add dangerous tools here if needed
        }
    )

    # Dangerous operations that require extra caution
    DANGEROUS_TOOLS: Set[str] = frozenset(
        {
            "shell",
            "git_push",
            "git_commit",
            "delete_file",
        }
    )

    def __init__(self, sandbox_mode: bool = False):
        """Initialize safety layer.

        Args:
            sandbox_mode: If True, block dangerous tools
        """
        self._sandbox_mode = sandbox_mode

    def check(self, tool_name: str, context: Optional[ToolAccessContext]) -> Tuple[bool, str]:
        """Check safety constraints for a tool."""
        if _tool_matches(tool_name, self.BLOCKED_TOOLS):
            return False, f"Tool '{tool_name}' is blocked for safety reasons"

        if self._sandbox_mode and _tool_matches(tool_name, self.DANGEROUS_TOOLS):
            return False, f"Tool '{tool_name}' is blocked in sandbox mode"

        return True, "Passed safety checks"


# =============================================================================
# Mode Layer (L1) - Mode-based Restrictions
# =============================================================================


class ModeLayer(AccessLayer, ModeAwareMixin):
    """Mode-based tool access control.

    Restricts tools based on current agent mode (BUILD, PLAN, EXPLORE).
    - BUILD: All tools allowed
    - PLAN: Read-only + sandbox edits
    - EXPLORE: Read-only + sandbox notes

    Precedence: 1 (after safety)
    """

    PRECEDENCE = AccessPrecedence.MODE.value
    NAME = "mode"

    def check(self, tool_name: str, context: Optional[ToolAccessContext]) -> Tuple[bool, str]:
        """Check mode restrictions for a tool."""
        # If BUILD mode (allow_all_tools=True), allow everything
        if self.is_build_mode:
            return True, "BUILD mode allows all tools"

        # Use mode controller's is_tool_allowed
        if not self.is_tool_allowed_by_mode(tool_name):
            # Allow write/edit through when an analysis-safe policy is active;
            # enforce_sandbox_path() gates individual paths inside the tool itself.
            if tool_name in {"write", "edit", "write_file", "edit_files"}:
                try:
                    from victor.tools.write_path_policy import (
                        WritePathTier,
                        get_active_write_policy,
                    )

                    policy = get_active_write_policy()
                    if policy is not None and policy.max_tier >= WritePathTier.LOCAL_ANALYSIS:
                        return (
                            True,
                            "Write tool allowed: analysis-safe path policy active",
                        )
                except Exception:
                    pass  # policy unavailable — fall through to mode block
            mode_name = self.current_mode_name
            return False, f"Tool '{tool_name}' not allowed in {mode_name} mode"

        return True, f"Allowed in {self.current_mode_name} mode"

    def get_allowed_tools(
        self, all_tools: Set[str], context: Optional[ToolAccessContext]
    ) -> Set[str]:
        """Get tools allowed by current mode."""
        if self.is_build_mode:
            return all_tools.copy()

        mode_info = self.get_mode_info()
        if mode_info.allowed_tools:
            # Intersection with mode's allowed tools
            return _filter_matching_tools(all_tools, mode_info.allowed_tools)

        # Remove disallowed tools
        disallowed = _filter_matching_tools(all_tools, mode_info.disallowed_tools)
        return all_tools - disallowed


# =============================================================================
# Session Layer (L2) - Session-enabled Tools
# =============================================================================


class SessionLayer(AccessLayer):
    """Session-based tool access control.

    Restricts tools to those explicitly enabled for the current session.
    If no session tools are set, all tools pass.

    Precedence: 2 (after mode)
    """

    PRECEDENCE = AccessPrecedence.SESSION.value
    NAME = "session"

    def check(self, tool_name: str, context: Optional[ToolAccessContext]) -> Tuple[bool, str]:
        """Check session restrictions for a tool."""
        if context is None or context.session_enabled_tools is None:
            return True, "No session restrictions"

        if _tool_matches(tool_name, context.session_enabled_tools):
            return True, "Tool enabled for this session"

        return False, f"Tool '{tool_name}' not enabled for this session"

    def get_allowed_tools(
        self, all_tools: Set[str], context: Optional[ToolAccessContext]
    ) -> Set[str]:
        """Get tools allowed by session."""
        if context is None or context.session_enabled_tools is None:
            return all_tools.copy()
        return _filter_matching_tools(all_tools, context.session_enabled_tools)


# =============================================================================
# Vertical Layer (L3) - TieredToolConfig
# =============================================================================


class VerticalLayer(AccessLayer):
    """Vertical-based tool access control.

    Restricts tools based on vertical's TieredToolConfig.
    Verticals define core, extension, and optional tools.

    Precedence: 3 (after session)
    """

    PRECEDENCE = AccessPrecedence.VERTICAL.value
    NAME = "vertical"

    def __init__(self, tiered_config: Optional[Any] = None):
        """Initialize with vertical's tiered config.

        Args:
            tiered_config: TieredToolConfig from active vertical
        """
        self._tiered_config = tiered_config

    def set_tiered_config(self, config: Any) -> None:
        """Update the tiered config (e.g., when vertical changes)."""
        self._tiered_config = config

    def check(self, tool_name: str, context: Optional[ToolAccessContext]) -> Tuple[bool, str]:
        """Check vertical restrictions for a tool."""
        if self._tiered_config is None:
            return True, "No vertical restrictions"

        # Get all allowed tools from tiered config (ISP fix: support both interfaces)
        allowed = self._get_allowed_from_config()

        if not allowed:
            return True, "Vertical has no tool restrictions"

        if _tool_matches(tool_name, allowed):
            return True, "Tool allowed by vertical config"

        vertical_name = context.vertical_name if context else "active"
        return False, f"Tool '{tool_name}' not in {vertical_name} vertical's tool set"

    def _get_allowed_from_config(self) -> Set[str]:
        """Extract allowed tools from tiered config (ISP-compliant).

        Supports both legacy interface (core_tools/extension_tools/optional_tools)
        and TieredToolConfig interface (mandatory/vertical_core/semantic_pool).
        """
        if self._tiered_config is None:
            return set()

        allowed: Set[str] = set()

        # TieredToolConfig interface (preferred)
        if hasattr(self._tiered_config, "mandatory"):
            allowed.update(self._tiered_config.mandatory or set())
        if hasattr(self._tiered_config, "vertical_core"):
            allowed.update(self._tiered_config.vertical_core or set())
        if hasattr(self._tiered_config, "semantic_pool"):
            allowed.update(self._tiered_config.semantic_pool or set())

        # Use helper method if available (TieredToolConfig.get_all_tools())
        if hasattr(self._tiered_config, "get_all_tools"):
            allowed.update(self._tiered_config.get_all_tools() or set())

        # Legacy interface fallback (deprecated)
        if hasattr(self._tiered_config, "core_tools"):
            allowed.update(self._tiered_config.core_tools or [])
        if hasattr(self._tiered_config, "extension_tools"):
            allowed.update(self._tiered_config.extension_tools or [])
        if hasattr(self._tiered_config, "optional_tools"):
            allowed.update(self._tiered_config.optional_tools or [])

        return allowed

    def get_allowed_tools(
        self, all_tools: Set[str], context: Optional[ToolAccessContext]
    ) -> Set[str]:
        """Get tools allowed by vertical (ISP fix: uses unified config extraction)."""
        if self._tiered_config is None:
            return all_tools.copy()

        allowed = self._get_allowed_from_config()

        if not allowed:
            return all_tools.copy()

        return _filter_matching_tools(all_tools, allowed)


# =============================================================================
# Stage Layer (L4) - Conversation Stage Filtering
# =============================================================================


class StageLayer(AccessLayer, ModeAwareMixin):
    """Stage-based tool access control.

    Filters write/execute tools during exploration/analysis stages.
    Core tools are always preserved for basic operation.

    Precedence: 4 (after vertical)
    """

    PRECEDENCE = AccessPrecedence.STAGE.value
    NAME = "stage"

    # Stages where write tools should be filtered
    EXPLORATION_STAGES: Set[str] = frozenset({"INITIAL", "PLANNING", "READING", "ANALYSIS"})

    # Write/execute tools to filter during exploration
    WRITE_TOOLS: Set[str] = frozenset(
        {
            "write",
            "edit",
            "shell",
            "git_commit",
            "git_push",
            "delete_file",
        }
    )

    # Core tools never filtered (basic operation)
    CORE_TOOLS: Set[str] = frozenset(
        {
            "read",
            "ls",
            "search",
            "code_search",
        }
    )

    def __init__(self, preserved_tools: Optional[Set[str]] = None):
        """Initialize stage layer.

        Args:
            preserved_tools: Additional tools to never filter (e.g., vertical core)
        """
        self._preserved_tools = preserved_tools or set()

    def set_preserved_tools(self, tools: Set[str]) -> None:
        """Update preserved tools (e.g., from vertical config)."""
        self._preserved_tools = tools

    def check(self, tool_name: str, context: Optional[ToolAccessContext]) -> Tuple[bool, str]:
        """Check stage restrictions for a tool."""
        # Skip stage filtering in BUILD mode
        if self.is_build_mode:
            return True, "BUILD mode skips stage filtering"

        # Get current stage
        stage = None
        if context and context.conversation_stage:
            stage = context.conversation_stage
        elif context and context.metadata.get("stage"):
            stage = context.metadata["stage"]

        if stage is None:
            return True, "No stage context available"

        # Get stage name for comparison
        stage_name = stage.name if hasattr(stage, "name") else str(stage)

        # Check if in exploration stage
        if stage_name not in self.EXPLORATION_STAGES:
            return True, f"Stage {stage_name} allows all tools"

        # Always allow core and preserved tools
        if _tool_matches(tool_name, self.CORE_TOOLS | self._preserved_tools):
            return True, f"Tool '{tool_name}' is core/preserved"

        if _core_tool_name(
            tool_name
        ) == ToolNames.SHELL and should_allow_shell_for_read_only_intent(
            getattr(context, "intent", None),
            getattr(context, "user_message", None),
        ):
            return True, "Explicit readonly shell request allowed during exploration"

        # Filter write tools during exploration
        if _tool_matches(tool_name, self.WRITE_TOOLS | get_write_tools_for_policy()):
            return False, f"Write tool '{tool_name}' filtered during {stage_name} stage"

        return True, f"Tool allowed during {stage_name} stage"


# =============================================================================
# Intent Layer (L5) - Action Intent Filtering
# =============================================================================


class IntentLayer(AccessLayer):
    """Intent-based tool access control.

    Filters tools based on detected user intent.
    - DISPLAY_ONLY: Block all write tools
    - READ_ONLY: Block all write tools
    - Other intents: Allow based on intent category

    Precedence: 5 (lowest - intent is advisory)
    """

    PRECEDENCE = AccessPrecedence.INTENT.value
    NAME = "intent"

    # Intents that block write tools
    READ_ONLY_INTENTS: Set[str] = frozenset({"DISPLAY_ONLY", "READ_ONLY", "EXPLAIN"})

    # Write tools blocked for read-only intents
    WRITE_TOOLS: Set[str] = frozenset(
        {
            "write",
            "edit",
            "shell",
            "git_commit",
            "git_push",
            "delete_file",
        }
    )

    def check(self, tool_name: str, context: Optional[ToolAccessContext]) -> Tuple[bool, str]:
        """Check intent restrictions for a tool."""
        if context is None or context.intent is None:
            return True, "No intent context available"

        # Get intent name
        intent = context.intent
        intent_name = intent.name if hasattr(intent, "name") else str(intent)

        # Check if read-only intent
        if intent_name in self.READ_ONLY_INTENTS:
            if is_tool_blocked_for_intent(
                tool_name,
                context.intent,
                getattr(context, "user_message", None),
            ):
                return (
                    False,
                    f"Write tool '{tool_name}' blocked for {intent_name} intent",
                )

        return True, f"Tool allowed for {intent_name} intent"


# =============================================================================
# Tool Access Controller
# =============================================================================


@dataclass
class ToolAccessController(IToolAccessController):
    """Unified tool access controller.

    Composes multiple access layers and checks them in precedence order.
    Provides a single interface for all tool access decisions.

    Attributes:
        registry: Tool registry for tool lookup
        layers: List of access layers in precedence order
        _cache: Cache for recent decisions (optional optimization)
        _allowed_tools_cache: Cached bulk allowed tool set (scalability fix)
    """

    registry: Optional["ToolRegistry"] = None
    layers: List[AccessLayer] = field(default_factory=list)
    _cache: Dict[str, ToolAccessDecision] = field(default_factory=dict)
    _cache_context_hash: Optional[int] = None
    _allowed_tools_cache: Optional[Set[str]] = None

    def __post_init__(self) -> None:
        """Initialize with default layers if none provided."""
        if not self.layers:
            self.layers = self._create_default_layers()

    def _create_default_layers(self) -> List[AccessLayer]:
        """Create default access layers in precedence order."""
        return [
            SafetyLayer(),
            ModeLayer(),
            SessionLayer(),
            VerticalLayer(),
            StageLayer(),
            IntentLayer(),
        ]

    def _get_context_hash(self, context: Optional[ToolAccessContext]) -> int:
        """Get hash for context to detect cache invalidation."""
        if context is None:
            return 0
        return hash(
            (
                context.current_mode,
                context.conversation_stage.name if context.conversation_stage else None,
                (
                    context.intent.name
                    if context.intent and hasattr(context.intent, "name")
                    else None
                ),
                (
                    frozenset(context.session_enabled_tools)
                    if context.session_enabled_tools
                    else None
                ),
            )
        )

    def _invalidate_cache_if_needed(self, context: Optional[ToolAccessContext]) -> None:
        """Clear cache if context has changed."""
        ctx_hash = self._get_context_hash(context)
        if ctx_hash != self._cache_context_hash:
            self._cache.clear()
            self._allowed_tools_cache = None  # Invalidate bulk cache too
            self._cache_context_hash = ctx_hash

    def check_access(
        self, tool_name: str, context: Optional[ToolAccessContext] = None
    ) -> ToolAccessDecision:
        """Check if a tool is allowed in the given context.

        Checks all layers in precedence order. First denial becomes
        the authoritative decision, but all layers are checked for audit.

        Args:
            tool_name: Name of the tool to check
            context: Access context

        Returns:
            ToolAccessDecision with result and explanation
        """
        self._invalidate_cache_if_needed(context)

        # Check cache
        cache_key = tool_name
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check all layers
        checked_layers: List[str] = []
        layer_results: Dict[str, bool] = {}
        first_denial: Optional[Tuple[AccessLayer, str]] = None

        for layer in self.layers:
            is_allowed, reason = layer.check(tool_name, context)
            checked_layers.append(layer.NAME)
            layer_results[layer.NAME] = is_allowed

            if not is_allowed and first_denial is None:
                first_denial = (layer, reason)

        # Build decision
        if first_denial:
            layer, reason = first_denial
            decision = ToolAccessDecision(
                allowed=False,
                tool_name=tool_name,
                reason=reason,
                source=layer.NAME,
                precedence_level=layer.PRECEDENCE,
                checked_layers=checked_layers,
                layer_results=layer_results,
            )
        else:
            decision = ToolAccessDecision(
                allowed=True,
                tool_name=tool_name,
                reason="All layers allow this tool",
                source="all",
                precedence_level=-1,  # No denial
                checked_layers=checked_layers,
                layer_results=layer_results,
            )

        # Cache and return
        self._cache[cache_key] = decision
        return decision

    def filter_tools(
        self, tools: List[str], context: Optional[ToolAccessContext] = None
    ) -> Tuple[List[str], List[ToolAccessDecision]]:
        """Filter a list of tools to only allowed ones.

        Args:
            tools: List of tool names to filter
            context: Access context

        Returns:
            Tuple of (allowed_tools, denial_decisions)
        """
        allowed: List[str] = []
        denials: List[ToolAccessDecision] = []

        for tool_name in tools:
            decision = self.check_access(tool_name, context)
            if decision.allowed:
                allowed.append(tool_name)
            else:
                denials.append(decision)

        return allowed, denials

    def get_allowed_tools(self, context: Optional[ToolAccessContext] = None) -> Set[str]:
        """Get all tools allowed in the given context.

        Uses layer-specific optimization and caching for scalability.
        On large tool sets, this avoids repeated per-tool iteration.

        Args:
            context: Access context

        Returns:
            Set of allowed tool names
        """
        # Invalidate cache if context changed
        self._invalidate_cache_if_needed(context)

        # Return cached result if available (scalability optimization)
        if self._allowed_tools_cache is not None:
            return self._allowed_tools_cache.copy()

        # Get all available tools from registry
        if self.registry is None:
            return set()

        all_tools = set()
        for tool in self.registry.list_tools(only_enabled=True):
            all_tools.add(tool.name)

        # Apply each layer's filter
        allowed = all_tools.copy()
        for layer in self.layers:
            allowed = layer.get_allowed_tools(allowed, context)

        # Cache the result
        self._allowed_tools_cache = allowed.copy()
        return allowed

    def explain_decision(self, tool_name: str, context: Optional[ToolAccessContext] = None) -> str:
        """Get detailed explanation for a tool access decision.

        Args:
            tool_name: Name of the tool
            context: Access context

        Returns:
            Human-readable explanation
        """
        decision = self.check_access(tool_name, context)
        return decision.explain()

    def get_layer(self, name: str) -> Optional[AccessLayer]:
        """Get a layer by name for configuration.

        Args:
            name: Layer name (e.g., "mode", "safety")

        Returns:
            AccessLayer or None if not found
        """
        for layer in self.layers:
            if layer.NAME == name:
                return layer
        return None

    def set_tiered_config(self, config: Any) -> None:
        """Update vertical's tiered config.

        Args:
            config: TieredToolConfig from active vertical
        """
        vertical_layer = self.get_layer("vertical")
        if isinstance(vertical_layer, VerticalLayer):
            vertical_layer.set_tiered_config(config)
        self._cache.clear()

    def set_preserved_tools(self, tools: Set[str]) -> None:
        """Update stage layer's preserved tools.

        Args:
            tools: Tools to never filter during exploration stages
        """
        stage_layer = self.get_layer("stage")
        if isinstance(stage_layer, StageLayer):
            stage_layer.set_preserved_tools(tools)
        self._cache.clear()


# =============================================================================
# Factory Function
# =============================================================================


def create_tool_access_controller(
    registry: Optional["ToolRegistry"] = None,
    sandbox_mode: bool = False,
    tiered_config: Optional[Any] = None,
    preserved_tools: Optional[Set[str]] = None,
) -> ToolAccessController:
    """Create a configured ToolAccessController instance.

    Factory function for DI registration.

    Args:
        registry: Tool registry for tool lookup
        sandbox_mode: If True, enable sandbox safety restrictions
        tiered_config: TieredToolConfig from active vertical
        preserved_tools: Tools to never filter during exploration

    Returns:
        Configured ToolAccessController instance
    """
    layers = [
        SafetyLayer(sandbox_mode=sandbox_mode),
        ModeLayer(),
        SessionLayer(),
        VerticalLayer(tiered_config=tiered_config),
        StageLayer(preserved_tools=preserved_tools),
        IntentLayer(),
    ]

    return ToolAccessController(registry=registry, layers=layers)
