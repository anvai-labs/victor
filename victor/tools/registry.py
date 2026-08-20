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

"""Tool registry for managing available tools."""

import logging
import threading
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from victor.core.registry import BaseRegistry
from victor.tools.enums import CostTier

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.tools.registration.strategies import ToolRegistrationStrategy
    from victor.tools.registration.registry import ToolRegistrationStrategyRegistry


# Type alias: Any at runtime, BaseTool for type-checkers
# This avoids circular import while preserving type safety
_ToolType = Any  # Runtime: avoid circular import
if TYPE_CHECKING:
    _ToolType = "BaseTool"


class HookError(Exception):
    """Raised when a critical hook fails."""

    def __init__(self, hook_name: str, original_error: Exception, tool_name: str = ""):
        self.hook_name = hook_name
        self.original_error = original_error
        self.tool_name = tool_name
        super().__init__(
            f"Critical hook '{hook_name}' failed for tool '{tool_name}': {original_error}"
        )


class Hook:
    """Tool execution hook with metadata.

    Hooks can be marked as critical, meaning their failure will prevent
    tool execution (useful for safety checks, validation, etc.).
    """

    def __init__(
        self,
        callback: Callable,
        name: str = "",
        critical: bool = False,
        description: str = "",
    ):
        """Initialize hook.

        Args:
            callback: The hook function to call
            name: Human-readable name for the hook
            critical: If True, hook failure blocks tool execution
            description: Description of what the hook does
        """
        self.callback = callback
        self.name = name or callback.__name__
        self.critical = critical
        self.description = description

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the hook callback."""
        return self.callback(*args, **kwargs)


class ToolRegistry(BaseRegistry[str, _ToolType]):
    """Registry for managing available tools.

    Extends BaseRegistry to provide tool-specific functionality including:
    - Hook system for pre/post execution callbacks
    - Tool enable/disable state management
    - Cost tier filtering
    - JSON schema generation

    The generic type is Any to avoid circular import with BaseTool,
    but values are always BaseTool instances.
    """

    def __init__(self) -> None:
        """Initialize tool registry."""
        super().__init__()
        # Import here to avoid circular dependency
        from victor.tools.base import BaseTool, ToolResult

        self._BaseTool = BaseTool
        self._ToolResult = ToolResult
        # Note: self._items is inherited from BaseRegistry, aliased to _tools for compatibility
        self._tool_enabled: Dict[str, bool] = {}  # Track enabled/disabled state
        self._before_hooks: List[Union[Hook, Callable[[str, Dict[str, Any]], None]]] = []
        self._after_hooks: List[Union[Hook, Callable[..., Any]]] = []

        # Schema cache: version-counter based invalidation for O(1) cache checks.
        # Key is (only_enabled: bool), value is (version_at_cache_time, schemas_list)
        self._schema_cache: Dict[Any, Optional[Tuple[int, List[Dict[str, Any]]]]] = {
            True: None,  # Backward-compatible cache for enabled advertised tools
            False: None,  # Backward-compatible cache for all registered tools
            (True, True): None,  # Enabled tools including folded tools
        }
        self._schema_cache_version: int = 0
        self._schema_cache_lock = threading.RLock()
        self._batch_depth: int = 0
        self._batch_dirty: bool = False
        self._dynamic_availability_tools: Set[str] = set()
        self._normalized_name_winners: Dict[str, str] = {}
        self._tool_normalized_names: Dict[str, str] = {}
        self._tool_sources: Dict[str, Any] = {}
        self._deduplication_whitelist: Set[str] = set()
        self._deduplication_blacklist: Set[str] = set()

        # Tool deduplication (optional, based on settings)
        self._deduplicator: Optional[Any] = None
        self._deduplication_enabled: bool = False
        try:
            from victor.config.tool_settings import get_tool_settings

            tool_settings = get_tool_settings()
            if tool_settings.enable_tool_deduplication:
                from victor.tools.deduplication import (
                    DeduplicationConfig,
                    ToolDeduplicator,
                )

                config = DeduplicationConfig(
                    enabled=tool_settings.enable_tool_deduplication,
                    priority_order=tool_settings.deduplication_priority_order,
                    whitelist=tool_settings.deduplication_whitelist,
                    blacklist=tool_settings.deduplication_blacklist,
                    naming_enforcement=tool_settings.deduplication_naming_enforcement,
                    semantic_similarity_threshold=tool_settings.deduplication_semantic_threshold,
                )
                self._deduplicator = ToolDeduplicator(config)
                self._deduplication_enabled = True
                self._deduplication_whitelist = set(config.whitelist)
                self._deduplication_blacklist = set(config.blacklist)
        except ImportError:
            pass  # Deduplication not available

        # Strategy pattern support (when flag enabled)
        # Initialize strategy registry if flag is enabled
        self._strategy_registry: Optional["ToolRegistrationStrategyRegistry"] = None
        try:
            from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

            if get_feature_flag_manager().is_enabled(
                FeatureFlag.USE_STRATEGY_BASED_TOOL_REGISTRATION
            ):
                from victor.tools.registration.registry import (
                    get_tool_registration_strategy_registry,
                )

                self._strategy_registry = get_tool_registration_strategy_registry()
        except ImportError:
            pass  # Feature flags not available

    @property
    def _tools(self) -> Dict[str, Any]:
        """Alias for _items to maintain backward compatibility."""
        return self._items

    @_tools.setter
    def _tools(self, value: Dict[str, Any]) -> None:
        """Setter for _tools alias."""
        self._items = value

    def _extract_tool_name(self, tool: Any) -> str:
        """Extract tool name from various tool types.

        Args:
            tool: Tool object (BaseTool, decorated function, etc.)

        Returns:
            Tool name
        """
        if hasattr(tool, "name"):
            return tool.name
        elif hasattr(tool, "__name__"):
            return tool.__name__
        else:
            return str(tool)

    def _prepare_deduplicated_registration(
        self,
        tool: Any,
        tool_name: str,
    ) -> tuple[bool, Optional[str], Optional[Any]]:
        """Resolve the outcome for a deduplicated registration attempt.

        Args:
            tool: Tool to check
            tool_name: Extracted tool name

        Returns:
            Tuple of ``(should_register, losing_tool_name, tool_source)``.
        """
        if not self._deduplication_enabled or not self._deduplicator:
            return True, None, None

        if tool_name in self._deduplication_blacklist:
            logger.debug("Skipping blacklisted tool '%s' during registration", tool_name)
            return False, None, None

        normalized_name = self._normalize_name(tool_name)
        new_tool_source = self._detect_tool_source(tool)
        existing_name = self._normalized_name_winners.get(normalized_name)

        if existing_name is None or existing_name == tool_name:
            return True, None, new_tool_source

        if existing_name in self._deduplication_whitelist:
            logger.debug(
                "Skipping '%s' because whitelisted tool '%s' already owns normalized name '%s'",
                tool_name,
                existing_name,
                normalized_name,
            )
            return False, None, new_tool_source

        if tool_name in self._deduplication_whitelist:
            logger.debug(
                "Replacing '%s' with whitelisted tool '%s' for normalized name '%s'",
                existing_name,
                tool_name,
                normalized_name,
            )
            return True, existing_name, new_tool_source

        existing_source = self._tool_sources.get(existing_name)
        if existing_source is None:
            existing_tool = self._tools.get(existing_name)
            if existing_tool is None:
                self._normalized_name_winners.pop(normalized_name, None)
                return True, None, new_tool_source
            existing_source = self._detect_tool_source(existing_tool)

        comparison = self._compare_sources(existing_source, new_tool_source)
        if comparison >= 0:
            logger.debug(
                "Skipping '%s' (source=%s) in favor of '%s' (source=%s)",
                tool_name,
                new_tool_source.value,
                existing_name,
                existing_source.value,
            )
            return False, None, new_tool_source

        logger.debug(
            "Replacing '%s' (source=%s) with '%s' (source=%s) for normalized name '%s'",
            existing_name,
            existing_source.value,
            tool_name,
            new_tool_source.value,
            normalized_name,
        )
        return True, existing_name, new_tool_source

    def _detect_tool_source(self, tool: Any) -> Any:
        """Detect tool source from metadata or heuristics.

        Args:
            tool: Tool to analyze

        Returns:
            Detected ToolSource
        """
        # Import here to avoid circular dependency
        from victor.tools.deduplication import ToolSource

        # Check for source metadata
        if hasattr(tool, "_tool_source"):
            try:
                return ToolSource(tool._tool_source)
            except (TypeError, ValueError):
                logger.debug(
                    "Ignoring invalid _tool_source metadata on %s: %r",
                    self._extract_tool_name(tool),
                    tool._tool_source,
                )

        # Heuristic detection based on tool name
        tool_name = self._extract_tool_name(tool).lower()

        if tool_name.startswith("lgc_") or tool_name.startswith("langchain_"):
            return ToolSource.LANGCHAIN
        elif tool_name.startswith("mcp_"):
            return ToolSource.MCP
        elif tool_name.startswith("plg_"):
            return ToolSource.PLUGIN
        else:
            return ToolSource.NATIVE

    def _normalize_name(self, name: str) -> str:
        """Normalize tool name for conflict detection.

        Args:
            name: Tool name to normalize

        Returns:
            Normalized name
        """
        normalized = name.lower()

        # Remove source prefixes
        for prefix in ["lgc_", "langchain_", "mcp_", "plg_", "plugin_"]:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break

        # Normalize separators
        normalized = normalized.replace("_", " ").replace("-", " ")

        # Remove extra whitespace
        normalized = " ".join(normalized.split())

        return normalized

    def _compare_sources(self, source1: Any, source2: Any) -> int:
        """Compare two tool sources by priority.

        Args:
            source1: First tool source
            source2: Second tool source

        Returns:
            Positive if source1 has higher priority, negative if source2 has higher priority, 0 if equal
        """
        return source1.priority_weight - source2.priority_weight

    def _tool_requires_dynamic_availability(self, tool: Any) -> bool:
        """Return whether tool availability can change without registry mutation."""
        requires_configuration = getattr(tool, "requires_configuration", False)
        if callable(requires_configuration):
            requires_configuration = requires_configuration()
        return bool(requires_configuration)

    def _track_registered_tool(
        self,
        name: str,
        tool: Any,
        *,
        tool_source: Optional[Any] = None,
    ) -> None:
        """Update auxiliary indexes for a registered tool."""
        normalized_name = self._normalize_name(name)
        self._tool_normalized_names[name] = normalized_name

        if self._tool_requires_dynamic_availability(tool):
            self._dynamic_availability_tools.add(name)
        else:
            self._dynamic_availability_tools.discard(name)

        if self._deduplication_enabled:
            source = tool_source or self._detect_tool_source(tool)
            self._tool_sources[name] = source
            self._normalized_name_winners[normalized_name] = name

    def _remove_registered_tool_state(self, name: str) -> None:
        """Remove auxiliary index state for a tool."""
        self._tool_enabled.pop(name, None)
        self._dynamic_availability_tools.discard(name)
        normalized_name = self._tool_normalized_names.pop(name, None)
        self._tool_sources.pop(name, None)

        if (
            normalized_name is not None
            and self._normalized_name_winners.get(normalized_name) == name
        ):
            self._normalized_name_winners.pop(normalized_name, None)

    def _remove_tool_without_invalidation(self, name: str) -> bool:
        """Remove a tool and its bookkeeping without touching schema cache state."""
        existed = name in self._tools
        if not existed:
            self._remove_registered_tool_state(name)
            return False

        self._remove_registered_tool_state(name)
        return super().unregister(name)

    def _store_tool(
        self,
        name: str,
        tool: Any,
        enabled: bool,
        *,
        tool_source: Optional[Any] = None,
    ) -> None:
        """Store a tool and update auxiliary state."""
        super().register(name, tool)
        self._tool_enabled[name] = enabled
        self._track_registered_tool(name, tool, tool_source=tool_source)

    def _invalidate_schema_cache(self) -> None:
        """Invalidate the schema cache by bumping the version counter.

        Called when tools are registered, unregistered, enabled, or disabled.
        Thread-safe using the schema cache lock. Uses O(1) version counter
        instead of rebuilding tool name tuples for validation.

        If batch_update() is active, defers invalidation until the batch ends.
        """
        if self._batch_depth > 0:
            self._batch_dirty = True
            return
        with self._schema_cache_lock:
            self._schema_cache_version += 1

    def _wrap_hook(
        self, hook: Union[Hook, Callable], critical: bool = False, name: str = ""
    ) -> Hook:
        """Wrap a callable into a Hook object if needed."""
        if isinstance(hook, Hook):
            return hook
        return Hook(
            callback=hook,
            name=name or getattr(hook, "__name__", "hook"),
            critical=critical,
        )

    def register_before_hook(
        self,
        hook: Union[Hook, Callable[[str, Dict[str, Any]], None]],
        critical: bool = False,
        name: str = "",
    ) -> None:
        """Register a hook to be called before a tool is executed.

        Args:
            hook: Hook instance or callable that takes (tool_name, arguments)
            critical: If True, hook failure will block tool execution
            name: Human-readable name for the hook (used for error messages)
        """
        wrapped = self._wrap_hook(hook, critical=critical, name=name)
        self._before_hooks.append(wrapped)

    def register_after_hook(
        self,
        hook: Union[Hook, Callable],
        critical: bool = False,
        name: str = "",
    ) -> None:
        """Register a hook to be called after a tool is executed.

        Args:
            hook: Hook instance or callable that takes (tool_result,)
            critical: If True, hook failure will raise an error
            name: Human-readable name for the hook (used for error messages)
        """
        wrapped = self._wrap_hook(hook, critical=critical, name=name)
        self._after_hooks.append(wrapped)

    def register(self, *args, enabled: bool = True, **kwargs) -> None:  # type: ignore[override]
        """Register a tool - supports both signatures for flexibility.

        Signatures:
            register(tool)           - Auto-extracts name from tool object (common case)
            register(key, value)     - Explicit key-value (LSP-compatible with BaseRegistry)

        This unified method provides both backwards compatibility and LSP compliance.
        The first form is preferred for tool registration; the second allows
        ToolRegistry to be used where BaseRegistry is expected.

        When feature flag USE_STRATEGY_BASED_TOOL_REGISTRATION is enabled,
        this method uses the strategy pattern to automatically determine the
        appropriate registration strategy based on the tool type.

        When tool deduplication is enabled, this method checks for conflicts
        with already-registered tools and skips registration if a higher-priority
        tool with the same normalized name exists.

        Args:
            *args: Either (tool,) or (key, value)
            enabled: Whether the tool is enabled by default (default: True)

        Examples:
            registry.register(my_tool)                    # Auto-extract name
            registry.register("custom_name", my_tool)     # Explicit name
            registry.register(decorated_function)         # From @tool decorator
        """
        if len(args) == 1:
            # Single argument: register(tool) - extract name automatically
            tool = args[0]

            # LazyToolProxy — register directly (bypass strategy pattern)
            if hasattr(tool, "is_loaded") and hasattr(tool, "execute"):
                tool_name = tool.name
                should_register, losing_tool_name, tool_source = (
                    self._prepare_deduplicated_registration(tool, tool_name)
                )
                if not should_register:
                    return
                if losing_tool_name is not None:
                    self._remove_tool_without_invalidation(losing_tool_name)
                self._store_tool(tool_name, tool, enabled, tool_source=tool_source)
                self._invalidate_schema_cache()
                return

            # Check feature flag for strategy-based registration
            try:
                from victor.core.feature_flags import (
                    get_feature_flag_manager,
                    FeatureFlag,
                )

                if get_feature_flag_manager().is_enabled(
                    FeatureFlag.USE_STRATEGY_BASED_TOOL_REGISTRATION
                ):
                    return self._register_with_strategy(tool, enabled)
            except ImportError:
                pass  # Feature flags not available

            # Use existing implementation
            if hasattr(tool, "Tool"):  # It's a decorated function
                tool_instance = tool.Tool
                tool_name = tool_instance.name
                tool_to_register = tool_instance
            elif isinstance(tool, self._BaseTool):  # It's a BaseTool instance
                tool_name = tool.name
                tool_to_register = tool
            elif hasattr(tool, "name") and hasattr(tool, "execute"):
                # LazyToolProxy or duck-typed tool with name + execute
                tool_name = tool.name
                tool_to_register = tool
            else:
                raise TypeError(
                    "Can only register BaseTool instances, @tool functions, or LazyToolProxy"
                )
        elif len(args) == 2:
            # Two arguments: register(key, value) - LSP-compatible
            key, value = args
            tool_name = key
            tool_to_register = value
        else:
            raise TypeError(
                f"register() takes 1 or 2 positional arguments but {len(args)} were given"
            )

        should_register, losing_tool_name, tool_source = self._prepare_deduplicated_registration(
            tool_to_register,
            tool_name,
        )
        if not should_register:
            return
        if losing_tool_name is not None:
            self._remove_tool_without_invalidation(losing_tool_name)
        self._store_tool(tool_name, tool_to_register, enabled, tool_source=tool_source)

        # Invalidate schema cache after registration
        self._invalidate_schema_cache()

    def _register_with_strategy(self, tool: Any, enabled: bool) -> None:
        """Register using strategy pattern.

        Args:
            tool: Tool to register
            enabled: Whether tool is enabled
        """
        from victor.tools.registration.registry import (
            get_tool_registration_strategy_registry,
        )

        if self._strategy_registry is None:
            self._strategy_registry = get_tool_registration_strategy_registry()

        strategy = self._strategy_registry.get_strategy_for(tool)
        if strategy is None:
            raise TypeError(f"No registration strategy found for tool type: {type(tool)}")

        strategy.register(self, tool, enabled)

    def _register_direct(self, name: str, tool: Any, enabled: bool = True) -> None:
        """Directly register a tool by name (used by strategies).

        This method is called by registration strategies to perform
        the actual registration with the BaseRegistry.

        Args:
            name: Tool name
            tool: Tool instance
            enabled: Whether tool is enabled
        """
        should_register, losing_tool_name, tool_source = self._prepare_deduplicated_registration(
            tool,
            name,
        )
        if not should_register:
            return
        if losing_tool_name is not None:
            self._remove_tool_without_invalidation(losing_tool_name)
        self._store_tool(name, tool, enabled, tool_source=tool_source)
        self._invalidate_schema_cache()

    def add_custom_strategy(self, strategy: "ToolRegistrationStrategy") -> None:
        """Add a custom registration strategy.

        Allows extending tool registration without modifying core code.

        Example:
            class PydanticModelStrategy:
                def can_handle(self, tool):
                    try:
                        from pydantic import BaseModel
                        return isinstance(tool, BaseModel)
                    except ImportError:
                        return False

                def register(self, registry, tool, enabled=True):
                    wrapper = self._create_wrapper(tool)
                    registry._register_direct(wrapper.name, wrapper, enabled)

                @property
                def priority(self):
                    return 75

            registry = ToolRegistry()
            registry.add_custom_strategy(PydanticModelStrategy())

        Args:
            strategy: Strategy to add
        """
        from victor.tools.registration.registry import (
            get_tool_registration_strategy_registry,
        )

        if self._strategy_registry is None:
            self._strategy_registry = get_tool_registration_strategy_registry()

        self._strategy_registry.register_strategy(strategy)

    # Alias for backwards compatibility with code using register_tool()
    def register_tool(self, tool: Any, enabled: bool = True) -> None:
        """Alias for register(tool). Kept for backwards compatibility."""
        self.register(tool, enabled=enabled)

    def register_dict(self, tool_dict: Dict[str, Any], enabled: bool = True) -> None:
        """Register a tool from a dictionary definition.

        Used primarily for MCP tool definitions that come as dictionaries.

        Args:
            tool_dict: Dictionary with 'name', 'description', and 'parameters' keys
            enabled: Whether the tool is enabled by default (default: True)
        """
        from victor.tools.base import BaseTool, ToolResult

        name = tool_dict.get("name", "")
        description = tool_dict.get("description", "")
        parameters = tool_dict.get("parameters", {"type": "object", "properties": {}})

        # Create a wrapper tool that stores the dictionary definition
        # This is a placeholder - actual execution is handled by mcp_call
        class DictTool(BaseTool):
            @property
            def name(self) -> str:
                return name

            @property
            def description(self) -> str:
                return description

            @property
            def parameters(self) -> Dict[str, Any]:
                return parameters

            async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
                # MCP tools are executed via mcp_call, not directly
                return ToolResult(
                    success=False,
                    output=None,
                    error="MCP tools should be called via mcp_call",
                )

        self.register(DictTool(), enabled=enabled)

    def unregister(self, name: str) -> bool:  # type: ignore[override]
        """Unregister a tool.

        Args:
            name: Tool name to unregister

        Returns:
            True if the tool was found and removed, False otherwise
        """
        result = self._remove_tool_without_invalidation(name)

        # Invalidate schema cache after unregistration
        if result:
            self._invalidate_schema_cache()

        return result

    def enable_tool(self, name: str) -> bool:
        """Enable a tool by name.

        Args:
            name: Tool name to enable

        Returns:
            True if tool exists and was enabled, False otherwise
        """
        if name in self._tools:
            self._tool_enabled[name] = True
            # Invalidate schema cache after enabling
            self._invalidate_schema_cache()
            return True
        return False

    def disable_tool(self, name: str) -> bool:
        """Disable a tool by name.

        Args:
            name: Tool name to disable

        Returns:
            True if tool exists and was disabled, False otherwise
        """
        if name in self._tools:
            self._tool_enabled[name] = False
            # Invalidate schema cache after disabling
            self._invalidate_schema_cache()
            return True
        return False

    def is_tool_enabled(self, name: str) -> bool:
        """Check if a tool is enabled.

        Args:
            name: Tool name

        Returns:
            True if tool is enabled, False otherwise
        """
        return self._tool_enabled.get(name, False)

    def set_tool_states(self, tool_states: Dict[str, bool]) -> None:
        """Set enabled/disabled states for multiple tools.

        Args:
            tool_states: Dictionary mapping tool names to enabled state
        """
        changed = False
        for name, enabled in tool_states.items():
            if name in self._tools:
                self._tool_enabled[name] = enabled
                changed = True

        # Invalidate schema cache if any states changed
        if changed:
            self._invalidate_schema_cache()

    def get_tool_states(self) -> Dict[str, bool]:
        """Get enabled/disabled states for all tools.

        Returns:
            Dictionary mapping tool names to enabled state
        """
        return self._tool_enabled.copy()

    @contextmanager
    def batch_update(self) -> Generator[None, None, None]:
        """Batch multiple tool mutations with a single cache invalidation.

        Use when registering, enabling, or disabling many tools at once
        (e.g., during startup or vertical activation) to avoid repeated
        schema cache rebuilds.

        Example::

            with registry.batch_update():
                registry.register(tool_a)
                registry.register(tool_b)
                registry.enable_tool("tool_a")
                registry.enable_tool("tool_b")
            # Cache invalidated once here
        """
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0 and self._batch_dirty:
                self._batch_dirty = False
                with self._schema_cache_lock:
                    self._schema_cache_version += 1

    def get(self, name: str) -> Optional[Any]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return super().get(name)

    def list_tools(self, only_enabled: bool = True, include_folded: bool = False) -> list:
        """List all registered tools.

        Args:
            only_enabled: If True, only return enabled tools (default: True)
            include_folded: If True, include tools folded into other advertised
                tools. Folded tools remain executable but are hidden from
                default enabled listings to reduce schema/tool-count burden.

        Returns:
            List of tool instances
        """
        if only_enabled:
            from victor.tools.folding import should_advertise_tool

            return [
                tool
                for name, tool in self._tools.items()
                if self._tool_enabled.get(name, False)
                and self._tool_is_available(tool)
                and should_advertise_tool(name, include_folded=include_folded)
            ]
        return list(self._tools.values())

    def get_tool_schemas(
        self, only_enabled: bool = True, include_folded: bool = False
    ) -> List[Dict[str, Any]]:
        """Get JSON schemas for all tools with caching.

        Uses O(1) version-counter invalidation. The cache is bumped automatically
        when tools are registered, unregistered, enabled, or disabled.

        Args:
            only_enabled: If True, only return schemas for enabled tools (default: True)
            include_folded: If True, include schemas for folded tools.

        Returns:
            List of tool JSON schemas
        """
        with self._schema_cache_lock:
            current_version = self._schema_cache_version
            use_cache = not (only_enabled and self._has_dynamic_availability_checks())
            if only_enabled and include_folded:
                cache_key: Any = (True, True)
            else:
                cache_key = bool(only_enabled)

            # O(1) cache check via version counter
            if use_cache:
                cache_entry = self._schema_cache.get(cache_key)
                if cache_entry is not None:
                    cached_version, cached_schemas = cache_entry
                    if cached_version == current_version:
                        return cached_schemas

            # Cache miss - generate schemas
            if only_enabled:
                from victor.tools.folding import should_advertise_tool

                schemas = [
                    tool.to_json_schema()
                    for name, tool in self._tools.items()
                    if self._tool_enabled.get(name, False)
                    and self._tool_is_available(tool)
                    and should_advertise_tool(name, include_folded=include_folded)
                ]
            else:
                schemas = [tool.to_json_schema() for tool in self._tools.values()]

            # Update cache with current version
            if use_cache:
                self._schema_cache[cache_key] = (current_version, schemas)

            return schemas

    def _tool_is_available(self, tool: Any) -> bool:
        """Return whether a tool is currently available for selection."""
        checker = getattr(tool, "is_available", None)
        if not callable(checker):
            return True
        return bool(checker())

    def _has_dynamic_availability_checks(self) -> bool:
        """Detect tools whose availability changes outside registry mutations."""
        return bool(self._dynamic_availability_tools)

    def clear(self) -> None:  # type: ignore[override]
        """Clear all registered tools and auxiliary registry state."""
        super().clear()
        self._tool_enabled.clear()
        self._dynamic_availability_tools.clear()
        self._normalized_name_winners.clear()
        self._tool_normalized_names.clear()
        self._tool_sources.clear()
        self._invalidate_schema_cache()

    def get_tool_cost(self, name: str) -> Optional[CostTier]:
        """Get the cost tier for a tool.

        Args:
            name: Tool name

        Returns:
            CostTier enum value or None if tool not found
        """
        tool = self.get(name)
        if tool:
            return tool.cost_tier
        return None

    def get_tools_by_cost(self, max_tier: CostTier = CostTier.HIGH, only_enabled: bool = True):
        """Get tools filtered by maximum cost tier.

        Args:
            max_tier: Maximum cost tier to include
            only_enabled: If True, only return enabled tools

        Returns:
            List of tools at or below the specified cost tier
        """
        tools = self.list_tools(only_enabled=only_enabled)
        return [t for t in tools if t.cost_tier.weight <= max_tier.weight]

    def get_cost_summary(self, only_enabled: bool = True) -> Dict[str, List[str]]:
        """Get a summary of tools grouped by cost tier.

        Args:
            only_enabled: If True, only include enabled tools

        Returns:
            Dictionary mapping cost tier names to lists of tool names
        """
        summary: Dict[str, List[str]] = {tier.value: [] for tier in CostTier}
        for tool in self.list_tools(only_enabled=only_enabled):
            summary[tool.cost_tier.value].append(tool.name)
        return summary

    async def execute(self, name: str, _exec_ctx: Dict[str, Any], **kwargs: Any):
        """Execute a tool by name.

        Args:
            name: Tool name
            _exec_ctx: Framework execution context (reserved name to avoid collision
                      with tool parameters). Contains shared resources.
            **kwargs: Tool parameters

        Returns:
            ToolResult with execution outcome
        """
        from victor.tools.base import ToolResult

        # Trigger before-execution hooks
        for hook in self._before_hooks:
            hook(name, kwargs)

        tool = self.get(name)
        if tool is None:
            result = ToolResult(
                success=False,
                output=None,
                error=f"Tool '{name}' not found",
                metadata=None,
            )
        elif not self.is_tool_enabled(name):
            result = ToolResult(
                success=False,
                output=None,
                error=f"Tool '{name}' is disabled",
                metadata=None,
            )
        else:
            # Use detailed validation for better error messages
            validation = tool.validate_parameters_detailed(**kwargs)
            if not validation.valid:
                error_msg = f"Invalid parameters for tool '{name}': " + "; ".join(validation.errors)
                result = ToolResult(
                    success=False,
                    output=None,
                    error=error_msg,
                    metadata={"invalid_params": validation.invalid_params},
                )
            else:
                try:
                    result = await tool.execute(_exec_ctx, **kwargs)
                except Exception as e:
                    result = ToolResult(
                        success=False,
                        output=None,
                        error=f"Tool execution failed: {str(e)}",
                        metadata={"exception": type(e).__name__},
                    )

        # Trigger after-execution hooks
        for hook in self._after_hooks:
            hook(result)

        return result

    # ========================================================================
    # Plugin-Based Tool Registration
    # ========================================================================

    def register_plugin(self, plugin: Any) -> None:
        """Register a tool plugin with the registry.

        Tool plugins enable external verticals to dynamically register
        tools at runtime. The plugin's register() method is called to perform
        the actual registration.

        This method accepts any object with a register() method, including:
        - VictorPlugin from victor_contracts.core.plugins
        - ToolFactoryAdapter from victor_contracts.verticals.protocols.tool_plugins
        - Any class with a register(registry) method

        Args:
            plugin: Object with register() method (VictorPlugin protocol)

        Raises:
            AttributeError: If plugin doesn't have a register() method
            Exception: If plugin.register() raises an exception

        Example:
            from victor_contracts.verticals.protocols import ToolPluginHelper

            # Using helper
            tools = {"code_search": CodeSearchTool(), "refactor": RefactoringTool()}
            plugin = ToolPluginHelper.from_instances(tools)
            registry.register_plugin(plugin)

        Example:
            # Custom plugin class
            class MyVerticalPlugin:
                def register(self, registry: ToolRegistry) -> None:
                    registry.register(CodeSearchTool())
                    registry.register(RefactoringTool())

            plugin = MyVerticalPlugin()
            registry.register_plugin(plugin)
        """
        if not hasattr(plugin, "register"):
            raise AttributeError(f"Tool plugin must have a 'register' method. Got: {type(plugin)}")

        # Call plugin's register method
        plugin.register(self)

    def discover_plugins(self, entry_point_group: str = "victor.plugins") -> int:
        """Discover and register tool plugins from entry points.

        Uses UnifiedEntryPointRegistry for efficient single-pass discovery
        instead of redundant importlib.metadata.entry_points() calls.

        Handles multiple plugin formats:
        - Objects with a ``register`` method (ToolPlugin protocol)
        - Lists/tuples of tool instances (wrapped via ToolPluginHelper)

        Args:
            entry_point_group: Entry point group to scan (default: victor.plugins)

        Returns:
            Number of plugins discovered and registered

        Example:
            # In setup.py / pyproject.toml:
            # [project.entry-points."victor.plugins"]
            # my_plugin = "my_package.plugin:MyPlugin"

            # At runtime:
            count = registry.discover_plugins()
            print(f"Registered {count} tool plugins")
        """
        from victor.framework.entry_point_registry import get_entry_point_objects

        count = 0

        try:
            group_eps = get_entry_point_objects(entry_point_group)

            for ep in group_eps:
                try:
                    plugin = ep.load()

                    if hasattr(plugin, "register"):
                        self.register_plugin(plugin)
                        count += 1
                    elif isinstance(plugin, (list, tuple)):
                        from victor_contracts.verticals.protocols import (
                            ToolPluginHelper,
                        )

                        tools = {f"tool_{i}": tool for i, tool in enumerate(plugin)}
                        adapter = ToolPluginHelper.from_instances(tools)
                        self.register_plugin(adapter)
                        count += 1

                except Exception as e:
                    import logging

                    logger = logging.getLogger(__name__)
                    ep_name = getattr(ep, "name", "unknown")
                    logger.warning(f"Failed to load tool plugin from '{ep_name}': {e}")

        except Exception:
            pass

        return count

    # Backward-compatible alias for discover_plugins()
    register_from_entry_points = discover_plugins

    # ========================================================================
    # Lightweight Tool Discovery (for CLI)
    # ========================================================================

    @classmethod
    def discover_lightweight(cls) -> List[Tuple[str, str, str]]:
        """Discover tools without full orchestrator initialization.

        Dynamically discovers tools from the victor/tools directory by scanning
        for @tool decorated functions and BaseTool subclasses. This is useful
        for CLI commands that need to list tools without the overhead of
        initializing the full agent.

        Returns:
            List of tuples: (name, description, cost_tier)
        """
        from victor.tools.base import BaseTool
        import importlib
        import inspect
        import os

        tools_dir = os.path.join(os.path.dirname(__file__), "..", "tools")
        excluded_files = {
            "__init__.py",
            "base.py",
            "decorators.py",
            "semantic_selector.py",
            "enums.py",
            "registry.py",
            "metadata.py",
            "metadata_registry.py",
            "tool_names.py",
            "output_utils.py",
            "shared_ast_utils.py",
            "dependency_graph.py",
            "plugin_registry.py",
        }

        discovered_tools = []

        for filename in os.listdir(tools_dir):
            if filename.endswith(".py") and filename not in excluded_files:
                module_name = f"victor.tools.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for _name, obj in inspect.getmembers(module):
                        # Check @tool decorated functions
                        if inspect.isfunction(obj) and getattr(obj, "_is_tool", False):
                            tool_instance = getattr(obj, "Tool", None)
                            if tool_instance:
                                name = tool_instance.name
                                description = tool_instance.description or "No description"
                                cost_tier = getattr(tool_instance, "cost_tier", None)
                                cost_str = cost_tier.value if cost_tier else "unknown"
                                discovered_tools.append((name, description, cost_str))
                        # Check BaseTool class instances
                        elif (
                            inspect.isclass(obj)
                            and issubclass(obj, BaseTool)
                            and obj is not BaseTool
                            and hasattr(obj, "name")
                        ):
                            try:
                                tool_instance = obj()
                                name = tool_instance.name
                                description = tool_instance.description or "No description"
                                cost_tier = getattr(tool_instance, "cost_tier", None)
                                cost_str = cost_tier.value if cost_tier else "unknown"
                                discovered_tools.append((name, description, cost_str))
                            except Exception:
                                # Skip tools that can't be instantiated
                                pass
                except Exception:
                    # Log but continue with other modules
                    pass

        return discovered_tools
