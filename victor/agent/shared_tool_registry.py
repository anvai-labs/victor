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

"""Shared tool registry for memory-efficient concurrent sessions.

This module provides a singleton registry that shares tool class definitions
across multiple AgentOrchestrator instances. This significantly reduces memory
footprint when running concurrent sessions, as tool classes are discovered
and stored only once.

Design Pattern: Singleton + Flyweight
====================================
- Singleton: Ensures only one SharedToolRegistry instance exists
- Flyweight: Tool classes are shared objects, while per-session state is kept separate

Memory Savings:
- Without sharing: 100 sessions x 61 tools x ~150KB = ~900MB
- With sharing: 1 registry + 100 sessions x per-session state = ~20MB

Usage:
    from victor.agent.shared_tool_registry import SharedToolRegistry

    # Get the singleton instance
    registry = SharedToolRegistry.get_instance()

    # Get all tool classes (cached after first call)
    tool_classes = registry.get_tool_classes()

    # Create a new tool instance (for per-session use if needed)
    tool_instance = registry.create_tool_instance("read_file")

    # For test isolation
    SharedToolRegistry.reset_instance()
"""

import importlib
import inspect
import logging
import os
import threading
from typing import Any, Dict, Iterable, List, Optional, Set, Type

logger = logging.getLogger(__name__)


# Web tools that should be filtered out in airgapped mode
WEB_TOOL_NAMES: Set[str] = {
    "web",
    "web_search",
    "web_fetch",
    "http_tool",
    "browser_tool",
    "http_request",
}


# Tools that remain callable through the umbrella ``graph(mode=...)`` surface but
# should not be advertised as separate default schemas. Keeping them out of the
# registered/default catalog reduces prompt size and avoids cache churn from a
# long tail of near-duplicate graph functions.
#
# NOTE: This set currently only affects the non-lazy full-discovery registration
# path (``get_all_tools_for_registration``). The production lazy path
# (``get_bootstrap_tools_for_registration``) and ``tool_selection`` do not yet
# consult it, so it is not a general "hide from LLM schema" lever. True
# advertisement-hiding for the granular primitives subsumed by the unified
# command domains (fs/shell/git/web/code) is a follow-up that belongs at the
# selection/schema-build layer.
SCHEMA_HIDDEN_TOOL_NAMES: Set[str] = {
    "graph_analytics",
    "graph_dependencies",
    "graph_neighbors",
    "graph_path",
    "graph_patterns",
    "graph_query",
    "graph_search",
    "graph_semantic",
    "graph_semantic_search",
    "impact_analysis",
}


BOOTSTRAP_TOOL_SPECS: Dict[str, tuple[str, str]] = {
    "git": ("victor.tools.unified.git_tool", "git_tool"),
    "search": ("victor.tools.unified.search_tool", "search_tool"),
    "code": ("victor.tools.unified.code_tool", "code_tool"),
    "web": ("victor.tools.unified.web_tool", "web_tool"),
    "read": ("victor.tools.filesystem", "read"),
    "write": ("victor.tools.filesystem", "write"),
    "ls": ("victor.tools.filesystem", "ls"),
    "project_overview": ("victor.tools.filesystem", "project_overview"),
    "edit": ("victor.tools.file_editor_tool", "edit"),
    "shell": ("victor.tools.bash", "shell"),
    "web_search": ("victor.tools.web_search_tool", "web_search"),
    "web_fetch": ("victor.tools.web_search_tool", "web_fetch"),
}


DEMAND_TOOL_SPECS: Dict[str, tuple[str, str]] = {
    **BOOTSTRAP_TOOL_SPECS,
    # graph lives in the optional victor-coding package (a vertical capability,
    # not a core tool) — hence demand-loaded, not bootstrap. The spec must point
    # at the real module; a stale "victor.tools.graph_tool" path (which doesn't
    # exist) silently failed demand-loading, leaving graph registrable only via
    # the victor-coding entry-point. Import fails gracefully when victor-coding
    # is absent (caught in _load_tool_spec → returns None).
    "graph": ("victor_coding.tools.graph_tool", "graph"),
}


GRAPH_DEMAND_KEYWORDS = frozenset(
    {
        "graph",
        "callers",
        "callees",
        "call graph",
        "dependency graph",
        "dependencies",
        "impact analysis",
        "pagerank",
        "centrality",
        "neighbors",
        "trace",
        "call flow",
    }
)


class SharedToolRegistry:
    """Singleton registry for sharing tool class definitions across sessions.

    This class provides a memory-efficient way to share tool definitions across
    multiple concurrent AgentOrchestrator instances. Instead of each session
    discovering and storing its own copy of tool classes, they all reference
    the same shared definitions.

    Thread-Safety:
        All operations are thread-safe. The singleton instance is protected by
        a lock, and tool discovery is performed atomically.

    Attributes:
        _instance: The singleton instance
        _lock: Threading lock for thread-safe singleton access
        _tool_classes: Cached dictionary mapping tool names to tool classes
        _decorated_tools: Cached dictionary mapping tool names to decorated functions
        _initialized: Flag indicating if tools have been discovered
    """

    _instance: Optional["SharedToolRegistry"] = None
    _lock: threading.Lock = threading.Lock()
    _instantiation_allowed: bool = False

    def __init__(self) -> None:
        """Initialize the SharedToolRegistry.

        Raises:
            RuntimeError: If called directly instead of via get_instance()
        """
        if not SharedToolRegistry._instantiation_allowed:
            raise RuntimeError(
                "SharedToolRegistry is a singleton. Use SharedToolRegistry.get_instance() "
                "to get the singleton instance."
            )

        self._tool_classes: Dict[str, Type[Any]] = {}
        self._decorated_tools: Dict[str, Any] = {}
        # Cache of instances created during discovery so that LazyToolProxy
        # can reuse them instead of re-instantiating the class on first use.
        # This eliminates the double-instantiation defect where _discover_tools()
        # created an instance to read .name and the proxy then called cls() again.
        self._tool_instances: Dict[str, Any] = {}
        self._initialized: bool = False
        self._discovery_lock: threading.Lock = threading.Lock()

        logger.debug("SharedToolRegistry instance created")

    @classmethod
    def get_instance(cls) -> "SharedToolRegistry":
        """Get the singleton instance of SharedToolRegistry.

        This method is thread-safe and ensures only one instance is created
        even when called concurrently from multiple threads.

        Returns:
            The singleton SharedToolRegistry instance
        """
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instantiation_allowed = True
                    try:
                        cls._instance = cls()
                    finally:
                        cls._instantiation_allowed = False
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance.

        This method is primarily intended for test isolation. It allows
        tests to start with a fresh SharedToolRegistry instance.

        Note:
            This method is thread-safe but should only be called during
            test setup/teardown, not during normal operation.
        """
        with cls._lock:
            cls._instance = None
            logger.debug("SharedToolRegistry instance reset")

    def get_tool_classes(self, airgapped_mode: bool = False) -> Dict[str, Type[Any]]:
        """Get all discovered tool classes.

        This method returns the cached dictionary of tool classes. If tools
        haven't been discovered yet, discovery is performed lazily.

        Args:
            airgapped_mode: If True, filter out web tools that require network

        Returns:
            Dictionary mapping tool names to their corresponding classes.
            For tools defined with @tool decorator, the class is retrieved
            from the .Tool attribute.

        Note:
            The returned dictionary should NOT be modified. If you need to
            filter tools, create a copy first.
        """
        if not self._initialized:
            with self._discovery_lock:
                if not self._initialized:
                    self._discover_tools()
                    self._initialized = True

        if airgapped_mode:
            return {
                name: cls for name, cls in self._tool_classes.items() if name not in WEB_TOOL_NAMES
            }

        return self._tool_classes

    def get_decorated_tools(self, airgapped_mode: bool = False) -> Dict[str, Any]:
        """Get all discovered decorated tool functions.

        This method returns decorated functions (those using @tool decorator)
        separately from class-based tools.

        Args:
            airgapped_mode: If True, filter out web tools that require network

        Returns:
            Dictionary mapping tool names to decorated functions
        """
        if not self._initialized:
            with self._discovery_lock:
                if not self._initialized:
                    self._discover_tools()
                    self._initialized = True

        if airgapped_mode:
            return {
                name: func
                for name, func in self._decorated_tools.items()
                if name not in WEB_TOOL_NAMES
            }

        return self._decorated_tools

    def get_tool_names(self, airgapped_mode: bool = False) -> List[str]:
        """Get list of all discovered tool names.

        Args:
            airgapped_mode: If True, filter out web tools that require network

        Returns:
            List of tool names
        """
        return list(self.get_tool_classes(airgapped_mode=airgapped_mode).keys())

    def create_tool_instance(self, tool_name: str) -> Optional[Any]:
        """Create a new instance of a tool.

        This method creates a fresh instance of the specified tool. Use this
        when you need a per-session tool instance with its own state.

        Args:
            tool_name: Name of the tool to instantiate

        Returns:
            New instance of the tool, or None if tool not found
        """
        # Ensure tools are discovered
        self.get_tool_classes()

        tool_class = self._tool_classes.get(tool_name)
        if tool_class is None:
            return None

        try:
            return tool_class()
        except Exception as e:
            logger.debug(f"Failed to create instance of {tool_name}: {e}")
            return None

    def _load_tool_spec(self, tool_name: str) -> Optional[Any]:
        """Load one known tool without scanning/importing the full tools tree."""
        spec = DEMAND_TOOL_SPECS.get(tool_name)
        if spec is None:
            return None

        module_name, member_name = spec
        try:
            module = importlib.import_module(module_name)
            tool_obj = getattr(module, member_name)
        except Exception as exc:
            logger.debug(
                "Failed to load bootstrap tool %s from %s: %s",
                tool_name,
                module_name,
                exc,
            )
            return None

        if inspect.isfunction(tool_obj) and getattr(tool_obj, "_is_tool", False):
            tool_instance = tool_obj.Tool
            self._tool_classes[tool_instance.name] = type(tool_instance)
            self._decorated_tools[tool_instance.name] = tool_obj
            return tool_instance

        try:
            tool_name_from_obj = getattr(tool_obj, "name", tool_name)
            self._tool_classes[tool_name_from_obj] = type(tool_obj)
            self._tool_instances[tool_name_from_obj] = tool_obj
            return tool_obj
        except Exception:
            return tool_obj

    def get_tools_for_names(
        self,
        tool_names: Iterable[str],
        *,
        airgapped_mode: bool = False,
    ) -> List[Any]:
        """Load specific known tools without forcing full catalog discovery."""
        result: List[Any] = []
        seen: Set[str] = set()
        for tool_name in tool_names:
            if tool_name in seen:
                continue
            seen.add(tool_name)
            if airgapped_mode and tool_name in WEB_TOOL_NAMES:
                continue
            if tool_name in SCHEMA_HIDDEN_TOOL_NAMES:
                continue
            tool_obj = self._load_tool_spec(tool_name)
            if tool_obj is not None:
                result.append(tool_obj)
        return result

    def get_bootstrap_tools_for_registration(self, airgapped_mode: bool = False) -> List[Any]:
        """Return the compact startup tool set without full filesystem discovery."""
        return self.get_tools_for_names(
            BOOTSTRAP_TOOL_SPECS.keys(),
            airgapped_mode=airgapped_mode,
        )

    def infer_demand_tools(self, text: str) -> List[str]:
        """Infer specialty tools that should be hydrated for a user request."""
        lowered = (text or "").lower()
        demand: List[str] = []
        if any(keyword in lowered for keyword in GRAPH_DEMAND_KEYWORDS):
            demand.append("graph")
        return demand

    def _discover_tools(self) -> None:
        """Discover and cache all tool classes from victor/tools directory.

        This method scans the victor/tools directory for tool definitions and
        stores references to their classes. It handles both:
        - Classes inheriting from BaseTool
        - Functions decorated with @tool

        Note:
            This method is called lazily on first access and only runs once
            (per singleton instance). Results are cached in _tool_classes.
        """
        # Import BaseTool for isinstance checks
        from victor.tools.base import BaseTool as BaseToolClass

        tools_dir = os.path.join(os.path.dirname(__file__), "..", "tools")
        excluded_files = {
            "__init__.py",
            "base.py",
            "decorators.py",
            "semantic_selector.py",
            "registry.py",
            "enums.py",
            "metadata.py",
            "metadata_registry.py",
            "selection_filters.py",
            "selection_common.py",
            "keyword_tool_selector.py",
            "hybrid_tool_selector.py",
            "output_utils.py",
            "shared_ast_utils.py",
            "common.py",
            "subprocess_executor.py",
            "tool_names.py",
            "composition.py",
            "plugin.py",
            "plugin_registry.py",
            "language_analyzer.py",
            "dependency_graph.py",
        }

        discovered_classes = 0
        discovered_decorated = 0

        for filename in os.listdir(tools_dir):
            if filename.endswith(".py") and filename not in excluded_files:
                module_name = f"victor.tools.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for member_name, obj in inspect.getmembers(module):
                        # Handle @tool decorated functions
                        if inspect.isfunction(obj) and getattr(obj, "_is_tool", False):
                            tool_instance = obj.Tool
                            tool_name = tool_instance.name
                            # Store the class, not the instance
                            self._tool_classes[tool_name] = type(tool_instance)
                            self._decorated_tools[tool_name] = obj
                            discovered_decorated += 1

                        # Handle BaseTool class instances
                        elif (
                            inspect.isclass(obj)
                            and issubclass(obj, BaseToolClass)
                            and obj is not BaseToolClass
                            and hasattr(obj, "name")
                        ):
                            # Only store the class reference if it's not abstract
                            try:
                                # Attempt to create an instance to get the name.
                                # NOTE: BaseTool.name is a @property, so we must
                                # instantiate to read it. We cache this instance so
                                # LazyToolProxy can reuse it instead of calling
                                # cls() again on first use (double-instantiation fix).
                                temp_instance = obj()
                                tool_name = temp_instance.name
                                self._tool_classes[tool_name] = obj
                                self._tool_instances[tool_name] = temp_instance
                                discovered_classes += 1
                            except Exception:
                                # Skip abstract classes or classes that can't be instantiated
                                pass

                except Exception as e:
                    logger.debug(f"Failed to load tools from {module_name}: {e}")

        # The flat scan above only visits top-level .py files. The unified
        # bash-style tools (fs/shell/search/web/code) live in the ``unified/``
        # subpackage; scan it with the same discovery logic so their @tool
        # decorators populate this registry's catalogs. Without this, the
        # unified tools are absent from get_all_tools_for_registration() and
        # therefore never registered on the full-discovery (full=True) path.
        unified_dir = os.path.join(tools_dir, "unified")
        if os.path.isdir(unified_dir):
            for filename in os.listdir(unified_dir):
                # Only the *_tool.py modules define @tool entrypoints; skip
                # registry/adapters/parser/__init__ helpers.
                if not filename.endswith("_tool.py"):
                    continue
                module_name = f"victor.tools.unified.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for _member_name, obj in inspect.getmembers(module):
                        if inspect.isfunction(obj) and getattr(obj, "_is_tool", False):
                            tool_instance = obj.Tool
                            tool_name = tool_instance.name
                            # The flat scan above runs first, so canonical tools
                            # (e.g. ``bash.shell``) already own their names. Skip
                            # unified duplicates to avoid silently shadowing them.
                            if tool_name in self._tool_classes:
                                continue
                            self._tool_classes[tool_name] = type(tool_instance)
                            self._decorated_tools[tool_name] = obj
                            discovered_decorated += 1
                except Exception as e:
                    logger.debug(f"Failed to load tools from {module_name}: {e}")

        logger.info(
            f"SharedToolRegistry discovered {len(self._tool_classes)} tools "
            f"({discovered_classes} classes, {discovered_decorated} decorated)"
        )

    def get_all_tools_for_registration(
        self,
        airgapped_mode: bool = False,
        *,
        include_schema_hidden: bool = False,
    ) -> List[Any]:
        """Get tool instances and decorated functions for registration.

        This method returns a list suitable for registering with ToolRegistry.
        It includes both class-based tool instances and decorated functions.

        Args:
            airgapped_mode: If True, filter out web tools that require network

        Returns:
            List of tool instances and decorated functions ready for registration
        """
        # Ensure tools are discovered
        self.get_tool_classes()

        result: List[Any] = []

        # Add decorated tools (functions with @tool decorator)
        for name, func in self.get_decorated_tools(airgapped_mode=airgapped_mode).items():
            if not include_schema_hidden and name in SCHEMA_HIDDEN_TOOL_NAMES:
                continue
            result.append(func)

        # Add class-based tools as lazy proxies (deferred initialization)
        # Tools are NOT instantiated until first execute() call — saves 1-2s startup
        from victor.tools.progressive import LazyToolProxy
        from victor.tools.enums import CostTier

        decorated_names = set(self._decorated_tools.keys())
        for name, cls in self.get_tool_classes(airgapped_mode=airgapped_mode).items():
            if name not in decorated_names:
                if not include_schema_hidden and name in SCHEMA_HIDDEN_TOOL_NAMES:
                    continue
                try:
                    cost_tier = getattr(cls, "cost_tier", CostTier.FREE)
                    if not isinstance(cost_tier, CostTier):
                        cost_tier = CostTier.FREE
                    description = getattr(cls, "description", "") or ""
                    if callable(description):
                        description = ""  # Skip property descriptors
                    # Reuse the instance created during discovery if available,
                    # so the proxy does not re-instantiate the class on first use.
                    cached_instance = self._tool_instances.get(name)
                    factory = (
                        (lambda inst=cached_instance: inst) if cached_instance is not None else cls
                    )
                    proxy = LazyToolProxy(
                        name=name,
                        factory=factory,
                        cost_tier=cost_tier,
                        description=description,
                    )
                    result.append(proxy)
                except Exception as e:
                    logger.debug(f"Skipped lazy proxy for {name}: {e}")

        return result
