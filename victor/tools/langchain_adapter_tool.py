# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""LangChain tool adapter — projects LangChain tools as native Victor tools.

Adapter Pattern: LangChainAdapterTool wraps a LangChain BaseTool into
Victor's BaseTool interface. The LLM sees native names and JSON Schema
parameters — indistinguishable from built-in tools.

Factory Pattern: LangChainToolProjector batch-creates adapters from
a list of LangChain tools with name collision handling.

Requires: pip install victor-ai[langchain]

Usage:
    from langchain_community.tools import DuckDuckGoSearchRun
    from victor.tools.langchain_adapter_tool import LangChainAdapterTool

    search = DuckDuckGoSearchRun()
    adapter = LangChainAdapterTool(search)
    tool_registry.register(adapter)

    # Or batch:
    from victor.tools.langchain_adapter_tool import LangChainToolProjector
    adapters = LangChainToolProjector.project([tool1, tool2, tool3])
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.tools.base import BaseTool, CostTier, ToolResult

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool as LCBaseTool

logger = logging.getLogger(__name__)

# Import ToolSource for metadata
try:
    from victor.tools.deduplication import ToolSource
except ImportError:
    # Fallback if deduplication not available
    class ToolSource:
        LANGCHAIN = "langchain"


# Constants for LangChain tool safeguards
MAX_LANGCHAIN_TOOLS = 10

# Default prefix for all LangChain tools (unified naming convention)
DEFAULT_LANGCHAIN_PREFIX = "lgc"

# Whitelist of allowed LangChain tools (unique functionality not available in Victor)
# Note: This is now used only for logging warnings - actual deduplication is handled by ToolDeduplicator
ALLOWED_LANGCHAIN_TOOLS = {
    "wikipedia",  # Unique functionality
    "wolfram_alpha",  # Unique functionality
    "tmdb",  # Movie database - unique
    "pubmed",  # Medical research - unique
    "arxiv",  # Academic papers - unique
}

# Blacklist of banned LangChain tools (duplicates of Victor built-in tools)
# Note: This is now used only for logging warnings - actual deduplication is handled by ToolDeduplicator
BANNED_LANGCHAIN_TOOLS = {
    "duckduckgo",  # Duplicate: web_search
    "google_search",  # Duplicate: web_search
    "requests",  # Duplicate: http
    "shell",  # Duplicate: shell
    "python_repl",  # Duplicate: sandbox
    "terminal",  # Duplicate: shell
    "file",  # Duplicate: read/write/edit
    "directory",  # Duplicate: ls
    "grep",  # Duplicate: code_search
}


def _are_tools_semantically_similar(name1: str, name2: str) -> bool:
    """Check if two tool names are semantically similar (potential duplicates).

    Uses simple heuristics:
    - Same base name (ignoring prefixes/suffixes)
    - Same functional keywords (search, fetch, request, etc.)

    Args:
        name1: First tool name
        name2: Second tool name

    Returns:
        True if tools are semantically similar, False otherwise
    """
    # Normalize names: lowercase, remove prefixes/suffixes
    norm1 = name1.lower().replace("_", " ")
    norm2 = name2.lower().replace("_", " ")

    # Extract base names (remove common prefixes)
    for prefix in ["lc", "langchain", "tool"]:
        norm1 = norm1.replace(prefix, "").strip()
        norm2 = norm2.replace(prefix, "").strip()

    # Check for exact match after normalization
    if norm1 == norm2:
        return True

    # Check for functional keyword overlap
    search_keywords = {"search", "find", "lookup", "query"}
    fetch_keywords = {"fetch", "get", "request", "http", "curl"}
    shell_keywords = {"shell", "terminal", "command", "exec", "run"}
    file_keywords = {"file", "read", "write", "edit", "directory"}

    # Both are search tools
    if any(kw in norm1 for kw in search_keywords) and any(kw in norm2 for kw in search_keywords):
        return True

    # Both are fetch tools
    if any(kw in norm1 for kw in fetch_keywords) and any(kw in norm2 for kw in fetch_keywords):
        return True

    # Both are shell tools
    if any(kw in norm1 for kw in shell_keywords) and any(kw in norm2 for kw in shell_keywords):
        return True

    # Both are file tools
    if any(kw in norm1 for kw in file_keywords) and any(kw in norm2 for kw in file_keywords):
        return True

    return False


def _check_langchain_tool_allowed(
    tool_name: str,
    existing_tools: List[str],
    langchain_count: int,
) -> tuple[bool, str]:
    """Check if a LangChain tool should be allowed to register.

    Enforces:
    1. Blacklist check (blocked duplicates)
    2. Whitelist check (if configured to be restrictive)
    3. Count limit (max MAX_LANGCHAIN_TOOLS)
    4. Duplicate detection (exact name)
    5. Semantic similarity detection

    Args:
        tool_name: Name of the LangChain tool to check
        existing_tools: List of already registered tool names
        langchain_count: Current number of registered LangChain tools

    Returns:
        Tuple of (allowed: bool, reason: str)
        - If allowed=True, tool can be registered
        - If allowed=False, reason explains why it was blocked
    """
    # Check blacklist (exact match)
    if tool_name.lower() in BANNED_LANGCHAIN_TOOLS:
        return (
            False,
            f"Tool '{tool_name}' is blacklisted (duplicate of built-in Victor tool)",
        )

    # Check whitelist (if configured to be restrictive)
    # Note: We're not enforcing whitelist strictly by default, only logging
    if ALLOWED_LANGCHAIN_TOOLS and tool_name.lower() not in ALLOWED_LANGCHAIN_TOOLS:
        logger.warning(
            "LangChain tool '%s' is not in whitelist. "
            "Consider using built-in Victor tools for better integration.",
            tool_name,
        )
        # Still allow, but log warning

    # Check count limit
    if langchain_count >= MAX_LANGCHAIN_TOOLS:
        return False, (
            f"Maximum LangChain tools ({MAX_LANGCHAIN_TOOLS}) reached. "
            f"Cannot register '{tool_name}'. "
            f"Consider removing unused LangChain tools or using built-in Victor tools."
        )

    # Check for exact duplicate
    if tool_name in existing_tools:
        return False, f"Tool '{tool_name}' already exists in tool registry"

    # Check for semantic similarity
    for existing_name in existing_tools:
        if _are_tools_semantically_similar(tool_name, existing_name):
            return False, (
                f"Tool '{tool_name}' is semantically similar to existing tool '{existing_name}'. "
                f"Skipping to avoid proliferation."
            )

    # All checks passed
    return True, "Tool allowed"


def _pydantic_to_json_schema(args_schema: Any) -> Dict[str, Any]:
    """Convert LangChain args_schema to JSON Schema dict.

    Handles:
    - Pydantic v2 models (.model_json_schema())
    - Pydantic v1 models (.schema())
    - Already-converted dicts (passthrough)
    - None (empty schema)
    """
    if args_schema is None:
        return {"type": "object", "properties": {}}

    if isinstance(args_schema, dict):
        return args_schema

    # Pydantic v2
    if hasattr(args_schema, "model_json_schema"):
        try:
            return args_schema.model_json_schema()
        except Exception:
            pass

    # Pydantic v1
    if hasattr(args_schema, "schema"):
        try:
            return args_schema.schema()
        except Exception:
            pass

    return {"type": "object", "properties": {}}


class LangChainAdapterTool(BaseTool):
    """Adapts a LangChain BaseTool to Victor's BaseTool interface.

    Makes LangChain tools first-class citizens in Victor's tool ecosystem.
    The LLM sees native names, descriptions, and JSON Schema parameters.
    Execution routes through LangChain's ainvoke() for async compatibility.

    All LangChain tools are automatically prefixed with 'lgc_' for unified
    naming convention and tagged with ToolSource.LANGCHAIN for deduplication.
    """

    def __init__(
        self,
        langchain_tool: "LCBaseTool",
        name_prefix: str = DEFAULT_LANGCHAIN_PREFIX,
        source: str = "langchain",
    ):
        self._lc_tool = langchain_tool
        self._name_prefix = name_prefix or DEFAULT_LANGCHAIN_PREFIX
        self._source = source
        self._json_schema = _pydantic_to_json_schema(getattr(langchain_tool, "args_schema", None))

        # Set tool source metadata for deduplication
        try:
            self._tool_source = ToolSource.LANGCHAIN
        except Exception:
            self._tool_source = "langchain"  # Fallback

    @property
    def name(self) -> str:
        """Return tool name with lgc_ prefix for unified naming convention."""
        base_name = self._lc_tool.name
        if self._name_prefix:
            return f"{self._name_prefix}_{base_name}"
        return base_name

    @property
    def description(self) -> str:
        desc = self._lc_tool.description or f"LangChain tool: {self._lc_tool.name}"
        return f"{desc} (via LangChain)"

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._json_schema

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.MEDIUM  # External execution

    @property
    def is_idempotent(self) -> bool:
        return False  # Cannot guarantee for LangChain tools

    @property
    def default_schema_level(self) -> str:
        """LangChain tools default to STUB schema for token efficiency."""
        return "stub"

    @property
    def langchain_tool_name(self) -> str:
        """The original LangChain tool name (without prefix)."""
        return self._lc_tool.name

    async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
        """Execute by routing through LangChain's ainvoke().

        ainvoke() handles async/sync bridging internally — if the tool
        only implements _run(), LangChain runs it in a thread pool.
        """
        try:
            # LangChain ainvoke accepts dict input
            tool_input = kwargs if kwargs else {}
            result = await self._lc_tool.ainvoke(tool_input)

            return ToolResult(
                success=True,
                output=result,
                error=None,
                metadata={
                    "source": self._source,
                    "langchain_tool": self._lc_tool.name,
                },
            )
        except Exception as e:
            logger.warning("LangChain tool %s failed: %s", self.name, e)
            return ToolResult(
                success=False,
                output="",
                error=f"LangChain tool execution failed: {e}",
                metadata={"source": self._source},
            )


class LangChainToolProjector:
    """Factory for batch-adapting LangChain tools with collision handling and safeguards.

    Same pattern as MCPToolProjector — iterates tools, creates adapters,
    handles name collisions, and enforces safeguards.
    """

    @staticmethod
    def project(
        tools: List["LCBaseTool"],
        prefix: str = DEFAULT_LANGCHAIN_PREFIX,
        conflict_strategy: str = "prefix_source",
        existing_tool_names: Optional[List[str]] = None,
        enable_safeguards: bool = True,
    ) -> List[LangChainAdapterTool]:
        """Create adapter tools for a list of LangChain tools with safeguards.

        Args:
            tools: List of LangChain BaseTool instances
            prefix: Optional prefix for all tool names (default: "lgc")
            conflict_strategy: How to handle name collisions:
                "prefix_source" — prepend source index (default)
                "skip" — skip duplicates, keep first
            existing_tool_names: List of already registered tool names (for duplicate detection)
            enable_safeguards: Whether to enforce LangChain safeguards (default: True)

        Returns:
            List of LangChainAdapterTool instances ready for registration

        Raises:
            ValueError: If count limit is exceeded and safeguards are enabled

        Note:
            Actual deduplication is handled by ToolDeduplicator in ToolRegistry.
            Safeguards here are lightweight checks (count limit, blacklist logging).
        """
        adapted: List[LangChainAdapterTool] = []
        seen_names: Dict[str, int] = {}
        blocked_count = 0

        # Initialize existing tools list
        if existing_tool_names is None:
            existing_tool_names = []

        for tool in tools:
            # Use default prefix if none provided
            effective_prefix = prefix or DEFAULT_LANGCHAIN_PREFIX
            adapter = LangChainAdapterTool(tool, name_prefix=effective_prefix)
            tool_name = adapter.name

            # Apply lightweight safeguards if enabled
            if enable_safeguards:
                # Check blacklist (for logging only - actual blocking done by ToolDeduplicator)
                if tool_name.lower() in BANNED_LANGCHAIN_TOOLS:
                    logger.warning(
                        "LangChain tool '%s' is blacklisted (duplicate of built-in tool). "
                        "ToolDeduplicator will handle deduplication during registration.",
                        tool_name,
                    )

                # Check whitelist (for logging only)
                if ALLOWED_LANGCHAIN_TOOLS and tool_name.lower() not in ALLOWED_LANGCHAIN_TOOLS:
                    logger.info(
                        "LangChain tool '%s' is not in recommended whitelist. "
                        "Consider using built-in Victor tools for better integration.",
                        tool_name,
                    )

                # Check count limit (still enforced here)
                if len(adapted) >= MAX_LANGCHAIN_TOOLS:
                    logger.error(
                        "Maximum LangChain tools (%d) reached. Cannot register '%s'.",
                        MAX_LANGCHAIN_TOOLS,
                        tool_name,
                    )
                    blocked_count += 1
                    continue

                # Log successful projection
                logger.info(
                    "Projecting LangChain tool '%s' (via LangChain). "
                    "LangChain tools projected: %d/%d",
                    tool_name,
                    len(adapted) + 1,
                    MAX_LANGCHAIN_TOOLS,
                )

            # Handle name conflicts within this batch
            if tool_name in seen_names:
                if conflict_strategy == "skip":
                    logger.debug("Skipping duplicate LangChain tool: %s", tool_name)
                    blocked_count += 1
                    continue
                # prefix_source: add index suffix
                idx = seen_names[tool_name]
                adapter = LangChainAdapterTool(
                    tool,
                    name_prefix=f"{effective_prefix}_{idx}",
                )
                tool_name = adapter.name

            seen_names[tool_name] = seen_names.get(tool_name, 0) + 1
            adapted.append(adapter)

        if blocked_count > 0:
            logger.warning(
                "Blocked %d LangChain tool(s) due to safeguards (count limit)",
                blocked_count,
            )

        logger.info("Projected %d LangChain tools (blocked: %d)", len(adapted), blocked_count)
        return adapted


def register_langchain_tools(
    tools: List["LCBaseTool"],
    tool_registry: Any,
    prefix: str = DEFAULT_LANGCHAIN_PREFIX,
    conflict_strategy: str = "prefix_source",
    enable_safeguards: bool = True,
) -> int:
    """Register LangChain tools with automatic safeguards and deduplication.

    Convenience function that projects tools and registers them,
    applying all safeguards by default. ToolDeduplicator in ToolRegistry
    will handle cross-source deduplication automatically.

    Args:
        tools: List of LangChain BaseTool instances
        tool_registry: Victor tool registry instance
        prefix: Optional prefix for all tool names (default: "lgc")
        conflict_strategy: How to handle name collisions
        enable_safeguards: Whether to enforce LangChain safeguards (default: True)

    Returns:
        Number of tools successfully registered

    Raises:
        ValueError: If count limit is exceeded and safeguards are enabled

    Example:
        from langchain_community.tools import WikipediaQueryRun
        from victor.tools.langchain_adapter_tool import register_langchain_tools
        from victor.framework.tools import get_tool_registry

        wiki = WikipediaQueryRun()
        registry = get_tool_registry()
        count = register_langchain_tools([wiki], registry)
    """
    # Get existing tool names from registry
    existing_names = []
    if hasattr(tool_registry, "get_tool_names"):
        existing_names = list(tool_registry.get_tool_names())
    elif hasattr(tool_registry, "tools"):
        existing_names = [t.name for t in tool_registry.tools]

    # Project tools with safeguards
    adapters = LangChainToolProjector.project(
        tools,
        prefix=prefix,
        conflict_strategy=conflict_strategy,
        existing_tool_names=existing_names,
        enable_safeguards=enable_safeguards,
    )

    # Register adapters (ToolDeduplicator will handle conflicts)
    registered_count = 0
    for adapter in adapters:
        try:
            tool_registry.register(adapter)
            registered_count += 1
            logger.info("Registered LangChain tool: %s", adapter.name)
        except Exception as e:
            logger.error("Failed to register LangChain tool %s: %s", adapter.name, e)

    logger.info(
        "LangChain tool registration complete: %d registered, %d blocked",
        registered_count,
        len(adapters) - registered_count,
    )

    return registered_count


__all__ = [
    "LangChainAdapterTool",
    "LangChainToolProjector",
    "register_langchain_tools",
    "MAX_LANGCHAIN_TOOLS",
    "ALLOWED_LANGCHAIN_TOOLS",
    "BANNED_LANGCHAIN_TOOLS",
]
