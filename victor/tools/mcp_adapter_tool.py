# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""MCP tool adapter — projects MCP tools as first-class Victor tools.

Adapter Pattern: MCPAdapterTool wraps an MCPTool + MCPRegistry into
the BaseTool interface. The LLM sees native tool names (e.g., "github_search")
with proper JSON Schema parameters — no bridge indirection.

Factory Pattern: MCPToolProjector creates MCPAdapterTool instances from
all connected MCP servers, handling name collisions and optional prefixing.

Usage:
    from victor.tools.mcp_adapter_tool import MCPToolProjector

    tools = MCPToolProjector.project(mcp_registry)
    for tool in tools:
        tool_registry.register(tool)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.tools.base import BaseTool, CostTier, ToolResult

if TYPE_CHECKING:
    from victor.integrations.mcp.protocol import MCPParameter, MCPTool
    from victor.integrations.mcp.registry import MCPRegistry

logger = logging.getLogger(__name__)

# Import ToolSource for metadata
try:
    from victor.tools.deduplication import ToolSource
except ImportError:
    # Fallback if deduplication not available
    class ToolSource:
        MCP = "mcp"


# Default prefix for all MCP tools (unified naming convention)
DEFAULT_MCP_PREFIX = "mcp"


def _mcp_param_to_json_schema(param: "MCPParameter") -> Dict[str, Any]:
    """Convert a single MCPParameter to JSON Schema property."""
    schema: Dict[str, Any] = {
        "type": param.type.value,
        "description": param.description,
    }
    if param.default is not None:
        schema["default"] = param.default
    return schema


def _mcp_params_to_json_schema(params: List["MCPParameter"]) -> Dict[str, Any]:
    """Convert MCPParameter list to JSON Schema parameters dict."""
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param in params:
        properties[param.name] = _mcp_param_to_json_schema(param)
        if param.required:
            required.append(param.name)

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


class MCPAdapterTool(BaseTool):
    """Adapts an MCP tool to Victor's BaseTool interface.

    Makes MCP tools indistinguishable from native tools in the agent's
    toolset. The LLM sees proper name, description, and JSON Schema
    parameters. Execution routes through MCPRegistry.call_tool().

    All MCP tools are automatically prefixed with 'mcp_' for unified
    naming convention and tagged with ToolSource.MCP for deduplication.
    """

    def __init__(
        self,
        mcp_tool: "MCPTool",
        mcp_registry: "MCPRegistry",
        server_name: str,
        name_prefix: str = DEFAULT_MCP_PREFIX,
    ):
        self._mcp_tool = mcp_tool
        self._registry = mcp_registry
        self._server_name = server_name
        self._name_prefix = name_prefix or DEFAULT_MCP_PREFIX
        self._json_schema = _mcp_params_to_json_schema(mcp_tool.parameters)

        # Set tool source metadata for deduplication
        try:
            self._tool_source = ToolSource.MCP
        except Exception:
            self._tool_source = "mcp"  # Fallback

    @property
    def name(self) -> str:
        """Return tool name with mcp_ prefix for unified naming convention."""
        if self._name_prefix:
            return f"{self._name_prefix}_{self._mcp_tool.name}"
        return self._mcp_tool.name

    @property
    def description(self) -> str:
        desc = self._mcp_tool.description or f"MCP tool: {self._mcp_tool.name}"
        return f"{desc} (via MCP server: {self._server_name})"

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._json_schema

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.MEDIUM  # MCP tools involve IPC

    @property
    def is_idempotent(self) -> bool:
        return False  # Cannot guarantee for external MCP tools

    @property
    def default_schema_level(self) -> str:
        """MCP tools default to STUB schema for token efficiency."""
        return "stub"

    @property
    def mcp_server_name(self) -> str:
        """The MCP server providing this tool."""
        return self._server_name

    @property
    def mcp_tool_name(self) -> str:
        """The original MCP tool name (without prefix)."""
        return self._mcp_tool.name

    async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
        """Execute by routing through MCPRegistry.call_tool()."""
        try:
            result = await self._registry.call_tool(self._mcp_tool.name, **kwargs)
            return ToolResult(
                success=result.success,
                output=result.result if result.success else result.error,
                error=result.error if not result.success else None,
                metadata={
                    "mcp_server": self._server_name,
                    "mcp_tool": self._mcp_tool.name,
                },
            )
        except Exception as e:
            logger.warning("MCP tool %s execution failed: %s", self.name, e)
            return ToolResult(
                success=False,
                output="",
                error=f"MCP tool execution failed: {e}",
                metadata={"mcp_server": self._server_name},
            )


class MCPToolProjector:
    """Factory that projects MCPRegistry tools into BaseTool instances.

    Iterates all connected MCP servers, creates an MCPAdapterTool for each
    tool, and handles name collisions by prefixing with server name.

    Supports optional relevance filtering: when a user_message is provided,
    only MCP tools whose name or description match the query are included.
    This prevents broadcasting all 48+ MCP tools every turn (~1,500 token
    savings for MCP-heavy setups).
    """

    # Default relevance threshold for keyword-based MCP filtering
    DEFAULT_RELEVANCE_THRESHOLD: float = 0.3

    @staticmethod
    def project(
        registry: "MCPRegistry",
        prefix: str = DEFAULT_MCP_PREFIX,
        conflict_strategy: str = "prefix_server",
        user_message: Optional[str] = None,
        max_mcp_tools: int = 12,
    ) -> List[MCPAdapterTool]:
        """Create adapter tools for all tools in connected MCP servers.

        Args:
            registry: MCPRegistry with connected servers
            prefix: Optional prefix for all tool names (default: "mcp")
            conflict_strategy: How to handle name collisions:
                "prefix_server" — prepend server name (default)
                "skip" — skip duplicates, keep first
            user_message: Optional query for relevance filtering.
                When provided, only MCP tools relevant to the query
                are returned (keyword matching on name + description).
                When None, all MCP tools are returned (backward compat).
            max_mcp_tools: Maximum MCP tools to include when filtering
                by relevance. Ignored when user_message is None.

        Returns:
            List of MCPAdapterTool instances ready for registration

        Note:
            Actual deduplication is handled by ToolDeduplicator in ToolRegistry.
            Prefix strategy here is for distinguishing tools from different MCP servers.
        """
        tools: List[MCPAdapterTool] = []
        seen_names: Dict[str, str] = {}  # name -> server_name

        for server_name, entry in registry._servers.items():
            if not entry.tools_cache:
                continue

            for mcp_tool in entry.tools_cache:
                # Use default prefix if none provided
                effective_prefix = prefix or DEFAULT_MCP_PREFIX
                adapter = MCPAdapterTool(
                    mcp_tool=mcp_tool,
                    mcp_registry=registry,
                    server_name=server_name,
                    name_prefix=effective_prefix,
                )
                tool_name = adapter.name

                if tool_name in seen_names:
                    if conflict_strategy == "skip":
                        logger.debug(
                            "Skipping duplicate MCP tool %s from %s (already from %s)",
                            tool_name,
                            server_name,
                            seen_names[tool_name],
                        )
                        continue
                    # prefix_server: create with server-prefixed name
                    adapter = MCPAdapterTool(
                        mcp_tool=mcp_tool,
                        mcp_registry=registry,
                        server_name=server_name,
                        name_prefix=f"{effective_prefix}_{server_name}",
                    )
                    tool_name = adapter.name
                    logger.info(
                        "MCP tool name collision: %s → %s (from %s)",
                        mcp_tool.name,
                        tool_name,
                        server_name,
                    )

                seen_names[tool_name] = server_name
                tools.append(adapter)

        total_projected = len(tools)

        # --- Relevance filtering: keyword match on name + description ---
        if user_message is not None and tools:
            tools = MCPToolProjector._filter_by_relevance(
                tools,
                user_message,
                max_mcp_tools,
            )

        logger.info(
            "Projected %d MCP tools from %d servers%s",
            len(tools),
            len({t.mcp_server_name for t in tools}),
            (f" (filtered from {total_projected})" if len(tools) < total_projected else ""),
        )
        return tools

    @staticmethod
    def _filter_by_relevance(
        tools: List[MCPAdapterTool],
        user_message: str,
        max_tools: int,
    ) -> List[MCPAdapterTool]:
        """Filter MCP tools by keyword relevance to the user message.

        Scores each tool by counting keyword overlaps between the query
        and the tool's name + description. Tools with zero relevance
        are excluded. Results are capped at max_tools.

        This is intentionally lightweight (no embeddings) since MCP tools
        are already STUB-level and the filtering runs per-turn.
        """
        msg_words = set(user_message.lower().split())
        # Remove very short/common words
        msg_words = {w for w in msg_words if len(w) > 2}

        scored: List[tuple] = []  # (score, tool)
        for tool in tools:
            # Build searchable text from tool name and description
            tool_text = f"{tool.name} {tool.description}".lower()
            tool_words = set(tool_text.split())
            tool_words = {w for w in tool_words if len(w) > 2}

            # Score: Jaccard-like overlap normalized by query size
            overlap = len(msg_words & tool_words)
            if overlap > 0:
                score = overlap / max(len(msg_words), 1)
                scored.append((score, tool))

        if not scored:
            # No matches — return top tools by name similarity as fallback
            # (at least return a few MCP tools so the model isn't blind)
            return tools[: min(3, max_tools)]

        # Sort by score descending, cap at max_tools
        scored.sort(key=lambda x: x[0], reverse=True)
        filtered = [tool for _, tool in scored[:max_tools]]

        logger.debug(
            "MCP relevance filter: %d/%d tools matched query " "(top: %s %.2f)",
            len(filtered),
            len(tools),
            filtered[0].name if filtered else "none",
            scored[0][0] if scored else 0.0,
        )
        return filtered


__all__ = [
    "MCPAdapterTool",
    "MCPToolProjector",
]
