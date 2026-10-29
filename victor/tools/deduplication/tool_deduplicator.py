"""Core tool deduplication engine with conflict detection and resolution.

This module provides the ToolDeduplicator class that handles cross-source
tool deduplication with priority-based resolution (Native > LangChain > MCP > Plugin).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolSource(str, Enum):
    """Tool source with priority for deduplication resolution.

    Higher priority sources are preferred when conflicts are detected.
    Priority order: NATIVE > LANGCHAIN > MCP > PLUGIN
    """

    NATIVE = "native"
    LANGCHAIN = "langchain"
    MCP = "mcp"
    PLUGIN = "plugin"

    @property
    def priority_weight(self) -> int:
        """Get priority weight for sorting (higher = more preferred)."""
        weights = {
            ToolSource.NATIVE: 100,
            ToolSource.LANGCHAIN: 75,
            ToolSource.MCP: 50,
            ToolSource.PLUGIN: 25,
        }
        return weights[self]

    def __lt__(self, other: ToolSource) -> bool:
        """Compare tools by priority (higher priority = "less than" for sorting)."""
        return self.priority_weight > other.priority_weight


class DeduplicationConfig(BaseModel):
    """Configuration for tool deduplication system."""

    enabled: bool = Field(
        default=True,
        description="Enable tool deduplication across sources",
    )
    priority_order: List[str] = Field(
        default_factory=lambda: ["native", "langchain", "mcp", "plugin"],
        description="Priority order for tool sources (highest to lowest)",
    )
    whitelist: List[str] = Field(
        default_factory=list,
        description="Tools to always allow (bypass deduplication)",
    )
    blacklist: List[str] = Field(
        default_factory=list,
        description="Tools to always skip (force deduplication)",
    )
    naming_enforcement: bool = Field(
        default=True,
        description="Enforce naming conventions (lgc_*, mcp_*, plg_*)",
    )
    semantic_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Threshold for semantic similarity detection (0.0-1.0)",
    )

    def get_priority_map(self) -> Dict[str, int]:
        """Get priority weights for each source based on priority_order."""
        return {
            source: len(self.priority_order) - i for i, source in enumerate(self.priority_order)
        }


@dataclass
class DeduplicationResult:
    """Result of tool deduplication process."""

    kept_tools: List[Any] = field(default_factory=list)
    skipped_tools: List[Any] = field(default_factory=list)
    conflicts_resolved: int = 0
    naming_changes: int = 0
    logs: List[str] = field(default_factory=list)

    def add_log(self, message: str) -> None:
        """Add a log message."""
        self.logs.append(message)
        logger.debug(f"ToolDeduplicator: {message}")


class ToolDeduplicator:
    """Deduplicates tools across multiple sources with priority-based resolution.

    Usage:
        config = DeduplicationConfig()
        deduplicator = ToolDeduplicator(config)
        result = deduplicator.deduplicate(tools)
    """

    def __init__(self, config: Optional[DeduplicationConfig] = None) -> None:
        """Initialize tool deduplicator.

        Args:
            config: Deduplication configuration (uses defaults if None)
        """
        self._config = config or DeduplicationConfig()
        self._priority_map = self._config.get_priority_map()

    def deduplicate(self, tools: List[Any]) -> DeduplicationResult:
        """Deduplicate a list of tools using priority-based resolution.

        Args:
            tools: List of tools to deduplicate

        Returns:
            DeduplicationResult with kept_tools, skipped_tools, and metadata
        """
        result = DeduplicationResult()

        if not self._config.enabled:
            result.kept_tools = list(tools)
            result.add_log("Deduplication disabled, returning all tools")
            return result

        if not tools:
            result.add_log("No tools to deduplicate")
            return result

        # Apply blacklist first (force skip these tools)
        tools_to_process = []
        for tool in tools:
            tool_name = self._get_tool_name(tool)
            if tool_name in self._config.blacklist:
                result.skipped_tools.append(tool)
                result.add_log(f"Blacklisted tool skipped: {tool_name}")
            else:
                tools_to_process.append(tool)

        # Group tools by normalized name for conflict detection
        groups = self._group_tools_by_name(tools_to_process)

        # Resolve conflicts within each group
        for normalized_name, group in groups.items():
            kept = self._resolve_group_conflict(group, result)
            if kept:
                result.kept_tools.append(kept)

        return result

    def _get_tool_name(self, tool: Any) -> str:
        """Extract tool name, handling various tool types."""
        if hasattr(tool, "name"):
            return tool.name
        elif hasattr(tool, "__name__"):
            return tool.__name__
        else:
            return str(tool)

    def _get_tool_source(self, tool: Any) -> ToolSource:
        """Detect tool source from metadata or heuristics."""
        # Check for source metadata first
        if hasattr(tool, "_tool_source"):
            source = ToolSource(tool._tool_source)
            # If source is explicitly set to non-NATIVE, trust it
            # If NATIVE, still check for prefix in case it's a mislabeled adapter tool
            if source != ToolSource.NATIVE:
                return source

        # Heuristic detection based on tool name prefix
        tool_name = self._get_tool_name(tool).lower()

        if tool_name.startswith("lgc_") or tool_name.startswith("langchain_"):
            return ToolSource.LANGCHAIN
        elif tool_name.startswith("mcp_"):
            return ToolSource.MCP
        elif tool_name.startswith("plg_"):
            return ToolSource.PLUGIN
        else:
            # Default to native for tools with no prefix
            return ToolSource.NATIVE

    def _group_tools_by_name(self, tools: List[Any]) -> Dict[str, List[Any]]:
        """Group tools by normalized name for conflict detection.

        Normalization: lowercase, remove source prefixes, normalize underscores/whitespace.
        """
        groups: Dict[str, List[Any]] = {}

        for tool in tools:
            tool_name = self._get_tool_name(tool)
            normalized = self._normalize_name(tool_name)

            if normalized not in groups:
                groups[normalized] = []
            groups[normalized].append(tool)

        return groups

    def _normalize_name(self, name: str) -> str:
        """Normalize tool name for conflict detection.

        Removes source prefixes (lgc_, mcp_, plg_), converts to lowercase,
        and normalizes underscores/hyphens to spaces.
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

    def _resolve_group_conflict(
        self, group: List[Any], result: DeduplicationResult
    ) -> Optional[Any]:
        """Resolve conflicts within a group of tools with the same normalized name.

        Args:
            group: List of tools with conflicting names
            result: DeduplicationResult to update with logs and skipped tools

        Returns:
            The tool to keep (highest priority), or None if all skipped
        """
        if len(group) == 1:
            # No conflict
            return group[0]

        result.conflicts_resolved += 1

        # Check for whitelist (bypass deduplication)
        for tool in group:
            tool_name = self._get_tool_name(tool)
            if tool_name in self._config.whitelist:
                result.add_log(f"Whitelisted tool wins conflict: {tool_name}")
                # Keep whitelisted tool, skip others
                for other in group:
                    if other is not tool:
                        result.skipped_tools.append(other)
                return tool

        # Sort by source priority
        sorted_tools = sorted(group, key=lambda t: self._get_tool_source(t))

        # Keep highest priority tool
        kept = sorted_tools[0]
        kept_source = self._get_tool_source(kept)
        kept_name = self._get_tool_name(kept)

        result.add_log(
            f"Conflict resolved: kept {kept_name} (source={kept_source.value}), "
            f"skipped {len(sorted_tools) - 1} lower-priority tools"
        )

        # Skip lower priority tools
        for tool in sorted_tools[1:]:
            tool_source = self._get_tool_source(tool)
            tool_name = self._get_tool_name(tool)
            result.add_log(
                f"Skipped {tool_name} (source={tool_source.value}) in favor of {kept_name}"
            )
            result.skipped_tools.append(tool)

        return kept

    def enforce_naming(self, tool: Any) -> str:
        """Enforce naming conventions by adding source prefix if needed.

        Args:
            tool: Tool to check/rename

        Returns:
            Original name if convention already followed, else new name with prefix
        """
        if not self._config.naming_enforcement:
            return self._get_tool_name(tool)

        tool_name = self._get_tool_name(tool)
        source = self._get_tool_source(tool)

        # Check if prefix already present
        prefixes = {
            ToolSource.LANGCHAIN: "lgc_",
            ToolSource.MCP: "mcp_",
            ToolSource.PLUGIN: "plg_",
        }

        expected_prefix = prefixes.get(source, "")
        if expected_prefix and not tool_name.lower().startswith(expected_prefix):
            new_name = f"{expected_prefix}{tool_name}"
            logger.debug(f"Renamed {tool_name} → {new_name} (naming enforcement)")
            return new_name

        return tool_name
