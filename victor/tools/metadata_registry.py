from __future__ import annotations

from victor.core.json_utils import json_dumps

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

"""Tool metadata registry for querying tool properties.

Provides centralized access to tool metadata (priority, access_mode, danger_level)
for tool selection, approval tracking, and danger warnings.

Key Features:
- Keyword-based tool selection with relevance scoring
- Category discovery from @tool decorators
- Stage-based tool selection for conversation state machine
- Metrics and observability for keyword matching
- Fallback chain for graceful degradation

Usage:
    from victor.tools.metadata_registry import ToolMetadataRegistry

    registry = ToolMetadataRegistry()
    registry.register(tool)

    # Query by properties
    critical_tools = registry.get_by_priority(Priority.CRITICAL)
    safe_tools = registry.get_safe_tools()
    dangerous_tools = registry.get_by_danger_level(DangerLevel.HIGH)

    # Scored keyword matching
    matches = registry.get_tools_matching_text_scored("run tests")
    for tool_name, score in matches:
        print(f"{tool_name}: {score}")

    # Stage-based tool selection
    reading_tools = registry.get_tools_by_stage("reading")
    all_stages = registry.get_all_stages()

Migration Pattern for Static Configs:
-------------------------------------
To migrate static tool configuration dictionaries to decorator-driven lookups:

1. **Add decorator attributes**: Update @tool decorators with the new attributes:
   ```python
   @tool(
       name="read_file",
       keywords=["read", "file", "content"],
       stages=["reading", "initial"],  # Conversation stages
       mandatory_keywords=["show file", "read content"],  # Force inclusion
       task_types=["analysis", "search"],  # Classification-aware selection
       signature_params=["path"],  # Loop detection - offset/limit excluded to enable pagination
       execution_category="read_only",  # Parallel execution planning
   )
   async def read_file(path: str, offset: int = 0) -> str:
       ...
   ```

2. **Update consumers to use registry**: Replace static dictionary lookups with
   registry queries:
   ```python
   # Before (static):
   tools = STAGE_TOOL_MAPPING.get(stage, set())
   mandatory = MANDATORY_TOOL_KEYWORDS.get(keyword, set())
   task_tools = TASK_TYPE_CATEGORIES.get(task_type, set())

   # After (registry-driven with fallback):
   from victor.tools.metadata_registry import (
       get_tools_by_stage,
       get_tools_matching_mandatory_keywords,
       get_tools_by_task_type,
       get_tools_by_execution_category,
       get_parallelizable_tools,
       get_signature_params,
   )
   registry_tools = get_tools_by_stage(stage.name.lower())
   tools = registry_tools | static_fallback  # Union for backward compat
   ```

3. **Available decorator attributes**:
   - `stages`: List of conversation stages (initial, planning, reading, analysis,
     execution, verification, completion)
   - `mandatory_keywords`: Phrases that force tool inclusion (e.g., "show diff")
   - `task_types`: Task classifications (analysis, action, generation, search, edit)
   - `signature_params`: Parameters to include in loop detection signatures
   - `execution_category`: One of read_only, write, compute, network, execute, mixed

4. **Fallback pattern**: Keep static configs as fallback during migration:
   ```python
   def _get_tools_for_stage(self, stage: ConversationStage) -> Set[str]:
       stage_name = stage.name.lower()
       registry_tools = registry_get_tools_by_stage(stage_name)
       static_tools = STAGE_TOOL_MAPPING.get(stage, set())
       return registry_tools | static_tools  # Union for coverage
   ```

5. **Parallel execution**: Use execution categories for safe parallelization:
   ```python
   parallelizable = get_parallelizable_tools()  # READ_ONLY, COMPUTE, NETWORK
   conflicts = registry.get_conflicting_tools(tool_name)
   ```

See `victor/agent/conversation_state.py` for a complete migration example.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from victor.tools.base import (
    AccessMode,
    BaseTool,
    CostTier,
    DangerLevel,
    ExecutionCategory,
    Priority,
)
from victor.tools.keyword_matcher import KeywordMatcher
from victor.tools.tool_category_resolver import ToolCategoryResolver
from victor.tools.cache_manager import CacheNamespace, ToolCacheManager

logger = logging.getLogger(__name__)

# Rust-accelerated pattern matching (with Python fallback)
_RUST_PATTERN_MATCHING_AVAILABLE = False
try:
    from victor.processing.native import find_all_patterns

    _RUST_PATTERN_MATCHING_AVAILABLE = True
except ImportError:
    pass


def _get_matched_pattern_indices(text: str, patterns: List[str]) -> List[int]:
    """Get indices of patterns that match in text (using Rust or Python fallback).

    Returns unique pattern indices (not all matches).
    """
    if _RUST_PATTERN_MATCHING_AVAILABLE:
        matches = find_all_patterns(text, patterns, case_insensitive=True)
        # Extract unique pattern indices
        return list({m.pattern_idx for m in matches})
    else:
        # Python fallback - simple substring matching
        text_lower = text.lower()
        return [i for i, p in enumerate(patterns) if p.lower() in text_lower]


# Default fallback tools when registry is empty or keyword matching fails
_FALLBACK_CRITICAL_TOOLS: Set[str] = {
    "read",
    "write",
    "edit",
    "ls",
    "shell",
    "search",
    "glob",
    "grep",
}


@dataclass
class KeywordMatchResult:
    """Result of keyword-based tool matching with relevance score.

    Attributes:
        tool_name: Name of the matched tool
        score: Relevance score (0.0 to 1.0) based on keyword match quality
        matched_keywords: Set of keywords that matched
        priority_boost: Additional score from tool priority
    """

    tool_name: str
    score: float
    matched_keywords: Set[str]
    priority_boost: float = 0.0

    @property
    def total_score(self) -> float:
        """Combined score with priority boost."""
        return self.score + self.priority_boost


@dataclass
class MatchingMetrics:
    """Metrics for keyword matching operations.

    Tracks performance and quality of keyword-based tool selection
    for observability and tuning.
    """

    total_queries: int = 0
    total_matches: int = 0
    empty_results: int = 0
    fallback_used: int = 0
    avg_match_time_ms: float = 0.0
    avg_tools_per_match: float = 0.0
    keyword_hit_counts: Dict[str, int] = field(default_factory=dict)
    category_hit_counts: Dict[str, int] = field(default_factory=dict)

    def record_match(
        self,
        duration_ms: float,
        tools_matched: int,
        keywords_matched: Set[str],
        used_fallback: bool = False,
    ) -> None:
        """Record a keyword matching operation."""
        self.total_queries += 1
        self.total_matches += tools_matched

        if tools_matched == 0:
            self.empty_results += 1

        if used_fallback:
            self.fallback_used += 1

        # Update running averages
        n = self.total_queries
        self.avg_match_time_ms = (self.avg_match_time_ms * (n - 1) + duration_ms) / n
        if self.total_queries > 0:
            self.avg_tools_per_match = self.total_matches / self.total_queries

        # Track keyword hits
        for kw in keywords_matched:
            self.keyword_hit_counts[kw] = self.keyword_hit_counts.get(kw, 0) + 1

    def record_category_hit(self, category: str) -> None:
        """Record a category-based lookup."""
        self.category_hit_counts[category] = self.category_hit_counts.get(category, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/export."""
        return {
            "total_queries": self.total_queries,
            "total_matches": self.total_matches,
            "empty_results": self.empty_results,
            "fallback_used": self.fallback_used,
            "avg_match_time_ms": round(self.avg_match_time_ms, 2),
            "avg_tools_per_match": round(self.avg_tools_per_match, 2),
            "top_keywords": dict(
                sorted(
                    self.keyword_hit_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),
            "top_categories": dict(
                sorted(
                    self.category_hit_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
            ),
        }


# Minimum keyword length to prevent false matches with common short words
MIN_KEYWORD_LENGTH = 3


@dataclass
class ToolMetadataEntry:
    """Metadata entry for a registered tool.

    Provides a snapshot of tool metadata for efficient querying without
    requiring access to the full tool instance.
    """

    name: str
    category: Optional[str]
    priority: Priority
    access_mode: AccessMode
    danger_level: DangerLevel
    cost_tier: CostTier
    aliases: Set[str] = field(default_factory=set)
    keywords: Set[str] = field(default_factory=set)
    stages: Set[str] = field(default_factory=set)  # Conversation stages where tool is relevant
    description: str = ""
    # NEW: Decorator-driven semantic selection fields
    mandatory_keywords: Set[str] = field(default_factory=set)  # Keywords that force inclusion
    task_types: Set[str] = field(
        default_factory=set
    )  # Task types for classification-aware selection
    signature_params: Set[str] = field(default_factory=set)  # Params for loop detection signatures
    execution_category: ExecutionCategory = ExecutionCategory.READ_ONLY  # For parallel execution
    # NEW: Availability check for optional tools
    _availability_check: Optional[Callable[[], bool]] = field(default=None, repr=False)

    # Derived properties for caching behavior (reuse access_mode instead of new fields)
    @property
    def is_idempotent(self) -> bool:
        """Tool result can be cached if access_mode is READONLY (no side effects)."""
        return self.access_mode == AccessMode.READONLY

    @property
    def cache_invalidating(self) -> bool:
        """Tool invalidates cache if access_mode modifies state."""
        return self.access_mode in {
            AccessMode.WRITE,
            AccessMode.EXECUTE,
            AccessMode.MIXED,
        }

    @property
    def requires_configuration(self) -> bool:
        """Check if this tool requires external configuration.

        Returns True if an availability_check was provided, indicating
        this tool needs configuration (API keys, credentials, etc.)
        before it can be used. Examples: Slack, Teams, Jira.
        """
        return self._availability_check is not None

    def is_available(self) -> bool:
        """Check if this tool is currently available for use.

        For tools without an availability_check, always returns True.
        For tools with an availability_check (e.g., Slack, Teams), calls
        the provided function to check if the tool is properly configured.

        Returns:
            True if the tool is available, False if it requires configuration
            that hasn't been completed.
        """
        if self._availability_check is None:
            return True
        try:
            return self._availability_check()
        except Exception as e:
            logger.warning(f"Availability check for tool '{self.name}' raised exception: {e}")
            return False

    @classmethod
    def from_tool(cls, tool: BaseTool) -> "ToolMetadataEntry":
        """Create metadata entry from a tool instance."""
        explicit_metadata = None
        get_metadata = getattr(tool, "get_metadata", None)
        if callable(get_metadata):
            try:
                explicit_metadata = get_metadata()
            except Exception:
                logger.debug(
                    "Tool '%s' metadata generation failed during registry indexing",
                    tool.name,
                )

        # Get properties with defaults for tools that don't have the new attributes
        priority = getattr(explicit_metadata, "priority", None) or getattr(
            tool, "priority", Priority.MEDIUM
        )
        access_mode = getattr(explicit_metadata, "access_mode", None) or getattr(
            tool, "access_mode", AccessMode.READONLY
        )
        danger_level = getattr(explicit_metadata, "danger_level", None) or getattr(
            tool, "danger_level", DangerLevel.SAFE
        )
        aliases = getattr(tool, "aliases", set())
        category = getattr(explicit_metadata, "category", None) or getattr(tool, "category", None)
        # Extract keywords from tool decorator metadata, filtering short keywords
        raw_keywords = (
            getattr(explicit_metadata, "keywords", None) or getattr(tool, "keywords", None) or []
        )
        keywords = (
            {k.lower() for k in raw_keywords if len(k) >= MIN_KEYWORD_LENGTH}
            if raw_keywords
            else set()
        )

        # Log warning if keywords were filtered out
        filtered_count = len(raw_keywords) - len(keywords) if raw_keywords else 0
        if filtered_count > 0:
            logger.debug(
                f"Tool '{tool.name}': filtered {filtered_count} keywords shorter than "
                f"{MIN_KEYWORD_LENGTH} chars"
            )

        # Extract stages from tool decorator (lowercased for consistency)
        raw_stages = (
            getattr(explicit_metadata, "stages", None) or getattr(tool, "stages", None) or []
        )
        stages = {s.lower() for s in raw_stages} if raw_stages else set()

        # NEW: Extract mandatory keywords (lowercased for matching)
        raw_mandatory = (
            getattr(explicit_metadata, "mandatory_keywords", None)
            or getattr(tool, "mandatory_keywords", None)
            or []
        )
        mandatory_keywords = {k.lower() for k in raw_mandatory} if raw_mandatory else set()

        # NEW: Extract task types (lowercased for consistency)
        raw_task_types = (
            getattr(explicit_metadata, "task_types", None)
            or getattr(tool, "task_types", None)
            or []
        )
        task_types = {t.lower() for t in raw_task_types} if raw_task_types else set()

        # NEW: Extract signature params
        raw_signature_params = (
            getattr(explicit_metadata, "signature_params", None)
            or getattr(tool, "signature_params", None)
            or []
        )
        signature_params = set(raw_signature_params) if raw_signature_params else set()

        # NEW: Extract execution category
        execution_category = (
            getattr(explicit_metadata, "execution_category", None)
            or getattr(tool, "execution_category", None)
            or ExecutionCategory.READ_ONLY
        )

        # NEW: Extract availability check for optional tools
        availability_check = getattr(tool, "_availability_check", None)

        return cls(
            name=tool.name,
            category=category,
            priority=priority,
            access_mode=access_mode,
            danger_level=danger_level,
            cost_tier=tool.cost_tier,
            aliases=aliases,
            keywords=keywords,
            stages=stages,
            description=tool.description[:200] if tool.description else "",
            mandatory_keywords=mandatory_keywords,
            task_types=task_types,
            signature_params=signature_params,
            execution_category=execution_category,
            _availability_check=availability_check,
        )

    @property
    def requires_approval(self) -> bool:
        """Check if this tool requires approval."""
        return self.access_mode.requires_approval or self.danger_level.requires_confirmation

    @property
    def is_safe(self) -> bool:
        """Check if this tool is safe (readonly, no danger)."""
        return self.access_mode.is_safe and self.danger_level == DangerLevel.SAFE

    @property
    def warning_message(self) -> str:
        """Get warning message for this tool."""
        return self.danger_level.warning_message

    def should_include_for_task(self, task_type: str) -> bool:
        """Check if this tool should be included for a task type."""
        return self.priority.should_include_for_task(task_type)


class ToolMetadataRegistry:
    """Registry for querying tool metadata.

    Provides efficient lookup and filtering of tools by their metadata properties.
    This is useful for:
    - Tool selection: Get tools by priority level
    - Approval tracking: Find tools that require confirmation
    - Safety checks: Identify dangerous operations
    - Cost optimization: Filter by cost tier

    Features:
    - Keyword-based tool selection with relevance scoring
    - Category discovery from @tool decorators
    - Metrics and observability for keyword matching
    - Fallback chain for graceful degradation
    """

    # Priority weights for scoring (higher priority = higher boost)
    _PRIORITY_WEIGHTS: Dict[Priority, float] = {
        Priority.CRITICAL: 0.3,
        Priority.HIGH: 0.2,
        Priority.MEDIUM: 0.1,
        Priority.LOW: 0.05,
    }

    def __init__(
        self,
        category_resolver: Optional[ToolCategoryResolver] = None,
        keyword_matcher: Optional[KeywordMatcher] = None,
        schema_cache: Optional[CacheNamespace] = None,
    ) -> None:
        """Initialize the registry.

        Args:
            category_resolver: Resolver for tool categories. Creates default if None.
            keyword_matcher: Matcher for keyword-based tool selection. Creates default if None.
            schema_cache: Cache for tool schemas. Creates default if None.
        """
        self._entries: Dict[str, ToolMetadataEntry] = {}
        self._by_priority: Dict[Priority, Set[str]] = {p: set() for p in Priority}
        self._by_access_mode: Dict[AccessMode, Set[str]] = {a: set() for a in AccessMode}
        self._by_danger_level: Dict[DangerLevel, Set[str]] = {d: set() for d in DangerLevel}
        self._by_category: Dict[str, Set[str]] = {}
        self._by_keyword: Dict[str, Set[str]] = {}  # keyword -> set of tool names
        self._by_stage: Dict[str, Set[str]] = {}  # stage -> set of tool names
        self._alias_map: Dict[str, str] = {}  # alias -> canonical name
        self._metrics: MatchingMetrics = MatchingMetrics()
        # NEW: Indexes for decorator-driven semantic selection
        self._by_mandatory_keyword: Dict[str, Set[str]] = {}  # mandatory_keyword -> tools
        self._by_task_type: Dict[str, Set[str]] = {}  # task_type -> tools
        self._by_execution_category: Dict[ExecutionCategory, Set[str]] = {
            ec: set() for ec in ExecutionCategory
        }

        # Composed extracted classes (injectable for testing)
        self._category_resolver = category_resolver or ToolCategoryResolver()
        self._keyword_matcher = keyword_matcher or KeywordMatcher()
        self._schema_cache = schema_cache or CacheNamespace(
            name="tool_schemas", max_entries=200, default_ttl=None
        )
        self._tools_hash: Optional[str] = None
        self._last_refresh_count: int = 0

    @classmethod
    def get_instance(cls) -> "ToolMetadataRegistry":
        """Get the global registry singleton.

        This compatibility entrypoint lets runtime callers migrate from the
        older ``victor.tools.metadata.ToolMetadataRegistry`` singleton API
        without having to special-case this richer registry implementation.
        """
        return get_global_registry()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the global registry singleton."""
        reset_global_registry()

    @staticmethod
    def _calculate_tools_hash(tools: List[BaseTool]) -> str:
        """Calculate a stable hash of tool metadata-relevant fields."""
        import hashlib
        import json

        snapshots = []
        for tool in sorted(tools, key=lambda candidate: candidate.name):
            snapshots.append(
                {
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                    "parameters": getattr(tool, "parameters", None),
                    "category": getattr(tool, "category", None),
                    "keywords": sorted(getattr(tool, "keywords", None) or []),
                    "aliases": sorted(getattr(tool, "aliases", None) or []),
                    "stages": sorted(getattr(tool, "stages", None) or []),
                    "mandatory_keywords": sorted(getattr(tool, "mandatory_keywords", None) or []),
                    "task_types": sorted(getattr(tool, "task_types", None) or []),
                    "signature_params": sorted(getattr(tool, "signature_params", None) or []),
                    "priority": str(getattr(tool, "priority", "")),
                    "access_mode": str(getattr(tool, "access_mode", "")),
                    "danger_level": str(getattr(tool, "danger_level", "")),
                    "cost_tier": str(getattr(tool, "cost_tier", "")),
                    "execution_category": str(getattr(tool, "execution_category", "")),
                }
            )

        payload = json_dumps(snapshots, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _reset_registered_state(self) -> None:
        """Clear registered metadata and any composed indexes."""
        self._entries.clear()
        self._by_priority = {p: set() for p in Priority}
        self._by_access_mode = {a: set() for a in AccessMode}
        self._by_danger_level = {d: set() for d in DangerLevel}
        self._by_category.clear()
        self._by_keyword.clear()
        self._by_stage.clear()
        self._alias_map.clear()
        self._metrics = MatchingMetrics()
        self._by_mandatory_keyword.clear()
        self._by_task_type.clear()
        self._by_execution_category = {ec: set() for ec in ExecutionCategory}

        if hasattr(self._schema_cache, "clear"):
            self._schema_cache.clear()

        # The composed helpers do not currently expose a public reset API.
        # Clear their backing stores so refresh_from_tools() can rebuild them.
        if hasattr(self._category_resolver, "_tool_categories"):
            self._category_resolver._tool_categories.clear()
        if hasattr(self._category_resolver, "_category_tools"):
            self._category_resolver._category_tools.clear()

        if hasattr(self._keyword_matcher, "_tool_keywords"):
            self._keyword_matcher._tool_keywords.clear()
        if hasattr(self._keyword_matcher, "_mandatory_keywords"):
            self._keyword_matcher._mandatory_keywords.clear()
        if hasattr(self._keyword_matcher, "_keyword_index"):
            self._keyword_matcher._keyword_index.clear()

    def needs_reindex(self, tools: List[BaseTool]) -> bool:
        """Return True when a tool refresh would change the registry."""
        if self._tools_hash is None:
            return True
        if len(tools) != self._last_refresh_count:
            return True
        return self._calculate_tools_hash(tools) != self._tools_hash

    def refresh_from_tools(self, tools: List[BaseTool], force: bool = False) -> bool:
        """Rebuild the registry from a tool list when needed.

        Returns True when a rebuild happened and False when the current cache
        already matches the provided tools.
        """
        if not force and not self.needs_reindex(tools):
            return False

        self._reset_registered_state()
        self.register_all(tools)
        self._tools_hash = self._calculate_tools_hash(tools)
        self._last_refresh_count = len(tools)
        return True

    def register_tool(self, tool: BaseTool) -> None:
        """Compatibility alias for register()."""
        self.register(tool)

    def register(self, tool: BaseTool) -> None:
        """Register a tool in the metadata registry.

        Args:
            tool: Tool instance to register
        """
        entry = ToolMetadataEntry.from_tool(tool)
        self._entries[entry.name] = entry

        # Index by properties
        self._by_priority[entry.priority].add(entry.name)
        self._by_access_mode[entry.access_mode].add(entry.name)
        self._by_danger_level[entry.danger_level].add(entry.name)

        if entry.category:
            if entry.category not in self._by_category:
                self._by_category[entry.category] = set()
            self._by_category[entry.category].add(entry.name)

        # Index keywords (from @tool decorator)
        for keyword in entry.keywords:
            if keyword not in self._by_keyword:
                self._by_keyword[keyword] = set()
            self._by_keyword[keyword].add(entry.name)

        # Index stages (from @tool decorator)
        for stage in entry.stages:
            if stage not in self._by_stage:
                self._by_stage[stage] = set()
            self._by_stage[stage].add(entry.name)

        # Index aliases
        for alias in entry.aliases:
            self._alias_map[alias] = entry.name

        # NEW: Index mandatory keywords
        for mk in entry.mandatory_keywords:
            if mk not in self._by_mandatory_keyword:
                self._by_mandatory_keyword[mk] = set()
            self._by_mandatory_keyword[mk].add(entry.name)

        # NEW: Index task types
        for tt in entry.task_types:
            if tt not in self._by_task_type:
                self._by_task_type[tt] = set()
            self._by_task_type[tt].add(entry.name)

        # NEW: Index execution category
        self._by_execution_category[entry.execution_category].add(entry.name)

        # Wire composed extracted classes
        if entry.category:
            self._category_resolver.register(entry.name, entry.category)
        if entry.keywords or entry.mandatory_keywords:
            self._keyword_matcher.index_tool(
                name=entry.name,
                keywords=list(entry.keywords),
                mandatory_keywords=(
                    list(entry.mandatory_keywords) if entry.mandatory_keywords else None
                ),
            )
        schema = getattr(tool, "parameters", None)
        if schema:
            self._schema_cache.set(entry.name, schema)

    def unregister_tool(self, tool_name: str) -> None:
        """Remove a tool and its indexes from the registry."""
        entry = self._entries.pop(tool_name, None)
        if entry is None:
            return

        self._by_priority.get(entry.priority, set()).discard(tool_name)
        self._by_access_mode.get(entry.access_mode, set()).discard(tool_name)
        self._by_danger_level.get(entry.danger_level, set()).discard(tool_name)

        if entry.category in self._by_category:
            self._by_category[entry.category].discard(tool_name)
            if not self._by_category[entry.category]:
                del self._by_category[entry.category]

        for keyword in entry.keywords:
            if keyword in self._by_keyword:
                self._by_keyword[keyword].discard(tool_name)
                if not self._by_keyword[keyword]:
                    del self._by_keyword[keyword]

        for stage in entry.stages:
            if stage in self._by_stage:
                self._by_stage[stage].discard(tool_name)
                if not self._by_stage[stage]:
                    del self._by_stage[stage]

        for alias in entry.aliases:
            if self._alias_map.get(alias) == tool_name:
                del self._alias_map[alias]

        for mandatory_keyword in entry.mandatory_keywords:
            if mandatory_keyword in self._by_mandatory_keyword:
                self._by_mandatory_keyword[mandatory_keyword].discard(tool_name)
                if not self._by_mandatory_keyword[mandatory_keyword]:
                    del self._by_mandatory_keyword[mandatory_keyword]

        for task_type in entry.task_types:
            if task_type in self._by_task_type:
                self._by_task_type[task_type].discard(tool_name)
                if not self._by_task_type[task_type]:
                    del self._by_task_type[task_type]

        self._by_execution_category.get(entry.execution_category, set()).discard(tool_name)

        if hasattr(self._category_resolver, "remove_tool"):
            self._category_resolver.remove_tool(tool_name)
        if hasattr(self._keyword_matcher, "remove_tool"):
            self._keyword_matcher.remove_tool(tool_name)
        if hasattr(self._schema_cache, "invalidate"):
            self._schema_cache.invalidate(tool_name)
        elif hasattr(self._schema_cache, "delete"):
            self._schema_cache.delete(tool_name)

    def get_tool_schema(self, tool_name: str) -> Optional[dict]:
        """Get cached schema for a tool by name."""
        return self._schema_cache.get(tool_name)

    def get_metadata(self, name: str) -> Optional[ToolMetadataEntry]:
        """Compatibility alias for get()."""
        return self.get(name)

    def get_all(self) -> Dict[str, ToolMetadataEntry]:
        """Return a copy of all registered metadata entries."""
        return dict(self._entries)

    def register_all(self, tools: List[BaseTool]) -> None:
        """Register multiple tools.

        Args:
            tools: List of tool instances to register
        """
        for tool in tools:
            self.register(tool)

    def get_core_readonly_tools(self) -> List[str]:
        """Get core tools that are explicitly read-only.

        Uses decorator metadata (category + access/exec hints) so the list
        stays current as new tools are added.
        """
        # Always-include read-only set for analysis/exploration safety
        default_readonly = {
            "read",
            "ls",
            "search",
            "overview",
            "docs_coverage",
            "scan",
            "metrics",
            "symbol",
            "refs",
            "graph",  # Code graph traversal for rapid codebase discovery
        }

        core_readonly: List[str] = list(default_readonly)

        # Env override: supports JSON list format (["a", "b"]) or comma-separated
        import os
        import json as _json

        env_raw = os.getenv("CORE_READONLY_TOOLS")
        if env_raw:
            env_raw_stripped = env_raw.strip()
            if env_raw_stripped.startswith("["):
                try:
                    parsed = _json.loads(env_raw_stripped)
                    if isinstance(parsed, list):
                        core_readonly.extend(str(item).strip() for item in parsed if item)
                except (ValueError, TypeError):
                    # Fallback to comma-separated if JSON parse fails
                    core_readonly.extend(
                        [item.strip() for item in env_raw.split(",") if item.strip()]
                    )
            else:
                core_readonly.extend([item.strip() for item in env_raw.split(",") if item.strip()])

        # Runtime override/extension via settings (env: CORE_READONLY_TOOLS)
        try:
            from victor.config.settings import load_settings

            configured = load_settings().core_readonly_tools
            if configured:
                core_readonly.extend(configured)
        except Exception:
            # Settings import failure should not block tool filtering
            pass

        for name, entry in self._entries.items():
            if entry.category != "core":
                # Allow explicitly curated defaults even if not tagged core
                if name not in default_readonly:
                    continue
            access_mode = entry.access_mode.value
            exec_cat = entry.execution_category.value
            if access_mode == "readonly" or exec_cat == "read_only":
                core_readonly.append(name)
        return core_readonly

    def get(self, name: str) -> Optional[ToolMetadataEntry]:
        """Get metadata entry by name (canonical or alias).

        Alias resolution order:
        1. Check canonical name directly
        2. Check @tool decorator aliases (from entry.aliases)
        3. Check global TOOL_ALIASES registry (tool_names.py)

        Args:
            name: Tool name (canonical or alias)

        Returns:
            ToolMetadataEntry or None if not found
        """
        # Check canonical name first
        if name in self._entries:
            return self._entries[name]

        # Check decorator aliases (from entry.aliases)
        canonical = self._alias_map.get(name)
        if canonical:
            return self._entries.get(canonical)

        # Fallback: Check global TOOL_ALIASES registry for backward compatibility
        # This is the single source of truth for legacy name mappings
        from victor.tools.tool_names import get_canonical_name

        resolved = get_canonical_name(name)
        if resolved != name and resolved in self._entries:
            return self._entries.get(resolved)

        return None

    def get_by_priority(self, priority: Priority) -> List[ToolMetadataEntry]:
        """Get all tools with specified priority.

        Args:
            priority: Priority level to filter by

        Returns:
            List of matching tool entries
        """
        return [self._entries[name] for name in self._by_priority.get(priority, set())]

    def get_by_access_mode(self, access_mode: AccessMode) -> List[ToolMetadataEntry]:
        """Get all tools with specified access mode.

        Args:
            access_mode: Access mode to filter by

        Returns:
            List of matching tool entries
        """
        return [self._entries[name] for name in self._by_access_mode.get(access_mode, set())]

    def get_by_danger_level(self, danger_level: DangerLevel) -> List[ToolMetadataEntry]:
        """Get all tools with specified danger level.

        Args:
            danger_level: Danger level to filter by

        Returns:
            List of matching tool entries
        """
        return [self._entries[name] for name in self._by_danger_level.get(danger_level, set())]

    def get_by_category(self, category: str) -> List[ToolMetadataEntry]:
        """Get all tools in specified category.

        Args:
            category: Category name

        Returns:
            List of matching tool entries
        """
        return [self._entries[name] for name in self._by_category.get(category, set())]

    def get_tools_by_category(self, category: str) -> Set[str]:
        """Get tool names in the specified category.

        Args:
            category: Category name (e.g., "git", "testing", "security")

        Returns:
            Set of tool names in this category
        """
        return self._by_category.get(category, set()).copy()

    def get_all_categories(self) -> Set[str]:
        """Get all registered categories.

        Returns:
            Set of all category names from tools with @tool(category="...")
        """
        return set(self._by_category.keys())

    def get_category_keywords(self, category: str) -> Set[str]:
        """Get all keywords from tools in a category.

        Aggregates keywords from all tools in the specified category,
        enabling dynamic category detection from user messages.

        Args:
            category: Category name to look up

        Returns:
            Set of all keywords from tools in this category
        """
        result: Set[str] = set()
        for tool_name in self._by_category.get(category, set()):
            entry = self._entries.get(tool_name)
            if entry:
                result.update(entry.keywords)
        return result

    def get_all_category_keywords(self) -> Dict[str, Set[str]]:
        """Get mapping of categories to aggregated keywords.

        Builds a complete mapping of category names to the combined keywords
        of all tools in each category. This replaces static CATEGORY_KEYWORDS
        dictionaries with decorator-driven discovery.

        Returns:
            Dictionary mapping category names to sets of keywords
        """
        return {cat: self.get_category_keywords(cat) for cat in self._by_category.keys()}

    def detect_categories_from_text(self, text: str) -> Set[str]:
        """Detect relevant categories from keywords in text.

        Scans the text for keywords that match tools in each category,
        returning categories where at least one keyword is found.

        Uses Rust-accelerated Aho-Corasick pattern matching when available,
        falling back to Python string matching otherwise.

        Args:
            text: Text to analyze for category keywords

        Returns:
            Set of category names where matching keywords were found
        """
        detected: Set[str] = set()

        for category in self._by_category.keys():
            keywords = self.get_category_keywords(category)
            if not keywords:
                continue

            keywords_list = list(keywords)

            # Use Rust Aho-Corasick when available (O(text_len) vs O(k * text_len))
            matched = _get_matched_pattern_indices(text, keywords_list)
            if matched:
                detected.add(category)

        return detected

    def get_by_keyword(self, keyword: str) -> List[ToolMetadataEntry]:
        """Get all tools with specified keyword.

        Args:
            keyword: Keyword to search for (case-insensitive)

        Returns:
            List of matching tool entries
        """
        keyword_lower = keyword.lower()
        return [self._entries[name] for name in self._by_keyword.get(keyword_lower, set())]

    def get_tools_by_keywords(self, keywords: List[str]) -> Set[str]:
        """Get tool names matching any of the given keywords.

        Args:
            keywords: List of keywords to match

        Returns:
            Set of tool names that have any matching keyword
        """
        result: Set[str] = set()
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in self._by_keyword:
                result.update(self._by_keyword[keyword_lower])
        return result

    def get_by_stage(self, stage: str) -> List[ToolMetadataEntry]:
        """Get all tools relevant for the specified conversation stage.

        Args:
            stage: Conversation stage name (e.g., "reading", "execution")
                   Case-insensitive.

        Returns:
            List of tools that have this stage in their stages list
        """
        stage_lower = stage.lower()
        return [self._entries[name] for name in self._by_stage.get(stage_lower, set())]

    def get_tools_by_stage(self, stage: str) -> Set[str]:
        """Get tool names relevant for the specified conversation stage.

        Args:
            stage: Conversation stage name (e.g., "reading", "execution")
                   Case-insensitive.

        Returns:
            Set of tool names that have this stage in their stages list
        """
        stage_lower = stage.lower()
        return self._by_stage.get(stage_lower, set()).copy()

    def get_all_stages(self) -> Set[str]:
        """Get all stages discovered from registered tools.

        Returns:
            Set of all stage names from tools with @tool(stages=[...])
        """
        return set(self._by_stage.keys())

    def get_stage_tool_mapping(self) -> Dict[str, Set[str]]:
        """Get complete mapping of stages to tool names.

        This replaces static STAGE_TOOL_MAPPING dictionaries by
        dynamically building the mapping from @tool decorator metadata.

        Returns:
            Dictionary mapping stage names to sets of tool names
        """
        return {stage: tools.copy() for stage, tools in self._by_stage.items()}

    # =========================================================================
    # NEW: Decorator-driven semantic selection methods
    # =========================================================================

    def get_tools_by_mandatory_keyword(self, keyword: str) -> Set[str]:
        """Get tools that must be included when this keyword is matched.

        Unlike regular keywords which influence ranking, mandatory keywords
        guarantee tool inclusion. Use for phrases like "show diff", "run tests".

        Args:
            keyword: Mandatory keyword to look up (case-insensitive)

        Returns:
            Set of tool names that have this as a mandatory keyword
        """
        return self._by_mandatory_keyword.get(keyword.lower(), set()).copy()

    def get_tools_matching_mandatory_keywords(self, text: str) -> Set[str]:
        """Find tools whose mandatory keywords appear in the given text.

        Scans the text for all registered mandatory keywords and returns
        tools that MUST be included when these phrases are matched.

        Args:
            text: Text to scan for mandatory keywords

        Returns:
            Set of tool names whose mandatory keywords match (guaranteed inclusion)
        """
        text_lower = text.lower()
        result: Set[str] = set()
        for keyword, tool_names in self._by_mandatory_keyword.items():
            if keyword in text_lower:
                result.update(tool_names)
        return result

    def get_all_mandatory_keywords(self) -> Set[str]:
        """Get all registered mandatory keywords.

        Returns:
            Set of all mandatory keywords from tools with @tool(mandatory_keywords=[...])
        """
        return set(self._by_mandatory_keyword.keys())

    def get_tools_by_task_type(self, task_type: str) -> Set[str]:
        """Get tools relevant for the specified task type.

        Enables classification-aware tool selection where tools can declare
        which task types they're designed for (analysis, action, generation, etc.).

        Args:
            task_type: Task type to look up (e.g., "analysis", "action", "generation")
                       Case-insensitive.

        Returns:
            Set of tool names that have this task type in their task_types list
        """
        return self._by_task_type.get(task_type.lower(), set()).copy()

    def get_tools_for_task_classification(self, task_type: str) -> List[ToolMetadataEntry]:
        """Get tool entries relevant for the specified task classification.

        Returns full metadata entries for task-type matching, enabling
        consumers to further filter by priority, danger level, etc.

        Args:
            task_type: Task classification (e.g., "analysis", "search", "edit")

        Returns:
            List of ToolMetadataEntry for tools matching this task type
        """
        tool_names = self.get_tools_by_task_type(task_type)
        return [self._entries[name] for name in tool_names if name in self._entries]

    def get_all_task_types(self) -> Set[str]:
        """Get all registered task types.

        Returns:
            Set of all task types from tools with @tool(task_types=[...])
        """
        return set(self._by_task_type.keys())

    def get_task_type_tool_mapping(self) -> Dict[str, Set[str]]:
        """Get complete mapping of task types to tool names.

        This replaces static TASK_TYPE_CATEGORIES dictionaries by
        dynamically building the mapping from @tool decorator metadata.

        Returns:
            Dictionary mapping task type names to sets of tool names
        """
        return {tt: tools.copy() for tt, tools in self._by_task_type.items()}

    def get_tools_by_execution_category(self, category: ExecutionCategory) -> Set[str]:
        """Get tools in the specified execution category.

        Used for parallel execution planning to identify which tools
        can safely run concurrently.

        Args:
            category: ExecutionCategory to look up

        Returns:
            Set of tool names in this execution category
        """
        return self._by_execution_category.get(category, set()).copy()

    def get_parallelizable_tools(self) -> Set[str]:
        """Get all tools that are safe to run in parallel.

        Returns tools in READ_ONLY, COMPUTE, and NETWORK categories
        which can safely run concurrently without conflicts.

        Returns:
            Set of tool names that are safe to parallelize
        """
        result: Set[str] = set()
        for category in ExecutionCategory:
            if category.can_parallelize:
                result.update(self._by_execution_category.get(category, set()))
        return result

    def get_conflicting_tools(self, tool_name: str) -> Set[str]:
        """Get tools that would conflict with the given tool.

        Used for dependency analysis in parallel execution.

        Args:
            tool_name: Name of the tool to check conflicts for

        Returns:
            Set of tool names that conflict with this tool's execution category
        """
        entry = self._entries.get(tool_name)
        if not entry:
            return set()

        conflicts = entry.execution_category.conflicts_with
        result: Set[str] = set()
        for conflicting_cat in conflicts:
            result.update(self._by_execution_category.get(conflicting_cat, set()))

        # Remove self from conflicts
        result.discard(tool_name)
        return result

    def get_signature_params(self, tool_name: str) -> Set[str]:
        """Get signature parameters for a tool.

        Signature params are the only parameters included in loop detection
        signatures. Pagination parameters (offset, limit, k, etc.) should be
        excluded to allow reading results in chunks.

        Args:
            tool_name: Name of the tool

        Returns:
            Set of parameter names to include in signatures, or empty set
        """
        entry = self._entries.get(tool_name)
        if entry:
            return entry.signature_params.copy()
        return set()

    def get_tools_with_signature_params(self) -> Dict[str, Set[str]]:
        """Get all tools that have signature parameters defined.

        Returns:
            Dictionary mapping tool names to their signature parameter sets
        """
        return {
            name: entry.signature_params.copy()
            for name, entry in self._entries.items()
            if entry.signature_params
        }

    def get_execution_category_mapping(self) -> Dict[str, Set[str]]:
        """Get complete mapping of execution categories to tool names.

        This replaces static TOOL_CATEGORIES dictionaries in parallel_executor
        by dynamically building the mapping from @tool decorator metadata.

        Returns:
            Dictionary mapping execution category values to sets of tool names
        """
        return {cat.value: tools.copy() for cat, tools in self._by_execution_category.items()}

    # =========================================================================
    # Access mode-based tool discovery (replaces static WRITE_TOOL_NAMES, etc.)
    # =========================================================================

    def get_write_tools(self) -> Set[str]:
        """Get tools that perform write/modify operations.

        Replaces static WRITE_TOOL_NAMES frozensets in safety.py, action_authorizer.py.
        Uses access_mode from @tool decorator as single source of truth.

        Returns:
            Set of tool names with WRITE, EXECUTE, or MIXED access_mode
        """
        result: Set[str] = set()
        for mode in [AccessMode.WRITE, AccessMode.EXECUTE, AccessMode.MIXED]:
            result.update(self._by_access_mode.get(mode, set()))
        return result

    def get_idempotent_tools(self) -> Set[str]:
        """Get tools that are safe to cache (readonly, no side effects).

        Replaces static DEFAULT_CACHEABLE_TOOLS in tool_executor.py.
        Uses access_mode=READONLY as indicator of idempotency.

        Returns:
            Set of tool names with READONLY access_mode
        """
        return self._by_access_mode.get(AccessMode.READONLY, set()).copy()

    def get_cache_invalidating_tools(self) -> Set[str]:
        """Get tools that modify state and should invalidate cache.

        Replaces static CACHE_INVALIDATING_TOOLS in tool_executor.py.
        Uses access_mode from @tool decorator.

        Returns:
            Set of tool names with WRITE, EXECUTE, or MIXED access_mode
        """
        # Same as get_write_tools() - cache invalidation = state modification
        return self.get_write_tools()

    def get_tools_matching_text(self, text: str) -> Set[str]:
        """Find tools whose keywords appear in the given text.

        Scans the text for all registered keywords and returns tools
        that have matching keywords. Useful for goal inference.

        Uses Rust-accelerated Aho-Corasick pattern matching when available,
        falling back to Python string matching otherwise.

        Args:
            text: Text to scan for keywords

        Returns:
            Set of tool names whose keywords match
        """
        result: Set[str] = set()

        # Get all keywords and their indices for Rust pattern matching
        all_keywords = list(self._by_keyword.keys())
        if not all_keywords:
            return result

        # Use Rust Aho-Corasick when available (single pass through text)
        matched_indices = _get_matched_pattern_indices(text, all_keywords)
        for idx in matched_indices:
            if 0 <= idx < len(all_keywords):
                keyword = all_keywords[idx]
                result.update(self._by_keyword[keyword])

        return result

    def get_tools_matching_text_scored(
        self,
        text: str,
        min_score: float = 0.0,
        max_results: Optional[int] = None,
        use_fallback: bool = True,
    ) -> List[KeywordMatchResult]:
        """Find tools matching text with relevance scores.

        Enhanced version of get_tools_matching_text() that returns scored results.
        Score is based on:
        - Number of matching keywords (more = higher score)
        - Keyword specificity (longer keywords score higher)
        - Tool priority (CRITICAL tools get a boost)

        Args:
            text: Text to scan for keywords
            min_score: Minimum score threshold (0.0 to 1.0)
            max_results: Maximum number of results to return
            use_fallback: If True, return fallback tools when no matches found

        Returns:
            List of KeywordMatchResult sorted by score (highest first)
        """
        start_time = time.time()

        # Collect matches per tool using Rust-accelerated pattern matching
        tool_matches: Dict[str, Set[str]] = {}
        matched_keywords: Set[str] = set()

        # Get all keywords for single-pass matching
        all_keywords = list(self._by_keyword.keys())
        if all_keywords:
            # Use Rust Aho-Corasick when available (O(text_len) vs O(k * text_len))
            matched_indices = _get_matched_pattern_indices(text, all_keywords)

            # Map matched indices back to keywords and tools
            for idx in matched_indices:
                if 0 <= idx < len(all_keywords):
                    keyword = all_keywords[idx]
                    matched_keywords.add(keyword)
                    for tool_name in self._by_keyword[keyword]:
                        if tool_name not in tool_matches:
                            tool_matches[tool_name] = set()
                        tool_matches[tool_name].add(keyword)

        # Calculate scores
        results: List[KeywordMatchResult] = []
        for tool_name, keywords in tool_matches.items():
            entry = self._entries.get(tool_name)
            if not entry:
                continue

            # Base score: ratio of matched keywords to total keywords
            total_keywords = len(entry.keywords) if entry.keywords else 1
            base_score = len(keywords) / total_keywords

            # Specificity bonus: longer keywords are more specific
            specificity_bonus = sum(len(k) for k in keywords) / (len(keywords) * 10)
            specificity_bonus = min(specificity_bonus, 0.2)  # Cap at 0.2

            score = min(base_score + specificity_bonus, 1.0)

            # Priority boost
            priority_boost = self._PRIORITY_WEIGHTS.get(entry.priority, 0.0)

            if score + priority_boost >= min_score:
                results.append(
                    KeywordMatchResult(
                        tool_name=tool_name,
                        score=score,
                        matched_keywords=keywords,
                        priority_boost=priority_boost,
                    )
                )

        # Sort by total score (descending)
        results.sort(key=lambda r: r.total_score, reverse=True)

        # Apply max_results limit
        if max_results and len(results) > max_results:
            results = results[:max_results]

        # Fallback to critical tools if no matches
        used_fallback = False
        if not results and use_fallback:
            used_fallback = True
            for tool_name in _FALLBACK_CRITICAL_TOOLS:
                if tool_name in self._entries:
                    results.append(
                        KeywordMatchResult(
                            tool_name=tool_name,
                            score=0.0,
                            matched_keywords=set(),
                            priority_boost=0.0,
                        )
                    )

        # Record metrics
        duration_ms = (time.time() - start_time) * 1000
        self._metrics.record_match(
            duration_ms=duration_ms,
            tools_matched=len(results),
            keywords_matched=matched_keywords,
            used_fallback=used_fallback,
        )

        return results

    def get_all_categories(self) -> Set[str]:
        """Get all categories discovered from registered tools.

        Returns:
            Set of all category names from tools with @tool(category=...)
        """
        return set(self._by_category.keys())

    def get_tools_by_category_with_fallback(
        self,
        category: str,
        fallback_categories: Optional[List[str]] = None,
    ) -> List[ToolMetadataEntry]:
        """Get tools by category with fallback chain.

        If the requested category has no tools, tries fallback categories
        in order until tools are found.

        Args:
            category: Primary category to search
            fallback_categories: List of fallback categories to try

        Returns:
            List of tools from the first category with matches
        """
        tools = self.get_by_category(category)
        if tools:
            self._metrics.record_category_hit(category)
            return tools

        if fallback_categories:
            for fallback in fallback_categories:
                tools = self.get_by_category(fallback)
                if tools:
                    self._metrics.record_category_hit(fallback)
                    logger.debug(f"Category '{category}' empty, using fallback '{fallback}'")
                    return tools

        return []

    @property
    def metrics(self) -> MatchingMetrics:
        """Get matching metrics for observability."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset matching metrics."""
        self._metrics = MatchingMetrics()

    def get_all_keywords(self) -> Set[str]:
        """Get all registered keywords.

        Returns:
            Set of all keywords from registered tools
        """
        return set(self._by_keyword.keys())

    def get_all_tool_names(self) -> Set[str]:
        """Get all registered tool names.

        Returns:
            Set of all registered tool names (canonical names only)
        """
        return set(self._entries.keys())

    def get_fallback_tools_for_category(self, category: str) -> Set[str]:
        """Get fallback tools for a logical category.

        Used when semantic selection returns empty results for a category.
        Returns tools from the specified execution category.

        Args:
            category: Logical category name (e.g., "file_ops", "git_ops")

        Returns:
            Set of tool names that serve as fallback for this category
        """
        # Map logical categories to execution categories or tool categories
        CATEGORY_TO_EXECUTION = {
            "file_ops": ExecutionCategory.READ_ONLY,
            "execution": ExecutionCategory.EXECUTE,
            "generation": ExecutionCategory.WRITE,
            "compute": ExecutionCategory.COMPUTE,
        }

        if category in CATEGORY_TO_EXECUTION:
            return self._by_execution_category.get(CATEGORY_TO_EXECUTION[category], set()).copy()

        # Try direct category lookup
        return self._by_category.get(category, set()).copy()

    def get_critical_tools(self) -> List[ToolMetadataEntry]:
        """Get all CRITICAL priority tools."""
        return self.get_by_priority(Priority.CRITICAL)

    def get_safe_tools(self) -> List[ToolMetadataEntry]:
        """Get all safe tools (readonly, no danger)."""
        return [entry for entry in self._entries.values() if entry.is_safe]

    def get_dangerous_tools(self) -> List[ToolMetadataEntry]:
        """Get all dangerous tools (HIGH or CRITICAL danger level)."""
        high = self.get_by_danger_level(DangerLevel.HIGH)
        critical = self.get_by_danger_level(DangerLevel.CRITICAL)
        return high + critical

    def get_tools_requiring_approval(self) -> List[ToolMetadataEntry]:
        """Get all tools that require user approval."""
        return [entry for entry in self._entries.values() if entry.requires_approval]

    def get_tools_for_task(self, task_type: str) -> List[ToolMetadataEntry]:
        """Get tools appropriate for a task type.

        Args:
            task_type: Task type (e.g., "analysis", "edit", "search")

        Returns:
            List of tools that should be included for this task
        """
        return [
            entry for entry in self._entries.values() if entry.should_include_for_task(task_type)
        ]

    def get_tools_up_to_priority(self, max_priority: Priority) -> List[ToolMetadataEntry]:
        """Get all tools with priority <= max_priority (lower value = higher priority).

        Args:
            max_priority: Maximum priority level to include

        Returns:
            List of tools at or above this priority
        """
        return [
            entry for entry in self._entries.values() if entry.priority.value <= max_priority.value
        ]

    def filter(
        self,
        priority: Optional[Priority] = None,
        access_mode: Optional[AccessMode] = None,
        danger_level: Optional[DangerLevel] = None,
        category: Optional[str] = None,
        safe_only: bool = False,
        requires_approval: Optional[bool] = None,
    ) -> List[ToolMetadataEntry]:
        """Filter tools by multiple criteria.

        Args:
            priority: Filter by priority level
            access_mode: Filter by access mode
            danger_level: Filter by danger level
            category: Filter by category
            safe_only: If True, only return safe tools
            requires_approval: Filter by approval requirement

        Returns:
            List of tools matching all specified criteria
        """
        results = list(self._entries.values())

        if priority is not None:
            priority_names = self._by_priority.get(priority, set())
            results = [e for e in results if e.name in priority_names]

        if access_mode is not None:
            access_names = self._by_access_mode.get(access_mode, set())
            results = [e for e in results if e.name in access_names]

        if danger_level is not None:
            danger_names = self._by_danger_level.get(danger_level, set())
            results = [e for e in results if e.name in danger_names]

        if category is not None:
            category_names = self._by_category.get(category, set())
            results = [e for e in results if e.name in category_names]

        if safe_only:
            results = [e for e in results if e.is_safe]

        if requires_approval is not None:
            results = [e for e in results if e.requires_approval == requires_approval]

        return results

    # =========================================================================
    # Availability filtering for optional tools
    # =========================================================================

    def get_available_tools(self) -> List[ToolMetadataEntry]:
        """Get all tools that are currently available for use.

        Filters out tools that require configuration (e.g., Slack, Teams, Jira)
        but are not yet configured. Tools without an availability_check are
        always considered available.

        Returns:
            List of ToolMetadataEntry for tools that are available
        """
        return [entry for entry in self._entries.values() if entry.is_available()]

    def get_unavailable_tools(self) -> List[ToolMetadataEntry]:
        """Get tools that require configuration but are not currently configured.

        Useful for informing users about tools that could be enabled with
        proper configuration.

        Returns:
            List of ToolMetadataEntry for tools that are unavailable
        """
        return [
            entry
            for entry in self._entries.values()
            if entry.requires_configuration and not entry.is_available()
        ]

    def get_tools_requiring_configuration(self) -> List[ToolMetadataEntry]:
        """Get all tools that require external configuration.

        These are tools with an availability_check (e.g., Slack, Teams, Jira)
        regardless of whether they are currently configured.

        Returns:
            List of ToolMetadataEntry for tools with availability_check
        """
        return [entry for entry in self._entries.values() if entry.requires_configuration]

    def get_available_tool_names(self) -> Set[str]:
        """Get names of all currently available tools.

        Convenience method for tool selection that returns just the names
        of available tools for filtering.

        Returns:
            Set of tool names that are currently available
        """
        return {entry.name for entry in self._entries.values() if entry.is_available()}

    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._entries)

    def __iter__(self) -> Iterator[ToolMetadataEntry]:
        """Iterate over all registered entries."""
        return iter(self._entries.values())

    def __contains__(self, name: str) -> bool:
        """Check if a tool name is registered (supports aliases).

        Checks:
        1. Canonical names in registry
        2. Decorator aliases (from @tool decorator)
        3. Global TOOL_ALIASES (from tool_names.py)
        """
        if name in self._entries:
            return True
        if name in self._alias_map:
            return True
        # Check global TOOL_ALIASES
        from victor.tools.tool_names import get_canonical_name

        resolved = get_canonical_name(name)
        return resolved != name and resolved in self._entries

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of registered tools.

        Returns:
            Dictionary with counts by priority, access_mode, danger_level, stage,
            task_type, execution_category, and other metadata.
        """
        return {
            "total": len(self._entries),
            "by_priority": {p.name: len(names) for p, names in self._by_priority.items()},
            "by_access_mode": {a.name: len(names) for a, names in self._by_access_mode.items()},
            "by_danger_level": {d.name: len(names) for d, names in self._by_danger_level.items()},
            "by_category": {cat: len(names) for cat, names in self._by_category.items()},
            "by_stage": {stage: len(names) for stage, names in self._by_stage.items()},
            # NEW: Additional semantic selection statistics
            "by_task_type": {tt: len(names) for tt, names in self._by_task_type.items()},
            "by_execution_category": {
                ec.value: len(names) for ec, names in self._by_execution_category.items()
            },
            "mandatory_keywords_count": len(self._by_mandatory_keyword),
            "tools_with_signature_params": len(
                [e for e in self._entries.values() if e.signature_params]
            ),
            "requiring_approval": len(self.get_tools_requiring_approval()),
            "safe_tools": len(self.get_safe_tools()),
            "dangerous_tools": len(self.get_dangerous_tools()),
            "parallelizable_tools": len(self.get_parallelizable_tools()),
            # Availability statistics
            "available_tools": len(self.get_available_tools()),
            "unavailable_tools": len(self.get_unavailable_tools()),
            "tools_requiring_configuration": len(self.get_tools_requiring_configuration()),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Compatibility summary matching the older registry API."""
        total_tools = len(self._entries)
        tools_with_explicit_metadata = sum(
            1 for entry in self._entries.values() if entry.category and entry.keywords
        )
        return {
            "total_tools": total_tools,
            "tools_with_explicit_metadata": tools_with_explicit_metadata,
            "tools_with_auto_metadata": total_tools - tools_with_explicit_metadata,
            "total_categories": len(self._by_category),
            "total_keywords": len(self._by_keyword),
            "categories": sorted(self._by_category.keys()),
        }


# Global singleton instance
_global_registry: Optional[ToolMetadataRegistry] = None


def get_global_registry() -> ToolMetadataRegistry:
    """Get or create the global metadata registry.

    Returns:
        The global ToolMetadataRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolMetadataRegistry()
    return _global_registry


def reset_global_registry() -> None:
    """Reset the global metadata registry singleton."""
    global _global_registry
    _global_registry = None


def register_tool_metadata(tool: BaseTool) -> None:
    """Register a tool in the global registry.

    Args:
        tool: Tool to register
    """
    get_global_registry().register(tool)


def get_tool_metadata(name: str) -> Optional[ToolMetadataEntry]:
    """Get tool metadata from the global registry.

    Args:
        name: Tool name (canonical or alias)

    Returns:
        ToolMetadataEntry or None
    """
    return get_global_registry().get(name)


def get_fallback_critical_tools() -> Set[str]:
    """Get the fallback critical tools set.

    Returns:
        Set of tool names used as fallback when no keyword matches found
    """
    return _FALLBACK_CRITICAL_TOOLS.copy()


def get_tools_matching_text_scored(
    text: str,
    min_score: float = 0.0,
    max_results: Optional[int] = None,
) -> List[KeywordMatchResult]:
    """Get tools matching text with scores from global registry.

    Convenience function for scored matching using the global registry.

    Args:
        text: Text to scan for keywords
        min_score: Minimum score threshold
        max_results: Maximum number of results

    Returns:
        List of KeywordMatchResult sorted by score
    """
    return get_global_registry().get_tools_matching_text_scored(
        text=text,
        min_score=min_score,
        max_results=max_results,
    )


def get_matching_metrics() -> MatchingMetrics:
    """Get matching metrics from global registry.

    Returns:
        MatchingMetrics instance with keyword matching statistics
    """
    return get_global_registry().metrics


def get_tools_by_stage(stage: str) -> Set[str]:
    """Get tool names relevant for the specified conversation stage.

    This function enables decorator-driven stage-based tool selection,
    replacing static STAGE_TOOL_MAPPING dictionaries.

    Args:
        stage: Conversation stage name (e.g., "reading", "execution")
               Case-insensitive.

    Returns:
        Set of tool names that have this stage in their stages list
    """
    return get_global_registry().get_tools_by_stage(stage)


def get_stage_tool_mapping() -> Dict[str, Set[str]]:
    """Get complete mapping of stages to tool names from global registry.

    This replaces static STAGE_TOOL_MAPPING dictionaries by
    dynamically building the mapping from @tool decorator metadata.

    Returns:
        Dictionary mapping stage names to sets of tool names
    """
    return get_global_registry().get_stage_tool_mapping()


def get_all_stages() -> Set[str]:
    """Get all stages discovered from registered tools.

    Returns:
        Set of all stage names from tools with @tool(stages=[...])
    """
    return get_global_registry().get_all_stages()


# =========================================================================
# NEW: Module-level convenience functions for decorator-driven selection
# =========================================================================


def get_tools_matching_mandatory_keywords(text: str) -> Set[str]:
    """Find tools whose mandatory keywords appear in the given text.

    Scans the text for all registered mandatory keywords and returns
    tools that MUST be included when these phrases are matched.

    Args:
        text: Text to scan for mandatory keywords

    Returns:
        Set of tool names whose mandatory keywords match (guaranteed inclusion)
    """
    return get_global_registry().get_tools_matching_mandatory_keywords(text)


def get_tools_by_task_type(task_type: str) -> Set[str]:
    """Get tools relevant for the specified task type.

    Enables classification-aware tool selection where tools can declare
    which task types they're designed for (analysis, action, generation, etc.).

    Args:
        task_type: Task type to look up (e.g., "analysis", "action", "generation")

    Returns:
        Set of tool names that have this task type in their task_types list
    """
    return get_global_registry().get_tools_by_task_type(task_type)


def get_task_type_tool_mapping() -> Dict[str, Set[str]]:
    """Get complete mapping of task types to tool names from global registry.

    This replaces static TASK_TYPE_CATEGORIES dictionaries by
    dynamically building the mapping from @tool decorator metadata.

    Returns:
        Dictionary mapping task type names to sets of tool names
    """
    return get_global_registry().get_task_type_tool_mapping()


def get_tools_by_execution_category(category: ExecutionCategory) -> Set[str]:
    """Get tools in the specified execution category.

    Used for parallel execution planning to identify which tools
    can safely run concurrently.

    Args:
        category: ExecutionCategory to look up

    Returns:
        Set of tool names in this execution category
    """
    return get_global_registry().get_tools_by_execution_category(category)


def get_parallelizable_tools() -> Set[str]:
    """Get all tools that are safe to run in parallel.

    Returns tools in READ_ONLY, COMPUTE, and NETWORK categories
    which can safely run concurrently without conflicts.

    Returns:
        Set of tool names that are safe to parallelize
    """
    return get_global_registry().get_parallelizable_tools()


def get_signature_params(tool_name: str) -> Set[str]:
    """Get signature parameters for a tool.

    Signature params are the only parameters included in loop detection
    signatures. Pagination parameters should be excluded.

    Args:
        tool_name: Name of the tool

    Returns:
        Set of parameter names to include in signatures, or empty set
    """
    return get_global_registry().get_signature_params(tool_name)


def get_execution_category_mapping() -> Dict[str, Set[str]]:
    """Get complete mapping of execution categories to tool names.

    This replaces static TOOL_CATEGORIES dictionaries in parallel_executor
    by dynamically building the mapping from @tool decorator metadata.

    Returns:
        Dictionary mapping execution category values to sets of tool names
    """
    return get_global_registry().get_execution_category_mapping()


# =========================================================================
# Access mode-based tool discovery (module-level convenience functions)
# =========================================================================


def get_write_tools() -> Set[str]:
    """Get tools that perform write/modify operations.

    Replaces static WRITE_TOOL_NAMES frozensets in safety.py, action_authorizer.py.
    Uses access_mode from @tool decorator as single source of truth.

    Returns:
        Set of tool names with WRITE, EXECUTE, or MIXED access_mode
    """
    return get_global_registry().get_write_tools()


def get_idempotent_tools() -> Set[str]:
    """Get tools that are safe to cache (readonly, no side effects).

    Replaces static DEFAULT_CACHEABLE_TOOLS in tool_executor.py.
    Uses access_mode=READONLY as indicator of idempotency.

    Returns:
        Set of tool names with READONLY access_mode
    """
    return get_global_registry().get_idempotent_tools()


def get_cache_invalidating_tools() -> Set[str]:
    """Get tools that modify state and should invalidate cache.

    Replaces static CACHE_INVALIDATING_TOOLS in tool_executor.py.
    Uses access_mode from @tool decorator.

    Returns:
        Set of tool names with WRITE, EXECUTE, or MIXED access_mode
    """
    return get_global_registry().get_cache_invalidating_tools()


# =========================================================================
# Category-based tool discovery (module-level convenience functions)
# =========================================================================


def get_tools_by_category(category: str) -> Set[str]:
    """Get tool names in the specified category.

    Args:
        category: Category name (e.g., "git", "testing", "security")

    Returns:
        Set of tool names in this category
    """
    return get_global_registry().get_tools_by_category(category)


def get_all_categories() -> Set[str]:
    """Get all registered categories.

    Returns:
        Set of all category names from tools with @tool(category="...")
    """
    return get_global_registry().get_all_categories()


def get_category_keywords(category: str) -> Set[str]:
    """Get all keywords from tools in a category.

    Aggregates keywords from all tools in the specified category,
    enabling dynamic category detection from user messages.

    Args:
        category: Category name to look up

    Returns:
        Set of all keywords from tools in this category
    """
    return get_global_registry().get_category_keywords(category)


def get_all_category_keywords() -> Dict[str, Set[str]]:
    """Get mapping of categories to aggregated keywords.

    Builds a complete mapping of category names to the combined keywords
    of all tools in each category. This replaces static CATEGORY_KEYWORDS
    dictionaries with decorator-driven discovery.

    Returns:
        Dictionary mapping category names to sets of keywords
    """
    return get_global_registry().get_all_category_keywords()


def detect_categories_from_text(text: str) -> Set[str]:
    """Detect relevant categories from keywords in text.

    Scans the text for keywords that match tools in each category,
    returning categories where at least one keyword is found.

    This replaces static CATEGORY_KEYWORDS-based detection in
    tool_selection.py with decorator-driven discovery.

    Args:
        text: Text to analyze for category keywords

    Returns:
        Set of category names where matching keywords were found
    """
    return get_global_registry().detect_categories_from_text(text)


# =========================================================================
# Availability filtering (module-level convenience functions)
# =========================================================================


def get_available_tools() -> List[ToolMetadataEntry]:
    """Get all tools that are currently available for use.

    Filters out tools that require configuration (e.g., Slack, Teams, Jira)
    but are not yet configured.

    Returns:
        List of ToolMetadataEntry for tools that are available
    """
    return get_global_registry().get_available_tools()


def get_unavailable_tools() -> List[ToolMetadataEntry]:
    """Get tools that require configuration but are not currently configured.

    Returns:
        List of ToolMetadataEntry for tools that are unavailable
    """
    return get_global_registry().get_unavailable_tools()


def get_tools_requiring_configuration() -> List[ToolMetadataEntry]:
    """Get all tools that require external configuration.

    Returns:
        List of ToolMetadataEntry for tools with availability_check
    """
    return get_global_registry().get_tools_requiring_configuration()


def get_available_tool_names() -> Set[str]:
    """Get names of all currently available tools.

    Returns:
        Set of tool names that are currently available
    """
    return get_global_registry().get_available_tool_names()


def is_tool_available(tool_name: str) -> bool:
    """Check if a specific tool is available.

    Args:
        tool_name: Name of the tool to check

    Returns:
        True if the tool exists and is available, False otherwise
    """
    entry = get_global_registry().get(tool_name)
    if entry is None:
        return False
    return entry.is_available()


def get_core_readonly_tools() -> List[str]:
    """Get names of core tools that are explicitly read-only."""
    return get_global_registry().get_core_readonly_tools()
