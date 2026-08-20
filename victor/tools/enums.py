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

"""Enumerations for tool framework."""

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from victor.core.severity import SeverityLevel

# Lazy-loaded presentation adapter for icons
_presentation = None


def _get_icon(name: str) -> str:
    """Get icon from presentation adapter (lazy initialization)."""
    global _presentation
    if _presentation is None:
        from victor.agent.presentation import create_presentation_adapter

        _presentation = create_presentation_adapter()
    return _presentation.icon(name, with_color=False)


class CostTier(Enum):
    """Cost tier for tools.

    Used for cost-aware tool selection to deprioritize expensive tools
    when cheaper alternatives exist.

    Tiers:
        FREE: Local operations with no external costs (filesystem, bash, git)
        LOW: Compute-only operations (code review, refactoring analysis)
        MEDIUM: External API calls (web search, web fetch)
        HIGH: Resource-intensive operations (batch processing 100+ files)
    """

    FREE = "free"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def weight(self) -> float:
        """Return numeric weight for cost comparison."""
        weights = {
            CostTier.FREE: 0.0,
            CostTier.LOW: 1.0,
            CostTier.MEDIUM: 2.0,
            CostTier.HIGH: 3.0,
        }
        return weights[self]


class Priority(Enum):
    """Tool priority levels for selection availability.

    Decoupled from category - determines when tools are included in LLM context.
    Higher priority tools are more likely to be included in limited contexts.

    Priority Levels:
        CRITICAL (1): Always available - essential for any task
                      Examples: read, ls, grep, shell
        HIGH (2): Included for most tasks - important operations
                  Examples: write, edit, git
        MEDIUM (3): Task-specific - included when relevant
                    Examples: docker, db, test
        LOW (4): Specialized - only when specifically needed
                 Examples: batch, scaffold, workflow
        CONTEXTUAL (5): Only included based on task classification
                        Examples: web_search (only for research tasks)
    """

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    CONTEXTUAL = 5

    @property
    def weight(self) -> int:
        """Lower weight = higher priority (more likely to be selected)."""
        return self.value

    def should_include_for_task(self, task_type: str) -> bool:
        """Check if this priority level should be included for a task type.

        Args:
            task_type: Task type (e.g., "analysis", "edit", "search")

        Returns:
            True if tool should be included
        """
        # CRITICAL and HIGH are always included
        if self.value <= 2:
            return True
        # MEDIUM included for most tasks
        if self.value == 3:
            return task_type != "simple"
        # LOW and CONTEXTUAL need specific conditions
        return False


class AccessMode(Enum):
    """Tool access mode for approval tracking and security policies.

    Determines what kind of access a tool requires, enabling:
    - Auto-approval for read-only tools in safe mode
    - Confirmation prompts for write operations
    - Strict approval for dangerous operations

    Access Modes:
        READONLY: Only reads data, no side effects
                  Examples: read, ls, grep, search
        WRITE: Modifies files or state
               Examples: write, edit
        EXECUTE: Runs external commands or code
                 Examples: shell, sandbox
        NETWORK: Makes external network calls
                 Examples: web_search, web_fetch, api calls
        MIXED: Multiple access types in same operation
               Examples: git (read+write+network), docker (execute+network)
    """

    READONLY = "readonly"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    MIXED = "mixed"

    @property
    def requires_approval(self) -> bool:
        """Check if this access mode requires user approval by default."""
        return self in {AccessMode.WRITE, AccessMode.EXECUTE, AccessMode.MIXED}

    @property
    def is_safe(self) -> bool:
        """Check if this access mode is considered safe (no side effects)."""
        return self == AccessMode.READONLY

    def approval_level(self) -> int:
        """Get approval level (higher = stricter approval needed).

        Returns:
            0: Auto-approve (readonly)
            1: Log only (network)
            2: Confirm on first use (write)
            3: Always confirm (execute, mixed)
        """
        levels = {
            AccessMode.READONLY: 0,
            AccessMode.NETWORK: 1,
            AccessMode.WRITE: 2,
            AccessMode.EXECUTE: 3,
            AccessMode.MIXED: 3,
        }
        return levels[self]


class ExecutionCategory(Enum):
    """Execution category for parallel tool execution and dependency analysis.

    Used by parallel_executor to determine which tools can safely run concurrently.
    Replaces the static TOOL_CATEGORIES dictionary with decorator-driven metadata.

    Categories:
        READ_ONLY: Pure read operations, safe to parallelize (read, ls, grep, search)
        WRITE: File modification operations, may conflict (write, edit)
        COMPUTE: CPU-intensive but isolated (code_review, refactor analysis)
        NETWORK: External network calls, rate-limited (web_search, api calls)
        EXECUTE: Shell commands, may have side effects (shell, sandbox)
        MIXED: Multiple categories, needs careful dependency analysis (git, docker)
    """

    READ_ONLY = "read_only"
    WRITE = "write"
    COMPUTE = "compute"
    NETWORK = "network"
    EXECUTE = "execute"
    MIXED = "mixed"

    @property
    def can_parallelize(self) -> bool:
        """Check if this category is safe to run in parallel with itself."""
        return self in {
            ExecutionCategory.READ_ONLY,
            ExecutionCategory.COMPUTE,
            ExecutionCategory.NETWORK,
        }

    @property
    def conflicts_with(self) -> set:
        """Get categories that conflict with this one (shouldn't run together)."""
        conflict_map = {
            ExecutionCategory.READ_ONLY: set(),  # Safe with everything
            ExecutionCategory.WRITE: {ExecutionCategory.WRITE, ExecutionCategory.MIXED},
            ExecutionCategory.COMPUTE: set(),  # Safe with everything
            ExecutionCategory.NETWORK: set(),  # Safe with everything (rate limiting handled separately)
            ExecutionCategory.EXECUTE: {
                ExecutionCategory.EXECUTE,
                ExecutionCategory.WRITE,
                ExecutionCategory.MIXED,
            },
            ExecutionCategory.MIXED: {
                ExecutionCategory.WRITE,
                ExecutionCategory.EXECUTE,
                ExecutionCategory.MIXED,
            },
        }
        return conflict_map.get(self, set())


class DangerLevel(Enum):
    """Danger level for potentially destructive operations.

    Indicates the potential impact of tool misuse or errors.
    Used for warning messages and approval escalation.

    Danger Levels:
        SAFE: No risk of data loss or system damage
              Examples: read, ls, grep
        LOW: Minor risk, easily reversible
             Examples: write (single file), edit
        MEDIUM: Moderate risk, may require effort to reverse
                Examples: git push, db queries
        HIGH: Significant risk, difficult to reverse
              Examples: shell (arbitrary commands), db drop, docker rm
        CRITICAL: Irreversible or system-wide impact
                  Examples: rm -rf, format disk, drop database
    """

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def weight(self) -> int:
        """Numeric weight for danger comparison."""
        weights = {
            DangerLevel.SAFE: 0,
            DangerLevel.LOW: 1,
            DangerLevel.MEDIUM: 2,
            DangerLevel.HIGH: 3,
            DangerLevel.CRITICAL: 4,
        }
        return weights[self]

    @property
    def requires_confirmation(self) -> bool:
        """Check if this danger level requires explicit user confirmation."""
        return self.weight >= 2  # MEDIUM and above

    @property
    def warning_message(self) -> str:
        """Get appropriate warning message for this danger level."""
        if self == DangerLevel.SAFE:
            return ""
        elif self == DangerLevel.LOW:
            return "This operation modifies data."
        elif self == DangerLevel.MEDIUM:
            return f"{_get_icon('warning')} This operation may be difficult to reverse."
        elif self == DangerLevel.HIGH:
            return f"{_get_icon('warning')} WARNING: This is a potentially destructive operation."
        else:  # CRITICAL
            return f"{_get_icon('stop')} DANGER: This operation may cause irreversible changes!"

    def to_severity(self) -> "SeverityLevel":
        """Convert to unified SeverityLevel.

        Returns:
            Corresponding SeverityLevel from victor.core.severity.
        """
        from victor.core.severity import SeverityLevel

        return SeverityLevel.from_danger_level(self)


class SchemaLevel(str, Enum):
    """Schema verbosity level for LLM broadcasting.

    Controls how much detail is included in tool schemas sent to LLMs.
    Using tiered schemas reduces token usage while maintaining functionality.

    Schema Levels:
        FULL: Complete description + all parameter details
              Use for: Core tools, high-priority tools
              Token cost: ~100-150 tokens per tool
        COMPACT: Shortened description + all params with brief descriptions
              Use for: Vertical-specific tools
              Token cost: ~60-80 tokens per tool (~20% reduction from FULL)
        STUB: Name + one-line description + required params only
              Use for: Semantic pool tools, low-priority tools
              Token cost: ~25-40 tokens per tool

    Example:
        # FULL schema for core tool
        {"name": "read", "description": "Read text/code file. TRUNCATION: ...",
         "parameters": {"path": {...}, "offset": {...}, "limit": {...}}}

        # COMPACT schema for vertical tool
        {"name": "git", "description": "Git version control operations.",
         "parameters": {"operation": {...}, "args": {...}}}

        # STUB schema for semantic pool
        {"name": "jira", "description": "Create/query Jira issues.",
         "parameters": {"action": {...}}}
    """

    FULL = "full"
    COMPACT = "compact"
    STUB = "stub"

    @property
    def is_verbose(self) -> bool:
        """Check if this level includes full details."""
        return self == SchemaLevel.FULL

    @property
    def max_description_chars(self) -> int:
        """Maximum description length for this level."""
        if self == SchemaLevel.FULL:
            return 500
        elif self == SchemaLevel.COMPACT:
            return 150
        else:
            return 80

    @property
    def max_param_description_chars(self) -> int:
        """Maximum parameter description length for this level."""
        if self == SchemaLevel.FULL:
            return 100
        elif self == SchemaLevel.COMPACT:
            return 50
        else:
            return 25

    @property
    def include_optional_params(self) -> bool:
        """Whether to include optional parameters."""
        return self in {SchemaLevel.FULL, SchemaLevel.COMPACT}


class GraphMode(str, Enum):
    """Mode for graph analysis operations."""

    FIND = "find"
    SEARCH = "search"
    OVERVIEW = "overview"
    NEIGHBORS = "neighbors"
    PAGERANK = "pagerank"
    CENTRALITY = "centrality"
    PATH = "path"
    IMPACT = "impact"
    CLUSTERS = "clusters"
    STATS = "stats"
    SUBGRAPH = "subgraph"
    FILE_DEPS = "file_deps"
    PATTERNS = "patterns"
    MODULE_PAGERANK = "module_pagerank"
    MODULE_CENTRALITY = "module_centrality"
    CALL_FLOW = "call_flow"
    CALLERS = "callers"
    CALLEES = "callees"
    TRACE = "trace"
    SEMANTIC = "semantic"
    QUERY = "query"
    SCHEMA = "schema"
