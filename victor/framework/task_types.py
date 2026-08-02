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

"""Unified Task Type Registry for the Victor Framework.

This module consolidates the ~5 different TaskType enums and scattered
hint/budget definitions into a single, extensible registry.

The registry provides:
- Canonical task type definitions with categories
- Configurable tool budgets and iteration limits
- System prompt hints for each task type
- Priority tool lists per task type
- Vertical-specific overrides
- Alias resolution for backward compatibility
- **CANONICAL ALIAS REGISTRY**: Single source of truth for semantic term mapping

Usage:
    # Get the singleton registry
    registry = TaskTypeRegistry.get_instance()

    # Look up a task type
    definition = registry.get("edit")
    print(definition.hint)  # "[EDIT] Modify existing code..."
    print(definition.tool_budget)  # 25

    # Resolve alias to canonical name
    canonical = registry.resolve_alias("bugfix")  # → "edit"
    canonical = registry.resolve_alias("refactor")  # → "edit"

    # Vertical-specific lookup
    devops_edit = registry.get("edit", vertical="devops")

    # Register custom task type
    registry.register(TaskTypeDefinition(
        name="custom_task",
        category=TaskCategory.MODIFICATION,
        hint="[CUSTOM] Custom task hint",
        tool_budget=15,
    ))

    # Register vertical override
    registry.register_for_vertical("devops", TaskTypeDefinition(
        name="edit",
        category=TaskCategory.MODIFICATION,
        hint="[DEVOPS EDIT] Edit infrastructure files carefully",
        tool_budget=30,
    ))

---

## Canonical Alias Registry

**IMPORTANT**: This is the SINGLE SOURCE OF TRUTH for semantic term mapping across:
- Planning step types (victor/agent/planning/readable_schema.py)
- Completion/fulfillment TaskTypes (victor/framework/completion_scorer.py)
- Enhanced completion evaluation (victor/framework/enhanced_completion_evaluation.py)
- Task classification (victor/classification/*)

When adding new aliases, update ONLY this registry. Do not add local mappings elsewhere.

### Semantic Categories

| Canonical Type | Semantic Aliases (use these in prompts/LLM calls) |
|----------------|---------------------------------------------------|
| **edit** | modify, change, update, fix, bugfix, refactor, restructure, reorganize, patch, correct, improve, adjust, adapt |
| **create** | new, add, write, generate, implement, feature, scaffold, initialize, setup_file, make |
| **create_simple** | generate, generation, simple_create, quick_generate, write_directly |
| **search** | find, locate, grep, look, search_code, find_in_files, where_is, locate_file |
| **analyze** | analysis, review, audit, examine, inspect, investigation, assess, evaluate, check, code_review |
| **analysis_deep** | comprehensive, thorough, full_analysis, deep_dive, extensive, complete_review, detailed_analysis, in_depth |
| **design** | plan, architect, conceptual, architecture, planning, design_system, high_level, strategy |
| **action** | execute, run, do, perform, operation, task_execution, carry_out, accomplish |
| **research** | web_search, internet, investigate, lookup, search_web, find_online, query_web, explore_web |
| **general** | default, chat, help, ambiguous, unclear, misc, other, various |
| **simple** | quick, brief, small, easy, straightforward, trivial, minor |
| **refactor** | restructure, reorganize, cleanup, clean_up, improve_structure, reorganize_code |
| **debug** | troubleshoot, diagnose, fix_bug, error_fix, resolve_issue, problem_solve, debug_issue |
| **test** | unit_test, testing, verify, validation, check, ensure, confirm, test_code |
| **fact_check** | verify_claim, validate_fact, confirm_truth, check_accuracy |
| **literature_review** | survey, systematic_review, literature_survey, paper_review, academic_review |
| **competitive_analysis** | compare, comparison, competitor_analysis, market_analysis, versus |
| **trend_research** | identify_trends, trend_analysis, find_patterns, pattern_recognition, emerging |
| **technical_research** | deep_dive, technical_deep_dive, tech_investigation, explore_technical |

### Cross-System Mapping

For planning/fulfillment compatibility:
- **Planning RESEARCH** → maps to: **analyze**, **research**, **design**
- **Planning IMPLEMENTATION** → maps to: **edit**, **create**, **refactor**
- **Planning TESTING** → maps to: **test**
- **Planning REVIEW** → maps to: **analyze**, **analysis_deep**
- **Planning DEPLOYMENT** → maps to: **action** (deployment tasks)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set

from victor.framework.tool_naming import canonicalize_tool_list

if TYPE_CHECKING:
    from victor.agent.planning.base import StepType
    from victor.framework.completion_scorer import TaskType

logger = logging.getLogger(__name__)


class TaskCategory(Enum):
    """High-level task categories for grouping related task types.

    Categories help with:
    - Tool selection (analysis tasks favor read-only tools)
    - Safety rules (modification tasks need file edit permissions)
    - UI grouping (show related tasks together)
    """

    MODIFICATION = "modification"
    """Edit, Create, Refactor - tasks that change files."""

    ANALYSIS = "analysis"
    """Analyze, Search, Review - tasks that read/explore code."""

    EXECUTION = "execution"
    """Action, Test, Build - tasks that run commands/processes."""

    CONVERSATION = "conversation"
    """Chat, General, Help - conversational or ambiguous tasks."""


@dataclass
class TaskTypeDefinition:
    """Definition of a task type with all associated metadata.

    This is the single source of truth for task type configuration,
    consolidating hints, budgets, tool lists, and constraints.

    Attributes:
        name: Canonical name of the task type (lowercase, snake_case)
        category: High-level category for grouping
        hint: System prompt hint injected for this task type
        tool_budget: Maximum tool calls allowed for this task type
        priority_tools: Tools to prefer when selecting for this task
        max_iterations: Maximum exploration iterations
        aliases: Alternative names that map to this type
        vertical: Vertical this definition belongs to (None = core)
        needs_tools: Whether this task type needs tools at all
        force_action_after_read: For EDIT tasks, force action after reading target
        stage_tools: Tools allowed at each conversation stage
        force_action_hints: Hints to show when forcing action
        exploration_multiplier: Multiplier for exploration limits (used by modes)
    """

    name: str
    category: TaskCategory
    hint: str = ""
    tool_budget: int = 10
    priority_tools: List[str] = field(default_factory=list)
    max_iterations: int = 30
    aliases: Set[str] = field(default_factory=set)
    vertical: Optional[str] = None
    needs_tools: bool = True
    force_action_after_read: bool = False
    stage_tools: Dict[str, List[str]] = field(default_factory=dict)
    force_action_hints: Dict[str, str] = field(default_factory=dict)
    exploration_multiplier: float = 1.0

    def __post_init__(self):
        """Normalize task-type metadata to canonical runtime forms."""
        self.name = self.name.lower()
        # Ensure aliases are a set
        if isinstance(self.aliases, list):
            self.aliases = set(self.aliases)
        self.priority_tools = canonicalize_tool_list(self.priority_tools)
        self.stage_tools = {
            stage: canonicalize_tool_list(tools) for stage, tools in self.stage_tools.items()
        }


class TaskTypeRegistry:
    """Central registry for all task type definitions.

    The registry is a singleton that provides:
    - Registration of core task types
    - Registration of vertical-specific overrides
    - Alias resolution (e.g., "create" -> "generation")
    - Lookup with vertical fallback to core

    Thread Safety:
        The registry is NOT thread-safe. It should be populated at startup
        and then only read during normal operation.

    Extensibility:
        Verticals can register their own task types or override core ones:
        ```python
        registry = TaskTypeRegistry.get_instance()
        registry.register_for_vertical("devops", TaskTypeDefinition(
            name="infrastructure",
            category=TaskCategory.MODIFICATION,
            hint="[INFRA] Create infrastructure configs",
            tool_budget=25,
        ))
        ```
    """

    _instance: Optional["TaskTypeRegistry"] = None

    def __init__(self):
        """Initialize empty registry. Use get_instance() for singleton access."""
        self._core_types: Dict[str, TaskTypeDefinition] = {}
        self._vertical_overrides: Dict[str, Dict[str, TaskTypeDefinition]] = {}
        self._aliases: Dict[str, str] = {}
        self._registration_hooks: List[Callable[["TaskTypeRegistry"], None]] = []

    @classmethod
    def get_instance(cls) -> "TaskTypeRegistry":
        """Get the singleton registry instance.

        Creates and populates the registry on first access.

        Returns:
            The global TaskTypeRegistry instance with defaults registered.
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_defaults()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def register(self, definition: TaskTypeDefinition) -> None:
        """Register a core task type definition.

        Args:
            definition: The task type definition to register

        Raises:
            ValueError: If a task type with the same name already exists
        """
        name = definition.name.lower()

        if name in self._core_types:
            logger.warning(f"TaskTypeRegistry: Overwriting core type '{name}'")

        self._core_types[name] = definition

        # Register aliases
        for alias in definition.aliases:
            alias_lower = alias.lower()
            self._aliases[alias_lower] = name
            logger.debug(f"TaskTypeRegistry: Alias '{alias_lower}' -> '{name}'")

        logger.debug(
            f"TaskTypeRegistry: Registered core type '{name}' "
            f"(category={definition.category.value}, budget={definition.tool_budget})"
        )

    def register_for_vertical(self, vertical: str, definition: TaskTypeDefinition) -> None:
        """Register a task type definition for a specific vertical.

        Vertical-specific definitions override core definitions when
        looking up with that vertical specified.

        Args:
            vertical: The vertical name (e.g., "devops", "data_analysis")
            definition: The task type definition to register
        """
        vertical = vertical.lower()
        name = definition.name.lower()
        definition.vertical = vertical

        if vertical not in self._vertical_overrides:
            self._vertical_overrides[vertical] = {}

        self._vertical_overrides[vertical][name] = definition

        logger.debug(
            f"TaskTypeRegistry: Registered '{name}' for vertical '{vertical}' "
            f"(budget={definition.tool_budget})"
        )

    def add_registration_hook(self, hook: Callable[["TaskTypeRegistry"], None]) -> None:
        """Add a hook to be called when the registry is populated.

        Hooks are called during _register_defaults after core types are registered.
        Verticals can use this to register their task types.

        Args:
            hook: Callable that receives the registry instance
        """
        self._registration_hooks.append(hook)

    def get(self, task_type: str, vertical: Optional[str] = None) -> Optional[TaskTypeDefinition]:
        """Look up a task type definition.

        Resolution order:
        1. If vertical specified, check vertical-specific definitions
        2. Resolve aliases to canonical names
        3. Look up in core types

        Args:
            task_type: Name or alias of the task type
            vertical: Optional vertical for specific override lookup

        Returns:
            TaskTypeDefinition if found, None otherwise
        """
        task_type = task_type.lower()

        # Resolve alias to canonical name
        canonical = self._aliases.get(task_type, task_type)

        # Check vertical-specific first
        if vertical:
            vertical = vertical.lower()
            overrides = self._vertical_overrides.get(vertical)
            if overrides:
                # Honor the exact registered name as well as the canonical form.
                # Vertical overrides registered under an alias (e.g. "refactor",
                # an alias of "edit") would otherwise be invisible because the
                # lookup canonicalizes "refactor" -> "edit" before matching.
                if task_type in overrides:
                    return overrides[task_type]
                if canonical in overrides:
                    return overrides[canonical]

        # Fall back to core types
        return self._core_types.get(canonical)

    def get_hint(self, task_type: str, vertical: Optional[str] = None) -> str:
        """Get the system prompt hint for a task type.

        Args:
            task_type: Name or alias of the task type
            vertical: Optional vertical for specific override

        Returns:
            Hint string, or empty string if task type not found
        """
        definition = self.get(task_type, vertical)
        return definition.hint if definition else ""

    def get_tool_budget(self, task_type: str, vertical: Optional[str] = None) -> int:
        """Get the tool budget for a task type.

        Args:
            task_type: Name or alias of the task type
            vertical: Optional vertical for specific override

        Returns:
            Tool budget, or 10 (default) if task type not found
        """
        definition = self.get(task_type, vertical)
        return definition.tool_budget if definition else 10

    def get_max_iterations(self, task_type: str, vertical: Optional[str] = None) -> int:
        """Get the max iterations for a task type.

        Args:
            task_type: Name or alias of the task type
            vertical: Optional vertical for specific override

        Returns:
            Max iterations, or 30 (default) if task type not found
        """
        definition = self.get(task_type, vertical)
        return definition.max_iterations if definition else 30

    def get_priority_tools(self, task_type: str, vertical: Optional[str] = None) -> List[str]:
        """Get priority tools for a task type.

        Args:
            task_type: Name or alias of the task type
            vertical: Optional vertical for specific override

        Returns:
            List of tool names to prioritize, or empty list
        """
        definition = self.get(task_type, vertical)
        return definition.priority_tools if definition else []

    def get_category(
        self, task_type: str, vertical: Optional[str] = None
    ) -> Optional[TaskCategory]:
        """Get the category for a task type.

        Args:
            task_type: Name or alias of the task type
            vertical: Optional vertical for specific override

        Returns:
            TaskCategory or None if not found
        """
        definition = self.get(task_type, vertical)
        return definition.category if definition else None

    def list_types(self, vertical: Optional[str] = None) -> List[str]:
        """List all registered task type names.

        Args:
            vertical: If specified, includes vertical-specific types

        Returns:
            Sorted list of task type names
        """
        names = set(self._core_types.keys())
        if vertical and vertical.lower() in self._vertical_overrides:
            names.update(self._vertical_overrides[vertical.lower()].keys())
        return sorted(names)

    def list_verticals(self) -> List[str]:
        """List all registered verticals.

        Returns:
            Sorted list of vertical names
        """
        return sorted(self._vertical_overrides.keys())

    def resolve_alias(self, name: str) -> str:
        """Resolve an alias to its canonical name.

        Args:
            name: Task type name or alias

        Returns:
            Canonical name if alias exists, otherwise the original name
        """
        return self._aliases.get(name.lower(), name.lower())

    def to_completion_task_type(
        self, task_type: str, vertical: Optional[str] = None
    ) -> "Optional[TaskType]":
        """Convert a task type alias to completion TaskType enum.

        This is the SINGLE SOURCE OF TRUTH for mapping task type strings to
        completion/fulfillment TaskType enums. Do not duplicate this mapping elsewhere.

        Args:
            task_type: Task type name or alias
            vertical: Optional vertical for specific override

        Returns:
            TaskType enum from victor.framework.completion_scorer, or None if not found

        Example:
            >>> registry = TaskTypeRegistry.get_instance()
            >>> registry.to_completion_task_type("bugfix")
            <TaskType.CODE_MODIFICATION: 'code_modification'>
            >>> registry.to_completion_task_type("refactor")
            <TaskType.CODE_MODIFICATION: 'code_modification'>
        """
        from victor.framework.completion_scorer import TaskType as CompletionTaskType

        # Resolve alias to canonical name
        canonical_name = self.resolve_alias(task_type)

        # Map canonical names to completion TaskType enum
        completion_map = {
            "edit": CompletionTaskType.CODE_MODIFICATION,
            "create": CompletionTaskType.CODE_GENERATION,
            "create_simple": CompletionTaskType.CODE_GENERATION,
            "search": CompletionTaskType.SEARCH,
            "analyze": CompletionTaskType.ANALYSIS,
            "analysis_deep": CompletionTaskType.ANALYSIS,
            "design": CompletionTaskType.ANALYSIS,
            "research": CompletionTaskType.ANALYSIS,
            "action": CompletionTaskType.DEPLOYMENT,
            "debug": CompletionTaskType.DEBUGGING,
            "test": CompletionTaskType.TESTING,
            "general": CompletionTaskType.UNKNOWN,
            "simple": CompletionTaskType.UNKNOWN,
            "refactor": CompletionTaskType.CODE_MODIFICATION,
            "setup": CompletionTaskType.SETUP,
            "documentation": CompletionTaskType.DOCUMENTATION,
            "deployment": CompletionTaskType.DEPLOYMENT,
        }

        return completion_map.get(canonical_name)

    def to_planning_step_type(
        self, task_type: str, vertical: Optional[str] = None
    ) -> "Optional[StepType]":
        """Convert a task type alias to planning StepType enum.

        This is the SINGLE SOURCE OF TRUTH for mapping task type strings to
        planning StepType enums. Do not duplicate this mapping elsewhere.

        Args:
            task_type: Task type name or alias
            vertical: Optional vertical for specific override

        Returns:
            StepType enum from victor.agent.planning.base, or None if not found

        Example:
            >>> registry = TaskTypeRegistry.get_instance()
            >>> registry.to_planning_step_type("bugfix")
            <StepType.IMPLEMENTATION: 'implementation'>
            >>> registry.to_planning_step_type("refactor")
            <StepType.IMPLEMENTATION: 'implementation'>
        """
        from victor.agent.planning.base import StepType

        # The ReadableTaskPlan vocabulary maps directly to planning step types and
        # takes precedence over alias resolution, which is tuned for tool-budget
        # categories and would otherwise collapse distinct planning steps (e.g.
        # "review" -> "analyze" -> RESEARCH, or drop "planning"/"deploy"/"doc").
        direct_step_map = {
            "research": StepType.RESEARCH,
            "planning": StepType.PLANNING,
            "feature": StepType.IMPLEMENTATION,
            "implementation": StepType.IMPLEMENTATION,
            "bugfix": StepType.IMPLEMENTATION,
            "refactor": StepType.IMPLEMENTATION,
            "test": StepType.TESTING,
            "testing": StepType.TESTING,
            "review": StepType.REVIEW,
            "deploy": StepType.DEPLOYMENT,
            "deployment": StepType.DEPLOYMENT,
            "analyze": StepType.RESEARCH,
            "analysis": StepType.RESEARCH,
            "doc": StepType.RESEARCH,
            "documentation": StepType.RESEARCH,
        }
        direct = direct_step_map.get(task_type.lower().strip())
        if direct is not None:
            return direct

        # Resolve alias to canonical name
        canonical_name = self.resolve_alias(task_type)

        # Map canonical names to planning StepType enum
        planning_map = {
            "edit": StepType.IMPLEMENTATION,
            "create": StepType.IMPLEMENTATION,
            "create_simple": StepType.IMPLEMENTATION,
            "search": StepType.RESEARCH,
            "analyze": StepType.RESEARCH,
            "analysis_deep": StepType.REVIEW,
            "design": StepType.PLANNING,
            "research": StepType.RESEARCH,
            "action": StepType.DEPLOYMENT,
            "debug": StepType.IMPLEMENTATION,
            "test": StepType.TESTING,
            "general": StepType.IMPLEMENTATION,
            "simple": StepType.IMPLEMENTATION,
            "refactor": StepType.IMPLEMENTATION,
            "documentation": StepType.RESEARCH,
            "deployment": StepType.DEPLOYMENT,
        }

        return planning_map.get(canonical_name)

    def _register_defaults(self) -> None:
        """Register all default task types.

        This is called automatically on first access to get_instance().
        Consolidates definitions from:
        - complexity_classifier.py: PROMPT_HINTS, DEFAULT_BUDGETS
        - unified_task_tracker.py: UnifiedTaskConfigLoader.DEFAULT_CONFIG
        - embeddings/task_classifier.py: TaskType enum
        """
        # =================================================================
        # Core Task Types
        # =================================================================

        # EDIT - Modify existing files
        self.register(
            TaskTypeDefinition(
                name="edit",
                category=TaskCategory.MODIFICATION,
                hint="[EDIT] Modify existing code. Read target files first, then make focused changes.",
                tool_budget=25,
                max_iterations=10,
                priority_tools=["read", "edit", "code_search"],
                aliases={
                    # Core semantic aliases
                    "modify",
                    "change",
                    "update",
                    "fix",
                    # Bug-related aliases
                    "bugfix",
                    "patch",
                    "correct",
                    "fix_bug",
                    # Structural change aliases
                    "refactor",
                    "restructure",
                    "reorganize",
                    # Improvement aliases
                    "improve",
                    "adjust",
                    "adapt",
                },
                force_action_after_read=True,
                stage_tools={
                    "initial": ["ls", "code_search"],
                    "reading": ["read", "code_search"],
                    "executing": ["edit", "read"],
                    "verifying": ["read", "run_tests"],
                },
                force_action_hints={
                    "after_target_read": "Use edit to make the change.",
                    "max_iterations": "Please make the change or explain blockers.",
                },
            )
        )

        # CREATE - Create new files with context
        self.register(
            TaskTypeDefinition(
                name="create",
                category=TaskCategory.MODIFICATION,
                hint="[CREATE] Create new files. Explore context first, then write files.",
                tool_budget=25,
                max_iterations=10,
                priority_tools=["write", "read", "ls"],
                aliases={
                    # Core semantic aliases
                    "new",
                    "add",
                    "write",
                    # Generation aliases
                    "generate",
                    "implement",
                    # Feature aliases
                    "feature",
                    # Setup aliases
                    "scaffold",
                    "initialize",
                    "setup_file",
                    # Creation aliases
                    "make",
                    "create_new",
                },
                stage_tools={
                    "initial": ["ls", "read"],
                    "reading": ["read"],
                    "executing": ["write", "edit"],
                    "verifying": ["read", "run_tests"],
                },
                force_action_hints={
                    "max_iterations": "Please create the file.",
                },
            )
        )

        # CREATE_SIMPLE - Generate code directly (no exploration)
        self.register(
            TaskTypeDefinition(
                name="create_simple",
                category=TaskCategory.MODIFICATION,
                hint="[GENERATE] Write code directly. Minimal exploration. Display or save as requested.",
                tool_budget=5,
                max_iterations=3,
                priority_tools=["write"],
                aliases={"generate", "generation", "simple_create"},
                stage_tools={
                    "initial": ["write"],
                    "executing": ["write"],
                    "verifying": ["read"],
                },
                force_action_hints={
                    "immediate": "Create the code directly using write.",
                },
            )
        )

        # SEARCH - Find/locate code or files
        self.register(
            TaskTypeDefinition(
                name="search",
                category=TaskCategory.ANALYSIS,
                hint="[SEARCH] Find code or files. Use search tools, then read relevant results.",
                tool_budget=25,
                max_iterations=10,
                priority_tools=[
                    "code_search",
                    "semantic_code_search",
                    "read",
                    "ls",
                ],
                aliases={
                    # Core semantic aliases
                    "find",
                    "locate",
                    "grep",
                    "look",
                    # Extended search aliases
                    "search_code",
                    "find_in_files",
                    "where_is",
                    "locate_file",
                    "find_file",
                    "search_for",
                },
                stage_tools={
                    "initial": ["ls", "code_search"],
                    "reading": ["read", "code_search"],
                    "executing": ["read"],
                    "verifying": ["read"],
                },
                force_action_hints={
                    "max_iterations": "Please summarize your findings.",
                },
            )
        )

        # ANALYZE - Count, measure, analyze code
        self.register(
            TaskTypeDefinition(
                name="analyze",
                category=TaskCategory.ANALYSIS,
                hint="[ANALYZE] Analyze code metrics. Read files and run analysis commands.",
                tool_budget=40,
                max_iterations=20,
                priority_tools=["read", "code_search", "shell"],
                aliases={
                    # Core semantic aliases
                    "analysis",
                    "review",
                    "audit",
                    "examine",
                    # Inspection aliases
                    "inspect",
                    "investigation",
                    "assess",
                    "evaluate",
                    # Check aliases
                    "check",
                    "code_review",
                    "validate",
                    # Understanding aliases
                    "understand",
                    "investigate",
                    "explore",
                },
                stage_tools={
                    "initial": ["ls", "code_search"],
                    "reading": ["read", "code_search", "shell"],
                    "executing": ["shell"],
                    "verifying": ["read"],
                },
                force_action_hints={
                    "max_iterations": "Please summarize your analysis.",
                },
            )
        )

        # ANALYSIS_DEEP - Comprehensive codebase analysis
        self.register(
            TaskTypeDefinition(
                name="analysis_deep",
                category=TaskCategory.ANALYSIS,
                hint="[DEEP ANALYSIS] Thorough exploration required. Examine all relevant modules. Comprehensive output.",
                tool_budget=60,
                max_iterations=50,
                priority_tools=[
                    "read",
                    "code_search",
                    "semantic_code_search",
                    "ls",
                ],
                aliases={
                    # Core semantic aliases
                    "comprehensive",
                    "thorough",
                    "full_analysis",
                    # Extended deep analysis aliases
                    "deep_dive",
                    "extensive",
                    "complete_review",
                    "detailed_analysis",
                    "in_depth",
                    "deep_analysis",
                    "exhaustive",
                    "complete",
                    "full_review",
                },
                exploration_multiplier=1.5,
            )
        )

        # DESIGN - Conceptual/planning tasks
        self.register(
            TaskTypeDefinition(
                name="design",
                category=TaskCategory.ANALYSIS,
                hint="[DESIGN] Architecture/design questions. Explore codebase to understand patterns and structure.",
                tool_budget=40,
                max_iterations=20,
                priority_tools=["read", "ls", "code_search"],
                aliases={
                    # Core semantic aliases
                    "plan",
                    "architect",
                    "conceptual",
                    # Architecture aliases
                    "architecture",
                    "design_system",
                    "high_level",
                    # Strategy aliases
                    "strategy",
                    "strategic",
                    "architectural",
                },
                needs_tools=True,  # Design tasks benefit from codebase exploration
                stage_tools={
                    "initial": ["ls", "code_search", "read"],
                    "reading": ["read", "code_search", "ls"],
                    "verifying": ["read"],
                },
                force_action_hints={
                    "max_iterations": "Please summarize the architecture and provide your recommendations.",
                },
            )
        )

        # ACTION - Execute, run, deploy
        self.register(
            TaskTypeDefinition(
                name="action",
                category=TaskCategory.EXECUTION,
                hint="[ACTION] Execute task. Multiple tool calls allowed. Continue until complete.",
                tool_budget=50,
                max_iterations=12,
                priority_tools=["shell", "git_commit", "run_tests"],
                aliases={"execute", "run", "do"},
            )
        )

        # RESEARCH - Web research tasks
        self.register(
            TaskTypeDefinition(
                name="research",
                category=TaskCategory.ANALYSIS,
                hint="[RESEARCH] Web research task. Search and fetch information from the web. Use 4-phase approach: discover → search → analyze → synthesize.",
                tool_budget=45,  # Increased from 20 - allow comprehensive research
                max_iterations=25,  # Increased from 10 - allow multi-phase analysis
                priority_tools=["web_search", "web_fetch", "read", "grep"],
                aliases={
                    # Core semantic aliases
                    "web_search",
                    "internet",
                    # Investigation aliases
                    "investigate",
                    "lookup",
                    "search_web",
                    "find_online",
                    # Query aliases
                    "query_web",
                    "explore_web",
                    "search_online",
                    # Information gathering
                    "gather_info",
                    "find_information",
                    "look_up",
                },
                stage_tools={
                    "initial": ["web_search"],  # Phase 1: Discover
                    "reading": ["web_fetch", "read"],  # Phase 2: Gather
                    "analysis": ["read", "grep"],  # Phase 3: Analyze
                    "executing": ["write"],  # Phase 4: Synthesize
                },
                force_action_hints={
                    "max_iterations": "Please summarize your research findings with specific paper IDs and key insights.",
                },
            )
        )

        # GENERAL - Ambiguous or conversational
        self.register(
            TaskTypeDefinition(
                name="general",
                category=TaskCategory.CONVERSATION,
                hint="[GENERAL] General task. Explore as needed and complete the task.",
                tool_budget=35,
                max_iterations=15,
                priority_tools=["read", "ls", "code_search"],
                aliases={"default", "chat", "help", "ambiguous"},
                stage_tools={
                    "initial": ["ls", "code_search", "read"],
                    "reading": ["read", "code_search"],
                    "executing": ["edit", "write", "shell"],
                    "verifying": ["read", "run_tests"],
                },
                force_action_hints={
                    "max_iterations": "Please complete the task or explain blockers.",
                },
            )
        )

        # =================================================================
        # Simple Query Types (low budget)
        # =================================================================

        self.register(
            TaskTypeDefinition(
                name="simple",
                category=TaskCategory.CONVERSATION,
                hint="[SIMPLE] Simple query. 1-2 tool calls max. Answer immediately after.",
                tool_budget=2,
                max_iterations=3,
                priority_tools=["ls", "read"],
                aliases={"quick", "brief"},
            )
        )

        # =================================================================
        # Extended Task Types (from embeddings/task_classifier.py)
        # =================================================================

        # Coding vertical granular types
        self.register(
            TaskTypeDefinition(
                name="refactor",
                category=TaskCategory.MODIFICATION,
                hint="[REFACTOR] Restructure existing code. Read carefully, then make incremental changes.",
                tool_budget=30,
                max_iterations=15,
                priority_tools=["read", "edit", "code_search"],
                aliases={"restructure", "reorganize"},
            )
        )

        self.register(
            TaskTypeDefinition(
                name="debug",
                category=TaskCategory.ANALYSIS,
                hint="[DEBUG] Find and fix bugs. Read code, trace issues, then fix.",
                tool_budget=25,
                max_iterations=12,
                priority_tools=["read", "code_search", "shell"],
                aliases={"troubleshoot", "diagnose"},
            )
        )

        self.register(
            TaskTypeDefinition(
                name="test",
                category=TaskCategory.EXECUTION,
                hint="[TEST] Write or run tests. Create test files or execute test commands.",
                tool_budget=25,
                max_iterations=10,
                priority_tools=["write", "shell", "read"],
                aliases={"unit_test", "testing"},
            )
        )

        # =================================================================
        # Research vertical types
        # =================================================================

        self.register(
            TaskTypeDefinition(
                name="fact_check",
                category=TaskCategory.ANALYSIS,
                hint="[FACT CHECK] Verify claims with sources. Search for authoritative information.",
                tool_budget=20,
                max_iterations=10,
                priority_tools=["web_search", "web_fetch"],
            )
        )

        self.register(
            TaskTypeDefinition(
                name="literature_review",
                category=TaskCategory.ANALYSIS,
                hint="[LITERATURE REVIEW] Systematic review of knowledge. Search broadly and synthesize.",
                tool_budget=35,
                max_iterations=20,
                priority_tools=["web_search", "web_fetch"],
            )
        )

        self.register(
            TaskTypeDefinition(
                name="competitive_analysis",
                category=TaskCategory.ANALYSIS,
                hint="[COMPETITIVE ANALYSIS] Compare products/services. Research and compare features.",
                tool_budget=35,
                max_iterations=20,
                priority_tools=["web_search", "web_fetch"],
            )
        )

        self.register(
            TaskTypeDefinition(
                name="trend_research",
                category=TaskCategory.ANALYSIS,
                hint="[TREND RESEARCH] Identify patterns and emerging developments.",
                tool_budget=35,
                max_iterations=20,
                priority_tools=["web_search", "web_fetch"],
            )
        )

        self.register(
            TaskTypeDefinition(
                name="technical_research",
                category=TaskCategory.ANALYSIS,
                hint="[TECHNICAL RESEARCH] Deep dive into technical topics.",
                tool_budget=35,
                max_iterations=20,
                priority_tools=["web_search", "web_fetch", "read"],
            )
        )

        # =================================================================
        # Call registration hooks for verticals
        # =================================================================

        for hook in self._registration_hooks:
            try:
                hook(self)
            except Exception as e:
                logger.warning(f"TaskTypeRegistry: Hook failed: {e}")

        logger.debug(f"TaskTypeRegistry: Initialized with {len(self._core_types)} core types")


# =================================================================
# Convenience Functions
# =================================================================


def get_task_type_registry() -> TaskTypeRegistry:
    """Get the global TaskTypeRegistry instance.

    Returns:
        The singleton TaskTypeRegistry instance
    """
    return TaskTypeRegistry.get_instance()


def canonicalize_task_type(value: str, overrides: Optional[Dict[str, str]] = None) -> str:
    """Resolve any task-type token to its canonical registered name.

    The single mapping authority used by the task-type *adapter* enums
    (``TrackerTaskType`` / ``ClassifierTaskType`` / ``pattern_registry.TaskType``)
    so they all collapse onto this registry instead of each carrying their own
    parallel taxonomy. ``overrides`` covers adapter-local tokens that have no alias
    in the default registry (e.g. coding-specific ``bug_fix`` -> ``debug``).

    Args:
        value: A task-type name/alias from any adapter enum.
        overrides: Optional adapter-local pre-map applied before alias resolution.

    Returns:
        The canonical task-type name (an alias resolves to its target; an already-
        canonical name is returned unchanged).
    """
    if overrides and value in overrides:
        value = overrides[value]
    return get_task_type_registry().resolve_alias(value)


def get_task_hint(task_type: str, vertical: Optional[str] = None) -> str:
    """Get the system prompt hint for a task type.

    Args:
        task_type: Name or alias of the task type
        vertical: Optional vertical for specific override

    Returns:
        Hint string for the task type
    """
    return TaskTypeRegistry.get_instance().get_hint(task_type, vertical)


def get_task_budget(task_type: str, vertical: Optional[str] = None) -> int:
    """Get the tool budget for a task type.

    Args:
        task_type: Name or alias of the task type
        vertical: Optional vertical for specific override

    Returns:
        Tool budget for the task type
    """
    return TaskTypeRegistry.get_instance().get_tool_budget(task_type, vertical)


def register_vertical_task_type(
    vertical: str,
    name: str,
    category: TaskCategory,
    hint: str,
    tool_budget: int = 10,
    **kwargs,
) -> None:
    """Register a task type for a specific vertical.

    Convenience function for verticals to register their task types.

    Args:
        vertical: Vertical name (e.g., "devops", "data_analysis")
        name: Task type name
        category: Task category
        hint: System prompt hint
        tool_budget: Maximum tool calls
        **kwargs: Additional TaskTypeDefinition fields
    """
    definition = TaskTypeDefinition(
        name=name,
        category=category,
        hint=hint,
        tool_budget=tool_budget,
        **kwargs,
    )
    TaskTypeRegistry.get_instance().register_for_vertical(vertical, definition)


# =================================================================
# Vertical Registration Hooks
# =================================================================


def register_devops_task_types(registry: TaskTypeRegistry) -> None:
    """Register DevOps-specific task types.

    This function is called as a registration hook to add DevOps vertical
    task types and overrides. Can be used directly or via add_registration_hook.

    Args:
        registry: The TaskTypeRegistry instance
    """
    # Infrastructure task
    registry.register_for_vertical(
        "devops",
        TaskTypeDefinition(
            name="infrastructure",
            category=TaskCategory.MODIFICATION,
            hint="[INFRA] Create or modify infrastructure configs. Review existing setup first.",
            tool_budget=30,
            max_iterations=15,
            priority_tools=[
                "read",
                "write",
                "shell",
                "ls",
            ],
        ),
    )

    # CI/CD task
    registry.register_for_vertical(
        "devops",
        TaskTypeDefinition(
            name="ci_cd",
            category=TaskCategory.MODIFICATION,
            hint="[CI/CD] Configure CI/CD pipelines. Understand workflow structure first.",
            tool_budget=25,
            max_iterations=12,
            priority_tools=["read", "write", "code_search"],
            aliases={"pipeline", "workflow", "github_actions"},
        ),
    )

    # Kubernetes task
    registry.register_for_vertical(
        "devops",
        TaskTypeDefinition(
            name="kubernetes",
            category=TaskCategory.MODIFICATION,
            hint="[K8S] Create or modify Kubernetes manifests. Check existing resources.",
            tool_budget=30,
            max_iterations=15,
            priority_tools=["read", "write", "shell"],
            aliases={"k8s", "helm"},
        ),
    )

    # Terraform task
    registry.register_for_vertical(
        "devops",
        TaskTypeDefinition(
            name="terraform",
            category=TaskCategory.MODIFICATION,
            hint="[TERRAFORM] Create or modify Terraform configs. Plan before applying.",
            tool_budget=35,
            max_iterations=18,
            priority_tools=["read", "write", "shell", "code_search"],
            aliases={"tf", "iac"},
        ),
    )

    # Docker task
    registry.register_for_vertical(
        "devops",
        TaskTypeDefinition(
            name="dockerfile",
            category=TaskCategory.MODIFICATION,
            hint="[DOCKER] Create or modify Dockerfiles. Follow best practices.",
            tool_budget=20,
            max_iterations=10,
            priority_tools=["read", "write"],
            aliases={"docker"},
        ),
    )

    # Docker Compose task
    registry.register_for_vertical(
        "devops",
        TaskTypeDefinition(
            name="docker_compose",
            category=TaskCategory.MODIFICATION,
            hint="[COMPOSE] Create or modify docker-compose files. Check service dependencies.",
            tool_budget=25,
            max_iterations=12,
            priority_tools=["read", "write", "code_search"],
            aliases={"compose"},
        ),
    )

    # Monitoring task
    registry.register_for_vertical(
        "devops",
        TaskTypeDefinition(
            name="monitoring",
            category=TaskCategory.MODIFICATION,
            hint="[MONITORING] Configure monitoring and alerting. Review current setup first.",
            tool_budget=30,
            max_iterations=15,
            priority_tools=["read", "write", "code_search"],
            aliases={"observability", "alerts"},
        ),
    )

    # Override edit for DevOps (more cautious with infrastructure)
    registry.register_for_vertical(
        "devops",
        TaskTypeDefinition(
            name="edit",
            category=TaskCategory.MODIFICATION,
            hint="[DEVOPS EDIT] Edit infrastructure files carefully. Verify syntax and dependencies.",
            tool_budget=30,
            max_iterations=12,
            priority_tools=["read", "edit", "shell"],
            force_action_after_read=True,
        ),
    )

    logger.debug("TaskTypeRegistry: Registered DevOps vertical task types")


def register_data_analysis_task_types(registry: TaskTypeRegistry) -> None:
    """Register Data Analysis-specific task types.

    Args:
        registry: The TaskTypeRegistry instance
    """
    # Data profiling task
    registry.register_for_vertical(
        "data_analysis",
        TaskTypeDefinition(
            name="data_profiling",
            category=TaskCategory.ANALYSIS,
            hint="[PROFILE] Profile dataset. Examine structure, types, distributions.",
            tool_budget=30,
            max_iterations=15,
            priority_tools=["read", "shell", "notebook_edit"],
        ),
    )

    # Statistical analysis task
    registry.register_for_vertical(
        "data_analysis",
        TaskTypeDefinition(
            name="statistical_analysis",
            category=TaskCategory.ANALYSIS,
            hint="[STATS] Perform statistical analysis. Calculate metrics and significance.",
            tool_budget=40,
            max_iterations=20,
            priority_tools=["read", "shell", "notebook_edit"],
            aliases={"statistics"},
        ),
    )

    # Correlation analysis task
    registry.register_for_vertical(
        "data_analysis",
        TaskTypeDefinition(
            name="correlation_analysis",
            category=TaskCategory.ANALYSIS,
            hint="[CORRELATION] Analyze relationships between variables.",
            tool_budget=30,
            max_iterations=15,
            priority_tools=["read", "shell", "notebook_edit"],
        ),
    )

    # Regression task
    registry.register_for_vertical(
        "data_analysis",
        TaskTypeDefinition(
            name="regression",
            category=TaskCategory.ANALYSIS,
            hint="[REGRESSION] Build regression models. Fit and evaluate.",
            tool_budget=40,
            max_iterations=20,
            priority_tools=["read", "shell", "notebook_edit", "write"],
        ),
    )

    # Clustering task
    registry.register_for_vertical(
        "data_analysis",
        TaskTypeDefinition(
            name="clustering",
            category=TaskCategory.ANALYSIS,
            hint="[CLUSTERING] Cluster data points. Find patterns and groupings.",
            tool_budget=40,
            max_iterations=20,
            priority_tools=["read", "shell", "notebook_edit"],
        ),
    )

    # Time series task
    registry.register_for_vertical(
        "data_analysis",
        TaskTypeDefinition(
            name="time_series",
            category=TaskCategory.ANALYSIS,
            hint="[TIME SERIES] Analyze temporal patterns. Decompose and forecast.",
            tool_budget=45,
            max_iterations=25,
            priority_tools=["read", "shell", "notebook_edit"],
        ),
    )

    # Visualization task
    registry.register_for_vertical(
        "data_analysis",
        TaskTypeDefinition(
            name="visualization",
            category=TaskCategory.MODIFICATION,
            hint="[VIZ] Create data visualizations. Choose appropriate chart types.",
            tool_budget=25,
            max_iterations=12,
            priority_tools=["read", "shell", "notebook_edit", "write"],
            aliases={"chart", "plot", "graph"},
        ),
    )

    # Override analyze for data analysis (higher budget)
    registry.register_for_vertical(
        "data_analysis",
        TaskTypeDefinition(
            name="analyze",
            category=TaskCategory.ANALYSIS,
            hint="[DATA ANALYZE] Analyze data comprehensively. Profile, explore, and summarize.",
            tool_budget=50,
            max_iterations=25,
            priority_tools=["read", "shell", "notebook_edit"],
        ),
    )

    logger.debug("TaskTypeRegistry: Registered Data Analysis vertical task types")


def register_coding_task_types(registry: TaskTypeRegistry) -> None:
    """Register Coding-specific task types.

    Args:
        registry: The TaskTypeRegistry instance
    """
    # Code generation task
    registry.register_for_vertical(
        "coding",
        TaskTypeDefinition(
            name="code_generation",
            category=TaskCategory.MODIFICATION,
            hint="[CODEGEN] Generate code from specification. Follow project patterns.",
            tool_budget=20,
            max_iterations=10,
            priority_tools=["read", "write", "code_search"],
        ),
    )

    # Override refactor for coding (more thorough)
    registry.register_for_vertical(
        "coding",
        TaskTypeDefinition(
            name="refactor",
            category=TaskCategory.MODIFICATION,
            hint="[REFACTOR] Refactor code systematically. Preserve behavior, improve structure.",
            tool_budget=35,
            max_iterations=20,
            priority_tools=["read", "edit", "code_search", "run_tests"],
        ),
    )

    # Override debug for coding
    registry.register_for_vertical(
        "coding",
        TaskTypeDefinition(
            name="debug",
            category=TaskCategory.ANALYSIS,
            hint="[DEBUG] Debug code issues. Read stack traces, add logging, trace execution.",
            tool_budget=30,
            max_iterations=15,
            priority_tools=["read", "code_search", "shell", "edit"],
        ),
    )

    # Override test for coding
    registry.register_for_vertical(
        "coding",
        TaskTypeDefinition(
            name="test",
            category=TaskCategory.EXECUTION,
            hint="[TEST] Write and run tests. Ensure coverage and edge cases.",
            tool_budget=30,
            max_iterations=15,
            priority_tools=["write", "shell", "read", "code_search"],
        ),
    )

    logger.debug("TaskTypeRegistry: Registered Coding vertical task types")


def register_research_task_types(registry: TaskTypeRegistry) -> None:
    """Register Research-specific task types.

    Args:
        registry: The TaskTypeRegistry instance
    """
    # General research query
    registry.register_for_vertical(
        "research",
        TaskTypeDefinition(
            name="general_query",
            category=TaskCategory.ANALYSIS,
            hint="[QUERY] Answer research question. Search and synthesize information.",
            tool_budget=25,
            max_iterations=12,
            priority_tools=["web_search", "web_fetch"],
        ),
    )

    # Override research for research vertical (higher budget)
    registry.register_for_vertical(
        "research",
        TaskTypeDefinition(
            name="research",
            category=TaskCategory.ANALYSIS,
            hint="[RESEARCH] Comprehensive web research. Search deeply, verify sources.",
            tool_budget=40,
            max_iterations=20,
            priority_tools=["web_search", "web_fetch"],
        ),
    )

    logger.debug("TaskTypeRegistry: Registered Research vertical task types")


def setup_vertical_task_types() -> None:
    """Set up all vertical task types.

    This function should be called during application startup to register
    all vertical-specific task types. It's safe to call multiple times.

    Usage:
        from victor.framework.task_types import setup_vertical_task_types
        setup_vertical_task_types()
    """
    registry = TaskTypeRegistry.get_instance()

    # Register all vertical task types
    register_devops_task_types(registry)
    register_data_analysis_task_types(registry)
    register_coding_task_types(registry)
    register_research_task_types(registry)

    logger.debug(
        f"TaskTypeRegistry: Set up vertical task types. " f"Verticals: {registry.list_verticals()}"
    )
