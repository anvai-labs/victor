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

"""Unified Task Tracker - Single source of truth for task progress and loop detection.

This module consolidates TaskMilestoneMonitor and LoopDetector into a single
unified system that provides:

- Unified task types (7 fine-grained types)
- Combined milestone and loop detection
- Model-aware exploration settings
- Single configuration source
- Clear, testable interface

Usage:
    tracker = UnifiedTaskTracker()
    tracker.set_task_type(TrackerTaskType.EDIT)
    tracker.set_model_capabilities(tool_calling_caps)

    # In the main loop:
    tracker.record_tool_call("read_file", {"path": "test.py"})

    decision = tracker.should_stop()
    if decision.should_stop:
        # Force completion with decision.hint
        break
"""

import hashlib
import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import yaml

from victor.agent.action_authorizer import build_write_tool_set
from victor.core.loop_thresholds import (
    DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD,
    derive_blocked_total_threshold,
)
from victor.tools.tool_names import ToolNames, get_canonical_name
from victor.tools.metadata_registry import get_signature_params
from victor.protocols.mode_aware import ModeAwareMixin

if TYPE_CHECKING:
    from victor.agent.unified_classifier import ClassificationResult

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class TrackerTaskType(Enum):
    """Task types for unified task tracking with progress milestones.

    Fine-grained types optimized for milestone-based progress tracking.

    Renamed from TaskType to be semantically distinct:
    - TaskType (victor.classification.pattern_registry): Canonical prompt classification
    - TrackerTaskType: Progress tracking with milestones
    - LoopDetectorTaskType: Loop detection thresholds
    - ClassifierTaskType: Unified classification output
    - FrameworkTaskType: Framework-level task abstraction
    """

    # Action tasks (modify/create files)
    EDIT = "edit"  # Modify existing files
    CREATE = "create"  # Create new files with context
    CREATE_SIMPLE = "create_simple"  # Create files directly (no context needed)

    # Analysis tasks (read/search/understand)
    SEARCH = "search"  # Find/locate code or files
    ANALYZE = "analyze"  # Count, measure, analyze code
    RESEARCH = "research"  # Web research tasks

    # Other tasks
    DESIGN = "design"  # Conceptual/planning (no tools)
    GENERAL = "general"  # Ambiguous or mixed tasks

    def to_canonical(self) -> str:
        """Canonical task-type name in the TaskTypeRegistry (tool-supply P7).

        Collapses this adapter enum onto the single registry taxonomy; this enum's
        values are already canonical names.
        """
        from victor.framework.task_types import canonicalize_task_type

        return canonicalize_task_type(self.value)

    @classmethod
    def from_canonical(cls, name: str) -> "Optional[TrackerTaskType]":
        """Best-effort reverse map: a canonical name -> the member that yields it."""
        from victor.framework.task_types import canonicalize_task_type

        target = canonicalize_task_type(name)
        for member in cls:
            if member.to_canonical() == target:
                return member
        return None


class Milestone(Enum):
    """Task milestones for progress tracking."""

    TARGET_IDENTIFIED = "target_identified"
    TARGET_READ = "target_read"
    CHANGE_MADE = "change_made"
    CHANGE_VERIFIED = "change_verified"
    SEARCH_COMPLETE = "search_complete"


class ConversationStage(Enum):
    """Conversation stages for tool availability."""

    INITIAL = "initial"
    READING = "reading"
    EXECUTING = "executing"
    VERIFYING = "verifying"


class TrackerStopReason(Enum):
    """Reason for stopping task execution.

    Renamed from StopReason to be semantically distinct:
    - TrackerStopReason (here): Task tracker stop reasons (budget, loop, iterations)
    - LoopStopRecommendation (victor.agent.loop_detector): Loop detection recommendation dataclass
    - DebugStopReason (victor.observability.debug.protocol): Debugger stop reasons (breakpoint, step)
    """

    NONE = "none"
    # NOTE: there is intentionally no TOOL_BUDGET stop reason. ``tool_budget`` is advisory — it
    # steers the model via the system prompt, planning fast-path decisions, and learned
    # adherence, but never hard-stops execution. The hard bound is MAX_ITERATIONS.
    LOOP_DETECTED = "loop_detected"
    MAX_ITERATIONS = "max_iterations"
    GOAL_FORCING = "goal_forcing"
    MANUAL_STOP = "manual_stop"


# Backward compatibility alias
StopReason = TrackerStopReason


# Tools that indicate research activity
RESEARCH_TOOLS = frozenset({"web_search", "web_fetch", "tavily_search", "search_web", "fetch_url"})


# Default limit for read_file when not specified
DEFAULT_READ_LIMIT = 500

# Canonical tool name for file reading (from centralized registry)
CANONICAL_READ_TOOL = ToolNames.READ
CANONICAL_LIST_TOOL = ToolNames.LS
CANONICAL_WRITE_TOOL = ToolNames.WRITE
CANONICAL_EDIT_TOOL = ToolNames.EDIT
CANONICAL_SHELL_TOOL = ToolNames.SHELL
CANONICAL_TEST_TOOL = ToolNames.TEST

FALLBACK_SIGNATURE_PARAMS: Dict[str, Tuple[str, ...]] = {
    CANONICAL_READ_TOOL: ("path",),
    CANONICAL_LIST_TOOL: ("path",),
    CANONICAL_WRITE_TOOL: ("path",),
    CANONICAL_SHELL_TOOL: ("cmd",),
    ToolNames.GREP: ("query", "pattern", "path"),
    ToolNames.CODE_SEARCH: ("query", "path"),
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FileReadRange:
    """Represents a range of lines read from a file."""

    offset: int
    limit: int

    @property
    def end(self) -> int:
        """End line (exclusive) of this range."""
        return self.offset + self.limit

    def overlaps(self, other: "FileReadRange") -> bool:
        """Check if this range overlaps with another."""
        return self.offset < other.end and other.offset < self.end

    def __hash__(self) -> int:
        return hash((self.offset, self.limit))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FileReadRange):
            return False
        return self.offset == other.offset and self.limit == other.limit


@dataclass
class TaskConfig:
    """Configuration for a specific task type."""

    max_exploration_iterations: int = 8
    force_action_after_target_read: bool = False
    tool_budget: int = 50
    loop_repeat_threshold: int = 4  # Warning at 3, block at 4
    needs_tools: bool = True
    required_tools: List[str] = field(default_factory=list)
    stage_tools: Dict[str, List[str]] = field(default_factory=dict)
    force_action_hints: Dict[str, str] = field(default_factory=dict)


@dataclass
class StopDecision:
    """Decision about whether to stop execution."""

    should_stop: bool
    reason: StopReason = StopReason.NONE
    hint: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    is_warning: bool = False


@dataclass
class UnifiedTaskProgress:
    """Single source of truth for task progress."""

    # Task classification
    task_type: TrackerTaskType = TrackerTaskType.GENERAL

    # Iteration tracking
    iteration_count: int = 0  # Productive iterations (tool calls)
    exploration_iterations: int = 0  # Read/search operations (counts toward exploration limit)
    action_iterations: int = 0  # Write/modify operations (don't count toward exploration limit)
    total_turns: int = 0  # All turns including continuations
    low_output_iterations: int = 0

    # Tool budget
    tool_calls: int = 0
    tool_budget: int = 50

    # Milestone tracking
    milestones: Set[Milestone] = field(default_factory=set)
    target_files: Set[str] = field(default_factory=set)
    target_entities: Set[str] = field(default_factory=set)
    files_read: Set[str] = field(default_factory=set)
    files_modified: Set[str] = field(default_factory=set)

    # Loop detection
    unique_resources: Set[str] = field(default_factory=set)
    file_read_ranges: Dict[str, List[FileReadRange]] = field(default_factory=dict)
    signature_history: deque = field(default_factory=lambda: deque(maxlen=10))
    base_resource_counts: Counter = field(default_factory=Counter)
    loop_warning_given: bool = False
    warned_signature: Optional[str] = None
    consecutive_research_calls: int = 0

    # Stage tracking
    stage: ConversationStage = ConversationStage.INITIAL

    # Manual stop
    forced_stop: Optional[str] = None

    # True if the prompt had explicit file/fix counts (informational signal set by the
    # prompt-requirement extractor; tool_budget itself is advisory and never hard-stops).
    has_prompt_requirements: bool = False

    # Log deduplication - only log forcing completion once
    completion_forcing_logged: bool = False

    # Response loop detection - tracks last response content to detect repeated responses
    last_response_content: str = ""
    response_loop_detected: bool = False


# =============================================================================
# Configuration Loader
# =============================================================================


class UnifiedTaskConfigLoader:
    """Loads unified task configuration from YAML."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "task_types": {
            "edit": {
                "max_exploration_iterations": 10,
                "force_action_after_target_read": True,
                "tool_budget": 25,
                "loop_repeat_threshold": 8,
                "required_tools": ["edit", "read"],
                "stage_tools": {
                    "initial": ["ls", "code_search"],
                    "reading": ["read", "code_search"],
                    "executing": ["edit", "read"],
                    "verifying": ["read", "test"],
                },
                "force_action_hints": {
                    "after_target_read": "Use edit to make the change.",
                    "max_iterations": "Please make the change or explain blockers.",
                },
            },
            "create": {
                "max_exploration_iterations": 10,
                "force_action_after_target_read": False,
                "tool_budget": 25,
                "loop_repeat_threshold": 8,
                "required_tools": ["write"],
                "stage_tools": {
                    "initial": ["ls", "read"],
                    "reading": ["read"],
                    "executing": ["write", "edit"],
                    "verifying": ["read", "test"],
                },
                "force_action_hints": {
                    "max_iterations": "Please create the file.",
                },
            },
            "create_simple": {
                "max_exploration_iterations": 10,
                "force_action_after_target_read": False,
                "tool_budget": 20,
                "loop_repeat_threshold": 6,
                "required_tools": ["write"],
                "stage_tools": {
                    "initial": ["write"],
                    "reading": [],
                    "executing": ["write"],
                    "verifying": ["read"],
                },
                "force_action_hints": {
                    "immediate": "Create the code directly using write.",
                },
            },
            "search": {
                "max_exploration_iterations": 10,
                "force_action_after_target_read": False,
                "tool_budget": 25,
                "loop_repeat_threshold": 6,
                "required_tools": ["code_search", "read"],
                "stage_tools": {
                    "initial": ["ls", "code_search"],
                    "reading": ["read", "code_search"],
                    "executing": ["read"],
                    "verifying": ["read"],
                },
                "force_action_hints": {
                    "max_iterations": "Please summarize your findings.",
                },
            },
            "analyze": {
                "max_exploration_iterations": 20,
                "force_action_after_target_read": False,
                "tool_budget": 40,
                "loop_repeat_threshold": 10,
                "required_tools": ["read", "shell"],
                "stage_tools": {
                    "initial": ["ls", "code_search"],
                    "reading": ["read", "code_search", "shell"],
                    "executing": ["shell"],
                    "verifying": ["read"],
                },
                "force_action_hints": {
                    "max_iterations": "Please summarize your analysis.",
                },
            },
            "research": {
                "max_exploration_iterations": 10,
                "force_action_after_target_read": False,
                "tool_budget": 20,
                "loop_repeat_threshold": 6,
                "required_tools": ["web_search", "web_fetch"],
                "stage_tools": {
                    "initial": ["web_search"],
                    "reading": ["web_fetch", "web_search"],
                    "executing": ["web_fetch"],
                    "verifying": [],
                },
                "force_action_hints": {
                    "max_iterations": "Please summarize your research findings.",
                },
            },
            "design": {
                # Architecture/design questions require codebase exploration
                # to understand key components, structure, and patterns.
                # Increased limits from 3/5 to match "analyze" task type.
                "max_exploration_iterations": 20,
                "force_action_after_target_read": False,
                "tool_budget": 40,
                "loop_repeat_threshold": 10,
                "needs_tools": True,
                "required_tools": ["read", "ls", "code_search"],
                "stage_tools": {
                    "initial": ["ls", "code_search", "read"],
                    "reading": ["read", "code_search", "ls"],
                    "executing": [],
                    "verifying": ["read"],
                },
                "force_action_hints": {
                    "max_iterations": "Please summarize the architecture and provide your recommendations.",
                },
            },
            "general": {
                "max_exploration_iterations": 15,
                "force_action_after_target_read": False,
                "tool_budget": 35,
                "loop_repeat_threshold": 8,
                "required_tools": ["read", "ls"],
                "stage_tools": {
                    "initial": ["ls", "code_search", "read"],
                    "reading": ["read", "code_search"],
                    "executing": ["edit", "write", "shell"],
                    "verifying": ["read", "test"],
                },
                "force_action_hints": {
                    "max_iterations": "Please complete the task or explain blockers.",
                },
            },
        },
        "model_overrides": {
            "deepseek*": {
                "exploration_multiplier": 1.5,
                "continuation_patience": 10,
            },
            "qwen*": {
                "exploration_multiplier": 1.3,
                "continuation_patience": 8,
            },
        },
        "global": {
            "max_total_iterations": 50,
            "min_content_threshold": 150,
            "signature_history_size": 10,
            "max_overlapping_reads_per_file": 3,
            "max_searches_per_query_prefix": 2,
        },
    }

    _instance: Optional["UnifiedTaskConfigLoader"] = None
    _config: Optional[Dict[str, Any]] = None

    def __new__(cls) -> "UnifiedTaskConfigLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._config is None:
            self._load_config()

    def _load_config(self) -> None:
        """Load configuration from existing YAML files.

        Uses:
        - task_tool_config.yaml for task type configurations
        - model_capabilities.yaml for model-specific settings (via capabilities loader)
        """
        # Load task config from existing task_tool_config.yaml
        task_config_path = Path(__file__).parent.parent / "config" / "task_tool_config.yaml"

        if task_config_path.exists():
            try:
                with open(task_config_path) as f:
                    task_config = yaml.safe_load(f) or {}

                # Restructure to match our expected format
                self._config = {
                    "task_types": self._normalize_task_types(task_config.get("task_types", {})),
                    "global": {
                        "max_total_iterations": 50,
                        "min_content_threshold": 150,
                        "signature_history_size": 10,
                        "max_overlapping_reads_per_file": 3,
                        "max_searches_per_query_prefix": 2,
                    },
                    # Model overrides come from model_capabilities.yaml via capabilities loader
                    "model_overrides": self.DEFAULT_CONFIG.get("model_overrides", {}),
                }
                logger.debug(f"Loaded task config from {task_config_path}")
            except Exception as e:
                logger.warning(f"Failed to load task config: {e}, using defaults")
                self._config = {
                    **self.DEFAULT_CONFIG,
                    "task_types": self._normalize_task_types(
                        self.DEFAULT_CONFIG.get("task_types", {})
                    ),
                }
        else:
            logger.debug("Using default task config (task_tool_config.yaml not found)")
            self._config = {
                **self.DEFAULT_CONFIG,
                "task_types": self._normalize_task_types(self.DEFAULT_CONFIG.get("task_types", {})),
            }

    def reload(self) -> None:
        """Force reload configuration."""
        self._config = None
        self._load_config()

    @staticmethod
    def _normalize_task_types(task_types: Any) -> Dict[str, Any]:
        """Normalize loaded task types to canonical runtime tool names."""
        if not isinstance(task_types, dict):
            return {}

        normalized: Dict[str, Any] = {}
        for task_name, task_data in task_types.items():
            if not isinstance(task_data, dict):
                continue
            task_copy = dict(task_data)
            required = task_data.get("required_tools", [])
            task_copy["required_tools"] = [
                get_canonical_name(tool) if isinstance(tool, str) else tool for tool in required
            ]

            raw_stage_tools = task_data.get("stage_tools", {})
            normalized_stage_tools = {}
            if isinstance(raw_stage_tools, dict):
                for stage_name, tools in raw_stage_tools.items():
                    if isinstance(tools, list):
                        normalized_stage_tools[stage_name] = [
                            get_canonical_name(tool) if isinstance(tool, str) else tool
                            for tool in tools
                        ]
                    else:
                        normalized_stage_tools[stage_name] = []
            task_copy["stage_tools"] = normalized_stage_tools
            normalized[task_name] = task_copy
        return normalized

    def get_task_config(self, task_type: TrackerTaskType) -> TaskConfig:
        """Get configuration for a specific task type."""
        config = self._config or self.DEFAULT_CONFIG
        task_configs = config.get("task_types", {})
        task_data = task_configs.get(task_type.value, task_configs.get("general", {}))

        return TaskConfig(
            max_exploration_iterations=task_data.get("max_exploration_iterations", 8),
            force_action_after_target_read=task_data.get("force_action_after_target_read", False),
            tool_budget=task_data.get("tool_budget", 50),
            loop_repeat_threshold=task_data.get(
                "loop_repeat_threshold", 4
            ),  # Warning at 3, block at 4
            needs_tools=task_data.get("needs_tools", True),
            required_tools=task_data.get("required_tools", []),
            stage_tools=task_data.get("stage_tools", {}),
            force_action_hints=task_data.get("force_action_hints", {}),
        )

    def get_global_config(self) -> Dict[str, Any]:
        """Get global configuration settings."""
        config = self._config or self.DEFAULT_CONFIG
        return config.get("global", {})

    def get_model_override(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific override settings."""
        config = self._config or self.DEFAULT_CONFIG
        overrides = config.get("model_overrides", {})

        model_lower = model_name.lower()
        for pattern, settings in overrides.items():
            # Simple glob matching
            if pattern.endswith("*"):
                if model_lower.startswith(pattern[:-1].lower()):
                    return settings
            elif model_lower == pattern.lower():
                return settings

        return {}


# =============================================================================
# Main Tracker Class
# =============================================================================


class UnifiedTaskTracker(ModeAwareMixin):
    """Unified task tracking combining milestones and loop detection.

    This class is the single source of truth for:
    - Task type and configuration
    - Iteration counting and limits
    - Milestone tracking
    - Loop detection
    - Stop decisions

    Uses ModeAwareMixin for consistent mode controller access (via self.is_build_mode,
    self.exploration_multiplier, etc.).

    Usage:
        tracker = UnifiedTaskTracker()
        tracker.set_task_type(TrackerTaskType.EDIT)

        # Record tool calls
        tracker.record_tool_call("read_file", {"path": "test.py"})

        # Check if should stop
        decision = tracker.should_stop()
        if decision.should_stop:
            print(decision.hint)
    """

    def __init__(self, budget_manager: Optional[Any] = None) -> None:
        """Initialize the unified task tracker.

        Args:
            budget_manager: Optional BudgetManager for unified budget tracking.
                          If provided, budget operations are delegated to it.
        """
        self._config_loader = UnifiedTaskConfigLoader()
        self._progress = UnifiedTaskProgress()
        self._task_config: Optional[TaskConfig] = None
        self._sticky_user_budget: bool = False
        self._allow_budget_override: bool = False
        self._sticky_user_iterations: bool = False
        self._allow_iteration_override: bool = False

        # Optional BudgetManager integration (parallel operation)
        self._budget_manager = budget_manager

        # Model-specific settings
        self._exploration_multiplier: float = 1.0
        self._continuation_patience: int = 10

        # Agent mode settings (plan/explore get higher multipliers)
        self._mode_exploration_multiplier: float = 1.0

        # Global settings
        global_config = self._config_loader.get_global_config()
        self._max_total_iterations = global_config.get("max_total_iterations", 50)
        self._min_content_threshold = global_config.get("min_content_threshold", 150)
        self._base_max_overlapping_reads = global_config.get("max_overlapping_reads_per_file", 3)
        self._base_max_searches_per_prefix = global_config.get("max_searches_per_query_prefix", 2)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def progress(self) -> UnifiedTaskProgress:
        """Get current progress state."""
        return self._progress

    @property
    def task_type(self) -> TrackerTaskType:
        """Get current task type."""
        return self._progress.task_type

    @property
    def iteration_count(self) -> int:
        """Get productive iteration count."""
        return self._progress.iteration_count

    @property
    def tool_calls(self) -> int:
        """Get total tool calls."""
        return self._progress.tool_calls

    @property
    def remaining_budget(self) -> int:
        """Get remaining tool budget."""
        return max(0, self._progress.tool_budget - self._progress.tool_calls)

    @property
    def milestones(self) -> Set[Milestone]:
        """Get achieved milestones."""
        return self._progress.milestones.copy()

    @property
    def stage(self) -> ConversationStage:
        """Get current conversation stage."""
        return self._progress.stage

    @property
    def max_overlapping_reads(self) -> int:
        """Get mode-aware max overlapping reads per file.

        PLAN/EXPLORE modes get higher limits to allow thorough file exploration.
        """
        effective = int(self._base_max_overlapping_reads * self._mode_exploration_multiplier)
        return max(self._base_max_overlapping_reads, effective)

    @property
    def max_searches_per_prefix(self) -> int:
        """Get mode-aware max searches per query prefix.

        PLAN/EXPLORE modes get higher limits to allow thorough search exploration.
        """
        effective = int(self._base_max_searches_per_prefix * self._mode_exploration_multiplier)
        return max(self._base_max_searches_per_prefix, effective)

    # =========================================================================
    # Configuration
    # =========================================================================

    def set_task_type(self, task_type: TrackerTaskType) -> None:
        """Set the task type and load corresponding configuration."""
        self._progress.task_type = task_type
        self._task_config = self._config_loader.get_task_config(task_type)

        # Respect sticky user budget - only update if not user-set
        if not self._sticky_user_budget:
            self._progress.tool_budget = self._task_config.tool_budget

        logger.info(
            f"UnifiedTaskTracker: task_type={task_type.value}, "
            f"max_iterations={self._task_config.max_exploration_iterations}, "
            f"tool_budget={self._task_config.tool_budget}"
        )

    def set_model_capabilities(
        self,
        exploration_multiplier: float = 1.0,
        continuation_patience: int = 10,
    ) -> None:
        """Set model-specific exploration settings."""
        self._exploration_multiplier = exploration_multiplier
        self._continuation_patience = continuation_patience

        logger.info(
            f"UnifiedTaskTracker: model settings - "
            f"multiplier={exploration_multiplier}, patience={continuation_patience}"
        )

    def set_target_files(self, files: Set[str]) -> None:
        """Set target files for the task."""
        self._progress.target_files = files

    def set_tool_budget(self, budget: int, user_override: bool = False) -> None:
        """Set the tool budget for this task.

        Args:
            budget: Maximum number of tool calls allowed
            user_override: Mark this budget as sticky (prevents auto-adjustment)
        """
        if self._sticky_user_budget and not user_override and not self._allow_budget_override:
            logger.debug("UnifiedTaskTracker: sticky user budget set; skipping auto-adjustment")
            return

        if user_override:
            self._allow_budget_override = True
            self._sticky_user_budget = True

        self._progress.tool_budget = budget
        logger.debug(f"UnifiedTaskTracker: tool_budget set to {budget}")
        self._allow_budget_override = False

    def set_max_iterations(self, iterations: int, user_override: bool = False) -> None:
        """Set the maximum total iterations allowed.

        Args:
            iterations: Maximum iterations
            user_override: Mark this limit as sticky (prevents auto-adjustment)
        """
        if (
            self._sticky_user_iterations
            and not user_override
            and not self._allow_iteration_override
        ):
            logger.debug(
                "UnifiedTaskTracker: sticky user max iterations set; skipping auto-adjustment"
            )
            return

        if user_override:
            self._allow_iteration_override = True
            self._sticky_user_iterations = True

        self._max_total_iterations = iterations
        logger.debug(f"UnifiedTaskTracker: max_total_iterations set to {iterations}")
        self._allow_iteration_override = False

    def set_target_entities(self, entities: Set[str]) -> None:
        """Set target entities (functions, classes) for the task."""
        self._progress.target_entities = entities

    def get_required_tools(self) -> Set[str]:
        """Get the required tools for the current task type.

        Returns:
            Set of tool names required for this task
        """
        if self._task_config is None:
            return set()
        return set(self._task_config.required_tools)

    @property
    def max_exploration_iterations(self) -> int:
        """Get max exploration iterations for current task type.

        Combines base task config with mode and model multipliers:
        - Plan mode: 2.5x multiplier (more exploration needed for planning)
        - Explore mode: 3.0x multiplier (exploration is the primary goal)
        - Model multiplier: varies by model capability

        Capped proportionally to the tool budget to prevent runaway exploration.
        """
        # Handle case where task_config hasn't been set yet
        if self._task_config is None:
            return 8  # Default fallback
        base = self._task_config.max_exploration_iterations
        combined_multiplier = self._exploration_multiplier * self._mode_exploration_multiplier
        raw = int(base * combined_multiplier)
        # Cap at 3x the current tool budget to prevent runaway exploration
        tool_budget = self._progress.tool_budget
        if tool_budget and tool_budget > 0:
            cap = max(tool_budget * 3, 30)  # At least 30
            return min(raw, cap)
        return min(raw, 100)  # Hard ceiling

    def set_mode_exploration_multiplier(self, multiplier: float) -> None:
        """Set the agent mode exploration multiplier.

        Plan mode (2.5x) and Explore mode (3.0x) need more iterations
        since their purpose is thorough exploration before action.

        Also syncs with BudgetManager if available.

        Args:
            multiplier: Mode-specific multiplier (1.0 for build, 2.5 for plan, 3.0 for explore)
        """
        self._mode_exploration_multiplier = multiplier

        # Sync with BudgetManager if available
        if self._budget_manager:
            self._budget_manager.set_mode_multiplier(multiplier)

        logger.info(
            f"UnifiedTaskTracker: mode_exploration_multiplier={multiplier}, "
            f"effective_max={self.max_exploration_iterations}"
        )

    def set_budget_manager(self, budget_manager: Any) -> None:
        """Set the BudgetManager for unified budget tracking.

        Enables parallel operation mode where both the tracker and
        BudgetManager track budgets. Useful for migration and comparison.

        Args:
            budget_manager: BudgetManager instance
        """
        self._budget_manager = budget_manager

        # Sync current multipliers
        if budget_manager:
            budget_manager.set_mode_multiplier(self._mode_exploration_multiplier)
            budget_manager.set_model_multiplier(self._exploration_multiplier)
            logger.debug(
                f"UnifiedTaskTracker: BudgetManager attached with "
                f"mode_multiplier={self._mode_exploration_multiplier}, "
                f"model_multiplier={self._exploration_multiplier}"
            )

    # =========================================================================
    # Recording
    # =========================================================================

    # Tools that perform write/modify actions (don't count toward exploration limit)
    WRITE_TOOLS = set(
        build_write_tool_set(
            "git_commit",
            "git_push",
            "create_file",
            "delete_file",
            "rename_file",
            "notebook_edit",
            "refactor",
            "rename",
        )
    )

    def record_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Record a tool call - updates milestones, loops, and budgets.

        Classifies tool calls as exploration (read/search) or action (write/modify).
        Only exploration iterations count toward the exploration limit in BUILD mode.

        If a BudgetManager is available, also delegates budget tracking to it.
        """
        self._progress.tool_calls += 1
        self._progress.iteration_count += 1
        self._progress.total_turns += 1
        canonical_tool_name = get_canonical_name(tool_name)

        # Classify as exploration vs action
        # Shell commands with write-like operations are actions
        is_write_operation = canonical_tool_name in self.WRITE_TOOLS
        if canonical_tool_name == CANONICAL_SHELL_TOOL and arguments:
            cmd = arguments.get("cmd") or arguments.get("command", "")
            # Check for write-like shell commands
            write_commands = [
                "mkdir",
                "touch",
                "echo",
                "cat >",
                "cp",
                "mv",
                "rm",
                "chmod",
                "chown",
            ]
            is_write_operation = any(wc in cmd for wc in write_commands)

        if is_write_operation:
            self._progress.action_iterations += 1
        else:
            self._progress.exploration_iterations += 1

        # Delegate to BudgetManager if available (parallel operation)
        if self._budget_manager:
            self._budget_manager.record_tool_call(canonical_tool_name, is_write_operation)

        # Update milestones
        self._update_milestones(canonical_tool_name, arguments)

        # Update loop detection
        self._update_loop_state(canonical_tool_name, arguments)

        # Track resources
        self._track_resource(canonical_tool_name, arguments)

        # Update stage
        self._update_stage(canonical_tool_name)

        logger.debug(
            f"UnifiedTaskTracker: tool_call={canonical_tool_name}, "
            f"iteration={self._progress.iteration_count}, "
            f"exploration={self._progress.exploration_iterations}, "
            f"action={self._progress.action_iterations}"
        )

    def increment_turn(self) -> None:
        """Record a turn without tool call (continuation prompt)."""
        self._progress.total_turns += 1
        logger.debug(
            f"UnifiedTaskTracker: turn without tool call, "
            f"total_turns={self._progress.total_turns}, "
            f"productive={self._progress.iteration_count}"
        )

    def record_iteration(self, content_length: int) -> None:
        """Record an iteration completion with content length."""
        if content_length < self._min_content_threshold:
            self._progress.low_output_iterations += 1

    def force_stop(self, reason: str) -> None:
        """Force the tracker to recommend stopping."""
        self._progress.forced_stop = reason

    def reset(self) -> None:
        """Reset all state for a new conversation.

        Preserves user-overridden sticky values (budget, max_iterations) to
        honor explicit user settings across conversation turns.
        """
        # Preserve sticky values before reset
        sticky_budget = self._progress.tool_budget if self._sticky_user_budget else None
        sticky_max_iter = self._max_total_iterations if self._sticky_user_iterations else None

        self._progress = UnifiedTaskProgress()
        self._task_config = None

        # Restore sticky values after reset
        if sticky_budget is not None:
            self._progress.tool_budget = sticky_budget
        if sticky_max_iter is not None:
            self._max_total_iterations = sticky_max_iter

    # =========================================================================
    # Stop Decision
    # =========================================================================

    def should_stop(self) -> StopDecision:
        """Single unified check for whether to stop.

        Checks in order:
        1. Manual force stop
        2. True loop detected
        3. Max iterations exceeded
        4. Goal-based forcing (milestone-aware)

        Note: ``tool_budget`` is advisory (it steers the model via the prompt/planning and is
        learned as adherence) and is deliberately *not* a stop condition here — the hard
        execution bound is max iterations.
        """
        details = self._get_details()

        # Manual stop
        if self._progress.forced_stop:
            return StopDecision(
                should_stop=True,
                reason=StopReason.MANUAL_STOP,
                hint=f"Manual stop: {self._progress.forced_stop}",
                details=details,
            )

        # NOTE: tool_budget is intentionally NOT enforced here. It is an *advisory* signal —
        # surfaced to the model in the system prompt, used for planning fast-path decisions, and
        # learned as adherence — but it never hard-stops execution. The hard execution bound is
        # MAX_ITERATIONS (checked below), enforced by both the streaming and headless loops via
        # max_total_iterations. (A soft-warn/hard-stop budget branch used to live here but was
        # never wired into either loop's stop path, so it was dead code; removed to match reality.)

        # Loop check
        loop = self._check_loop()
        if loop:
            return StopDecision(
                should_stop=True,
                reason=StopReason.LOOP_DETECTED,
                hint=f"Loop detected: {loop}",
                details=details,
            )

        # Hard iteration limit
        if self._progress.iteration_count >= self._max_total_iterations:
            return StopDecision(
                should_stop=True,
                reason=StopReason.MAX_ITERATIONS,
                hint=f"Max iterations reached ({self._progress.iteration_count}/{self._max_total_iterations})",
                details=details,
            )

        # Task-specific iteration limit (with model multiplier)
        # In BUILD mode (allow_all_tools), only exploration iterations count toward limit
        # This allows agents to continue creating files without hitting exploration limit
        effective_max = self._calculate_effective_max()

        # Determine which counter to use for exploration limit
        # Uses ModeAwareMixin for consistent mode controller access
        is_build_mode = self.is_build_mode

        # In BUILD mode, use exploration_iterations; otherwise use total iteration_count
        exploration_count = (
            self._progress.exploration_iterations
            if is_build_mode
            else self._progress.iteration_count
        )

        if exploration_count > effective_max:
            hint = self._get_completion_hint()
            # Only log once to avoid duplicate messages
            if not self._progress.completion_forcing_logged:
                self._progress.completion_forcing_logged = True
                logger.info(
                    f"UnifiedTaskTracker: Forcing completion at exploration_count={exploration_count} "
                    f"(task_type={self._progress.task_type.value}, "
                    f"base_max={self._get_base_max()}, effective_max={effective_max}, "
                    f"total_iterations={self._progress.iteration_count}, "
                    f"action_iterations={self._progress.action_iterations}, "
                    f"is_build_mode={is_build_mode})"
                )
            return StopDecision(
                should_stop=True,
                reason=StopReason.MAX_ITERATIONS,
                hint=hint,
                details=details,
            )

        # Goal-based forcing (EDIT tasks after target read)
        goal_decision = self._check_goal_forcing()
        if goal_decision:
            return goal_decision

        return StopDecision(should_stop=False, details=details)

    def check_loop_warning(self) -> Optional[str]:
        """Check if approaching loop threshold (warning before hard stop).

        When a signature is warned, it's immediately added to the permanent block list
        so it will be blocked on any future attempt, even if the model tries other
        operations in between.
        """
        # Initialize permanently blocked set if needed
        if not hasattr(self._progress, "permanently_blocked"):
            self._progress.permanently_blocked = set()

        if len(self._progress.signature_history) < 3:
            return None

        threshold = self._get_loop_threshold()
        recent = list(self._progress.signature_history)[
            -min(len(self._progress.signature_history), 8) :
        ]

        if recent:
            sig_counts = Counter(recent)
            for sig, count in sig_counts.items():
                # Skip if already permanently blocked
                if sig in self._progress.permanently_blocked:
                    continue
                if count == threshold - 1:
                    self._progress.loop_warning_given = True
                    self._progress.warned_signature = sig
                    # Add to permanent block list IMMEDIATELY when warning is given
                    # This ensures it stays blocked even if model tries other operations
                    self._progress.permanently_blocked.add(sig)

                    # Parse signature for logging
                    sig_parts = sig.split("|")
                    tool_name = sig_parts[0] if sig_parts else sig

                    logger.info(
                        f"[dedup-warning] Tool call approaching loop threshold: {tool_name} | "
                        f"signature={sig[:80]}... | "
                        f"count={count}/{threshold} | "
                        f"Adding to permanent block list"
                    )

                    return f"Approaching loop ({count}/{threshold}): {sig[:80]}"

        return None

    def is_blocked_after_warning(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Check if a tool call is blocked due to being in the permanent block list.

        Once a signature is warned (in check_loop_warning), it's permanently blocked
        for the entire conversation. This prevents the model from cycling between
        different tools and returning to the blocked operation.
        """
        # Initialize permanently blocked set if needed
        if not hasattr(self._progress, "permanently_blocked"):
            self._progress.permanently_blocked = set()

        proposed_sig = self._get_signature(tool_name, arguments)

        # Check if this signature is permanently blocked
        if proposed_sig in self._progress.permanently_blocked:
            # Log detailed information about the blocked tool call
            args_summary = ", ".join(
                f"{k}={repr(v)[:30]}" for k, v in arguments.items() if k not in {"offset", "limit"}
            )
            logger.info(
                f"[dedup-block] Tool call BLOCKED: {tool_name}({args_summary}) | "
                f"signature={proposed_sig[:60]}... | "
                f"permanently_blocked after loop warning"
            )
            return f"Blocked: same operation after warning ({proposed_sig[:50]})"

        return None

    def get_loop_patience_limits(self) -> tuple[int, int]:
        """Get the patience limits for forced completion after blocked attempts.

        Returns:
            Tuple of (consecutive_limit, total_limit):
            - consecutive_limit: Force completion after N consecutive blocked attempts
            - total_limit: Force completion after N total blocked attempts (across conversation)
        """
        threshold = self._get_loop_threshold()
        consecutive_limit = threshold
        total_limit = derive_blocked_total_threshold(threshold)
        return consecutive_limit, total_limit

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get current progress metrics for logging/monitoring."""
        return {
            "task_type": self._progress.task_type.value,
            "iteration_count": self._progress.iteration_count,
            "total_turns": self._progress.total_turns,
            "tool_calls": self._progress.tool_calls,
            "tool_budget": self._progress.tool_budget,
            "remaining_budget": self.remaining_budget,
            "unique_resources": len(self._progress.unique_resources),
            "files_read": len(self._progress.files_read),
            "files_modified": len(self._progress.files_modified),
            "milestones": [m.value for m in self._progress.milestones],
            "stage": self._progress.stage.value,
            "exploration_multiplier": self._exploration_multiplier,
            "effective_max": self._calculate_effective_max(),
        }

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _update_milestones(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Update milestones based on tool call."""
        if tool_name in {CANONICAL_LIST_TOOL, "code_search", "semantic_code_search"}:
            self._progress.milestones.add(Milestone.TARGET_IDENTIFIED)

        elif tool_name == CANONICAL_READ_TOOL:
            path = arguments.get("path", "")
            if path:
                self._progress.files_read.add(path)
                if path in self._progress.target_files:
                    self._progress.milestones.add(Milestone.TARGET_READ)

        elif tool_name in {CANONICAL_EDIT_TOOL, CANONICAL_WRITE_TOOL}:
            self._progress.milestones.add(Milestone.CHANGE_MADE)
            if tool_name == CANONICAL_EDIT_TOOL:
                ops = arguments.get("ops") or arguments.get("files", [])
                if isinstance(ops, list):
                    for op in ops:
                        if isinstance(op, dict):
                            path = op.get("path") or op.get("new_path")
                            if path:
                                self._progress.files_modified.add(path)
            else:
                path = arguments.get("path", "")
                if path:
                    self._progress.files_modified.add(path)

        elif tool_name in {CANONICAL_TEST_TOOL, CANONICAL_SHELL_TOOL}:
            if Milestone.CHANGE_MADE in self._progress.milestones:
                self._progress.milestones.add(Milestone.CHANGE_VERIFIED)

    def _update_loop_state(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Update loop detection state."""
        # Track file reads with offset-aware detection
        if tool_name == CANONICAL_READ_TOOL:
            self._track_file_read(arguments)
        else:
            # Track base resources for non-file operations
            base_key = self._get_base_resource_key(tool_name, arguments)
            if base_key:
                self._progress.base_resource_counts[base_key] += 1

        # Track signature
        signature = self._get_signature(tool_name, arguments)

        # Clear warning if truly different signature
        if self._progress.loop_warning_given:
            recent = list(self._progress.signature_history)[-3:]
            if signature not in recent:
                self._progress.loop_warning_given = False
                self._progress.warned_signature = None

        self._progress.signature_history.append(signature)

        # Track research calls
        if tool_name in RESEARCH_TOOLS:
            self._progress.consecutive_research_calls += 1
        else:
            self._progress.consecutive_research_calls = 0

    def _track_file_read(self, arguments: Dict[str, Any]) -> None:
        """Track file read with offset-aware overlap detection."""
        path = arguments.get("path", "")
        if not path:
            return

        offset = arguments.get("offset", 0)
        limit = arguments.get("limit", DEFAULT_READ_LIMIT)
        new_range = FileReadRange(offset=offset, limit=limit)

        if path not in self._progress.file_read_ranges:
            self._progress.file_read_ranges[path] = []

        self._progress.file_read_ranges[path].append(new_range)

    def _track_resource(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Track unique resource access."""
        resource_key = self._get_resource_key(tool_name, arguments)
        if resource_key:
            self._progress.unique_resources.add(resource_key)

    def _update_stage(self, tool_name: str) -> None:
        """Update conversation stage based on tool usage."""
        if tool_name in {CANONICAL_READ_TOOL, ToolNames.GREP, ToolNames.CODE_SEARCH}:
            if self._progress.stage == ConversationStage.INITIAL:
                self._progress.stage = ConversationStage.READING

        elif tool_name in {CANONICAL_EDIT_TOOL, CANONICAL_WRITE_TOOL}:
            self._progress.stage = ConversationStage.EXECUTING

        elif tool_name in {CANONICAL_TEST_TOOL, CANONICAL_SHELL_TOOL}:
            if Milestone.CHANGE_MADE in self._progress.milestones:
                self._progress.stage = ConversationStage.VERIFYING

    def _check_loop(self) -> Optional[str]:
        """Check for loop patterns."""
        # Check for read-only loop (too many reads without any writes)
        readonly_loop = self._check_readonly_loop()
        if readonly_loop:
            return readonly_loop

        # Check file read overlaps
        file_loop = self._check_file_read_loops()
        if file_loop:
            return file_loop

        # Check search query repetition (mode-aware limit)
        for base_key, count in self._progress.base_resource_counts.items():
            if base_key.startswith("search:") and count > self.max_searches_per_prefix:
                return f"Similar search repeated {count} times: {base_key}"

        # Check signature repetition
        if len(self._progress.signature_history) < 3:
            return None

        threshold = self._get_loop_threshold()
        recent = list(self._progress.signature_history)[
            -min(len(self._progress.signature_history), 8) :
        ]

        if recent:
            sig_counts = Counter(recent)
            for sig, count in sig_counts.items():
                if count >= threshold:
                    return f"Same signature repeated {count} times: {sig[:50]}"

        return None

    def _check_readonly_loop(self) -> Optional[str]:
        """Check for read-only loop - too many reads without any writes.

        This catches the case where the agent keeps reading different files
        without making any progress on the actual task (no edits/creates/writes).
        """
        # Only check for analysis/research tasks after minimum iterations
        if self._progress.task_type not in {
            TrackerTaskType.ANALYZE,
            TrackerTaskType.SEARCH,
            TrackerTaskType.RESEARCH,
        }:
            return None

        # Need at least some iterations before checking
        if self._progress.iteration_count < 10:
            return None

        # Check if we've read many files without any write operations
        files_read = len(self._progress.files_read)
        files_modified = len(self._progress.files_modified)

        # If we've read more than 20 files without modifying anything, flag it
        if files_read > 20 and files_modified == 0:
            return f"Read-only loop: {files_read} files read, 0 files modified"

        # If we've read more than 50 files with fewer than 5 modifications, flag it
        if files_read > 50 and files_modified < 5:
            return f"Read-heavy loop: {files_read} files read, only {files_modified} files modified"

        return None

    def _check_file_read_loops(self) -> Optional[str]:
        """Check for overlapping file read loops."""
        for path, ranges in self._progress.file_read_ranges.items():
            if len(ranges) < 2:
                continue

            for i, current in enumerate(ranges):
                overlap_count = sum(
                    1 for j, other in enumerate(ranges) if i != j and current.overlaps(other)
                )
                total = overlap_count + 1
                # Use mode-aware overlapping reads limit
                if total > self.max_overlapping_reads:
                    return (
                        f"Same file region read {total} times: {path} " f"[offset={current.offset}]"
                    )

        return None

    def check_response_loop(self, content: str, similarity_threshold: float = 0.9) -> bool:
        """Check if response content is a repeat of the previous response.

        This detects when the model keeps responding with similar text but makes
        no tool calls - a different type of loop than tool call loops.

        P0 FIX: Improved detection with proper loop exit handling.
        - Also tracks consecutive loops to prevent repeated detections
        - Sets forced_stop flag to ensure proper termination

        NOTE: Threshold raised from 0.7 to 0.9 to reduce false positives.
        When exploring directories, responses like "Let me examine dir1" and
        "Let me examine dir2" have high word overlap (~80%) but represent progress.
        A 0.9 threshold catches near-verbatim repeats while allowing variation.

        Args:
            content: The current response content to check
            similarity_threshold: Word overlap ratio to consider as repeated (default 0.9)

        Returns:
            True if a response loop is detected, False otherwise
        """
        content_for_comparison = (content or "").strip()[:500]
        last_content = self._progress.last_response_content

        # P0 FIX: Track consecutive loop detections for proper exit
        if not hasattr(self._progress, "consecutive_response_loops"):
            self._progress.consecutive_response_loops = 0

        is_repeated = False
        if last_content and content_for_comparison:
            # Simple word overlap similarity check
            current_words = set(content_for_comparison.lower().split())
            last_words = set(last_content.lower().split())
            if current_words and last_words:
                overlap = len(current_words & last_words)
                max_words = max(len(current_words), len(last_words))
                similarity = overlap / max_words if max_words > 0 else 0
                if similarity > similarity_threshold:
                    self._progress.consecutive_response_loops += 1
                    # P0 FIX: Force stop after 2 consecutive detections
                    if self._progress.consecutive_response_loops >= 2:
                        is_repeated = True
                        self._progress.response_loop_detected = True
                        self._progress.forced_stop = "response_loop"
                        logger.warning(
                            f"Response loop detected (similarity={similarity:.2f}, consecutive={self._progress.consecutive_response_loops}), "
                            "forcing completion to prevent infinite loop"
                        )
                    else:
                        # First detection - just log and continue
                        logger.info(
                            f"Potential response loop detected (similarity={similarity:.2f}), "
                            f"will force completion on next repetition"
                        )
                else:
                    # Reset counter when content is different
                    self._progress.consecutive_response_loops = 0

        # Track content for next iteration's comparison
        self._progress.last_response_content = content_for_comparison
        return is_repeated

    @property
    def response_loop_detected(self) -> bool:
        """Check if a response loop has been detected."""
        return self._progress.response_loop_detected

    def _check_goal_forcing(self) -> Optional[StopDecision]:
        """Check for goal-based forcing (EDIT after target read)."""
        if not self._task_config:
            return None

        if (
            self._task_config.force_action_after_target_read
            and Milestone.TARGET_READ in self._progress.milestones
            and Milestone.CHANGE_MADE not in self._progress.milestones
            and self._progress.iteration_count >= self._calculate_effective_max()
        ):
            hint = self._task_config.force_action_hints.get(
                "after_target_read",
                "You have read the target. Please make the change.",
            )
            return StopDecision(
                should_stop=True,
                reason=StopReason.GOAL_FORCING,
                hint=hint,
                details=self._get_details(),
            )

        return None

    def _calculate_effective_max(self) -> int:
        """Calculate effective max iterations with model and mode adjustments."""
        base_max = self._get_base_max()

        # Apply combined multiplier (model * mode)
        # Mode multipliers: Build=1.0, Plan=2.5, Explore=3.0
        combined_multiplier = self._exploration_multiplier * self._mode_exploration_multiplier
        model_adjusted = int(base_max * combined_multiplier)

        # Apply productivity ratio adjustment
        total = self._progress.total_turns
        productive = self._progress.iteration_count

        if total > 0 and productive > 0:
            productivity = productive / total
            if productivity > 0:
                productivity_mult = min(2.0, max(1.0, 1.0 / productivity))
                return int(model_adjusted * productivity_mult)

        return model_adjusted

    def _get_base_max(self) -> int:
        """Get base max iterations from task config."""
        if self._task_config:
            return self._task_config.max_exploration_iterations
        return 8

    def _get_loop_threshold(self) -> int:
        """Get loop repeat threshold from task config with mode awareness.

        Warning triggers at threshold - 1, block at threshold.
        Default is 4 (warning at 3, block at 4).

        Mode multipliers are applied to allow more exploration in PLAN/EXPLORE modes:
        - BUILD mode: 1.0x (default threshold)
        - PLAN mode: 2.5x (e.g., threshold 4 becomes 10)
        - EXPLORE mode: 3.0x (e.g., threshold 4 becomes 12)
        """
        base_threshold = DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD
        if self._task_config:
            base_threshold = self._task_config.loop_repeat_threshold

        # Apply mode multiplier for PLAN/EXPLORE modes
        # This allows more exploration before triggering loop detection
        effective_threshold = int(base_threshold * self._mode_exploration_multiplier)

        # Ensure minimum threshold of base to prevent immediate loops
        return max(base_threshold, effective_threshold)

    def _get_completion_hint(self) -> str:
        """Get appropriate completion hint based on task type."""
        if self._task_config:
            return self._task_config.force_action_hints.get(
                "max_iterations",
                "Please complete the task or explain what's blocking you.",
            )

        # Fallback hints by task type
        if self._progress.task_type == TrackerTaskType.SEARCH:
            return "Please summarize your findings and provide the answer."
        elif self._progress.task_type == TrackerTaskType.ANALYZE:
            return "Please summarize your analysis."
        elif self._progress.task_type in {TrackerTaskType.EDIT, TrackerTaskType.CREATE}:
            return "Please complete the change or explain what's blocking you."
        else:
            return "Please complete the task or explain what's blocking you."

    def _get_signature(
        self, tool_name: str, arguments: Dict[str, Any], include_stage: bool = True
    ) -> str:
        """Generate context-aware signature for loop detection.

        Tools declare signature_params via @tool decorator to explicitly control
        which parameters are included in loop detection signatures. Only params in
        signature_params affect the signature - pagination parameters (offset, limit, k)
        are excluded to allow reading results in chunks.

        Examples:
            read(path="file.py", offset=0)   → signature="read|file.py|stage:initial"
            read(path="file.py", offset=500) → signature="read|file.py|stage:initial" (SAME!)
            This enables pagination - different chunks of same file = exploration, not loop.

        Context-Awareness:
        - Includes conversation stage to prevent false loop detection when the same
          operation is performed in different stages (e.g., reading a file in
          READING stage vs EXECUTING stage)

        Args:
            tool_name: Name of the tool being called
            arguments: Tool call arguments
            include_stage: Whether to include stage in signature (default True)

        Returns:
            Context-aware signature string for loop detection
        """
        canonical_tool_name = get_canonical_name(tool_name)
        params = list(get_signature_params(canonical_tool_name))
        if not params:
            params = list(FALLBACK_SIGNATURE_PARAMS.get(canonical_tool_name, ()))
        if params:
            # Use only declared signature_params - explicit positive allowlist
            sig_parts = [canonical_tool_name]
            for param in params:
                value = arguments.get(param, "")
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100]
                sig_parts.append(str(value))

            # Add stage for context-awareness
            if include_stage:
                sig_parts.append(f"stage:{self._progress.stage.value}")

            return "|".join(sig_parts)
        else:
            # For tools without signature_params, use all arguments
            args_str = str(sorted(arguments.items()))
            base_sig = f"{canonical_tool_name}:{hashlib.md5(args_str.encode()).hexdigest()[:8]}"

            # Add stage for context-awareness
            if include_stage:
                base_sig += f"|stage:{self._progress.stage.value}"

            return base_sig

    def _get_resource_key(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Generate resource key for tracking unique resources."""
        canonical = get_canonical_name(tool_name)
        if canonical == CANONICAL_READ_TOOL:
            path = arguments.get("path", "")
            offset = arguments.get("offset", 0)
            return f"file:{path}:{offset}" if path else None
        elif canonical == ToolNames.LS:
            path = arguments.get("path", "")
            return f"dir:{path}" if path else None
        elif canonical in {ToolNames.GREP, ToolNames.CODE_SEARCH}:
            query = arguments.get("query", "")
            directory = arguments.get("directory", ".")
            return f"search:{directory}:{query[:50]}" if query else None
        elif canonical == ToolNames.SHELL:
            command = arguments.get("cmd") or arguments.get("command", "")
            return f"bash:{command[:50]}" if command else None
        return None

    def _get_base_resource_key(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Generate base resource key for loop detection."""
        if tool_name == CANONICAL_LIST_TOOL:
            path = arguments.get("path", "")
            return f"dir:{path}" if path else None
        elif tool_name in {"code_search", "semantic_code_search"}:
            query = arguments.get("query", "")
            directory = arguments.get("directory", ".")
            return f"search:{directory}:{query[:20]}" if query else None
        elif tool_name == CANONICAL_SHELL_TOOL:
            command = arguments.get("cmd") or arguments.get("command", "")
            if command:
                parts = command.strip().split()
                if parts:
                    base = parts[0]
                    if base == "git" and len(parts) > 1:
                        base = f"git {parts[1]}"
                    return f"bash:{base}"
        return None

    def _get_details(self) -> Dict[str, Any]:
        """Get details dict for stop decision."""
        return {
            "task_type": self._progress.task_type.value,
            "iteration_count": self._progress.iteration_count,
            "total_turns": self._progress.total_turns,
            "tool_calls": self._progress.tool_calls,
            "tool_budget": self._progress.tool_budget,
            "milestones": [m.value for m in self._progress.milestones],
            "effective_max": self._calculate_effective_max(),
        }

    # =========================================================================
    # Backward Compatibility Methods (for orchestrator integration)
    # =========================================================================

    @property
    def config(self) -> "CompatConfig":
        """Return config-like object for backward compatibility with LoopDetector."""
        return CompatConfig(self)

    @property
    def iterations(self) -> int:
        """Alias for iteration_count (LoopDetector compatibility)."""
        return self._progress.iteration_count

    @property
    def unique_resources(self) -> Set[str]:
        """Get unique resources (LoopDetector compatibility)."""
        return self._progress.unique_resources.copy()

    def detect_task_type(self, message: str) -> TrackerTaskType:
        """Detect task type from message and configure tracker.

        Args:
            message: User message to classify

        Returns:
            Detected TrackerTaskType
        """
        from victor.storage.embeddings.task_classifier import (
            TaskType as ClassifierTaskType,
        )
        from victor.storage.embeddings.task_classifier import TaskTypeClassifier

        # Use the singleton classifier instance
        classifier = TaskTypeClassifier.get_instance()
        # Initialize synchronously if not already initialized
        classifier.initialize_sync()
        result = classifier.classify_sync(message)
        classifier_type = result.task_type

        type_map = {
            ClassifierTaskType.EDIT: TrackerTaskType.EDIT,
            ClassifierTaskType.CREATE: TrackerTaskType.CREATE,
            ClassifierTaskType.CREATE_SIMPLE: TrackerTaskType.CREATE_SIMPLE,
            ClassifierTaskType.SEARCH: TrackerTaskType.SEARCH,
            ClassifierTaskType.ANALYZE: TrackerTaskType.ANALYZE,
            ClassifierTaskType.DESIGN: TrackerTaskType.DESIGN,
            ClassifierTaskType.GENERAL: TrackerTaskType.GENERAL,
            ClassifierTaskType.GENERAL_QUERY: TrackerTaskType.GENERAL,
            # Map additional types to closest match
            ClassifierTaskType.ACTION: TrackerTaskType.GENERAL,  # Actions use general limits
            ClassifierTaskType.ANALYSIS_DEEP: TrackerTaskType.ANALYZE,  # Deep analysis uses analyze limits
        }

        task_type = type_map.get(classifier_type, TrackerTaskType.GENERAL)
        self.set_task_type(task_type)
        return task_type

    def detect_task_type_with_negation(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[TrackerTaskType, "ClassificationResult"]:
        """Detect task type using negation-aware keyword classification.

        Uses UnifiedTaskClassifier for robust classification that handles:
        - Negation patterns ("don't analyze" won't classify as analysis)
        - Positive overrides ("don't analyze, just run" classifies as action)
        - Context boosting from conversation history
        - Confidence scoring for reliability assessment

        Args:
            message: User message to classify
            history: Optional conversation history for context boosting

        Returns:
            Tuple of (TrackerTaskType, ClassificationResult) for detailed inspection
        """
        from victor.agent.unified_classifier import (
            ClassifierTaskType as UnifiedTaskType,
            get_unified_classifier,
        )

        classifier = get_unified_classifier()

        # Use context-aware classification if history provided
        if history:
            result = classifier.classify_with_context(message, history)
        else:
            result = classifier.classify(message)

        # Map UnifiedTaskClassifier.ClassifierTaskType to UnifiedTaskTracker.TrackerTaskType
        type_map = {
            UnifiedTaskType.EDIT: TrackerTaskType.EDIT,
            UnifiedTaskType.SEARCH: TrackerTaskType.SEARCH,
            UnifiedTaskType.ANALYSIS: TrackerTaskType.ANALYZE,
            UnifiedTaskType.GENERATION: TrackerTaskType.CREATE,
            UnifiedTaskType.ACTION: TrackerTaskType.GENERAL,  # ACTION maps to GENERAL for broad tool access
            UnifiedTaskType.DEFAULT: TrackerTaskType.GENERAL,
        }

        task_type = type_map.get(result.task_type, TrackerTaskType.GENERAL)

        # Use recommended tool budget from classifier
        self.set_task_type(task_type)
        if result.recommended_tool_budget:
            self.set_tool_budget(result.recommended_tool_budget)

        # Log negation info for debugging
        if result.negated_keywords:
            negated_kws = [m.keyword for m in result.negated_keywords]
            logger.info(
                f"UnifiedTaskTracker: Negated keywords detected: {negated_kws}, "
                f"final task_type={task_type.value}, confidence={result.confidence:.2f}"
            )

        return task_type, result

    def classify_with_negation_awareness(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Classify message with full negation awareness and return detailed results.

        Combines UnifiedTaskClassifier (keyword + negation) with UnifiedTaskTracker
        (budget + milestone tracking) for comprehensive task analysis.

        Args:
            message: User message to classify
            history: Optional conversation history

        Returns:
            Dictionary with classification details:
            - task_type: TrackerTaskType enum value
            - confidence: Classification confidence (0-1)
            - is_action_task: Whether task involves actions
            - is_analysis_task: Whether task involves analysis
            - negated_keywords: List of negated keyword strings
            - recommended_budget: Tool call budget
            - source: Classification source (keyword/semantic/context)
        """
        task_type, result = self.detect_task_type_with_negation(message, history)

        return {
            "task_type": task_type.value,
            "confidence": result.confidence,
            "is_action_task": result.is_action_task,
            "is_analysis_task": result.is_analysis_task,
            "is_generation_task": result.is_generation_task,
            "needs_execution": result.needs_execution,
            "negated_keywords": [m.keyword for m in result.negated_keywords],
            "matched_keywords": [m.keyword for m in result.matched_keywords],
            "recommended_budget": result.recommended_tool_budget,
            "source": result.source,
            "context_boost": result.context_boost,
        }

    def update_from_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update tracker from tool call (TaskMilestoneMonitor compatibility).

        Args:
            tool_name: Name of the tool called
            arguments: Tool arguments
            result: Optional tool result (unused, for compatibility)
        """
        # This is called by orchestrator, but record_tool_call already handles everything
        # Just update milestones if needed (record_tool_call handles most of it)
        pass  # record_tool_call already called separately

    def increment_iteration(self) -> None:
        """Increment iteration count (TaskMilestoneMonitor compatibility).

        Note: This is typically called after record_tool_call, but record_tool_call
        already increments iteration_count, so this is a no-op to avoid double counting.
        """
        # Already handled in record_tool_call
        pass

    def should_force_action(self) -> Tuple[bool, Optional[str]]:
        """Check if action should be forced (TaskMilestoneMonitor compatibility).

        Returns:
            Tuple of (should_force, hint_message)
        """
        decision = self.should_stop()
        if decision.should_stop and decision.reason in (
            StopReason.MAX_ITERATIONS,
            StopReason.GOAL_FORCING,
        ):
            return True, decision.hint
        return False, None

    def set_model_exploration_settings(
        self,
        exploration_multiplier: float = 1.0,
        continuation_patience: int = 3,
    ) -> None:
        """Set model exploration settings (TaskMilestoneMonitor compatibility).

        Alias for set_model_capabilities.
        """
        self.set_model_capabilities(exploration_multiplier, continuation_patience)


class CompatConfig:
    """Compatibility wrapper for LoopDetector's config attribute."""

    def __init__(self, tracker: UnifiedTaskTracker) -> None:
        self._tracker = tracker

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key (dict-like interface).

        Args:
            key: Configuration key name
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if key == "max_total_iterations":
            return self._tracker._max_total_iterations
        elif key == "tool_budget":
            return self._tracker._progress.tool_budget
        elif key == "remaining_tool_budget":
            return max(
                0,
                self._tracker._progress.tool_budget - self._tracker._progress.tool_calls,
            )
        return default

    @property
    def max_total_iterations(self) -> int:
        return self._tracker._max_total_iterations

    @max_total_iterations.setter
    def max_total_iterations(self, value: int) -> None:
        if self._tracker._sticky_user_iterations and not self._tracker._allow_iteration_override:
            logger.debug(
                "UnifiedTaskTracker: sticky user max iterations set; skipping auto-adjustment"
            )
            return
        self._tracker._max_total_iterations = value

    @property
    def tool_budget(self) -> int:
        return self._tracker._progress.tool_budget

    @tool_budget.setter
    def tool_budget(self, value: int) -> None:
        if self._tracker._sticky_user_budget and not self._tracker._allow_budget_override:
            logger.debug("UnifiedTaskTracker: sticky user budget set; skipping auto-adjustment")
            return
        self._tracker._progress.tool_budget = value


# =============================================================================
# Factory Functions
# =============================================================================


def create_tracker_for_task(task_type: TrackerTaskType) -> UnifiedTaskTracker:
    """Create a tracker configured for a specific task type.

    Args:
        task_type: The task type to configure for

    Returns:
        Configured UnifiedTaskTracker
    """
    tracker = UnifiedTaskTracker()
    tracker.set_task_type(task_type)
    return tracker


def create_tracker_from_message(
    message: str,
) -> Tuple[UnifiedTaskTracker, TrackerTaskType]:
    """Create a tracker by classifying a message.

    Args:
        message: User message to classify

    Returns:
        Tuple of (tracker, detected_task_type)
    """
    # Import here to avoid circular dependency
    from victor.storage.embeddings.task_classifier import TaskType as ClassifierTaskType
    from victor.storage.embeddings.task_classifier import classify_task_type

    # Classify the message
    classifier_type = classify_task_type(message)

    # Map classifier types to unified types
    type_map = {
        ClassifierTaskType.EDIT: TrackerTaskType.EDIT,
        ClassifierTaskType.CREATE: TrackerTaskType.CREATE,
        ClassifierTaskType.CREATE_SIMPLE: TrackerTaskType.CREATE_SIMPLE,
        ClassifierTaskType.SEARCH: TrackerTaskType.SEARCH,
        ClassifierTaskType.ANALYZE: TrackerTaskType.ANALYZE,
        ClassifierTaskType.DESIGN: TrackerTaskType.DESIGN,
        ClassifierTaskType.GENERAL: TrackerTaskType.GENERAL,
    }

    task_type = type_map.get(classifier_type, TrackerTaskType.GENERAL)

    tracker = UnifiedTaskTracker()
    tracker.set_task_type(task_type)

    return tracker, task_type


def create_tracker_with_negation_awareness(
    message: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[UnifiedTaskTracker, TrackerTaskType, Dict[str, Any]]:
    """Create a tracker using negation-aware keyword classification.

    This is the recommended way to create a tracker when negation detection
    is important (e.g., "don't analyze, just run the tests").

    Args:
        message: User message to classify
        history: Optional conversation history for context boosting

    Returns:
        Tuple of (tracker, task_type, classification_details)

    Example:
        tracker, task_type, details = create_tracker_with_negation_awareness(
            "Don't analyze the code, just run the tests"
        )
        print(task_type)  # TrackerTaskType.GENERAL (not ANALYZE)
        print(details["negated_keywords"])  # ["analyze"]
        print(details["matched_keywords"])  # ["run"]
    """
    tracker = UnifiedTaskTracker()
    task_type, result = tracker.detect_task_type_with_negation(message, history)

    details = {
        "task_type": task_type.value,
        "confidence": result.confidence,
        "is_action_task": result.is_action_task,
        "is_analysis_task": result.is_analysis_task,
        "is_generation_task": result.is_generation_task,
        "needs_execution": result.needs_execution,
        "negated_keywords": [m.keyword for m in result.negated_keywords],
        "matched_keywords": [m.keyword for m in result.matched_keywords],
        "recommended_budget": result.recommended_tool_budget,
        "source": result.source,
    }

    return tracker, task_type, details


def create_tracker_with_prompt_requirements(
    message: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[UnifiedTaskTracker, TrackerTaskType, Dict[str, Any]]:
    """Create a tracker using prompt requirement extraction.

    This function extracts explicit requirements from the prompt (e.g.,
    "read 9 files", "top 3 fixes") and adjusts budgets dynamically.

    Combines:
    - Negation-aware keyword classification
    - Prompt requirement extraction for dynamic budgets
    - Soft limits instead of hard stops

    Args:
        message: User message to classify
        history: Optional conversation history for context boosting

    Returns:
        Tuple of (tracker, task_type, details with prompt_requirements)

    Example:
        tracker, task_type, details = create_tracker_with_prompt_requirements(
            "Review 9 files and provide top 3 fixes"
        )
        print(details["prompt_requirements"]["file_count"])  # 9
        print(details["prompt_requirements"]["fix_count"])   # 3
        print(details["prompt_requirements"]["tool_budget"]) # 40+ (dynamic)
    """
    from victor.agent.prompt_requirement_extractor import extract_prompt_requirements

    # First, use negation-aware classification
    tracker = UnifiedTaskTracker()
    task_type, result = tracker.detect_task_type_with_negation(message, history)

    # Extract explicit requirements from prompt
    prompt_requirements = extract_prompt_requirements(message)

    # Apply dynamic budgets if requirements were found
    if prompt_requirements.has_explicit_requirements():
        # Mark that we have explicit prompt requirements (enables lenient limits)
        tracker._progress.has_prompt_requirements = True

        # Use the larger of: default budget or extracted requirement budget
        current_budget = tracker._progress.tool_budget
        if prompt_requirements.tool_budget and prompt_requirements.tool_budget > current_budget:
            tracker.set_tool_budget(prompt_requirements.tool_budget, user_override=False)
            logger.info(
                f"UnifiedTaskTracker: Dynamic budget from prompt requirements: "
                f"{prompt_requirements.tool_budget} (files={prompt_requirements.file_count}, "
                f"fixes={prompt_requirements.fix_count})"
            )

        # Also adjust max iterations if needed
        current_iterations = tracker._task_config.max_exploration_iterations
        if (
            prompt_requirements.iteration_budget
            and prompt_requirements.iteration_budget > current_iterations
        ):
            tracker.set_max_iterations(prompt_requirements.iteration_budget, user_override=False)
            logger.info(
                f"UnifiedTaskTracker: Dynamic iterations from prompt requirements: "
                f"{prompt_requirements.iteration_budget}"
            )

    details = {
        "task_type": task_type.value,
        "confidence": result.confidence,
        "is_action_task": result.is_action_task,
        "is_analysis_task": result.is_analysis_task,
        "is_generation_task": result.is_generation_task,
        "needs_execution": result.needs_execution,
        "negated_keywords": [m.keyword for m in result.negated_keywords],
        "matched_keywords": [m.keyword for m in result.matched_keywords],
        "recommended_budget": tracker._progress.tool_budget,  # Use adjusted budget
        "source": result.source,
        "prompt_requirements": prompt_requirements.to_dict(),
    }

    return tracker, task_type, details
