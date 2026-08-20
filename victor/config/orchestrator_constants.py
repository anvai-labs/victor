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

"""Centralized constants for orchestrator components.

This module provides dataclasses containing magic numbers and thresholds
used throughout the Victor framework. Centralizing these values makes the
codebase more maintainable and easier to tune.

**Design Rationale:**
Previously, threshold values were scattered across multiple files (context_compactor.py,
conversation_controller.py, etc.), making it difficult to understand and tune the system.
This module provides a single source of truth for all orchestrator-related constants.

**Usage:**
    from victor.config.orchestrator_constants import (
        CONTEXT_LIMITS,
        SEMANTIC_THRESHOLDS,
        BUDGET_LIMITS,
        TOOL_SELECTION_PRESETS,
    )

    # Check context overflow
    if utilization >= CONTEXT_LIMITS.overflow_threshold:
        compact_history()

    # Filter by semantic similarity
    if similarity >= SEMANTIC_THRESHOLDS.tool_selection:
        include_tool()

**Note:** These constants are defaults. Most can be overridden via Settings or ProfileConfig.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ContextLimits:
    """Context window utilization thresholds.

    These thresholds control when context compaction and warnings are triggered.

    Attributes:
        overflow_threshold: Utilization % to trigger overflow warnings (0.0-1.0)
        warning_threshold: Utilization % to show warnings in UI (0.0-1.0)
        proactive_compaction_threshold: Utilization % to trigger proactive compaction (0.0-1.0)
        critical_threshold: Utilization % indicating critical overflow risk (0.0-1.0)
        compaction_target: Target utilization after compaction (0.0-1.0)
        max_context_chars: Default maximum context size in characters
        chars_per_token_estimate: Approximate characters per token for estimation

    **Design Notes:**
    - max_context_chars keeps the long-standing 800K-character default for compatibility
      while still targeting modern 128K-200K token models. This is intentionally a little
      generous because multiple runtime components and tests assume the historical default.
    - overflow_threshold (0.85) triggers at ~170K tokens on a 200K model, leaving 30K headroom.
    - proactive_compaction_threshold (0.75) triggers at ~150K tokens — conservative to avoid
      overflows due to inaccurate token estimation (different parts of system use 3.0-4.0).
    - warning_threshold (0.65) warns at ~130K tokens, giving user advance notice.
    - compaction_target (0.45) compacts down to ~90K tokens — room for continued growth.
    - chars_per_token_estimate preserves the historical rounded value of 4 for
      compatibility with conversation budgeting tests and existing heuristics.
    """

    overflow_threshold: float = 0.85
    warning_threshold: float = 0.65
    proactive_compaction_threshold: float = (
        0.75  # Reduced from 0.85 to avoid overflows due to token estimation inaccuracy
    )
    critical_threshold: float = 0.95
    compaction_target: float = 0.45
    # Compatibility-preserving default for modern large-context models.
    # Override via VICTOR_MAX_CONTEXT_CHARS env var or profiles.yaml if using a smaller model.
    max_context_chars: int = 800000
    chars_per_token_estimate: int = 4


@dataclass(frozen=True)
class SemanticThresholds:
    """Semantic similarity thresholds for various operations.

    These thresholds control when content is considered "similar enough" for
    different semantic operations throughout the framework.

    Attributes:
        tool_selection: Minimum similarity to include a tool in the selection
        code_search: Minimum similarity for code search results
        history_retrieval: Minimum similarity for retrieving historical messages
        compaction_relevance: Minimum similarity for message retention during compaction

    **Design Notes:**
    - tool_selection (0.3) is lenient to avoid missing useful tools
    - code_search (0.5) is stricter to reduce false positives
    - history_retrieval (0.4) balances context preservation with relevance
    - compaction_relevance (0.3) matches conversation_controller.py
    """

    tool_selection: float = 0.3
    code_search: float = 0.5
    history_retrieval: float = 0.4
    compaction_relevance: float = 0.3


@dataclass(frozen=True)
class BudgetLimits:
    """Tool budget limits by task complexity.

    These budgets control how many tool calls are allowed based on the
    detected task type. Higher budgets allow more exploratory behavior.

    Attributes:
        simple_task: Budget for simple tasks (quick queries, single file edits)
        medium_task: Budget for medium tasks (multi-file analysis, refactoring)
        complex_task: Budget for complex tasks (architecture changes, migrations)
        action_task: Budget for action-heavy tasks (building features, large refactors)
        analysis_task: Budget for analysis-heavy tasks (codebase exploration, audits)
        max_session_budget: Maximum tool calls per session (hard limit)
        warning_threshold_pct: % of budget used to trigger warnings (0.0-1.0)

    **Design Notes:**
    - Budgets scale exponentially with complexity (2 → 6 → 15 → 50 → 60)
    - action_task (50) allows building non-trivial features
    - analysis_task (60) supports deep codebase exploration
    - max_session_budget (300) prevents runaway tool usage
    - warning_threshold_pct (0.83) gives early warning at 250/300 calls
    """

    # Significantly increased to match Claude Code's approach of unlimited exploration
    # Claude Code lets the model explore as much as needed to complete the task
    simple_task: int = 20  # Quick tasks still bounded (was 5)
    medium_task: int = 50  # Multi-file tasks need room (was 15)
    complex_task: int = 100  # Complex tasks get generous budget (was 30)
    action_task: int = 200  # BUILD mode tasks - let task completion decide (was 75)
    analysis_task: int = 500  # PLAN/EXPLORE modes - effectively unlimited (was 100)
    max_session_budget: int = 2000  # Very high - rely on task completion detection (was 500)
    warning_threshold_pct: float = 0.90  # Only warn at 90% (1800/2000)


@dataclass(frozen=True)
class CompactionConfig:
    """Configuration for context compaction strategies.

    Attributes:
        min_messages_after_compact: Minimum messages to keep after compaction
        tool_result_retention_weight: Weight multiplier for tool results (higher = keep longer)
        recent_message_weight: Weight multiplier for recent messages (higher = keep longer)
        tool_result_max_chars: Maximum characters for tool results before truncation
        tool_result_max_lines: Maximum lines for tool results before truncation
        output_reserve_pct: % of context to reserve for output generation (0.0-1.0)
        parallel_read_target_files: Target number of parallel file reads to optimize for
        chars_per_token: Approximate characters per token for calculations
        enable_fast_pruning: Enable fast pruning before LLM compaction (P1: OpenDev finding)
        enable_proactive: Enable proactive compaction at threshold (not just overflow)

    **Design Notes:**
    - min_messages_after_compact (6) preserves enough context for coherent conversation
    - tool_result_retention_weight (1.2) moderate priority for tool results
    - recent_message_weight (2.0) heavily weights recent messages
    - tool_result_max_chars (5120) allows ~12-15 parallel reads within 32K usable tokens
    - output_reserve_pct (0.5) reserves half the context for model output
    - parallel_read_target_files (12) optimizes for common read patterns
    - enable_fast_pruning (True) reduces LLM compaction cost by 30-40%
    - chars_per_token (3.5) standardized across all modules (was 3.0, inconsistent)
    """

    min_messages_after_compact: int = 6  # Reduced from 8
    tool_result_retention_weight: float = 1.2  # Reduced from 1.5
    recent_message_weight: float = 2.0
    tool_result_max_chars: int = 5120  # Reduced from 8192 (~37% reduction)
    tool_result_max_lines: int = 150  # Reduced from 230 (~35% reduction)
    output_reserve_pct: float = 0.5
    parallel_read_target_files: int = 12  # Increased from 10
    chars_per_token: float = 3.5  # Standardized from 3.0 (matches ContextLimits)
    # P1 FIX: Fast pruning before LLM compaction (OpenDev finding)
    enable_fast_pruning: bool = True
    enable_proactive: bool = False
    # L1: reference-aware tool-result pruning. Prunes old, large tool results that no later
    # assistant turn cites; preserves the rest. Default ON — flipped after the evidence sweep
    # showed it is lossless (drops 0 cited results vs size-based pruning dropping every cited
    # result) and recovers most of the savings (-62% to -78% in the realistic read-many/cite-few
    # regime), while fixing the reference-blind size-based path that silently dropped cited tool
    # output at compaction. Replaces size-based fast-pruning when active (see ContextCompactor).
    enable_reference_aware_pruning: bool = True
    reference_window_turns: int = 3


# Task-specific compaction configurations
# Research tasks need more context preservation than other tasks
TASK_COMPACTION_CONFIGS: Dict[str, CompactionConfig] = {
    "research": CompactionConfig(
        min_messages_after_compact=100,  # Keep 100+ messages (vs 6 default)
        tool_result_retention_weight=3.0,  # High priority on tool results (vs 1.2)
        recent_message_weight=1.5,  # Lower priority on recency (vs 2.0)
        tool_result_max_chars=10240,  # Allow larger tool results (vs 5120)
        tool_result_max_lines=300,  # Allow more lines (vs 150)
        output_reserve_pct=0.4,  # Reserve less for output, more for context
        parallel_read_target_files=20,  # Optimize for more parallel reads
        chars_per_token=3.0,
    ),
    # Add other task types as needed
    # "coding": CompactionConfig(...),
    # "analysis": CompactionConfig(...),
}


@dataclass(frozen=True)
class ToolSelectionPresets:
    """Presets for adaptive tool selection by model size.

    These presets configure semantic tool selection thresholds based on
    model size/capability. Smaller models get stricter filtering to avoid
    overwhelming them with tool options.

    **Design Notes:**
    - tiny (0.5B-3B): Strict filtering, minimal tools to avoid confusion
    - small (7B-8B): Moderate filtering, suitable for most tasks
    - medium (13B-15B): Lenient filtering, can handle more tools
    - large (30B+): Very lenient, can handle complex tool sets
    - cloud (Claude/GPT): Optimized for commercial models
    """

    tiny: Dict[str, float] = None  # Will be set in __post_init__
    small: Dict[str, float] = None
    medium: Dict[str, float] = None
    large: Dict[str, float] = None
    cloud: Dict[str, float] = None

    def __post_init__(self):
        # Use object.__setattr__ since dataclass is frozen
        object.__setattr__(
            self,
            "tiny",
            {
                "base_threshold": 0.35,
                "base_max_tools": 5,
            },
        )
        object.__setattr__(
            self,
            "small",
            {
                "base_threshold": 0.25,
                "base_max_tools": 7,
            },
        )
        object.__setattr__(
            self,
            "medium",
            {
                "base_threshold": 0.20,
                "base_max_tools": 10,
            },
        )
        object.__setattr__(
            self,
            "large",
            {
                "base_threshold": 0.15,
                "base_max_tools": 12,
            },
        )
        object.__setattr__(
            self,
            "cloud",
            {
                "base_threshold": 0.18,
                "base_max_tools": 10,
            },
        )


@dataclass(frozen=True)
class StageToolLimits:
    """Maximum tools per conversation stage.

    Controls how many tools are broadcast to the LLM at each stage.
    Higher limits for execution stages, lower for exploration stages.

    **Design Notes:**
    - INITIAL/READING: Lower limit since only read tools needed
    - EXECUTING: Higher limit to allow write/edit/test tools
    - BUILD mode uses executing_max for all stages
    - These limits are applied AFTER semantic/keyword selection
    """

    initial_max: int = 10  # Early exploration - focus on read tools
    planning_max: int = 10  # Planning stage
    reading_max: int = 10  # Reading/analysis
    analysis_max: int = 12  # Analysis may need more tools
    executing_max: int = 15  # Execution needs write/edit/shell/git/test
    verification_max: int = 12  # Verification stage
    completion_max: int = 8  # Completion - minimal tools
    build_mode_max: int = 15  # BUILD mode override


STAGE_TOOL_LIMITS = StageToolLimits()


@dataclass(frozen=True)
class SemanticCacheConfig:
    """Configuration for semantic tool result cache (FAISS-based).

    Attributes:
        max_entries: Maximum cache entries before LRU eviction
        cleanup_interval: Seconds between TTL cleanup runs
        default_ttl: Default TTL for cached results (seconds)
        enabled: Whether semantic caching is enabled

    **Design Notes:**
    - max_entries (500) balances memory usage vs cache effectiveness
    - cleanup_interval (60) provides regular cleanup without overhead
    - Uses mtime-based invalidation for file content, TTL as fallback
    """

    max_entries: int = 500
    cleanup_interval: float = 60.0
    default_ttl: int = 3600  # 1 hour (mtime handles freshness)
    enabled: bool = True


@dataclass(frozen=True)
class RLLearnerConfig:
    """Configuration for RL-based context pruning learner.

    Attributes:
        learning_rate: Q-value update rate (alpha)
        discount_factor: Future reward discount (gamma)
        enabled: Whether RL-based pruning is enabled
        min_samples: Minimum samples before confident recommendation
        prior_alpha: Thompson Sampling prior alpha (Beta distribution)
        prior_beta: Thompson Sampling prior beta (Beta distribution)

    **Design Notes:**
    - learning_rate (0.15) provides moderate learning speed
    - discount_factor (0.90) values future rewards appropriately
    - Uses Thompson Sampling for exploration-exploitation balance
    """

    learning_rate: float = 0.15
    discount_factor: float = 0.90
    enabled: bool = True
    min_samples: int = 10
    prior_alpha: float = 1.0
    prior_beta: float = 1.0


SEMANTIC_CACHE_CONFIG = SemanticCacheConfig()
RL_LEARNER_CONFIG = RLLearnerConfig()


@dataclass(frozen=True)
class SessionLedgerConfig:
    """Configuration for session ledger (structured session state).

    Attributes:
        max_entries: Maximum ledger entries before oldest-first eviction
        max_render_chars: Hard cap on rendered ledger output
        file_summary_max_len: Max chars for per-file summary strings
    """

    max_entries: int = 200
    max_render_chars: int = 3000
    file_summary_max_len: int = 80


@dataclass(frozen=True)
class SummarizationConfig:
    """Turn-based summarization compaction configuration.

    When conversation exceeds trigger_turn_count, oldest turns are
    summarized into a single compact message using a fast edge model.
    Original messages remain in SQLite for audit but are removed from
    the active context window.

    Attributes:
        enabled: Whether summarization compaction is active
        trigger_turn_count: Summarize after this many turns in the conversation
        max_turns_to_summarize: Number of oldest turns to summarize at once
        summary_max_tokens: Maximum tokens for the summary output
        summary_timeout_seconds: Timeout for the edge model summarization call
    """

    enabled: bool = True
    trigger_turn_count: int = 15
    max_turns_to_summarize: int = 10
    summary_max_tokens: int = 400
    summary_timeout_seconds: float = 8.0


SUMMARIZATION_CONFIG = SummarizationConfig()


@dataclass(frozen=True)
class DeduplicationConfig:
    """Configuration for tool result deduplication.

    Attributes:
        enabled: Whether deduplication is active
        stub_template: Template for replacement stubs
        dedup_tool_names: Tool names considered as read operations
        min_content_chars_to_dedup: Minimum content size to trigger dedup
    """

    enabled: bool = True
    stub_template: str = "[Previously read: {path} -- {lines} lines]"
    dedup_tool_names: tuple = (
        "read",
        "cat",
        "read_file",
        "head",
        "tail",  # File reads
        "ls",
        "glob",
        "find",  # Directory listings
        "code_search",
        "grep",
        "search",  # Search results
    )
    min_content_chars_to_dedup: int = 500


@dataclass(frozen=True)
class ContextAssemblerConfig:
    """Configuration for turn-boundary context assembly.

    Attributes:
        history_budget_pct: Fraction of max_context_chars for history
        response_reserve_pct: Fraction reserved for response generation
        full_turn_count: Number of recent turns kept in full
        min_scored_messages: Minimum older messages to score-select
        ledger_budget_pct: Fraction of max_context_chars for ledger rendering
    """

    history_budget_pct: float = 0.70
    response_reserve_pct: float = 0.30
    full_turn_count: int = 3
    min_scored_messages: int = 4
    ledger_budget_pct: float = 0.05


SESSION_LEDGER_CONFIG = SessionLedgerConfig()
DEDUPLICATION_CONFIG = DeduplicationConfig()
CONTEXT_ASSEMBLER_CONFIG = ContextAssemblerConfig()


# ============================================================================
# Singleton instances (read-only)
# ============================================================================
# These are the default values used throughout Victor. Import these instances
# directly rather than creating new ones.

CONTEXT_LIMITS = ContextLimits()
SEMANTIC_THRESHOLDS = SemanticThresholds()
BUDGET_LIMITS = BudgetLimits()
COMPACTION_CONFIG = CompactionConfig()
TOOL_SELECTION_PRESETS = ToolSelectionPresets()


# ============================================================================
# Convenience functions
# ============================================================================


def get_budget_for_task(task_type: str) -> int:
    """Get recommended tool budget for a task type.

    Args:
        task_type: Task type (simple, medium, complex, action, analysis)

    Returns:
        Recommended tool budget

    Raises:
        ValueError: If task_type is not recognized
    """
    task_type_lower = task_type.lower()

    if task_type_lower in ("simple", "quick", "trivial"):
        return BUDGET_LIMITS.simple_task
    elif task_type_lower in ("medium", "moderate", "standard"):
        return BUDGET_LIMITS.medium_task
    elif task_type_lower in ("complex", "difficult", "advanced"):
        return BUDGET_LIMITS.complex_task
    elif task_type_lower in ("action", "build", "implement", "feature"):
        return BUDGET_LIMITS.action_task
    elif task_type_lower in ("analysis", "explore", "research", "audit"):
        return BUDGET_LIMITS.analysis_task
    else:
        raise ValueError(
            f"Unknown task type: {task_type}. "
            f"Valid types: simple, medium, complex, action, analysis"
        )


def is_budget_warning_threshold(used: int, total: int) -> bool:
    """Check if tool budget usage has reached warning threshold.

    Args:
        used: Number of tool calls used
        total: Total tool budget

    Returns:
        True if warning threshold reached
    """
    if total <= 0:
        return False
    utilization = used / total
    return utilization >= BUDGET_LIMITS.warning_threshold_pct


def should_compact_context(utilization: float, proactive: bool = True) -> bool:
    """Determine if context should be compacted based on utilization.

    Args:
        utilization: Current context utilization (0.0-1.0)
        proactive: If True, use proactive threshold; otherwise use overflow threshold

    Returns:
        True if compaction should be triggered
    """
    if proactive:
        return utilization >= CONTEXT_LIMITS.proactive_compaction_threshold
    else:
        return utilization >= CONTEXT_LIMITS.overflow_threshold
