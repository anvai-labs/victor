"""Tool execution, selection, and retry configuration."""

from __future__ import annotations

import os
from typing import Dict, List, Set

from pydantic import BaseModel, Field

from victor.config.model_capabilities import _load_tool_capable_patterns_from_yaml
from victor.config.orchestrator_constants import BUDGET_LIMITS


class ToolSettings(BaseModel):
    """Tool execution, selection, and retry configuration."""

    tool_call_budget: int = Field(default_factory=lambda: BUDGET_LIMITS.max_session_budget)
    tool_call_budget_warning_threshold: int = Field(
        default_factory=lambda: int(
            BUDGET_LIMITS.max_session_budget * BUDGET_LIMITS.warning_threshold_pct
        )
    )
    tool_call_budget_warning_pct: float = Field(
        default=0.8,
        ge=0.5,
        le=0.99,
        description="Warn when tool usage reaches this fraction of the turn budget",
    )
    tool_call_budget_warning_remaining: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Warn when this many or fewer tool calls remain in the current turn",
    )
    tool_budget_progress_relief_enabled: bool = Field(
        default=True,
        description="Allow a bounded one-time budget extension when action tasks show progress",
    )
    tool_budget_progress_relief_amount: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of extra tool calls to grant for progress-based relief",
    )
    tool_budget_progress_relief_max_uses: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Maximum number of progress-based budget relief grants per turn",
    )
    tool_budget_calibration_enabled: bool = Field(
        default=False,
        description=(
            "Opt-in RL-driven tool-budget calibration. When True, the session "
            "overlay is derived from aggregated rl_tool_q + decision_outcome "
            "signals via BudgetCalibrator. Default False = unchanged behavior; "
            "low-confidence recommendations always retain the baseline."
        ),
    )
    tool_budget_calibration_min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum calibrator confidence required to apply a calibrated "
            "overlay; below this the baseline settings are returned unchanged. "
            "Default 0.7 per FEP-0002 review (see Open Questions #1)."
        ),
    )
    tool_calling_models: Dict[str, list[str]] = Field(
        default_factory=_load_tool_capable_patterns_from_yaml
    )
    tool_retry_enabled: bool = True
    tool_retry_max_attempts: int = 3
    tool_retry_base_delay: float = 1.0
    tool_retry_max_delay: float = 10.0
    fallback_max_tools: int = 8
    enable_tool_deduplication: bool = True
    tool_deduplication_window_size: int = 20
    use_semantic_tool_selection: bool = True
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    tool_cache_enabled: bool = True
    tool_cache_ttl: int = 600
    tool_cache_allowlist: List[str] = Field(
        default_factory=lambda: [
            "read",
            "ls",
            "grep",
            "overview",
            "diff",
            "code_search",
            "semantic_code_search",
            "list_directory",
            "plan_files",
        ]
    )
    generic_result_cache_enabled: bool = False
    generic_result_cache_ttl: int = 300
    tool_selection_cache_enabled: bool = True
    tool_selection_cache_ttl: int = 300
    tool_validation_mode: str = "lenient"
    # Token budget for tool schemas broadcast to LLM.
    # When total estimated tokens exceed this budget, lowest-ranked tools
    # are demoted (COMPACT→STUB) or dropped (tail STUBs removed).
    # Default 4000 is a safe ceiling: with fallback_max_tools=8, the tiered
    # selection typically uses ~650 tokens, so this only triggers if someone
    # raises fallback_max_tools significantly. Set 0 to disable enforcement.
    max_tool_schema_tokens: int = 4000
    # Promote high-confidence semantic matches from STUB→COMPACT.
    # Tools with semantic similarity >= this threshold get richer schemas.
    schema_promotion_threshold: float = 0.8
    # Maximum MCP tools to broadcast per turn (when relevance filtering is active)
    max_mcp_tools_per_turn: int = 12
    # Enable cross-turn tool result deduplication (session-scoped cache)
    cross_turn_dedup_enabled: bool = True
    cross_turn_dedup_ttl: int = 300  # seconds

    # Tool output preview configuration (user display only, does NOT affect LLM)
    tool_output_preview_enabled: bool = Field(
        default=True,
        description="Show tool output preview to user (default: yes for transparency)",
    )
    tool_output_preview_lines: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of lines to show in tool output preview (default: 3)",
    )
    tool_output_pruning_enabled: bool = Field(
        default=True,
        description="[DEPRECATED - Now only affects user preview, NOT LLM input] Enable tool output pruning for user display",
    )
    tool_output_pruning_safe_only: bool = Field(
        default=True,
        description="Prune only safe verbose read-heavy tool outputs by default; set false for broader opt-in pruning",
    )
    tool_output_show_transparency: bool = Field(
        default=True,
        description="Show preview status to user when output was truncated for display (default: yes)",
    )
    tool_output_expand_hotkey: str = Field(
        default="^O",
        description="Hotkey to expand tool output preview (default: Ctrl+O, format: ^X for Ctrl+X)",
    )

    # Tool output byte limit for display
    # This is separate from pruning and affects what users see in terminal/UI
    # When pruning is disabled (accuracy-first), byte limit should be higher
    tool_output_byte_limit_mb: float = Field(
        default=10.0,  # 10MB default (much higher when accuracy matters)
        ge=1.0,  # Minimum 1MB
        le=100.0,  # Maximum 100MB
        description="Maximum tool output size in MB for display (default: 10MB, higher when pruning disabled)",
    )

    # Shell output condensation (rtk-style, applies to the LLM path)
    # Command-aware structural compression of shell output: pytest failures+summary,
    # git status grouping, install/build noise stripping. Diagnostic content is
    # preserved verbatim; raw output is teed to .victor/tool_output/ for recovery.
    shell_output_condensation_enabled: bool = Field(
        default=True,
        description="Condense recognized shell command output (pytest, git, pip/npm/cargo) before LLM injection; fail-open passthrough on parse anomalies",
    )
    shell_output_raw_tee_enabled: bool = Field(
        default=True,
        description="Tee raw shell output to .victor/tool_output/ when condensed, so the full log stays retrievable",
    )
    shell_frugal_flags_enabled: bool = Field(
        default=True,
        description="Rewrite recognized commands to quieter forms before execution (e.g. pytest -q --tb=short -rxX); explicit user flags are never overridden",
    )

    # Adaptive preview sizing (context-aware)
    tool_output_preview_adaptive: bool = Field(
        default=True,
        description="Adjust preview size based on content type and size (default: enabled)",
    )
    tool_output_preview_lines_min: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Minimum preview lines for adaptive mode (default: 1)",
    )
    tool_output_preview_lines_max: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum preview lines for adaptive mode (default: 10)",
    )

    # Tool grouping for content organization
    enable_tool_grouping: bool = Field(
        default=True,
        description=(
            "DEPRECATED (no-op): per-result category group headers were removed "
            "from the live renderer — results arrive in execution order, so "
            "arrival-order grouping grouped nothing. Field kept for config "
            "compatibility."
        ),
    )

    tool_progress_streaming_enabled: bool = Field(
        default=True,
        description=(
            "Stream partial stdout/stderr from long-running tools (shell, "
            "code_search) into a live terminal block while they run (default: "
            "enabled). UI-only; never affects what the model sees."
        ),
    )

    # Rich formatting configuration
    rich_formatting_enabled: bool = Field(
        default=True,
        description="Enable Rich markup formatting for tool outputs (default: True)",
    )
    rich_formatting_tools: List[str] = Field(
        default_factory=lambda: [
            "test",
            "pytest",
            "run_tests",
            "code_search",
            "semantic_code_search",
            "git",
            "http",
            "https",
            "database",
            "db",
            "sql",
            "refactor",
            "refactoring",
            "docker",
            "security",
            "security_scan",
            # Phase 7: Additional formatters
            "shell",
            "bash",
            "exec",
            "filesystem",
            "ls",
            "find",
            "cat",
            "read",
            "overview",
            "network",
            "ping",
            "traceroute",
            "dns",
            "build",
            "make",
            "cmake",
            "cargo",
            "npm",
            "pip",
        ],
        description="Tools that use Rich formatting (default: major tools with formatters)",
    )
    rich_formatting_max_output_size: int = Field(
        default=1_000_000,
        ge=100_000,
        le=10_000_000,
        description="Maximum formatted output size in bytes (default: 1MB)",
    )
    rich_formatting_max_time_ms: int = Field(
        default=200,
        ge=50,
        le=5000,
        description="Maximum formatting time in milliseconds (default: 200ms)",
    )
    rich_formatting_cache_enabled: bool = Field(
        default=True,
        description="Enable caching of formatted outputs (default: True)",
    )
    rich_formatting_cache_ttl: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Formatted output cache TTL in seconds (default: 300s = 5 minutes)",
    )
    rich_formatting_cache_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum number of formatted outputs to cache (default: 100)",
    )
    rich_formatting_fallback_enabled: bool = Field(
        default=True,
        description="Enable fallback to plain text on formatting errors (default: True)",
    )
    rich_formatting_validation_enabled: bool = Field(
        default=True,
        description="Enable input validation before formatting (default: True)",
    )

    # Embedding-intensive tool concurrency configuration
    # These tools use embedding models and need lower concurrency limits
    # to prevent resource exhaustion (memory, CPU, embedding model contention)
    max_embedding_concurrent: int = Field(
        default_factory=lambda: int(os.getenv("VICTOR_MAX_EMBEDDING_CONCURRENT", "2")),
        ge=1,
        le=10,
        description="Maximum concurrent embedding-intensive tool executions (default: 2, env: VICTOR_MAX_EMBEDDING_CONCURRENT)",
    )
    embedding_intensive_tools: Set[str] = Field(
        default_factory=lambda: set(
            os.getenv(
                "VICTOR_EMBEDDING_INTENSIVE_TOOLS",
                "code_search,semantic_code_search",
            ).split(",")
        ),
        description="Tool names that require embedding concurrency limits (default: code_search,semantic_code_search, env: VICTOR_EMBEDDING_INTENSIVE_TOOLS)",
    )

    # Tool deduplication configuration
    # Unified deduplication across native, LangChain, and MCP tools
    enable_tool_deduplication: bool = Field(
        default=True,
        description="Enable cross-source tool deduplication (default: True)",
    )
    deduplication_priority_order: List[str] = Field(
        default_factory=lambda: ["native", "langchain", "mcp", "plugin"],
        description="Priority order for tool sources (highest to lowest) (default: native,langchain,mcp,plugin)",
    )
    deduplication_whitelist: List[str] = Field(
        default_factory=list,
        description="Tools to always allow (bypass deduplication) (default: [])",
    )
    deduplication_blacklist: List[str] = Field(
        default_factory=list,
        description="Tools to always skip (force deduplication) (default: [])",
    )
    deduplication_naming_enforcement: bool = Field(
        default=True,
        description="Enforce naming conventions (lgc_*, mcp_*, plg_*) (default: True)",
    )
    deduplication_semantic_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Threshold for semantic similarity detection (0.0-1.0) (default: 0.85)",
    )

    # Provider-specific tool broadcasting optimization
    cloud_core_tool_set: str = Field(
        default="default",
        description="Cloud provider core tool set: default, expanded, or minimal (default: default)",
    )
    local_core_tool_set: str = Field(
        default="minimal",
        description="Local provider core tool set: minimal or expanded (default: minimal)",
    )
    schema_optimization_level: str = Field(
        default="aggressive",
        description="Schema optimization level: aggressive, moderate, or conservative (default: aggressive)",
    )
    max_dynamic_tools_local: int = Field(
        default=12,
        ge=0,
        le=20,
        description="Maximum dynamic tools per turn for local providers (default: 12)",
    )
    min_dynamic_tools_local: int = Field(
        default=8,
        ge=0,
        le=15,
        description="Minimum dynamic tools per turn for local providers (default: 8)",
    )


def get_tool_settings() -> ToolSettings:
    """Get the current tool settings from the global settings singleton.

    Returns:
        ToolSettings instance with current configuration

    Example:
        >>> from victor.config.tool_settings import get_tool_settings
        >>> settings = get_tool_settings()
        >>> if settings.tool_output_preview_enabled:
        ...     show_preview()
    """
    from victor.config.settings import get_settings

    global_settings = get_settings()
    if global_settings.tool_settings is None:
        # Return default ToolSettings if not configured
        return ToolSettings()
    return global_settings.tool_settings
