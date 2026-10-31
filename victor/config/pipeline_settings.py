"""Runtime-intelligence settings, quality scoring, and recovery."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, Field

from victor.core.loop_thresholds import (
    DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD,
    DEFAULT_BLOCKED_TOTAL_THRESHOLD,
)


class PipelineSettings(BaseModel):
    """Runtime-intelligence settings, quality scoring, and recovery."""

    # =================================================================
    # Decision Chain Configuration
    # =================================================================
    # Unified fallback chain for ALL decision points in the agentic loop.
    # Each decision type maps to an ordered list of strategies:
    #   "heuristic" — fast keyword/pattern/rule-based (default first)
    #   "llm"       — LLM decision service via edge or main model
    #
    # The chain tries strategies in order. If a strategy returns a result
    # with confidence >= its threshold, it's used. Otherwise, next in chain.
    #
    # Default: heuristic-first for all decisions (proven at 60% SWE-bench).
    # Override per decision type or globally via decision_chain_default.
    #
    # Example ~/.victor/config.yaml:
    #   pipeline:
    #     decision_chain_default: ["heuristic", "llm"]
    #     decision_chain:
    #       stage_detection: ["heuristic", "llm"]
    #       tool_selection: ["heuristic"]        # no LLM for tools
    #       task_completion: ["llm", "heuristic"] # LLM-first for completion
    # =================================================================

    # Global default chain — applied to any decision type not in decision_chain
    decision_chain_default: List[str] = Field(default=["heuristic", "llm"])

    # Per-decision-type overrides (keys match DecisionType enum values)
    decision_chain: Dict[str, List[str]] = Field(default_factory=dict)

    # Heuristic confidence threshold — below this, fallback to next in chain
    heuristic_confidence_threshold: float = 0.7

    # Parallel exploration — spawn concurrent subagents during READING stage
    parallel_exploration: bool = True
    max_exploration_agents: int = 3
    exploration_tool_budget: int = 50
    exploration_timeout: int = 90  # seconds (increase for local models)

    runtime_intelligence_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "runtime_intelligence_enabled",
            "intelligent_pipeline_enabled",
        ),
    )
    intelligent_quality_scoring: bool = True
    intelligent_mode_learning: bool = True
    intelligent_prompt_optimization: bool = True
    intelligent_grounding_verification: bool = True
    intelligent_min_quality_threshold: float = 0.5
    intelligent_grounding_threshold: float = 0.7
    intelligent_exploration_rate: float = 0.3
    intelligent_learning_rate: float = 0.1
    intelligent_discount_factor: float = 0.9
    serialization_enabled: bool = True
    serialization_default_format: Optional[str] = None
    serialization_min_savings_threshold: float = 0.15
    serialization_include_format_hint: bool = True
    serialization_min_rows_for_tabular: int = 3
    serialization_debug_mode: bool = False
    # 5x'd 2026-10-31: agents run longer autonomous sessions
    max_exploration_iterations: int = 1000
    max_exploration_iterations_action: int = 2500
    max_exploration_iterations_analysis: int = 5000
    min_content_threshold: int = 50
    max_research_iterations: int = 250
    recovery_empty_response_threshold: int = 5
    recovery_blocked_consecutive_threshold: int = DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD
    recovery_blocked_total_threshold: int = DEFAULT_BLOCKED_TOTAL_THRESHOLD
    max_continuation_prompts_analysis: int = 6
    max_continuation_prompts_action: int = 5
    max_continuation_prompts_default: int = 3
    continuation_prompt_overrides: dict = Field(default_factory=dict)
    enable_continuation_rl_learning: bool = True
    session_idle_timeout: int = 180


def resolve_runtime_intelligence_enabled(settings: Any, default: bool = True) -> bool:
    """Resolve the canonical runtime-intelligence gate from nested or legacy settings."""
    pipeline_settings = getattr(settings, "pipeline", None)
    if pipeline_settings is not None and hasattr(pipeline_settings, "runtime_intelligence_enabled"):
        return bool(pipeline_settings.runtime_intelligence_enabled)

    if hasattr(settings, "runtime_intelligence_enabled"):
        return bool(settings.runtime_intelligence_enabled)

    if hasattr(settings, "intelligent_pipeline_enabled"):
        return bool(settings.intelligent_pipeline_enabled)

    return default
