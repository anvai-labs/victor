"""Exploration iteration limits, recovery, and continuation prompts."""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field

from victor.core.loop_thresholds import (
    DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD,
    DEFAULT_BLOCKED_TOTAL_THRESHOLD,
)


class ExplorationSettings(BaseModel):
    """Exploration iteration limits, recovery, and continuation prompts.

    Iteration caps 5x'd 2026-10-31: agents run longer autonomous sessions.
    Stuck-agent / continuation-prompt guards below are intentionally unchanged.
    """

    max_exploration_iterations: int = 40
    max_exploration_iterations_action: int = 60
    max_exploration_iterations_analysis: int = 250
    chat_max_iterations: int = 250
    max_consecutive_tool_calls: int = 20
    max_research_iterations: int = 30
    min_content_threshold: int = 150
    recovery_empty_response_threshold: int = 5
    recovery_blocked_consecutive_threshold: int = DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD
    recovery_blocked_total_threshold: int = DEFAULT_BLOCKED_TOTAL_THRESHOLD
    max_continuation_prompts_default: int = 3
    max_continuation_prompts_action: int = 5
    max_continuation_prompts_analysis: int = 6
    continuation_prompt_overrides: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    enable_continuation_rl_learning: bool = True

    # Convergence guards (shared TurnEvaluationController; both agentic loops).
    # Search-novelty safety-net: stop a re-search loop that is no longer surfacing new files.
    search_novelty_guard_enabled: bool = True
    novelty_ratio_threshold: float = 0.34
    novelty_consecutive_low_limit: int = 3
    novelty_nudge_after_low: int = 2
    novelty_min_search_turns: int = 2
    min_iterations_before_force_complete: int = 2
    # Fulfillment-complete: stop redundant verification once a substantial answer + findings exist.
    fulfillment_completion_enabled: bool = True
    fulfillment_summary_min_chars: int = 800
    fulfillment_min_findings: int = 3
    min_iterations_before_fulfillment: int = 4
