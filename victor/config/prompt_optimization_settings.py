"""Prompt optimization configuration.

Strategy-agnostic settings for evolving system prompt sections at runtime.
GEPA is the default (and currently only) strategy, but the design supports
future strategies via the section_strategies mapping.

Configuration hierarchy:
  prompt_optimization.enabled          → master switch (all strategies)
  prompt_optimization.default_strategies → ["gepa"] applied to all sections
  prompt_optimization.section_strategies → per-section overrides
  prompt_optimization.gepa.*           → GEPA-specific config (tiers, bloat, etc.)

Section strategy resolution:
  1. If section listed in section_strategies → use that list
  2. Else → use default_strategies
  3. Empty list [] → skip optimization for that section
  4. List with entries → apply strategies in order (layered)
"""

from __future__ import annotations

import os
from typing import Dict, List

from pydantic import BaseModel, Field

from victor.config.gepa_settings import GEPASettings


def frozen_prompts_enabled() -> bool:
    """Whether this process requests static baseline prompt guidance.

    Set before creating sessions. Preserve the existing nonempty environment
    flag semantics; this does not freeze task context or provider cache state.
    """
    return bool(os.environ.get("VICTOR_FROZEN_PROMPTS"))


def get_builtin_section_strategies() -> Dict[str, List[str]]:
    """Return registry-backed default strategies for prompt sections."""
    try:
        from victor.agent.prompt_section_registry import get_default_section_strategies

        strategy_map = get_default_section_strategies()
        if strategy_map:
            return strategy_map
    except Exception:
        pass
    return {}


class MIPROv2Settings(BaseModel):
    """MIPROv2 few-shot demonstration mining configuration."""

    max_examples: int = 3
    min_completion_score: float = 0.7
    example_diversity: bool = True
    max_example_chars: int = 400


class CoTDistillationSettings(BaseModel):
    """Chain-of-Thought distillation configuration.

    Source and target providers are NOT hardcoded — they are determined
    dynamically at runtime from benchmark data:
    - source: auto-detected as the highest-scoring provider
    - target: all providers scoring below source
    - Can be overridden per call via evolve(source_provider=..., target_provider=...)
    """

    auto_detect_source: bool = True  # Auto-pick best provider as source
    min_source_score: float = 0.7
    max_steps: int = 5
    min_score_gap: float = 0.15  # Only distill if source-target gap >= 15pp


class PrefPOSettings(BaseModel):
    """Pairwise preference prompt optimization configuration."""

    max_guidance_items: int = 2
    min_failure_count: int = 1
    max_prompt_growth_chars: int = 240


class VerbositySettings(BaseModel):
    """Automated verbosity detection for CONCISE_MODE_GUIDANCE evolution.

    Tracks agent response verbosity to generate implicit feedback signals
    for PrefPO optimization. Responses exceeding thresholds are flagged
    as 'verbosity' failures, driving CONCISE_MODE_GUIDANCE refinements.
    """

    enabled: bool = True
    max_response_chars: int = 2000
    max_response_lines: int = 50
    auto_feedback_weight: float = 0.5
    # Minimum verbosity ratio to trigger feedback (actual / max)
    min_verbosity_ratio: float = 1.2


class PromptOptimizationSettings(BaseModel):
    """Top-level prompt optimization configuration."""

    # Master switch — gates ALL prompt optimization
    enabled: bool = True

    # Perf: cache the per-iteration learning-trace collection (JSONL + conversation merge)
    # for this many seconds within a task. The traces are stable across iterations of one
    # turn-sequence, so a short TTL avoids re-reading + re-merging them every iteration with
    # an identical result. 0 disables (default — behavior-preserving). 30–60s is typical.
    cache_traces_ttl_seconds: float = 0.0

    # Default strategy list applied to all evolvable sections.
    # Strategies are applied in order (layered).
    default_strategies: List[str] = Field(default_factory=lambda: ["gepa"])

    # Per-section overrides. Key = section name, value = strategy list.
    # Empty list [] = skip optimization for that section.
    # Missing key = use default_strategies.
    section_strategies: Dict[str, List[str]] = Field(default_factory=dict)

    # Reward-loop bootstrap: a freshly-evolved candidate starts at sample_count=0,
    # so the live-confidence serve gate (confidence>0.6 and not is_baseline) would
    # never serve it and it could never earn a reward. exploration_enabled lets a
    # structurally-valid fresh (or benchmark-approved) candidate be served with
    # probability exploration_epsilon when no proven candidate exists, so the
    # Thompson posterior can start accumulating evidence. Bounded by the persist
    # structural gate (growth_exceeded/repeated_trigrams).
    exploration_enabled: bool = True
    exploration_epsilon: float = 0.1

    # Strategy-specific configurations (nested)
    gepa: GEPASettings = Field(default_factory=GEPASettings)
    miprov2: MIPROv2Settings = Field(default_factory=MIPROv2Settings)
    cot_distillation: CoTDistillationSettings = Field(default_factory=CoTDistillationSettings)
    prefpo: PrefPOSettings = Field(default_factory=PrefPOSettings)
    verbosity: VerbositySettings = Field(default_factory=VerbositySettings)

    def get_strategies_for_section(self, section_name: str) -> List[str]:
        """Resolve which strategies apply to a given section.

        Returns:
            List of strategy names. Empty list means skip.
        """
        if not self.enabled:
            return []
        if section_name in self.section_strategies:
            return self.section_strategies[section_name]
        builtin_section_strategies = get_builtin_section_strategies()
        if section_name in builtin_section_strategies:
            return list(builtin_section_strategies[section_name])
        return self.default_strategies

    def is_strategy_active(self, strategy_name: str) -> bool:
        """Check if a strategy is used by any section."""
        if not self.enabled:
            return False
        if strategy_name in self.default_strategies:
            return True
        for strategies in self.section_strategies.values():
            if strategy_name in strategies:
                return True
        for strategies in get_builtin_section_strategies().values():
            if strategy_name in strategies:
                return True
        return False
