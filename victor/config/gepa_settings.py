"""GEPA v2 prompt optimizer configuration.

Configures trace enrichment (ASI), tiered model selection for
reflection/mutation, Pareto-based candidate selection, and
prompt bloat control.

Three model tiers:
- economic: Local model (Ollama qwen3:8b) — post-convergence maintenance
- balanced: Mid-tier cloud (GPT-4.1-mini) — default
- performance: Frontier (Sonnet) — initial convergence
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GEPAModelSpec(BaseModel):
    """A (provider, model) specification for one GEPA tier."""

    provider: str = "ollama"
    model: str = "qwen3:8b"
    # 30s does not cover a reasoning model rewriting a 1500-char section; the
    # observed calls run one to three minutes.
    timeout_s: float = 180.0
    # 1000 tokens predates reasoning models and is below what any measured
    # mutator needs. Both deepseek-v4-pro and kimi-k3 return *nothing* at 1000 on
    # a realistic mutate prompt — the budget goes to hidden reasoning before the
    # rewrite starts — so every call paid a wasted attempt before the retry
    # escalated it. deepseek needs 16384 to answer in full (992 chars at 4096,
    # 1345 at 8192, 1485 at 16384); this starts where the answers actually are.
    max_tokens: int = 8192
    base_url: str = ""  # Non-empty overrides provider default (e.g. coding endpoint)


class GEPASettings(BaseModel):
    """All GEPA prompt optimizer configuration."""

    enabled: bool = True

    # --- Trace enrichment (ASI) ---
    capture_reasoning: bool = True
    capture_tool_args: bool = True
    capture_tool_output: bool = True
    max_output_chars: int = 2000
    max_reasoning_chars: int = 1000

    # --- Tiered models for reflection/mutation ---
    economic_model: GEPAModelSpec = Field(
        default_factory=lambda: GEPAModelSpec(provider="ollama", model="gemma4")
    )
    balanced_model: GEPAModelSpec = Field(
        default_factory=lambda: GEPAModelSpec(provider="openai", model="gpt-4.1-mini")
    )
    performance_model: GEPAModelSpec = Field(
        default_factory=lambda: GEPAModelSpec(
            provider="anthropic", model="claude-sonnet-4-20250514"
        )
    )
    default_tier: str = "balanced"
    use_main_model: bool = True  # Use the main chat model instead of hardcoded tiers

    # --- Auto-tier switching ---
    auto_tier_switch: bool = True
    convergence_window: int = 10
    convergence_threshold: float = 0.02

    # --- Prompt bloat control ---
    max_prompt_chars: int = 1500

    # --- KV cache isolation ---
    # When set AND different from the session model, GEPA reflection uses this
    # model on the SAME provider → different cache namespace → no thrashing.
    # When None or same as session model → GEPA shares the main agent's cache
    # (mid-session reflection can cause KV cache miss + full-token-price cost).
    # The coordinator auto-resolves this to the provider's "edge" tier model
    # when available (e.g., glm-4.7-flash for zai). Set explicitly to override.
    model: str | None = None

    # --- Evolution thresholds ---
    min_traces_for_evolution: int = 5
    max_training_traces: int = 50
    max_candidates_per_section: int = 20
