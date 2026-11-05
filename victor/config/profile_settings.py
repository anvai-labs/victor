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

"""Model profile configuration (self-analysis round 2, W5).

ProfileConfig moved VERBATIM from victor/config/settings.py — the
stratification pattern the file already uses for its other groups
(*_settings.py modules imported back into the Settings facade). The
class is a complete BaseSettings surface with one validator; it has no
dependencies on the Settings class, so it extracts cleanly.
victor.config.settings re-exports it (import sites unchanged).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from victor.config.orchestrator_constants import TOOL_SELECTION_PRESETS
from victor.core.model_parameters import ReasoningEffort

__all__ = ["ProfileConfig", "ThinkingConfig"]


# ---------------------------------------------------------------------------
# Thinking configuration — first-class profile field (Proposal 5)
# ---------------------------------------------------------------------------

# Anthropic provider slugs that enforce the temperature+thinking constraint.
# When a profile uses one of these providers with thinking.enabled=true,
# the cross-field validator rejects a co-present `temperature` setting.
_ANTHROPIC_THINKING_PROVIDERS = frozenset({"anthropic", "bedrock"})


class ThinkingConfig(BaseModel):
    """Extended-thinking configuration for Anthropic (and compatible) models.

    When ``enabled`` is ``True`` the provider receives a ``thinking`` block
    in the request.  Anthropic's API **rejects** ``temperature`` alongside
    an enabled thinking block — Victor strips it automatically, but a profile
    that explicitly sets both fields will be rejected at load time so the
    conflict is surfaced early rather than silently swallowed.

    Example profile YAML::

        profiles:
          my_thinking_profile:
            provider: anthropic
            model: claude-sonnet-4-5-20251101
            thinking:
              enabled: true
              budget_tokens: 12000
            # temperature: 0.7  ← would raise ValidationError

    Attributes:
        enabled: Activate extended thinking for this profile (default: False).
        budget_tokens: Token budget for the thinking block.  Anthropic models
            have per-model maximums; see model_capabilities.yaml for defaults.
            Must be >= 1024 (Anthropic minimum) and <= 100000.
    """

    enabled: bool = False
    budget_tokens: int = Field(
        10000,
        ge=1024,
        le=100000,
        description=(
            "Token budget for the thinking block.  Anthropic's minimum is 1024; "
            "per-model maximums are listed in model_capabilities.yaml."
        ),
    )


class ProfileContextConfig(BaseModel):
    """Per-profile context-window overrides (narrow, by design).

    Only the keys victor actually consumes from a profile's context group.
    Flows into ``settings.context`` (ContextSettings) at factory time — see
    ``AgentFactory._apply_profile_context_window`` — so
    ``resolve_context_window`` honors it with ``source=settings``. Untyped
    profile keys in this group previously survived only as
    ``__pydantic_extra__`` and never reached the resolver.
    """

    max_context_tokens: Optional[int] = Field(
        None,
        gt=0,
        description=(
            "Context window (tokens) for this profile's model. Overrides the "
            "model-pattern window from provider_context_limits.yaml; "
            "VICTOR_MAX_CONTEXT_TOKENS and other user-set settings still win."
        ),
    )


class ProfileConfig(BaseSettings):
    """Configuration for a model profile."""

    model_config = SettingsConfigDict(extra="allow")

    provider: str = Field(..., description="Provider name (ollama, anthropic, openai, google)")
    model: str = Field(..., description="Model identifier")
    account: Optional[str] = Field(
        None,
        description=(
            "Provider account name (credential identity). Decoupled from the provider "
            "transport slug: multiple accounts can share one provider/dialect (e.g. several "
            "anthropic-dialect upstreams) while each resolves its own credential. When unset, "
            "the provider's default account is used."
        ),
    )
    temperature: float = Field(0.6, ge=0.0, le=2.0)  # ADR-013 default flip 0.7→0.6 (A/B 2026-06-22)
    temperatures: Optional[Dict[str, float]] = Field(
        None,
        description="Per-task temperature overrides (task_type -> temperature), e.g. {'plan': 0.5}. "
        "Highest precedence in the ADR-013 resolution chain; absent keys defer to settings/constants.",
    )
    max_tokens: int = Field(4096, gt=0)
    reasoning_effort: Optional[ReasoningEffort] = Field(
        None,
        description="Default reasoning effort for capable models; explicit requests override it. "
        "Allowed levels depend on the selected provider and model.",
    )
    thinking: Optional[ThinkingConfig] = Field(
        None,
        description=(
            "Extended-thinking configuration for Anthropic thinking models "
            "(claude-sonnet-4-5, claude-sonnet-5, claude-opus-4-8, etc.). "
            "When thinking.enabled=true, temperature must NOT be set — "
            "Anthropic's API rejects the combination. Victor enforces this at "
            "profile-load time so the error surfaces before any API call is made."
        ),
    )
    thinking_budget_tokens: Optional[int] = Field(
        None,
        ge=1024,
        le=100000,
        description=(
            "Convenience alias: sets thinking.budget_tokens without requiring the "
            "full thinking block. Ignored when thinking is None or thinking.enabled=False."
        ),
    )
    description: Optional[str] = Field(None, description="Optional profile description")
    context: Optional[ProfileContextConfig] = Field(
        None,
        description=(
            "Context-window overrides for this profile (e.g. "
            "context: {max_context_tokens: 262144}). Applied to settings at "
            "factory time without clobbering user-set values."
        ),
    )
    tool_selection: Optional[Dict[str, Any]] = Field(
        None, description="Tool selection configuration for adaptive thresholds"
    )

    # -------------------------------------------------------------------------
    # Planning Configuration (P3-1)
    # -------------------------------------------------------------------------
    # Optional override for planning-specific model selection
    # Allows using a different (often more capable or cost-effective) model for
    # structured planning vs regular chat execution
    #
    # Example: Use local model for chat (fast, cheap) but cloud model for planning
    # (better at structured output, more reliable JSON generation)
    #
    # Set planning_provider and planning_model to override defaults for planning
    # If not set, uses the default provider/model from the profile
    # -------------------------------------------------------------------------

    planning_provider: Optional[str] = Field(
        None,
        description="Override provider for planning (e.g., 'deepseek', 'anthropic')",
    )
    planning_model: Optional[str] = Field(
        None,
        description="Override model for planning tasks (e.g., 'deepseek-chat', 'claude-sonnet-4-5')",
    )

    # -------------------------------------------------------------------------
    # Provider Tuning Options (P3-1)
    # -------------------------------------------------------------------------
    # These settings allow fine-tuning agent behavior per provider/model.
    # Override defaults from Settings for provider-specific optimizations.

    # Loop detection thresholds - controls when to detect repetitive tool calls
    loop_repeat_threshold: Optional[int] = Field(
        None, description="Number of identical calls before triggering loop detection"
    )
    max_continuation_prompts: Optional[int] = Field(
        None,
        description="Max consecutive continuation prompts before forcing completion",
    )

    # Quality thresholds - controls response quality requirements
    quality_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum quality score to accept response"
    )
    grounding_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for grounding verification",
    )

    # Tool behavior - controls tool usage patterns
    max_tool_calls_per_turn: Optional[int] = Field(
        None, gt=0, description="Maximum tool calls allowed in single model turn"
    )
    tool_cache_enabled: Optional[bool] = Field(
        None, description="Enable/disable tool result caching for this profile"
    )
    tool_deduplication_enabled: Optional[bool] = Field(
        None, description="Enable/disable tool call deduplication"
    )

    # Timeout and session limits
    session_idle_timeout: Optional[int] = Field(
        None,
        gt=0,
        description="Maximum idle time in seconds (resets on provider response/tool execution)",
    )
    # Future: session_time_limit for total session duration cap (regardless of activity)
    timeout: Optional[int] = Field(None, gt=0, description="Request timeout in seconds")

    @model_validator(mode="after")
    def validate_thinking_temperature_compat(self) -> "ProfileConfig":
        """Reject profiles that set both temperature and thinking.enabled.

        Anthropic's API returns a 400 error when ``temperature`` is present
        alongside an enabled ``thinking`` block.  Catching it here surfaces a
        clear ValidationError at profile-load time so the user can fix their
        profiles.yaml before any API call is attempted.

        Only enforced for Anthropic-family providers (``anthropic``, ``bedrock``)
        because other providers may accept both fields independently.
        """
        if (
            self.provider in _ANTHROPIC_THINKING_PROVIDERS
            and self.thinking is not None
            and self.thinking.enabled
            # temperature has a default of 0.6; only flag it when the field
            # was explicitly supplied (i.e. the user set it in their YAML).
            # We detect "explicitly supplied" by checking model_fields_set,
            # which Pydantic populates only for fields present in the input.
            and "temperature" in self.model_fields_set
        ):
            raise ValueError(
                f"Profile '{getattr(self, 'model', '?')}' on provider '{self.provider}': "
                "'temperature' must not be set when 'thinking.enabled' is true. "
                "Anthropic's API rejects both together. "
                "Remove 'temperature' from your profile or set 'thinking.enabled: false'."
            )
        return self

    @field_validator("tool_selection")
    @classmethod
    def validate_tool_selection(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate tool_selection configuration.

        Args:
            v: Tool selection configuration dictionary

        Returns:
            Validated configuration with expanded tier shortcuts

        Raises:
            ValueError: If configuration is invalid
        """
        if v is None:
            return None

        # Predefined model size tiers for convenience (from orchestrator_constants)
        TIER_PRESETS = {
            "tiny": TOOL_SELECTION_PRESETS.tiny,  # 0.5B-3B
            "small": TOOL_SELECTION_PRESETS.small,  # 7B-8B
            "medium": TOOL_SELECTION_PRESETS.medium,  # 13B-15B
            "large": TOOL_SELECTION_PRESETS.large,  # 30B+
            "cloud": TOOL_SELECTION_PRESETS.cloud,  # Claude/GPT
        }

        # Expand tier shortcuts
        if "model_size_tier" in v:
            tier = v["model_size_tier"]
            if tier in TIER_PRESETS:
                # Apply preset values, but allow manual overrides
                preset = TIER_PRESETS[tier].copy()
                preset.update(v)  # Manual values override preset
                v = preset

        # Validate base_threshold
        if "base_threshold" in v:
            threshold = v["base_threshold"]
            if not isinstance(threshold, (int, float)):
                raise ValueError(f"base_threshold must be a number, got {type(threshold)}")
            if not (0.0 <= threshold <= 1.0):
                raise ValueError(f"base_threshold must be between 0.0 and 1.0, got {threshold}")

        # Validate base_max_tools
        if "base_max_tools" in v:
            max_tools = v["base_max_tools"]
            if not isinstance(max_tools, int):
                raise ValueError(f"base_max_tools must be an integer, got {type(max_tools)}")
            if max_tools < 1:
                raise ValueError(f"base_max_tools must be positive, got {max_tools}")

        return v
