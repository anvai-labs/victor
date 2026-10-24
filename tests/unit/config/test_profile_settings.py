"""ProfileConfig extraction pin (self-analysis round 2, W5).

The class moved verbatim to victor/config/profile_settings.py; the
Settings facade re-exports it. Pinned: identity through the facade, the
tier-shortcut validator behavior (the class's one piece of real logic),
and field surface.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from victor.config.profile_settings import ProfileConfig
from victor.config.settings import ProfileConfig as FacadeProfileConfig


def test_facade_reexport_is_the_same_class():
    assert FacadeProfileConfig is ProfileConfig


def test_tool_selection_tier_shortcut_expands_with_overrides():
    config = ProfileConfig(
        provider="ollama",
        model="qwen3",
        tool_selection={
            "model_size_tier": "small",
            "base_threshold": 0.9,  # manual value wins over the preset
        },
    )
    expanded = config.tool_selection
    assert expanded["base_threshold"] == 0.9
    assert "base_max_tools" in expanded  # preset fields applied


def test_tool_selection_validator_rejects_bad_values():
    with pytest.raises(ValidationError):
        ProfileConfig(provider="x", model="y", tool_selection={"base_threshold": 5.0})
    with pytest.raises(ValidationError):
        ProfileConfig(provider="x", model="y", tool_selection={"base_max_tools": 0})


def test_field_surface_unchanged():
    expected = {
        "provider",
        "model",
        "account",
        "temperature",
        "temperatures",
        "max_tokens",
        "reasoning_effort",
        "description",
        "tool_selection",
        "planning_provider",
        "planning_model",
        "loop_repeat_threshold",
        "max_continuation_prompts",
        "quality_threshold",
        "grounding_threshold",
        "max_tool_calls_per_turn",
        "tool_cache_enabled",
        "tool_deduplication_enabled",
        "session_idle_timeout",
        "timeout",
    }
    assert set(ProfileConfig.model_fields) == expected
