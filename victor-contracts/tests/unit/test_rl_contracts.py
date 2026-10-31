"""Tests for SDK-owned RL contracts.

``RLOutcome`` and ``RLRecommendation`` are the dependency-free mirrors of the
runtime classes in ``victor.framework.rl.base`` (the module the
``rl_runtime`` bridge resolves). These tests pin the aligned shape and, when
the victor-ai host is installed, differentially compare it against the host
classes so the two definitions cannot drift apart again.
"""

from __future__ import annotations

import dataclasses
import os
import typing
from datetime import datetime
from typing import Any, Dict, Optional

import pytest

from victor_contracts.rl import (
    BaseRLConfig,
    DEFAULT_ACTIVE_LEARNERS,
    DEFAULT_PATIENCE_MAP,
    LearnerType,
    RLRecommendation,
    RLOutcome,
)


def test_learner_type_values_remain_stable() -> None:
    assert LearnerType.TOOL_SELECTOR.value == "tool_selector"
    assert LearnerType.MODE_TRANSITION.value == "mode_transition"


def test_base_rl_config_uses_shared_defaults() -> None:
    config = BaseRLConfig()

    assert config.active_learners == DEFAULT_ACTIVE_LEARNERS
    assert config.default_patience == DEFAULT_PATIENCE_MAP
    assert config.get_tools_for_task("missing") == []
    assert config.get_quality_threshold("missing") == 0.80
    assert config.get_patience("unknown") == 4


def _field_shape(cls: type) -> list[tuple[str, object, object]]:
    """``(name, resolved type, default-or-required marker)`` per field.

    Types are resolved with :func:`typing.get_type_hints` so classes defined
    with and without ``from __future__ import annotations`` compare equal.
    """
    hints = typing.get_type_hints(cls)
    shape = []
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            default: object = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            produced = f.default_factory()  # type: ignore[misc]
            default = f"<factory:{type(produced).__name__}>"
        else:
            default = "<required>"
        shape.append((f.name, hints[f.name], default))
    return shape


def test_rl_outcome_shape_is_pinned() -> None:
    assert _field_shape(RLOutcome) == [
        ("provider", str, "<required>"),
        ("model", str, "<required>"),
        ("task_type", str, "<required>"),
        ("success", bool, "<required>"),
        ("quality_score", float, "<required>"),
        ("timestamp", str, "<factory:str>"),
        ("metadata", Dict[str, Any], "<factory:dict>"),
        ("vertical", str, ""),
        ("segment_rewards", Optional[Dict[int, float]], None),
    ]


def test_rl_recommendation_shape_is_pinned() -> None:
    assert _field_shape(RLRecommendation) == [
        ("value", Any, "<required>"),
        ("confidence", float, "<required>"),
        ("reason", str, "<required>"),
        ("sample_size", int, "<required>"),
        ("is_baseline", bool, False),
        ("metadata", Dict[str, Any], "<factory:dict>"),
    ]


def test_rl_outcome_timestamp_is_iso8601() -> None:
    outcome = RLOutcome(
        provider="anthropic",
        model="claude",
        task_type="default",
        success=True,
        quality_score=0.75,
    )

    assert isinstance(outcome.timestamp, str)
    assert datetime.fromisoformat(outcome.timestamp) is not None
    assert outcome.metadata == {}
    assert outcome.segment_rewards is None


def test_rl_outcome_requires_quality_score() -> None:
    """The runtime class requires it; a silently-defaulted 0.0 poisoned learners."""

    with pytest.raises(TypeError):
        RLOutcome(provider="anthropic", model="claude", task_type="default", success=True)  # type: ignore[call-arg]


def test_rl_recommendation_defaults_to_learned_not_baseline() -> None:
    recommendation = RLRecommendation(value=4, confidence=0.6, reason="warm-up", sample_size=3)

    assert recommendation.is_baseline is False
    assert recommendation.metadata == {}


def test_rl_outcome_matches_host_runtime_shape() -> None:
    """Differential: the SDK mirror equals the host class field for field."""
    host = pytest.importorskip(
        "victor.framework.rl.base",
        reason="host shape parity requires the victor-ai package",
    )
    env_vertical = os.environ.get("VICTOR_DEFAULT_VERTICAL")

    sdk_shape = _field_shape(RLOutcome)
    host_shape = _field_shape(host.RLOutcome)
    if env_vertical:
        # The host default is env-coupled; the SDK mirrors the unset-env default.
        host_shape = [f for f in host_shape if f[0] != "vertical"]

    assert sdk_shape == host_shape


def test_rl_recommendation_matches_host_runtime_shape() -> None:
    host = pytest.importorskip(
        "victor.framework.rl.base",
        reason="host shape parity requires the victor-ai package",
    )

    assert _field_shape(RLRecommendation) == _field_shape(host.RLRecommendation)


def test_sdk_outcome_is_accepted_by_host_serialisation() -> None:
    """A vertical-built SDK outcome must be field-complete for host storage."""
    host = pytest.importorskip(
        "victor.framework.rl.base",
        reason="host round-trip requires the victor-ai package",
    )

    outcome = RLOutcome(
        provider="anthropic",
        model="claude",
        task_type="default",
        success=True,
        quality_score=0.75,
        vertical="coding",
        metadata={"turns": 3},
    )

    host_outcome = host.RLOutcome(**dataclasses.asdict(outcome))

    assert dataclasses.asdict(host_outcome) == dataclasses.asdict(outcome)
    assert host.RLOutcome.from_dict(host_outcome.to_dict()) == host_outcome
