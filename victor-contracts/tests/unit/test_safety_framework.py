"""Tests for SDK framework-style safety compatibility helpers."""

import pytest

from victor_contracts.safety.framework import (
    SafetyConfig,
    SafetyEnforcer,
    SafetyLevel,
    SafetyRule,
)


def test_framework_safety_enforcer_blocks_matching_high_rule() -> None:
    enforcer = SafetyEnforcer(SafetyConfig(level=SafetyLevel.HIGH))
    enforcer.add_rule(
        SafetyRule(
            name="block_delete",
            description="Block delete",
            check_fn=lambda operation: "delete" in operation,
            level=SafetyLevel.HIGH,
        )
    )

    allowed, reason = enforcer.check_operation("delete all")

    assert allowed is False
    assert reason == "Blocked by safety rule: block_delete - Block delete"


def test_framework_safety_enforcer_rejects_duplicate_rules() -> None:
    enforcer = SafetyEnforcer(SafetyConfig())
    rule = SafetyRule(
        name="duplicate",
        description="Duplicate",
        check_fn=lambda operation: False,
    )

    enforcer.add_rule(rule)

    with pytest.raises(ValueError):
        enforcer.add_rule(rule)


def test_framework_safety_config_round_trips_dict() -> None:
    config = SafetyConfig.from_dict(
        {
            "level": "low",
            "require_confirmation": True,
            "blocked_operations": ["upload"],
            "audit_log": False,
            "dry_run": True,
        }
    )

    assert config.level is SafetyLevel.LOW
    assert config.to_dict() == {
        "level": "low",
        "require_confirmation": True,
        "blocked_operations": ["upload"],
        "audit_log": False,
        "dry_run": True,
    }


def test_framework_safety_rule_coerces_to_host_level_when_available() -> None:
    pytest.importorskip("victor", reason="host-level coercion requires the victor-ai package")
    from victor.framework.config import SafetyLevel as HostSafetyLevel

    rule = SafetyRule(
        name="host",
        description="Host level",
        check_fn=lambda operation: True,
        level=SafetyLevel.HIGH,
    )

    assert rule.level == HostSafetyLevel.HIGH


def test_safety_level_is_one_object_across_every_import_path() -> None:
    """The former top-level str-Enum fork is collapsed onto this enum.

    ``victor_contracts.safety.SafetyLevel`` used to be a second, independent
    str-Enum while the host runtime bound ``safety.framework.SafetyLevel`` as
    ``victor.framework.config.SafetyLevel`` — leveling rules through the
    top-level export silently disarmed the host enforcer (round-5 triage A1).
    """
    import victor_contracts
    import victor_contracts.safety

    assert victor_contracts.safety.SafetyLevel is SafetyLevel
    assert victor_contracts.SafetyLevel is SafetyLevel


def test_safety_level_str_enum_behavior_is_pinned() -> None:
    """Python 3.12 str-Enum subtleties, pinned (additive str-ness).

    ``SafetyLevel`` became a ``str, Enum`` mixin in the round-5 unification;
    these assertions freeze the observable behavior so no consumer (core,
    verticals, external SDK users) can drift on string equality, rendering,
    dict keys or JSON serialization.
    """
    import json

    # Value-ness: members compare equal to their raw value strings ...
    assert SafetyLevel.LOW == "low"
    assert SafetyLevel.MEDIUM == "medium"
    assert SafetyLevel.HIGH == "high"
    assert SafetyLevel.OFF == "off"
    # ... but stay real enum members.
    assert type(SafetyLevel.LOW) is SafetyLevel
    assert SafetyLevel("low") is SafetyLevel.LOW
    assert SafetyLevel.LOW.value == "low"
    assert issubclass(SafetyLevel, str)

    # Rendering: str()/f-string keep the Enum repr path on Python 3.12+
    # (NOT the value; only enum.StrEnum would render the bare value).
    assert str(SafetyLevel.LOW) == "SafetyLevel.LOW"
    assert f"{SafetyLevel.HIGH}" == "SafetyLevel.HIGH"
    assert repr(SafetyLevel.LOW) == "<SafetyLevel.LOW: 'low'>"

    # Dict-key behavior: a member and its value string are interchangeable
    # keys in both directions (equality and hash agree).
    assert {"low": 1}[SafetyLevel.LOW] == 1
    assert {SafetyLevel.LOW: 1}["low"] == 1

    # JSON: a str-mixin member serializes as its value string (the plain
    # Enum previously raised TypeError under json.dumps).
    assert json.dumps(SafetyLevel.LOW) == '"low"'
    assert json.dumps({"level": SafetyLevel.LOW}) == '{"level": "low"}'


def test_framework_safety_config_coerces_to_host_level_when_available() -> None:
    pytest.importorskip("victor", reason="host-level coercion requires the victor-ai package")
    from victor.framework.config import SafetyLevel as HostSafetyLevel

    assert SafetyConfig(level=SafetyLevel.HIGH).level == HostSafetyLevel.HIGH
    # Raw strings and foreign enums with the same values bind to the host too
    assert SafetyConfig(level="high").level == HostSafetyLevel.HIGH


def test_framework_safety_enforcer_coerces_config_level_when_available() -> None:
    pytest.importorskip("victor", reason="host-level coercion requires the victor-ai package")
    from victor.framework.config import SafetyLevel as HostSafetyLevel

    enforcer = SafetyEnforcer(SafetyConfig(level=SafetyLevel.LOW))

    assert enforcer.config.level == HostSafetyLevel.LOW


def test_framework_safety_objects_stay_self_contained_without_host(monkeypatch) -> None:
    """Without a host registered, SDK objects keep SDK enum semantics."""
    import victor_contracts.safety.framework as sdk_framework

    monkeypatch.setattr(sdk_framework, "_host_config_module", None, raising=False)

    assert SafetyConfig(level="high").level is SafetyLevel.HIGH
    rule = SafetyRule(
        name="standalone",
        description="Standalone",
        check_fn=lambda operation: True,
        level="high",
    )
    assert rule.level is SafetyLevel.HIGH


def test_framework_safety_level_resolution_delegates_to_the_host_module(
    monkeypatch,
) -> None:
    """Level resolution flows through the registered host hook.

    The host re-exports this module's SafetyLevel (same object) and SafetyLevel
    instances pass through coercion unchanged, so identity on enum inputs
    cannot prove delegation: construct with a string, patch the host module's
    enum with a marker, and assert the resolved level follows it. Unregistering
    the bridge fails this test.
    """
    pytest.importorskip("victor", reason="host-level coercion requires the victor-ai package")
    from enum import Enum

    import victor.framework.config as host_config
    import victor_contracts.safety.framework as sdk_framework

    assert sdk_framework._host_safety_module() is host_config

    class MarkerLevel(Enum):
        OFF = "off"
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    monkeypatch.setattr(host_config, "SafetyLevel", MarkerLevel, raising=False)

    assert SafetyConfig(level="high").level is MarkerLevel.HIGH


def test_framework_safety_enum_instances_pass_through_with_identity_preserved(
    monkeypatch,
) -> None:
    """SafetyLevel instances are never re-bound, even when the host differs."""
    pytest.importorskip("victor", reason="host-level coercion requires the victor-ai package")
    from enum import Enum

    import victor.framework.config as host_config

    class MarkerLevel(Enum):
        OFF = "off"
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    monkeypatch.setattr(host_config, "SafetyLevel", MarkerLevel, raising=False)

    assert SafetyConfig(level=SafetyLevel.HIGH).level is SafetyLevel.HIGH
    rule = SafetyRule(
        name="passthrough",
        description="identity preserved",
        check_fn=lambda operation: True,
        level=SafetyLevel.MEDIUM,
    )
    assert rule.level is SafetyLevel.MEDIUM


def test_framework_safety_enforcer_fails_closed_on_raising_check_fn() -> None:
    """A rule whose check_fn raises must block, never allow (fail-closed)."""

    def exploding(operation: str) -> bool:
        raise RuntimeError("predicate exploded")

    enforcer = SafetyEnforcer(SafetyConfig(level=SafetyLevel.LOW))
    enforcer.add_rule(
        SafetyRule(
            name="exploding",
            description="Raises",
            check_fn=exploding,
            level=SafetyLevel.MEDIUM,
        )
    )

    assert enforcer.check_operation("anything") == (
        False,
        "Safety rule could not be evaluated: exploding",
    )


def test_framework_safety_config_from_dict_parses_levels_like_host() -> None:
    """Level parsing is case-insensitive and raises a friendly ValueError."""
    config = SafetyConfig.from_dict({"level": "High"})

    assert config.level is SafetyLevel.HIGH

    with pytest.raises(ValueError, match="Invalid safety level"):
        SafetyConfig.from_dict({"level": "extreme"})


def test_framework_safety_construction_rejects_invalid_levels() -> None:
    """Invalid level values raise a friendly ValueError at construction.

    Previously ``SafetyConfig(level="bogus")`` stored the bad value silently;
    ``from_dict`` accepted same-enum members. Both now validate eagerly.
    """
    with pytest.raises(ValueError, match="Invalid safety level"):
        SafetyConfig(level="bogus")
    with pytest.raises(ValueError, match="Invalid safety level"):
        SafetyConfig(level=None)
    with pytest.raises(ValueError, match="Invalid safety level"):
        SafetyRule(
            name="bad_level",
            description="Bad level",
            check_fn=lambda operation: True,
            level="bogus",
        )
    with pytest.raises(ValueError, match="Invalid safety level"):
        SafetyConfig.from_dict({"level": SafetyLevel.HIGH})
