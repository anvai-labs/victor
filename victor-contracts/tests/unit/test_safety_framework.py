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
