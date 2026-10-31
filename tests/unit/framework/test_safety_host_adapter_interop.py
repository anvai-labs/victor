"""Interop tests for SDK safety host adapters."""

import pytest

from victor.framework.config import (
    SafetyConfig as CoreSafetyConfig,
    SafetyEnforcer as CoreSafetyEnforcer,
    SafetyLevel as CoreSafetyLevel,
    SafetyRule as CoreSafetyRule,
)
from victor.framework.safety import (
    create_file_safety_rules as core_create_file_safety_rules,
    create_git_safety_rules as core_create_git_safety_rules,
)
from victor.security.safety.code_patterns import (
    BUILD_DEPLOY_PATTERNS as CoreBuildDeployPatterns,
    CodePatternScanner as CoreCodePatternScanner,
)
from victor.security.safety.infrastructure import (
    InfrastructureScanner as CoreInfrastructureScanner,
)
from victor.security.safety.secrets import SecretScanner as CoreSecretScanner
from victor.security.safety.source_credibility import (
    validate_source_credibility as core_validate_source_credibility,
)
from victor_contracts.safety import SafetyLevel as SdkTopLevelSafetyLevel
from victor_contracts.safety.framework import (
    SafetyConfig as SdkFrameworkConfig,
    SafetyEnforcer as SdkFrameworkEnforcer,
    SafetyLevel as SdkFrameworkLevel,
    SafetyRule as SdkFrameworkRule,
)
from victor_contracts.safety_patterns import (
    BUILD_DEPLOY_PATTERNS as SdkBuildDeployPatterns,
    CodePatternScanner as SdkCodePatternScanner,
    InfrastructureScanner as SdkInfrastructureScanner,
    SecretScanner as SdkSecretScanner,
    validate_source_credibility as sdk_validate_source_credibility,
)
from victor_contracts.safety_policy import (
    SafetyConfig as SdkSafetyConfig,
    SafetyEnforcer as SdkSafetyEnforcer,
    SafetyLevel as SdkSafetyLevel,
    SafetyRule as SdkSafetyRule,
    create_file_safety_rules as sdk_create_file_safety_rules,
    create_git_safety_rules as sdk_create_git_safety_rules,
)


def test_safety_policy_identity_is_shared() -> None:
    assert CoreSafetyConfig is SdkSafetyConfig
    assert CoreSafetyEnforcer is SdkSafetyEnforcer
    assert CoreSafetyLevel is SdkSafetyLevel
    assert CoreSafetyRule is SdkSafetyRule
    assert sdk_create_file_safety_rules.__name__ == core_create_file_safety_rules.__name__
    assert sdk_create_file_safety_rules.__module__ == core_create_file_safety_rules.__module__
    assert sdk_create_git_safety_rules.__name__ == core_create_git_safety_rules.__name__
    assert sdk_create_git_safety_rules.__module__ == core_create_git_safety_rules.__module__


def test_safety_pattern_identity_is_shared() -> None:
    assert CoreBuildDeployPatterns is SdkBuildDeployPatterns
    assert CoreCodePatternScanner is SdkCodePatternScanner
    assert CoreInfrastructureScanner is SdkInfrastructureScanner
    assert CoreSecretScanner is SdkSecretScanner
    assert core_validate_source_credibility is sdk_validate_source_credibility


def test_sdk_top_level_level_cannot_disarm_core_enforcer() -> None:
    """Round-5 triage repro: a rule leveled via the SDK top-level enum must block.

    Before the SDK SafetyLevel unification, ``victor_contracts.SafetyLevel`` was
    a separate str-Enum, so that foreign member compared unequal to every
    ``SafetyLevel`` branch in the core enforcer's enum-identity leveling and
    ``check_operation`` silently ALLOWED the operation (fail-open: the exact
    ``(True, None)`` repro from the triage). With one enum object the HIGH
    branch is hit again at the enum-identity layer. (The enforcer's own
    fail-closed hardening for genuinely unknown level objects is tracked
    separately and intentionally NOT changed here.)
    """
    import victor_contracts

    assert victor_contracts.SafetyLevel is CoreSafetyLevel

    enforcer = CoreSafetyEnforcer(config=CoreSafetyConfig(level=CoreSafetyLevel.LOW))
    enforcer.add_rule(
        CoreSafetyRule(
            name="block_delete",
            description="Block deletes",
            check_fn=lambda operation: "delete" in operation,
            level=victor_contracts.SafetyLevel.HIGH,
            allow_override=False,
        )
    )

    allowed, reason = enforcer.check_operation("delete all data")

    assert allowed is False
    assert reason == "Blocked by safety rule: block_delete - Block deletes"
    # Level filtering is enum-identity based in core; the SDK member must be
    # found there too (it used to be an object from a foreign enum class).
    assert enforcer.get_rules_by_level(victor_contracts.SafetyLevel.HIGH) == [enforcer.rules[0]]
    assert enforcer.get_rules_by_level(CoreSafetyLevel.MEDIUM) == []


# =============================================================================
# SafetyConfig / SafetyEnforcer host bridge (round-5 PR: fail-closed unification)
# =============================================================================

_LEVEL_NAMES = ("OFF", "LOW", "MEDIUM", "HIGH")


def _predicate(kind: str):
    """Build a check_fn for the requested outcome: allow, deny or raise."""
    if kind == "allow":
        return lambda operation: False
    if kind == "deny":
        return lambda operation: True

    def raising(operation: str) -> bool:
        raise RuntimeError("predicate exploded")

    return raising


def _enforcer_decision(world, config_level: str, rule_level: str, kind: str):
    """Run one matrix cell through the given (core or SDK) enforcer kit."""
    enforcer_cls, config_cls, rule_cls, level_enum = world
    enforcer = enforcer_cls(config=config_cls(level=level_enum[config_level]))
    enforcer.add_rule(
        rule_cls(
            name="matrix_rule",
            description="matrix rule",
            check_fn=_predicate(kind),
            level=level_enum[rule_level],
        )
    )
    return enforcer.check_operation("dangerous operation")


def _core_world():
    return (
        CoreSafetyEnforcer,
        CoreSafetyConfig,
        CoreSafetyRule,
        CoreSafetyLevel,
    )


def _sdk_world():
    return (
        SdkFrameworkEnforcer,
        SdkFrameworkConfig,
        SdkFrameworkRule,
        SdkFrameworkLevel,
    )


@pytest.mark.parametrize("rule_level", _LEVEL_NAMES)
@pytest.mark.parametrize("config_level", _LEVEL_NAMES)
@pytest.mark.parametrize("kind", ["allow", "deny", "raise"])
def test_enforcer_decisions_match_across_core_and_sdk(kind, config_level, rule_level) -> None:
    """Core and SDK enforcers agree on every level x outcome cell (host registered)."""
    core_decision = _enforcer_decision(_core_world(), config_level, rule_level, kind)
    sdk_decision = _enforcer_decision(_sdk_world(), config_level, rule_level, kind)

    assert sdk_decision == core_decision, (
        f"core={core_decision} sdk={sdk_decision} for "
        f"config={config_level} rule={rule_level} check_fn={kind}"
    )


@pytest.mark.parametrize("rule_level", _LEVEL_NAMES)
@pytest.mark.parametrize("config_level", _LEVEL_NAMES)
@pytest.mark.parametrize("kind", ["allow", "deny", "raise"])
def test_enforcer_decisions_match_with_host_hook_unregistered(
    monkeypatch, kind, config_level, rule_level
) -> None:
    """Core and SDK enforcers still agree with the SDK host hook unregistered.

    Simulates a host-less SDK install: the SDK keeps self-contained (fail-closed)
    semantics while the core enforcer runs its own, and the decisions match.
    """
    import victor_contracts.safety.framework as sdk_framework

    # raising=False keeps this test meaningful against trees that predate the
    # bridge attribute (the negative baseline run uses such a tree).
    monkeypatch.setattr(sdk_framework, "_host_config_module", None, raising=False)

    core_decision = _enforcer_decision(_core_world(), config_level, rule_level, kind)
    sdk_decision = _enforcer_decision(_sdk_world(), config_level, rule_level, kind)

    assert sdk_decision == core_decision, (
        f"core={core_decision} sdk={sdk_decision} for "
        f"config={config_level} rule={rule_level} check_fn={kind} (host unregistered)"
    )


def test_sdk_safety_config_binds_to_host_level_when_registered() -> None:
    """SDK SafetyConfig levels resolve through the host hook (rule-style bridge).

    SDK-enum instances pass through unchanged (the host re-exports the same
    enum, so identity holds trivially); foreign enums and raw strings are
    normalized through the hook.
    """
    config = SdkFrameworkConfig(level=SdkFrameworkLevel.HIGH)

    assert config.level is CoreSafetyLevel.HIGH

    # Foreign enum and raw string levels are normalized through the same hook
    assert SdkFrameworkConfig(level=SdkTopLevelSafetyLevel.HIGH).level is CoreSafetyLevel.HIGH
    assert SdkFrameworkConfig(level="high").level is CoreSafetyLevel.HIGH


def test_sdk_safety_enum_instances_pass_through_with_identity_preserved(monkeypatch) -> None:
    """SafetyLevel instances are never re-bound, even when the host differs.

    Pinning the pass-through: with a foreign host enum patched in, an SDK-enum
    level must still come back as the very same object (no rebind), while the
    same construction with a string follows the host enum.
    """
    from enum import Enum

    import victor.framework.config as host_config

    class MarkerLevel(Enum):
        OFF = "off"
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    monkeypatch.setattr(host_config, "SafetyLevel", MarkerLevel, raising=False)

    assert SdkFrameworkConfig(level=SdkFrameworkLevel.HIGH).level is SdkFrameworkLevel.HIGH
    rule = SdkFrameworkRule(
        name="passthrough",
        description="identity preserved",
        check_fn=lambda operation: True,
        level=SdkFrameworkLevel.MEDIUM,
    )
    assert rule.level is SdkFrameworkLevel.MEDIUM

    # Strings still follow the host hook
    assert SdkFrameworkConfig(level="high").level is MarkerLevel.HIGH


def test_sdk_safety_enforcer_binds_config_to_host_level_when_registered() -> None:
    """SDK SafetyEnforcer normalizes its config's level through the host hook."""
    enforcer = SdkFrameworkEnforcer(config=SdkFrameworkConfig(level=SdkTopLevelSafetyLevel.LOW))

    assert enforcer.config.level is CoreSafetyLevel.LOW


def test_sdk_safety_objects_stay_self_contained_without_host(monkeypatch) -> None:
    """With no host registered, SDK objects keep SDK semantics (still fail-closed)."""
    import victor_contracts.safety.framework as sdk_framework

    monkeypatch.setattr(sdk_framework, "_host_config_module", None, raising=False)

    config = SdkFrameworkConfig(level=SdkTopLevelSafetyLevel.HIGH)
    assert config.level is SdkFrameworkLevel.HIGH

    enforcer = SdkFrameworkEnforcer(config=config)
    enforcer.add_rule(
        SdkFrameworkRule(
            name="exploding",
            description="raises",
            check_fn=_predicate("raise"),
            level=SdkFrameworkLevel.HIGH,
        )
    )
    assert enforcer.check_operation("dangerous operation") == (
        False,
        "Safety rule could not be evaluated: exploding",
    )


def test_sdk_safety_level_resolution_delegates_to_the_host_module(monkeypatch) -> None:
    """SDK level resolution flows through the registered host hook.

    The host and SDK SafetyLevel objects are the same enum in this tree (the
    host re-exports the SDK definition) and SDK-enum instances pass through
    coercion unchanged, so identity on enum inputs cannot prove the delegation:
    construct with a string and a foreign enum, patch the host module's enum
    with a marker, and assert the resolved levels follow it. Unregistering the
    bridge (hook returning ``None``) fails this test.
    """
    from enum import Enum

    import victor.framework.config as host_config
    import victor_contracts.safety.framework as sdk_framework

    # The hook is registered and resolves to the host module
    assert sdk_framework._host_safety_module() is host_config

    class MarkerLevel(Enum):
        OFF = "off"
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    monkeypatch.setattr(host_config, "SafetyLevel", MarkerLevel, raising=False)

    assert SdkFrameworkConfig(level="high").level is MarkerLevel.HIGH
    # Since the SafetyLevel unification the top-level export IS the SDK enum,
    # so SDK-enum levels pass through UNCHANGED (identity preserved) instead
    # of re-binding to the host marker; delegation applies to string and
    # foreign-enum inputs.
    assert SdkFrameworkConfig(level=SdkTopLevelSafetyLevel.LOW).level is SdkTopLevelSafetyLevel.LOW
    enforcer = SdkFrameworkEnforcer(config=SdkFrameworkConfig(level="low"))
    assert enforcer.config.level is MarkerLevel.LOW


def test_sdk_safety_config_from_dict_matches_host_parsing() -> None:
    """SDK from_dict level parsing matches the host (case-insensitive, friendly error)."""
    # Case-insensitive level lookup succeeds on both sides
    assert SdkFrameworkConfig.from_dict({"level": "High"}).level is CoreSafetyLevel.HIGH
    assert CoreSafetyConfig.from_dict({"level": "High"}).level is CoreSafetyLevel.HIGH

    # Invalid levels raise the same friendly ValueError on both sides
    with pytest.raises(ValueError, match="Invalid safety level"):
        SdkFrameworkConfig.from_dict({"level": "extreme"})
    with pytest.raises(ValueError, match="Invalid safety level"):
        CoreSafetyConfig.from_dict({"level": "extreme"})
