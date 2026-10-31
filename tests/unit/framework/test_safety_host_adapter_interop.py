"""Interop tests for SDK safety host adapters."""

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
