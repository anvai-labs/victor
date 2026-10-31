"""SDK-owned safety contracts and pattern declarations."""

from victor_contracts.safety.framework import SafetyLevel
from victor_contracts.safety.patterns import (
    SafetyPatternDeclaration,
    SafetyPatternType,
    SafetySeverity,
)
from victor_contracts.safety.pii import (
    ANONYMIZATION_SUGGESTIONS,
    PII_COLUMN_PATTERNS,
    PII_CONTENT_PATTERNS,
    PII_SEVERITY,
    PIIMatch,
    PIIScanner,
    PIISeverity,
    PIIType,
    detect_pii_columns,
    detect_pii_in_content,
    get_anonymization_suggestion,
    get_pii_severity,
    get_pii_types,
    get_safety_reminders,
    has_pii,
)
from victor_contracts.safety.runtime import (
    SafetyAction,
    SafetyCategory,
    SafetyCheckResult,
    SafetyCoordinator,
    SafetyRule,
    SafetyStats,
)

# SafetyLevel is owned by victor_contracts.safety.framework (the object the host
# runtime binds as victor.framework.config.SafetyLevel) and only re-exported
# here; this top-level name used to be a separate str-Enum fork, so rules
# leveled through it were silently disarmed by the host enforcer.

__all__ = [
    "ANONYMIZATION_SUGGESTIONS",
    "PII_COLUMN_PATTERNS",
    "PII_CONTENT_PATTERNS",
    "PII_SEVERITY",
    "PIIMatch",
    "PIIScanner",
    "PIISeverity",
    "PIIType",
    "SafetyAction",
    "SafetyCategory",
    "SafetyCheckResult",
    "SafetyCoordinator",
    "SafetyLevel",
    "SafetyPatternDeclaration",
    "SafetyPatternType",
    "SafetyRule",
    "SafetySeverity",
    "SafetyStats",
    "detect_pii_columns",
    "detect_pii_in_content",
    "get_anonymization_suggestion",
    "get_pii_severity",
    "get_pii_types",
    "get_safety_reminders",
    "has_pii",
]
