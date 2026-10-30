"""SDK host adapters for reusable safety scanners and pattern sets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from victor_contracts._lazy import make_lazy_getattr

if TYPE_CHECKING:
    from victor.security.safety.code_patterns import (
        BUILD_DEPLOY_PATTERNS,
        GIT_PATTERNS,
        PACKAGE_MANAGER_PATTERNS,
        REFACTORING_PATTERNS,
        SENSITIVE_FILE_PATTERNS,
        CodePatternScanner,
    )
    from victor.security.safety.content_patterns import (
        CONTENT_WARNING_PATTERNS,
        scan_content_warnings,
    )
    from victor.security.safety.infrastructure import (
        CLOUD_PATTERNS,
        DESTRUCTIVE_PATTERNS,
        DOCKER_PATTERNS,
        KUBERNETES_PATTERNS,
        TERRAFORM_PATTERNS,
        InfraScanResult,
        InfrastructureScanner,
        get_safety_reminders,
        validate_dockerfile,
        validate_kubernetes_manifest,
    )
    from victor.security.safety.secrets import CREDENTIAL_PATTERNS, SecretScanner
    from victor.security.safety.source_credibility import (
        SOURCE_CREDIBILITY_PATTERNS,
        get_source_safety_reminders,
        validate_source_credibility,
    )

__all__ = [
    "BUILD_DEPLOY_PATTERNS",
    "CLOUD_PATTERNS",
    "CONTENT_WARNING_PATTERNS",
    "CREDENTIAL_PATTERNS",
    "CodePatternScanner",
    "DESTRUCTIVE_PATTERNS",
    "DOCKER_PATTERNS",
    "GIT_PATTERNS",
    "InfraScanResult",
    "InfrastructureScanner",
    "KUBERNETES_PATTERNS",
    "PACKAGE_MANAGER_PATTERNS",
    "REFACTORING_PATTERNS",
    "SENSITIVE_FILE_PATTERNS",
    "SOURCE_CREDIBILITY_PATTERNS",
    "SecretScanner",
    "TERRAFORM_PATTERNS",
    "get_safety_reminders",
    "get_source_safety_reminders",
    "scan_content_warnings",
    "validate_dockerfile",
    "validate_kubernetes_manifest",
    "validate_source_credibility",
]

_LAZY_IMPORTS = {
    "BUILD_DEPLOY_PATTERNS": "victor.security.safety.code_patterns",
    "CodePatternScanner": "victor.security.safety.code_patterns",
    "GIT_PATTERNS": "victor.security.safety.code_patterns",
    "PACKAGE_MANAGER_PATTERNS": "victor.security.safety.code_patterns",
    "REFACTORING_PATTERNS": "victor.security.safety.code_patterns",
    "SENSITIVE_FILE_PATTERNS": "victor.security.safety.code_patterns",
    "CONTENT_WARNING_PATTERNS": "victor.security.safety.content_patterns",
    "scan_content_warnings": "victor.security.safety.content_patterns",
    "CLOUD_PATTERNS": "victor.security.safety.infrastructure",
    "DESTRUCTIVE_PATTERNS": "victor.security.safety.infrastructure",
    "DOCKER_PATTERNS": "victor.security.safety.infrastructure",
    "InfraScanResult": "victor.security.safety.infrastructure",
    "InfrastructureScanner": "victor.security.safety.infrastructure",
    "KUBERNETES_PATTERNS": "victor.security.safety.infrastructure",
    "TERRAFORM_PATTERNS": "victor.security.safety.infrastructure",
    "get_safety_reminders": "victor.security.safety.infrastructure",
    "validate_dockerfile": "victor.security.safety.infrastructure",
    "validate_kubernetes_manifest": "victor.security.safety.infrastructure",
    "CREDENTIAL_PATTERNS": "victor.security.safety.secrets",
    "SecretScanner": "victor.security.safety.secrets",
    "SOURCE_CREDIBILITY_PATTERNS": "victor.security.safety.source_credibility",
    "get_source_safety_reminders": "victor.security.safety.source_credibility",
    "validate_source_credibility": "victor.security.safety.source_credibility",
}


__getattr__ = make_lazy_getattr(__name__, _LAZY_IMPORTS)
