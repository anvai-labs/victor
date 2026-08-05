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

"""Safety patterns and utilities for Victor.

This module provides consolidated safety utilities for:
- Secret detection (API keys, credentials, tokens)
- PII detection (email, SSN, credit cards, etc.)
- Code safety patterns (git, refactoring, package management)
- Infrastructure safety patterns (Kubernetes, Docker, Terraform)
- Unified safety registry for pluggable scanners

Example usage:
    from victor.security.safety import (
        detect_secrets,
        detect_pii_columns,
        CodePatternScanner,
        InfrastructureScanner,
        SafetyRegistry,
    )

    # Detect secrets in code
    secrets = detect_secrets(code_content)

    # Detect PII columns
    pii_cols = detect_pii_columns(["email", "name", "score"])

    # Scan git commands
    scanner = CodePatternScanner()
    patterns = scanner.scan_command("git push --force")

    # Infrastructure scanning
    infra_scanner = InfrastructureScanner()
    result = infra_scanner.scan_command("kubectl delete namespace prod")

    # Use registry for unified scanning
    registry = SafetyRegistry()
    registry.register("secrets", SecretScanner())
    findings = registry.scan_all(content)
"""

# Types
from victor.security.safety.types import SafetyPattern

# Registry
from victor.security.safety.registry import ISafetyScanner, SafetyRegistry

# Secrets
from victor.security.safety.secrets import (
    CREDENTIAL_PATTERNS,
    SecretMatch,
    SecretScanner,
    SecretSeverity,
    detect_secrets,
    get_secret_types,
    has_secrets,
    mask_secrets,
)

# PII
from victor.security.safety.pii import (
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
    get_safety_reminders as get_pii_safety_reminders,
    has_pii,
)

# Code patterns
from victor.security.safety.code_patterns import (
    BUILD_DEPLOY_PATTERNS,
    CodePatternCategory,
    CodePatternScanner,
    GIT_PATTERNS,
    PACKAGE_MANAGER_PATTERNS,
    REFACTORING_PATTERNS,
    SENSITIVE_FILE_PATTERNS,
    ScanResult,
    get_all_patterns as get_all_code_patterns,
    is_sensitive_file,
    scan_command,
)

# Infrastructure patterns
from victor.security.safety.infrastructure import (
    CLOUD_PATTERNS,
    DESTRUCTIVE_PATTERNS,
    DOCKER_PATTERNS,
    InfraPatternCategory,
    InfraScanResult,
    InfrastructureScanner,
    KUBERNETES_PATTERNS,
    TERRAFORM_PATTERNS,
    get_all_infrastructure_patterns,
    get_safety_reminders as get_infra_safety_reminders,
    scan_infrastructure_command,
    validate_dockerfile,
    validate_kubernetes_manifest,
)

# Source credibility patterns (extracted from research/safety.py)
from victor.security.safety.source_credibility import (
    DOMAIN_TYPES,
    SOURCE_CREDIBILITY_PATTERNS,
    CredibilityLevel,
    CredibilityMatch,
    SourceCredibilityScanner,
    get_credibility_level,
    get_source_safety_reminders,
    is_high_credibility,
    is_low_credibility,
    validate_source_credibility,
)

# Content warning patterns (extracted from research/safety.py)
from victor.security.safety.content_patterns import (
    ADVICE_RISK_PATTERNS,
    CONTENT_WARNING_PATTERNS,
    MISINFORMATION_RISK_PATTERNS,
    ContentPatternScanner,
    ContentWarningLevel,
    ContentWarningMatch,
    detect_advice_risk,
    detect_misinformation_risk,
    get_content_safety_reminders,
    get_high_severity_warnings,
    has_content_warnings,
    scan_content_warnings,
)

__all__ = [
    # Types
    "SafetyPattern",
    # Registry
    "ISafetyScanner",
    "SafetyRegistry",
    # Secrets
    "CREDENTIAL_PATTERNS",
    "SecretMatch",
    "SecretScanner",
    "SecretSeverity",
    "detect_secrets",
    "get_secret_types",
    "has_secrets",
    "mask_secrets",
    # PII
    "ANONYMIZATION_SUGGESTIONS",
    "PII_COLUMN_PATTERNS",
    "PII_CONTENT_PATTERNS",
    "PII_SEVERITY",
    "PIIMatch",
    "PIIScanner",
    "PIISeverity",
    "PIIType",
    "detect_pii_columns",
    "detect_pii_in_content",
    "get_anonymization_suggestion",
    "get_pii_severity",
    "get_pii_types",
    "get_pii_safety_reminders",
    "has_pii",
    # Code patterns
    "BUILD_DEPLOY_PATTERNS",
    "CodePatternCategory",
    "CodePatternScanner",
    "GIT_PATTERNS",
    "PACKAGE_MANAGER_PATTERNS",
    "REFACTORING_PATTERNS",
    "SENSITIVE_FILE_PATTERNS",
    "ScanResult",
    "get_all_code_patterns",
    "is_sensitive_file",
    "scan_command",
    # Infrastructure patterns
    "CLOUD_PATTERNS",
    "DESTRUCTIVE_PATTERNS",
    "DOCKER_PATTERNS",
    "InfraPatternCategory",
    "InfraScanResult",
    "InfrastructureScanner",
    "KUBERNETES_PATTERNS",
    "TERRAFORM_PATTERNS",
    "get_all_infrastructure_patterns",
    "get_infra_safety_reminders",
    "scan_infrastructure_command",
    "validate_dockerfile",
    "validate_kubernetes_manifest",
    # Source credibility patterns
    "DOMAIN_TYPES",
    "SOURCE_CREDIBILITY_PATTERNS",
    "CredibilityLevel",
    "CredibilityMatch",
    "SourceCredibilityScanner",
    "get_credibility_level",
    "get_source_safety_reminders",
    "is_high_credibility",
    "is_low_credibility",
    "validate_source_credibility",
    # Content warning patterns
    "ADVICE_RISK_PATTERNS",
    "CONTENT_WARNING_PATTERNS",
    "MISINFORMATION_RISK_PATTERNS",
    "ContentPatternScanner",
    "ContentWarningLevel",
    "ContentWarningMatch",
    "detect_advice_risk",
    "detect_misinformation_risk",
    "get_content_safety_reminders",
    "get_high_severity_warnings",
    "has_content_warnings",
    "scan_content_warnings",
]
