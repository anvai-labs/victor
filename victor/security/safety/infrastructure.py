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

"""Infrastructure safety patterns for DevOps operations.

This module provides consolidated safety patterns for:
- Kubernetes operations (destructive and security)
- Docker operations (containers, images, volumes)
- Terraform operations (infrastructure)
- Cloud provider operations (AWS, GCP, Azure)
- Container security configurations

These patterns are used across verticals (devops, data_analysis) to detect
potentially dangerous operations and security misconfigurations.

Example usage:
    from victor.security.safety.infrastructure import (
        InfrastructureScanner,
        InfraPatternCategory,
        validate_dockerfile,
        validate_kubernetes_manifest,
    )

    # Scan infrastructure commands
    scanner = InfrastructureScanner()
    result = scanner.scan_command("kubectl delete namespace production")
    if result.has_critical:
        print("CRITICAL: Production namespace deletion!")

    # Validate Dockerfile
    warnings = validate_dockerfile(dockerfile_content)
    for warning in warnings:
        print(f"- {warning}")

    # Validate Kubernetes manifest
    issues = validate_kubernetes_manifest(manifest_yaml)
    for issue in issues:
        print(f"- {issue}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from victor.security.safety.types import SafetyPattern

# =============================================================================
# Enumerations
# =============================================================================


class InfraPatternCategory(Enum):
    """Categories for infrastructure safety patterns."""

    DESTRUCTIVE = "destructive"
    KUBERNETES = "kubernetes"
    DOCKER = "docker"
    TERRAFORM = "terraform"
    CLOUD = "cloud"
    DATABASE = "database"
    SECURITY = "security"


class RiskLevel(Enum):
    """Risk levels for safety patterns."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# =============================================================================
# Pattern Definitions
# =============================================================================


# Destructive operations - high data loss risk
DESTRUCTIVE_PATTERNS: List[SafetyPattern] = [
    SafetyPattern(
        pattern=r"rm\s+-rf\s+/(?!tmp)",
        description="Destructive filesystem deletion",
        risk_level="CRITICAL",
        category="destructive",
    ),
    SafetyPattern(
        pattern=r"rm\s+-rf\s+\*",
        description="Wildcard deletion",
        risk_level="HIGH",
        category="destructive",
    ),
    SafetyPattern(
        pattern=r"DROP\s+DATABASE",
        description="Database destruction",
        risk_level="CRITICAL",
        category="database",
    ),
    SafetyPattern(
        pattern=r"DROP\s+TABLE",
        description="Table destruction",
        risk_level="HIGH",
        category="database",
    ),
    SafetyPattern(
        pattern=r"TRUNCATE\s+TABLE",
        description="Table truncation",
        risk_level="HIGH",
        category="database",
    ),
    SafetyPattern(
        pattern=r"DELETE\s+FROM\s+\w+\s*(?:;|$)",
        description="Unfiltered table deletion",
        risk_level="HIGH",
        category="database",
    ),
]


# Kubernetes patterns
KUBERNETES_PATTERNS: List[SafetyPattern] = [
    # Destructive operations
    SafetyPattern(
        pattern=r"kubectl\s+delete\s+(?:namespace|ns)\s+(?!test|dev|local|staging)",
        description="Production namespace deletion",
        risk_level="CRITICAL",
        category="kubernetes",
    ),
    SafetyPattern(
        pattern=r"kubectl\s+delete\s+(?:deployment|deploy|service|svc|pod|pv|pvc)",
        description="Kubernetes resource deletion",
        risk_level="HIGH",
        category="kubernetes",
    ),
    SafetyPattern(
        pattern=r"kubectl\s+delete\s+--all",
        description="Delete all resources of type",
        risk_level="CRITICAL",
        category="kubernetes",
    ),
    SafetyPattern(
        pattern=r"kubectl\s+apply\s+-f\s+.*--force",
        description="Force apply configuration",
        risk_level="MEDIUM",
        category="kubernetes",
    ),
    SafetyPattern(
        pattern=r"kubectl\s+rollout\s+undo",
        description="Rollback deployment",
        risk_level="MEDIUM",
        category="kubernetes",
    ),
    SafetyPattern(
        pattern=r"kubectl\s+scale.*replicas=0",
        description="Scale to zero replicas",
        risk_level="MEDIUM",
        category="kubernetes",
    ),
    # Security misconfigurations
    SafetyPattern(
        pattern=r"privileged:\s*true",
        description="Privileged container mode",
        risk_level="MEDIUM",
        category="security",
    ),
    SafetyPattern(
        pattern=r"hostNetwork:\s*true",
        description="Host network access",
        risk_level="MEDIUM",
        category="security",
    ),
    SafetyPattern(
        pattern=r"hostPID:\s*true",
        description="Host PID namespace",
        risk_level="MEDIUM",
        category="security",
    ),
    SafetyPattern(
        pattern=r"hostIPC:\s*true",
        description="Host IPC namespace",
        risk_level="MEDIUM",
        category="security",
    ),
    SafetyPattern(
        pattern=r"capabilities:.*SYS_ADMIN",
        description="SYS_ADMIN capability",
        risk_level="MEDIUM",
        category="security",
    ),
    SafetyPattern(
        pattern=r"allowPrivilegeEscalation:\s*true",
        description="Privilege escalation enabled",
        risk_level="MEDIUM",
        category="security",
    ),
    SafetyPattern(
        pattern=r"runAsUser:\s*0",
        description="Running as root",
        risk_level="MEDIUM",
        category="security",
    ),
    SafetyPattern(
        pattern=r"runAsNonRoot:\s*false",
        description="Root container allowed",
        risk_level="MEDIUM",
        category="security",
    ),
]


# Docker patterns
DOCKER_PATTERNS: List[SafetyPattern] = [
    # Destructive operations
    SafetyPattern(
        pattern=r"docker\s+system\s+prune\s+-a",
        description="Docker full system prune",
        risk_level="HIGH",
        category="docker",
    ),
    SafetyPattern(
        pattern=r"docker\s+volume\s+prune",
        description="Remove all unused volumes",
        risk_level="HIGH",
        category="docker",
    ),
    SafetyPattern(
        pattern=r"docker\s+container\s+prune\s+-f",
        description="Force remove stopped containers",
        risk_level="MEDIUM",
        category="docker",
    ),
    SafetyPattern(
        pattern=r"docker\s+image\s+prune\s+-a",
        description="Remove all unused images",
        risk_level="MEDIUM",
        category="docker",
    ),
    SafetyPattern(
        pattern=r"docker\s+rm\s+-f",
        description="Force remove container",
        risk_level="MEDIUM",
        category="docker",
    ),
    SafetyPattern(
        pattern=r"docker\s+rmi\s+-f",
        description="Force remove image",
        risk_level="MEDIUM",
        category="docker",
    ),
    # Security patterns
    SafetyPattern(
        pattern=r"--net=host",
        description="Docker host networking",
        risk_level="MEDIUM",
        category="security",
    ),
    SafetyPattern(
        pattern=r"--pid=host",
        description="Docker host PID namespace",
        risk_level="MEDIUM",
        category="security",
    ),
    SafetyPattern(
        pattern=r"--privileged",
        description="Docker privileged mode",
        risk_level="MEDIUM",
        category="security",
    ),
    SafetyPattern(
        pattern=r"-v\s+/:/",
        description="Mount root filesystem",
        risk_level="HIGH",
        category="security",
    ),
    SafetyPattern(
        pattern=r"-v\s+/var/run/docker.sock",
        description="Docker socket mount",
        risk_level="MEDIUM",
        category="security",
    ),
    # Best practices (LOW risk)
    SafetyPattern(
        pattern=r"FROM\s+\S+:latest",
        description="Using latest tag",
        risk_level="LOW",
        category="docker",
    ),
]


# Terraform patterns
TERRAFORM_PATTERNS: List[SafetyPattern] = [
    SafetyPattern(
        pattern=r"terraform\s+destroy(?!\s+--target)",
        description="Full infrastructure destruction",
        risk_level="CRITICAL",
        category="terraform",
    ),
    SafetyPattern(
        pattern=r"terraform\s+destroy\s+--auto-approve",
        description="Auto-approved infrastructure destruction",
        risk_level="CRITICAL",
        category="terraform",
    ),
    SafetyPattern(
        pattern=r"terraform\s+apply\s+--auto-approve",
        description="Auto-approved infrastructure apply",
        risk_level="MEDIUM",
        category="terraform",
    ),
    SafetyPattern(
        pattern=r"terraform\s+state\s+rm",
        description="Remove resource from state",
        risk_level="HIGH",
        category="terraform",
    ),
    SafetyPattern(
        pattern=r"terraform\s+taint",
        description="Mark resource for recreation",
        risk_level="MEDIUM",
        category="terraform",
    ),
    SafetyPattern(
        pattern=r"terraform\s+import",
        description="Import existing infrastructure",
        risk_level="LOW",
        category="terraform",
    ),
]


# Cloud provider patterns
CLOUD_PATTERNS: List[SafetyPattern] = [
    # AWS
    SafetyPattern(
        pattern=r"aws\s+s3\s+rb\s+--force",
        description="Force delete S3 bucket",
        risk_level="CRITICAL",
        category="cloud",
    ),
    SafetyPattern(
        pattern=r"aws\s+ec2\s+terminate-instances",
        description="Terminate EC2 instances",
        risk_level="HIGH",
        category="cloud",
    ),
    SafetyPattern(
        pattern=r"aws\s+rds\s+delete-db-instance",
        description="Delete RDS instance",
        risk_level="CRITICAL",
        category="cloud",
    ),
    SafetyPattern(
        pattern=r"aws\s+iam\s+delete-user",
        description="Delete IAM user",
        risk_level="HIGH",
        category="cloud",
    ),
    # GCP
    SafetyPattern(
        pattern=r"gcloud\s+compute\s+instances\s+delete",
        description="Delete GCP instance",
        risk_level="HIGH",
        category="cloud",
    ),
    SafetyPattern(
        pattern=r"gcloud\s+sql\s+instances\s+delete",
        description="Delete Cloud SQL instance",
        risk_level="CRITICAL",
        category="cloud",
    ),
    # Azure
    SafetyPattern(
        pattern=r"az\s+group\s+delete",
        description="Delete Azure resource group",
        risk_level="CRITICAL",
        category="cloud",
    ),
    SafetyPattern(
        pattern=r"az\s+vm\s+delete",
        description="Delete Azure VM",
        risk_level="HIGH",
        category="cloud",
    ),
]


# =============================================================================
# InfrastructureScanner
# =============================================================================


@dataclass
class InfraScanResult:
    """Result from scanning infrastructure commands or configs.

    Attributes:
        matches: List of matched patterns
        risk_summary: Dict of risk level to count
        has_critical: Whether any CRITICAL patterns matched
        has_high: Whether any HIGH patterns matched
        security_issues: Security-specific matches
    """

    matches: List[SafetyPattern] = field(default_factory=list)
    risk_summary: Dict[str, int] = field(default_factory=dict)
    has_critical: bool = False
    has_high: bool = False
    security_issues: List[SafetyPattern] = field(default_factory=list)

    def add_match(self, pattern: SafetyPattern) -> None:
        """Add a matched pattern."""
        self.matches.append(pattern)
        level = pattern.risk_level
        self.risk_summary[level] = self.risk_summary.get(level, 0) + 1
        if level == "CRITICAL":
            self.has_critical = True
        elif level == "HIGH":
            self.has_high = True
        if pattern.category == "security":
            self.security_issues.append(pattern)


class InfrastructureScanner:
    """Scanner for infrastructure safety patterns.

    Provides comprehensive scanning for dangerous operations in:
    - Kubernetes commands and manifests
    - Docker commands and Dockerfiles
    - Terraform operations
    - Cloud provider CLI commands
    - Security misconfigurations

    Example:
        scanner = InfrastructureScanner()

        # Scan a command
        result = scanner.scan_command("terraform destroy --auto-approve")
        if result.has_critical:
            print("CRITICAL operation detected!")

        # Validate Dockerfile
        warnings = scanner.validate_dockerfile(dockerfile_content)

        # Validate Kubernetes manifest
        issues = scanner.validate_kubernetes_manifest(manifest_yaml)
    """

    def __init__(
        self,
        include_destructive: bool = True,
        include_kubernetes: bool = True,
        include_docker: bool = True,
        include_terraform: bool = True,
        include_cloud: bool = True,
        custom_patterns: Optional[List[SafetyPattern]] = None,
    ):
        """Initialize the scanner.

        Args:
            include_destructive: Include destructive patterns
            include_kubernetes: Include Kubernetes patterns
            include_docker: Include Docker patterns
            include_terraform: Include Terraform patterns
            include_cloud: Include cloud provider patterns
            custom_patterns: Additional custom patterns
        """
        self._patterns: List[SafetyPattern] = []

        if include_destructive:
            self._patterns.extend(DESTRUCTIVE_PATTERNS)
        if include_kubernetes:
            self._patterns.extend(KUBERNETES_PATTERNS)
        if include_docker:
            self._patterns.extend(DOCKER_PATTERNS)
        if include_terraform:
            self._patterns.extend(TERRAFORM_PATTERNS)
        if include_cloud:
            self._patterns.extend(CLOUD_PATTERNS)

        if custom_patterns:
            self._patterns.extend(custom_patterns)

        # Compile patterns for efficiency
        self._compiled_patterns = [
            (re.compile(p.pattern, re.IGNORECASE | re.MULTILINE), p) for p in self._patterns
        ]

    def scan_command(self, command: str) -> InfraScanResult:
        """Scan a command for dangerous patterns.

        Args:
            command: Shell command to scan

        Returns:
            InfraScanResult with matched patterns
        """
        result = InfraScanResult()
        for regex, pattern in self._compiled_patterns:
            if regex.search(command):
                result.add_match(pattern)
        return result

    def scan_content(self, content: str) -> InfraScanResult:
        """Scan content (like YAML) for dangerous patterns.

        Args:
            content: Content to scan (e.g., Kubernetes manifest)

        Returns:
            InfraScanResult with matched patterns
        """
        result = InfraScanResult()
        for regex, pattern in self._compiled_patterns:
            if regex.search(content):
                result.add_match(pattern)
        return result

    def validate_dockerfile(self, content: str) -> List[str]:
        """Validate Dockerfile security best practices.

        Args:
            content: Dockerfile content

        Returns:
            List of security warnings found
        """
        warnings = []

        # Check for latest tag
        if re.search(r"FROM\s+\S+:latest", content, re.IGNORECASE):
            warnings.append("Using ':latest' tag - pin to specific version")

        # Check for root user
        if not re.search(r"USER\s+(?!root)\w+", content):
            warnings.append("No non-root USER specified")

        # Check for COPY with --chown
        if re.search(r"COPY\s+(?!--chown)", content) and "USER" in content:
            warnings.append("Consider using COPY --chown for file ownership")

        # Check for health check
        if "HEALTHCHECK" not in content:
            warnings.append("No HEALTHCHECK instruction")

        # Check for multi-stage build (positive check)
        if content.count("FROM ") > 1:
            pass  # Multi-stage build is good

        # Check for apt-get clean
        if re.search(r"apt-get\s+install", content, re.IGNORECASE):
            if not re.search(r"apt-get\s+clean|rm\s+-rf\s+/var/lib/apt", content):
                warnings.append(
                    "apt-get install without cleanup - add 'apt-get clean && rm -rf /var/lib/apt/lists/*'"
                )

        # Check for ADD vs COPY
        if re.search(r"ADD\s+(?!https?://)", content, re.IGNORECASE):
            warnings.append("Use COPY instead of ADD for local files")

        # Check for sensitive files
        if re.search(r"COPY.*\.env|ADD.*\.env", content, re.IGNORECASE):
            warnings.append("Avoid copying .env files - use ARG or secrets")

        return warnings

    def validate_kubernetes_manifest(self, content: str) -> List[str]:
        """Validate Kubernetes manifest security.

        Args:
            content: Kubernetes YAML manifest content

        Returns:
            List of security warnings found
        """
        warnings = []

        # Check for privileged containers
        if re.search(r"privileged:\s*true", content, re.IGNORECASE):
            warnings.append("Privileged container detected")

        # Check for resource limits
        if "limits:" not in content:
            warnings.append("No resource limits defined")

        # Check for probes
        if "livenessProbe:" not in content:
            warnings.append("No liveness probe defined")
        if "readinessProbe:" not in content:
            warnings.append("No readiness probe defined")

        # Check for security context
        if "securityContext:" not in content:
            warnings.append("No security context defined")

        # Check for latest tag
        if re.search(r"image:\s*\S+:latest", content, re.IGNORECASE):
            warnings.append("Using ':latest' image tag - pin to specific version")

        # Check for host network
        if re.search(r"hostNetwork:\s*true", content, re.IGNORECASE):
            warnings.append("Host network enabled")

        # Check for service account
        if re.search(r"automountServiceAccountToken:\s*true", content, re.IGNORECASE):
            warnings.append("Service account token auto-mount enabled - disable if not needed")

        # Check for run as root
        if re.search(r"runAsUser:\s*0", content):
            warnings.append("Running as root (UID 0)")

        # Check for privilege escalation
        if re.search(r"allowPrivilegeEscalation:\s*true", content, re.IGNORECASE):
            warnings.append("Privilege escalation allowed")

        return warnings

    def get_patterns_by_category(self, category: InfraPatternCategory) -> List[SafetyPattern]:
        """Get patterns for a specific category.

        Args:
            category: Category to filter by

        Returns:
            List of patterns in category
        """
        return [p for p in self._patterns if p.category == category.value]

    def get_patterns_by_risk(self, risk_level: str) -> List[SafetyPattern]:
        """Get patterns by risk level.

        Args:
            risk_level: Risk level to filter by

        Returns:
            List of patterns at risk level
        """
        return [p for p in self._patterns if p.risk_level == risk_level]

    @property
    def all_patterns(self) -> List[SafetyPattern]:
        """Get all patterns."""
        return self._patterns.copy()


# =============================================================================
# Convenience Functions
# =============================================================================


def scan_infrastructure_command(command: str) -> List[SafetyPattern]:
    """Scan an infrastructure command for dangerous patterns.

    Convenience function for quick scanning.

    Args:
        command: Command to scan

    Returns:
        List of matched patterns
    """
    scanner = InfrastructureScanner()
    return scanner.scan_command(command).matches


def validate_dockerfile(content: str) -> List[str]:
    """Validate Dockerfile security best practices.

    Convenience function for quick validation.

    Args:
        content: Dockerfile content

    Returns:
        List of security warnings
    """
    scanner = InfrastructureScanner()
    return scanner.validate_dockerfile(content)


def validate_kubernetes_manifest(content: str) -> List[str]:
    """Validate Kubernetes manifest security.

    Convenience function for quick validation.

    Args:
        content: Kubernetes YAML content

    Returns:
        List of security warnings
    """
    scanner = InfrastructureScanner()
    return scanner.validate_kubernetes_manifest(content)


def get_all_infrastructure_patterns() -> List[SafetyPattern]:
    """Get all infrastructure safety patterns.

    Returns:
        Combined list of all patterns
    """
    return (
        DESTRUCTIVE_PATTERNS
        + KUBERNETES_PATTERNS
        + DOCKER_PATTERNS
        + TERRAFORM_PATTERNS
        + CLOUD_PATTERNS
    )


def get_safety_reminders() -> List[str]:
    """Get safety reminders for infrastructure work.

    Returns:
        List of safety reminder strings
    """
    return [
        "Never commit secrets to version control",
        "Use environment variables or secrets managers for credentials",
        "Test infrastructure changes in non-production first",
        "Enable audit logging for compliance",
        "Use least-privilege permissions",
        "Pin all dependency and image versions",
        "Enable resource limits for all containers",
        "Use network policies to restrict pod communication",
        "Run containers as non-root users",
        "Enable pod security standards",
    ]


__all__ = [
    # Enums
    "InfraPatternCategory",
    "RiskLevel",
    # Pattern lists
    "DESTRUCTIVE_PATTERNS",
    "KUBERNETES_PATTERNS",
    "DOCKER_PATTERNS",
    "TERRAFORM_PATTERNS",
    "CLOUD_PATTERNS",
    # Classes
    "InfraScanResult",
    "InfrastructureScanner",
    # Functions
    "scan_infrastructure_command",
    "validate_dockerfile",
    "validate_kubernetes_manifest",
    "get_all_infrastructure_patterns",
    "get_safety_reminders",
]
