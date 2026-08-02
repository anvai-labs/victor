from __future__ import annotations

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

"""IaC Scanner Protocol - Unified interface for infrastructure-as-code analysis.

This module defines the abstract interface and data structures for
scanning Infrastructure-as-Code files for security issues, misconfigurations,
and best practice violations.

Supported IaC formats:
- Terraform (.tf, .tfvars)
- Docker (Dockerfile, docker-compose.yml)
- Kubernetes (*.yaml, *.yml manifests)
- CloudFormation (.yaml, .json)
- Ansible (*.yml playbooks)
- Helm Charts
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class IaCPlatform(str, Enum):
    """Supported IaC platforms."""

    TERRAFORM = "terraform"
    DOCKER = "docker"
    DOCKER_COMPOSE = "docker_compose"
    KUBERNETES = "kubernetes"
    CLOUDFORMATION = "cloudformation"
    ANSIBLE = "ansible"
    HELM = "helm"
    UNKNOWN = "unknown"


class IaCSeverity(str, Enum):
    """Issue severity levels for IaC scanning, aligned with security standards.

    Renamed from Severity to be semantically distinct from other severity types:
    - CVESeverity (victor.security.protocol): CVE/CVSS-based severity
    - AuditSeverity: Audit event severity (like log levels)
    - IaCSeverity (here): IaC issue severity
    - ReviewSeverity: Code review severity
    """

    CRITICAL = "critical"  # Immediate exploitation risk
    HIGH = "high"  # Significant security weakness
    MEDIUM = "medium"  # Moderate risk or best practice violation
    LOW = "low"  # Minor issue or informational
    INFO = "info"  # Informational finding


# Backward compatibility alias
Severity = IaCSeverity


class Category(str, Enum):
    """Issue categories for classification."""

    SECRETS = "secrets"  # Hardcoded secrets, API keys
    PERMISSIONS = "permissions"  # Overly permissive IAM, RBAC
    ENCRYPTION = "encryption"  # Missing or weak encryption
    NETWORK = "network"  # Network exposure, firewall rules
    AUTHENTICATION = "authentication"  # Auth/authz issues
    LOGGING = "logging"  # Missing audit/logging config
    BACKUP = "backup"  # Missing backup configuration
    COMPLIANCE = "compliance"  # Regulatory compliance issues
    BEST_PRACTICE = "best_practice"  # General best practice violations
    VULNERABILITY = "vulnerability"  # Known CVEs in base images


@dataclass
class IaCResource:
    """A resource defined in IaC configuration."""

    resource_type: str  # aws_s3_bucket, kubernetes_deployment, etc.
    name: str
    file_path: Path
    line_number: int
    properties: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_type": self.resource_type,
            "name": self.name,
            "file_path": str(self.file_path),
            "line_number": self.line_number,
            "properties": self.properties,
            "tags": self.tags,
        }


@dataclass
class IaCFinding:
    """A security or configuration finding in IaC."""

    rule_id: str  # Unique identifier for the rule (e.g., AWS001, K8S005)
    severity: IaCSeverity
    category: Category
    message: str
    description: str
    file_path: Path
    line_number: int = 0
    resource_type: str | None = None
    resource_name: str | None = None
    remediation: str = ""
    documentation_url: str = ""
    false_positive_likely: bool = False
    cwe_id: str | None = None  # CWE reference
    cvss_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "description": self.description,
            "file_path": str(self.file_path),
            "line_number": self.line_number,
            "resource_type": self.resource_type,
            "resource_name": self.resource_name,
            "remediation": self.remediation,
            "documentation_url": self.documentation_url,
            "false_positive_likely": self.false_positive_likely,
            "cwe_id": self.cwe_id,
            "cvss_score": self.cvss_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IaCFinding":
        return cls(
            rule_id=data["rule_id"],
            severity=IaCSeverity(data["severity"]),
            category=Category(data["category"]),
            message=data["message"],
            description=data.get("description", ""),
            file_path=Path(data["file_path"]),
            line_number=data.get("line_number", 0),
            resource_type=data.get("resource_type"),
            resource_name=data.get("resource_name"),
            remediation=data.get("remediation", ""),
            documentation_url=data.get("documentation_url", ""),
            false_positive_likely=data.get("false_positive_likely", False),
            cwe_id=data.get("cwe_id"),
            cvss_score=data.get("cvss_score"),
        )


@dataclass
class IaCConfig:
    """Parsed IaC configuration."""

    platform: IaCPlatform
    file_path: Path
    resources: list[IaCResource] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    raw_content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform.value,
            "file_path": str(self.file_path),
            "resources": [r.to_dict() for r in self.resources],
            "variables": self.variables,
        }


@dataclass
class IaCScanResult:
    """Complete scan result for Infrastructure-as-Code files.

    Renamed from ScanResult to be semantically distinct:
    - IaCScanResult (here): Infrastructure-as-Code scan results
    - SafetyScanResult (victor.security.safety.code_patterns): Safety pattern matching results
    """

    configs: list[IaCConfig]
    findings: list[IaCFinding]
    files_scanned: int
    total_resources: int
    scan_duration_ms: int
    scanned_at: datetime = field(default_factory=datetime.now)

    # Summary counts by severity
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    info_count: int = 0

    def __post_init__(self):
        """Calculate summary counts."""
        for finding in self.findings:
            if finding.severity == IaCSeverity.CRITICAL:
                self.critical_count += 1
            elif finding.severity == IaCSeverity.HIGH:
                self.high_count += 1
            elif finding.severity == IaCSeverity.MEDIUM:
                self.medium_count += 1
            elif finding.severity == IaCSeverity.LOW:
                self.low_count += 1
            else:
                self.info_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "configs": [c.to_dict() for c in self.configs],
            "findings": [f.to_dict() for f in self.findings],
            "files_scanned": self.files_scanned,
            "total_resources": self.total_resources,
            "scan_duration_ms": self.scan_duration_ms,
            "scanned_at": self.scanned_at.isoformat(),
            "summary": {
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
                "info": self.info_count,
                "total": len(self.findings),
            },
        }


# Backward compatibility alias
ScanResult = IaCScanResult


@dataclass
class ScanPolicy:
    """Policy configuration for IaC scanning."""

    enabled_platforms: list[IaCPlatform] = field(default_factory=lambda: list(IaCPlatform))
    min_severity: IaCSeverity = IaCSeverity.LOW
    excluded_rules: list[str] = field(default_factory=list)
    excluded_paths: list[str] = field(default_factory=list)
    fail_on_severity: IaCSeverity = IaCSeverity.HIGH  # Fail CI if findings above this
    custom_rules: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled_platforms": [p.value for p in self.enabled_platforms],
            "min_severity": self.min_severity.value,
            "excluded_rules": self.excluded_rules,
            "excluded_paths": self.excluded_paths,
            "fail_on_severity": self.fail_on_severity.value,
        }


class IaCScannerProtocol(ABC):
    """Abstract protocol for IaC scanning.

    Implementations provide platform-specific parsing and security
    analysis for different IaC formats.
    """

    @property
    @abstractmethod
    def platform(self) -> IaCPlatform:
        """Return the platform this scanner handles."""
        ...

    @abstractmethod
    async def detect_files(self, root_path: Path) -> list[Path]:
        """Find IaC files in a project.

        Args:
            root_path: Project root directory

        Returns:
            List of paths to IaC files
        """
        ...

    @abstractmethod
    async def parse_config(self, config_path: Path) -> IaCConfig:
        """Parse an IaC configuration file.

        Args:
            config_path: Path to the config file

        Returns:
            Parsed IaC configuration
        """
        ...

    @abstractmethod
    async def scan(self, config: IaCConfig, policy: ScanPolicy | None = None) -> list[IaCFinding]:
        """Scan an IaC configuration for security issues.

        Args:
            config: Parsed IaC configuration
            policy: Optional scan policy for filtering

        Returns:
            List of security findings
        """
        ...


class SecretPattern:
    """Pattern for detecting secrets in IaC files."""

    def __init__(
        self,
        name: str,
        pattern: str,
        severity: IaCSeverity = IaCSeverity.CRITICAL,
        description: str = "",
    ):
        self.name = name
        self.pattern = pattern
        self.severity = severity
        self.description = description or f"Detected hardcoded {name}"


# Common secret patterns used across all scanners
COMMON_SECRET_PATTERNS = [
    SecretPattern(
        "AWS Access Key",
        r"AKIA[0-9A-Z]{16}",
        IaCSeverity.CRITICAL,
        "AWS Access Key ID exposed in configuration",
    ),
    SecretPattern(
        "AWS Secret Key",
        r"['\"][0-9a-zA-Z/+]{40}['\"]",
        IaCSeverity.CRITICAL,
        "Potential AWS Secret Access Key in configuration",
    ),
    SecretPattern(
        "Private Key",
        r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
        IaCSeverity.CRITICAL,
        "Private key embedded in configuration",
    ),
    SecretPattern(
        "Generic API Key",
        r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"][a-zA-Z0-9]{16,}['\"]",
        IaCSeverity.HIGH,
        "API key hardcoded in configuration",
    ),
    SecretPattern(
        "Password",
        r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"][^'\"$]+['\"]",
        IaCSeverity.HIGH,
        "Password hardcoded in configuration",
    ),
    SecretPattern(
        "Bearer Token",
        r"(?i)bearer\s+[a-zA-Z0-9_\-\.=]+",
        IaCSeverity.HIGH,
        "Bearer token exposed in configuration",
    ),
    SecretPattern(
        "GitHub Token",
        r"ghp_[0-9a-zA-Z]{36}|gho_[0-9a-zA-Z]{36}|github_pat_[0-9a-zA-Z]{22}_[0-9a-zA-Z]{59}",
        IaCSeverity.CRITICAL,
        "GitHub personal access token exposed",
    ),
    SecretPattern(
        "Slack Token",
        r"xox[baprs]-[0-9a-zA-Z]{10,48}",
        IaCSeverity.HIGH,
        "Slack token exposed in configuration",
    ),
    SecretPattern(
        "JWT Token",
        r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*",
        IaCSeverity.HIGH,
        "JWT token exposed in configuration",
    ),
    SecretPattern(
        "Google API Key",
        r"AIza[0-9A-Za-z\-_]{35}",
        IaCSeverity.HIGH,
        "Google API key exposed in configuration",
    ),
]
