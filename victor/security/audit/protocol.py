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

"""Audit Protocol - Unified interface for compliance and audit logging.

This module defines the abstract interface and data structures for
enterprise compliance tracking, audit logging, and governance reporting.

Supports compliance frameworks:
- SOC 2 Type II
- GDPR
- HIPAA
- PCI DSS
- ISO 27001
- SOX
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class AuditEventType(str, Enum):
    """Types of audit events."""

    # File operations
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    FILE_RENAME = "file_rename"

    # Code operations
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_REFACTOR = "code_refactor"

    # Tool operations
    TOOL_EXECUTION = "tool_execution"
    TOOL_FAILURE = "tool_failure"

    # Security events
    SECRET_DETECTED = "secret_detected"
    SECRET_ACCESS = "secret_access"  # API key / credential access attempts
    SECURITY_EVENT = (
        "security_event"  # General security events (HOME manipulation, path traversal, etc.)
    )
    SECURITY_SCAN = "security_scan"
    ACCESS_DENIED = "access_denied"

    # Session events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    MODEL_SWITCH = "model_switch"

    # Data events
    SENSITIVE_DATA_ACCESS = "sensitive_data_access"
    PII_DETECTED = "pii_detected"
    DATA_EXPORT = "data_export"


class AuditSeverity(str, Enum):
    """Event severity levels for audit logging.

    Renamed from Severity to be semantically distinct from other severity types:
    - CVESeverity (victor.security.protocol): CVE/CVSS-based severity
    - AuditSeverity (here): Audit event severity (like log levels)
    - IaCSeverity: IaC issue severity
    - ReviewSeverity: Code review severity
    """

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Backward compatibility alias
Severity = AuditSeverity


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""

    SOC2 = "soc2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    SOX = "sox"
    CUSTOM = "custom"


@dataclass
class AuditEvent:
    """A single audit event."""

    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    severity: AuditSeverity
    actor: str  # User or system identifier
    action: str  # Description of the action
    resource: str | None = None  # Resource affected (file path, tool name, etc.)
    outcome: str = "success"  # success, failure, partial
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    ip_address: str | None = None
    correlation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "outcome": self.outcome,
            "metadata": self.metadata,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        return cls(
            event_id=data["event_id"],
            event_type=AuditEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            severity=AuditSeverity(data["severity"]),
            actor=data["actor"],
            action=data["action"],
            resource=data.get("resource"),
            outcome=data.get("outcome", "success"),
            metadata=data.get("metadata", {}),
            session_id=data.get("session_id"),
            ip_address=data.get("ip_address"),
            correlation_id=data.get("correlation_id"),
        )


@dataclass
class ComplianceRule:
    """A compliance rule definition."""

    rule_id: str
    framework: ComplianceFramework
    name: str
    description: str
    event_types: list[AuditEventType] = field(default_factory=list)
    required_fields: list[str] = field(default_factory=list)
    retention_days: int = 365
    alert_on_violation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "framework": self.framework.value,
            "name": self.name,
            "description": self.description,
            "event_types": [e.value for e in self.event_types],
            "required_fields": self.required_fields,
            "retention_days": self.retention_days,
            "alert_on_violation": self.alert_on_violation,
        }


@dataclass
class ComplianceViolation:
    """A compliance rule violation."""

    violation_id: str
    rule: ComplianceRule
    event: AuditEvent
    violation_type: str
    message: str
    detected_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    resolution_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "rule_id": self.rule.rule_id,
            "event_id": self.event.event_id,
            "violation_type": self.violation_type,
            "message": self.message,
            "detected_at": self.detected_at.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "resolution_notes": self.resolution_notes,
        }


@dataclass
class AuditReport:
    """Audit report for a time period."""

    report_id: str
    start_date: datetime
    end_date: datetime
    framework: ComplianceFramework | None
    total_events: int
    events_by_type: dict[str, int] = field(default_factory=dict)
    events_by_severity: dict[str, int] = field(default_factory=dict)
    violations: list[ComplianceViolation] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "framework": self.framework.value if self.framework else None,
            "total_events": self.total_events,
            "events_by_type": self.events_by_type,
            "events_by_severity": self.events_by_severity,
            "violations": [v.to_dict() for v in self.violations],
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class RetentionPolicy:
    """Data retention policy configuration."""

    default_retention_days: int = 365
    framework_overrides: dict[ComplianceFramework, int] = field(default_factory=dict)
    sensitive_data_retention_days: int = 90
    auto_purge: bool = True
    archive_before_purge: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_retention_days": self.default_retention_days,
            "framework_overrides": {k.value: v for k, v in self.framework_overrides.items()},
            "sensitive_data_retention_days": self.sensitive_data_retention_days,
            "auto_purge": self.auto_purge,
            "archive_before_purge": self.archive_before_purge,
        }


@dataclass
class AuditConfig:
    """Configuration for audit logging."""

    enabled: bool = True
    log_file_operations: bool = True
    log_tool_executions: bool = True
    log_model_interactions: bool = True
    detect_secrets: bool = True
    detect_pii: bool = True
    frameworks: list[ComplianceFramework] = field(default_factory=list)
    retention: RetentionPolicy = field(default_factory=RetentionPolicy)
    export_format: str = "json"  # json, csv, syslog

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "log_file_operations": self.log_file_operations,
            "log_tool_executions": self.log_tool_executions,
            "log_model_interactions": self.log_model_interactions,
            "detect_secrets": self.detect_secrets,
            "detect_pii": self.detect_pii,
            "frameworks": [f.value for f in self.frameworks],
            "retention": self.retention.to_dict(),
            "export_format": self.export_format,
        }


class AuditLoggerProtocol(ABC):
    """Abstract protocol for audit logging.

    Implementations provide different storage backends for audit events.
    """

    @abstractmethod
    async def log_event(self, event: AuditEvent) -> None:
        """Log an audit event.

        Args:
            event: Event to log
        """
        ...

    @abstractmethod
    async def query_events(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        severity: AuditSeverity | None = None,
        actor: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events.

        Args:
            start_date: Start of time range
            end_date: End of time range
            event_types: Filter by event types
            severity: Filter by severity
            actor: Filter by actor
            limit: Maximum events to return

        Returns:
            List of matching events
        """
        ...

    @abstractmethod
    async def export_events(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "json",
    ) -> Path:
        """Export events to a file.

        Args:
            start_date: Start of time range
            end_date: End of time range
            format: Export format (json, csv)

        Returns:
            Path to exported file
        """
        ...


class ComplianceCheckerProtocol(ABC):
    """Abstract protocol for compliance checking."""

    @abstractmethod
    async def check_event(
        self, event: AuditEvent, rules: list[ComplianceRule]
    ) -> list[ComplianceViolation]:
        """Check an event against compliance rules.

        Args:
            event: Event to check
            rules: Rules to check against

        Returns:
            List of violations found
        """
        ...

    @abstractmethod
    async def generate_report(
        self,
        start_date: datetime,
        end_date: datetime,
        framework: ComplianceFramework | None = None,
    ) -> AuditReport:
        """Generate a compliance report.

        Args:
            start_date: Start of reporting period
            end_date: End of reporting period
            framework: Specific framework to report on

        Returns:
            Audit report
        """
        ...


# Default compliance rules
DEFAULT_COMPLIANCE_RULES: list[ComplianceRule] = [
    ComplianceRule(
        rule_id="SOC2-AC-1",
        framework=ComplianceFramework.SOC2,
        name="Access Control Logging",
        description="All access attempts must be logged",
        event_types=[AuditEventType.FILE_READ, AuditEventType.ACCESS_DENIED],
        required_fields=["actor", "resource", "timestamp"],
        retention_days=365,
    ),
    ComplianceRule(
        rule_id="SOC2-CC-6.1",
        framework=ComplianceFramework.SOC2,
        name="Change Management",
        description="All changes to code must be logged",
        event_types=[
            AuditEventType.FILE_WRITE,
            AuditEventType.FILE_DELETE,
            AuditEventType.CODE_GENERATION,
        ],
        required_fields=["actor", "resource", "timestamp", "outcome"],
        retention_days=365,
    ),
    ComplianceRule(
        rule_id="GDPR-A30",
        framework=ComplianceFramework.GDPR,
        name="Records of Processing",
        description="Processing activities involving personal data must be recorded",
        event_types=[
            AuditEventType.SENSITIVE_DATA_ACCESS,
            AuditEventType.PII_DETECTED,
            AuditEventType.DATA_EXPORT,
        ],
        required_fields=["actor", "resource", "timestamp"],
        retention_days=180,
    ),
    ComplianceRule(
        rule_id="HIPAA-164.312",
        framework=ComplianceFramework.HIPAA,
        name="Audit Controls",
        description="Mechanisms to record and examine activity",
        event_types=[
            AuditEventType.SENSITIVE_DATA_ACCESS,
            AuditEventType.FILE_READ,
            AuditEventType.FILE_WRITE,
        ],
        required_fields=["actor", "resource", "timestamp", "outcome"],
        retention_days=2190,  # 6 years
    ),
    ComplianceRule(
        rule_id="PCI-10.2",
        framework=ComplianceFramework.PCI_DSS,
        name="Audit Trails",
        description="Track and monitor all access to cardholder data",
        event_types=[
            AuditEventType.SENSITIVE_DATA_ACCESS,
            AuditEventType.SECRET_DETECTED,
        ],
        required_fields=["actor", "resource", "timestamp", "ip_address"],
        retention_days=365,
        alert_on_violation=True,
    ),
]
