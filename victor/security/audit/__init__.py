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

"""Compliance & Audit Module - Enterprise governance for Victor.

This module provides comprehensive audit logging and compliance
checking for enterprise environments.

Supported Compliance Frameworks:
- SOC 2 Type II
- GDPR
- HIPAA
- PCI DSS
- ISO 27001
- SOX

Features:
- Audit event logging with rotation
- Compliance rule checking
- Report generation
- Data retention policies
- PII detection

Usage:
    from victor.security.audit import AuditManager

    # Get singleton instance
    manager = AuditManager.get_instance()

    # Log events
    await manager.log_file_operation("read", "/path/to/file")
    await manager.log_tool_execution("search", {"query": "foo"})

    # Generate report
    report = await manager.generate_report(days=30)
    print(f"Total events: {report.total_events}")
"""

from .checker import DefaultComplianceChecker, get_compliance_summary
from .logger import FileAuditLogger, create_event
from .manager import AuditManager
from .protocol import (
    AuditConfig,
    AuditEvent,
    AuditEventType,
    AuditLoggerProtocol,
    AuditReport,
    ComplianceCheckerProtocol,
    ComplianceFramework,
    ComplianceRule,
    ComplianceViolation,
    DEFAULT_COMPLIANCE_RULES,
    RetentionPolicy,
    Severity,
)

__all__ = [
    # Manager
    "AuditManager",
    # Protocols
    "AuditLoggerProtocol",
    "ComplianceCheckerProtocol",
    # Data classes
    "AuditEventType",
    "Severity",
    "ComplianceFramework",
    "AuditEvent",
    "ComplianceRule",
    "ComplianceViolation",
    "AuditReport",
    "AuditConfig",
    "RetentionPolicy",
    # Implementations
    "FileAuditLogger",
    "DefaultComplianceChecker",
    # Utilities
    "create_event",
    "get_compliance_summary",
    # Constants
    "DEFAULT_COMPLIANCE_RULES",
]
