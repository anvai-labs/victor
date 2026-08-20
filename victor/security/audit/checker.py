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

"""Compliance Checker - Rule-based compliance validation.

This module provides compliance checking against various frameworks
and generates audit reports.
"""

import logging
import re
import uuid
from datetime import datetime
from typing import Any

from .protocol import (
    AuditEvent,
    AuditReport,
    ComplianceCheckerProtocol,
    ComplianceFramework,
    ComplianceRule,
    ComplianceViolation,
    DEFAULT_COMPLIANCE_RULES,
)
from .logger import FileAuditLogger

logger = logging.getLogger(__name__)


# PII detection patterns
PII_PATTERNS = [
    # Email
    (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "email"),
    # SSN (US)
    (r"\b\d{3}-\d{2}-\d{4}\b", "ssn"),
    # Phone numbers
    (r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "phone"),
    # Credit card numbers
    (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "credit_card"),
    # IP addresses
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "ip_address"),
]


class DefaultComplianceChecker(ComplianceCheckerProtocol):
    """Default implementation of compliance checking."""

    def __init__(
        self,
        audit_logger: FileAuditLogger | None = None,
        rules: list[ComplianceRule] | None = None,
    ):
        """Initialize the compliance checker.

        Args:
            audit_logger: Logger for querying events
            rules: Custom rules to use (defaults to DEFAULT_COMPLIANCE_RULES)
        """
        self._logger = audit_logger or FileAuditLogger()
        self._rules = rules or list(DEFAULT_COMPLIANCE_RULES)

    async def check_event(
        self, event: AuditEvent, rules: list[ComplianceRule] | None = None
    ) -> list[ComplianceViolation]:
        """Check an event against compliance rules."""
        violations = []
        rules_to_check = rules or self._rules

        for rule in rules_to_check:
            # Check if rule applies to this event type
            if event.event_type not in rule.event_types:
                continue

            # Check required fields
            event_dict = event.to_dict()
            for field in rule.required_fields:
                if field not in event_dict or event_dict[field] is None:
                    violations.append(
                        ComplianceViolation(
                            violation_id=str(uuid.uuid4()),
                            rule=rule,
                            event=event,
                            violation_type="missing_field",
                            message=f"Required field '{field}' is missing for {rule.name}",
                        )
                    )

        return violations

    async def generate_report(
        self,
        start_date: datetime,
        end_date: datetime,
        framework: ComplianceFramework | None = None,
    ) -> AuditReport:
        """Generate a compliance report for a time period."""
        # Query events
        events = await self._logger.query_events(
            start_date=start_date,
            end_date=end_date,
            limit=100000,
        )

        # Count by type
        events_by_type: dict[str, int] = {}
        for event in events:
            type_key = event.event_type.value
            events_by_type[type_key] = events_by_type.get(type_key, 0) + 1

        # Count by severity
        events_by_severity: dict[str, int] = {}
        for event in events:
            sev_key = event.severity.value
            events_by_severity[sev_key] = events_by_severity.get(sev_key, 0) + 1

        # Check for violations
        violations = []
        rules_to_check = self._rules
        if framework:
            rules_to_check = [r for r in self._rules if r.framework == framework]

        for event in events:
            event_violations = await self.check_event(event, rules_to_check)
            violations.extend(event_violations)

        return AuditReport(
            report_id=str(uuid.uuid4()),
            start_date=start_date,
            end_date=end_date,
            framework=framework,
            total_events=len(events),
            events_by_type=events_by_type,
            events_by_severity=events_by_severity,
            violations=violations,
        )

    def get_rules_for_framework(self, framework: ComplianceFramework) -> list[ComplianceRule]:
        """Get compliance rules for a specific framework."""
        return [r for r in self._rules if r.framework == framework]

    def add_rule(self, rule: ComplianceRule) -> None:
        """Add a custom compliance rule."""
        self._rules.append(rule)

    def detect_pii(self, text: str) -> list[dict[str, Any]]:
        """Detect PII in text content.

        Args:
            text: Text to scan for PII

        Returns:
            List of detected PII with type and position
        """
        detected = []

        for pattern, pii_type in PII_PATTERNS:
            for match in re.finditer(pattern, text):
                detected.append(
                    {
                        "type": pii_type,
                        "start": match.start(),
                        "end": match.end(),
                        "masked": self._mask_pii(match.group(), pii_type),
                    }
                )

        return detected

    def _mask_pii(self, value: str, pii_type: str) -> str:
        """Mask PII value for logging."""
        if pii_type == "email":
            parts = value.split("@")
            if len(parts) == 2:
                return f"{parts[0][:2]}***@{parts[1]}"
        elif pii_type == "ssn":
            return "***-**-" + value[-4:]
        elif pii_type == "credit_card":
            return "****-****-****-" + value[-4:]
        elif pii_type == "phone":
            return "***-***-" + value[-4:]

        # Default masking
        if len(value) > 4:
            return value[:2] + "*" * (len(value) - 4) + value[-2:]
        return "*" * len(value)


def get_compliance_summary(
    events: list[AuditEvent],
    framework: ComplianceFramework | None = None,
) -> dict[str, Any]:
    """Generate a quick compliance summary.

    Args:
        events: List of audit events
        framework: Optional framework to focus on

    Returns:
        Summary dictionary
    """
    checker = DefaultComplianceChecker()
    rules = checker._rules
    if framework:
        rules = checker.get_rules_for_framework(framework)

    # Check coverage
    covered_event_types = set()
    for rule in rules:
        covered_event_types.update(rule.event_types)

    actual_event_types = {e.event_type for e in events}

    return {
        "total_events": len(events),
        "covered_event_types": len(covered_event_types & actual_event_types),
        "total_rules": len(rules),
        "frameworks": list({r.framework.value for r in rules}),
        "compliant": True,  # Simplified
    }
