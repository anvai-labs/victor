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

"""Audit Tool - Compliance and audit logging for Victor tools.

This tool provides integration with Victor's tool system for
audit logging, compliance checking, and report generation.
"""

import logging
from typing import Any

from victor.security.audit import (
    AuditManager,
    AuditReport,
    ComplianceFramework,
)
from victor.tools.base import (
    AccessMode,
    BaseTool,
    CostTier,
    DangerLevel,
    Priority,
    ToolMetadata,
    ToolResult,
)
from victor.tools.tool_names import ToolNames

logger = logging.getLogger(__name__)

# Lazy-loaded presentation adapter for icons
_presentation = None


def _get_icon(name: str) -> str:
    """Get icon from presentation adapter (lazy initialization)."""
    global _presentation
    if _presentation is None:
        from victor.agent.presentation import create_presentation_adapter

        _presentation = create_presentation_adapter()
    return _presentation.icon(name, with_color=False)


class AuditTool(BaseTool):
    """Tool for audit logging and compliance checking."""

    name = ToolNames.AUDIT
    description = """Access audit logs and compliance reports.

Supported frameworks:
- SOC 2 Type II
- GDPR
- HIPAA
- PCI DSS
- ISO 27001

Actions:
- summary: Get audit activity summary
- report: Generate compliance report
- query: Query audit events
- compliance: Check compliance status for a framework
- export: Export audit logs to file"""

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["summary", "report", "query", "compliance", "export"],
                "description": "Action to perform",
            },
            "framework": {
                "type": "string",
                "enum": ["soc2", "gdpr", "hipaa", "pci_dss", "iso_27001"],
                "description": "Compliance framework for report/compliance actions",
            },
            "days": {
                "type": "integer",
                "description": "Number of days to include (default: 7 for summary, 30 for report)",
            },
            "event_type": {
                "type": "string",
                "description": "Filter by event type for query action",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum events to return for query action (default: 50)",
            },
            "format": {
                "type": "string",
                "enum": ["json", "csv"],
                "description": "Export format (default: json)",
            },
        },
        "required": ["action"],
    }

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.FREE

    @property
    def priority(self) -> Priority:
        """Tool priority for selection availability."""
        return Priority.LOW  # Specialized compliance tool

    @property
    def access_mode(self) -> AccessMode:
        """Tool access mode for approval tracking."""
        return AccessMode.MIXED  # Reads logs and can export files

    @property
    def danger_level(self) -> DangerLevel:
        """Danger level for warning/confirmation logic."""
        return DangerLevel.SAFE  # No harmful side effects

    @property
    def metadata(self) -> ToolMetadata:
        """Inline semantic metadata for dynamic tool selection."""
        return ToolMetadata(
            category="audit",
            keywords=[
                "audit",
                "compliance",
                "soc2",
                "gdpr",
                "hipaa",
                "pci dss",
                "iso 27001",
                "audit log",
                "security audit",
                "compliance report",
                "pii detection",
                "data retention",
                "audit trail",
            ],
            use_cases=[
                "audit logging",
                "compliance checking",
                "SOC2 compliance",
                "GDPR compliance",
                "HIPAA compliance",
                "PII detection",
                "audit reports",
            ],
            examples=[
                "generating compliance report",
                "checking SOC2 compliance status",
                "viewing audit logs",
                "detecting PII in operations",
                "exporting audit logs for review",
            ],
            priority_hints=[
                "Use for compliance and audit requirements",
                "Supports multiple compliance frameworks",
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute audit action."""
        action = kwargs.get("action", "summary")
        framework_str = kwargs.get("framework")
        days = kwargs.get("days")
        event_type = kwargs.get("event_type")
        limit = kwargs.get("limit", 50)
        export_format = kwargs.get("format", "json")

        try:
            manager = AuditManager.get_instance()

            if action == "summary":
                days = days or 7
                summary = await manager.get_summary(days=days)
                return ToolResult(
                    success=True,
                    output=self._format_summary(summary),
                    metadata=summary,
                )

            elif action == "report":
                days = days or 30
                framework = ComplianceFramework(framework_str) if framework_str else None
                report = await manager.generate_report(days=days, framework=framework)
                return ToolResult(
                    success=True,
                    output=self._format_report(report),
                    metadata=report.to_dict(),
                )

            elif action == "query":
                from datetime import datetime, timedelta
                from victor.security.audit import AuditEventType

                days = days or 7
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)

                event_types = None
                if event_type:
                    try:
                        event_types = [AuditEventType(event_type)]
                    except ValueError:
                        return ToolResult(
                            success=False,
                            output=f"Invalid event type: {event_type}",
                            error="Invalid parameter",
                        )

                events = await manager.query_events(
                    start_date=start_date,
                    end_date=end_date,
                    event_types=event_types,
                    limit=limit,
                )

                return ToolResult(
                    success=True,
                    output=self._format_events(events),
                    metadata={"count": len(events)},
                )

            elif action == "compliance":
                if not framework_str:
                    return ToolResult(
                        success=False,
                        output="framework is required for compliance action",
                        error="Missing parameter",
                    )

                framework = ComplianceFramework(framework_str)
                status = await manager.check_compliance(framework)
                return ToolResult(
                    success=True,
                    output=self._format_compliance(status),
                    metadata=status,
                )

            elif action == "export":
                days = days or 30
                export_path = await manager.export_audit_log(
                    days=days,
                    format=export_format,
                )
                return ToolResult(
                    success=True,
                    output=f"{_get_icon('success')} Exported audit log to:\n{export_path}",
                    metadata={"path": str(export_path)},
                )

            else:
                return ToolResult(
                    success=False,
                    output=f"Unknown action: {action}",
                    error="Invalid action",
                )

        except Exception as e:
            logger.exception(f"Audit operation failed: {e}")
            return ToolResult(
                success=False,
                output=f"Audit operation failed: {e}",
                error=str(e),
            )

    def _format_summary(self, summary: dict[str, Any]) -> str:
        """Format audit summary."""
        lines = ["**Audit Activity Summary**", ""]

        lines.append(f"**Period:** Last {summary['period_days']} days")
        lines.append(f"**Total Events:** {summary['total_events']}")
        lines.append("")

        # Compliance status
        status_icon = (
            _get_icon("success")
            if summary["compliance_status"] == "compliant"
            else _get_icon("warning")
        )
        lines.append(f"**Compliance Status:** {status_icon} {summary['compliance_status']}")
        if summary["violations"] > 0:
            lines.append(f"**Violations:** {summary['violations']}")
        lines.append("")

        # By type
        if summary["events_by_type"]:
            lines.append("**Events by Type:**")
            for event_type, count in sorted(
                summary["events_by_type"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]:
                lines.append(f"- {event_type}: {count}")
            lines.append("")

        # By severity
        if summary["events_by_severity"]:
            lines.append("**Events by Severity:**")
            severity_icons = {
                "critical": _get_icon("level_critical"),
                "error": _get_icon("level_high"),
                "warning": _get_icon("level_medium"),
                "info": _get_icon("level_info"),
                "debug": _get_icon("level_unknown"),
            }
            for sev, count in summary["events_by_severity"].items():
                icon = severity_icons.get(sev, _get_icon("level_unknown"))
                lines.append(f"- {icon} {sev}: {count}")

        return "\n".join(lines)

    def _format_report(self, report: AuditReport) -> str:
        """Format audit report."""
        lines = ["**Audit Report**", ""]

        lines.append(
            f"**Period:** {report.start_date.strftime('%Y-%m-%d')} to {report.end_date.strftime('%Y-%m-%d')}"
        )
        if report.framework:
            lines.append(f"**Framework:** {report.framework.value.upper()}")
        lines.append(f"**Total Events:** {report.total_events}")
        lines.append("")

        # Violations
        if report.violations:
            lines.append(f"**Violations ({len(report.violations)}):**")
            for v in report.violations[:5]:
                lines.append(f"- {_get_icon('warning')} {v.rule.name}: {v.message}")
            if len(report.violations) > 5:
                lines.append(f"  ... and {len(report.violations) - 5} more")
            lines.append("")
        else:
            lines.append(f"**Violations:** {_get_icon('success')} None")
            lines.append("")

        # Event breakdown
        if report.events_by_type:
            lines.append("**Event Breakdown:**")
            for event_type, count in sorted(
                report.events_by_type.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]:
                lines.append(f"- {event_type}: {count}")

        return "\n".join(lines)

    def _format_events(self, events: list) -> str:
        """Format audit events."""
        if not events:
            return "No audit events found for the specified criteria."

        lines = [f"**Audit Events ({len(events)})**", ""]

        for event in events[:20]:  # Limit display
            timestamp = event.timestamp.strftime("%Y-%m-%d %H:%M")
            severity_icons = {
                "critical": _get_icon("level_critical"),
                "error": _get_icon("level_high"),
                "warning": _get_icon("level_medium"),
                "info": _get_icon("level_info"),
                "debug": _get_icon("level_unknown"),
            }
            severity_icon = severity_icons.get(event.severity.value, _get_icon("level_unknown"))

            lines.append(f"{severity_icon} **{event.event_type.value}** - {timestamp}")
            lines.append(f"   {event.action}")
            if event.resource:
                lines.append(f"   Resource: {event.resource}")
            lines.append("")

        if len(events) > 20:
            lines.append(f"... and {len(events) - 20} more events")

        return "\n".join(lines)

    def _format_compliance(self, status: dict[str, Any]) -> str:
        """Format compliance status."""
        lines = ["**Compliance Check**", ""]

        framework = status["framework"].upper()
        compliant = status["compliant"]
        icon = _get_icon("success") if compliant else _get_icon("error")

        lines.append(f"**Framework:** {framework}")
        lines.append(f"**Status:** {icon} {'Compliant' if compliant else 'Non-Compliant'}")
        lines.append(f"**Events Analyzed:** {status['total_events']}")
        lines.append(f"**Period:** Last {status['report_period_days']} days")
        lines.append("")

        if status["violations"] > 0:
            lines.append(f"**Violations ({status['violations']}):**")
            for v in status["violation_details"]:
                lines.append(f"- {_get_icon('warning')} {v['violation_type']}: {v['message']}")

        return "\n".join(lines)
