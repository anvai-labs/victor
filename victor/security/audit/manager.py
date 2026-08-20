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

"""Audit Manager - Orchestrates audit logging and compliance checking.

This module provides the AuditManager class that coordinates audit
logging, compliance checking, and report generation.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from victor.config.settings import get_project_paths, load_settings
from victor.core.async_utils import run_sync

from .checker import DefaultComplianceChecker
from .logger import FileAuditLogger, create_event
from .protocol import (
    AuditConfig,
    AuditEvent,
    AuditEventType,
    AuditReport,
    ComplianceFramework,
    RetentionPolicy,
    Severity,
)

logger = logging.getLogger(__name__)


class AuditManager:
    """Manager for audit logging and compliance.

    This class orchestrates:
    - Audit event logging
    - Compliance rule checking
    - Report generation
    - Data retention

    Configuration is driven by settings.py for consistency with Victor.
    """

    _instance: "AuditManager | None" = None

    def __init__(self, root_path: str | Path | None = None):
        """Initialize the audit manager.

        Args:
            root_path: Root directory for audit data. Defaults to .victor/
        """
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self._settings = load_settings()
        self._paths = get_project_paths(self.root_path)
        self._config_file = self._paths.project_victor_dir / "audit_config.json"

        # Initialize config
        self._config = self._load_config()

        # Initialize logger and checker
        self._logger = FileAuditLogger(self._paths.project_victor_dir.parent)
        self._checker = DefaultComplianceChecker(self._logger)

        # Track current session
        self._session_id: str | None = None

    @classmethod
    def get_instance(cls, root_path: str | Path | None = None) -> "AuditManager":
        """Get or create the singleton instance.

        Args:
            root_path: Root path for first initialization

        Returns:
            AuditManager instance
        """
        if cls._instance is None:
            cls._instance = cls(root_path)
        return cls._instance

    def _load_config(self) -> AuditConfig:
        """Load audit configuration."""
        if not self._config_file.exists():
            return AuditConfig()

        try:
            with open(self._config_file, encoding="utf-8") as f:
                data = json.load(f)

                # Parse nested objects
                retention_data = data.get("retention", {})
                framework_overrides = {}
                for k, v in retention_data.get("framework_overrides", {}).items():
                    framework_overrides[ComplianceFramework(k)] = v

                retention = RetentionPolicy(
                    default_retention_days=retention_data.get("default_retention_days", 365),
                    framework_overrides=framework_overrides,
                    sensitive_data_retention_days=retention_data.get(
                        "sensitive_data_retention_days", 90
                    ),
                    auto_purge=retention_data.get("auto_purge", True),
                    archive_before_purge=retention_data.get("archive_before_purge", True),
                )

                return AuditConfig(
                    enabled=data.get("enabled", True),
                    log_file_operations=data.get("log_file_operations", True),
                    log_tool_executions=data.get("log_tool_executions", True),
                    log_model_interactions=data.get("log_model_interactions", True),
                    detect_secrets=data.get("detect_secrets", True),
                    detect_pii=data.get("detect_pii", True),
                    frameworks=[ComplianceFramework(f) for f in data.get("frameworks", [])],
                    retention=retention,
                    export_format=data.get("export_format", "json"),
                )
        except Exception as e:
            logger.warning(f"Failed to load audit config: {e}")
            return AuditConfig()

    async def save_config(self, config: AuditConfig) -> None:
        """Save audit configuration.

        Args:
            config: Configuration to save
        """
        self._config = config
        self._config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self._config_file, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)

        logger.info("Saved audit configuration")

    @property
    def config(self) -> AuditConfig:
        """Get current audit configuration."""
        return self._config

    @property
    def enabled(self) -> bool:
        """Check if audit logging is enabled."""
        return self._config.enabled

    def set_session(self, session_id: str) -> None:
        """Set the current session ID.

        Args:
            session_id: Session identifier
        """
        self._session_id = session_id

    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        details: dict[str, Any] | None = None,
        resource: str | None = None,
        severity: Severity = Severity.INFO,
    ) -> None:
        """Synchronous wrapper for logging audit events.

        This is a convenience method for synchronous callers that wraps
        the async log() method. Used by security modules that need to
        log events from synchronous code paths.

        Args:
            event_type: Type of event
            action: Description of the action
            details: Event details/metadata
            resource: Affected resource
            severity: Event severity
        """
        if not self._config.enabled:
            return

        # Create event synchronously
        event = create_event(
            event_type=event_type,
            action=action,
            severity=severity,
            resource=resource,
            metadata=details or {},
            session_id=self._session_id,
        )

        # Try to log asynchronously if possible, otherwise log synchronously
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - schedule the log
            loop.create_task(self._logger.log_event(event))
        except RuntimeError:
            # No running loop - bridge through the shared sync boundary.
            try:
                run_sync(self._logger.log_event(event))
            except Exception as e:
                logger.debug(f"Failed to log audit event: {e}")

    async def log(
        self,
        event_type: AuditEventType,
        action: str,
        resource: str | None = None,
        severity: Severity = Severity.INFO,
        outcome: str = "success",
        metadata: dict[str, Any] | None = None,
        actor: str | None = None,
    ) -> None:
        """Log an audit event.

        Args:
            event_type: Type of event
            action: Description of the action
            resource: Affected resource
            severity: Event severity
            outcome: Result of the action
            metadata: Additional metadata
            actor: Who performed the action
        """
        if not self._config.enabled:
            return

        # Check if this event type should be logged
        if event_type in [
            AuditEventType.FILE_READ,
            AuditEventType.FILE_WRITE,
            AuditEventType.FILE_DELETE,
            AuditEventType.FILE_RENAME,
        ]:
            if not self._config.log_file_operations:
                return

        if event_type in [AuditEventType.TOOL_EXECUTION, AuditEventType.TOOL_FAILURE]:
            if not self._config.log_tool_executions:
                return

        event = create_event(
            event_type=event_type,
            action=action,
            actor=actor,
            severity=severity,
            resource=resource,
            outcome=outcome,
            metadata=metadata,
            session_id=self._session_id,
        )

        await self._logger.log_event(event)

        # Check compliance rules
        violations = await self._checker.check_event(event)
        for violation in violations:
            logger.warning(f"Compliance violation: {violation.message}")

    async def log_file_operation(
        self,
        operation: str,
        file_path: str,
        outcome: str = "success",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a file operation.

        Args:
            operation: Type of operation (read, write, delete, rename)
            file_path: Path to the file
            outcome: Result of the operation
            metadata: Additional metadata
        """
        event_type_map = {
            "read": AuditEventType.FILE_READ,
            "write": AuditEventType.FILE_WRITE,
            "delete": AuditEventType.FILE_DELETE,
            "rename": AuditEventType.FILE_RENAME,
        }

        event_type = event_type_map.get(operation, AuditEventType.FILE_READ)

        await self.log(
            event_type=event_type,
            action=f"File {operation}",
            resource=file_path,
            outcome=outcome,
            metadata=metadata,
        )

    async def log_tool_execution(
        self,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
        outcome: str = "success",
        duration_ms: int | None = None,
    ) -> None:
        """Log a tool execution.

        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            outcome: Result of the execution
            duration_ms: Execution duration in milliseconds
        """
        event_type = (
            AuditEventType.TOOL_EXECUTION if outcome == "success" else AuditEventType.TOOL_FAILURE
        )

        metadata = {}
        if parameters:
            # Sanitize sensitive parameters
            metadata["parameters"] = self._sanitize_params(parameters)
        if duration_ms:
            metadata["duration_ms"] = duration_ms

        await self.log(
            event_type=event_type,
            action=f"Executed tool: {tool_name}",
            resource=tool_name,
            severity=Severity.INFO if outcome == "success" else Severity.WARNING,
            outcome=outcome,
            metadata=metadata,
        )

    async def log_security_event(
        self,
        event_type: AuditEventType,
        message: str,
        resource: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a security-related event.

        Args:
            event_type: Type of security event
            message: Description
            resource: Affected resource
            metadata: Additional metadata
        """
        await self.log(
            event_type=event_type,
            action=message,
            resource=resource,
            severity=Severity.WARNING,
            metadata=metadata,
        )

    def _sanitize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Sanitize sensitive parameters before logging."""
        sensitive_keys = [
            "password",
            "secret",
            "token",
            "key",
            "api_key",
            "apikey",
            "credential",
        ]

        sanitized = {}
        for key, value in params.items():
            key_lower = key.lower()
            if any(s in key_lower for s in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_params(value)
            else:
                sanitized[key] = value

        return sanitized

    async def query_events(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events.

        Args:
            start_date: Start of time range
            end_date: End of time range
            event_types: Filter by event types
            limit: Maximum events to return

        Returns:
            List of matching events
        """
        return await self._logger.query_events(
            start_date=start_date,
            end_date=end_date,
            event_types=event_types,
            limit=limit,
        )

    async def generate_report(
        self,
        days: int = 30,
        framework: ComplianceFramework | None = None,
    ) -> AuditReport:
        """Generate an audit report.

        Args:
            days: Number of days to include
            framework: Specific framework to report on

        Returns:
            Audit report
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        return await self._checker.generate_report(
            start_date=start_date,
            end_date=end_date,
            framework=framework,
        )

    async def export_audit_log(
        self,
        days: int = 30,
        format: str | None = None,
    ) -> Path:
        """Export audit log to file.

        Args:
            days: Number of days to export
            format: Export format (json, csv)

        Returns:
            Path to exported file
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        export_format = format or self._config.export_format

        return await self._logger.export_events(
            start_date=start_date,
            end_date=end_date,
            format=export_format,
        )

    async def check_compliance(self, framework: ComplianceFramework) -> dict[str, Any]:
        """Check compliance status for a framework.

        Args:
            framework: Framework to check

        Returns:
            Compliance status
        """
        report = await self.generate_report(days=30, framework=framework)

        compliant = len(report.violations) == 0

        return {
            "framework": framework.value,
            "compliant": compliant,
            "total_events": report.total_events,
            "violations": len(report.violations),
            "violation_details": [v.to_dict() for v in report.violations[:10]],
            "report_period_days": 30,
        }

    async def get_summary(self, days: int = 7) -> dict[str, Any]:
        """Get a summary of audit activity.

        Args:
            days: Number of days to summarize

        Returns:
            Summary dictionary
        """
        report = await self.generate_report(days=days)

        return {
            "period_days": days,
            "total_events": report.total_events,
            "events_by_type": report.events_by_type,
            "events_by_severity": report.events_by_severity,
            "violations": len(report.violations),
            "compliance_status": ("compliant" if not report.violations else "non_compliant"),
        }

    async def apply_retention_policy(self) -> dict[str, Any]:
        """Apply data retention policy.

        Returns:
            Results of retention enforcement
        """
        if not self._config.retention.auto_purge:
            return {"purged": 0, "skipped": True}

        retention_days = self._config.retention.default_retention_days

        if self._config.retention.archive_before_purge:
            # Export before purging
            cutoff = datetime.now() - timedelta(days=retention_days)
            await self._logger.export_events(
                start_date=datetime(2020, 1, 1),  # Very old date
                end_date=cutoff,
                format="json",
            )

        purged = await self._logger.purge_old_events(retention_days)

        return {
            "purged": purged,
            "retention_days": retention_days,
        }
