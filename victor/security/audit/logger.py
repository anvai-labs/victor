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

"""Audit Logger - File-based audit event storage.

This module provides a file-based implementation of the audit logger
protocol with support for rotation and export.
"""

import csv
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from victor.config.settings import get_project_paths

from .protocol import (
    AuditEvent,
    AuditEventType,
    AuditLoggerProtocol,
    Severity,
)

logger = logging.getLogger(__name__)

# Environment sentinels indicating the process is running under pytest. Mirrors
# the telemetry-redirect logic in ``victor/core/bootstrap.py`` so test sessions
# never pollute the project audit database (two-database / test-isolation rule).
_TEST_MODE_SENTINELS = ("TEST_MODE", "PYTEST_XDIST_WORKER", "PYTEST_CURRENT_TEST")


def _is_test_mode() -> bool:
    """Return True when the process is running under the pytest test harness."""
    return any(os.getenv(v) for v in _TEST_MODE_SENTINELS)


def _resolve_audit_dir(root_path: Path | None) -> Path:
    """Resolve the audit directory, redirecting to a temp dir under test mode.

    Mirrors the usage-logger redirect in ``victor/core/bootstrap.py``: when a
    test sentinel is set, audit events go to ``<tmp>/victor_test_audit`` so they
    never land in the project ``./audit/`` directory. In production the audit
    directory stays alongside the provided (or project) root.
    """
    if _is_test_mode():
        audit_dir = Path(tempfile.gettempdir()) / "victor_test_audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        return audit_dir
    if root_path:
        return root_path / "audit"
    paths = get_project_paths()
    return paths.project_victor_dir / "audit"


class FileAuditLogger(AuditLoggerProtocol):
    """File-based audit logger with JSON storage."""

    def __init__(self, root_path: Path | None = None):
        """Initialize the file audit logger.

        Args:
            root_path: Root path for audit logs. Defaults to .victor/audit/

        Under test mode (any pytest sentinel env var set) audit logs are
        redirected to ``<tmp>/victor_test_audit`` to avoid polluting the project
        audit database — mirroring the usage-logger redirect in bootstrap.py.
        """
        self._audit_dir = _resolve_audit_dir(root_path)

        self._audit_dir.mkdir(parents=True, exist_ok=True)
        self._current_log_file = self._get_log_file()

    def _get_log_file(self) -> Path:
        """Get the current log file path (daily rotation)."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self._audit_dir / f"audit_{today}.jsonl"

    async def log_event(self, event: AuditEvent) -> None:
        """Log an audit event to file."""
        # Check for daily rotation
        log_file = self._get_log_file()
        if log_file != self._current_log_file:
            self._current_log_file = log_file

        # Append event to log file
        try:
            with open(self._current_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")

            logger.debug(f"Logged audit event: {event.event_type.value} - {event.action}")
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")

    async def query_events(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        severity: Severity | None = None,
        actor: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events from log files."""
        events = []

        # Determine which log files to read
        log_files = sorted(self._audit_dir.glob("audit_*.jsonl"))

        if start_date:
            start_str = start_date.strftime("%Y-%m-%d")
            log_files = [f for f in log_files if f.stem.split("_")[1] >= start_str]

        if end_date:
            end_str = end_date.strftime("%Y-%m-%d")
            log_files = [f for f in log_files if f.stem.split("_")[1] <= end_str]

        # Read events from files
        for log_file in log_files:
            try:
                with open(log_file, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue

                        try:
                            data = json.loads(line)
                            event = AuditEvent.from_dict(data)

                            # Apply filters
                            if start_date and event.timestamp < start_date:
                                continue
                            if end_date and event.timestamp > end_date:
                                continue
                            if event_types and event.event_type not in event_types:
                                continue
                            if severity and event.severity != severity:
                                continue
                            if actor and event.actor != actor:
                                continue

                            events.append(event)

                            if len(events) >= limit:
                                break

                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Invalid event in {log_file}: {e}")
                            continue

            except Exception as e:
                logger.warning(f"Failed to read log file {log_file}: {e}")

            if len(events) >= limit:
                break

        return events[:limit]

    async def export_events(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "json",
    ) -> Path:
        """Export events to a file."""
        events = await self.query_events(
            start_date=start_date,
            end_date=end_date,
            limit=100000,  # Large limit for export
        )

        export_dir = self._audit_dir / "exports"
        export_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        if format == "csv":
            export_file = export_dir / f"audit_export_{start_str}_{end_str}_{timestamp}.csv"
            await self._export_csv(events, export_file)
        else:
            export_file = export_dir / f"audit_export_{start_str}_{end_str}_{timestamp}.json"
            await self._export_json(events, export_file)

        logger.info(f"Exported {len(events)} events to {export_file}")
        return export_file

    async def _export_json(self, events: list[AuditEvent], path: Path) -> None:
        """Export events to JSON format."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "export_date": datetime.now().isoformat(),
                    "event_count": len(events),
                    "events": [e.to_dict() for e in events],
                },
                f,
                indent=2,
            )

    async def _export_csv(self, events: list[AuditEvent], path: Path) -> None:
        """Export events to CSV format."""
        if not events:
            return

        fieldnames = [
            "event_id",
            "event_type",
            "timestamp",
            "severity",
            "actor",
            "action",
            "resource",
            "outcome",
            "session_id",
            "ip_address",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for event in events:
                row = event.to_dict()
                row["timestamp"] = event.timestamp.isoformat()
                writer.writerow(row)

    async def purge_old_events(self, retention_days: int) -> int:
        """Purge events older than retention period.

        Args:
            retention_days: Days to retain events

        Returns:
            Number of files purged
        """
        cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        purged = 0
        for log_file in self._audit_dir.glob("audit_*.jsonl"):
            file_date = log_file.stem.split("_")[1]
            # Calculate days difference
            file_datetime = datetime.strptime(file_date, "%Y-%m-%d")
            days_old = (cutoff - file_datetime).days

            if days_old > retention_days:
                try:
                    log_file.unlink()
                    purged += 1
                    logger.info(f"Purged old audit log: {log_file}")
                except Exception as e:
                    logger.warning(f"Failed to purge {log_file}: {e}")

        return purged


def create_event(
    event_type: AuditEventType,
    action: str,
    actor: str | None = None,
    severity: Severity = Severity.INFO,
    resource: str | None = None,
    outcome: str = "success",
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> AuditEvent:
    """Helper to create an audit event.

    Args:
        event_type: Type of event
        action: Description of the action
        actor: Who performed the action
        severity: Event severity
        resource: Affected resource
        outcome: Result of the action
        metadata: Additional metadata
        session_id: Session identifier

    Returns:
        New AuditEvent
    """
    return AuditEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        timestamp=datetime.now(),
        severity=severity,
        actor=actor or os.getenv("USER", "system"),
        action=action,
        resource=resource,
        outcome=outcome,
        metadata=metadata or {},
        session_id=session_id,
    )
