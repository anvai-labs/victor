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

"""Telemetry collection for recovery system.

This module provides comprehensive telemetry for:
- Failure occurrence tracking
- Recovery attempt monitoring
- Strategy effectiveness analysis
- Model-specific failure patterns

Data can be exported to:
- Prometheus metrics
- JSON reports
- SQLite for persistence
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps
import logging
import sqlite3
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.agent.recovery.protocols import (
    FailureType,
    RecoveryAction,
    RecoveryContext,
    RecoveryResult,
    TelemetryCollector,
)

logger = logging.getLogger(__name__)


@dataclass
class FailureEvent:
    """Record of a failure occurrence."""

    timestamp: datetime
    failure_type: FailureType
    provider: str
    model: str
    task_type: str
    consecutive_count: int
    context_hash: str


@dataclass
class RecoveryEvent:
    """Record of a recovery attempt."""

    timestamp: datetime
    failure_type: FailureType
    strategy_name: str
    action: RecoveryAction
    success: bool
    quality_improvement: float
    provider: str
    model: str
    context_hash: str


@dataclass
class AggregatedStats:
    """Aggregated statistics for a time window."""

    start_time: datetime
    end_time: datetime
    total_failures: int = 0
    total_recoveries: int = 0
    successful_recoveries: int = 0
    failures_by_type: Dict[str, int] = field(default_factory=dict)
    recoveries_by_strategy: Dict[str, Dict[str, int]] = field(default_factory=dict)
    failures_by_model: Dict[str, int] = field(default_factory=dict)
    avg_quality_improvement: float = 0.0

    @property
    def recovery_rate(self) -> float:
        if self.total_recoveries == 0:
            return 0.0
        return self.successful_recoveries / self.total_recoveries


class RecoveryTelemetryCollector:
    """Telemetry collector for the recovery system.

    Implements TelemetryCollector protocol.

    Features:
    - In-memory ring buffer for recent events
    - SQLite persistence for long-term analysis
    - Prometheus-compatible metrics export
    - Thread-safe operations
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        max_memory_events: int = 1000,
        prometheus_enabled: bool = False,
    ):
        self._db_path = db_path
        self._max_memory_events = max_memory_events
        self._prometheus_enabled = prometheus_enabled

        # In-memory storage (ring buffer)
        self._failure_events: List[FailureEvent] = []
        self._recovery_events: List[RecoveryEvent] = []
        self._lock = threading.RLock()

        # Aggregated counters (for Prometheus)
        self._failure_counts: Dict[str, int] = defaultdict(int)
        self._recovery_counts: Dict[str, int] = defaultdict(int)
        self._success_counts: Dict[str, int] = defaultdict(int)

        # Initialize database
        if db_path:
            self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS failure_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    failure_type TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    task_type TEXT,
                    consecutive_count INTEGER,
                    context_hash TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recovery_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    failure_type TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    quality_improvement REAL,
                    provider TEXT,
                    model TEXT,
                    context_hash TEXT
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_failure_timestamp
                ON failure_events(timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_recovery_timestamp
                ON recovery_events(timestamp)
            """)

            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to initialize telemetry database: {e}")

    def record_failure(self, context: RecoveryContext) -> None:
        """Record a failure occurrence."""
        event = FailureEvent(
            timestamp=datetime.now(),
            failure_type=context.failure_type,
            provider=context.provider_name,
            model=context.model_name,
            task_type=context.task_type,
            consecutive_count=context.consecutive_failures,
            context_hash=context.to_state_key()[:64],
        )

        with self._lock:
            # Add to memory buffer
            self._failure_events.append(event)
            if len(self._failure_events) > self._max_memory_events:
                self._failure_events.pop(0)

            # Update counters
            self._failure_counts[context.failure_type.name] += 1
            self._failure_counts[f"{context.provider_name}:{context.failure_type.name}"] += 1

        # Persist to database
        if self._db_path:
            self._persist_failure(event)

        logger.debug(
            f"Recorded failure: {context.failure_type.name} "
            f"({context.provider_name}/{context.model_name})"
        )

    def record_recovery_attempt(
        self,
        context: RecoveryContext,
        result: RecoveryResult,
    ) -> None:
        """Record a recovery attempt."""
        event = RecoveryEvent(
            timestamp=datetime.now(),
            failure_type=context.failure_type,
            strategy_name=result.strategy_name,
            action=result.action,
            success=False,  # Will be updated by record_recovery_outcome
            quality_improvement=0.0,
            provider=context.provider_name,
            model=context.model_name,
            context_hash=context.to_state_key()[:64],
        )

        with self._lock:
            self._recovery_events.append(event)
            if len(self._recovery_events) > self._max_memory_events:
                self._recovery_events.pop(0)

            # Update counters
            self._recovery_counts[result.strategy_name] += 1

    def record_recovery_outcome(
        self,
        context: RecoveryContext,
        result: RecoveryResult,
        success: bool,
        quality_improvement: float,
    ) -> None:
        """Record the outcome of a recovery attempt."""
        event = RecoveryEvent(
            timestamp=datetime.now(),
            failure_type=context.failure_type,
            strategy_name=result.strategy_name,
            action=result.action,
            success=success,
            quality_improvement=quality_improvement,
            provider=context.provider_name,
            model=context.model_name,
            context_hash=context.to_state_key()[:64],
        )

        with self._lock:
            # Update the most recent matching event if exists
            for i in range(len(self._recovery_events) - 1, -1, -1):
                existing = self._recovery_events[i]
                if (
                    existing.context_hash == event.context_hash
                    and existing.strategy_name == result.strategy_name
                    and not existing.success
                ):
                    self._recovery_events[i] = event
                    break
            else:
                self._recovery_events.append(event)
                if len(self._recovery_events) > self._max_memory_events:
                    self._recovery_events.pop(0)

            # Update counters
            if success:
                self._success_counts[result.strategy_name] += 1

        # Persist to database
        if self._db_path:
            self._persist_recovery(event)

        logger.debug(
            f"Recorded recovery outcome: {result.strategy_name} "
            f"success={success}, quality_improvement={quality_improvement:.2f}"
        )

    def _persist_failure(self, event: FailureEvent) -> None:
        """Persist failure event to database."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO failure_events
                (timestamp, failure_type, provider, model, task_type, consecutive_count, context_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event.timestamp.isoformat(),
                    event.failure_type.name,
                    event.provider,
                    event.model,
                    event.task_type,
                    event.consecutive_count,
                    event.context_hash,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to persist failure event: {e}")

    def _persist_recovery(self, event: RecoveryEvent) -> None:
        """Persist recovery event to database."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO recovery_events
                (timestamp, failure_type, strategy_name, action, success, quality_improvement, provider, model, context_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event.timestamp.isoformat(),
                    event.failure_type.name,
                    event.strategy_name,
                    event.action.name,
                    1 if event.success else 0,
                    event.quality_improvement,
                    event.provider,
                    event.model,
                    event.context_hash,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to persist recovery event: {e}")

    def get_failure_stats(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get failure statistics for a time window."""
        cutoff = datetime.now() - timedelta(hours=time_window_hours)

        with self._lock:
            recent_failures = [e for e in self._failure_events if e.timestamp >= cutoff]
            recent_recoveries = [e for e in self._recovery_events if e.timestamp >= cutoff]

        # Aggregate by type
        failures_by_type: Dict[str, int] = defaultdict(int)
        failures_by_model: Dict[str, int] = defaultdict(int)
        failures_by_provider: Dict[str, int] = defaultdict(int)

        for event in recent_failures:
            failures_by_type[event.failure_type.name] += 1
            failures_by_model[event.model] += 1
            failures_by_provider[event.provider] += 1

        # Recovery stats
        successful = sum(1 for e in recent_recoveries if e.success)
        total_quality_improvement = sum(e.quality_improvement for e in recent_recoveries)

        return {
            "time_window_hours": time_window_hours,
            "total_failures": len(recent_failures),
            "total_recovery_attempts": len(recent_recoveries),
            "successful_recoveries": successful,
            "recovery_rate": successful / max(len(recent_recoveries), 1),
            "avg_quality_improvement": total_quality_improvement / max(len(recent_recoveries), 1),
            "failures_by_type": dict(failures_by_type),
            "failures_by_model": dict(failures_by_model),
            "failures_by_provider": dict(failures_by_provider),
        }

    def get_strategy_effectiveness(self) -> Dict[str, Dict[str, float]]:
        """Get effectiveness metrics per strategy."""
        with self._lock:
            recovery_by_strategy: Dict[str, List[RecoveryEvent]] = defaultdict(list)
            for event in self._recovery_events:
                recovery_by_strategy[event.strategy_name].append(event)

        effectiveness = {}
        for strategy, events in recovery_by_strategy.items():
            successful = sum(1 for e in events if e.success)
            total_quality = sum(e.quality_improvement for e in events)
            effectiveness[strategy] = {
                "total_attempts": len(events),
                "successful": successful,
                "success_rate": successful / max(len(events), 1),
                "avg_quality_improvement": total_quality / max(len(events), 1),
            }

        return effectiveness

    def get_model_failure_patterns(self) -> Dict[str, Dict[str, int]]:
        """Get failure patterns per model."""
        with self._lock:
            patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for event in self._failure_events:
                patterns[event.model][event.failure_type.name] += 1

        return {model: dict(types) for model, types in patterns.items()}

    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Failure counts
        lines.append("# HELP recovery_failures_total Total number of failures by type")
        lines.append("# TYPE recovery_failures_total counter")
        with self._lock:
            for key, count in self._failure_counts.items():
                if ":" in key:
                    provider, failure_type = key.split(":", 1)
                    lines.append(
                        f'recovery_failures_total{{provider="{provider}",type="{failure_type}"}} {count}'
                    )
                else:
                    lines.append(f'recovery_failures_total{{type="{key}"}} {count}')

        # Recovery counts
        lines.append("# HELP recovery_attempts_total Total recovery attempts by strategy")
        lines.append("# TYPE recovery_attempts_total counter")
        with self._lock:
            for strategy, count in self._recovery_counts.items():
                lines.append(f'recovery_attempts_total{{strategy="{strategy}"}} {count}')

        # Success counts
        lines.append("# HELP recovery_successes_total Successful recoveries by strategy")
        lines.append("# TYPE recovery_successes_total counter")
        with self._lock:
            for strategy, count in self._success_counts.items():
                lines.append(f'recovery_successes_total{{strategy="{strategy}"}} {count}')

        return "\n".join(lines)

    def export_json_report(self, time_window_hours: int = 24) -> str:
        """Export a JSON report of telemetry data."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "time_window_hours": time_window_hours,
            "failure_stats": self.get_failure_stats(time_window_hours),
            "strategy_effectiveness": self.get_strategy_effectiveness(),
            "model_failure_patterns": self.get_model_failure_patterns(),
        }
        return json_dumps(report, indent=2, default=str)

    def clear_old_events(self, max_age_hours: int = 168) -> int:
        """Clear events older than max_age_hours. Returns count of cleared events."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        cleared = 0

        with self._lock:
            original_failures = len(self._failure_events)
            self._failure_events = [e for e in self._failure_events if e.timestamp >= cutoff]
            cleared += original_failures - len(self._failure_events)

            original_recoveries = len(self._recovery_events)
            self._recovery_events = [e for e in self._recovery_events if e.timestamp >= cutoff]
            cleared += original_recoveries - len(self._recovery_events)

        # Also clear from database
        if self._db_path:
            try:
                conn = sqlite3.connect(str(self._db_path))
                cursor = conn.cursor()
                cutoff_str = cutoff.isoformat()
                cursor.execute("DELETE FROM failure_events WHERE timestamp < ?", (cutoff_str,))
                cursor.execute("DELETE FROM recovery_events WHERE timestamp < ?", (cutoff_str,))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning(f"Failed to clear old events from database: {e}")

        logger.info(f"Cleared {cleared} old telemetry events")
        return cleared
