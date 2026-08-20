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

"""Persistent metrics collection for serialization optimization feedback.

Collects and persists serialization metrics for:
- Analysis of format effectiveness per tool/provider
- Optimization feedback for format selection
- Token savings tracking over time
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

if TYPE_CHECKING:
    from victor.processing.serialization.adaptive import (
        SerializationContext,
        SerializationMetrics,
    )
    from victor.processing.serialization.strategy import DataCharacteristics

logger = logging.getLogger(__name__)


@dataclass
class SerializationMetricRecord:
    """Single serialization metric record for persistence."""

    timestamp: str
    tool_name: Optional[str]
    tool_operation: Optional[str]
    provider: Optional[str]
    model: Optional[str]
    format_selected: str
    selection_reason: str
    original_tokens: int
    serialized_tokens: int
    token_savings_percent: float
    char_savings_percent: float
    data_structure_type: str
    array_length: int
    has_nested_objects: bool
    analysis_time_ms: float
    encoding_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class SerializationMetricsCollector:
    """Collects and persists serialization metrics.

    Uses SQLite for lightweight, embedded persistence.
    Thread-safe for concurrent access.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize collector.

        Args:
            db_path: Path to SQLite database. Defaults to ~/.victor/cache/serialization_metrics.db
        """
        if db_path is None:
            try:
                from victor.config.settings import get_project_paths

                paths = get_project_paths()
                db_path = paths.global_cache_dir / "serialization_metrics.db"
            except ImportError:
                db_path = Path.home() / ".victor" / "cache" / "serialization_metrics.db"

        self._db_path = db_path
        self._lock = threading.Lock()
        self._initialized = False

    def _ensure_db(self) -> None:
        """Ensure database and tables exist."""
        if self._initialized:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS serialization_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    tool_name TEXT,
                    tool_operation TEXT,
                    provider TEXT,
                    model TEXT,
                    format_selected TEXT NOT NULL,
                    selection_reason TEXT,
                    original_tokens INTEGER,
                    serialized_tokens INTEGER,
                    token_savings_percent REAL,
                    char_savings_percent REAL,
                    data_structure_type TEXT,
                    array_length INTEGER,
                    has_nested_objects INTEGER,
                    analysis_time_ms REAL,
                    encoding_time_ms REAL
                )
            """)

            # Create indices for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_tool
                ON serialization_metrics(tool_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_format
                ON serialization_metrics(format_selected)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
                ON serialization_metrics(timestamp)
            """)

            conn.commit()

        self._initialized = True

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get database connection with thread safety."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path), timeout=10.0)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def record(self, record: SerializationMetricRecord) -> None:
        """Record a serialization metric.

        Args:
            record: Metric record to persist
        """
        self._ensure_db()

        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO serialization_metrics (
                        timestamp, tool_name, tool_operation, provider, model,
                        format_selected, selection_reason, original_tokens,
                        serialized_tokens, token_savings_percent, char_savings_percent,
                        data_structure_type, array_length, has_nested_objects,
                        analysis_time_ms, encoding_time_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.timestamp,
                        record.tool_name,
                        record.tool_operation,
                        record.provider,
                        record.model,
                        record.format_selected,
                        record.selection_reason,
                        record.original_tokens,
                        record.serialized_tokens,
                        record.token_savings_percent,
                        record.char_savings_percent,
                        record.data_structure_type,
                        record.array_length,
                        1 if record.has_nested_objects else 0,
                        record.analysis_time_ms,
                        record.encoding_time_ms,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to record serialization metric: {e}")

    def get_tool_stats(self, tool_name: str) -> Dict[str, Any]:
        """Get aggregated statistics for a specific tool.

        Args:
            tool_name: Tool name to query

        Returns:
            Dictionary with aggregated statistics
        """
        self._ensure_db()

        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_serializations,
                        AVG(token_savings_percent) as avg_savings,
                        MAX(token_savings_percent) as max_savings,
                        MIN(token_savings_percent) as min_savings,
                        SUM(original_tokens - serialized_tokens) as total_tokens_saved,
                        AVG(encoding_time_ms) as avg_encoding_time
                    FROM serialization_metrics
                    WHERE tool_name = ?
                    """,
                    (tool_name,),
                )
                row = cursor.fetchone()

                if row and row["total_serializations"] > 0:
                    return {
                        "tool_name": tool_name,
                        "total_serializations": row["total_serializations"],
                        "avg_savings_percent": round(row["avg_savings"] or 0, 2),
                        "max_savings_percent": round(row["max_savings"] or 0, 2),
                        "min_savings_percent": round(row["min_savings"] or 0, 2),
                        "total_tokens_saved": row["total_tokens_saved"] or 0,
                        "avg_encoding_time_ms": round(row["avg_encoding_time"] or 0, 2),
                    }

        except Exception as e:
            logger.warning(f"Failed to get tool stats: {e}")

        return {
            "tool_name": tool_name,
            "total_serializations": 0,
            "avg_savings_percent": 0,
            "max_savings_percent": 0,
            "min_savings_percent": 0,
            "total_tokens_saved": 0,
            "avg_encoding_time_ms": 0,
        }

    def get_format_stats(self) -> List[Dict[str, Any]]:
        """Get aggregated statistics per format.

        Returns:
            List of format statistics
        """
        self._ensure_db()

        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT
                        format_selected,
                        COUNT(*) as usage_count,
                        AVG(token_savings_percent) as avg_savings,
                        SUM(original_tokens - serialized_tokens) as total_tokens_saved
                    FROM serialization_metrics
                    GROUP BY format_selected
                    ORDER BY usage_count DESC
                    """)

                return [
                    {
                        "format": row["format_selected"],
                        "usage_count": row["usage_count"],
                        "avg_savings_percent": round(row["avg_savings"] or 0, 2),
                        "total_tokens_saved": row["total_tokens_saved"] or 0,
                    }
                    for row in cursor.fetchall()
                ]

        except Exception as e:
            logger.warning(f"Failed to get format stats: {e}")
            return []

    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall serialization statistics.

        Returns:
            Dictionary with overall statistics
        """
        self._ensure_db()

        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT
                        COUNT(*) as total_serializations,
                        AVG(token_savings_percent) as avg_savings,
                        SUM(original_tokens) as total_original_tokens,
                        SUM(serialized_tokens) as total_serialized_tokens,
                        SUM(original_tokens - serialized_tokens) as total_tokens_saved
                    FROM serialization_metrics
                    """)
                row = cursor.fetchone()

                if row and row["total_serializations"] > 0:
                    return {
                        "total_serializations": row["total_serializations"],
                        "avg_savings_percent": round(row["avg_savings"] or 0, 2),
                        "total_original_tokens": row["total_original_tokens"] or 0,
                        "total_serialized_tokens": row["total_serialized_tokens"] or 0,
                        "total_tokens_saved": row["total_tokens_saved"] or 0,
                    }

        except Exception as e:
            logger.warning(f"Failed to get overall stats: {e}")

        return {
            "total_serializations": 0,
            "avg_savings_percent": 0,
            "total_original_tokens": 0,
            "total_serialized_tokens": 0,
            "total_tokens_saved": 0,
        }

    def clear_metrics(self) -> None:
        """Clear all metrics (for testing)."""
        self._ensure_db()

        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM serialization_metrics")
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to clear metrics: {e}")


# Global collector instance
_collector: Optional[SerializationMetricsCollector] = None


def get_metrics_collector() -> SerializationMetricsCollector:
    """Get the global metrics collector.

    Returns:
        SerializationMetricsCollector instance
    """
    global _collector
    if _collector is None:
        _collector = SerializationMetricsCollector()
    return _collector


def reset_metrics_collector() -> None:
    """Reset the global metrics collector (for testing)."""
    global _collector
    _collector = None


def record_serialization_metrics(
    metrics: "SerializationMetrics",
    context: Optional["SerializationContext"] = None,
    characteristics: Optional["DataCharacteristics"] = None,
) -> None:
    """Convenience function to record metrics from AdaptiveSerializer.

    Args:
        metrics: SerializationMetrics from serializer
        context: Optional SerializationContext
        characteristics: Optional DataCharacteristics
    """
    # Import here to avoid circular imports

    record = SerializationMetricRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        tool_name=context.tool_name if context else None,
        tool_operation=context.tool_operation if context else None,
        provider=context.provider if context else None,
        model=context.model if context else None,
        format_selected=metrics.format_selected,
        selection_reason=metrics.selection_reason,
        original_tokens=metrics.original_json_tokens,
        serialized_tokens=metrics.serialized_tokens,
        token_savings_percent=metrics.token_savings_percent,
        char_savings_percent=metrics.char_savings_percent,
        data_structure_type=(
            characteristics.structure_type.value if characteristics else "unknown"
        ),
        array_length=characteristics.array_length if characteristics else 0,
        has_nested_objects=(characteristics.has_nested_objects if characteristics else False),
        analysis_time_ms=metrics.analysis_time_ms,
        encoding_time_ms=metrics.encoding_time_ms,
    )

    get_metrics_collector().record(record)
