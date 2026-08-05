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

"""Metrics persistence backends.

Provides storage backends for metrics data including:
- SQLite backend for local persistence
- PostgreSQL backend for production use
- In-memory backend for testing

Example:
    store = SQLiteMetricsStore(":memory:")
    await store.store_metrics(agent_metrics)
    results = await store.query_metrics(
        agent_id="my-agent",
        start_time=time.time() - 3600
    )
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps, json_loads
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from victor.framework.observability.metrics import (
    AgentMetrics,
    Metric,
    MetricType,
    ToolCallMetrics,
    LLMCallMetrics,
)


class MetricsStore(ABC):
    """Abstract protocol for metrics storage backends.

    Implementations must support:
    - Storing AgentMetrics
    - Querying by time range
    - Querying by agent/session
    - Aggregation and downsampling
    """

    @abstractmethod
    async def store(self, metrics: AgentMetrics) -> None:
        """Store agent metrics.

        Args:
            metrics: AgentMetrics to store
        """

    @abstractmethod
    async def query(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
    ) -> List[AgentMetrics]:
        """Query stored metrics.

        Args:
            agent_id: Filter by agent ID
            session_id: Filter by session ID
            start_time: Start of time range (unix timestamp)
            end_time: End of time range (unix timestamp)
            limit: Maximum results to return

        Returns:
            List of matching AgentMetrics
        """

    @abstractmethod
    async def aggregate(
        self,
        metric_name: str,
        group_by: Optional[List[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Aggregate metrics by name.

        Args:
            metric_name: Name of metric to aggregate
            group_by: Fields to group by (e.g., ["agent_id"])
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Aggregated metric values
        """

    @abstractmethod
    async def delete_old(self, cutoff_hours: float = 24.0) -> int:
        """Delete metrics older than cutoff.

        Args:
            cutoff_hours: Hours to retain data

        Returns:
            Number of metrics deleted
        """


class SQLiteMetricsStore(MetricsStore):
    """SQLite metrics storage backend.

    Provides local file-based persistence for metrics data.
    Automatically creates tables on initialization.

    Example:
        store = SQLiteMetricsStore("/path/to/metrics.db")
        await store.store_metrics(agent_metrics)
        results = await store.query_metrics(agent_id="my-agent")
    """

    def __init__(self, path: Union[str, Path] = ":memory:") -> None:
        """Initialize the SQLite store.

        Args:
            path: Database file path (":memory:" for in-memory)
        """
        self._path = str(path)
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize database schema."""
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        cursor = self._conn.cursor()

        # Create agent metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                session_id TEXT,
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,

                total_input_tokens INTEGER DEFAULT 0,
                total_output_tokens INTEGER DEFAULT 0,
                total_cache_read_tokens INTEGER DEFAULT 0,
                total_cache_write_tokens INTEGER DEFAULT 0,
                total_reasoning_tokens INTEGER DEFAULT 0,

                state_transitions INTEGER DEFAULT 0,
                current_state TEXT,

                errors TEXT,

                INDEX (agent_id),
                INDEX (session_id),
                INDEX (created_at)
            )
        """)

        # Create tool calls table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_metric_id INTEGER NOT NULL,
                tool_name TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                success INTEGER DEFAULT 0,
                error_message TEXT,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,

                FOREIGN KEY (agent_metric_id) REFERENCES agent_metrics (id)
            )
        """)

        # Create LLM calls table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_metric_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                success INTEGER DEFAULT 0,
                error_message TEXT,

                FOREIGN KEY (agent_metric_id) REFERENCES agent_metrics (id)
            )
        """)

        self._conn.commit()

    async def store(self, metrics: AgentMetrics) -> None:
        """Store agent metrics.

        Args:
            metrics: AgentMetrics to store
        """
        cursor = self._conn.cursor()

        # Insert main metrics
        cursor.execute(
            """
            INSERT INTO agent_metrics (
                agent_id, session_id, created_at, started_at, completed_at,
                total_input_tokens, total_output_tokens,
                total_cache_read_tokens, total_cache_write_tokens,
                total_reasoning_tokens, state_transitions, current_state, errors
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metrics.agent_id,
                metrics.session_id,
                metrics.created_at,
                metrics.started_at,
                metrics.completed_at,
                metrics.total_input_tokens,
                metrics.total_output_tokens,
                metrics.total_cache_read_tokens,
                metrics.total_cache_write_tokens,
                metrics.total_reasoning_tokens,
                metrics.state_transitions,
                metrics.current_state,
                json_dumps(metrics.errors),
            ),
        )

        metric_id = cursor.lastrowid

        # Insert tool calls
        for tool_call in metrics.tool_calls:
            cursor.execute(
                """
                INSERT INTO tool_calls (
                    agent_metric_id, tool_name, start_time, end_time,
                    success, error_message, input_tokens, output_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metric_id,
                    tool_call.tool_name,
                    tool_call.start_time,
                    tool_call.end_time,
                    1 if tool_call.success else 0,
                    tool_call.error_message,
                    tool_call.input_tokens,
                    tool_call.output_tokens,
                ),
            )

        # Insert LLM calls
        for llm_call in metrics.llm_calls:
            cursor.execute(
                """
                INSERT INTO llm_calls (
                    agent_metric_id, provider, model, start_time, end_time,
                    input_tokens, output_tokens, cache_read_tokens,
                    cache_write_tokens, reasoning_tokens, success, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metric_id,
                    llm_call.provider,
                    llm_call.model,
                    llm_call.start_time,
                    llm_call.end_time,
                    llm_call.input_tokens,
                    llm_call.output_tokens,
                    llm_call.cache_read_tokens,
                    llm_call.cache_write_tokens,
                    llm_call.reasoning_tokens,
                    1 if llm_call.success else 0,
                    llm_call.error_message,
                ),
            )

        self._conn.commit()

    async def query(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
    ) -> List[AgentMetrics]:
        """Query stored metrics.

        Args:
            agent_id: Filter by agent ID
            session_id: Filter by session ID
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum results

        Returns:
            List of AgentMetrics
        """
        cursor = self._conn.cursor()

        # Build query
        conditions = []
        params = []

        if agent_id:
            conditions.append("agent_id = ?")
            params.append(agent_id)

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)

        if start_time:
            conditions.append("created_at >= ?")
            params.append(start_time)

        if end_time:
            conditions.append("created_at <= ?")
            params.append(end_time)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT * FROM agent_metrics
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)

        cursor.execute(query, params)

        results = []
        for row in cursor.fetchall():
            # Load tool calls
            cursor.execute(
                """
                SELECT * FROM tool_calls
                WHERE agent_metric_id = ?
                ORDER BY start_time
            """,
                (row["id"],),
            )

            tool_calls = []
            for tc_row in cursor.fetchall():
                tool_calls.append(
                    ToolCallMetrics(
                        tool_name=tc_row["tool_name"],
                        start_time=tc_row["start_time"],
                        end_time=tc_row["end_time"],
                        success=bool(tc_row["success"]),
                        error_message=tc_row["error_message"],
                        input_tokens=tc_row["input_tokens"],
                        output_tokens=tc_row["output_tokens"],
                    )
                )

            # Load LLM calls
            cursor.execute(
                """
                SELECT * FROM llm_calls
                WHERE agent_metric_id = ?
                ORDER BY start_time
            """,
                (row["id"],),
            )

            llm_calls = []
            for lc_row in cursor.fetchall():
                llm_calls.append(
                    LLMCallMetrics(
                        provider=lc_row["provider"],
                        model=lc_row["model"],
                        start_time=lc_row["start_time"],
                        end_time=lc_row["end_time"],
                        input_tokens=lc_row["input_tokens"],
                        output_tokens=lc_row["output_tokens"],
                        cache_read_tokens=lc_row["cache_read_tokens"],
                        cache_write_tokens=lc_row["cache_write_tokens"],
                        reasoning_tokens=lc_row["reasoning_tokens"],
                        success=bool(lc_row["success"]),
                        error_message=lc_row["error_message"],
                    )
                )

            # Reconstruct AgentMetrics
            metrics = AgentMetrics(
                agent_id=row["agent_id"],
                session_id=row["session_id"],
                created_at=row["created_at"],
                started_at=row["started_at"],
                completed_at=row["completed_at"],
                total_input_tokens=row["total_input_tokens"],
                total_output_tokens=row["total_output_tokens"],
                total_cache_read_tokens=row["total_cache_read_tokens"],
                total_cache_write_tokens=row["total_cache_write_tokens"],
                total_reasoning_tokens=row["total_reasoning_tokens"],
                tool_calls=tool_calls,
                llm_calls=llm_calls,
                state_transitions=row["state_transitions"],
                current_state=row["current_state"],
                errors=json_loads(row["errors"]) if row["errors"] else [],
            )

            results.append(metrics)

        return results

    async def aggregate(
        self,
        metric_name: str,
        group_by: Optional[List[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Aggregate metrics by name.

        Args:
            metric_name: Name of metric to aggregate
            group_by: Fields to group by
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Aggregated values
        """
        cursor = self._conn.cursor()

        # Map metric names to columns
        metric_columns = {
            "total_input_tokens": "total_input_tokens",
            "total_output_tokens": "total_output_tokens",
            "total_tokens": "(total_input_tokens + total_output_tokens)",
        }

        column = metric_columns.get(metric_name)
        if not column:
            return {}

        # Build aggregation query
        group_fields = group_by or []
        group_clause = ", ".join(group_fields) if group_fields else "NULL"

        query = f"""
            SELECT
                {group_fields} as group_key,
                SUM({column}) as total,
                AVG({column}) as average,
                MIN({column}) as minimum,
                MAX({column}) as maximum,
                COUNT(*) as count
            FROM agent_metrics
            WHERE 1=1
        """

        params = []
        if start_time:
            query += " AND created_at >= ?"
            params.append(start_time)

        if end_time:
            query += " AND created_at <= ?"
            params.append(end_time)

        query += f" GROUP BY {group_clause}"

        cursor.execute(query, params)

        return [dict(row) for row in cursor.fetchall()]

    async def delete_old(self, cutoff_hours: float = 24.0) -> int:
        """Delete metrics older than cutoff.

        Args:
            cutoff_hours: Hours to retain

        Returns:
            Number deleted
        """
        cursor = self._conn.cursor()

        cutoff = time.time() - (cutoff_hours * 3600)

        # Delete child rows first
        cursor.execute(
            """
            DELETE FROM tool_calls
            WHERE agent_metric_id IN (
                SELECT id FROM agent_metrics WHERE created_at < ?
            )
        """,
            (cutoff,),
        )

        cursor.execute(
            """
            DELETE FROM llm_calls
            WHERE agent_metric_id IN (
                SELECT id FROM agent_metrics WHERE created_at < ?
            )
        """,
            (cutoff,),
        )

        # Delete parent rows
        cursor.execute(
            """
            DELETE FROM agent_metrics WHERE created_at < ?
        """,
            (cutoff,),
        )

        deleted = cursor.rowcount
        self._conn.commit()

        return deleted

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


class InMemoryMetricsStore(MetricsStore):
    """In-memory metrics storage for testing.

    Example:
        store = InMemoryMetricsStore()
        await store.store_metrics(agent_metrics)
        results = await store.query_metrics(agent_id="my-agent")
    """

    def __init__(self, max_size: int = 10000) -> None:
        """Initialize the in-memory store.

        Args:
            max_size: Maximum number of metrics to store
        """
        self._metrics: List[AgentMetrics] = []
        self._max_size = max_size

    async def store(self, metrics: AgentMetrics) -> None:
        """Store agent metrics.

        Args:
            metrics: AgentMetrics to store
        """
        self._metrics.append(metrics)

        # Enforce max size (FIFO)
        if len(self._metrics) > self._max_size:
            self._metrics.pop(0)

    async def query(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
    ) -> List[AgentMetrics]:
        """Query stored metrics.

        Args:
            agent_id: Filter by agent ID
            session_id: Filter by session ID
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum results

        Returns:
            List of AgentMetrics
        """
        results = self._metrics

        if agent_id:
            results = [m for m in results if m.agent_id == agent_id]

        if session_id:
            results = [m for m in results if m.session_id == session_id]

        if start_time:
            results = [m for m in results if m.created_at >= start_time]

        if end_time:
            results = [m for m in results if m.created_at <= end_time]

        return results[:limit]

    async def aggregate(
        self,
        metric_name: str,
        group_by: Optional[List[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Aggregate metrics by name.

        Args:
            metric_name: Name of metric to aggregate
            group_by: Fields to group by
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Aggregated values
        """
        results = await self.query(
            start_time=start_time,
            end_time=end_time,
            limit=len(self._metrics),
        )

        if metric_name == "total_tokens":
            values = [m.total_tokens for m in results]
        elif metric_name == "total_input_tokens":
            values = [m.total_input_tokens for m in results]
        elif metric_name == "total_output_tokens":
            values = [m.total_output_tokens for m in results]
        else:
            values = []

        if not values:
            return {}

        return {
            "total": sum(values),
            "average": sum(values) / len(values),
            "minimum": min(values),
            "maximum": max(values),
            "count": len(values),
        }

    async def delete_old(self, cutoff_hours: float = 24.0) -> int:
        """Delete metrics older than cutoff.

        Args:
            cutoff_hours: Hours to retain

        Returns:
            Number deleted
        """
        cutoff = time.time() - (cutoff_hours * 3600)
        original_len = len(self._metrics)

        self._metrics = [m for m in self._metrics if m.created_at >= cutoff]

        return original_len - len(self._metrics)


__all__ = [
    "MetricsStore",
    "SQLiteMetricsStore",
    "InMemoryMetricsStore",
]
