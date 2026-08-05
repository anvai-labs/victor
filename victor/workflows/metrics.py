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

"""Workflow metrics collection for Phase 3.0 prerequisites.

This module provides comprehensive workflow-level metrics collection:
- Success rates and failure analysis
- Latency tracking by node
- Tool usage statistics
- Cost tracking per workflow

Integrates with existing EventBus for event-driven metrics collection.
Supports both in-memory and persistent storage (SQLite, JSON).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

from victor.workflows.streaming import WorkflowEventType, WorkflowStreamChunk

logger = logging.getLogger(__name__)


# =============================================================================
# Metrics Data Models
# =============================================================================


@dataclass
class NodeMetrics:
    """Metrics for a single node execution.

    Attributes:
        node_id: Node identifier
        execution_count: Number of times node was executed
        total_duration: Total time spent in node (seconds)
        avg_duration: Average execution time (seconds)
        min_duration: Minimum execution time (seconds)
        max_duration: Maximum execution time (seconds)
        success_count: Successful executions
        failure_count: Failed executions
        last_execution: Timestamp of last execution
    """

    node_id: str
    execution_count: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    min_duration: float = float("inf")
    max_duration: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    last_execution: Optional[float] = None

    def update(self, duration: float, success: bool) -> None:
        """Update metrics with new execution.

        Args:
            duration: Execution duration in seconds
            success: Whether execution succeeded
        """
        self.execution_count += 1
        self.total_duration += duration
        self.avg_duration = self.total_duration / self.execution_count
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)

        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        self.last_execution = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "execution_count": self.execution_count,
            "total_duration": self.total_duration,
            "avg_duration": self.avg_duration,
            "min_duration": (self.min_duration if self.min_duration != float("inf") else 0.0),
            "max_duration": self.max_duration,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_execution": self.last_execution,
        }


@dataclass
class ToolUsageMetrics:
    """Metrics for tool usage during workflow execution.

    Attributes:
        tool_name: Tool identifier
        call_count: Number of times tool was called
        total_duration: Total time spent in tool (seconds)
        error_count: Number of errors
        last_used: Timestamp of last usage
    """

    tool_name: str
    call_count: int = 0
    total_duration: float = 0.0
    error_count: int = 0
    last_used: Optional[float] = None

    def update(self, duration: float, success: bool) -> None:
        """Update metrics with new tool call.

        Args:
            duration: Tool call duration in seconds
            success: Whether call succeeded
        """
        self.call_count += 1
        self.total_duration += duration

        if not success:
            self.error_count += 1

        self.last_used = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "call_count": self.call_count,
            "total_duration": self.total_duration,
            "error_count": self.error_count,
            "last_used": self.last_used,
        }


@dataclass
class WorkflowMetrics:
    """Aggregated metrics for a workflow.

    Attributes:
        workflow_id: Workflow identifier
        workflow_name: Human-readable workflow name
        total_executions: Total number of executions
        successful_executions: Number of successful executions
        failed_executions: Number of failed executions
        success_rate: Success rate (0-1)
        total_duration: Total duration across all executions (seconds)
        avg_duration: Average execution duration (seconds)
        node_metrics: Metrics per node
        tool_metrics: Metrics per tool
        cost_tracking: Total cost tracking (if available)
        first_execution: Timestamp of first execution
        last_execution: Timestamp of last execution
    """

    workflow_id: str
    workflow_name: str = ""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    node_metrics: Dict[str, NodeMetrics] = field(default_factory=dict)
    tool_metrics: Dict[str, ToolUsageMetrics] = field(default_factory=dict)
    cost_tracking: float = 0.0
    first_execution: Optional[float] = None
    last_execution: Optional[float] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    def update_execution(self, duration: float, success: bool) -> None:
        """Update with new workflow execution.

        Args:
            duration: Execution duration in seconds
            success: Whether execution succeeded
        """
        self.total_executions += 1
        self.total_duration += duration
        self.avg_duration = self.total_duration / self.total_executions

        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1

        now = time.time()
        if self.first_execution is None:
            self.first_execution = now
        self.last_execution = now

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": self.success_rate,
            "total_duration": self.total_duration,
            "avg_duration": self.avg_duration,
            "node_metrics": {k: v.to_dict() for k, v in self.node_metrics.items()},
            "tool_metrics": {k: v.to_dict() for k, v in self.tool_metrics.items()},
            "cost_tracking": self.cost_tracking,
            "first_execution": self.first_execution,
            "last_execution": self.last_execution,
        }


# =============================================================================
# Metrics Collector
# =============================================================================


class WorkflowMetricsCollector:
    """Collects and aggregates workflow-level metrics.

    This collector integrates with the EventBus via StreamingObserver protocol
    to collect metrics during workflow execution. It tracks:
    - Workflow success rates
    - Node-level latency statistics
    - Tool usage patterns
    - Cost tracking (if available)

    Storage backends:
    - memory: In-memory only (default)
    - json: Persist to JSON file
    - sqlite: Persist to SQLite database

    Example:
        from victor.workflows.metrics import WorkflowMetricsCollector
        from victor.workflows.observability import ObservabilityEmitter

        # Create collector with JSON persistence
        collector = WorkflowMetricsCollector(
            storage_backend="json",
            storage_path="workflow_metrics.json"
        )

        # Register with emitter
        emitter = ObservabilityEmitter(workflow_id="wf_123")
        emitter.add_observer(collector)

        # Execute workflow
        result = await workflow.invoke(state)

        # Get metrics
        metrics = collector.get_workflow_metrics("wf_123")
        print(f"Success rate: {metrics.success_rate:.2%}")
        print(f"Avg duration: {metrics.avg_duration:.2f}s")

        # Or get all metrics
        all_metrics = collector.get_all_metrics()
    """

    def __init__(
        self,
        storage_backend: str = "memory",
        storage_path: Optional[str] = None,
        auto_save: bool = True,
    ):
        """Initialize metrics collector.

        Args:
            storage_backend: Storage backend ('memory', 'json', 'sqlite')
            storage_path: Path for JSON/SQLite storage (required if backend != 'memory')
            auto_save: Automatically save metrics after each workflow completion
        """
        self.storage_backend = storage_backend
        self.storage_path = storage_path
        self.auto_save = auto_save

        # In-memory metrics storage
        self._workflows: Dict[str, WorkflowMetrics] = {}

        # Active execution tracking
        self._active_executions: Dict[str, Dict[str, Any]] = {}

        # Initialize storage
        if storage_backend != "memory":
            if not storage_path:
                raise ValueError(f"storage_path required for backend '{storage_backend}'")

            if storage_backend == "json":
                self._load_from_json()
            elif storage_backend == "sqlite":
                self._init_sqlite()
            else:
                raise ValueError(f"Unknown storage backend: {storage_backend}")

    def on_event(self, chunk: WorkflowStreamChunk) -> None:
        """Handle workflow event (StreamingObserver protocol).

        Args:
            chunk: Workflow stream chunk event
        """
        try:
            if chunk.event_type == WorkflowEventType.WORKFLOW_START:
                self._on_workflow_start(chunk)
            elif chunk.event_type == WorkflowEventType.NODE_COMPLETE:
                self._on_node_complete(chunk)
            elif chunk.event_type == WorkflowEventType.NODE_ERROR:
                self._on_node_error(chunk)
            elif chunk.event_type == WorkflowEventType.AGENT_TOOL_CALL:
                self._on_tool_call(chunk)
            elif chunk.event_type == WorkflowEventType.WORKFLOW_COMPLETE:
                self._on_workflow_complete(chunk, success=True)
            elif chunk.event_type == WorkflowEventType.WORKFLOW_ERROR:
                self._on_workflow_complete(chunk, success=False)
        except Exception as e:
            logger.warning(f"Metrics collection error: {e}")

    def get_filter(self) -> Optional[set]:
        """Get event filter (all events)."""
        return None

    def _on_workflow_start(self, chunk: WorkflowStreamChunk) -> None:
        """Track workflow start."""
        workflow_id = chunk.workflow_id

        if workflow_id not in self._workflows:
            self._workflows[workflow_id] = WorkflowMetrics(
                workflow_id=workflow_id,
                workflow_name=chunk.metadata.get("workflow_name", ""),
            )

        # Track active execution
        self._active_executions[workflow_id] = {
            "start_time": time.time(),
            "nodes_started": {},
        }

    def _on_node_complete(self, chunk: WorkflowStreamChunk) -> None:
        """Track node completion."""
        workflow_id = chunk.workflow_id
        node_id = chunk.node_id

        if workflow_id not in self._workflows:
            return

        metrics = self._workflows[workflow_id]

        # Initialize node metrics if needed
        if node_id not in metrics.node_metrics:
            metrics.node_metrics[node_id] = NodeMetrics(node_id=node_id)

        # Extract duration from metadata
        duration = chunk.metadata.get("duration_seconds", 0.0)

        # Update node metrics
        metrics.node_metrics[node_id].update(duration=duration, success=True)

    def _on_node_error(self, chunk: WorkflowStreamChunk) -> None:
        """Track node error."""
        workflow_id = chunk.workflow_id
        node_id = chunk.node_id

        if workflow_id not in self._workflows:
            return

        metrics = self._workflows[workflow_id]

        # Initialize node metrics if needed
        if node_id not in metrics.node_metrics:
            metrics.node_metrics[node_id] = NodeMetrics(node_id=node_id)

        # Extract duration from metadata
        duration = chunk.metadata.get("duration_seconds", 0.0)

        # Update node metrics with failure
        metrics.node_metrics[node_id].update(duration=duration, success=False)

    def _on_tool_call(self, chunk: WorkflowStreamChunk) -> None:
        """Track tool usage."""
        workflow_id = chunk.workflow_id

        if workflow_id not in self._workflows:
            return

        metrics = self._workflows[workflow_id]

        # Extract tool calls from chunk
        tool_calls = chunk.tool_calls or []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "unknown")

            # Initialize tool metrics if needed
            if tool_name not in metrics.tool_metrics:
                metrics.tool_metrics[tool_name] = ToolUsageMetrics(tool_name=tool_name)

            # Update tool metrics
            duration = tool_call.get("duration", 0.0)
            error = tool_call.get("error")
            success = error is None
            metrics.tool_metrics[tool_name].update(duration=duration, success=success)

    def _on_workflow_complete(self, chunk: WorkflowStreamChunk, success: bool) -> None:
        """Track workflow completion."""
        workflow_id = chunk.workflow_id

        if workflow_id not in self._active_executions:
            return

        # Calculate duration
        start_time = self._active_executions[workflow_id]["start_time"]
        duration = time.time() - start_time

        # Update workflow metrics
        if workflow_id in self._workflows:
            self._workflows[workflow_id].update_execution(duration=duration, success=success)

        # Clean up active execution
        del self._active_executions[workflow_id]

        # Auto-save if enabled
        if self.auto_save:
            self.save()

    def get_workflow_metrics(self, workflow_id: str) -> Optional[WorkflowMetrics]:
        """Get metrics for a specific workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            WorkflowMetrics or None if not found
        """
        return self._workflows.get(workflow_id)

    def get_all_metrics(self) -> Dict[str, WorkflowMetrics]:
        """Get metrics for all workflows.

        Returns:
            Dictionary mapping workflow_id to WorkflowMetrics
        """
        return self._workflows.copy()

    def get_node_metrics(self, workflow_id: str, node_id: str) -> Optional[NodeMetrics]:
        """Get metrics for a specific node.

        Args:
            workflow_id: Workflow identifier
            node_id: Node identifier

        Returns:
            NodeMetrics or None if not found
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None
        return workflow.node_metrics.get(node_id)

    def get_tool_metrics(self, workflow_id: str, tool_name: str) -> Optional[ToolUsageMetrics]:
        """Get metrics for a specific tool.

        Args:
            workflow_id: Workflow identifier
            tool_name: Tool name

        Returns:
            ToolUsageMetrics or None if not found
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None
        return workflow.tool_metrics.get(tool_name)

    def reset_workflow_metrics(self, workflow_id: str) -> None:
        """Reset metrics for a specific workflow.

        Args:
            workflow_id: Workflow identifier
        """
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]

    def reset_all_metrics(self) -> None:
        """Reset all metrics."""
        self._workflows.clear()
        self._active_executions.clear()

    # =============================================================================
    # Persistence Methods
    # =============================================================================

    def save(self) -> None:
        """Save metrics to storage backend."""
        if self.storage_backend == "json":
            self._save_to_json()
        elif self.storage_backend == "sqlite":
            self._save_to_sqlite()

    def load(self) -> None:
        """Load metrics from storage backend."""
        if self.storage_backend == "json":
            self._load_from_json()
        elif self.storage_backend == "sqlite":
            self._load_from_sqlite()

    def _save_to_json(self) -> None:
        """Save metrics to JSON file."""
        if not self.storage_path:
            return

        try:
            path = Path(self.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "workflows": {
                    wf_id: metrics.to_dict() for wf_id, metrics in self._workflows.items()
                },
                "saved_at": datetime.now().isoformat(),
            }

            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=float)

            logger.debug(f"Saved metrics to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics to JSON: {e}")

    def _load_from_json(self) -> None:
        """Load metrics from JSON file."""
        if not self.storage_path:
            return

        path = Path(self.storage_path)
        if not path.exists():
            logger.debug(f"No existing metrics file at {self.storage_path}")
            return

        try:
            with open(path) as f:
                data = json.load(f)

            for wf_id, metrics_dict in data.get("workflows", {}).items():
                # Recreate workflow metrics
                workflow_metrics = WorkflowMetrics(
                    workflow_id=metrics_dict["workflow_id"],
                    workflow_name=metrics_dict.get("workflow_name", ""),
                    total_executions=metrics_dict.get("total_executions", 0),
                    successful_executions=metrics_dict.get("successful_executions", 0),
                    failed_executions=metrics_dict.get("failed_executions", 0),
                    total_duration=metrics_dict.get("total_duration", 0.0),
                    avg_duration=metrics_dict.get("avg_duration", 0.0),
                    cost_tracking=metrics_dict.get("cost_tracking", 0.0),
                    first_execution=metrics_dict.get("first_execution"),
                    last_execution=metrics_dict.get("last_execution"),
                )

                # Recreate node metrics
                for node_id, node_dict in metrics_dict.get("node_metrics", {}).items():
                    node_metrics = NodeMetrics(node_id=node_id)
                    node_metrics.execution_count = node_dict.get("execution_count", 0)
                    node_metrics.total_duration = node_dict.get("total_duration", 0.0)
                    node_metrics.avg_duration = node_dict.get("avg_duration", 0.0)
                    node_metrics.min_duration = node_dict.get("min_duration", 0.0)
                    node_metrics.max_duration = node_dict.get("max_duration", 0.0)
                    node_metrics.success_count = node_dict.get("success_count", 0)
                    node_metrics.failure_count = node_dict.get("failure_count", 0)
                    node_metrics.last_execution = node_dict.get("last_execution")
                    workflow_metrics.node_metrics[node_id] = node_metrics

                # Recreate tool metrics
                for tool_name, tool_dict in metrics_dict.get("tool_metrics", {}).items():
                    tool_metrics = ToolUsageMetrics(tool_name=tool_name)
                    tool_metrics.call_count = tool_dict.get("call_count", 0)
                    tool_metrics.total_duration = tool_dict.get("total_duration", 0.0)
                    tool_metrics.error_count = tool_dict.get("error_count", 0)
                    tool_metrics.last_used = tool_dict.get("last_used")
                    workflow_metrics.tool_metrics[tool_name] = tool_metrics

                self._workflows[wf_id] = workflow_metrics

            logger.debug(f"Loaded metrics from {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to load metrics from JSON: {e}")

    def _init_sqlite(self) -> None:
        """Initialize SQLite database."""
        if not self.storage_path:
            return

        try:
            path = Path(self.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(str(path))
            cursor = conn.cursor()

            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_metrics (
                    workflow_id TEXT PRIMARY KEY,
                    workflow_name TEXT,
                    total_executions INTEGER DEFAULT 0,
                    successful_executions INTEGER DEFAULT 0,
                    failed_executions INTEGER DEFAULT 0,
                    total_duration REAL DEFAULT 0.0,
                    avg_duration REAL DEFAULT 0.0,
                    cost_tracking REAL DEFAULT 0.0,
                    first_execution REAL,
                    last_execution REAL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS node_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT,
                    node_id TEXT,
                    execution_count INTEGER DEFAULT 0,
                    total_duration REAL DEFAULT 0.0,
                    avg_duration REAL DEFAULT 0.0,
                    min_duration REAL DEFAULT 0.0,
                    max_duration REAL DEFAULT 0.0,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    last_execution REAL,
                    FOREIGN KEY (workflow_id) REFERENCES workflow_metrics (workflow_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tool_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT,
                    tool_name TEXT,
                    call_count INTEGER DEFAULT 0,
                    total_duration REAL DEFAULT 0.0,
                    error_count INTEGER DEFAULT 0,
                    last_used REAL,
                    FOREIGN KEY (workflow_id) REFERENCES workflow_metrics (workflow_id)
                )
            """)

            conn.commit()
            conn.close()

            logger.debug(f"Initialized SQLite database at {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")

    def _save_to_sqlite(self) -> None:
        """Save metrics to SQLite database."""
        if not self.storage_path:
            return

        try:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()

            for workflow_id, metrics in self._workflows.items():
                # Upsert workflow metrics
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO workflow_metrics
                    (workflow_id, workflow_name, total_executions, successful_executions,
                     failed_executions, total_duration, avg_duration, cost_tracking,
                     first_execution, last_execution)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        workflow_id,
                        metrics.workflow_name,
                        metrics.total_executions,
                        metrics.successful_executions,
                        metrics.failed_executions,
                        metrics.total_duration,
                        metrics.avg_duration,
                        metrics.cost_tracking,
                        metrics.first_execution,
                        metrics.last_execution,
                    ),
                )

                # Delete existing node/tool metrics for this workflow
                cursor.execute("DELETE FROM node_metrics WHERE workflow_id = ?", (workflow_id,))
                cursor.execute("DELETE FROM tool_metrics WHERE workflow_id = ?", (workflow_id,))

                # Insert node metrics
                for node_id, node_metrics in metrics.node_metrics.items():
                    cursor.execute(
                        """
                        INSERT INTO node_metrics
                        (workflow_id, node_id, execution_count, total_duration, avg_duration,
                         min_duration, max_duration, success_count, failure_count, last_execution)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            workflow_id,
                            node_id,
                            node_metrics.execution_count,
                            node_metrics.total_duration,
                            node_metrics.avg_duration,
                            node_metrics.min_duration,
                            node_metrics.max_duration,
                            node_metrics.success_count,
                            node_metrics.failure_count,
                            node_metrics.last_execution,
                        ),
                    )

                # Insert tool metrics
                for tool_name, tool_metrics in metrics.tool_metrics.items():
                    cursor.execute(
                        """
                        INSERT INTO tool_metrics
                        (workflow_id, tool_name, call_count, total_duration, error_count, last_used)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            workflow_id,
                            tool_name,
                            tool_metrics.call_count,
                            tool_metrics.total_duration,
                            tool_metrics.error_count,
                            tool_metrics.last_used,
                        ),
                    )

            conn.commit()
            conn.close()

            logger.debug(f"Saved metrics to SQLite database at {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics to SQLite: {e}")

    def _load_from_sqlite(self) -> None:
        """Load metrics from SQLite database."""
        if not self.storage_path:
            return

        path = Path(self.storage_path)
        if not path.exists():
            logger.debug(f"No existing SQLite database at {self.storage_path}")
            return

        try:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()

            # Load workflow metrics
            cursor.execute("SELECT * FROM workflow_metrics")
            rows = cursor.fetchall()

            for row in rows:
                (
                    workflow_id,
                    workflow_name,
                    total_executions,
                    successful_executions,
                    failed_executions,
                    total_duration,
                    avg_duration,
                    cost_tracking,
                    first_execution,
                    last_execution,
                ) = row

                workflow_metrics = WorkflowMetrics(
                    workflow_id=workflow_id,
                    workflow_name=workflow_name,
                    total_executions=total_executions,
                    successful_executions=successful_executions,
                    failed_executions=failed_executions,
                    total_duration=total_duration,
                    avg_duration=avg_duration,
                    cost_tracking=cost_tracking,
                    first_execution=first_execution,
                    last_execution=last_execution,
                )

                self._workflows[workflow_id] = workflow_metrics

            # Load node metrics
            for row in cursor.execute("SELECT * FROM node_metrics"):
                (
                    _,
                    workflow_id,
                    node_id,
                    execution_count,
                    total_duration,
                    avg_duration,
                    min_duration,
                    max_duration,
                    success_count,
                    failure_count,
                    last_execution,
                ) = row

                if workflow_id in self._workflows:
                    node_metrics = NodeMetrics(node_id=node_id)
                    node_metrics.execution_count = execution_count
                    node_metrics.total_duration = total_duration
                    node_metrics.avg_duration = avg_duration
                    node_metrics.min_duration = min_duration
                    node_metrics.max_duration = max_duration
                    node_metrics.success_count = success_count
                    node_metrics.failure_count = failure_count
                    node_metrics.last_execution = last_execution
                    self._workflows[workflow_id].node_metrics[node_id] = node_metrics

            # Load tool metrics
            for row in cursor.execute("SELECT * FROM tool_metrics"):
                (
                    _,
                    workflow_id,
                    tool_name,
                    call_count,
                    total_duration,
                    error_count,
                    last_used,
                ) = row

                if workflow_id in self._workflows:
                    tool_metrics = ToolUsageMetrics(tool_name=tool_name)
                    tool_metrics.call_count = call_count
                    tool_metrics.total_duration = total_duration
                    tool_metrics.error_count = error_count
                    tool_metrics.last_used = last_used
                    self._workflows[workflow_id].tool_metrics[tool_name] = tool_metrics

            conn.close()

            logger.debug(f"Loaded metrics from SQLite database at {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to load metrics from SQLite: {e}")


__all__ = [
    # Metrics models
    "NodeMetrics",
    "ToolUsageMetrics",
    "WorkflowMetrics",
    # Collector
    "WorkflowMetricsCollector",
]
