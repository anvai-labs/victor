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

"""Tests for WorkflowMetricsCollector (Phase 3.0 prerequisite).

These tests verify workflow-level metrics collection including:
- Success rate tracking
- Latency metrics by node
- Tool usage statistics
- Persistence backends (JSON, SQLite)
"""

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from victor.workflows.metrics import (
    NodeMetrics,
    ToolUsageMetrics,
    WorkflowMetrics,
    WorkflowMetricsCollector,
)
from victor.workflows.streaming import WorkflowEventType, WorkflowStreamChunk

# =============================================================================
# Test: Node Metrics
# =============================================================================


class TestNodeMetrics:
    """Tests for NodeMetrics class."""

    def test_node_metrics_initialization(self):
        """Should initialize with zero values."""
        metrics = NodeMetrics(node_id="test_node")

        assert metrics.node_id == "test_node"
        assert metrics.execution_count == 0
        assert metrics.total_duration == 0.0
        assert metrics.success_count == 0
        assert metrics.failure_count == 0

    def test_node_metrics_update_success(self):
        """Should update metrics for successful execution."""
        metrics = NodeMetrics(node_id="test_node")

        metrics.update(duration=1.5, success=True)

        assert metrics.execution_count == 1
        assert metrics.total_duration == 1.5
        assert metrics.avg_duration == 1.5
        assert metrics.min_duration == 1.5
        assert metrics.max_duration == 1.5
        assert metrics.success_count == 1
        assert metrics.failure_count == 0
        assert metrics.last_execution is not None

    def test_node_metrics_update_failure(self):
        """Should update metrics for failed execution."""
        metrics = NodeMetrics(node_id="test_node")

        metrics.update(duration=0.5, success=False)

        assert metrics.execution_count == 1
        assert metrics.success_count == 0
        assert metrics.failure_count == 1

    def test_node_metrics_multiple_updates(self):
        """Should aggregate metrics across multiple executions."""
        metrics = NodeMetrics(node_id="test_node")

        metrics.update(duration=1.0, success=True)
        metrics.update(duration=2.0, success=True)
        metrics.update(duration=0.5, success=False)

        assert metrics.execution_count == 3
        assert metrics.total_duration == 3.5
        assert metrics.avg_duration == pytest.approx(3.5 / 3)
        assert metrics.min_duration == 0.5
        assert metrics.max_duration == 2.0
        assert metrics.success_count == 2
        assert metrics.failure_count == 1

    def test_node_metrics_to_dict(self):
        """Should convert to dictionary."""
        metrics = NodeMetrics(node_id="test")
        metrics.update(duration=1.0, success=True)

        data = metrics.to_dict()

        assert data["node_id"] == "test"
        assert data["execution_count"] == 1
        assert data["avg_duration"] == 1.0


# =============================================================================
# Test: Tool Usage Metrics
# =============================================================================


class TestToolUsageMetrics:
    """Tests for ToolUsageMetrics class."""

    def test_tool_metrics_initialization(self):
        """Should initialize with zero values."""
        metrics = ToolUsageMetrics(tool_name="test_tool")

        assert metrics.tool_name == "test_tool"
        assert metrics.call_count == 0
        assert metrics.total_duration == 0.0
        assert metrics.error_count == 0

    def test_tool_metrics_update_success(self):
        """Should update metrics for successful tool call."""
        metrics = ToolUsageMetrics(tool_name="test_tool")

        metrics.update(duration=0.5, success=True)

        assert metrics.call_count == 1
        assert metrics.total_duration == 0.5
        assert metrics.error_count == 0
        assert metrics.last_used is not None

    def test_tool_metrics_update_failure(self):
        """Should update metrics for failed tool call."""
        metrics = ToolUsageMetrics(tool_name="test_tool")

        metrics.update(duration=0.3, success=False)

        assert metrics.call_count == 1
        assert metrics.error_count == 1

    def test_tool_metrics_to_dict(self):
        """Should convert to dictionary."""
        metrics = ToolUsageMetrics(tool_name="test")
        metrics.update(duration=1.0, success=True)

        data = metrics.to_dict()

        assert data["tool_name"] == "test"
        assert data["call_count"] == 1


# =============================================================================
# Test: Workflow Metrics
# =============================================================================


class TestWorkflowMetrics:
    """Tests for WorkflowMetrics class."""

    def test_workflow_metrics_initialization(self):
        """Should initialize with zero values."""
        metrics = WorkflowMetrics(workflow_id="test_wf")

        assert metrics.workflow_id == "test_wf"
        assert metrics.total_executions == 0
        assert metrics.success_rate == 0.0

    def test_workflow_metrics_update_execution_success(self):
        """Should update for successful execution."""
        metrics = WorkflowMetrics(workflow_id="test_wf")

        metrics.update_execution(duration=10.0, success=True)

        assert metrics.total_executions == 1
        assert metrics.successful_executions == 1
        assert metrics.failed_executions == 0
        assert metrics.total_duration == 10.0
        assert metrics.success_rate == 1.0
        assert metrics.first_execution is not None
        assert metrics.last_execution is not None

    def test_workflow_metrics_update_execution_failure(self):
        """Should update for failed execution."""
        metrics = WorkflowMetrics(workflow_id="test_wf")

        metrics.update_execution(duration=5.0, success=False)

        assert metrics.total_executions == 1
        assert metrics.successful_executions == 0
        assert metrics.failed_executions == 1
        assert metrics.success_rate == 0.0

    def test_workflow_metrics_success_rate_calculation(self):
        """Should calculate success rate correctly."""
        metrics = WorkflowMetrics(workflow_id="test_wf")

        metrics.update_execution(duration=1.0, success=True)
        metrics.update_execution(duration=1.0, success=True)
        metrics.update_execution(duration=1.0, success=False)

        assert metrics.total_executions == 3
        assert metrics.success_rate == pytest.approx(2 / 3)

    def test_workflow_metrics_to_dict(self):
        """Should convert to dictionary."""
        metrics = WorkflowMetrics(workflow_id="test")
        metrics.update_execution(duration=10.0, success=True)

        data = metrics.to_dict()

        assert data["workflow_id"] == "test"
        assert data["total_executions"] == 1
        assert data["success_rate"] == 1.0


# =============================================================================
# Test: Metrics Collector - Basic Functionality
# =============================================================================


class TestMetricsCollectorBasic:
    """Tests for WorkflowMetricsCollector basic functionality."""

    def test_collector_initialization_memory(self):
        """Should initialize with memory backend."""
        collector = WorkflowMetricsCollector(storage_backend="memory")

        assert collector.storage_backend == "memory"
        assert len(collector._workflows) == 0

    def test_collector_initialization_json_requires_path(self):
        """Should require storage_path for JSON backend."""
        with pytest.raises(ValueError, match="storage_path required"):
            WorkflowMetricsCollector(storage_backend="json")

    def test_collector_initialization_sqlite_requires_path(self):
        """Should require storage_path for SQLite backend."""
        with pytest.raises(ValueError, match="storage_path required"):
            WorkflowMetricsCollector(storage_backend="sqlite")

    def test_collector_on_event_filter(self):
        """Should return None (accept all events)."""
        collector = WorkflowMetricsCollector(storage_backend="memory")

        assert collector.get_filter() is None


# =============================================================================
# Test: Metrics Collector - Event Handling
# =============================================================================


class TestMetricsCollectorEvents:
    """Tests for event handling in WorkflowMetricsCollector."""

    def test_on_workflow_start(self):
        """Should track workflow start."""
        collector = WorkflowMetricsCollector(storage_backend="memory")

        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.WORKFLOW_START,
            workflow_id="test_wf",
            progress=0.0,
            metadata={"workflow_name": "Test Workflow"},
        )

        collector.on_event(chunk)

        assert "test_wf" in collector._workflows
        assert "test_wf" in collector._active_executions

        metrics = collector.get_workflow_metrics("test_wf")
        assert metrics.workflow_name == "Test Workflow"

    def test_on_node_complete(self):
        """Should track node completion."""
        collector = WorkflowMetricsCollector(storage_backend="memory")

        # First start workflow
        collector.on_event(
            WorkflowStreamChunk(
                event_type=WorkflowEventType.WORKFLOW_START,
                workflow_id="test_wf",
                progress=0.0,
            )
        )

        # Then complete node
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.NODE_COMPLETE,
            workflow_id="test_wf",
            node_id="test_node",
            progress=50.0,
            metadata={"duration_seconds": 1.5},
        )

        collector.on_event(chunk)

        metrics = collector.get_workflow_metrics("test_wf")
        assert "test_node" in metrics.node_metrics

        node_metrics = metrics.node_metrics["test_node"]
        assert node_metrics.execution_count == 1
        assert node_metrics.total_duration == 1.5

    def test_on_node_error(self):
        """Should track node error."""
        collector = WorkflowMetricsCollector(storage_backend="memory")

        # Start workflow
        collector.on_event(
            WorkflowStreamChunk(
                event_type=WorkflowEventType.WORKFLOW_START,
                workflow_id="test_wf",
                progress=0.0,
            )
        )

        # Node error
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.NODE_ERROR,
            workflow_id="test_wf",
            node_id="failing_node",
            progress=50.0,
            error="Node failed",
            metadata={"duration_seconds": 0.5},
        )

        collector.on_event(chunk)

        metrics = collector.get_workflow_metrics("test_wf")
        node_metrics = metrics.node_metrics["failing_node"]

        assert node_metrics.execution_count == 1
        assert node_metrics.failure_count == 1
        assert node_metrics.success_count == 0

    def test_on_workflow_complete_success(self):
        """Should track successful workflow completion."""
        collector = WorkflowMetricsCollector(storage_backend="memory")

        # Start workflow
        collector.on_event(
            WorkflowStreamChunk(
                event_type=WorkflowEventType.WORKFLOW_START,
                workflow_id="test_wf",
                progress=0.0,
            )
        )

        # Complete workflow
        import time

        time.sleep(0.1)  # Small delay to measure duration

        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.WORKFLOW_COMPLETE,
            workflow_id="test_wf",
            progress=100.0,
            is_final=True,
            metadata={"success": True},
        )

        collector.on_event(chunk)

        metrics = collector.get_workflow_metrics("test_wf")
        assert metrics.total_executions == 1
        assert metrics.successful_executions == 1
        assert metrics.total_duration > 0

    def test_on_workflow_complete_failure(self):
        """Should track failed workflow completion."""
        collector = WorkflowMetricsCollector(storage_backend="memory")

        # Start workflow
        collector.on_event(
            WorkflowStreamChunk(
                event_type=WorkflowEventType.WORKFLOW_START,
                workflow_id="test_wf",
                progress=0.0,
            )
        )

        # Workflow error
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.WORKFLOW_ERROR,
            workflow_id="test_wf",
            progress=50.0,
            error="Workflow failed",
            is_final=True,
        )

        collector.on_event(chunk)

        metrics = collector.get_workflow_metrics("test_wf")
        assert metrics.total_executions == 1
        assert metrics.failed_executions == 1
        assert metrics.success_rate == 0.0


# =============================================================================
# Test: Metrics Collector - Tool Usage
# =============================================================================


class TestMetricsCollectorToolUsage:
    """Tests for tool usage tracking."""

    def test_on_tool_call(self):
        """Should track tool calls."""
        collector = WorkflowMetricsCollector(storage_backend="memory")

        # Start workflow
        collector.on_event(
            WorkflowStreamChunk(
                event_type=WorkflowEventType.WORKFLOW_START,
                workflow_id="test_wf",
                progress=0.0,
            )
        )

        # Tool call
        chunk = WorkflowStreamChunk(
            event_type=WorkflowEventType.AGENT_TOOL_CALL,
            workflow_id="test_wf",
            node_id="agent_node",
            progress=50.0,
            tool_calls=[
                {"name": "search", "duration": 1.0},
                {"name": "read_file", "duration": 0.5, "error": "File not found"},
            ],
        )

        collector.on_event(chunk)

        metrics = collector.get_workflow_metrics("test_wf")
        assert "search" in metrics.tool_metrics
        assert "read_file" in metrics.tool_metrics

        search_metrics = metrics.tool_metrics["search"]
        assert search_metrics.call_count == 1
        assert search_metrics.error_count == 0

        read_metrics = metrics.tool_metrics["read_file"]
        assert read_metrics.call_count == 1
        assert read_metrics.error_count == 1


# =============================================================================
# Test: Metrics Retrieval
# =============================================================================


class TestMetricsRetrieval:
    """Tests for metrics retrieval methods."""

    def test_get_workflow_metrics_exists(self):
        """Should return workflow metrics if exists."""
        collector = WorkflowMetricsCollector(storage_backend="memory")

        # Add metrics by sending events
        collector.on_event(
            WorkflowStreamChunk(
                event_type=WorkflowEventType.WORKFLOW_START,
                workflow_id="test_wf",
                progress=0.0,
            )
        )

        metrics = collector.get_workflow_metrics("test_wf")

        assert metrics is not None
        assert metrics.workflow_id == "test_wf"

    def test_get_workflow_metrics_not_exists(self):
        """Should return None if workflow not found."""
        collector = WorkflowMetricsCollector(storage_backend="memory")

        metrics = collector.get_workflow_metrics("nonexistent")

        assert metrics is None

    def test_get_all_metrics(self):
        """Should return all workflow metrics."""
        collector = WorkflowMetricsCollector(storage_backend="memory")

        # Add multiple workflows
        for wf_id in ["wf1", "wf2", "wf3"]:
            collector.on_event(
                WorkflowStreamChunk(
                    event_type=WorkflowEventType.WORKFLOW_START,
                    workflow_id=wf_id,
                    progress=0.0,
                )
            )

        all_metrics = collector.get_all_metrics()

        assert len(all_metrics) == 3
        assert "wf1" in all_metrics
        assert "wf2" in all_metrics
        assert "wf3" in all_metrics

    def test_get_node_metrics(self):
        """Should get metrics for specific node."""
        collector = WorkflowMetricsCollector(storage_backend="memory")

        # Setup
        collector.on_event(
            WorkflowStreamChunk(
                event_type=WorkflowEventType.WORKFLOW_START,
                workflow_id="test_wf",
                progress=0.0,
            )
        )

        collector.on_event(
            WorkflowStreamChunk(
                event_type=WorkflowEventType.NODE_COMPLETE,
                workflow_id="test_wf",
                node_id="test_node",
                progress=50.0,
                metadata={"duration_seconds": 1.0},
            )
        )

        node_metrics = collector.get_node_metrics("test_wf", "test_node")

        assert node_metrics is not None
        assert node_metrics.node_id == "test_node"
        assert node_metrics.execution_count == 1

    def test_get_tool_metrics(self):
        """Should get metrics for specific tool."""
        collector = WorkflowMetricsCollector(storage_backend="memory")

        # Setup
        collector.on_event(
            WorkflowStreamChunk(
                event_type=WorkflowEventType.WORKFLOW_START,
                workflow_id="test_wf",
                progress=0.0,
            )
        )

        collector.on_event(
            WorkflowStreamChunk(
                event_type=WorkflowEventType.AGENT_TOOL_CALL,
                workflow_id="test_wf",
                node_id="agent",
                progress=50.0,
                tool_calls=[{"name": "search", "duration": 1.0}],
            )
        )

        tool_metrics = collector.get_tool_metrics("test_wf", "search")

        assert tool_metrics is not None
        assert tool_metrics.tool_name == "search"
        assert tool_metrics.call_count == 1


# =============================================================================
# Test: Metrics Reset
# =============================================================================


class TestMetricsReset:
    """Tests for metrics reset functionality."""

    def test_reset_workflow_metrics(self):
        """Should reset metrics for specific workflow."""
        collector = WorkflowMetricsCollector(storage_backend="memory")

        # Add workflow
        collector.on_event(
            WorkflowStreamChunk(
                event_type=WorkflowEventType.WORKFLOW_START,
                workflow_id="test_wf",
                progress=0.0,
            )
        )

        assert collector.get_workflow_metrics("test_wf") is not None

        # Reset
        collector.reset_workflow_metrics("test_wf")

        assert collector.get_workflow_metrics("test_wf") is None

    def test_reset_all_metrics(self):
        """Should reset all metrics."""
        collector = WorkflowMetricsCollector(storage_backend="memory")

        # Add multiple workflows
        for wf_id in ["wf1", "wf2"]:
            collector.on_event(
                WorkflowStreamChunk(
                    event_type=WorkflowEventType.WORKFLOW_START,
                    workflow_id=wf_id,
                    progress=0.0,
                )
            )

        assert len(collector.get_all_metrics()) == 2

        # Reset all
        collector.reset_all_metrics()

        assert len(collector.get_all_metrics()) == 0


# =============================================================================
# Test: JSON Persistence
# =============================================================================


class TestJSONPersistence:
    """Tests for JSON storage backend."""

    def test_json_save_and_load(self):
        """Should save and load metrics from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "metrics.json")

            # Create collector and add metrics
            collector = WorkflowMetricsCollector(
                storage_backend="json", storage_path=json_path, auto_save=False
            )

            collector.on_event(
                WorkflowStreamChunk(
                    event_type=WorkflowEventType.WORKFLOW_START,
                    workflow_id="test_wf",
                    progress=0.0,
                    metadata={"workflow_name": "Test"},
                )
            )

            collector.on_event(
                WorkflowStreamChunk(
                    event_type=WorkflowEventType.WORKFLOW_COMPLETE,
                    workflow_id="test_wf",
                    progress=100.0,
                    is_final=True,
                )
            )

            # Save
            collector.save()

            # Create new collector and load
            collector2 = WorkflowMetricsCollector(
                storage_backend="json", storage_path=json_path, auto_save=False
            )
            collector2.load()

            metrics = collector2.get_workflow_metrics("test_wf")
            assert metrics is not None
            assert metrics.workflow_name == "Test"
            assert metrics.total_executions == 1

    def test_json_load_creates_file_if_not_exists(self):
        """Should not error if JSON file doesn't exist on load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "nonexistent.json")

            collector = WorkflowMetricsCollector(
                storage_backend="json", storage_path=json_path, auto_save=False
            )

            # Should not raise error
            collector.load()

            assert len(collector.get_all_metrics()) == 0


# =============================================================================
# Test: SQLite Persistence
# =============================================================================


class TestSQLitePersistence:
    """Tests for SQLite storage backend."""

    def test_sqlite_init_creates_tables(self):
        """Should initialize SQLite database with tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "metrics.db")

            collector = WorkflowMetricsCollector(
                storage_backend="sqlite", storage_path=db_path, auto_save=False
            )

            # Verify tables exist
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables = [row[0] for row in cursor.fetchall()]

            assert "workflow_metrics" in tables
            assert "node_metrics" in tables
            assert "tool_metrics" in tables

            conn.close()

    def test_sqlite_save_and_load(self):
        """Should save and load metrics from SQLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "metrics.db")

            # Create collector and add metrics
            collector = WorkflowMetricsCollector(
                storage_backend="sqlite", storage_path=db_path, auto_save=False
            )

            collector.on_event(
                WorkflowStreamChunk(
                    event_type=WorkflowEventType.WORKFLOW_START,
                    workflow_id="test_wf",
                    progress=0.0,
                    metadata={"workflow_name": "Test"},
                )
            )

            collector.on_event(
                WorkflowStreamChunk(
                    event_type=WorkflowEventType.NODE_COMPLETE,
                    workflow_id="test_wf",
                    node_id="test_node",
                    progress=50.0,
                    metadata={"duration_seconds": 1.5},
                )
            )

            collector.on_event(
                WorkflowStreamChunk(
                    event_type=WorkflowEventType.WORKFLOW_COMPLETE,
                    workflow_id="test_wf",
                    progress=100.0,
                    is_final=True,
                )
            )

            # Save
            collector.save()

            # Create new collector and load
            collector2 = WorkflowMetricsCollector(
                storage_backend="sqlite", storage_path=db_path, auto_save=False
            )
            collector2.load()

            metrics = collector2.get_workflow_metrics("test_wf")
            assert metrics is not None
            assert metrics.workflow_name == "Test"
            assert metrics.total_executions == 1
            assert "test_node" in metrics.node_metrics

    def test_sqlite_load_creates_db_if_not_exists(self):
        """Should create database if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "new_metrics.db")

            # Database doesn't exist yet
            assert not os.path.exists(db_path)

            collector = WorkflowMetricsCollector(
                storage_backend="sqlite", storage_path=db_path, auto_save=False
            )

            # Load should initialize database
            collector.load()

            assert os.path.exists(db_path)
