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

"""Tests for Bayesian orchestration monitoring."""

import sqlite3
from pathlib import Path

import pytest

from victor.framework.rl.monitoring.bayesian_monitor import (
    ASCIIChartRenderer,
    BayesianMetricsMonitor,
    MetricsExporter,
)


class TestBayesianMetricsMonitor:
    """Test Bayesian metrics monitor."""

    def test_init_with_db(self, tmp_path):
        """Test initialization with database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        monitor = BayesianMetricsMonitor(conn)

        assert monitor.db == conn

    def test_get_belief_evolution(self, tmp_path):
        """Test getting belief evolution."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        # Create tables using BayesianOrchestrationService
        BayesianOrchestrationService = __import__(
            "victor.framework.rl.orchestration.bayesian_orchestrator",
            fromlist=["BayesianOrchestrationService"],
        ).BayesianOrchestrationService

        service = BayesianOrchestrationService(
            db_connection=conn,
            observation_learner=None,
            reliability_learner=None,
            voi_controller=None,
        )

        monitor = BayesianMetricsMonitor(conn)

        # Insert test data
        conn.execute(
            """INSERT INTO rl_belief_history
               (belief_id, success_prob, failure_prob, entropy, agent_id, message, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "test_belief",
                0.5,
                0.5,
                0.693,
                "agent_a",
                "Yes",
                datetime.now().isoformat(),
            ),
        )

        # Get evolution
        evolution = monitor.get_belief_evolution("test_belief")

        assert len(evolution) == 1
        assert evolution[0]["success_prob"] == 0.5

    def test_get_system_summary(self, tmp_path):
        """Test getting system summary."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        # Create tables
        BayesianOrchestrationService = __import__(
            "victor.framework.rl.orchestration.bayesian_orchestrator",
            fromlist=["BayesianOrchestrationService"],
        ).BayesianOrchestrationService

        service = BayesianOrchestrationService(
            db_connection=conn,
            observation_learner=None,
            reliability_learner=None,
            voi_controller=None,
        )

        # Insert test data into various tables
        conn.execute(
            """INSERT INTO rl_belief_history
               (belief_id, success_prob, failure_prob, entropy, agent_id, message, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "test_belief",
                0.5,
                0.5,
                0.693,
                "agent_a",
                "Yes",
                datetime.now().isoformat(),
            ),
        )

        monitor = BayesianMetricsMonitor(conn)
        summary = monitor.get_system_summary(days=7)

        assert "belief_states" in summary
        assert summary["belief_states"]["unique_belief_states"] >= 1


class TestMetricsExporter:
    """Test metrics exporter."""

    def test_export_belief_evolution_csv(self, tmp_path):
        """Test exporting belief evolution to CSV."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        # Create tables and insert test data
        BayesianOrchestrationService = __import__(
            "victor.framework.rl.orchestration.bayesian_orchestrator",
            fromlist=["BayesianOrchestrationService"],
        ).BayesianOrchestrationService

        service = BayesianOrchestrationService(
            db_connection=conn,
            observation_learner=None,
            reliability_learner=None,
            voi_controller=None,
        )

        # Insert test data
        conn.execute(
            """INSERT INTO rl_belief_history
               (belief_id, success_prob, failure_prob, entropy, agent_id, message, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "test_belief",
                0.5,
                0.5,
                0.693,
                "agent_a",
                "Yes",
                datetime.now().isoformat(),
            ),
        )

        monitor = BayesianMetricsMonitor(conn)
        exporter = MetricsExporter(monitor)

        # Export to CSV
        output_path = tmp_path / "belief_evolution.csv"
        exporter.export_belief_evolution_csv("test_belief", str(output_path))

        # Check file was created
        assert output_path.exists()

        # Check contents
        content = output_path.read_text()
        assert "timestamp,success_prob,failure_prob" in content
        assert "0.5000" in content

    def test_export_summary_json(self, tmp_path):
        """Test exporting summary to JSON."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        # Create tables
        BayesianOrchestrationService = __import__(
            "victor.framework.rl.orchestration.bayesian_orchestrator",
            fromlist=["BayesianOrchestrationService"],
        ).BayesianOrchestrationService

        service = BayesianOrchestrationService(
            db_connection=conn,
            observation_learner=None,
            reliability_learner=None,
            voi_controller=None,
        )

        monitor = BayesianMetricsMonitor(conn)
        exporter = MetricsExporter(monitor)

        # Export to JSON
        output_path = tmp_path / "summary.json"
        exporter.export_summary_json(str(output_path), days=7)

        # Check file was created
        assert output_path.exists()

        # Check contents
        content = output_path.read_text()
        assert "summary" in content
        assert "consensus_performance" in content


class TestASCIIChartRenderer:
    """Test ASCII chart renderer."""

    def test_render_bar_chart(self):
        """Test rendering bar chart."""
        data = {"agent_a": 0.8, "agent_b": 0.6, "agent_c": 0.4}

        renderer = ASCIIChartRenderer()
        chart = renderer.render_bar_chart(data, "Test Chart")

        assert "Test Chart" in chart
        assert "agent_a" in chart
        assert "agent_b" in chart
        assert "agent_c" in chart

    def test_render_empty_bar_chart(self):
        """Test rendering empty bar chart."""
        data = {}

        renderer = ASCIIChartRenderer()
        chart = renderer.render_bar_chart(data, "Empty Chart")

        assert "Empty Chart" in chart
        assert "(No data)" in chart

    def test_render_line_chart(self):
        """Test rendering line chart."""
        data = [("t1", 0.5), ("t2", 0.7), ("t3", 0.9)]

        renderer = ASCIIChartRenderer()
        chart = renderer.render_line_chart(data, "Trend Chart")

        assert "Trend Chart" in chart
        assert "t1" in chart or "t2" in chart or "t3" in chart

    def test_render_heatmap(self):
        """Test rendering correlation heatmap."""
        matrix = {
            "agent_a": {"agent_a": 1.0, "agent_b": 0.8, "agent_c": -0.5},
            "agent_b": {"agent_a": 0.8, "agent_b": 1.0, "agent_c": 0.3},
            "agent_c": {"agent_a": -0.5, "agent_b": 0.3, "agent_c": 1.0},
        }

        renderer = ASCIIChartRenderer()
        heatmap = renderer.render_heatmap(matrix, "Correlation Matrix")

        assert "Correlation Matrix" in heatmap
        assert "Legend:" in heatmap
        assert "agent_a" in heatmap

    def test_render_empty_heatmap(self):
        """Test rendering empty heatmap."""
        matrix = {}

        renderer = ASCIIChartRenderer()
        heatmap = renderer.render_heatmap(matrix, "Empty Matrix")

        assert "Empty Matrix" in heatmap
        assert "(No data)" in heatmap


# Import for tests
from datetime import datetime
