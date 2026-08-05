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

"""A/B test manager for experiment lifecycle management.

This module provides the main ABTestManager class for creating, managing,
and analyzing A/B experiments.
"""

import asyncio
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.experiments.ab_testing.models import (
    AllocationStrategy,
    AggregatedMetrics,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    ExperimentVariant,
    ExecutionMetrics,
    VariantResult,
)
from victor.experiments.ab_testing.allocator import (
    StickyAllocator,
    TrafficAllocator,
    create_allocator,
)
from victor.experiments.ab_testing.paths import get_default_ab_test_db_path


class ABTestManager:
    """Manages A/B test lifecycle.

    This class handles:
    - Creating and storing experiments
    - Starting and stopping experiments
    - Allocating variants to users
    - Recording execution metrics
    - Analyzing results with statistical tests

    Usage:
        manager = ABTestManager(storage_path="/path/to/experiments.db")

        # Create experiment
        experiment_id = await manager.create_experiment(config)

        # Start experiment
        await manager.start_experiment(experiment_id)

        # Allocate variant
        variant_id = await manager.allocate_variant(experiment_id, user_id)

        # Record execution
        await manager.record_execution(metrics)

        # Get results
        results = await manager.get_results(experiment_id)
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        allocation_strategy: AllocationStrategy = AllocationStrategy.STICKY,
    ):
        """Initialize A/B test manager.

        Args:
            storage_path: Path to SQLite database (default: ~/.victor/ab_tests.db)
            allocation_strategy: Default allocation strategy
        """
        if storage_path is None:
            storage_path = str(get_default_ab_test_db_path())

        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize allocator
        self.allocator = create_allocator(allocation_strategy)

        # Initialize database
        self._init_database()

        # In-memory cache for active experiments
        self._active_experiments: Dict[str, ExperimentConfig] = {}
        self._experiment_status: Dict[str, ExperimentStatus] = {}

    def _init_database(self) -> None:
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                hypothesis TEXT,
                config_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                paused_at REAL,
                tags_json TEXT
            )
        """)

        # Variants table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS variants (
                experiment_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                workflow_type TEXT,
                config_json TEXT,
                parameter_overrides_json TEXT,
                traffic_weight REAL NOT NULL,
                is_control INTEGER NOT NULL DEFAULT 0,
                tags_json TEXT,
                PRIMARY KEY (experiment_id, variant_id),
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
            )
        """)

        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                experiment_id TEXT NOT NULL,
                metric_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                metric_type TEXT NOT NULL,
                optimization_goal TEXT NOT NULL,
                is_primary INTEGER NOT NULL DEFAULT 0,
                config_json TEXT,
                PRIMARY KEY (experiment_id, metric_id),
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
            )
        """)

        # Executions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS executions (
                execution_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                execution_time REAL NOT NULL,
                node_times_json TEXT,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                tool_calls_count INTEGER NOT NULL DEFAULT 0,
                tool_calls_by_name_json TEXT,
                tool_errors INTEGER NOT NULL DEFAULT 0,
                success INTEGER NOT NULL DEFAULT 1,
                error_message TEXT,
                estimated_cost REAL NOT NULL DEFAULT 0.0,
                custom_metrics_json TEXT,
                timestamp REAL NOT NULL,
                workflow_name TEXT,
                workflow_type TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
            )
        """)

        # Allocations table (for sticky allocation tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS allocations (
                experiment_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                allocated_at REAL NOT NULL,
                context_json TEXT,
                PRIMARY KEY (experiment_id, user_id),
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
            )
        """)

        # Results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                experiment_id TEXT PRIMARY KEY,
                winning_variant_id TEXT,
                confidence REAL,
                statistical_significance INTEGER NOT NULL DEFAULT 0,
                p_value REAL,
                confidence_interval_json TEXT,
                variant_results_json TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                reasoning TEXT,
                analyzed_at REAL NOT NULL,
                total_samples INTEGER NOT NULL DEFAULT 0,
                total_duration_seconds REAL NOT NULL DEFAULT 0.0,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
            )
        """)

        # Create indexes for performance
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_executions_experiment_variant ON executions(experiment_id, variant_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_executions_timestamp ON executions(timestamp)"
        )

        conn.commit()
        conn.close()

    async def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment.

        Args:
            config: Experiment configuration

        Returns:
            experiment_id: The created experiment ID

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate configuration
        if not config.variants:
            raise ValueError("Experiment must have at least one variant")

        if not config.primary_metric:
            raise ValueError("Experiment must have a primary metric")

        # Validate traffic weights
        total_weight = sum(v.traffic_weight for v in config.variants)
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Traffic weights must sum to 1.0, got {total_weight}")

        # Validate at least one control
        if not any(v.is_control for v in config.variants):
            raise ValueError("Experiment must have at least one control variant")

        # Store in database
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        import json

        # Insert experiment
        cursor.execute(
            """
            INSERT INTO experiments (
                experiment_id, name, description, hypothesis, config_json,
                status, created_at, tags_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                config.experiment_id,
                config.name,
                config.description,
                config.hypothesis,
                json.dumps(asdict(config)),
                "draft",
                config.created_at,
                json.dumps(config.tags),
            ),
        )

        # Insert variants
        for variant in config.variants:
            cursor.execute(
                """
                INSERT INTO variants (
                    experiment_id, variant_id, name, description,
                    workflow_type, config_json, parameter_overrides_json,
                    traffic_weight, is_control, tags_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    config.experiment_id,
                    variant.variant_id,
                    variant.name,
                    variant.description,
                    variant.workflow_type,
                    json.dumps(variant.workflow_config),
                    json.dumps(variant.parameter_overrides),
                    variant.traffic_weight,
                    1 if variant.is_control else 0,
                    json.dumps(variant.tags),
                ),
            )

        # Insert primary metric
        cursor.execute(
            """
            INSERT INTO metrics (
                experiment_id, metric_id, name, description,
                metric_type, optimization_goal, is_primary, config_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                config.experiment_id,
                config.primary_metric.metric_id,
                config.primary_metric.name,
                config.primary_metric.description,
                config.primary_metric.metric_type,
                config.primary_metric.optimization_goal,
                1,
                json.dumps(asdict(config.primary_metric)),
            ),
        )

        # Insert secondary metrics
        for metric in config.secondary_metrics:
            cursor.execute(
                """
                INSERT INTO metrics (
                    experiment_id, metric_id, name, description,
                    metric_type, optimization_goal, is_primary, config_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    config.experiment_id,
                    metric.metric_id,
                    metric.name,
                    metric.description,
                    metric.metric_type,
                    metric.optimization_goal,
                    0,
                    json.dumps(asdict(metric)),
                ),
            )

        conn.commit()
        conn.close()

        # Cache experiment
        self._active_experiments[config.experiment_id] = config
        self._experiment_status[config.experiment_id] = ExperimentStatus(status="draft")

        return config.experiment_id

    async def start_experiment(self, experiment_id: str) -> None:
        """Start an experiment.

        Args:
            experiment_id: Experiment identifier

        Raises:
            ValueError: If experiment not found or already running
        """
        # Load experiment
        experiment = await self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Update status
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE experiments
            SET status = 'running', started_at = ?
            WHERE experiment_id = ?
        """,
            (time.time(), experiment_id),
        )

        conn.commit()
        conn.close()

        # Update cache
        if experiment_id in self._experiment_status:
            status = self._experiment_status[experiment_id]
            status.status = "running"
            status.started_at = time.time()

    async def stop_experiment(self, experiment_id: str) -> None:
        """Stop an experiment.

        Args:
            experiment_id: Experiment identifier

        Raises:
            ValueError: If experiment not found
        """
        # Update status
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE experiments
            SET status = 'completed', completed_at = ?
            WHERE experiment_id = ?
        """,
            (time.time(), experiment_id),
        )

        conn.commit()
        conn.close()

        # Update cache
        if experiment_id in self._experiment_status:
            status = self._experiment_status[experiment_id]
            status.status = "completed"
            status.completed_at = time.time()

    async def get_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """Get experiment configuration.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Experiment configuration or None if not found
        """
        # Check cache first
        if experiment_id in self._active_experiments:
            return self._active_experiments[experiment_id]

        # Load from database
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT config_json FROM experiments WHERE experiment_id = ?
        """,
            (experiment_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        import json

        config_dict = json.loads(row[0])
        config = ExperimentConfig(**config_dict)

        # Cache it
        self._active_experiments[experiment_id] = config

        return config

    async def allocate_variant(
        self,
        experiment_id: str,
        user_id: str,
        context: Optional[Dict] = None,
    ) -> str:
        """Allocate a user to a variant.

        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            context: Optional context for allocation

        Returns:
            variant_id: The allocated variant ID

        Raises:
            ValueError: If experiment not found or not running
        """
        # Get experiment
        experiment = await self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Check if running
        status = await self.get_status(experiment_id)
        if not status or status.status != "running":
            raise ValueError(f"Experiment {experiment_id} is not running")

        # Check for existing allocation (for sticky allocator)
        if isinstance(self.allocator, StickyAllocator):
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT variant_id FROM allocations
                WHERE experiment_id = ? AND user_id = ?
            """,
                (experiment_id, user_id),
            )

            row = cursor.fetchone()
            conn.close()

            if row:
                return row[0]

        # Allocate new variant
        variant_id = await self.allocator.allocate_variant(user_id, experiment, context)

        # Store allocation
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        import json

        cursor.execute(
            """
            INSERT OR REPLACE INTO allocations (
                experiment_id, user_id, variant_id, allocated_at, context_json
            ) VALUES (?, ?, ?, ?, ?)
        """,
            (
                experiment_id,
                user_id,
                variant_id,
                time.time(),
                json.dumps(context or {}),
            ),
        )

        conn.commit()
        conn.close()

        return variant_id

    async def record_execution(self, metrics: ExecutionMetrics) -> None:
        """Record execution metrics.

        Args:
            metrics: Execution metrics to record
        """
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        import json

        cursor.execute(
            """
            INSERT INTO executions (
                execution_id, experiment_id, variant_id, user_id,
                execution_time, node_times_json,
                prompt_tokens, completion_tokens, total_tokens,
                tool_calls_count, tool_calls_by_name_json, tool_errors,
                success, error_message, estimated_cost, custom_metrics_json,
                timestamp, workflow_name, workflow_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metrics.execution_id,
                metrics.experiment_id,
                metrics.variant_id,
                metrics.user_id,
                metrics.execution_time,
                json.dumps(metrics.node_times),
                metrics.prompt_tokens,
                metrics.completion_tokens,
                metrics.total_tokens,
                metrics.tool_calls_count,
                json.dumps(metrics.tool_calls_by_name),
                metrics.tool_errors,
                1 if metrics.success else 0,
                metrics.error_message,
                metrics.estimated_cost,
                json.dumps(metrics.custom_metrics),
                metrics.timestamp,
                metrics.workflow_name,
                metrics.workflow_type,
            ),
        )

        conn.commit()
        conn.close()

        # Update cache
        if metrics.experiment_id in self._experiment_status:
            status = self._experiment_status[metrics.experiment_id]
            status.total_samples += 1
            status.variant_samples[metrics.variant_id] = (
                status.variant_samples.get(metrics.variant_id, 0) + 1
            )

    async def get_status(self, experiment_id: str) -> Optional[ExperimentStatus]:
        """Get experiment status.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Experiment status or None if not found
        """
        # Check cache first
        if experiment_id in self._experiment_status:
            return self._experiment_status[experiment_id]

        # Load from database
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT status, started_at, completed_at, paused_at
            FROM experiments WHERE experiment_id = ?
        """,
            (experiment_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        status = ExperimentStatus(
            status=row[0],
            started_at=row[1],
            completed_at=row[2],
            paused_at=row[3],
        )

        # Load variant sample counts
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT variant_id, COUNT(*) as count
            FROM executions
            WHERE experiment_id = ?
            GROUP BY variant_id
        """,
            (experiment_id,),
        )

        for row in cursor.fetchall():
            status.variant_samples[row[0]] = row[1]
            status.total_samples += row[1]

        conn.close()

        # Cache it
        self._experiment_status[experiment_id] = status

        return status

    async def get_results(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get experiment results.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Experiment results or None if not analyzed yet
        """
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT winning_variant_id, confidence, statistical_significance,
                   p_value, confidence_interval_json, variant_results_json,
                   recommendation, reasoning, analyzed_at, total_samples,
                   total_duration_seconds
            FROM results WHERE experiment_id = ?
        """,
            (experiment_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        import json

        result = ExperimentResult(
            experiment_id=experiment_id,
            winning_variant_id=row[0],
            confidence=row[1],
            statistical_significance=bool(row[2]),
            p_value=row[3],
            confidence_interval=json.loads(row[4]) if row[4] else None,
            variant_results={k: VariantResult(**v) for k, v in json.loads(row[5]).items()},
            recommendation=row[6],
            reasoning=row[7],
            analyzed_at=row[8],
            total_samples=row[9],
            total_duration_seconds=row[10],
        )

        return result
