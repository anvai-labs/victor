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

"""SQLite storage backend for experiment tracking.

This module provides a SQLite-based implementation of the storage backend,
suitable for local development and single-user deployments.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.experiments.entities import (
    Artifact,
    Experiment,
    ExperimentQuery,
    Metric,
    Run,
)
from victor.experiments.storage import IStorageBackend, StorageBackendError


class SQLiteStorage:
    """SQLite-based storage backend for experiment tracking.

    This implementation uses SQLite for persistent storage, with automatic
    schema creation and migration support.

    Thread-safe: Uses connection-per-thread pattern with SQLite.

    Args:
        db_path: Path to SQLite database file (default: ~/.victor/experiments.db)

    Example:
        storage = SQLiteStorage()
        experiment = Experiment(name="test")
        storage.create_experiment(experiment)
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize SQLite storage.

        Args:
            db_path: Path to database file. Defaults to ~/.victor/experiments.db
        """
        if db_path is None:
            victor_dir = Path.home() / ".victor"
            victor_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(victor_dir / "experiments.db")

        self.db_path = Path(db_path).expanduser()
        self._local = threading.local()
        self._ensure_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _ensure_tables(self) -> None:
        """Create database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                hypothesis TEXT,
                tags TEXT,
                parameters TEXT,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                git_commit_sha TEXT,
                git_branch TEXT,
                git_dirty INTEGER DEFAULT 0,
                parent_id TEXT,
                group_id TEXT,
                workflow_name TEXT,
                vertical TEXT DEFAULT 'coding',
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (parent_id) REFERENCES experiments(experiment_id)
            )
        """)

        # Runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                metrics_summary TEXT,
                parameters TEXT,
                error_message TEXT,
                python_version TEXT,
                os_info TEXT,
                victor_version TEXT,
                dependencies TEXT,
                provider TEXT,
                model TEXT,
                task_type TEXT,
                artifact_count INTEGER DEFAULT 0,
                artifact_size_bytes INTEGER DEFAULT 0,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
            )
        """)

        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                step INTEGER,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            )
        """)

        # Artifacts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            )
        """)

        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_key ON metrics(key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id)")

        conn.commit()

    # Experiment operations

    def create_experiment(self, experiment: Experiment) -> str:
        """Create a new experiment."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO experiments (
                    experiment_id, name, description, hypothesis, tags, parameters,
                    created_at, status, git_commit_sha, git_branch, git_dirty,
                    parent_id, group_id, workflow_name, vertical, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.experiment_id,
                    experiment.name,
                    experiment.description,
                    experiment.hypothesis,
                    json.dumps(experiment.tags),
                    json.dumps(experiment.parameters),
                    experiment.created_at.isoformat(),
                    experiment.status.value,
                    experiment.git_commit_sha,
                    experiment.git_branch,
                    1 if experiment.git_dirty else 0,
                    experiment.parent_id,
                    experiment.group_id,
                    experiment.workflow_name,
                    experiment.vertical,
                    (experiment.started_at.isoformat() if experiment.started_at else None),
                    (experiment.completed_at.isoformat() if experiment.completed_at else None),
                ),
            )
            conn.commit()
            return experiment.experiment_id
        except sqlite3.IntegrityError as e:
            raise StorageBackendError(f"Experiment already exists: {e}") from e
        except sqlite3.Error as e:
            raise StorageBackendError(f"Failed to create experiment: {e}") from e

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_experiment(row)

    def update_experiment(self, experiment_id: str, updates: Dict[str, Any]) -> bool:
        """Update an experiment."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Build UPDATE statement dynamically
        set_clauses = []
        values = []

        for key, value in updates.items():
            if key in ["tags", "parameters"]:
                value = json.dumps(value)
            elif key == "status" and hasattr(value, "value"):
                value = value.value
            elif isinstance(value, datetime):
                value = value.isoformat()
            elif key == "git_dirty":
                value = 1 if value else 0

            set_clauses.append(f"{key} = ?")
            values.append(value)

        if not set_clauses:
            return False

        values.append(experiment_id)
        sql = f"UPDATE experiments SET {', '.join(set_clauses)} WHERE experiment_id = ?"

        cursor.execute(sql, values)
        conn.commit()

        return cursor.rowcount > 0

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and all its runs."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
        conn.commit()

        return cursor.rowcount > 0

    def list_experiments(self, query: Optional[ExperimentQuery] = None) -> List[Experiment]:
        """List experiments with optional filtering."""
        conn = self._get_connection()
        cursor = conn.cursor()

        sql = "SELECT * FROM experiments WHERE 1=1"
        params = []

        if query:
            if query.name_contains:
                sql += " AND name LIKE ?"
                params.append(f"%{query.name_contains}%")

            if query.status:
                sql += " AND status = ?"
                params.append(query.status.value)

            if query.tags_any:
                # Check if any tag in list matches
                tag_conditions = []
                for tag in query.tags_any:
                    tag_conditions.append("tags LIKE ?")
                    params.append(f"%{tag}%")
                sql += f" AND ({' OR '.join(tag_conditions)})"

            # Sorting
            order_column = (
                query.sort_by if query.sort_by in ["name", "created_at", "status"] else "created_at"
            )
            order_direction = query.sort_order.upper()
            sql += f" ORDER BY {order_column} {order_direction}"

            # Pagination
            sql += " LIMIT ? OFFSET ?"
            params.extend([query.limit, query.offset])

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        return [self._row_to_experiment(row) for row in rows]

    # Run operations

    def create_run(self, run: Run) -> str:
        """Create a new run."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO runs (
                    run_id, experiment_id, name, status, started_at, completed_at,
                    metrics_summary, parameters, error_message, python_version,
                    os_info, victor_version, dependencies, provider, model,
                    task_type, artifact_count, artifact_size_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.experiment_id,
                    run.name,
                    run.status.value,
                    run.started_at.isoformat(),
                    run.completed_at.isoformat() if run.completed_at else None,
                    json.dumps(run.metrics_summary),
                    json.dumps(run.parameters),
                    run.error_message,
                    run.python_version,
                    run.os_info,
                    run.victor_version,
                    json.dumps(run.dependencies),
                    run.provider,
                    run.model,
                    run.task_type,
                    run.artifact_count,
                    run.artifact_size_bytes,
                ),
            )
            conn.commit()
            return run.run_id
        except sqlite3.Error as e:
            raise StorageBackendError(f"Failed to create run: {e}") from e

    def get_run(self, run_id: str) -> Optional[Run]:
        """Get a run by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_run(row)

    def update_run(self, run_id: str, updates: Dict[str, Any]) -> bool:
        """Update a run."""
        conn = self._get_connection()
        cursor = conn.cursor()

        set_clauses = []
        values = []

        for key, value in updates.items():
            if key in ["metrics_summary", "parameters", "dependencies"]:
                value = json.dumps(value)
            elif key == "status" and hasattr(value, "value"):
                value = value.value
            elif isinstance(value, datetime):
                value = value.isoformat()

            set_clauses.append(f"{key} = ?")
            values.append(value)

        if not set_clauses:
            return False

        values.append(run_id)
        sql = f"UPDATE runs SET {', '.join(set_clauses)} WHERE run_id = ?"

        cursor.execute(sql, values)
        conn.commit()

        return cursor.rowcount > 0

    def list_runs(self, experiment_id: str) -> List[Run]:
        """List all runs for an experiment."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM runs WHERE experiment_id = ? ORDER BY started_at DESC",
            (experiment_id,),
        )
        rows = cursor.fetchall()

        return [self._row_to_run(row) for row in rows]

    # Metric operations

    def log_metric(self, metric: Metric) -> None:
        """Log a metric for a run."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO metrics (run_id, key, value, timestamp, step)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    metric.run_id,
                    metric.key,
                    metric.value,
                    metric.timestamp.isoformat(),
                    metric.step,
                ),
            )
            conn.commit()
        except sqlite3.Error as e:
            raise StorageBackendError(f"Failed to log metric: {e}") from e

    def get_metrics(self, run_id: str) -> List[Metric]:
        """Get all metrics for a run."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM metrics WHERE run_id = ? ORDER BY timestamp", (run_id,))
        rows = cursor.fetchall()

        return [
            Metric(
                run_id=row["run_id"],
                key=row["key"],
                value=row["value"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                step=row["step"],
            )
            for row in rows
        ]

    def get_metric_history(self, run_id: str, metric_key: str) -> List[Metric]:
        """Get history of a specific metric."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM metrics
            WHERE run_id = ? AND key = ?
            ORDER BY timestamp
            """,
            (run_id, metric_key),
        )
        rows = cursor.fetchall()

        return [
            Metric(
                run_id=row["run_id"],
                key=row["key"],
                value=row["value"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                step=row["step"],
            )
            for row in rows
        ]

    # Artifact operations

    def log_artifact(self, artifact: Artifact) -> None:
        """Log an artifact for a run."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, run_id, artifact_type, filename, file_path,
                    file_size_bytes, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.artifact_id,
                    artifact.run_id,
                    artifact.artifact_type.value,
                    artifact.filename,
                    artifact.file_path,
                    artifact.file_size_bytes,
                    artifact.created_at.isoformat(),
                    json.dumps(artifact.metadata),
                ),
            )
            conn.commit()
        except sqlite3.Error as e:
            raise StorageBackendError(f"Failed to log artifact: {e}") from e

    def get_artifacts(self, run_id: str) -> List[Artifact]:
        """Get all artifacts for a run."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        )
        rows = cursor.fetchall()

        return [self._row_to_artifact(row) for row in rows]

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Get an artifact by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_artifact(row)

    # Utility methods

    def close(self) -> None:
        """Close the storage backend."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            delattr(self._local, "conn")

    # Helper methods

    def _row_to_experiment(self, row: sqlite3.Row) -> Experiment:
        """Convert database row to Experiment entity."""
        return Experiment(
            experiment_id=row["experiment_id"],
            name=row["name"],
            description=row["description"] or "",
            hypothesis=row["hypothesis"] or "",
            tags=json.loads(row["tags"]) if row["tags"] else [],
            parameters=json.loads(row["parameters"]) if row["parameters"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
            status=row["status"],
            git_commit_sha=row["git_commit_sha"] or "",
            git_branch=row["git_branch"] or "",
            git_dirty=bool(row["git_dirty"]),
            parent_id=row["parent_id"],
            group_id=row["group_id"],
            workflow_name=row["workflow_name"],
            vertical=row["vertical"] or "coding",
            started_at=(datetime.fromisoformat(row["started_at"]) if row["started_at"] else None),
            completed_at=(
                datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
            ),
        )

    def _row_to_run(self, row: sqlite3.Row) -> Run:
        """Convert database row to Run entity."""
        return Run(
            run_id=row["run_id"],
            experiment_id=row["experiment_id"],
            name=row["name"],
            status=row["status"],
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=(
                datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
            ),
            metrics_summary=(json.loads(row["metrics_summary"]) if row["metrics_summary"] else {}),
            parameters=json.loads(row["parameters"]) if row["parameters"] else {},
            error_message=row["error_message"],
            python_version=row["python_version"] or "",
            os_info=row["os_info"] or "",
            victor_version=row["victor_version"] or "",
            dependencies=json.loads(row["dependencies"]) if row["dependencies"] else {},
            provider=row["provider"] or "",
            model=row["model"] or "",
            task_type=row["task_type"] or "",
            artifact_count=row["artifact_count"] or 0,
            artifact_size_bytes=row["artifact_size_bytes"] or 0,
        )

    def _row_to_artifact(self, row: sqlite3.Row) -> Artifact:
        """Convert database row to Artifact entity."""
        from victor.experiments.entities import ArtifactType

        return Artifact(
            artifact_id=row["artifact_id"],
            run_id=row["run_id"],
            artifact_type=ArtifactType(row["artifact_type"]),
            filename=row["filename"],
            file_path=row["file_path"],
            file_size_bytes=row["file_size_bytes"],
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )


# Singleton instance for default storage
_default_storage: Optional[SQLiteStorage] = None


def get_default_storage() -> SQLiteStorage:
    """Get the default SQLite storage instance.

    Returns:
        Shared SQLiteStorage instance
    """
    global _default_storage
    if _default_storage is None:
        _default_storage = SQLiteStorage()
    return _default_storage
