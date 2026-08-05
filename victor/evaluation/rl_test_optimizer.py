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

"""RL-based test path optimization for efficient test execution.

Uses reinforcement learning to:
1. Prioritize tests most likely to fail based on code changes
2. Optimize test execution order for faster feedback
3. Learn test dependencies for intelligent test selection
4. Skip redundant tests based on historical patterns

Example usage:
    from victor.evaluation.rl_test_optimizer import (
        RLTestOptimizer,
        TestPrioritizer,
        TestDependencyLearner,
    )

    optimizer = RLTestOptimizer()

    # Get prioritized test order for changed files
    changed_files = ["src/module.py", "src/utils.py"]
    prioritized_tests = optimizer.prioritize_tests(
        test_files=["tests/test_module.py", "tests/test_utils.py"],
        changed_files=changed_files,
    )

    # Run tests in optimized order
    for test in prioritized_tests:
        result = run_test(test)
        optimizer.record_outcome(test, result.passed, result.duration_ms)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TestMetadata:
    """Metadata about a test for RL-based optimization."""

    test_name: str
    test_file: str
    dependencies: List[str] = field(default_factory=list)
    avg_duration_ms: float = 0.0
    failure_rate: float = 0.0
    last_run_passed: bool = True
    run_count: int = 0
    flakiness_score: float = 0.0  # 0-1, higher = more flaky


@dataclass
class TestPriorityResult:
    """Result of test prioritization with explanation."""

    test_name: str
    priority_score: float
    reason: str
    estimated_duration_ms: float
    failure_probability: float


class TestDependencyLearner:
    """Learns test dependencies from execution patterns.

    Uses Q-learning to learn which tests tend to fail together,
    enabling intelligent test selection based on changed files.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
    ):
        """Initialize dependency learner.

        Args:
            db_path: Path to SQLite database for persistence
            learning_rate: Q-learning alpha parameter
            discount_factor: Q-learning gamma parameter
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # SQLite storage
        if db_path is None:
            db_path = Path.home() / ".victor" / "rl_data" / "test_optimization.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._ensure_tables()

        # In-memory Q-table for fast access
        # Key: (file_hash, test_name), Value: Q-value for failure probability
        self._q_table: Dict[Tuple[str, str], float] = {}
        self._load_q_table()

    def _ensure_tables(self) -> None:
        """Create database tables if they don't exist."""
        cursor = self.db.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS test_dependencies (
                file_hash TEXT,
                test_name TEXT,
                q_value REAL DEFAULT 0.0,
                sample_count INTEGER DEFAULT 0,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (file_hash, test_name)
            );

            CREATE TABLE IF NOT EXISTS test_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                passed INTEGER NOT NULL,
                duration_ms REAL,
                changed_files TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS test_metadata (
                test_name TEXT PRIMARY KEY,
                test_file TEXT,
                avg_duration_ms REAL DEFAULT 0.0,
                failure_rate REAL DEFAULT 0.0,
                run_count INTEGER DEFAULT 0,
                flakiness_score REAL DEFAULT 0.0
            );

            CREATE INDEX IF NOT EXISTS idx_test_history_name ON test_history(test_name);
            CREATE INDEX IF NOT EXISTS idx_test_history_time ON test_history(timestamp);
        """)
        self.db.commit()

    def _load_q_table(self) -> None:
        """Load Q-table from database."""
        cursor = self.db.cursor()
        cursor.execute("SELECT file_hash, test_name, q_value FROM test_dependencies")
        for row in cursor.fetchall():
            self._q_table[(row["file_hash"], row["test_name"])] = row["q_value"]
        logger.debug(f"Loaded {len(self._q_table)} Q-values from database")

    def _file_hash(self, file_path: str) -> str:
        """Create a hash for a file path."""
        return hashlib.md5(file_path.encode()).hexdigest()[:16]

    def get_failure_probability(self, test_name: str, changed_files: List[str]) -> float:
        """Get estimated failure probability for a test given changed files.

        Args:
            test_name: Name of the test
            changed_files: List of changed file paths

        Returns:
            Estimated failure probability (0-1)
        """
        if not changed_files:
            return 0.0

        # Aggregate Q-values across all changed files
        total_q = 0.0
        count = 0
        for file_path in changed_files:
            file_hash = self._file_hash(file_path)
            key = (file_hash, test_name)
            if key in self._q_table:
                total_q += self._q_table[key]
                count += 1

        if count == 0:
            return 0.1  # Default probability for unknown tests

        # Convert Q-value to probability (sigmoid)
        avg_q = total_q / count
        return 1 / (1 + 2.718 ** (-avg_q))

    def record_outcome(
        self,
        test_name: str,
        passed: bool,
        duration_ms: float,
        changed_files: List[str],
    ) -> None:
        """Record test outcome for learning.

        Args:
            test_name: Name of the test
            passed: Whether the test passed
            duration_ms: Test duration in milliseconds
            changed_files: Files that were changed before this test run
        """
        cursor = self.db.cursor()

        # Record in history
        cursor.execute(
            """
            INSERT INTO test_history (test_name, passed, duration_ms, changed_files)
            VALUES (?, ?, ?, ?)
            """,
            (test_name, 1 if passed else 0, duration_ms, json.dumps(changed_files)),
        )

        # Update Q-values for file->test relationships
        reward = -1.0 if not passed else 0.1  # Negative reward for failures
        for file_path in changed_files:
            file_hash = self._file_hash(file_path)
            key = (file_hash, test_name)

            # Q-learning update
            old_q = self._q_table.get(key, 0.0)
            new_q = old_q + self.learning_rate * (reward - old_q)
            self._q_table[key] = new_q

            # Persist to database
            cursor.execute(
                """
                INSERT INTO test_dependencies (file_hash, test_name, q_value, sample_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(file_hash, test_name) DO UPDATE SET
                    q_value = ?,
                    sample_count = sample_count + 1,
                    last_updated = CURRENT_TIMESTAMP
                """,
                (file_hash, test_name, new_q, new_q),
            )

        # Update test metadata
        self._update_test_metadata(cursor, test_name, passed, duration_ms)
        self.db.commit()

    def _update_test_metadata(
        self, cursor: sqlite3.Cursor, test_name: str, passed: bool, duration_ms: float
    ) -> None:
        """Update aggregated test metadata."""
        # Get existing metadata
        cursor.execute(
            "SELECT avg_duration_ms, failure_rate, run_count FROM test_metadata WHERE test_name = ?",
            (test_name,),
        )
        row = cursor.fetchone()

        if row:
            # Update running averages
            old_avg_duration = row["avg_duration_ms"]
            old_failure_rate = row["failure_rate"]
            run_count = row["run_count"] + 1

            new_avg_duration = (old_avg_duration * (run_count - 1) + duration_ms) / run_count
            new_failure_rate = (
                old_failure_rate * (run_count - 1) + (0 if passed else 1)
            ) / run_count

            cursor.execute(
                """
                UPDATE test_metadata SET
                    avg_duration_ms = ?,
                    failure_rate = ?,
                    run_count = ?
                WHERE test_name = ?
                """,
                (new_avg_duration, new_failure_rate, run_count, test_name),
            )
        else:
            # Insert new metadata
            cursor.execute(
                """
                INSERT INTO test_metadata (test_name, avg_duration_ms, failure_rate, run_count)
                VALUES (?, ?, ?, 1)
                """,
                (test_name, duration_ms, 0.0 if passed else 1.0),
            )

    def get_test_metadata(self, test_name: str) -> Optional[TestMetadata]:
        """Get metadata for a test."""
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT test_name, test_file, avg_duration_ms, failure_rate, run_count, flakiness_score
            FROM test_metadata WHERE test_name = ?
            """,
            (test_name,),
        )
        row = cursor.fetchone()
        if row:
            return TestMetadata(
                test_name=row["test_name"],
                test_file=row["test_file"] or "",
                avg_duration_ms=row["avg_duration_ms"],
                failure_rate=row["failure_rate"],
                run_count=row["run_count"],
                flakiness_score=row["flakiness_score"],
            )
        return None


class TestPrioritizer:
    """Prioritizes tests based on multiple factors using RL.

    Combines:
    - Failure probability (from dependency learner)
    - Historical failure rate
    - Test duration (faster tests first for quick feedback)
    - Flakiness (stable tests preferred)
    """

    def __init__(
        self,
        dependency_learner: Optional[TestDependencyLearner] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize prioritizer.

        Args:
            dependency_learner: Learner for test dependencies
            weights: Weights for different factors (failure_prob, duration, flakiness)
        """
        self.learner = dependency_learner or TestDependencyLearner()
        self.weights = weights or {
            "failure_probability": 0.5,  # Most important: likely to fail
            "historical_failure": 0.2,  # Past failures matter
            "duration": 0.2,  # Prefer fast tests for quick feedback
            "flakiness": 0.1,  # Slightly penalize flaky tests
        }

    def prioritize(
        self,
        test_names: List[str],
        changed_files: Optional[List[str]] = None,
        max_tests: Optional[int] = None,
    ) -> List[TestPriorityResult]:
        """Prioritize tests based on changed files.

        Args:
            test_names: List of test names to prioritize
            changed_files: Files that changed (for dependency-based prioritization)
            max_tests: Maximum number of tests to return (None = all)

        Returns:
            Prioritized list of TestPriorityResult
        """
        changed_files = changed_files or []
        results = []

        for test_name in test_names:
            # Get failure probability from RL
            failure_prob = self.learner.get_failure_probability(test_name, changed_files)

            # Get test metadata
            metadata = self.learner.get_test_metadata(test_name)

            if metadata:
                historical_failure = metadata.failure_rate
                duration_score = 1.0 / (1 + metadata.avg_duration_ms / 1000)  # Normalize
                flakiness = metadata.flakiness_score
                estimated_duration = metadata.avg_duration_ms
            else:
                historical_failure = 0.0
                duration_score = 0.5
                flakiness = 0.0
                estimated_duration = 1000.0  # Default 1 second

            # Calculate priority score (higher = run first)
            priority = (
                self.weights["failure_probability"] * failure_prob
                + self.weights["historical_failure"] * historical_failure
                + self.weights["duration"] * duration_score
                - self.weights["flakiness"] * flakiness  # Subtract flakiness
            )

            # Generate reason
            reasons = []
            if failure_prob > 0.5:
                reasons.append(f"high failure prob ({failure_prob:.0%})")
            if historical_failure > 0.3:
                reasons.append(f"historical failures ({historical_failure:.0%})")
            if metadata and metadata.avg_duration_ms < 500:
                reasons.append("fast test")

            reason = ", ".join(reasons) if reasons else "standard priority"

            results.append(
                TestPriorityResult(
                    test_name=test_name,
                    priority_score=priority,
                    reason=reason,
                    estimated_duration_ms=estimated_duration,
                    failure_probability=failure_prob,
                )
            )

        # Sort by priority (highest first)
        results.sort(key=lambda x: x.priority_score, reverse=True)

        if max_tests:
            results = results[:max_tests]

        return results


class RLTestOptimizer:
    """Main interface for RL-based test optimization.

    Combines dependency learning and prioritization to provide:
    1. Optimized test execution order
    2. Test selection based on changed files
    3. Continuous learning from test outcomes
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        enable_learning: bool = True,
    ):
        """Initialize optimizer.

        Args:
            db_path: Path to database for persistence
            enable_learning: Whether to learn from test outcomes
        """
        self.learner = TestDependencyLearner(db_path=db_path)
        self.prioritizer = TestPrioritizer(dependency_learner=self.learner)
        self.enable_learning = enable_learning
        self._current_changed_files: List[str] = []

    def set_changed_files(self, changed_files: List[str]) -> None:
        """Set the files that changed for this test run.

        Args:
            changed_files: List of changed file paths
        """
        self._current_changed_files = changed_files

    def prioritize_tests(
        self,
        test_files: Optional[List[str]] = None,
        test_names: Optional[List[str]] = None,
        changed_files: Optional[List[str]] = None,
        max_tests: Optional[int] = None,
    ) -> List[TestPriorityResult]:
        """Get prioritized test execution order.

        Args:
            test_files: List of test file paths
            test_names: List of specific test names
            changed_files: Files that changed (uses set_changed_files if not provided)
            max_tests: Maximum tests to return

        Returns:
            Prioritized list of tests with explanations
        """
        changed = changed_files or self._current_changed_files

        # Convert test files to test names if needed
        if test_files and not test_names:
            test_names = [Path(f).stem for f in test_files]

        if not test_names:
            return []

        return self.prioritizer.prioritize(
            test_names=test_names,
            changed_files=changed,
            max_tests=max_tests,
        )

    def record_outcome(
        self,
        test_name: str,
        passed: bool,
        duration_ms: float = 0.0,
        changed_files: Optional[List[str]] = None,
    ) -> None:
        """Record test outcome for learning.

        Args:
            test_name: Name of the test
            passed: Whether the test passed
            duration_ms: Test duration
            changed_files: Files that were changed (uses set_changed_files if not provided)
        """
        if not self.enable_learning:
            return

        changed = changed_files or self._current_changed_files
        self.learner.record_outcome(test_name, passed, duration_ms, changed)

    def get_test_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        cursor = self.learner.db.cursor()

        cursor.execute("SELECT COUNT(*) FROM test_history")
        total_runs = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT test_name) FROM test_metadata")
        unique_tests = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(failure_rate) FROM test_metadata")
        avg_failure_rate = cursor.fetchone()[0] or 0.0

        cursor.execute("SELECT SUM(sample_count) FROM test_dependencies")
        dependency_samples = cursor.fetchone()[0] or 0

        return {
            "total_test_runs": total_runs,
            "unique_tests_tracked": unique_tests,
            "avg_failure_rate": avg_failure_rate,
            "dependency_samples": dependency_samples,
            "q_table_size": len(self.learner._q_table),
        }

    def suggest_tests_for_changes(
        self,
        changed_files: List[str],
        all_tests: List[str],
        threshold: float = 0.3,
    ) -> List[str]:
        """Suggest which tests to run based on changed files.

        Uses RL-learned dependencies to identify tests that are likely
        to be affected by the changes.

        Args:
            changed_files: Files that changed
            all_tests: All available test names
            threshold: Minimum failure probability to include test

        Returns:
            List of suggested test names
        """
        suggested = []
        for test_name in all_tests:
            prob = self.learner.get_failure_probability(test_name, changed_files)
            if prob >= threshold:
                suggested.append(test_name)

        # Always include tests with unknown dependencies (new tests)
        for test_name in all_tests:
            if test_name not in suggested:
                # Check if we have any data for this test
                metadata = self.learner.get_test_metadata(test_name)
                if not metadata or metadata.run_count < 5:
                    suggested.append(test_name)

        return suggested


# Factory function
def get_rl_test_optimizer() -> RLTestOptimizer:
    """Get a global RL test optimizer instance."""
    return RLTestOptimizer()
