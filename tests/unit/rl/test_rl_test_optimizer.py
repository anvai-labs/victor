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

"""Tests for RL-based test path optimizer."""

import tempfile
from pathlib import Path

import pytest

from victor.evaluation.rl_test_optimizer import (
    RLTestOptimizer,
    TestDependencyLearner,
    TestMetadata,
    TestPrioritizer,
    TestPriorityResult,
)


class TestTestDependencyLearner:
    """Tests for TestDependencyLearner."""

    @pytest.fixture
    def learner(self) -> TestDependencyLearner:
        """Create a learner with temp database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_optimization.db"
            yield TestDependencyLearner(db_path=db_path)

    def test_init_creates_tables(self, learner: TestDependencyLearner) -> None:
        """Test that initialization creates required tables."""
        cursor = learner.db.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='test_dependencies'"
        )
        assert cursor.fetchone() is not None

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_history'")
        assert cursor.fetchone() is not None

    def test_get_failure_probability_no_data(self, learner: TestDependencyLearner) -> None:
        """Test failure probability with no historical data."""
        prob = learner.get_failure_probability("test_foo", ["src/foo.py"])
        # Default probability for unknown tests
        assert prob == 0.1

    def test_get_failure_probability_no_files(self, learner: TestDependencyLearner) -> None:
        """Test failure probability with no changed files."""
        prob = learner.get_failure_probability("test_foo", [])
        assert prob == 0.0

    def test_record_outcome_creates_history(self, learner: TestDependencyLearner) -> None:
        """Test that recording outcome creates history entry."""
        learner.record_outcome(
            test_name="test_bar",
            passed=True,
            duration_ms=100.0,
            changed_files=["src/bar.py"],
        )

        cursor = learner.db.cursor()
        cursor.execute("SELECT * FROM test_history WHERE test_name = 'test_bar'")
        row = cursor.fetchone()
        assert row is not None
        assert row["passed"] == 1
        assert row["duration_ms"] == 100.0

    def test_record_outcome_updates_q_value(self, learner: TestDependencyLearner) -> None:
        """Test that recording outcome updates Q-value."""
        # Record a failure
        learner.record_outcome(
            test_name="test_baz",
            passed=False,
            duration_ms=50.0,
            changed_files=["src/baz.py"],
        )

        # Q-value should increase (more negative = higher failure probability)
        file_hash = learner._file_hash("src/baz.py")
        key = (file_hash, "test_baz")
        assert key in learner._q_table
        assert learner._q_table[key] < 0  # Negative from failure

    def test_failure_probability_increases_with_failures(
        self, learner: TestDependencyLearner
    ) -> None:
        """Test that failure probability increases with recorded failures."""
        changed_files = ["src/module.py"]

        # Initial probability
        initial_prob = learner.get_failure_probability("test_module", changed_files)

        # Record multiple failures
        for _ in range(5):
            learner.record_outcome(
                test_name="test_module",
                passed=False,
                duration_ms=100.0,
                changed_files=changed_files,
            )

        # Probability should increase
        final_prob = learner.get_failure_probability("test_module", changed_files)
        assert final_prob > initial_prob

    def test_get_test_metadata(self, learner: TestDependencyLearner) -> None:
        """Test retrieving test metadata."""
        # Record some outcomes
        learner.record_outcome("test_meta", True, 100.0, ["file1.py"])
        learner.record_outcome("test_meta", False, 150.0, ["file1.py"])
        learner.record_outcome("test_meta", True, 120.0, ["file1.py"])

        metadata = learner.get_test_metadata("test_meta")
        assert metadata is not None
        assert metadata.test_name == "test_meta"
        assert metadata.run_count == 3
        # One failure out of 3 runs
        assert abs(metadata.failure_rate - 0.333) < 0.1


class TestTestPrioritizer:
    """Tests for TestPrioritizer."""

    @pytest.fixture
    def prioritizer(self) -> TestPrioritizer:
        """Create a prioritizer with temp database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_optimization.db"
            learner = TestDependencyLearner(db_path=db_path)
            yield TestPrioritizer(dependency_learner=learner)

    def test_prioritize_empty_list(self, prioritizer: TestPrioritizer) -> None:
        """Test prioritizing an empty test list."""
        results = prioritizer.prioritize(test_names=[], changed_files=[])
        assert results == []

    def test_prioritize_returns_all_tests(self, prioritizer: TestPrioritizer) -> None:
        """Test that prioritize returns all tests."""
        test_names = ["test_a", "test_b", "test_c"]
        results = prioritizer.prioritize(test_names=test_names)
        assert len(results) == 3
        assert all(isinstance(r, TestPriorityResult) for r in results)

    def test_prioritize_respects_max_tests(self, prioritizer: TestPrioritizer) -> None:
        """Test that max_tests is respected."""
        test_names = ["test_a", "test_b", "test_c", "test_d"]
        results = prioritizer.prioritize(test_names=test_names, max_tests=2)
        assert len(results) == 2

    def test_prioritize_with_historical_failures(self, prioritizer: TestPrioritizer) -> None:
        """Test that tests with historical failures are prioritized."""
        # Record failures for test_fail
        for _ in range(5):
            prioritizer.learner.record_outcome(
                test_name="test_fail",
                passed=False,
                duration_ms=100.0,
                changed_files=["src/fail.py"],
            )

        # Record successes for test_pass
        for _ in range(5):
            prioritizer.learner.record_outcome(
                test_name="test_pass",
                passed=True,
                duration_ms=100.0,
                changed_files=["src/pass.py"],
            )

        results = prioritizer.prioritize(
            test_names=["test_pass", "test_fail"],
            changed_files=["src/fail.py"],
        )

        # test_fail should be prioritized higher
        assert results[0].test_name == "test_fail"


class TestRLTestOptimizer:
    """Tests for RLTestOptimizer."""

    @pytest.fixture
    def optimizer(self) -> RLTestOptimizer:
        """Create an optimizer with temp database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_optimization.db"
            yield RLTestOptimizer(db_path=db_path)

    def test_set_changed_files(self, optimizer: RLTestOptimizer) -> None:
        """Test setting changed files."""
        optimizer.set_changed_files(["file1.py", "file2.py"])
        assert optimizer._current_changed_files == ["file1.py", "file2.py"]

    def test_prioritize_tests_basic(self, optimizer: RLTestOptimizer) -> None:
        """Test basic test prioritization."""
        results = optimizer.prioritize_tests(
            test_names=["test_a", "test_b"],
            changed_files=["src/module.py"],
        )
        assert len(results) == 2

    def test_record_outcome(self, optimizer: RLTestOptimizer) -> None:
        """Test recording test outcome."""
        optimizer.set_changed_files(["src/module.py"])
        optimizer.record_outcome(
            test_name="test_record",
            passed=True,
            duration_ms=150.0,
        )

        # Check that metadata was recorded
        metadata = optimizer.learner.get_test_metadata("test_record")
        assert metadata is not None
        assert metadata.run_count == 1

    def test_get_test_stats(self, optimizer: RLTestOptimizer) -> None:
        """Test getting optimizer statistics."""
        # Record some outcomes
        optimizer.record_outcome("test_1", True, 100.0, ["file.py"])
        optimizer.record_outcome("test_2", False, 200.0, ["file.py"])

        stats = optimizer.get_test_stats()
        assert stats["total_test_runs"] == 2
        assert stats["unique_tests_tracked"] == 2

    def test_suggest_tests_for_changes(self, optimizer: RLTestOptimizer) -> None:
        """Test suggesting tests based on changed files."""
        # Record failures for specific file changes
        for _ in range(10):
            optimizer.record_outcome(
                test_name="test_related",
                passed=False,
                duration_ms=100.0,
                changed_files=["src/critical.py"],
            )

        # Record successes for test_unrelated
        for _ in range(10):
            optimizer.record_outcome(
                test_name="test_unrelated",
                passed=True,
                duration_ms=100.0,
                changed_files=["src/other.py"],
            )

        suggested = optimizer.suggest_tests_for_changes(
            changed_files=["src/critical.py"],
            all_tests=["test_related", "test_unrelated"],
            threshold=0.3,
        )

        # test_related should be suggested due to high failure correlation
        assert "test_related" in suggested

    def test_learning_disabled(self) -> None:
        """Test that learning can be disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_optimization.db"
            optimizer = RLTestOptimizer(db_path=db_path, enable_learning=False)

            optimizer.record_outcome("test_disabled", True, 100.0, ["file.py"])

            # No metadata should be recorded
            metadata = optimizer.learner.get_test_metadata("test_disabled")
            assert metadata is None
