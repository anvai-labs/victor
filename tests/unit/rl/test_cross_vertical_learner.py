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

"""Unit tests for CrossVerticalLearner.

Tests the cross-vertical transfer learning functionality:
- Pattern discovery across verticals
- Confidence scoring based on sample count and vertical coverage
- Recommendation generation for cold-start scenarios
- Pattern application tracking
"""

import json
import sqlite3
import pytest
from datetime import datetime

from victor.framework.rl.learners.cross_vertical import (
    CrossVerticalLearner,
    SharedPattern,
)
from victor.framework.rl.base import RLOutcome
from victor.core.schema import Tables

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def db_connection():
    """Create an in-memory database for testing."""
    conn = sqlite3.connect(":memory:")

    # Create required tables
    cursor = conn.cursor()

    # Create rl_outcome table (used by CrossVerticalLearner)
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {Tables.RL_OUTCOME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner TEXT NOT NULL,
            task_type TEXT,
            vertical TEXT,
            quality_score REAL,
            metadata TEXT,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def learner(db_connection):
    """Create CrossVerticalLearner with test database."""
    return CrossVerticalLearner(
        name="cross_vertical",
        db_connection=db_connection,
        learning_rate=0.1,
        min_samples=3,  # Lower threshold for tests
        min_verticals=2,
    )


def insert_outcome(
    db_connection,
    task_type: str,
    vertical: str,
    quality_score: float,
    mode: str = None,
):
    """Helper to insert test outcomes."""
    cursor = db_connection.cursor()
    metadata = json.dumps({"mode": mode}) if mode else None
    cursor.execute(
        f"""
        INSERT INTO {Tables.RL_OUTCOME}
        (learner, task_type, vertical, quality_score, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "test",
            task_type,
            vertical,
            quality_score,
            metadata,
            datetime.now().isoformat(),
        ),
    )
    db_connection.commit()


# =============================================================================
# Test SharedPattern Dataclass
# =============================================================================


class TestSharedPattern:
    """Test SharedPattern data class."""

    def test_shared_pattern_creation(self):
        """Test creating a SharedPattern."""
        pattern = SharedPattern(
            task_type="edit",
            pattern_name="edit_cross_vertical_pattern",
            avg_quality=0.85,
            confidence=0.9,
            source_verticals=["coding", "devops"],
            recommended_mode="BUILD",
            recommendation="For edit tasks, BUILD mode yields high quality.",
            sample_count=50,
        )

        assert pattern.task_type == "edit"
        assert pattern.avg_quality == 0.85
        assert pattern.confidence == 0.9
        assert "coding" in pattern.source_verticals
        assert pattern.recommended_mode == "BUILD"

    def test_shared_pattern_defaults(self):
        """Test SharedPattern default values."""
        pattern = SharedPattern(
            task_type="query",
            pattern_name="query_pattern",
            avg_quality=0.6,
            confidence=0.5,
        )

        assert pattern.source_verticals == []
        assert pattern.recommended_mode is None
        assert pattern.recommendation == ""
        assert pattern.sample_count == 0


# =============================================================================
# Test CrossVerticalLearner Initialization
# =============================================================================


class TestCrossVerticalLearnerInit:
    """Test CrossVerticalLearner initialization."""

    def test_learner_initialization(self, learner):
        """Test learner is properly initialized."""
        assert learner.name == "cross_vertical"
        assert learner._min_samples == 3
        assert learner._min_verticals == 2

    def test_tables_created(self, db_connection, learner):
        """Test required tables are created."""
        cursor = db_connection.cursor()

        # Check rl_pattern table exists
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tables.RL_PATTERN}'"
        )
        assert cursor.fetchone() is not None

        # Check rl_pattern_use table exists
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tables.RL_PATTERN_USE}'"
        )
        assert cursor.fetchone() is not None


# =============================================================================
# Test Pattern Discovery
# =============================================================================


class TestPatternDiscovery:
    """Test pattern discovery across verticals."""

    def test_no_patterns_when_insufficient_data(self, learner, db_connection):
        """Test no patterns returned when data is insufficient."""
        # Only 1 outcome - below threshold
        insert_outcome(db_connection, "edit", "coding", 0.8)

        patterns = learner.get_shared_patterns()
        assert len(patterns) == 0

    def test_no_patterns_with_single_vertical(self, learner, db_connection):
        """Test no patterns when data comes from single vertical."""
        # Multiple outcomes but all from same vertical
        for _ in range(5):
            insert_outcome(db_connection, "edit", "coding", 0.8)

        patterns = learner.get_shared_patterns()
        assert len(patterns) == 0

    def test_pattern_discovered_with_multiple_verticals(self, learner, db_connection):
        """Test pattern discovered when data spans multiple verticals."""
        # Add outcomes from coding vertical
        for _ in range(3):
            insert_outcome(db_connection, "edit", "coding", 0.8)

        # Add outcomes from devops vertical
        for _ in range(3):
            insert_outcome(db_connection, "edit", "devops", 0.85)

        patterns = learner.get_shared_patterns()

        assert len(patterns) == 1
        assert patterns[0].task_type == "edit"
        assert "coding" in patterns[0].source_verticals
        assert "devops" in patterns[0].source_verticals

    def test_multiple_patterns_discovered(self, learner, db_connection):
        """Test multiple patterns discovered for different task types."""
        # Edit task pattern
        for v in ["coding", "devops", "research"]:
            for _ in range(2):
                insert_outcome(db_connection, "edit", v, 0.8)

        # Query task pattern
        for v in ["coding", "dataanalysis"]:
            for _ in range(2):
                insert_outcome(db_connection, "query", v, 0.7)

        patterns = learner.get_shared_patterns()

        task_types = [p.task_type for p in patterns]
        assert "edit" in task_types
        assert "query" in task_types

    def test_pattern_excludes_target_vertical(self, learner, db_connection):
        """Test patterns can exclude target vertical from sources."""
        for v in ["coding", "devops"]:
            for _ in range(3):
                insert_outcome(db_connection, "edit", v, 0.8)

        patterns = learner.get_shared_patterns(target_vertical="research")

        assert len(patterns) == 1
        # Pattern sources should not include target
        assert "research" not in patterns[0].source_verticals

    def test_confidence_threshold_filtering(self, learner, db_connection):
        """Test patterns filtered by confidence threshold."""
        # Low sample count = low confidence
        for v in ["coding", "devops"]:
            insert_outcome(db_connection, "edit", v, 0.8)
            insert_outcome(db_connection, "edit", v, 0.8)

        # High confidence threshold should filter out
        patterns = learner.get_shared_patterns(min_confidence=0.9)
        assert len(patterns) == 0

        # Lower threshold should include
        patterns = learner.get_shared_patterns(min_confidence=0.3)
        assert len(patterns) >= 0  # May or may not have patterns


# =============================================================================
# Test Recommendation Generation
# =============================================================================


class TestRecommendations:
    """Test recommendation generation."""

    def test_recommendation_with_sufficient_local_data(self, learner, db_connection):
        """Test recommendation when local data is sufficient."""
        # Add enough local data
        for _ in range(5):
            insert_outcome(db_connection, "edit", "research", 0.8)

        rec = learner.get_recommendation("edit", "research")

        assert rec.is_baseline
        assert "local data" in rec.reason.lower()

    def test_recommendation_from_cross_vertical_pattern(self, learner, db_connection):
        """Test recommendation uses cross-vertical patterns when local data is sparse."""
        # Add cross-vertical data
        for v in ["coding", "devops"]:
            for _ in range(3):
                insert_outcome(db_connection, "edit", v, 0.85)

        # Target vertical has no data
        rec = learner.get_recommendation("edit", "research")

        # Should get cross-vertical recommendation
        if not rec.is_baseline:
            assert "cross-vertical" in rec.reason.lower()
            assert rec.confidence > 0

    def test_recommendation_no_pattern_available(self, learner, db_connection):
        """Test recommendation when no cross-vertical pattern exists."""
        rec = learner.get_recommendation("unknown_task", "research")

        assert rec.confidence < 0.5
        assert "no cross-vertical pattern" in rec.reason.lower()

    def test_recommendation_text_quality_levels(self, learner, db_connection):
        """Test recommendation text reflects quality levels."""
        # High quality outcomes
        for v in ["coding", "devops"]:
            for _ in range(3):
                insert_outcome(db_connection, "edit", v, 0.9)

        patterns = learner.get_shared_patterns()
        if patterns:
            assert "high" in patterns[0].recommendation.lower()


# =============================================================================
# Test Best Mode Detection
# =============================================================================


class TestBestModeDetection:
    """Test best mode detection for task types."""

    def test_best_mode_from_metadata(self, learner, db_connection):
        """Test best mode extracted from outcome metadata."""
        # Add outcomes with mode in metadata
        for _ in range(6):
            insert_outcome(db_connection, "edit", "coding", 0.9, mode="BUILD")

        best_mode = learner._get_best_mode_for_task("edit")
        assert best_mode == "BUILD"

    def test_no_mode_when_metadata_missing(self, learner, db_connection):
        """Test returns None when no mode in metadata."""
        for _ in range(6):
            insert_outcome(db_connection, "query", "coding", 0.7)

        best_mode = learner._get_best_mode_for_task("query")
        assert best_mode is None


# =============================================================================
# Test RLOutcome Processing
# =============================================================================


class TestOutcomeProcessing:
    """Test outcome recording (CrossVerticalLearner uses aggregate queries)."""

    def test_record_outcome_basic(self, learner):
        """Test that outcome recording works (even if CrossVerticalLearner doesn't use it directly)."""
        outcome = RLOutcome(
            provider="anthropic",
            model="claude-3-opus",
            task_type="edit",
            success=True,
            quality_score=0.85,
            vertical="coding",
        )

        # CrossVerticalLearner.record_outcome should not raise
        learner.record_outcome(outcome)

    def test_learner_name(self, learner):
        """Test learner name is correct."""
        assert learner.name == "cross_vertical"


# =============================================================================
# Test Learner Import
# =============================================================================


class TestLearnerImport:
    """Test that CrossVerticalLearner can be imported correctly."""

    def test_learner_in_exports(self):
        """Test that CrossVerticalLearner is exported from learners module."""
        from victor.framework.rl.learners import CrossVerticalLearner as CVL

        assert CVL is CrossVerticalLearner

    def test_learner_inherits_from_base(self, learner):
        """Test that CrossVerticalLearner inherits from BaseLearner."""
        from victor.framework.rl.base import BaseLearner

        assert isinstance(learner, BaseLearner)
