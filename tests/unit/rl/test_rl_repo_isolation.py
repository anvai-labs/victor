# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for RL per-repo isolation and test fixture filtering."""

import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest

from victor.framework.rl.coordinator import RLCoordinator
from victor.framework.rl.base import RLOutcome
from victor.core.database import reset_database
from victor.core.schema import Tables


@pytest.fixture
def coordinator(tmp_path) -> Generator[RLCoordinator, None, None]:
    """Create isolated RLCoordinator with temp database."""
    db_path = tmp_path / "test.db"

    with patch("victor.framework.rl.coordinator.get_database") as mock_db:
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        mock_manager = type(
            "MockDB",
            (),
            {
                "get_connection": lambda self: conn,
                "db_path": db_path,
            },
        )()
        mock_db.return_value = mock_manager

        coord = RLCoordinator(storage_path=tmp_path)
        yield coord
        conn.close()


def _make_outcome(tool_name: str = "read", success: bool = True) -> RLOutcome:
    return RLOutcome(
        provider="test",
        model="test-model",
        task_type="bug_fix",
        success=success,
        quality_score=0.8 if success else 0.2,
        metadata={"tool_name": tool_name},
    )


class TestRepoIsolation:
    """Tests for per-repo RL outcome tagging."""

    def test_set_repo_context(self, coordinator):
        """set_repo_context sets internal _repo_id."""
        assert coordinator._repo_id is None
        coordinator.set_repo_context("django__django")
        assert coordinator._repo_id == "django__django"

    def test_clear_repo_context(self, coordinator):
        """set_repo_context(None) clears repo_id."""
        coordinator.set_repo_context("astropy")
        coordinator.set_repo_context(None)
        assert coordinator._repo_id is None

    def test_repo_id_in_outcome(self, coordinator):
        """Recorded outcomes include repo_id when context is set."""
        coordinator.set_repo_context("django__django")

        # Register a dummy learner that accepts anything
        from unittest.mock import MagicMock

        mock_learner = MagicMock()
        coordinator._learners["tool_selector"] = mock_learner

        outcome = _make_outcome()
        coordinator.record_outcome("tool_selector", outcome)

        # Check database
        cursor = coordinator.db.cursor()
        cursor.execute(f"SELECT repo_id FROM {Tables.RL_OUTCOME}")
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "django__django"

    def test_no_repo_id_when_global(self, coordinator):
        """Outcomes without repo context have NULL repo_id."""
        from unittest.mock import MagicMock

        mock_learner = MagicMock()
        coordinator._learners["tool_selector"] = mock_learner

        outcome = _make_outcome()
        coordinator.record_outcome("tool_selector", outcome)

        cursor = coordinator.db.cursor()
        cursor.execute(f"SELECT repo_id FROM {Tables.RL_OUTCOME}")
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][0] is None


class TestFixtureFiltering:
    """Tests for test fixture tool filtering."""

    def test_filters_dummy_tools(self, coordinator):
        """Tools prefixed with dummy_ are filtered from recording."""
        from unittest.mock import MagicMock

        mock_learner = MagicMock()
        coordinator._learners["tool_selector"] = mock_learner

        outcome = _make_outcome(tool_name="dummy_tool")
        coordinator.record_outcome("tool_selector", outcome)

        # Should not be recorded
        mock_learner.record_outcome.assert_not_called()

    def test_filters_test_tools(self, coordinator):
        """Tools prefixed with test_ are filtered."""
        from unittest.mock import MagicMock

        mock_learner = MagicMock()
        coordinator._learners["tool_selector"] = mock_learner

        outcome = _make_outcome(tool_name="test_helper")
        coordinator.record_outcome("tool_selector", outcome)

        mock_learner.record_outcome.assert_not_called()

    def test_allows_real_tools(self, coordinator):
        """Normal tools are not filtered."""
        from unittest.mock import MagicMock

        mock_learner = MagicMock()
        coordinator._learners["tool_selector"] = mock_learner

        outcome = _make_outcome(tool_name="read")
        coordinator.record_outcome("tool_selector", outcome)

        mock_learner.record_outcome.assert_called_once()

    def test_should_record_tool_checks(self, coordinator):
        """Verify _should_record_tool for various prefixes."""
        assert coordinator._should_record_tool("read") is True
        assert coordinator._should_record_tool("edit") is True
        assert coordinator._should_record_tool("code_search") is True
        assert coordinator._should_record_tool("dummy_tool") is False
        assert coordinator._should_record_tool("test_helper") is False
        assert coordinator._should_record_tool("mock_service") is False
        assert coordinator._should_record_tool("flaky_test") is False
        assert coordinator._should_record_tool("error_handler") is False
        assert coordinator._should_record_tool("always_fail") is False


class TestMigration:
    """Tests for repo_id schema migration."""

    def test_migrate_adds_repo_id_column(self, coordinator):
        """Migration adds repo_id column to existing table."""
        cursor = coordinator.db.cursor()
        cursor.execute(f"PRAGMA table_info({Tables.RL_OUTCOME})")
        columns = {row[1] for row in cursor.fetchall()}
        assert "repo_id" in columns

    def test_migration_is_idempotent(self, coordinator):
        """Running migration twice doesn't error."""
        # Migration ran in __init__, run again
        coordinator._migrate_add_repo_id()
        # Should not raise
