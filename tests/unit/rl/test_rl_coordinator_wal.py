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

"""Tests verifying SQLite WAL mode is enabled for the unified database.

WAL (Write-Ahead Logging) mode enables concurrent readers with a single
writer, which is critical for RLCoordinator under team concurrency.
"""

import tempfile
from pathlib import Path

import pytest

from victor.core.database import DatabaseManager, reset_database, get_database


@pytest.fixture
def wal_db():
    """Create a fresh DatabaseManager with a temp database."""
    reset_database()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_wal.db"
        db = get_database(db_path)
        yield db
        reset_database()


class TestWALMode:
    """Tests for WAL mode configuration in the unified database."""

    def test_wal_mode_enabled(self, wal_db: DatabaseManager) -> None:
        """Verify PRAGMA journal_mode returns 'wal'."""
        conn = wal_db.get_connection()
        result = conn.execute("PRAGMA journal_mode").fetchone()
        assert result[0] == "wal", f"Expected WAL mode, got {result[0]}"

    def test_synchronous_normal(self, wal_db: DatabaseManager) -> None:
        """Verify PRAGMA synchronous returns NORMAL (1)."""
        conn = wal_db.get_connection()
        result = conn.execute("PRAGMA synchronous").fetchone()
        # NORMAL = 1
        assert result[0] == 1, f"Expected synchronous=NORMAL (1), got {result[0]}"
