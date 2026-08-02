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

"""Meta-tests guarding global-database isolation for the unit-test suite.

Production ``~/.victor/victor.db`` was found polluted with unit-test RL
fixtures (``failing_tool``, ``123invalid``, ``nonexistent_tool``,
``timed_tool`` — 12,035 ``rl_tool_outcome`` rows): tests that exercised the
RL store wrote through ``victor.core.database.get_database()`` straight into
the developer's real global database.

The session-scoped autouse fixture ``isolate_global_victor_db``
(tests/unit/conftest.py) sandboxes ``$HOME`` so ``DatabaseManager`` resolves
its ``Path.home()/.victor/victor.db`` path inside a throwaway temp dir. These
tests are the regression net for that fixture: if it is ever removed, renamed,
or stops taking effect, they fail loudly BEFORE the suite silently resumes
polluting the real database.

Ordering note: the pure path checks run first and import nothing from victor,
so running this file without conftest fixtures (the "red" state) fails on the
path assertions without ever opening — let alone writing — the real database.
"""

import os
import pwd
from pathlib import Path


def _real_home() -> Path:
    """The user's real home from the passwd database (immune to $HOME patching)."""
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


class TestGlobalDbPathIsSandboxed:
    """Pure path checks — no victor imports, no database instantiation."""

    def test_home_is_redirected_away_from_real_home(self):
        """$HOME must point at a sandbox, not the developer's real home."""
        assert Path.home() != _real_home(), (
            "Path.home() resolves to the REAL home directory — the "
            "isolate_global_victor_db autouse fixture (tests/unit/conftest.py) "
            "is missing or inactive, so any global-DB write would hit the real "
            "~/.victor/victor.db."
        )

    def test_resolved_global_db_path_is_not_the_real_victor_db(self):
        """The path DatabaseManager would resolve must not live under real $HOME."""
        # Mirrors DatabaseManager.__init__: Path.home() / ".victor" / "victor.db"
        resolved = Path.home() / ".victor" / "victor.db"
        real_db = _real_home() / ".victor" / "victor.db"
        assert resolved != real_db, (
            f"Global database path resolves to the production DB ({real_db}); "
            "unit tests would write RL fixture rows into it."
        )
        assert not str(resolved).startswith(
            str(_real_home()) + os.sep
        ), f"Global database path {resolved} is under the real home directory."


class TestGlobalDbWriteStaysInSandbox:
    """End-to-end: an RL-table write through get_database() must not touch
    the real ~/.victor/victor.db (same size + mtime before/after)."""

    def test_rl_outcome_write_does_not_touch_real_victor_db(self):
        real_db = _real_home() / ".victor" / "victor.db"
        before = None
        if real_db.exists():
            stat = real_db.stat()
            before = (stat.st_size, stat.st_mtime_ns)

        from victor.core.database import get_database
        from victor.core.schema import Schema

        db = get_database()
        db_path = Path(db.db_path)

        # The live manager must point inside the sandboxed home.
        assert str(db_path).startswith(str(Path.home()) + os.sep), (
            f"get_database() resolved {db_path}, which is outside the "
            f"sandboxed home {Path.home()} — global-DB isolation is broken."
        )
        assert not str(db_path).startswith(
            str(_real_home()) + os.sep
        ), f"get_database() resolved the REAL global database at {db_path}."

        # Write a marker row into the unified RL outcome table (the table the
        # historical pollution accumulated in via the RL store).
        db.execute(Schema.RL_OUTCOME)  # CREATE TABLE IF NOT EXISTS
        db.execute(
            "INSERT INTO rl_outcome (learner_id, provider, model, task_type, success) "
            "VALUES (?, ?, ?, ?, ?)",
            ("test_isolation_marker", "test", "test", "isolation_meta_test", 1),
        )
        rows = db.query(
            "SELECT COUNT(*) AS n FROM rl_outcome WHERE learner_id = ?",
            ("test_isolation_marker",),
        )
        assert rows[0]["n"] >= 1, "marker insert did not land in the sandboxed DB"

        # The real production database must be byte-for-byte untouched.
        if before is not None:
            stat = real_db.stat()
            assert (stat.st_size, stat.st_mtime_ns) == before, (
                "The real ~/.victor/victor.db changed during an isolated "
                "unit-test RL write — isolation is leaking."
            )
        else:
            assert not real_db.exists(), (
                "The real ~/.victor/victor.db was CREATED by an isolated "
                "unit-test RL write — isolation is leaking."
            )
