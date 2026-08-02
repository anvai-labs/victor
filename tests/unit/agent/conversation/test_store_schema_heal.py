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

"""ConversationStore schema self-heal for legacy `sessions` tables."""

import sqlite3

from victor.agent.conversation.store_schema import ConversationStoreSchema


def test_migrate_sessions_table_heals_all_missing_core_columns(tmp_path):
    """A legacy sessions table missing core columns is healed in one pass.

    Regression: an earlier fix added only session_id, so init still failed on the
    next missing column (last_activity). The heal must add the whole core set.
    """
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    # Ancient table — missing session_id, last_activity, and most core columns.
    conn.execute("CREATE TABLE sessions (id TEXT, created_at TIMESTAMP)")
    conn.commit()

    schema = ConversationStoreSchema.__new__(ConversationStoreSchema)
    schema.migrate_sessions_table(conn)

    cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    for required in (
        "session_id",
        "last_activity",
        "project_path",
        "model",
        "profile",
        "max_tokens",
        "reserved_tokens",
        "metadata",
    ):
        assert required in cols, f"missing healed column: {required}"

    # An INSERT OR REPLACE using the core columns now succeeds (no "no such column").
    conn.execute(
        "INSERT OR REPLACE INTO sessions (session_id, created_at, last_activity) "
        "VALUES (?, ?, ?)",
        ("s1", "2026-01-01", "2026-01-01"),
    )
    conn.commit()
    conn.close()
