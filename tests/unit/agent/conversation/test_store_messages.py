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

"""Mirror suite for victor.agent.conversation.store_messages.

Bulk transcript persistence: _message_to_row row parity with the
add_message path and _persist_session_messages atomic replace semantics.
"""

import json
import sqlite3

import pytest

from victor.agent.conversation.store import ConversationStore
from victor.agent.conversation.types import MessageRole


@pytest.fixture
def temp_store(tmp_path):
    return ConversationStore(db_path=tmp_path / "test_project.db")


class TestPersistSessionMessages:
    """save_session must persist the transcript, not just the session row."""


class TestSaveSessionPersistsMessages:
    """save_session must persist the transcript, not just the session row.

    Regression guard for the import bug where ``/resume --from-file`` claimed
    "Messages: N" but a fresh process loaded zero messages: save_session used
    to mutate only the in-memory session and never write ``messages`` rows.
    """

    def _save_three(self, store, session_id="s_persist_001"):
        return store.save_session(
            conversation={
                "messages": [
                    {
                        "role": "user",
                        "content": "first question",
                        "timestamp": "2026-11-06T12:00:01",
                    },
                    {
                        "role": "assistant",
                        "content": "first answer",
                        "timestamp": "2026-11-06T12:00:05",
                    },
                    {
                        "role": "user",
                        "content": "second question",
                        "timestamp": "2026-11-06T12:00:09",
                    },
                ]
            },
            model="test-model",
            provider="test-provider",
            session_id=session_id,
            title="persist me",
        )

    def test_save_session_writes_messages_rows(self, temp_store):
        """P0 regression: a fresh store (fresh cache) reloads the transcript."""
        self._save_three(temp_store)

        fresh = ConversationStore(db_path=temp_store.db_path)
        loaded = fresh.load_session("s_persist_001")
        assert loaded is not None
        msgs = loaded["conversation"]["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        assert msgs[0]["content"] == "first question"
        assert msgs[2]["content"] == "second question"

        with sqlite3.connect(temp_store.db_path) as conn:
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = 's_persist_001'"
            ).fetchone()
        assert count == 3

    def test_save_session_replace_semantics(self, temp_store):
        """Re-saving with the same session id replaces, not appends."""
        self._save_three(temp_store)
        temp_store.save_session(
            conversation={"messages": [{"role": "user", "content": "only one"}]},
            model="test-model",
            provider="test-provider",
            session_id="s_persist_001",
        )

        fresh = ConversationStore(db_path=temp_store.db_path)
        msgs = fresh.load_session("s_persist_001")["conversation"]["messages"]
        assert len(msgs) == 1
        assert msgs[0]["content"] == "only one"

    def test_save_session_does_not_write_preview_sidecar_to_messages(self, temp_store):
        """Preview messages live in the sessions.metadata sidecar only."""
        temp_store.save_session(
            conversation={
                "messages": [
                    {"role": "user", "content": "real"},
                    {"role": "assistant", "content": "reply"},
                ],
                "preview_messages": [
                    {
                        "role": "assistant",
                        "content": "preview text",
                        "metadata": {"preview_kind": "reflection"},
                    },
                ],
            },
            model="test-model",
            provider="test-provider",
            session_id="s_preview_001",
        )

        with sqlite3.connect(temp_store.db_path) as conn:
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = 's_preview_001'"
            ).fetchone()
        assert count == 2  # transcript only; preview is in the sidecar

        fresh = ConversationStore(db_path=temp_store.db_path)
        loaded = fresh.load_session("s_preview_001")
        assert len(loaded["conversation"]["messages"]) == 2
        assert len(loaded["conversation"]["preview_messages"]) == 1
        assert loaded["conversation"]["preview_messages"][0]["content"] == "preview text"

    def test_bulk_persist_matches_row_path(self, temp_store):
        """Bulk rows are byte-identical to add_message rows (same builder)."""
        big_tool_output = "x" * (ConversationStore._TOOL_OUTPUT_STORE_LIMIT + 5000)
        shared_metadata = {"agent_id": "ag1", "plan_id": "plan9"}

        temp_store.create_session(session_id="s_row_a", provider="p", model="m")
        tool_calls = [{"id": "call_1", "name": "bash", "arguments": {"cmd": "ls"}}]
        temp_store.add_message(
            "s_row_a",
            MessageRole.TOOL,
            big_tool_output,
            tool_name="bash",
            tool_call_id="call_1",
            metadata=dict(shared_metadata),
        )
        temp_store.add_message(
            "s_row_a",
            MessageRole.ASSISTANT,
            "running it",
            tool_calls=list(tool_calls),
        )

        temp_store.save_session(
            conversation={
                "messages": [
                    {
                        "role": "tool",
                        "content": big_tool_output,
                        "tool_name": "bash",
                        "tool_call_id": "call_1",
                        "metadata": dict(shared_metadata),
                        "token_count": 4242,
                    },
                    {
                        "role": "assistant",
                        "content": "running it",
                        "tool_calls": list(tool_calls),
                    },
                ]
            },
            model="m",
            provider="p",
            session_id="s_row_b",
        )

        with sqlite3.connect(temp_store.db_path) as conn:
            conn.row_factory = sqlite3.Row
            all_rows = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM messages WHERE session_id IN ('s_row_a','s_row_b')"
                )
            ]
        # Row-builder transforms must be identical across paths. metadata is
        # excluded from exact equality: add_message additionally injects trace
        # metadata (source tracking) that imports deliberately do not get.
        by_role = {(row["session_id"], row["role"]): row for row in all_rows}
        a_tool = by_role[("s_row_a", "tool")]
        b_tool = by_role[("s_row_b", "tool")]
        a_asst = by_role[("s_row_a", "assistant")]
        b_asst = by_role[("s_row_b", "assistant")]
        # token_count and priority: add_message derives them (estimate /
        # role-based) when absent, while the dict path carries the caller's
        # values — different input defaulting, not a row-builder divergence.
        ignored = {"id", "session_id", "timestamp", "metadata", "token_count", "priority"}
        for key in set(a_tool) - ignored:
            assert a_tool[key] == b_tool[key], f"column {key} diverged (tool)"
        # explicit token_count survives the dict/bulk path verbatim
        assert b_tool["token_count"] == 4242
        for key in set(a_asst) - ignored:
            assert a_asst[key] == b_asst[key], f"column {key} diverged (assistant)"
        for row in (a_tool, b_tool):
            assert len(row["content"]) <= ConversationStore._TOOL_OUTPUT_STORE_LIMIT + 100
            assert "truncated from" in row["content"]
            assert row["agent_id"] == "ag1" and row["plan_id"] == "plan9"
        # tool_calls merged into metadata JSON identically on both paths
        for row in (a_asst, b_asst):
            merged = json.loads(row["metadata"])
            assert merged.get("tool_calls") == tool_calls

    def test_save_session_updates_fts(self, temp_store):
        """Bulk-written rows are searchable from a fresh store (trigger sync)."""
        self._save_three(temp_store, session_id="s_fts_001")

        fresh = ConversationStore(db_path=temp_store.db_path)
        results = fresh.search_sessions("first question")
        assert any(r["session_id"] == "s_fts_001" for r in results)

    def test_imported_tied_timestamps_order_stable(self, temp_store):
        """Tied timestamps keep insertion order via the rowid tiebreak."""
        temp_store.save_session(
            conversation={
                "messages": [
                    {"role": "user", "content": "alpha", "timestamp": "2026-11-06T12:00:00"},
                    {"role": "assistant", "content": "beta", "timestamp": "2026-11-06T12:00:00"},
                    {"role": "user", "content": "gamma", "timestamp": "2026-11-06T12:00:00"},
                ]
            },
            model="m",
            provider="p",
            session_id="s_ties_001",
        )
        fresh = ConversationStore(db_path=temp_store.db_path)
        msgs = fresh.load_session("s_ties_001")["conversation"]["messages"]
        assert [m["content"] for m in msgs] == ["alpha", "beta", "gamma"]

    def test_message_from_row_restores_tool_calls(self, temp_store):
        """tool_calls survive save → fresh load (popped back out of metadata)."""
        temp_store.save_session(
            conversation={
                "messages": [
                    {
                        "role": "assistant",
                        "content": "running the tool",
                        "tool_calls": [
                            {"id": "call_1", "name": "bash", "arguments": {"cmd": "ls"}}
                        ],
                    }
                ]
            },
            model="m",
            provider="p",
            session_id="s_tc_001",
        )
        fresh = ConversationStore(db_path=temp_store.db_path)
        session = fresh.get_session("s_tc_001")
        msg = session.messages[0]
        assert msg.tool_calls is not None
        assert msg.tool_calls[0]["name"] == "bash"
        assert "tool_calls" not in msg.metadata  # popped, not duplicated
