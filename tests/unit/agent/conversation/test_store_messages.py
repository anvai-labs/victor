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


class TestSaveSessionLiveSaveFidelity:
    """/save must not degrade rows that add_message already persisted.

    Regression guard for the live-save path: MessageHistory.to_dict payloads
    (id-less dicts) used to replace persisted rows with degraded copies
    (metadata={}, token_count=0, save-time timestamps, NULL lineage).
    """

    def _persist_live_session(self, store):
        """Simulate the live chat flow: add_message with source metadata."""
        from victor.agent.conversation.types import MessageSource

        store.create_session(session_id="live_001", provider="p", model="m")
        store.add_message(
            "live_001",
            MessageRole.USER,
            "run the analysis",
            metadata={"source": MessageSource.USER_TYPED.value, "agent_id": "ag1"},
        )
        store.add_message(
            "live_001",
            MessageRole.ASSISTANT,
            "analysis complete, 42 records matched",
            metadata={"source": MessageSource.AGENT_CONTINUATION.value},
        )
        store.add_message(
            "live_001",
            MessageRole.USER,
            "thanks",
            metadata={"source": MessageSource.USER_TYPED.value},
        )

    def _snapshot_rows(self, db_path):
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM messages WHERE session_id='live_001' ORDER BY rowid"
                )
            ]

    def _history_payload(self, count, with_metadata=True):
        """MessageHistory.to_dict-equivalent message dicts (no ids/counts)."""
        base = [
            {
                "role": "user",
                "content": "run the analysis",
                "metadata": {"source": "user_typed", "agent_id": "ag1"},
            },
            {
                "role": "assistant",
                "content": "analysis complete, 42 records matched",
                "metadata": {"source": "agent_continuation"},
            },
            {"role": "user", "content": "thanks", "metadata": {"source": "user_typed"}},
        ]
        msgs = base[:count]
        if not with_metadata:
            msgs = [{k: v for k, v in m.items() if k != "metadata"} for m in msgs]
        return {"messages": msgs}

    def test_live_save_leaves_persisted_rows_untouched(self, temp_store):
        """Equal-count id-less save (the /save shape) rewrites nothing."""
        self._persist_live_session(temp_store)
        before = self._snapshot_rows(temp_store.db_path)
        assert len(before) == 3

        temp_store.save_session(
            conversation=self._history_payload(3),
            model="m",
            provider="p",
            session_id="live_001",
            title="after live save",
        )

        after = self._snapshot_rows(temp_store.db_path)
        assert after == before, "live /save must not rewrite persisted rows"
        # session row still updates (title)
        loaded = ConversationStore(db_path=temp_store.db_path).load_session("live_001")
        assert loaded["metadata"]["title"] == "after live save"

    def test_count_mismatch_rewrites_with_enrichment(self, temp_store):
        """Post-/compact shape (fewer messages) falls through to rewrite."""
        self._persist_live_session(temp_store)
        temp_store.save_session(
            conversation=self._history_payload(1),
            model="m",
            provider="p",
            session_id="live_001",
        )
        rows = self._snapshot_rows(temp_store.db_path)
        assert len(rows) == 1
        # metadata from the payload survives; token_count estimated, not zero
        assert "user_typed" in rows[0]["metadata"] or "source" in rows[0]["metadata"]
        assert rows[0]["token_count"] > 0

    def test_missing_token_count_is_estimated(self, temp_store):
        temp_store.save_session(
            conversation={"messages": [{"role": "user", "content": "estimate me"}]},
            model="m",
            provider="p",
            session_id="est_001",
        )
        rows = self._snapshot_rows(temp_store.db_path)
        assert len(rows) == 0  # different session id — query the right one
        with sqlite3.connect(temp_store.db_path) as conn:
            (tokens,) = conn.execute(
                "SELECT token_count FROM messages WHERE session_id='est_001'"
            ).fetchone()
        assert tokens > 0

    def test_explicit_token_count_still_wins(self, temp_store):
        temp_store.save_session(
            conversation={"messages": [{"role": "user", "content": "exact", "token_count": 4242}]},
            model="m",
            provider="p",
            session_id="exact_001",
        )
        with sqlite3.connect(temp_store.db_path) as conn:
            (tokens,) = conn.execute(
                "SELECT token_count FROM messages WHERE session_id='exact_001'"
            ).fetchone()
        assert tokens == 4242

    def test_live_save_still_syncs_preview_sidecar(self, temp_store):
        """The skip path must not freeze previews: new sidecar entries land."""
        self._persist_live_session(temp_store)
        temp_store.save_session(
            conversation={
                "messages": self._history_payload(3)["messages"],
                "preview_messages": [
                    {
                        "role": "assistant",
                        "content": "reflection preview",
                        "metadata": {"preview_kind": "reflection"},
                    }
                ],
            },
            model="m",
            provider="p",
            session_id="live_001",
        )
        # message rows untouched…
        assert len(self._snapshot_rows(temp_store.db_path)) == 3
        # …but the sidecar picked up the new preview
        loaded = ConversationStore(db_path=temp_store.db_path).load_session("live_001")
        previews = loaded["conversation"]["preview_messages"]
        assert len(previews) == 1
        assert previews[0]["content"] == "reflection preview"


class TestCacheHitRateScale:
    """`cache_hit_rate` is a fraction 0..1 everywhere — the canonical scale of
    the metrics-service producers, the `{:.0%}` presenters, and the TUI bar.
    The store used to emit a percentage under the same key (audit B5): any
    consumer wiring the store's value into a fraction-expecting renderer showed
    "9000% cached". Both producers are pinned here."""

    def test_session_stats_hit_rate_is_fraction(self, temp_store):
        self._seed(temp_store, "s_hit_001")
        stats = temp_store.get_session_token_stats("s_hit_001")
        assert stats is not None
        assert stats["cache_hit_rate"] == pytest.approx(0.9)

    def test_aggregate_hit_rate_is_fraction(self, temp_store):
        self._seed(temp_store, "s_hit_002")
        usage = temp_store.get_total_token_usage()
        assert usage["cache_hit_rate"] == pytest.approx(0.9)

    def test_hit_rate_zero_when_no_prompt_tokens(self, temp_store):
        temp_store.save_session(
            conversation={"messages": []},
            model="test-model",
            provider="test-provider",
            session_id="s_hit_003",
        )
        stats = temp_store.get_session_token_stats("s_hit_003")
        assert stats is not None
        assert stats["cache_hit_rate"] == 0.0

    @staticmethod
    def _seed(store, session_id):
        store.save_session(
            conversation={"messages": [{"role": "user", "content": "hi"}]},
            model="test-model",
            provider="test-provider",
            session_id=session_id,
        )
        store.update_session_token_usage(
            session_id,
            prompt_tokens=1000,
            completion_tokens=100,
            cached_tokens=900,
        )
