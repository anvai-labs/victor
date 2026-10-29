"""Tests for cross-session context linker."""

from unittest.mock import MagicMock

import pytest

from victor.agent.session_context_linker import (
    SessionContextLinker,
)
from victor.agent.session_ledger import SessionLedger


@pytest.fixture
def mock_persistence():
    persistence = MagicMock()
    return persistence


@pytest.fixture
def sample_session_data():
    ledger = SessionLedger()
    ledger.record_file_read("auth.py", "Auth module", 1)
    ledger.record_decision("Use factory pattern", 2)
    ledger.record_pending_action("Write tests", 3)

    return {
        "metadata": {
            "session_id": "test-session-123",
            "title": "Auth Implementation",
            "updated_at": "2026-03-19T10:00:00",
        },
        "conversation": {"messages": [{"role": "user", "content": "hello"}]},
        "session_ledger": ledger.to_dict(),
        "execution_state": {
            "observed_files": ["auth.py"],
            "executed_tools": ["read"],
            "tool_calls_used": 5,
        },
    }


@pytest.fixture
def linker(mock_persistence):
    return SessionContextLinker(session_persistence=mock_persistence)


class TestSessionContextLinker:
    def test_build_resume_context_restores_ledger(
        self, linker, mock_persistence, sample_session_data
    ):
        mock_persistence.load_session.return_value = sample_session_data
        ctx = linker.build_resume_context("test-session-123")

        assert ctx.ledger is not None
        assert len(ctx.ledger.entries) == 3
        files = ctx.ledger.get_files_read()
        assert "auth.py" in files

    def test_build_resume_context_restores_execution_state(
        self, linker, mock_persistence, sample_session_data
    ):
        mock_persistence.load_session.return_value = sample_session_data
        ctx = linker.build_resume_context("test-session-123")

        assert ctx.execution_state is not None
        assert ctx.execution_state.tool_calls_used == 5

    def test_resume_summary_includes_files_and_decisions(
        self, linker, mock_persistence, sample_session_data
    ):
        mock_persistence.load_session.return_value = sample_session_data
        ctx = linker.build_resume_context("test-session-123")

        assert "auth.py" in ctx.resume_summary
        assert (
            "factory pattern" in ctx.resume_summary.lower()
            or "decided" in ctx.resume_summary.lower()
        )

    def test_resume_summary_empty_session(self, linker, mock_persistence):
        mock_persistence.load_session.return_value = None
        ctx = linker.build_resume_context("nonexistent")

        assert "not found" in ctx.resume_summary.lower()
        assert ctx.ledger is None

    def test_find_related_sessions_uses_search(self, mock_persistence):
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {"session_id": "s1", "title": "Related"},
        ]
        linker = SessionContextLinker(
            session_persistence=mock_persistence,
            conversation_store=mock_store,
        )

        results = linker.find_related_sessions("auth implementation")
        assert len(results) == 1
        mock_store.search.assert_called_once()

    def test_cross_session_context_respects_max_chars(self, mock_persistence):
        # Create session with large ledger
        ledger = SessionLedger()
        for i in range(50):
            ledger.record_file_read(f"file_{i}.py", f"Module {i} " * 20, i)

        mock_persistence.load_session.return_value = {
            "metadata": {"title": "Big session"},
            "session_ledger": ledger.to_dict(),
        }

        linker = SessionContextLinker(session_persistence=mock_persistence)
        result = linker.build_cross_session_context("query", ["s1"], max_chars=500)

        assert len(result) <= 600  # Some tolerance for formatting

    def test_no_store_graceful_degradation(self, mock_persistence):
        linker = SessionContextLinker(
            session_persistence=mock_persistence,
            conversation_store=None,
        )

        results = linker.find_related_sessions("query")
        assert results == []

        context = linker.build_cross_session_context("query", ["s1"])
        assert context == ""

    def test_build_resume_context_populates_compaction_summaries(
        self, linker, mock_persistence, sample_session_data
    ):
        """compaction_summaries is populated from persisted hierarchy data (Fix 1)."""
        sample_session_data["compaction_hierarchy"] = {
            "individual_summaries": [
                {"summary": "implemented auth module", "turn_index": 5},
            ],
            "epochs": [],
            "max_individual": 3,
            "epoch_threshold": 6,
        }
        mock_persistence.load_session.return_value = sample_session_data

        ctx = linker.build_resume_context("test-session-123")

        assert len(ctx.compaction_summaries) == 1
        assert "auth module" in ctx.compaction_summaries[0]

    def test_build_resume_context_no_summaries_when_no_hierarchy(
        self, linker, mock_persistence, sample_session_data
    ):
        """compaction_summaries is empty when session has no persisted hierarchy."""
        mock_persistence.load_session.return_value = sample_session_data  # no hierarchy key

        ctx = linker.build_resume_context("test-session-123")

        assert ctx.compaction_summaries == []

    def test_resume_summary_includes_preview_paths(
        self, linker, mock_persistence, sample_session_data
    ):
        """Resume summary should surface persisted preview sidecar files."""
        sample_session_data["conversation"]["preview_messages"] = [
            {
                "role": "system",
                "content": "FILE PREVIEW: app.py",
                "metadata": {
                    "preview_kind": "file_preview",
                    "preview_path": "app.py",
                    "preview_language": "python",
                    "preview_body": "print('hello')\n",
                },
                "after_message_index": 1,
            }
        ]
        mock_persistence.load_session.return_value = sample_session_data

        ctx = linker.build_resume_context("test-session-123")

        assert "previews:" in ctx.resume_summary.lower()
        assert "app.py" in ctx.resume_summary
