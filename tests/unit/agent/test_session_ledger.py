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

"""Tests for SessionLedger — structured session state tracking."""

import pytest

from victor.agent.session_ledger import LedgerEntry, SessionLedger
from victor.config.orchestrator_constants import SessionLedgerConfig


class TestLedgerEntry:
    def test_creation(self):
        entry = LedgerEntry(
            timestamp=1000.0,
            category="file_read",
            key="/path/to/file.py",
            summary="50 lines: class Foo",
            turn_index=3,
        )
        assert entry.category == "file_read"
        assert entry.key == "/path/to/file.py"
        assert entry.turn_index == 3
        assert entry.resolved is False

    def test_frozen(self):
        entry = LedgerEntry(
            timestamp=1000.0,
            category="decision",
            key="d1",
            summary="test",
            turn_index=1,
        )
        with pytest.raises(AttributeError):
            entry.summary = "changed"


class TestSessionLedgerInit:
    def test_default_config(self):
        ledger = SessionLedger()
        assert ledger.config.max_entries == 200
        assert ledger.config.max_render_chars == 3000

    def test_custom_config(self):
        config = SessionLedgerConfig(max_entries=50, max_render_chars=1000, file_summary_max_len=40)
        ledger = SessionLedger(config=config)
        assert ledger.config.max_entries == 50
        assert ledger.config.max_render_chars == 1000


class TestSessionLedgerRecording:
    def test_record_file_read(self):
        ledger = SessionLedger()
        ledger.record_file_read("/src/main.py", "Main entry point module", turn_index=1)
        assert len(ledger.entries) == 1
        assert ledger.entries[0].category == "file_read"
        assert ledger.entries[0].key == "/src/main.py"
        assert "/src/main.py" in ledger.get_files_read()

    def test_record_file_modified(self):
        ledger = SessionLedger()
        ledger.record_file_modified("/src/main.py", "Added new function", turn_index=2)
        assert len(ledger.entries) == 1
        assert ledger.entries[0].category == "file_modified"

    def test_record_decision(self):
        ledger = SessionLedger()
        ledger.record_decision("Use factory pattern for initialization", turn_index=3)
        assert len(ledger.entries) == 1
        assert ledger.entries[0].category == "decision"

    def test_record_recommendation(self):
        ledger = SessionLedger()
        ledger.record_recommendation("Refactor the module into smaller classes", turn_index=4)
        assert len(ledger.entries) == 1
        assert ledger.entries[0].category == "recommendation"

    def test_record_pending_action(self):
        ledger = SessionLedger()
        ledger.record_pending_action("Update tests for new API", turn_index=5)
        assert len(ledger.entries) == 1
        assert ledger.entries[0].category == "pending_action"
        assert ledger.entries[0].resolved is False

    def test_resolve_pending_action(self):
        ledger = SessionLedger()
        ledger.record_pending_action("Update tests", turn_index=5)
        action_key = ledger.entries[0].key
        ledger.resolve_pending_action(action_key)
        assert ledger.entries[0].resolved is True

    def test_resolve_nonexistent_action(self):
        ledger = SessionLedger()
        # Should not raise
        ledger.resolve_pending_action("nonexistent_key")


class TestUpdateFromToolResult:
    def test_read_tool_extraction(self):
        ledger = SessionLedger()
        result = "     1\tclass Foo:\n     2\t    pass\n     3\t"
        ledger.update_from_tool_result("read", {"path": "/src/foo.py"}, result, turn_index=1)
        assert "/src/foo.py" in ledger.get_files_read()
        entries = [e for e in ledger.entries if e.category == "file_read"]
        assert len(entries) == 1

    def test_legacy_read_tool_extraction(self):
        ledger = SessionLedger()
        ledger.update_from_tool_result(
            "read_file",
            {"path": "/src/legacy.py"},
            "line 1\nline 2",
            turn_index=1,
        )
        assert "/src/legacy.py" in ledger.get_files_read()

    def test_edit_tool_extraction(self):
        ledger = SessionLedger()
        ledger.update_from_tool_result(
            "edit",
            {"path": "/src/foo.py", "content": "new content"},
            "OK",
            turn_index=2,
        )
        entries = [e for e in ledger.entries if e.category == "file_modified"]
        assert len(entries) == 1

    def test_legacy_create_file_counts_as_write(self):
        ledger = SessionLedger()
        ledger.update_from_tool_result(
            "create_file",
            {"path": "/src/new.py", "content": "print('x')"},
            "OK",
            turn_index=2,
        )
        entries = [e for e in ledger.entries if e.category == "file_modified"]
        assert len(entries) == 1

    def test_unknown_tool_noop(self):
        ledger = SessionLedger()
        ledger.update_from_tool_result("shell", {"cmd": "ls"}, "output", turn_index=1)
        assert len(ledger.entries) == 0

    def test_malformed_content_graceful(self):
        ledger = SessionLedger()
        # Should not raise
        ledger.update_from_tool_result("read", {"path": "/x"}, "", turn_index=1)
        assert len(ledger.entries) == 1

    def test_tool_output_marker_parsing(self):
        ledger = SessionLedger()
        result = '<TOOL_OUTPUT tool="read" path="/src/bar.py">content here</TOOL_OUTPUT>'
        ledger.update_from_tool_result("shell", {}, result, turn_index=1)
        assert "/src/bar.py" in ledger.get_files_read()

    def test_tool_output_marker_parsing_legacy_alias(self):
        ledger = SessionLedger()
        result = (
            '<TOOL_OUTPUT tool="read_file" path="/src/legacy_bar.py">content here</TOOL_OUTPUT>'
        )
        ledger.update_from_tool_result("shell", {}, result, turn_index=1)
        assert "/src/legacy_bar.py" in ledger.get_files_read()


class TestUpdateFromAssistantResponse:
    def test_recommendation_extraction(self):
        ledger = SessionLedger()
        content = "I recommend refactoring the database module into smaller classes."
        ledger.update_from_assistant_response(content, turn_index=3)
        recs = [e for e in ledger.entries if e.category == "recommendation"]
        assert len(recs) >= 1

    def test_decision_extraction(self):
        ledger = SessionLedger()
        content = "I will use the factory pattern for creating components."
        ledger.update_from_assistant_response(content, turn_index=3)
        decisions = [e for e in ledger.entries if e.category == "decision"]
        assert len(decisions) >= 1

    def test_no_patterns_found(self):
        ledger = SessionLedger()
        content = "Here is the file content you requested."
        ledger.update_from_assistant_response(content, turn_index=3)
        assert len(ledger.entries) == 0

    def test_technical_should_is_not_a_recommendation(self):
        """FEP-0023 P1: bare 'should' in technical prose must not extract.

        On 12.7k real assistant messages the old bare-``should`` pattern dropped
        recommendation precision to ~62% by matching prose like these. Regression
        guard for the hardened patterns.
        """
        ledger = SessionLedger()
        for prose in (
            "The function should return None on an empty input.",
            "You should see the test output above.",
            "Note that the cache should be invalidated on model switch.",
            "This should work now.",
        ):
            ledger.update_from_assistant_response(prose, turn_index=1)
        recs = [e for e in ledger.entries if e.category == "recommendation"]
        assert recs == [], f"unexpected recommendation extractions: {[e.summary for e in recs]}"

    def test_we_should_is_a_recommendation(self):
        """A genuine 'we should X' recommendation is still captured."""
        ledger = SessionLedger()
        ledger.update_from_assistant_response(
            "I think we should measure token cost before defaulting the flag ON.",
            turn_index=1,
        )
        recs = [e for e in ledger.entries if e.category == "recommendation"]
        assert len(recs) >= 1

    def test_empty_content(self):
        ledger = SessionLedger()
        ledger.update_from_assistant_response("", turn_index=1)
        assert len(ledger.entries) == 0


class TestSessionLedgerRender:
    def test_basic_render(self):
        ledger = SessionLedger()
        ledger.record_file_read("/src/main.py", "50 lines", turn_index=1)
        ledger.record_decision("Use factory pattern", turn_index=2)
        rendered = ledger.render()
        assert "<SESSION_STATE>" in rendered
        assert "</SESSION_STATE>" in rendered
        assert "/src/main.py" in rendered
        assert "Use factory pattern" in rendered

    def test_max_chars_truncation(self):
        ledger = SessionLedger()
        for i in range(100):
            ledger.record_file_read(f"/src/file_{i}.py", f"Content of file {i}", turn_index=i)
        rendered = ledger.render(max_chars=500)
        assert len(rendered) <= 520  # Allow small overshoot for closing tag
        assert "</SESSION_STATE>" in rendered

    def test_empty_ledger(self):
        ledger = SessionLedger()
        assert ledger.render() == ""

    def test_category_grouping(self):
        ledger = SessionLedger()
        ledger.record_file_read("/a.py", "file a", turn_index=1)
        ledger.record_decision("decision 1", turn_index=2)
        rendered = ledger.render()
        assert "<file_reads>" in rendered
        assert "<decisions>" in rendered


class TestSessionLedgerMaxEntries:
    def test_eviction_when_max_exceeded(self):
        config = SessionLedgerConfig(max_entries=5)
        ledger = SessionLedger(config=config)
        for i in range(10):
            ledger.record_file_read(f"/file_{i}.py", f"file {i}", turn_index=i)
        assert len(ledger.entries) == 5
        # Oldest entries should be evicted
        assert ledger.entries[0].key == "/file_5.py"


class TestSessionLedgerCheckpoint:
    def test_to_dict_from_dict_roundtrip(self):
        ledger = SessionLedger()
        ledger.record_file_read("/src/main.py", "Main module", turn_index=1)
        ledger.record_decision("Use protocol pattern", turn_index=2)
        ledger.record_pending_action("Write tests", turn_index=3)

        data = ledger.to_dict()
        restored = SessionLedger.from_dict(data)

        assert len(restored.entries) == len(ledger.entries)
        assert restored.get_files_read() == ledger.get_files_read()
        for orig, rest in zip(ledger.entries, restored.entries):
            assert orig.category == rest.category
            assert orig.key == rest.key
            assert orig.summary == rest.summary

    def test_from_dict_empty(self):
        restored = SessionLedger.from_dict({})
        assert len(restored.entries) == 0


class TestGetRecentActionableItems:
    def test_returns_unresolved_actionable(self):
        ledger = SessionLedger()
        ledger.record_file_read("/a.py", "file", turn_index=1)
        ledger.record_decision("decision 1", turn_index=2)
        ledger.record_recommendation("rec 1", turn_index=3)
        ledger.record_pending_action("action 1", turn_index=4)

        items = ledger.get_recent_actionable_items(limit=10)
        categories = {i.category for i in items}
        assert "file_read" not in categories
        assert "decision" in categories
        assert "recommendation" in categories
        assert "pending_action" in categories

    def test_excludes_resolved(self):
        ledger = SessionLedger()
        ledger.record_pending_action("action 1", turn_index=1)
        key = ledger.entries[0].key
        ledger.resolve_pending_action(key)
        items = ledger.get_recent_actionable_items()
        assert len(items) == 0

    def test_respects_limit(self):
        ledger = SessionLedger()
        for i in range(10):
            ledger.record_decision(f"decision {i} is important", turn_index=i)
        items = ledger.get_recent_actionable_items(limit=3)
        assert len(items) == 3
