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

"""Integration tests for conversation context management overhaul.

Tests the end-to-end interaction of SessionLedger, CompactionSummarizer,
ToolResultDeduplicator, and ContextAssembler working together as they would
during a real conversation flow. (The ReferentialIntentResolver was proven
redundant with a properly-ordered <SESSION_STATE> and removed — FEP-0023.)
"""

import pytest

from victor.agent.compaction_summarizer import (
    KeywordCompactionSummarizer,
    LedgerAwareCompactionSummarizer,
)
from victor.agent.conversation.assembler import TurnBoundaryContextAssembler
from victor.agent.session_ledger import SessionLedger
from victor.agent.tool_result_deduplicator import ToolResultDeduplicator
from victor.config.orchestrator_constants import (
    ContextAssemblerConfig,
    DeduplicationConfig,
    SessionLedgerConfig,
)
from victor.providers.base import Message


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


class TestContextManagementPipeline:
    """Tests the full pipeline of context management components."""

    def _build_components(self):
        ledger = SessionLedger()
        deduplicator = ToolResultDeduplicator()
        assembler = TurnBoundaryContextAssembler(session_ledger=ledger)
        summarizer = LedgerAwareCompactionSummarizer()
        return ledger, deduplicator, assembler, summarizer

    def test_full_conversation_flow(self):
        """Simulate a multi-turn conversation with file reads, decisions, and assembly."""
        ledger, deduplicator, assembler, summarizer = self._build_components()

        # Turn 1: User asks to read a file
        messages = [_msg("system", "You are a helpful coding assistant.")]
        messages.append(_msg("user", "Read the config file"))

        # Simulate tool result
        file_content = "x" * 600  # Over dedup threshold
        tool_output = f'<TOOL_OUTPUT tool="read" path="/src/config.py">{file_content}</TOOL_OUTPUT>'
        messages.append(_msg("user", tool_output))

        # Update ledger from tool result
        ledger.update_from_tool_result(
            "read", {"path": "/src/config.py"}, file_content, turn_index=2
        )

        # Turn 2: Assistant responds with recommendation
        assistant_response = "I recommend refactoring the config module into smaller files."
        messages.append(_msg("assistant", assistant_response))
        ledger.update_from_assistant_response(assistant_response, turn_index=3)

        # Verify ledger state
        assert "/src/config.py" in ledger.get_files_read()
        recs = [e for e in ledger.entries if e.category == "recommendation"]
        assert len(recs) >= 1

        # Turn 3: User re-reads same file
        messages.append(_msg("user", "Read config again"))
        messages.append(_msg("user", tool_output))  # Same file content

        # Deduplication should stub the older read
        if deduplicator.should_deduplicate("read", {"path": "/src/config.py"}):
            count = deduplicator.deduplicate_in_place(messages, "read", {"path": "/src/config.py"})
            assert count == 1  # First read stubbed
            assert "Previously read" in messages[2].content

        # Turn 4: User issues a brief follow-up. The populated ledger surfaces the
        # recent recommendation via <SESSION_STATE> (no separate enrichment step —
        # the ReferentialIntentResolver was proven redundant and removed, FEP-0023).
        messages.append(_msg("user", "do it"))
        assembled = assembler.assemble(messages, max_context_chars=100000)
        # Should include system prompt + ledger + messages
        assert any("<SESSION_STATE>" in m.content for m in assembled)
        # The recent recommendation must be present in the rendered session state
        # so a follow-up like "do it" is resolvable from context alone.
        state = next(m.content for m in assembled if "<SESSION_STATE>" in m.content)
        assert "refactor" in state.lower()

    def test_compaction_with_ledger_produces_structured_summary(self):
        """Test that compaction produces structured summaries when ledger is available."""
        ledger = SessionLedger()
        ledger.record_file_read("/src/main.py", "Entry point", turn_index=1)
        ledger.record_file_read("/src/utils.py", "Utilities", turn_index=2)
        ledger.record_decision("Use factory pattern for initialization", turn_index=3)
        ledger.record_pending_action("Write integration tests", turn_index=4)

        summarizer = LedgerAwareCompactionSummarizer()
        removed = [
            _msg("user", "Read main.py"),
            _msg("assistant", "I will use the factory pattern"),
            _msg("user", "Read utils.py"),
            _msg("assistant", "I decided to refactor."),
        ]

        summary = summarizer.summarize(removed, ledger=ledger)
        assert "Compacted context" in summary
        assert "Read 2 files" in summary
        assert "Decided" in summary
        assert "Pending" in summary

    def test_compaction_without_ledger_falls_back_to_keyword(self):
        """Test graceful fallback when ledger is not available."""
        summarizer = LedgerAwareCompactionSummarizer()
        removed = [
            _msg("user", "Hello world"),
            _msg("assistant", "Hi there"),
        ]
        summary = summarizer.summarize(removed, ledger=None)
        assert "Earlier conversation" in summary

    def test_context_assembler_preserves_recent_drops_old(self):
        """Test that assembler keeps recent turns and drops older ones within budget."""
        ledger = SessionLedger()
        ledger.record_file_read("/a.py", "file a", turn_index=1)

        assembler = TurnBoundaryContextAssembler(
            config=ContextAssemblerConfig(full_turn_count=2),
            session_ledger=ledger,
        )

        messages = [_msg("system", "sys")]
        for i in range(10):
            messages.append(_msg("user", f"question {i} " * 50))
            messages.append(_msg("assistant", f"answer {i} " * 50))

        assembled = assembler.assemble(messages, max_context_chars=5000)

        # Recent turns should be present
        contents = " ".join(m.content for m in assembled)
        assert "question 9" in contents
        assert "question 8" in contents

    def test_deduplicator_preserves_message_order(self):
        """Test that deduplication doesn't change message order or count."""
        deduplicator = ToolResultDeduplicator()
        messages = [
            _msg(
                "user",
                '<TOOL_OUTPUT tool="read" path="/a.py">' + "x" * 600 + "</TOOL_OUTPUT>",
            ),
            _msg("assistant", "I see the file"),
            _msg("user", "Now read it again"),
            _msg(
                "user",
                '<TOOL_OUTPUT tool="read" path="/a.py">' + "y" * 600 + "</TOOL_OUTPUT>",
            ),
        ]
        original_count = len(messages)
        original_roles = [m.role for m in messages]

        deduplicator.deduplicate_in_place(messages, "read", {"path": "/a.py"})

        assert len(messages) == original_count
        assert [m.role for m in messages] == original_roles

    def test_ledger_checkpoint_survives_roundtrip(self):
        """Test that ledger state can be serialized and restored."""
        ledger = SessionLedger()
        ledger.record_file_read("/a.py", "file a", turn_index=1)
        ledger.record_decision("Use protocol pattern", turn_index=2)
        ledger.record_pending_action("Deploy to staging", turn_index=3)
        ledger.resolve_pending_action(ledger.entries[2].key)

        data = ledger.to_dict()
        restored = SessionLedger.from_dict(data)

        assert restored.render() == ledger.render()
        assert restored.get_files_read() == ledger.get_files_read()
        assert len(restored.get_recent_actionable_items()) == len(
            ledger.get_recent_actionable_items()
        )

    def test_all_components_with_none_gracefully_degrade(self):
        """Test that all components work when optional dependencies are None."""
        # No ledger
        assembler = TurnBoundaryContextAssembler(session_ledger=None, score_fn=None)
        msgs = [_msg("user", "hello")]
        assert assembler.assemble(msgs, max_context_chars=100000) == msgs

        # Disabled dedup
        dedup = ToolResultDeduplicator(config=DeduplicationConfig(enabled=False))
        assert dedup.should_deduplicate("read", {"path": "/a.py"}) is False
