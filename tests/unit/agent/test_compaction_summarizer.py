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

"""Tests for compaction summarizer strategies."""

import pytest

from victor.agent.compaction_summarizer import (
    KeywordCompactionSummarizer,
    LedgerAwareCompactionSummarizer,
)
from victor.agent.session_ledger import SessionLedger
from victor.providers.base import Message


def _make_messages(roles_contents):
    return [Message(role=r, content=c) for r, c in roles_contents]


class TestKeywordCompactionSummarizer:
    def test_basic_summary(self):
        summarizer = KeywordCompactionSummarizer()
        msgs = _make_messages(
            [
                ("user", "Please read the config file"),
                ("assistant", "Here is the ConfigManager class"),
                ("user", "Now update the test_config module"),
            ]
        )
        result = summarizer.summarize(msgs)
        assert "2 user messages" in result
        assert "Earlier conversation" in result

    def test_empty_messages(self):
        summarizer = KeywordCompactionSummarizer()
        assert summarizer.summarize([]) == ""

    def test_matches_existing_output_format(self):
        summarizer = KeywordCompactionSummarizer()
        msgs = _make_messages(
            [
                ("user", "hello"),
                ("tool", "tool output here"),
            ]
        )
        result = summarizer.summarize(msgs)
        assert result.startswith("[Earlier conversation:")
        assert result.endswith("]")


class TestLedgerAwareCompactionSummarizer:
    def test_with_populated_ledger(self):
        ledger = SessionLedger()
        ledger.record_file_read("/src/main.py", "50 lines", turn_index=1)
        ledger.record_decision("Use factory pattern", turn_index=2)
        ledger.record_pending_action("Write unit tests", turn_index=3)

        summarizer = LedgerAwareCompactionSummarizer()
        msgs = _make_messages(
            [
                ("user", "Read main.py"),
                ("assistant", "I will use the factory pattern"),
                ("user", "Sounds good"),
            ]
        )
        result = summarizer.summarize(msgs, ledger=ledger)
        assert "Compacted context" in result
        assert "/src/main.py" in result or "1 file" in result

    def test_with_empty_ledger_falls_back(self):
        ledger = SessionLedger()
        summarizer = LedgerAwareCompactionSummarizer()
        msgs = _make_messages([("user", "hello world")])
        result = summarizer.summarize(msgs, ledger=ledger)
        # Falls back to keyword summarizer
        assert "Earlier conversation" in result or result == ""

    def test_without_ledger_falls_back(self):
        summarizer = LedgerAwareCompactionSummarizer()
        msgs = _make_messages([("user", "hello"), ("assistant", "hi")])
        result = summarizer.summarize(msgs, ledger=None)
        assert "Earlier conversation" in result


class TestCompactionSummaryContent:
    def test_decisions_preserved(self):
        ledger = SessionLedger()
        ledger.record_decision("Refactor into smaller modules", turn_index=1)
        summarizer = LedgerAwareCompactionSummarizer()
        msgs = _make_messages([("user", "x"), ("assistant", "y")])
        result = summarizer.summarize(msgs, ledger=ledger)
        assert "Decided" in result

    def test_files_mentioned(self):
        ledger = SessionLedger()
        ledger.record_file_read("/a.py", "file a", turn_index=1)
        ledger.record_file_read("/b.py", "file b", turn_index=2)
        summarizer = LedgerAwareCompactionSummarizer()
        msgs = _make_messages([("user", "x")] * 3)
        result = summarizer.summarize(msgs, ledger=ledger)
        assert "Read 2 files" in result

    def test_pending_actions_included(self):
        ledger = SessionLedger()
        ledger.record_pending_action("Deploy to staging", turn_index=1)
        summarizer = LedgerAwareCompactionSummarizer()
        msgs = _make_messages([("user", "x")])
        result = summarizer.summarize(msgs, ledger=ledger)
        assert "Pending" in result
