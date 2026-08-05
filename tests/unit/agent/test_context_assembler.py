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

"""Tests for TurnBoundaryContextAssembler."""

import pytest

from victor.agent.conversation.assembler import TurnBoundaryContextAssembler
from victor.agent.session_ledger import SessionLedger
from victor.config.orchestrator_constants import ContextAssemblerConfig
from victor.providers.base import Message


def _msg(role, content):
    return Message(role=role, content=content)


class TestContextAssemblerConfig:
    def test_defaults(self):
        config = ContextAssemblerConfig()
        assert config.full_turn_count == 3
        assert config.history_budget_pct == 0.70


class TestAssembleBasic:
    def test_short_conversation_passes_through(self):
        assembler = TurnBoundaryContextAssembler()
        msgs = [
            _msg("system", "You are helpful"),
            _msg("user", "hello"),
            _msg("assistant", "hi"),
        ]
        result = assembler.assemble(msgs, max_context_chars=100000)
        # No ledger/score_fn = passthrough
        assert len(result) == 3

    def test_empty_messages(self):
        assembler = TurnBoundaryContextAssembler()
        assert assembler.assemble([], max_context_chars=100000) == []


class TestAssembleWithLedger:
    def test_ledger_rendered_and_injected(self):
        ledger = SessionLedger()
        ledger.record_file_read("/src/main.py", "Main module", turn_index=1)

        assembler = TurnBoundaryContextAssembler(session_ledger=ledger)
        msgs = [
            _msg("system", "You are helpful"),
            _msg("user", "read main.py"),
            _msg("assistant", "Here is the file"),
        ]
        result = assembler.assemble(msgs, max_context_chars=100000)
        # Should have system + ledger + user + assistant
        assert len(result) >= 4
        # Second message should be the ledger
        assert "<SESSION_STATE>" in result[1].content


class TestAssembleTokenBudget:
    def test_respects_max_context_chars(self):
        ledger = SessionLedger()
        assembler = TurnBoundaryContextAssembler(session_ledger=ledger)
        # Create a long conversation
        msgs = [_msg("system", "sys" * 10)]
        for i in range(20):
            msgs.append(_msg("user", f"question {i} " * 100))
            msgs.append(_msg("assistant", f"answer {i} " * 100))

        result = assembler.assemble(msgs, max_context_chars=5000)
        total_chars = sum(len(m.content) for m in result)
        # Should be within budget (approximately)
        assert total_chars < 10000  # Generous bound since recent turns always kept


class TestAssembleRecentTurns:
    def test_last_3_turns_always_kept(self):
        ledger = SessionLedger()
        assembler = TurnBoundaryContextAssembler(session_ledger=ledger)
        msgs = [_msg("system", "sys")]
        for i in range(10):
            msgs.append(_msg("user", f"q{i}"))
            msgs.append(_msg("assistant", f"a{i}"))

        result = assembler.assemble(msgs, max_context_chars=100000)
        contents = [m.content for m in result]
        # Last 3 user messages should be present
        assert "q7" in contents
        assert "q8" in contents
        assert "q9" in contents


class TestAssembleOlderMessageSelection:
    def test_scoring_function_called(self):
        calls = []

        def score_fn(messages, query):
            calls.append(len(messages))
            return [(msg, 1.0 / (i + 1)) for i, msg in enumerate(messages)]

        ledger = SessionLedger()
        assembler = TurnBoundaryContextAssembler(session_ledger=ledger, score_fn=score_fn)
        msgs = [_msg("system", "sys")]
        for i in range(10):
            msgs.append(_msg("user", f"q{i}"))
            msgs.append(_msg("assistant", f"a{i}"))

        assembler.assemble(msgs, max_context_chars=100000, current_query="test")
        assert len(calls) > 0


class TestAssembleGracefulDegradation:
    def test_no_ledger_no_score_returns_raw(self):
        assembler = TurnBoundaryContextAssembler()
        msgs = [_msg("user", "hello"), _msg("assistant", "hi")]
        result = assembler.assemble(msgs, max_context_chars=100000)
        assert len(result) == 2
        assert result[0].content == "hello"
        assert result[1].content == "hi"
