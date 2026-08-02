# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""FEP-0023 Phase 1: SessionLedger population at the agentic-loop turn boundary.

Exercises ``AgenticLoop._populate_session_ledger`` — the single shared seam that
both the buffered (``run``) and streaming (``run_streaming``) loops call once per
turn — verifying the ``USE_SESSION_LEDGER`` gate, that it populates the live
ledger from tool results + assistant text when enabled, and that it degrades
safely (no ledger / malformed entries).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from victor.agent.session_ledger import SessionLedger
from victor.core.feature_flags import (
    FeatureFlag,
    get_feature_flag_manager,
    reset_feature_flag_manager,
)
from victor.framework.agentic_loop import AgenticLoop


@pytest.fixture(autouse=True)
def _reset_flags():
    """Isolate flag state so USE_SESSION_LEDGER never leaks across tests."""
    reset_feature_flag_manager()
    yield
    reset_feature_flag_manager()


def _make_loop(ledger):
    """AgenticLoop over a mock orchestrator exposing a real SessionLedger."""
    orchestrator = MagicMock()
    orchestrator._session_ledger = ledger
    return AgenticLoop(orchestrator=orchestrator, enable_fulfillment_check=False)


def _turn_result(content="", tool_results=None):
    """A minimal TurnResult-shaped object (``.content`` + ``.tool_results``)."""
    return SimpleNamespace(content=content, tool_results=tool_results or [])


def _enable():
    get_feature_flag_manager().enable(FeatureFlag.USE_SESSION_LEDGER)


def _disable():
    get_feature_flag_manager().disable(FeatureFlag.USE_SESSION_LEDGER)


class TestSessionLedgerPopulation:
    def test_flag_off_is_a_noop(self):
        """With the flag explicitly OFF: the ledger stays empty and <SESSION_STATE> inert."""
        ledger = SessionLedger()
        loop = _make_loop(ledger)
        _disable()

        action = _turn_result(
            content="I recommend refactoring the config module.",
            tool_results=[
                {"tool_name": "read", "args": {"path": "/src/config.py"}, "result": "line1\nline2"}
            ],
        )
        loop._populate_session_ledger(action, turn_index=1)

        assert ledger.get_files_read() == {}
        assert ledger.render() == ""

    def test_flag_on_populates_files_read(self):
        ledger = SessionLedger()
        loop = _make_loop(ledger)
        _enable()

        action = _turn_result(
            tool_results=[
                {"tool_name": "read", "args": {"path": "/src/config.py"}, "result": "a\nb\nc"}
            ]
        )
        loop._populate_session_ledger(action, turn_index=2)

        assert "/src/config.py" in ledger.get_files_read()
        assert "<SESSION_STATE>" in ledger.render()

    def test_flag_on_records_file_modification(self):
        ledger = SessionLedger()
        loop = _make_loop(ledger)
        _enable()

        action = _turn_result(
            tool_results=[{"tool_name": "edit", "args": {"path": "/src/main.py"}, "result": "ok"}]
        )
        loop._populate_session_ledger(action, turn_index=3)

        rendered = ledger.render()
        assert "file_modified" in rendered
        assert "/src/main.py" in rendered

    def test_flag_on_extracts_recommendation_from_assistant_text(self):
        ledger = SessionLedger()
        loop = _make_loop(ledger)
        _enable()

        action = _turn_result(content="I recommend refactoring the config module into pieces.")
        loop._populate_session_ledger(action, turn_index=4)

        recs = [e for e in ledger.entries if e.category == "recommendation"]
        assert len(recs) >= 1

    def test_alternate_key_name_is_accepted(self):
        """Tool-result dicts keyed ``name`` (recovery path) are handled too."""
        ledger = SessionLedger()
        loop = _make_loop(ledger)
        _enable()

        action = _turn_result(
            tool_results=[
                {"name": "read", "args": {"path": "/src/x.py"}, "full_result": "one\ntwo"}
            ]
        )
        loop._populate_session_ledger(action, turn_index=5)

        assert "/src/x.py" in ledger.get_files_read()

    def test_missing_ledger_does_not_raise(self):
        """A partially-built orchestrator without a ledger degrades to a no-op."""
        orchestrator = MagicMock(spec=[])  # no _session_ledger attribute
        loop = AgenticLoop(orchestrator=orchestrator, enable_fulfillment_check=False)
        _enable()

        # Must not raise.
        loop._populate_session_ledger(_turn_result(content="do something"), turn_index=1)

    def test_loop_without_orchestrator_attribute_does_not_raise(self):
        """A bare __new__-constructed loop (no ``orchestrator`` attr) is a safe no-op.

        Mirrors how the streaming unit tests build the loop; the flag defaults ON,
        so the helper must guard the attribute access itself, not just its value.
        """
        loop = AgenticLoop.__new__(AgenticLoop)  # __init__ bypassed: no self.orchestrator
        _enable()
        assert not hasattr(loop, "orchestrator")

        # Must not raise.
        loop._populate_session_ledger(_turn_result(content="do it"), turn_index=1)

    def test_malformed_entries_are_skipped(self):
        ledger = SessionLedger()
        loop = _make_loop(ledger)
        _enable()

        action = _turn_result(
            content="",
            tool_results=[
                None,
                "not-a-dict",
                {"args": {"path": "/skip.py"}},  # no tool name
                {"tool_name": "read", "args": {"path": "/keep.py"}, "result": "x\ny"},
            ],
        )
        loop._populate_session_ledger(action, turn_index=6)

        files = ledger.get_files_read()
        assert "/keep.py" in files
        assert "/skip.py" not in files
