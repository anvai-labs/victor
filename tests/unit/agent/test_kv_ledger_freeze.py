"""KV prefix stability: per-user-turn ledger render freeze (F7 fix).

The session ledger is injected at message index 1 and the AgenticLoop
live-updates it after every tool call. Re-rendering it per provider call
mutated already-sent bytes immediately after the system prompt — the measured
production failure (cache hits frozen at the tools+system head). These tests
pin the freeze: intra-turn renders are byte-stable; a new anchor re-renders.
"""

from __future__ import annotations

from victor.agent.conversation.assembler import TurnBoundaryContextAssembler
from victor.agent.session_ledger import SessionLedger
from victor.providers.base import Message


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


def _history_with_anchor() -> list[Message]:
    return [
        _msg("system", "You are helpful"),
        _msg("user", "read src/a.py and summarize"),
        _msg("assistant", "reading"),
        _msg("tool", "TOOL_OUTPUT src/a.py: 10 lines"),
        _msg("assistant", "a.py defines helpers"),
    ]


def _ledger_message(result: list[Message]) -> Message:
    assert result[0].role == "system"
    assert result[1].role == "assistant", "ledger must sit at index 1"
    return result[1]


def _make_assembler(ledger: SessionLedger) -> TurnBoundaryContextAssembler:
    return TurnBoundaryContextAssembler(session_ledger=ledger)


def test_ledger_render_frozen_intra_turn():
    ledger = SessionLedger()
    ledger.record_file_read("src/a.py", "10 lines", 0)
    assembler = _make_assembler(ledger)
    # Same Message objects across calls (append-only production flow); the
    # assembler receives a fresh list view but identical object identities.
    history = _history_with_anchor()

    first = _ledger_message(assembler.assemble(history, 100000))
    # Mid-loop ledger updates (the production mutator):
    ledger.record_file_read("src/b.py", "20 lines", 1)
    ledger.record_decision("use helper module", 1)
    ledger.record_pending_action("deploy", 1)
    ledger.resolve_pending_action("deploy")
    second = _ledger_message(assembler.assemble(history, 100000))

    assert second.content == first.content, "intra-turn ledger re-render defeats prefix cache"


def test_new_user_turn_re_renders_ledger():
    ledger = SessionLedger()
    ledger.record_file_read("src/a.py", "10 lines", 0)
    assembler = _make_assembler(ledger)
    history = _history_with_anchor()

    first = _ledger_message(assembler.assemble(history, 100000))
    ledger.record_file_read("src/b.py", "20 lines", 1)
    history.append(_msg("user", "now read src/c.py too"))
    second = _ledger_message(assembler.assemble(history, 100000))

    assert second.content != first.content, "freshness must return at the user-turn boundary"
    assert "src/b.py" in second.content


def test_reset_turn_freezes_forces_re_render():
    ledger = SessionLedger()
    ledger.record_file_read("src/a.py", "10 lines", 0)
    assembler = _make_assembler(ledger)
    history = _history_with_anchor()

    first = _ledger_message(assembler.assemble(history, 100000))
    ledger.record_file_read("src/b.py", "20 lines", 1)
    assembler.reset_turn_freezes()
    second = _ledger_message(assembler.assemble(history, 100000))

    assert second.content != first.content
    assert "src/b.py" in second.content


def test_hidden_runtime_message_does_not_move_the_anchor():
    """Appended hidden failure hints must not count as a new user turn."""
    from victor.agent.conversation.history_metadata import (
        build_internal_history_metadata,
        is_user_turn,
    )

    history = _history_with_anchor()
    before = TurnBoundaryContextAssembler._turn_anchor(history)
    history.append(
        Message(
            role="user",
            content="[hint] previous tool call failed; adjust arguments",
            metadata=build_internal_history_metadata(kind="failure_hint"),
        )
    )
    after = TurnBoundaryContextAssembler._turn_anchor(history)
    assert before is not None
    assert after == before
    assert not is_user_turn(history[-1].role, history[-1].metadata)
