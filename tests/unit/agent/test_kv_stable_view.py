"""KV prefix stability: dedup direction, sticky selection, max_history unification."""

from __future__ import annotations

import logging

import pytest

from victor.agent.conversation.assembler import TurnBoundaryContextAssembler
from victor.agent.conversation.history_metadata import (
    build_internal_history_metadata,
    mid_user_turn,
)
from victor.agent.message_history import MessageHistory
from victor.agent.tool_result_deduplicator import ToolResultDeduplicator
from victor.config.conversation_settings import DEFAULT_MAX_CONVERSATION_HISTORY
from victor.config.settings import Settings
from victor.providers.base import Message


def _msg(role: str, content: str, **kwargs) -> Message:
    return Message(role=role, content=content, **kwargs)


def _dup_content(tag: str) -> str:
    return f"<TOOL_OUTPUT> read src/{tag}.py\n" + ("export function stable() { return 1; }\n" * 40)


# ── F12: dedup stub direction ──────────────────────────────────────────


class TestDedupDirection:
    def test_older_message_stays_full_newer_duplicate_stubbed(self):
        dedup = ToolResultDeduplicator()
        messages = [
            _msg("user", "task"),
            _msg("user", _dup_content("big")),
            _msg("assistant", "re-reading"),
            _msg("user", _dup_content("big")),
        ]
        view, stubbed = dedup.deduplicate_history_view(messages)
        assert stubbed == 1
        assert view[1].content == _dup_content("big"), "older already-sent bytes must stay full"
        assert view[3].content.startswith("[Duplicate tool output")
        assert "see earlier message" in view[3].content

    def test_stub_is_stable_across_repeated_assembles(self):
        dedup = ToolResultDeduplicator()
        messages = [
            _msg("user", _dup_content("big")),
            _msg("user", _dup_content("big")),
        ]
        first, _ = dedup.deduplicate_history_view(list(messages))
        second, _ = dedup.deduplicate_history_view(list(messages))
        assert [m.content for m in first] == [m.content for m in second]

    def test_three_copy_chain_keeps_first_full(self):
        dedup = ToolResultDeduplicator()
        messages = [_msg("user", _dup_content("same")) for _ in range(3)]
        view, stubbed = dedup.deduplicate_history_view(messages)
        assert stubbed == 2
        assert view[0].content == _dup_content("same")
        assert view[1].content.startswith("[Duplicate")
        assert view[2].content.startswith("[Duplicate")

    def test_stub_preserves_metadata(self):
        dedup = ToolResultDeduplicator()
        meta = build_internal_history_metadata(kind="tool_result")
        messages = [
            _msg("user", _dup_content("big")),
            _msg("user", _dup_content("big"), metadata=meta),
        ]
        view, stubbed = dedup.deduplicate_history_view(messages)
        assert stubbed == 1
        assert view[1].metadata == meta, "compaction scoring keys on metadata"


# ── F9/F10/F11: sticky selection ───────────────────────────────────────


def _scored_assembler() -> TurnBoundaryContextAssembler:
    def score_fn(messages, current_query=None):
        return [(m, 1.0) for m in messages]

    return TurnBoundaryContextAssembler(score_fn=score_fn)


def _long_history() -> list[Message]:
    # > full_turn_count (3) genuine user turns so the selection path is active
    history = [_msg("system", "sys")]
    for i in range(5):
        history.append(_msg("user", f"turn {i} task"))
        history.append(_msg("assistant", f"work {i}"))
        history.append(_msg("tool", f"TOOL_OUTPUT t{i}", tool_call_id=f"c{i}"))
    return history


class TestStickySelection:
    def test_growing_recent_tail_does_not_drop_selected_older(self):
        assembler = _scored_assembler()
        history = _long_history()

        first = assembler.assemble(history, 100000)
        older_before = [m.content for m in first if m.content.startswith(("turn", "work"))]
        assert older_before, "selection path must include older messages"

        # Intra-loop: recent tail grows massively (budget pressure)
        for j in range(8):
            history.append(_msg("tool", "big " + "x" * 400, tool_call_id=f"g{j}"))

        second = assembler.assemble(history, 100000)
        older_after = [m.content for m in second if m.content.startswith(("turn", "work"))]
        assert older_after == older_before, "intra-loop budget pressure dropped selected older"

    def test_payload_prefix_extends_intra_loop(self):
        from victor.agent.kv_prefix_observability import serialize_payload

        assembler = _scored_assembler()
        history = _long_history()
        first = serialize_payload(assembler.assemble(history, 100000))
        history.append(_msg("tool", "more output", tool_call_id="g0"))
        second = serialize_payload(assembler.assemble(history, 100000))
        assert second.startswith(first)

    def test_new_turn_recomputes_selection(self):
        assembler = _scored_assembler()
        history = _long_history()
        assembler.assemble(history, 100000)
        frozen_at_first_turn = assembler._frozen_selection
        assert frozen_at_first_turn is not None

        history.append(_msg("user", "brand new turn"))
        history.append(_msg("assistant", "reply"))
        assembler.assemble(history, 100000)
        # Anchor changed → the selection recomputed (new freeze object with a
        # new anchor) instead of reusing the intra-turn frozen set.
        assert assembler._frozen_selection is not frozen_at_first_turn
        assert assembler._frozen_selection[0] != frozen_at_first_turn[0]

    def test_membership_miss_after_drops_recomputes(self):
        assembler = _scored_assembler()
        history = _long_history()
        assembler.assemble(history, 100000)
        frozen_ids = list(assembler._frozen_selection[1])
        assert frozen_ids
        # Simulate compaction dropping a selected message from history
        target = next(m for m in history if id(m) == frozen_ids[0])
        history.remove(target)
        assembler.assemble(history, 100000)
        assert frozen_ids[0] not in assembler._frozen_selection[1]


# ── mid_user_turn helper ───────────────────────────────────────────────


class TestMidUserTurn:
    def test_false_at_turn_boundary(self):
        history = [_msg("user", "task"), _msg("assistant", "done")]
        assert mid_user_turn(history) is True  # assistant after anchor
        history2 = [_msg("user", "task")]
        assert mid_user_turn(history2) is False

    def test_hidden_user_messages_do_not_end_turn(self):
        history = [
            _msg("user", "task"),
            _msg("assistant", "work"),
            _msg(
                "user",
                "[hint] adjust args",
                metadata=build_internal_history_metadata(kind="failure_hint"),
            ),
        ]
        assert mid_user_turn(history) is True  # still inside the turn

    def test_no_genuine_user_message(self):
        assert mid_user_turn([_msg("assistant", "hi")]) is False


# ── F13: max_history unification ───────────────────────────────────────


class TestMaxHistoryUnification:
    def test_all_defaults_agree(self):
        assert MessageHistory()._max_history == DEFAULT_MAX_CONVERSATION_HISTORY
        assert Settings().max_conversation_history == DEFAULT_MAX_CONVERSATION_HISTORY

    def test_round_trip_preserves_custom_max(self):
        history = MessageHistory(max_history_messages=555)
        history.add_user_message("hi")
        restored = MessageHistory.from_dict(history.to_dict())
        assert restored._max_history == 555

    def test_legacy_dict_without_key_uses_constant(self):
        restored = MessageHistory.from_dict(
            {"system_prompt": "s", "system_added": True, "messages": []}
        )
        assert restored._max_history == DEFAULT_MAX_CONVERSATION_HISTORY

    def test_trim_logs_sanctioned_boundary(self, caplog):
        history = MessageHistory(max_history_messages=4)
        history.add_user_message("u1")
        history.add_message("assistant", "a1")
        history.add_message("assistant", "a2")
        history.add_message("assistant", "a3")
        with caplog.at_level(logging.INFO, logger="victor.agent.message_history"):
            history.add_message("assistant", "a4")  # len 5 > cap 4 → trim
        assert any("[history-trim] sanctioned boundary" in r.message for r in caplog.records)
