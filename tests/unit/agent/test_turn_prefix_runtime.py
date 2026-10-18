"""KV prefix stability: per-user-turn turn-prefix freeze (F8 fix).

The composed turn prefix was re-rendered for every intra-loop provider call
and prepended to the anchor (last genuine user) message — the same message
resent each call — so any dynamic guidance change mutated already-sent bytes.
These tests pin: intra-loop prefix bytes frozen; mid-loop failure signals
delivered as appended hidden messages; new user turns recompose.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from victor.agent.conversation.history_metadata import (
    build_internal_history_metadata,
    is_hidden_from_interactive_history,
    is_user_turn,
)
from victor.agent.kv_prefix_observability import serialize_payload
from victor.providers.base import Message


def _load_guard_module():
    """Reuse the invariant guard's orchestrator scaffold (same wiring)."""
    path = Path(__file__).parent.parent / "framework" / "test_kv_prefix_invariant.py"
    spec = importlib.util.spec_from_file_location("kv_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed(orch) -> None:
    orch.conversation.add_user_message("read src/a.py and summarize its imports")
    orch.conversation.add_message("assistant", "reading")
    orch.conversation.add_message("tool", "TOOL_OUTPUT src/a.py: 10 lines", tool_call_id="c0")


def _anchor_content(orch) -> str:
    assembled = orch.get_assembled_messages(current_query="read src/a.py")
    for message in reversed(assembled):
        if is_user_turn(message.role, message.metadata):
            return message.content
    raise AssertionError("no genuine user message in assembled payload")


def _hidden_failure_hints(orch) -> list[Message]:
    return [
        m
        for m in orch.conversation.messages
        if m.role == "user"
        and is_hidden_from_interactive_history(m.metadata)
        and m.metadata.get("internal_prompt_kind") == "failure_hint"
    ]


async def test_turn_prefix_frozen_intra_loop():
    guard = _load_guard_module()
    orch = guard._make_orchestrator()
    _seed(orch)

    first = _anchor_content(orch)
    # Mid-loop activity: more tool exchange + a failure signal raised after a
    # failed iteration (previously re-rendered into the prefix).
    orch.conversation.add_message("assistant", "plan B")
    orch.conversation.add_message("tool", "TOOL_OUTPUT error: file not found", tool_call_id="c1")
    injector = orch._optimization_injector
    injector._last_failure_category = "tool_error"
    injector._last_failure_error = "file not found"

    second = _anchor_content(orch)

    assert second == first, "intra-loop turn-prefix re-render mutates sent bytes"
    # The failure signal was delivered as an appended hidden message instead.
    hints = _hidden_failure_hints(orch)
    assert len(hints) == 1
    assert hints[0].content  # non-empty corrective hint
    assert injector._last_failure_category is None


async def test_intra_loop_payloads_extend_despite_failure_signals():
    guard = _load_guard_module()
    orch = guard._make_orchestrator()
    _seed(orch)

    first = serialize_payload(orch.get_assembled_messages(current_query="task"))
    orch.conversation.add_message("assistant", "step")
    orch.conversation.add_message("tool", "TOOL_OUTPUT ok", tool_call_id="c2")
    injector = orch._optimization_injector
    injector._last_failure_category = "tool_error"
    injector._last_failure_error = "boom"
    second = serialize_payload(orch.get_assembled_messages(current_query="task"))

    assert second.startswith(first)
    # The appended hint must be visible in the payload tail (not a re-render).
    orch.conversation.add_message("assistant", "retry")
    third = serialize_payload(orch.get_assembled_messages(current_query="task"))
    assert third.startswith(second)


async def test_new_user_turn_recomposes_prefix():
    guard = _load_guard_module()
    orch = guard._make_orchestrator()
    _seed(orch)
    first = _anchor_content(orch)

    orch.conversation.add_user_message("now edit src/a.py to add a docstring")
    second = _anchor_content(orch)

    assert second != first, "boundary must recompose (fresh dynamic guidance)"
