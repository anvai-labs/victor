"""The runtime ConversationStore satisfies the storage adapters' port."""

from __future__ import annotations

import inspect

import pytest

from victor.agent.conversation.store import ConversationStore
from victor.storage.memory.adapters import ConversationHistoryStore

_MEMBERS = [
    name
    for name, value in vars(ConversationHistoryStore).items()
    if inspect.isfunction(value) and not name.startswith("_")
]


def _mismatches(impl: type) -> list[str]:
    problems = []
    for name in _MEMBERS:
        port = inspect.signature(getattr(ConversationHistoryStore, name))
        actual = getattr(impl, name, None)
        if actual is None:
            problems.append(f"{name}: missing")
            continue
        if list(inspect.signature(actual).parameters) != list(port.parameters):
            problems.append(f"{name}: parameters differ")
        if inspect.iscoroutinefunction(actual) != inspect.iscoroutinefunction(
            getattr(ConversationHistoryStore, name)
        ):
            problems.append(f"{name}: sync/async differs")
    return problems


def test_port_covers_every_store_method_the_adapters_call():
    assert set(_MEMBERS) == {
        "add_message",
        "get_recent_messages",
        "get_semantically_relevant_messages",
        "aget_semantically_relevant_messages",
        "aget_dual_trace_relevant_messages",
        "get_message_trace_text",
        "get_historical_tool_results",
    }


def test_concrete_store_matches_the_port():
    assert _mismatches(ConversationStore) == []


@pytest.mark.parametrize(
    "drift",
    ["missing", "renamed_param", "sync_async"],
)
def test_port_check_rejects_drifted_implementations(drift):
    class Drifted(ConversationStore):
        pass

    if drift == "missing":
        Drifted.get_recent_messages = None  # type: ignore[assignment]
    elif drift == "renamed_param":

        def get_recent_messages(self, sid, count=10):  # noqa: ANN001
            return []

        Drifted.get_recent_messages = get_recent_messages  # type: ignore[assignment]
    else:

        async def as_async(self, message, trace_kind=None):  # noqa: ANN001
            return ""

        Drifted.get_message_trace_text = as_async  # type: ignore[assignment]
    assert _mismatches(Drifted)
