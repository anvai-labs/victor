"""text_call_tools: the sticky-offer rule for calls that must answer in text."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from victor.agent.services.tool_selection_runtime import text_call_tools
from victor.agent.tool_selection import stable_definitions

TOOL_HISTORY = [
    {"role": "user", "content": "find the parser"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "grep"}]},
    {"role": "tool", "content": "parser.py", "tool_call_id": "c1"},
]
PLAIN_HISTORY = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
ALL = ["all-a", "all-b"]
CURATED = ["curated-a"]


@pytest.fixture(autouse=True)
def _stable(monkeypatch):
    monkeypatch.setattr(stable_definitions, "stable_all_definitions", lambda selector: ALL)
    monkeypatch.setattr(stable_definitions, "stable_curated_definitions", lambda selector: CURATED)


def _host(*, supports=True, model_supports=True, curated=None):
    return SimpleNamespace(
        provider=SimpleNamespace(supports_tools=lambda: supports),
        _model_supports_tool_calls=lambda: model_supports,
        tool_selector=SimpleNamespace(_enabled_tools=curated),
    )


def test_tool_history_gets_the_full_stable_offer():
    assert text_call_tools(_host(), TOOL_HISTORY) == ALL


def test_curated_session_gets_its_curated_offer():
    assert text_call_tools(_host(curated={"read"}), TOOL_HISTORY) == CURATED


@pytest.mark.parametrize("history", [PLAIN_HISTORY, [], None])
def test_tool_free_history_keeps_the_tool_free_request(history):
    assert text_call_tools(_host(), history) is None


@pytest.mark.parametrize(
    "host",
    [
        _host(supports=False),
        _host(model_supports=False),
        None,
        SimpleNamespace(provider=None),
    ],
    ids=["provider-no-tools", "model-no-tools", "no-host", "no-provider"],
)
def test_hosts_that_cannot_offer_tools_send_none(host):
    assert text_call_tools(host, TOOL_HISTORY) is None


def test_failure_degrades_to_none():
    host = _host()
    host.provider = SimpleNamespace(supports_tools=MagicMock(side_effect=RuntimeError("x")))
    assert text_call_tools(host, TOOL_HISTORY) is None


def test_empty_stable_supply_is_none_not_an_empty_list(monkeypatch):
    monkeypatch.setattr(stable_definitions, "stable_all_definitions", lambda selector: [])
    assert text_call_tools(_host(), TOOL_HISTORY) is None
