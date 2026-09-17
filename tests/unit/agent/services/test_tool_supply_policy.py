"""Real ToolService policy gates resolve opt-in pruning at each invocation."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from victor.agent.services.tool_service import ToolService, ToolServiceConfig
from victor.agent.services.tool_supply_policy import pruning_disabled_for
from victor.tools.enums import Priority


@pytest.fixture(params=["semantic", "context"])
def policy_call(request, monkeypatch):
    # Fix only the provider economics and token estimator; both public service
    # methods and their actual policy gates run unchanged.
    monkeypatch.setattr(
        "victor.config.tool_tiers.resolve_tool_supply_profile",
        lambda *_args, **_kwargs: SimpleNamespace(
            budget_tokens=100, cap_mode="hard", session_lock="none"
        ),
    )
    tools = [SimpleNamespace(name=name, priority=Priority.HIGH) for name in ("a", "b")]
    provider = SimpleNamespace(
        context_window=lambda _model: 8192, supports_prompt_caching=lambda: False
    )

    def invoke(service):
        monkeypatch.setattr(service, "estimate_tool_tokens", lambda *_args, **_kwargs: 60)
        if request.param == "semantic":
            return service.semantic_select_tools(tools, 100)
        return service.apply_context_aware_strategy(tools, provider=provider, model="test")

    return invoke


def _service(override=None, settings_enabled=False):
    return ToolService(
        ToolServiceConfig(tool_selection_enabled=override),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        settings=SimpleNamespace(tools=SimpleNamespace(tool_selection_enabled=settings_enabled)),
    )


@pytest.mark.parametrize(
    "override,settings_enabled,env,expected_count",
    [
        (None, False, "", 2),
        (None, True, "", 1),
        (None, False, "1", 1),
        (None, True, "0", 1),
        (False, True, "1", 2),
        (True, False, "0", 1),
    ],
)
def test_policy_gates_honor_canonical_precedence(
    policy_call, monkeypatch, override, settings_enabled, env, expected_count
):
    monkeypatch.setenv("VICTOR_TOOL_SELECTION", env)
    service = _service(override, settings_enabled)
    assert len(policy_call(service)) == expected_count
    assert service._config.tool_selection_enabled is override


@pytest.mark.parametrize("toggle", ["settings", "environment"])
def test_policy_gates_resolve_changes_after_service_initialization(
    policy_call, monkeypatch, toggle
):
    monkeypatch.delenv("VICTOR_TOOL_SELECTION", raising=False)
    service = _service()
    assert len(policy_call(service)) == 2
    if toggle == "settings":
        service._settings.tools.tool_selection_enabled = True
    else:
        monkeypatch.setenv("VICTOR_TOOL_SELECTION", "1")
    assert len(policy_call(service)) == 1
    if toggle == "settings":
        service._settings.tools.tool_selection_enabled = False
    else:
        monkeypatch.setenv("VICTOR_TOOL_SELECTION", "0")
    assert len(policy_call(service)) == 2
    assert service._config.tool_selection_enabled is None


def test_missing_config_preserves_legacy_service_behavior(monkeypatch):
    monkeypatch.delenv("VICTOR_TOOL_SELECTION", raising=False)
    assert pruning_disabled_for(None) is False
    assert pruning_disabled_for(SimpleNamespace()) is False
