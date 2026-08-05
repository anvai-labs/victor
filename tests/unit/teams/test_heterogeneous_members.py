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

"""Tests for per-member provider/model selection (heterogeneous teams)."""

import types

from victor.agent.subagents import orchestrator as orch_mod
from victor.agent.subagents.base import SubAgentConfig
from victor.agent.subagents.orchestrator import SubAgentOrchestrator
from victor.core.shared_types import SubAgentRole
from victor.framework.teams import TeamMemberSpec
from victor.teams.unified_coordinator import UnifiedTeamCoordinator

# -- spec -> member passthrough --------------------------------------------


def test_spec_passes_provider_model_to_member():
    spec = TeamMemberSpec(
        role="reviewer", goal="review", provider="anthropic", model="claude-opus-4-8"
    )
    member = spec.to_team_member()
    assert member.provider == "anthropic"
    assert member.model == "claude-opus-4-8"


def test_spec_defaults_provider_model_none():
    member = TeamMemberSpec(role="executor", goal="do").to_team_member()
    assert member.provider is None
    assert member.model is None
    assert member.temperature is None


def test_spec_passes_temperature_to_member():
    member = TeamMemberSpec(role="reviewer", goal="review", temperature=0.0).to_team_member()
    assert member.temperature == 0.0


# -- _resolve_override_provider --------------------------------------------


async def test_resolve_override_provider_success(monkeypatch):
    sentinel = object()
    created = {}

    class _FakeFactory:
        @staticmethod
        async def create(*, provider_name, model, api_key):
            created["args"] = (provider_name, model, api_key)
            return sentinel

    monkeypatch.setattr(orch_mod, "ManagedProviderFactory", _FakeFactory, raising=False)
    # ManagedProviderFactory is imported inside the method; patch the source module too.
    import victor.providers.factory as factory_mod

    monkeypatch.setattr(factory_mod, "ManagedProviderFactory", _FakeFactory)
    monkeypatch.setattr("victor.config.api_keys.get_api_key", lambda p: "KEY")

    so = SubAgentOrchestrator(types.SimpleNamespace(model="parent-model"))
    result = await so._resolve_override_provider("openai", "gpt-5")
    assert result is sentinel
    assert created["args"] == ("openai", "gpt-5", "KEY")


async def test_resolve_override_provider_fail_open(monkeypatch, caplog):
    import victor.providers.factory as factory_mod

    class _BoomFactory:
        @staticmethod
        async def create(**kwargs):
            raise RuntimeError("no credentials")

    monkeypatch.setattr(factory_mod, "ManagedProviderFactory", _BoomFactory)
    monkeypatch.setattr("victor.config.api_keys.get_api_key", lambda p: None)

    so = SubAgentOrchestrator(types.SimpleNamespace(model="parent-model"))
    with caplog.at_level("WARNING"):
        result = await so._resolve_override_provider("openai", None)
    assert result is None
    assert any("could not be resolved" in r.message for r in caplog.records)


# -- spawn() threads the override into SubAgentConfig ----------------------


async def _spawn_capturing_config(
    monkeypatch, *, provider, model, resolved, temperature=None, reasoning_effort=None
):
    """Run spawn() with SubAgent + resolution stubbed; return the captured config."""
    captured = {}

    class _FakeSubAgent:
        def __init__(self, config, parent, **kwargs):
            captured["config"] = config

        async def execute(self):
            return orch_mod.SubAgentResult(
                success=True,
                summary="ok",
                details={},
                tool_calls_used=0,
                context_size=0,
                duration_seconds=0.0,
            )

    monkeypatch.setattr(orch_mod, "SubAgent", _FakeSubAgent)

    so = SubAgentOrchestrator(
        types.SimpleNamespace(model="parent-model", provider_name="anthropic")
    )

    async def _fake_resolve(p, m):
        return resolved

    monkeypatch.setattr(so, "_resolve_override_provider", _fake_resolve)

    await so.spawn(
        SubAgentRole.REVIEWER,
        "task",
        provider=provider,
        model=model,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )
    return captured["config"]


async def test_spawn_sets_override_when_provider_given(monkeypatch):
    sentinel = object()
    config = await _spawn_capturing_config(
        monkeypatch, provider="openai", model="gpt-5", resolved=sentinel
    )
    assert config.provider_override is sentinel
    assert config.model_override == "gpt-5"


async def test_spawn_no_override_when_provider_absent(monkeypatch):
    config = await _spawn_capturing_config(monkeypatch, provider=None, model=None, resolved=None)
    assert config.provider_override is None
    assert config.model_override is None


async def test_spawn_fail_open_keeps_override_none(monkeypatch):
    # Resolution returns None (factory failed) -> config carries no override.
    config = await _spawn_capturing_config(
        monkeypatch, provider="openai", model="gpt-5", resolved=None
    )
    assert config.provider_override is None
    assert config.model_override is None


async def test_spawn_threads_temperature_override(monkeypatch):
    config = await _spawn_capturing_config(
        monkeypatch, provider=None, model=None, resolved=None, temperature=0.0
    )
    assert config.temperature_override == 0.0


# -- coordinator forwards member.provider/model to spawn -------------------


async def test_make_executor_forwards_provider_model(monkeypatch):
    recorded = {}

    async def _recording_spawn(self, role, task, **kwargs):
        recorded.update(kwargs)
        return types.SimpleNamespace(
            success=True,
            summary="done",
            error=None,
            tool_calls_used=0,
            duration_seconds=0.1,
        )

    monkeypatch.setattr(SubAgentOrchestrator, "spawn", _recording_spawn)

    coord = UnifiedTeamCoordinator(types.SimpleNamespace(), lightweight_mode=True)
    member = TeamMemberSpec(
        role="reviewer",
        goal="review",
        provider="anthropic",
        model="claude-opus-4-8",
        temperature=0.0,
    ).to_team_member()

    adapters = coord._adapt_team_members([member])
    await adapters[0].execute_task("review this", {})

    assert recorded.get("provider") == "anthropic"
    assert recorded.get("model") == "claude-opus-4-8"
    assert recorded.get("temperature") == 0.0


# -- _create_constrained_orchestrator uses the override --------------------


def test_constrained_orchestrator_uses_override_provider(monkeypatch):
    from victor.agent.subagents.base import SubAgent

    captured = {}

    class _FakeRegistry:
        def clear(self):
            pass

        def register(self, tool):
            pass

        def get(self, name):
            return None

    class _FakeOrchestrator:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.tool_registry = _FakeRegistry()
            self._vertical_context = None

    import victor.agent.orchestrator as orch_module

    monkeypatch.setattr(orch_module, "AgentOrchestrator", _FakeOrchestrator)

    override_provider = object()
    config = SubAgentConfig(
        role=SubAgentRole.REVIEWER,
        task="t",
        allowed_tools=[],
        tool_budget=5,
        context_limit=1000,
        system_prompt_override="ROLE PROMPT",
        provider_override=override_provider,
        model_override="claude-opus-4-8",
    )

    sa = SubAgent.__new__(SubAgent)
    sa.config = config
    sa._context = types.SimpleNamespace(
        settings=types.SimpleNamespace(tool_budget=1, max_context_chars=1),
        provider=object(),  # parent provider (should NOT be used)
        model="parent-model",
        temperature=0.7,
        provider_name="anthropic",
        vertical_context=None,
        tool_registry=_FakeRegistry(),
    )

    sa._create_constrained_orchestrator()

    assert captured["provider"] is override_provider
    assert captured["model"] == "claude-opus-4-8"


def test_constrained_orchestrator_inherits_when_no_override(monkeypatch):
    from victor.agent.subagents.base import SubAgent

    captured = {}

    class _FakeRegistry:
        def clear(self):
            pass

        def register(self, tool):
            pass

        def get(self, name):
            return None

    class _FakeOrchestrator:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.tool_registry = _FakeRegistry()
            self._vertical_context = None

    import victor.agent.orchestrator as orch_module

    monkeypatch.setattr(orch_module, "AgentOrchestrator", _FakeOrchestrator)

    parent_provider = object()
    config = SubAgentConfig(
        role=SubAgentRole.EXECUTOR,
        task="t",
        allowed_tools=[],
        tool_budget=5,
        context_limit=1000,
        system_prompt_override="ROLE PROMPT",
    )

    sa = SubAgent.__new__(SubAgent)
    sa.config = config
    sa._context = types.SimpleNamespace(
        settings=types.SimpleNamespace(tool_budget=1, max_context_chars=1),
        provider=parent_provider,
        model="parent-model",
        temperature=0.7,
        provider_name="anthropic",
        vertical_context=None,
        tool_registry=_FakeRegistry(),
    )

    sa._create_constrained_orchestrator()

    assert captured["provider"] is parent_provider
    assert captured["model"] == "parent-model"
    # No temperature override -> inherits the parent's temperature.
    assert captured["temperature"] == 0.7


def test_constrained_orchestrator_uses_temperature_override(monkeypatch):
    from victor.agent.subagents.base import SubAgent

    captured = {}

    class _FakeRegistry:
        def clear(self):
            pass

        def register(self, tool):
            pass

        def get(self, name):
            return None

    class _FakeOrchestrator:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.tool_registry = _FakeRegistry()
            self._vertical_context = None

    import victor.agent.orchestrator as orch_module

    monkeypatch.setattr(orch_module, "AgentOrchestrator", _FakeOrchestrator)

    config = SubAgentConfig(
        role=SubAgentRole.REVIEWER,
        task="t",
        allowed_tools=[],
        tool_budget=5,
        context_limit=1000,
        system_prompt_override="ROLE PROMPT",
        temperature_override=0.0,  # deterministic reviewer
    )

    sa = SubAgent.__new__(SubAgent)
    sa.config = config
    sa._context = types.SimpleNamespace(
        settings=types.SimpleNamespace(tool_budget=1, max_context_chars=1),
        provider=object(),
        model="parent-model",
        temperature=0.7,  # parent default (should be overridden)
        provider_name="anthropic",
        vertical_context=None,
        tool_registry=_FakeRegistry(),
    )

    sa._create_constrained_orchestrator()

    # temperature_override=0.0 must win over the parent's 0.7 (not be treated as falsy).
    assert captured["temperature"] == 0.0


# -- reasoning_effort threads the same chain as temperature -----------------


def test_spec_passes_reasoning_effort_to_member():
    member = TeamMemberSpec(
        role="reviewer", goal="review", reasoning_effort="high"
    ).to_team_member()
    assert member.reasoning_effort == "high"


def test_spec_defaults_reasoning_effort_none():
    member = TeamMemberSpec(role="executor", goal="write").to_team_member()
    assert member.reasoning_effort is None


async def test_spawn_threads_reasoning_effort_override(monkeypatch):
    config = await _spawn_capturing_config(
        monkeypatch, provider=None, model=None, resolved=None, reasoning_effort="high"
    )
    assert config.reasoning_effort_override == "high"


async def test_spawn_no_reasoning_effort_override_by_default(monkeypatch):
    config = await _spawn_capturing_config(monkeypatch, provider=None, model=None, resolved=None)
    assert config.reasoning_effort_override is None


async def test_make_executor_forwards_reasoning_effort(monkeypatch):
    recorded = {}

    async def _recording_spawn(self, role, task, **kwargs):
        recorded.update(kwargs)
        return types.SimpleNamespace(
            success=True,
            summary="done",
            error=None,
            tool_calls_used=0,
            duration_seconds=0.1,
        )

    monkeypatch.setattr(SubAgentOrchestrator, "spawn", _recording_spawn)

    coord = UnifiedTeamCoordinator(types.SimpleNamespace(), lightweight_mode=True)
    member = TeamMemberSpec(
        role="reviewer", goal="review", reasoning_effort="high"
    ).to_team_member()

    adapters = coord._adapt_team_members([member])
    await adapters[0].execute_task("review this", {})

    assert recorded.get("reasoning_effort") == "high"


def _make_constrained(monkeypatch, *, config, context_reasoning=None):
    """Build a SubAgent with a fake AgentOrchestrator; return captured init kwargs."""
    from victor.agent.subagents.base import SubAgent

    captured = {}

    class _FakeRegistry:
        def clear(self):
            pass

        def register(self, tool):
            pass

        def get(self, name):
            return None

    class _FakeOrchestrator:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.tool_registry = _FakeRegistry()
            self._vertical_context = None

    import victor.agent.orchestrator as orch_module

    monkeypatch.setattr(orch_module, "AgentOrchestrator", _FakeOrchestrator)

    sa = SubAgent.__new__(SubAgent)
    sa.config = config
    sa._context = types.SimpleNamespace(
        settings=types.SimpleNamespace(tool_budget=1, max_context_chars=1),
        provider=object(),
        model="parent-model",
        temperature=0.7,
        reasoning_effort=context_reasoning,
        provider_name="anthropic",
        vertical_context=None,
        tool_registry=_FakeRegistry(),
    )
    sa._create_constrained_orchestrator()
    return captured


def test_constrained_orchestrator_uses_reasoning_effort_override(monkeypatch):
    config = SubAgentConfig(
        role=SubAgentRole.REVIEWER,
        task="t",
        allowed_tools=[],
        tool_budget=5,
        context_limit=1000,
        system_prompt_override="ROLE PROMPT",
        reasoning_effort_override="high",
    )
    captured = _make_constrained(monkeypatch, config=config, context_reasoning="low")
    # Override wins over the parent context value.
    assert captured["reasoning_effort"] == "high"


def test_constrained_orchestrator_inherits_reasoning_effort(monkeypatch):
    config = SubAgentConfig(
        role=SubAgentRole.EXECUTOR,
        task="t",
        allowed_tools=[],
        tool_budget=5,
        context_limit=1000,
        system_prompt_override="ROLE PROMPT",
    )
    captured = _make_constrained(monkeypatch, config=config, context_reasoning="medium")
    # No override -> inherits the parent's reasoning_effort.
    assert captured["reasoning_effort"] == "medium"
