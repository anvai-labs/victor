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

"""Tests for first-class ReflectionFormation wiring (multi-round critique)."""

import types

import pytest

from victor.coordination.formations import (
    HierarchicalFormation,
    ReflectionFormation,
    SequentialFormation,
)
from victor.coordination.formations.base import TeamContext
from victor.framework.teams import AgentTeam, TeamFormation, TeamMemberSpec
from victor.teams.types import AgentMessage, MessageType
from victor.teams.unified_coordinator import (
    UnifiedTeamCoordinator,
    _MemberContextAgent,
    _member_formation_role,
)

# -- fakes ------------------------------------------------------------------


class _FakeMember:
    """Minimal ITeamMember: id + formation_role + execute_task."""

    def __init__(self, id_, formation_role=None, ret="output"):
        self.id = id_
        self.formation_role = formation_role
        self._ret = ret
        self.calls = 0

    async def execute_task(self, task, context):
        self.calls += 1
        return self._ret


def _coord():
    return UnifiedTeamCoordinator(types.SimpleNamespace(), lightweight_mode=True)


# -- enum + dispatch + capability flag --------------------------------------


def test_reflection_enum_and_dispatch_registered():
    assert TeamFormation.REFLECTION.value == "reflection"
    assert TeamFormation.REFLECTION in _coord()._formations


def test_consumes_context_agents_flag():
    assert ReflectionFormation().consumes_context_agents() is True
    assert SequentialFormation().consumes_context_agents() is False
    # Regression guard: HIERARCHICAL declares required roles but must NOT be
    # treated as a context-agent consumer (it uses the members list).
    assert HierarchicalFormation().consumes_context_agents() is False


# -- _MemberContextAgent shim normalization ---------------------------------


async def test_shim_normalizes_mapping_output():
    agent = _MemberContextAgent(_FakeMember("m", ret={"output": "X"}))
    assert await agent.execute("prompt") == "X"


async def test_shim_normalizes_string_and_none():
    assert await _MemberContextAgent(_FakeMember("m", ret="hi")).execute("p") == "hi"
    assert await _MemberContextAgent(_FakeMember("m", ret=None)).execute("p") == ""


async def test_shim_calls_execute_task():
    member = _FakeMember("m", ret="done")
    await _MemberContextAgent(member).execute("p", {"k": "v"})
    assert member.calls == 1


def test_member_formation_role_reads_through_adapter():
    raw = _FakeMember("m", formation_role="critic")
    assert _member_formation_role(raw) == "critic"
    # Through an adapter exposing `.member`
    adapter = types.SimpleNamespace(member=raw)
    assert _member_formation_role(adapter) == "critic"
    assert _member_formation_role(_FakeMember("m")) is None


# -- _populate_context_agents binding ---------------------------------------


def test_populate_binds_by_role_ignoring_order():
    gen = _FakeMember("g", "generator")
    crit = _FakeMember("c", "critic")
    # Pass reversed; role binding must still map correctly.
    agents = _coord()._populate_context_agents(ReflectionFormation(), [crit, gen])
    assert set(agents) == {"generator", "critic"}
    assert agents["generator"]._member is gen
    assert agents["critic"]._member is crit


def test_populate_positional_fallback_when_role_unset():
    m0 = _FakeMember("0")
    m1 = _FakeMember("1")
    agents = _coord()._populate_context_agents(ReflectionFormation(), [m0, m1])
    assert agents["generator"]._member is m0
    assert agents["critic"]._member is m1


def test_populate_noop_for_non_consuming_formation():
    members = [_FakeMember("0"), _FakeMember("1")]
    assert _coord()._populate_context_agents(SequentialFormation(), members) == {}


def test_populate_partial_when_too_few_members():
    agents = _coord()._populate_context_agents(ReflectionFormation(), [_FakeMember("0")])
    assert "generator" in agents
    assert "critic" not in agents


# -- rounds override (context-driven) ---------------------------------------


async def test_rounds_override_limits_iterations():
    # Critic never satisfied -> loop runs until the bound.
    gen = _FakeMember("g", "generator", ret="solution")
    crit = _FakeMember("c", "critic", ret="needs work, not done")
    # Wrap as the coordinator does, so the formation's .execute(str, ctx) works.
    ctx = TeamContext(
        team_id="t",
        formation="reflection",
        shared_state={
            "generator": _MemberContextAgent(gen),
            "critic": _MemberContextAgent(crit),
            "reflection_max_iterations": 1,
        },
    )
    task = AgentMessage(sender_id="t", content="do it", message_type=MessageType.TASK)

    results = await ReflectionFormation(max_iterations=5).execute([], ctx, task)

    assert gen.calls == 1  # capped at 1 despite instance max_iterations=5
    assert crit.calls == 1
    assert results[0].metadata["iterations"] == 1


async def test_defaults_to_instance_max_iterations_without_override():
    gen = _FakeMember("g", "generator", ret="solution")
    crit = _FakeMember("c", "critic", ret="needs work")
    ctx = TeamContext(
        team_id="t",
        formation="reflection",
        shared_state={  # no override
            "generator": _MemberContextAgent(gen),
            "critic": _MemberContextAgent(crit),
        },
    )
    task = AgentMessage(sender_id="t", content="do it", message_type=MessageType.TASK)

    await ReflectionFormation(max_iterations=2).execute([], ctx, task)
    assert gen.calls == 2


# -- create_reflection_team preset ------------------------------------------


@pytest.fixture
def _stub_coordinator(monkeypatch):
    monkeypatch.setattr(
        "victor.framework.teams.create_coordinator",
        lambda orchestrator, **kw: types.SimpleNamespace(),
    )


async def test_create_reflection_team_config(_stub_coordinator):
    team = await AgentTeam.create_reflection_team(
        orchestrator=types.SimpleNamespace(),
        name="Refine",
        goal="ship a correct change",
        generator=TeamMemberSpec(role="executor", goal="write", provider="openai"),
        critic=TeamMemberSpec(role="reviewer", goal="review", provider="anthropic"),
        rounds=2,
    )
    cfg = team._config
    assert cfg.formation == TeamFormation.REFLECTION
    assert cfg.shared_context["reflection_max_iterations"] == 2
    assert len(cfg.members) == 2
    assert cfg.members[0].formation_role == "generator"
    assert cfg.members[1].formation_role == "critic"


async def test_create_reflection_team_warns_same_vendor(_stub_coordinator, caplog):
    with caplog.at_level("WARNING"):
        await AgentTeam.create_reflection_team(
            orchestrator=types.SimpleNamespace(),
            name="SameVendor",
            goal="goal",
            generator=TeamMemberSpec(role="executor", goal="w", provider="openai"),
            critic=TeamMemberSpec(role="reviewer", goal="r", provider="openai"),
        )
    assert any("share provider" in r.message for r in caplog.records)


# -- satisfaction judging (verdict-preferred, keyword fallback) --------------


def test_verdict_satisfied_wins_over_keyword_absence():
    f = ReflectionFormation()
    # Explicit verdict, no positive keywords at all.
    assert f._is_satisfied("Looks fine overall.\nVERDICT: SATISFIED") is True


def test_verdict_needs_work_overrides_positive_keywords():
    f = ReflectionFormation()
    # Contains "good" but the critic's verdict is NEEDS_WORK -> not satisfied.
    assert f._is_satisfied("This is good in places.\nVERDICT: NEEDS_WORK") is False


def test_verdict_tolerates_markdown_formatting():
    f = ReflectionFormation()
    assert f._is_satisfied("**VERDICT:** SATISFIED") is True
    assert f._is_satisfied("verdict - needs_work") is False


def test_keyword_fallback_when_no_verdict():
    f = ReflectionFormation()
    assert f._is_satisfied("this looks good to me") is True


def test_keyword_fallback_respects_negation():
    f = ReflectionFormation()
    # "not good" / "not acceptable" must NOT read as satisfaction.
    assert f._is_satisfied("this is not good yet") is False
    assert f._is_satisfied("the result is not acceptable") is False


def test_empty_feedback_not_satisfied():
    assert ReflectionFormation()._is_satisfied("") is False
    assert ReflectionFormation()._is_satisfied(None) is False


# -- richer critique prompt (includes original task + verdict request) -------


async def test_critique_prompt_includes_original_task_and_verdict_request():
    captured = {}

    class _RecordingCritic(_FakeMember):
        async def execute(self, prompt, context=None):  # context-agent shim API
            captured["prompt"] = prompt
            return "VERDICT: SATISFIED"

    class _Gen(_FakeMember):
        async def execute(self, prompt, context=None):
            self.calls += 1
            return "the solution"

    gen = _Gen("g", "generator")
    crit = _RecordingCritic("c", "critic")
    ctx = TeamContext(
        team_id="t",
        formation="reflection",
        shared_state={"generator": gen, "critic": crit},
    )
    task = AgentMessage(
        sender_id="t",
        content="Implement feature X correctly",
        message_type=MessageType.TASK,
    )

    results = await ReflectionFormation(max_iterations=3).execute([], ctx, task)

    # Original task surfaced to the critic, and a verdict line was requested.
    assert "Implement feature X correctly" in captured["prompt"]
    assert "VERDICT" in captured["prompt"]
    # SATISFIED verdict -> early termination after one round.
    assert gen.calls == 1
    assert results[0].metadata["satisfied"] is True
