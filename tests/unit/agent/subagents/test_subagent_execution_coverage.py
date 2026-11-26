# Copyright 2026 Vijaykumar Singh <vijay@anaiops.com>
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

"""TDD coverage for the subagent execution path (FOU-A1 arc).

Targets the regions the 197-test subagent suite leaves uncovered — the
parent-visible tool digest, the streaming error terminal, orchestrator
construction overrides, the fan-out exception funnel, and the run-registry
list/emission edges. Every test here pins REAL behavior, not mocks
echoing themselves: inputs are constructed, outputs are asserted against
the documented contracts in the source.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.subagents.base import SubAgent
from victor.agent.subagents.base import SubAgentConfig
from victor.agent.subagents.orchestrator import FanOutResult, SubAgentTask
from victor.agent.subagents.run_registry import SubAgentRunRegistry, SubAgentRunStatus
from victor.agent.subagents import SubAgentRole


def _make_subagent(**config_overrides) -> SubAgent:
    config = SubAgentConfig(
        role=SubAgentRole.RESEARCHER,
        task="test task",
        allowed_tools=["read_file"],
        tool_budget=3,
        context_limit=4000,
        **config_overrides,
    )
    parent = MagicMock()
    subagent = SubAgent(config, parent)
    return subagent


class TestToolResultDigest:
    """`_build_tool_evidence_handoff` — the tool digest the parent consumes."""

    def _subagent_with_messages(self, messages):
        subagent = _make_subagent()
        subagent._get_orchestrator_messages = lambda: messages
        return subagent

    def test_dict_messages_with_explicit_name(self):
        sub = self._subagent_with_messages(
            [
                {"role": "user", "content": "q"},
                {"role": "tool", "name": "read_file", "content": "file body"},
                {"role": "assistant", "content": "thinking"},
            ]
        )
        digest = sub._build_tool_evidence_handoff()
        assert digest["count"] == 1
        assert digest["tool_names"] == ["read_file"]
        assert digest["total_chars"] == len("file body")
        assert digest["summary"].startswith("read_file: file body")

    def test_dict_message_metadata_tool_name_and_name_fallbacks(self):
        sub = self._subagent_with_messages(
            [
                {"role": "tool", "content": "a", "metadata": {"tool_name": "grep"}},
                {"role": "tool", "content": "b", "metadata": {"name": "glob"}},
                {"role": "tool", "content": "c"},
            ]
        )
        digest = sub._build_tool_evidence_handoff()
        # Deduplicated tool name list, bare-unknown falls back to "tool".
        assert digest["tool_names"] == ["grep", "glob", "tool"]
        assert digest["count"] == 3

    def test_object_messages_with_role_enum_and_metadata(self):
        tool_msg = SimpleNamespace(
            role=SimpleNamespace(value="tool"),
            name=None,
            metadata={"tool_name": "bash"},
            content="ran the thing",
        )
        other = SimpleNamespace(
            role=SimpleNamespace(value="assistant"),
            content="not a tool",
        )
        sub = self._subagent_with_messages([tool_msg, other])
        digest = sub._build_tool_evidence_handoff()
        assert digest["count"] == 1
        assert digest["tool_names"] == ["bash"]
        assert "ran the thing" in digest["summary"]

    def test_object_message_name_attribute_wins(self):
        tool_msg = SimpleNamespace(
            role="tool",
            name="search",
            metadata={"tool_name": "ignored"},
            content="hit",
        )
        sub = self._subagent_with_messages([tool_msg])
        assert sub._build_tool_evidence_handoff()["tool_names"] == ["search"]

    def test_blank_content_is_skipped(self):
        sub = self._subagent_with_messages(
            [
                {"role": "tool", "name": "x", "content": "   "},
                {"role": "tool", "name": "y", "content": "\n\t"},
                {"role": "tool", "name": "z", "content": "real"},
            ]
        )
        digest = sub._build_tool_evidence_handoff()
        assert digest["count"] == 1
        assert digest["tool_names"] == ["z"]

    def test_whitespace_collapsed_and_snippet_capped_at_700(self):
        content = "line1\nline2\ttabbed   spaced\n" + "x" * 900
        sub = self._subagent_with_messages([{"role": "tool", "name": "big", "content": content}])
        digest = sub._build_tool_evidence_handoff()
        entry = digest["summary"].split(": ", 1)[1]
        assert "\n" not in entry
        assert len(entry) <= 700
        # total_chars counts the ORIGINAL content, pre-snippet.
        assert digest["total_chars"] == len(content)

    def test_digest_caps_at_24_entries(self):
        messages = [{"role": "tool", "name": f"tool{i}", "content": f"c{i}"} for i in range(30)]
        sub = self._subagent_with_messages(messages)
        digest = sub._build_tool_evidence_handoff()
        assert digest["count"] == 24
        # Order preserved: first 24 tools.
        assert digest["tool_names"][0] == "tool0"
        assert digest["tool_names"][-1] == "tool23"

    def test_empty_history_gives_zeroed_digest(self):
        sub = self._subagent_with_messages([])
        digest = sub._build_tool_evidence_handoff()
        assert digest == {
            "count": 0,
            "tool_names": [],
            "total_chars": 0,
            "summary": "",
        }


class TestStreamExecuteErrorTerminal:
    """The stream's exception path must still emit a final chunk."""

    @pytest.mark.asyncio
    async def test_exception_yields_final_error_chunk_with_orchestrator_stats(self):
        sub = _make_subagent()
        orch = MagicMock()
        orch.tool_calls_used = 7
        orch.get_messages.return_value = ["m1", "m2"]
        sub.orchestrator = orch
        orch.stream_chat = MagicMock(side_effect=RuntimeError("provider exploded"))

        chunks = []
        # stream_execute drives orchestrator.stream() internally; force the
        # failure at that seam regardless of the surrounding setup by
        # pointing the internal walk at a raising callable.
        try:
            async for chunk in sub.stream_execute():
                chunks.append(chunk)
        except RuntimeError:
            pytest.fail("stream_execute must convert exceptions to a final chunk")

        final = chunks[-1]
        assert final.is_final is True
        assert final.content == ""
        assert final.metadata["error_type"] == "RuntimeError"
        assert "provider exploded" in final.metadata["error"]

    @pytest.mark.asyncio
    async def test_get_messages_failure_does_not_mask_stream_error(self):
        sub = _make_subagent()
        orch = MagicMock()
        orch.tool_calls_used = 3
        orch.get_messages.side_effect = ValueError("corrupt history")
        orch.stream_chat = MagicMock(side_effect=RuntimeError("boom"))
        sub.orchestrator = orch

        chunks = [c async for c in sub.stream_execute()]
        final = chunks[-1]
        assert final.is_final is True
        # The context-size fallback degraded to zero, error still surfaced —
        # and it is the ORIGINAL stream error, not the size error.
        assert final.metadata["tool_calls_used"] == 3
        assert final.metadata["error_type"] == "RuntimeError"
        assert "boom" in final.metadata["error"]


class TestCreateOrchestratorOverrides:
    """Constructor override resolution on the real `_create_orchestrator`."""

    def _run_create(self, monkeypatch, config_overrides, context_attrs=None):
        recorded = {}

        class FakeOrchestrator:
            def __init__(self, **kwargs):
                recorded.update(kwargs)
                self._get_session_runtime = MagicMock()
                self._vertical_context = None
                from types import SimpleNamespace as _NS

                self.tool_registry = MagicMock()
                self._session_state_manager = _NS(execution_state=_NS(disable_embeddings=False))
                self._tools_registered = []

        import victor.agent.orchestrator as orch_module

        monkeypatch.setattr(orch_module, "AgentOrchestrator", FakeOrchestrator)

        sub = _make_subagent(**config_overrides)
        # A MagicMock parent fails the runtime protocol check and is wrapped
        # in SubAgentContextAdapter; set inherited fields on the wrapped mock.
        for attr, value in (context_attrs or {}).items():
            setattr(sub._context._orchestrator, attr, value)

        orch = sub._create_constrained_orchestrator()
        return orch, recorded, sub

    def test_override_fields_flow_into_construction(self, monkeypatch):
        provider_override = MagicMock(provider_name="zai")
        _, recorded, _ = self._run_create(
            monkeypatch,
            {
                "provider_override": provider_override,
                "model_override": "glm-5.3",
                "temperature_override": 0.2,
                "reasoning_effort_override": "high",
            },
        )
        assert recorded["provider"] is provider_override
        assert recorded["model"] == "glm-5.3"
        assert recorded["temperature"] == 0.2
        assert recorded["reasoning_effort"] == "high"
        assert recorded["provider_name"] == "zai"

    def test_unset_overrides_inherit_parent_context(self, monkeypatch):
        parent_provider = MagicMock()
        _, recorded, _ = self._run_create(
            monkeypatch,
            {},
            {
                "provider": parent_provider,
                "model": "parent-model",
                "provider_name": "parent",
                "temperature": 0.7,
            },
        )
        assert recorded["provider"] is parent_provider
        assert recorded["model"] == "parent-model"
        assert recorded["provider_name"] == "parent"
        assert recorded["temperature"] == 0.7

    def test_disable_embeddings_sets_execution_flag(self, monkeypatch):
        orch, _, _ = self._run_create(monkeypatch, {"disable_embeddings": True})
        execution_state = orch._session_state_manager.execution_state
        assert execution_state.disable_embeddings is True

    def test_embeddings_left_enabled_by_default(self, monkeypatch):
        orch, _, _ = self._run_create(monkeypatch, {})
        execution_state = orch._session_state_manager.execution_state
        assert not execution_state.disable_embeddings

    def test_role_prompt_becomes_system_prompt_override(self, monkeypatch):
        _, recorded, _ = self._run_create(monkeypatch, {})
        assert recorded["system_prompt_override"]


class TestSpawnManyFanOut:
    """`fan_out` — semaphore fan-out with exception funneling."""

    def _orchestrator_with(self, outcomes):
        from victor.agent.subagents.orchestrator import SubAgentOrchestrator

        orch = SubAgentOrchestrator.__new__(SubAgentOrchestrator)
        calls = []

        async def fake_spawn(**kwargs):
            calls.append(kwargs)
            outcome = outcomes[len(calls) - 1]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        orch.spawn = fake_spawn
        return orch, calls

    @pytest.mark.asyncio
    async def test_mixed_success_exception_and_failure(self):
        from victor.agent.subagents.base import SubAgentResult

        ok = SubAgentResult(
            success=True,
            summary="done",
            details={},
            tool_calls_used=5,
            context_size=100,
            duration_seconds=1.0,
        )
        failed = SubAgentResult(
            success=False,
            summary="nope",
            details={},
            tool_calls_used=1,
            context_size=10,
            duration_seconds=0.5,
            error="tool refused",
        )
        orch, calls = self._orchestrator_with([ok, RuntimeError("worker died"), failed])
        tasks = [
            SubAgentTask(SubAgentRole.RESEARCHER, "a"),
            SubAgentTask(SubAgentRole.EXECUTOR, "b"),
            SubAgentTask(SubAgentRole.TESTER, "c"),
        ]
        result = await orch.fan_out(tasks)

        assert isinstance(result, FanOutResult)
        assert result.all_success is False
        assert len(result.results) == 3
        assert result.results[0] is ok
        assert result.results[1].success is False
        assert "worker died" in result.results[1].error
        assert result.results[2] is failed
        assert result.total_tool_calls == 6
        assert len(result.errors) == 2
        assert any("RESEARCHER" not in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_all_success_when_every_task_succeeds(self):
        from victor.agent.subagents.base import SubAgentResult

        ok = SubAgentResult(
            success=True,
            summary="",
            details={},
            tool_calls_used=0,
            context_size=0,
            duration_seconds=0.1,
        )
        orch, _ = self._orchestrator_with([ok, ok])
        result = await orch.fan_out([SubAgentTask(SubAgentRole.RESEARCHER, "x")] * 2)
        assert result.all_success is True
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_task_fields_pass_through_to_spawn(self):
        from victor.agent.subagents.base import SubAgentResult

        ok = SubAgentResult(
            success=True,
            summary="",
            details={},
            tool_calls_used=0,
            context_size=0,
            duration_seconds=0.0,
        )
        orch, calls = self._orchestrator_with([ok])
        task = SubAgentTask(
            role=SubAgentRole.RESEARCHER,
            task="mapped",
            member_id="m1",
            agent_id="agent_x",
            display_name="X",
            team_id="t1",
            plan_id="p1",
            plan_step_id="s1",
            parent_session_id="root",
            child_session_id="child",
        )
        await orch.fan_out([task])
        assert calls[0]["member_id"] == "m1"
        assert calls[0]["agent_id"] == "agent_x"
        assert calls[0]["child_session_id"] == "child"
        assert calls[0]["task"] == "mapped"


class TestRunRegistryListAndEmission:
    """Registry listing filters and terminal-event emission edges."""

    def _registry_with_runs(self):
        registry = SubAgentRunRegistry()
        a = registry.start_run(prompt="a", child_session_id="cs-a")
        b = registry.start_run(prompt="b", child_session_id="cs-b")
        c = registry.start_run(prompt="c", child_session_id="cs-c")
        registry.complete_run(a.subagent_id, {"success": True, "summary": "done"})
        registry.fail_run(b.subagent_id, error="kaput")
        # c stays running
        return registry, a, b, c

    def test_list_newest_first_with_status_filter_and_limit(self):
        registry, a, b, c = self._registry_with_runs()
        # _finish move_to_end's terminal runs: dict order [c, a, b] (insertion,
        # then a and b re-appended at finish time) reversed -> most recently
        # finished first, still-running last.
        all_runs = registry.list_runs()
        assert [r["subagent_id"] for r in all_runs] == [
            b.subagent_id,
            a.subagent_id,
            c.subagent_id,
        ]

        completed = registry.list_runs(status=SubAgentRunStatus.COMPLETED)
        assert [r["subagent_id"] for r in completed] == [a.subagent_id]

        limited = registry.list_runs(limit=2)
        assert [r["subagent_id"] for r in limited] == [b.subagent_id, a.subagent_id]

    def test_result_dict_tolerates_malformed_details(self):
        registry = SubAgentRunRegistry()
        record = registry.start_run(prompt="t", child_session_id="cs-t")
        registry.complete_run(
            record.subagent_id,
            {"success": "yes", "summary": None, "tool_calls_used": "4", "details": "not-a-dict"},
        )
        refreshed = registry.get(record.subagent_id)
        fields = refreshed.result_dict()
        assert fields["success"] is True
        assert fields["content"] == ""
        assert fields["tool_calls_used"] == 4

    def test_unknown_id_lookups_are_refused(self):
        registry = SubAgentRunRegistry()
        assert registry.get("does-not-exist") is None
        assert registry.complete_run("nope", {}) is None
        assert registry.fail_run("nope", error="x") is None
        assert registry.cancel_run("nope") is None
        assert registry.attach_task("nope", object()) is False

    def test_terminal_states_refuse_double_transitions(self):
        registry = SubAgentRunRegistry()
        record = registry.start_run(prompt="t", child_session_id="cs-x")
        sid = record.subagent_id
        assert registry.complete_run(sid, {"success": True}) is not None
        # Terminal is terminal: no second transition of any kind.
        assert registry.fail_run(sid, error="late") is None
        assert registry.cancel_run(sid) is None


class TestExecuteErrorTerminal:
    """The buffered sibling of the stream error terminal (#2028's missed twin)."""

    @pytest.mark.asyncio
    async def test_get_messages_failure_does_not_mask_execute_error(self):
        sub = _make_subagent()
        orch = MagicMock()
        orch.tool_calls_used = 4
        orch.get_messages.side_effect = ValueError("corrupt history")
        orch.chat = AsyncMock(side_effect=RuntimeError("execute boom"))
        sub.orchestrator = orch

        result = await sub.execute()

        assert result.success is False
        assert "execute boom" in result.summary
        # The details dict must carry the ORIGINAL error, not crash with
        # UnboundLocalError when the context-size fallback also fails.
        assert result.details["error_type"] == "RuntimeError"
        assert "execute boom" in result.details["error_message"]
