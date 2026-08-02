"""FEP-0029 Phase 3a: faithful replay of a durably-paused turn.

`resume_paused_run` finds the gated tool_call in the paused assistant message (the one with no
result yet), executes it (approve — via the raw tool executor, bypassing the ASK) or skips it
(reject), appends the `role=tool` result, and drives continuation turns until the model stops
calling tools — WITHOUT re-sampling the model for the original call. Exercised with fakes for the
conversation controller / tool service / turn executor (no real provider).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from victor.agent.durable_resume import ResumeError, resume_paused_run
from victor.framework.approval_pause import ApprovalDecision


class _Msg:
    def __init__(
        self,
        role: str,
        content: str = "",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
    ) -> None:
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id
        self.metadata: Dict[str, Any] = {}


class _Controller:
    def __init__(self, messages: List[_Msg]) -> None:
        self.messages = messages
        self.appended: List[tuple] = []

    def add_tool_result(self, tool_call_id: str, content: str, **kw: Any) -> None:
        self.appended.append((tool_call_id, content))
        self.messages.append(_Msg("tool", content, tool_call_id=tool_call_id))


class _ToolService:
    def __init__(self) -> None:
        self.calls: List[tuple] = []  # raw execute_tool (pre-approved gated call)
        self.pipeline_calls: List[dict] = []  # execute_tool_call (siblings, via pipeline)

    async def execute_tool(self, tool_name: str, arguments: Any) -> Any:
        self.calls.append((tool_name, arguments))
        return SimpleNamespace(success=True, output=f"ran {tool_name}")

    async def execute_tool_call(self, tool_call: dict) -> dict:
        self.pipeline_calls.append(tool_call)
        return {"success": True, "result": f"sibling {tool_call.get('name')}"}


class _TurnExecutor:
    """Yields a scripted sequence of (content, tool_calls, has_tool_calls) per execute_turn call."""

    def __init__(self, script: List[tuple]) -> None:
        self._script = list(script)
        self.calls: List[str] = []

    async def execute_turn(self, user_message: str, *a: Any, **k: Any) -> Any:
        self.calls.append(user_message)
        content, tool_calls, has = self._script.pop(0)
        return SimpleNamespace(
            response=SimpleNamespace(content=content, tool_calls=tool_calls),
            has_tool_calls=has,
        )


def _orchestrator(messages: List[_Msg], script: List[tuple]) -> Any:
    return SimpleNamespace(
        _conversation_controller=_Controller(messages),
        _tool_service=_ToolService(),
        turn_executor=_TurnExecutor(script),
    )


def _paused(tool_name: str = "run_command", arguments: Any = None) -> Any:
    return SimpleNamespace(
        pending_tool={"tool_name": tool_name, "arguments": arguments or {"cmd": "ls"}},
        session_id="s1",
    )


def _gated_conversation() -> List[_Msg]:
    return [
        _Msg("user", "delete the temp dir"),
        _Msg(
            "assistant",
            "",
            tool_calls=[{"id": "call_1", "name": "run_command", "arguments": {"cmd": "rm -rf x"}}],
        ),
    ]


# ── approve ───────────────────────────────────────────────────────


async def test_approve_executes_gated_call_appends_result_and_continues() -> None:
    orch = _orchestrator(_gated_conversation(), script=[("all done", [], False)])
    out = await resume_paused_run(orch, _paused(), ApprovalDecision(approved=True))

    # The exact persisted call ran through the RAW executor (no re-ASK), with its arguments.
    assert orch._tool_service.calls == [("run_command", {"cmd": "rm -rf x"})]
    # Its result was appended, linked to the gated call's id.
    assert orch._conversation_controller.appended == [("call_1", "ran run_command")]
    # Continuation drove the model to a final answer (no new user message re-sampled the call).
    assert out.final_content == "all done"
    assert out.approved is True and out.gated_tool == "run_command"
    assert out.continuation_turns == 1
    # The continuation turn reused the original user message for tool-selection context.
    assert orch.turn_executor.calls == ["delete the temp dir"]


async def test_continuation_loops_until_no_more_tool_calls() -> None:
    orch = _orchestrator(
        _gated_conversation(),
        script=[("thinking", [{"id": "c2"}], True), ("final", [], False)],
    )
    out = await resume_paused_run(orch, _paused(), ApprovalDecision(approved=True))
    assert out.continuation_turns == 2
    assert out.final_content == "final"


# ── reject ────────────────────────────────────────────────────────


async def test_reject_skips_execution_and_appends_error() -> None:
    orch = _orchestrator(_gated_conversation(), script=[("understood", [], False)])
    out = await resume_paused_run(
        orch, _paused(), ApprovalDecision(approved=False, response="too risky")
    )

    assert orch._tool_service.calls == []  # tool never ran
    tc_id, content = orch._conversation_controller.appended[0]
    assert tc_id == "call_1" and "rejected by human" in content and "too risky" in content
    assert out.approved is False


# ── gated-call selection ──────────────────────────────────────────


async def test_picks_the_unresolved_call_when_a_sibling_already_ran() -> None:
    messages = [
        _Msg("user", "do two things"),
        _Msg(
            "assistant",
            "",
            tool_calls=[
                {"id": "done_1", "name": "read_file", "arguments": {}},
                {"id": "gated_2", "name": "run_command", "arguments": {"cmd": "x"}},
            ],
        ),
        _Msg("tool", "file contents", tool_call_id="done_1"),  # sibling already resolved
    ]
    orch = _orchestrator(messages, script=[("ok", [], False)])
    await resume_paused_run(orch, _paused(), ApprovalDecision(approved=True))
    assert orch._tool_service.calls == [("run_command", {"cmd": "x"})]
    assert orch._conversation_controller.appended[0][0] == "gated_2"


async def test_no_unresolved_call_raises() -> None:
    messages = [
        _Msg("assistant", "", tool_calls=[{"id": "c1", "name": "t"}]),
        _Msg("tool", "res", tool_call_id="c1"),  # already resolved
    ]
    orch = _orchestrator(messages, script=[])
    with pytest.raises(ResumeError):
        await resume_paused_run(orch, _paused(), ApprovalDecision(approved=True))


async def test_json_string_arguments_are_parsed() -> None:
    messages = [
        _Msg("user", "go"),
        _Msg(
            "assistant",
            "",
            tool_calls=[{"id": "c1", "name": "run_command", "arguments": '{"cmd": "ls"}'}],
        ),
    ]
    orch = _orchestrator(messages, script=[("done", [], False)])
    await resume_paused_run(orch, _paused(), ApprovalDecision(approved=True))
    assert orch._tool_service.calls == [("run_command", {"cmd": "ls"})]


async def test_missing_runtime_surface_raises() -> None:
    orch = SimpleNamespace(_conversation_controller=None, _tool_service=None, turn_executor=None)
    with pytest.raises(ResumeError):
        await resume_paused_run(orch, _paused(), ApprovalDecision(approved=True))


# ── multi-tool batch partiality ───────────────────────────────────


def _batch_conversation() -> List[_Msg]:
    # A parallel-tool pause: the whole batch aborted at the gate, so BOTH are unresolved.
    return [
        _Msg("user", "do two things"),
        _Msg(
            "assistant",
            "",
            tool_calls=[
                {"id": "gated", "name": "run_command", "arguments": {"cmd": "rm -rf x"}},
                {"id": "sib", "name": "read_file", "arguments": {"path": "a.txt"}},
            ],
        ),
    ]


async def test_batch_approve_gated_and_runs_siblings_via_pipeline() -> None:
    orch = _orchestrator(_batch_conversation(), script=[("done", [], False)])
    out = await resume_paused_run(orch, _paused(), ApprovalDecision(approved=True))

    # Gated call ran through the RAW executor (pre-approved, no re-ASK).
    assert orch._tool_service.calls == [("run_command", {"cmd": "rm -rf x"})]
    # The sibling ran through the normal pipeline (policy-honoring), not the raw executor.
    assert [c["id"] for c in orch._tool_service.pipeline_calls] == ["sib"]
    # BOTH results were appended (so no tool_call is left without a result before continuing).
    assert {tc_id for tc_id, _ in orch._conversation_controller.appended} == {"gated", "sib"}
    assert out.executed_siblings == 1
    assert out.gated_tool == "run_command" and out.final_content == "done"


async def test_batch_reject_gated_still_runs_siblings() -> None:
    orch = _orchestrator(_batch_conversation(), script=[("ok", [], False)])
    out = await resume_paused_run(orch, _paused(), ApprovalDecision(approved=False, response="no"))
    assert orch._tool_service.calls == []  # gated tool never executed
    assert [c["id"] for c in orch._tool_service.pipeline_calls] == ["sib"]  # sibling still ran
    appended = dict(orch._conversation_controller.appended)
    assert "rejected by human" in appended["gated"]
    assert appended["sib"] == "sibling read_file"
    assert out.executed_siblings == 1


async def test_batch_gated_pick_disambiguated_by_pending_tool() -> None:
    # The gated one is the SECOND call; pending_tool names it, so it (not the first) is decided.
    messages = [
        _Msg("user", "go"),
        _Msg(
            "assistant",
            "",
            tool_calls=[
                {"id": "first", "name": "read_file", "arguments": {}},
                {"id": "second", "name": "run_command", "arguments": {"cmd": "x"}},
            ],
        ),
    ]
    orch = _orchestrator(messages, script=[("done", [], False)])
    await resume_paused_run(orch, _paused(tool_name="run_command"), ApprovalDecision(approved=True))
    # run_command (the gated one) went to the raw executor; read_file (sibling) to the pipeline.
    assert orch._tool_service.calls == [("run_command", {"cmd": "x"})]
    assert [c["id"] for c in orch._tool_service.pipeline_calls] == ["first"]
