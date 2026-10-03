"""Exact-action resume: real policy/pipeline/executor with an in-memory transcript.

Tests use the service-owned runtime, real policy/pipeline/executor and canonical
result post-processing with a fake host. No provider or external effect runs.
"""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from victor.agent.durable_resume import ResumeError, resume_paused_run
from victor.agent.middleware_chain import MiddlewareChain
from victor.agent.paused_run_store import (
    InMemoryPausedRunStore,
    record_pause_from_approval,
    reset_paused_run_store,
    set_paused_run_store,
)
from victor.agent.tool_executor import ToolExecutor
from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig
from victor.framework.approval_pause import (
    ApprovalDecision,
    ApprovalPause,
    current_durable_pause_enabled,
)
from victor.framework.policies import (
    AskOnToolsPolicy,
    PolicyContext,
    PolicyEngine,
    PolicyEngineMiddleware,
)
from victor.tools.decorators import tool
from victor.tools.enums import AccessMode
from victor.tools.registry import ToolRegistry


@pytest.fixture
def store():
    value = InMemoryPausedRunStore()
    set_paused_run_store(value)
    yield value
    reset_paused_run_store()


class Controller:
    def __init__(self, call):
        self.messages = [
            SimpleNamespace(role="user", content="submit record"),
            SimpleNamespace(role="assistant", tool_calls=[call]),
        ]
        self.appended = []

    def add_tool_result(self, call_id, content):
        self.appended.append((call_id, content))
        self.messages.append(SimpleNamespace(role="tool", tool_call_id=call_id, content=content))


async def paused_runtime(store, *, argument=None, empty=False, effect_error=None):
    effects = []

    @tool(access_mode=AccessMode.WRITE)
    async def submit_record(payload: str, _exec_ctx=None):
        effects.append(payload)
        if effect_error is not None:
            raise effect_error
        return "receipt"

    if empty:

        @tool(name="submit_record", access_mode=AccessMode.WRITE)
        async def empty_record(_exec_ctx=None):
            effects.append("original")
            return "receipt"

        submit_record = empty_record
    registry = ToolRegistry()
    registry.register(submit_record)
    executor = ToolExecutor(tool_registry=registry, retry_delay=0)
    engine = PolicyEngine([AskOnToolsPolicy({"submit_record"})])
    scope = {"session_id": "s1", "labels": {"role": "operator"}}

    async def ask(request):
        raise ApprovalPause(request)

    middleware = PolicyEngineMiddleware(
        engine, lambda: PolicyContext(**scope), approval_handler=ask
    )
    chain = MiddlewareChain()
    chain.add(middleware)
    pipeline = ToolPipeline(
        tool_registry=registry,
        tool_executor=executor,
        middleware_chain=chain,
        config=ToolPipelineConfig(enable_caching=False, enable_semantic_caching=False),
    )
    call = {
        "id": "call_1",
        "name": "submit_record",
        "arguments": {} if empty else (argument or {"payload": "original"}),
    }
    controller = Controller(call)

    from victor.agent.services.tool_execution_runtime import ToolExecutionRuntime
    from victor.agent.services.tool_service import process_tool_results_with_context
    from unittest.mock import MagicMock

    turn = SimpleNamespace(
        response=SimpleNamespace(content="all done", tool_calls=[]), has_tool_calls=False
    )
    controller.get_context_metrics = lambda: SimpleNamespace(
        remaining_tokens=8000, max_tokens=10000
    )

    def add_message(role, content, **kwargs):
        assert role == "tool"
        controller.add_tool_result(kwargs["tool_call_id"], content)

    orch = SimpleNamespace(
        _conversation_controller=controller,
        active_session_id="s1",
        _tool_pipeline=pipeline,
        _get_tool_context=lambda: {},
        _tool_service=SimpleNamespace(process_tool_results=process_tool_results_with_context),
        executed_tools=[],
        observed_files=set(),
        failed_tool_signatures=set(),
        _shown_tool_errors=set(),
        _continuation_prompts=0,
        _asking_input_prompts=0,
        tool_calls_used=0,
        _record_tool_execution=None,
        conversation_state=None,
        unified_tracker=None,
        usage_logger=None,
        console=MagicMock(),
        _presentation=None,
        add_message=add_message,
        _tool_output_formatter=SimpleNamespace(format_tool_output=lambda **kw: str(kw["output"])),
        turn_executor=SimpleNamespace(execute_turn=AsyncMock(return_value=turn)),
    )
    runtime = ToolExecutionRuntime(orch)
    runtime.execute_tool_calls = AsyncMock(wraps=runtime.execute_tool_calls)
    orch._get_tool_execution_runtime = lambda: runtime
    token = current_durable_pause_enabled.set(True)
    try:
        with pytest.raises(ApprovalPause) as caught:
            await pipeline.execute_tool_calls([deepcopy(call)], {})
    finally:
        current_durable_pause_enabled.reset(token)
    run_id, _ = record_pause_from_approval(caught.value.request, session_id="s1", agent_id="a1")
    return SimpleNamespace(
        orch=orch,
        paused=store.get(run_id),
        effects=effects,
        pipeline=pipeline,
        scope=scope,
        runtime=runtime,
        controller=controller,
    )


@pytest.mark.parametrize("empty", [False, True])
async def test_approve_exact_call_through_policy_appends_once_and_continues(store, empty):
    state = await paused_runtime(store, empty=empty)
    out = await resume_paused_run(state.orch, state.paused, ApprovalDecision(True))
    assert state.effects == ["original"]
    assert state.controller.appended == [("call_1", "receipt")]
    assert out.approved and out.final_content == "all done" and out.continuation_turns == 1
    assert out.executed_siblings == 0
    state.runtime.execute_tool_calls.assert_awaited_once()
    state.orch.turn_executor.execute_turn.assert_awaited_once_with("submit record")


async def test_approval_rejects_changed_payload_before_any_execution(store):
    state = await paused_runtime(store)
    state.controller.messages[1].tool_calls[0]["arguments"] = {"payload": "different"}
    with pytest.raises(ResumeError):
        await resume_paused_run(state.orch, state.paused, ApprovalDecision(True))
    assert state.effects == []
    state.runtime.execute_tool_calls.assert_not_awaited()


async def test_reject_skips_execution_and_appends_error(store):
    state = await paused_runtime(store)
    out = await resume_paused_run(state.orch, state.paused, ApprovalDecision(False, response="no"))
    assert not out.approved and state.effects == []
    assert (
        len(state.controller.appended) == 1
        and "rejected by human" in state.controller.appended[0][1]
    )
    state.runtime.execute_tool_calls.assert_not_awaited()


@pytest.mark.parametrize(
    "change",
    [
        "legacy",
        "duplicate",
        "sibling",
        "same_name",
        "request",
        "expiry",
        "scope",
        "member",
        "session",
        "malformed",
        "non_bool",
        "resolved",
        "incomplete_evidence",
        "malformed_evidence",
        "surface",
    ],
)
async def test_invalid_binding_never_dispatches(store, change):
    state = await paused_runtime(store)
    decision = ApprovalDecision(True)
    calls = state.controller.messages[1].tool_calls
    if change == "legacy":
        del state.paused.pending_tool["binding"]
    elif change == "duplicate":
        calls.append(deepcopy(calls[0]))
    elif change in ("sibling", "same_name"):
        calls.append(
            {
                **deepcopy(calls[0]),
                "id": "other",
                "name": "read" if change == "sibling" else "submit_record",
            }
        )
    elif change == "request":
        state.paused.approval_request["id"] = "other"
    elif change == "expiry":
        state.paused.approval_request["created_at"] = 1
        state.paused.pending_tool["binding"]["expires_at"] = 301
    elif change == "scope":
        state.scope["labels"] = {"role": "different"}
    elif change == "member":
        state.paused.approval_request["context"]["member_id"] = "member-1"
    elif change == "session":
        state.orch.active_session_id = "other"
    elif change == "malformed":
        calls[0]["arguments"] = '{"payload": broken}'
    elif change == "non_bool":
        decision = ApprovalDecision("yes")
    elif change in {"incomplete_evidence", "malformed_evidence"}:
        state.paused.approval_request["context"]["batch_result_publication"] = (
            {"schema_version": 1, "status": "incomplete"}
            if change == "incomplete_evidence"
            else False
        )
    elif change == "resolved":
        state.controller.add_tool_result("call_1", "previous receipt")
    else:
        state.orch._get_tool_execution_runtime = None
    with pytest.raises(ResumeError):
        await resume_paused_run(state.orch, state.paused, decision)
    assert state.effects == []
    state.orch.turn_executor.execute_turn.assert_not_awaited()


async def test_json_string_arguments_are_parsed(store):
    state = await paused_runtime(store, argument='{"payload": "original"}')
    await resume_paused_run(state.orch, state.paused, ApprovalDecision(True))
    assert state.effects == ["original"]


async def test_resolved_sibling_is_not_reexecuted(store):
    state = await paused_runtime(store)
    state.controller.messages[1].tool_calls.insert(
        0, {"id": "done", "name": "read", "arguments": {}}
    )
    state.controller.add_tool_result("done", "prior receipt")
    await resume_paused_run(state.orch, state.paused, ApprovalDecision(True))
    assert state.effects == ["original"]
    assert len(state.runtime.execute_tool_calls.call_args.args[0]) == 1


@pytest.mark.parametrize(
    "change",
    [
        "deny",
        "different_ask",
        "policy_arguments",
        "hook_arguments",
        "normalization",
        "missing_policy",
        "schema",
        "authority",
        "later_middleware",
        "dispatch_expiry",
        "dispatch_session",
        "dispatch_scope",
        "dispatch_context_failure",
        "dispatch_multiple_scopes",
        "dispatch_resolution_expiry",
    ],
)
async def test_current_policy_and_final_payload_are_enforced(store, change, monkeypatch):
    from victor.framework.policies import Policy, PolicyVerdict

    state = await paused_runtime(store)

    class NewPolicy(Policy):
        name = "new"

        async def evaluate(self, event):
            if change == "deny":
                return PolicyVerdict.deny("revoked")
            if change == "different_ask":
                return PolicyVerdict.ask("new approval", policy_name="other")
            return PolicyVerdict.allow(modified_arguments={"payload": "changed"})

    if change in ("deny", "different_ask", "policy_arguments"):
        state.pipeline.middleware_chain = MiddlewareChain()
        state.pipeline.middleware_chain.add(
            PolicyEngineMiddleware(
                PolicyEngine([NewPolicy()]), lambda: PolicyContext(**state.scope)
            )
        )
    elif change == "hook_arguments":
        state.pipeline.executor._run_before_hooks = lambda name, args: args.update(
            payload="changed"
        )
    elif change == "normalization":
        state.pipeline.executor.normalizer.normalize_arguments = lambda args, name: (
            {"payload": "changed"},
            None,
        )
    elif change == "missing_policy":
        state.pipeline.middleware_chain = None
    elif change == "authority":
        state.pipeline.executor.current_user = "different-user"
    elif change == "later_middleware":
        original_before = state.pipeline.middleware_chain.process_before

        async def mutate(name, args):
            from victor.core.verticals.protocols import MiddlewareResult

            await original_before(name, args)
            return MiddlewareResult(proceed=True, modified_arguments={"payload": "changed"})

        state.pipeline.middleware_chain.process_before = mutate
    elif change == "dispatch_scope":
        state.pipeline.executor._run_before_hooks = lambda name, args: state.scope.update(
            labels={"role": "revoked"}
        )
    elif change == "dispatch_context_failure":
        state.pipeline.executor._run_before_hooks = lambda name, args: state.scope.update(
            cost_usd=float("nan")
        )
    elif change == "dispatch_multiple_scopes":
        other_scope = deepcopy(state.scope)
        state.pipeline.middleware_chain.add(
            PolicyEngineMiddleware(PolicyEngine([]), lambda: PolicyContext(**other_scope))
        )
        # A later unchanged owner must not replace the first owner's fresh check.
        state.pipeline.executor._run_before_hooks = lambda name, args: state.scope.update(
            labels={"role": "revoked"}
        )
    elif change == "dispatch_resolution_expiry":
        from victor.framework import approval_binding

        def refresh_context():
            grant = approval_binding.current_approval_grant.get()
            if grant.scope_resolvers:
                # A synchronous refresh can cross expiry without any async yield.
                monkeypatch.setattr(
                    approval_binding, "time", SimpleNamespace(time=lambda: grant.expires_at + 1)
                )
            return PolicyContext(**state.scope)

        state.pipeline.middleware_chain._middleware[0]._context_provider = refresh_context
    elif change == "dispatch_session":
        state.pipeline.executor._run_before_hooks = lambda name, args: setattr(
            state.orch, "active_session_id", "other"
        )
    elif change == "dispatch_expiry":
        from victor.framework.approval_binding import current_approval_grant

        state.pipeline.executor._run_before_hooks = lambda name, args: setattr(
            current_approval_grant.get(), "expires_at", 1
        )
    else:
        # Registry contract change between approval and resume, not an invented version label.
        from unittest.mock import PropertyMock, patch

        original = state.pipeline.tools.get("submit_record")
        with patch.object(
            type(original), "parameters", new_callable=PropertyMock, return_value={"type": "object"}
        ):
            with pytest.raises(ResumeError):
                await resume_paused_run(state.orch, state.paused, ApprovalDecision(True))
        assert state.effects == []
        return
    with pytest.raises(ResumeError):
        await resume_paused_run(state.orch, state.paused, ApprovalDecision(True))
    assert state.effects == []


async def test_continuation_loops_and_chained_pause_parks_again(store):
    state = await paused_runtime(store)
    from victor.framework.hitl import ApprovalRequest

    request = ApprovalRequest(
        id="new",
        title="Approve next action",
        description="",
        context={"tool_name": "next", "arguments": {}},
    )
    state.orch.turn_executor.execute_turn.side_effect = [
        SimpleNamespace(
            response=SimpleNamespace(content="thinking", tool_calls=[{"id": "next"}]),
            has_tool_calls=True,
        ),
        ApprovalPause(request),
    ]
    out = await resume_paused_run(state.orch, state.paused, ApprovalDecision(True))
    assert state.effects == ["original"]
    assert out.continuation_turns == 1 and out.awaiting_run_id
    assert store.get(out.awaiting_run_id).metadata["chained_from"] == state.paused.run_id


async def test_missing_result_after_dispatch_never_continues_or_replays(store):
    state = await paused_runtime(store)

    def missing(*args, **kwargs):
        raise OSError("result storage unavailable")

    state.orch.add_message = missing
    assert store.mark_resumed(state.paused.run_id)
    with pytest.raises(ResumeError, match="reconcile"):
        await resume_paused_run(state.orch, state.paused, ApprovalDecision(True))
    assert state.effects == ["original"]
    state.orch.turn_executor.execute_turn.assert_not_awaited()
    assert not store.mark_resumed(state.paused.run_id)


@pytest.mark.parametrize("arguments", ['{"x": 1, "x": 2}', '{"x": NaN}', "[]", {1: "bad"}])
def test_approval_arguments_reject_ambiguous_json(arguments):
    from victor.framework.approval_binding import ApprovalBindingError, parse_arguments

    with pytest.raises(ApprovalBindingError):
        parse_arguments(arguments)


@pytest.mark.parametrize("signal", ["cancel", "pause"])
async def test_control_signal_resets_call_and_grant_context(store, signal):
    import asyncio
    from victor.framework.approval_binding import current_approval_call, current_approval_grant
    from victor.framework.hitl import ApprovalRequest

    state = await paused_runtime(store)
    # A fresh initial pause already unwound the per-call wrapper.
    assert current_approval_call.get() is None and current_approval_grant.get() is None
    error = (
        asyncio.CancelledError()
        if signal == "cancel"
        else ApprovalPause(ApprovalRequest(id="new", title="new", description=""))
    )

    def stop(name, args):
        raise error

    state.pipeline.executor._run_before_hooks = stop
    with pytest.raises(type(error)):
        await resume_paused_run(state.orch, state.paused, ApprovalDecision(True))
    assert state.effects == []
    assert current_approval_call.get() is None and current_approval_grant.get() is None
    # A subsequent call must ASK again; the previous grant cannot leak.
    state.pipeline.executor._run_before_hooks = lambda *args: None
    token = current_durable_pause_enabled.set(True)
    try:
        with pytest.raises(ApprovalPause):
            await state.pipeline.execute_tool_calls(
                [{"id": "fresh", "name": "submit_record", "arguments": {"payload": "original"}}]
            )
    finally:
        current_durable_pause_enabled.reset(token)
    assert state.effects == []


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "timeout",
        "reported_failure",
        "cancel",
        "intent",
        "observation",
        "cancel_observation",
        "publication",
        "after_hook",
        "output_deny",
        "output_ask",
    ],
)
async def test_durable_action_observation_at_real_dispatch(tmp_path, monkeypatch, failure):
    import asyncio
    import sqlite3
    from victor.agent.paused_run_store import ProjectDbPausedRunStore

    durable = ProjectDbPausedRunStore(tmp_path / "actions.db")
    set_paused_run_store(durable)
    try:
        error = {
            "reported_failure": TimeoutError(),
            "cancel": asyncio.CancelledError(),
            "cancel_observation": asyncio.CancelledError(),
        }.get(failure)
        state = await paused_runtime(durable, effect_error=error)
        if failure == "timeout":

            async def committed_timeout(**kwargs):
                with sqlite3.connect(tmp_path / "backend.db") as backend:
                    backend.execute("CREATE TABLE receipt (action_id TEXT PRIMARY KEY)")
                    backend.execute(
                        "INSERT INTO receipt VALUES (?)",
                        (kwargs["_exec_ctx"]["durable_action_id"],),
                    )
                state.effects.append("original")
                raise TimeoutError()

            monkeypatch.setattr(
                state.pipeline.executor.tools.get("submit_record"), "execute", committed_timeout
            )
        run_id = state.paused.run_id
        assert durable.mark_resumed(run_id)
        observation_attempts = []
        if failure in {"intent", "observation", "cancel_observation"}:
            method = "begin_action" if failure == "intent" else "observe_action"

            def broken(*args, **kwargs):
                observation_attempts.append(args)
                raise OSError("storage unavailable")

            monkeypatch.setattr(durable, method, broken)
        if failure in {"output_deny", "output_ask"}:
            from victor.framework.policies import Policy, Phase, PolicyVerdict

            class WithholdResult(Policy):
                def phases(self):
                    return {Phase.TOOL_RESULT}

                async def evaluate(self, event):
                    return (
                        PolicyVerdict.ask("withhold")
                        if failure == "output_ask"
                        else PolicyVerdict.deny("withhold")
                    )

            state.pipeline.middleware_chain.add(
                PolicyEngineMiddleware(
                    PolicyEngine([WithholdResult()]), lambda: PolicyContext(**state.scope)
                )
            )
        publication_attempts = []
        if failure in {"publication", "after_hook"}:

            def fail_after_effect(*args, **kwargs):
                publication_attempts.append(kwargs)
                raise OSError("post-effect bookkeeping unavailable")

            if failure == "publication":
                state.orch.add_message = fail_after_effect
            else:
                monkeypatch.setattr(state.pipeline.executor, "_run_after_hooks", fail_after_effect)
        if failure in {"cancel", "cancel_observation"}:
            with pytest.raises(asyncio.CancelledError) as caught:
                await resume_paused_run(
                    state.orch, state.paused, ApprovalDecision(True), action_store=durable
                )
            assert caught.value is error
        elif failure:
            with pytest.raises(ResumeError, match="action"):
                await resume_paused_run(
                    state.orch, state.paused, ApprovalDecision(True), action_store=durable
                )
        else:
            result = await resume_paused_run(
                state.orch, state.paused, ApprovalDecision(True), action_store=durable
            )
            assert result.final_content == "all done"
        reopened = ProjectDbPausedRunStore(tmp_path / "actions.db")
        action = reopened.get(run_id).action
        assert state.effects == ([] if failure == "intent" else ["original"])
        if failure == "intent":
            assert action is None
        else:
            expected = (
                "pending"
                if failure in {"observation", "cancel_observation"}
                else (
                    "returned"
                    if failure
                    in {
                        None,
                        "reported_failure",
                        "publication",
                        "after_hook",
                        "output_deny",
                        "output_ask",
                    }
                    else "unknown"
                )
            )
            assert action["state"] == expected
            assert action["backend_receipt"] is None
        if failure in {"intent", "observation", "cancel_observation"}:
            assert len(observation_attempts) == 1
        if failure == "publication":
            assert len(publication_attempts) == 1
            assert publication_attempts[0]["require_persistence"] is True
        if failure == "timeout":
            with sqlite3.connect(tmp_path / "backend.db") as backend:
                assert backend.execute("SELECT action_id FROM receipt").fetchall() == [
                    (action["action_id"],)
                ]
        if failure:
            state.orch.turn_executor.execute_turn.assert_not_awaited()
        assert reopened.mark_resumed(run_id) is False
    finally:
        reset_paused_run_store()


@pytest.mark.parametrize("change", ["scope", "session", "expiry", "prior_intent"])
async def test_durable_intent_cannot_bypass_final_authority_or_restart_guard(
    tmp_path, monkeypatch, change
):
    from victor.agent.paused_run_store import ProjectDbPausedRunStore
    from victor.framework import approval_binding

    durable = ProjectDbPausedRunStore(tmp_path / "actions.db")
    set_paused_run_store(durable)
    try:
        state = await paused_runtime(durable)
        run_id = state.paused.run_id
        assert durable.mark_resumed(run_id)
        if change == "prior_intent":
            durable.begin_action(run_id, state.paused.pending_tool["binding"])
            durable = ProjectDbPausedRunStore(tmp_path / "actions.db")
        else:
            begin = durable.begin_action

            def save_then_change(*args):
                action = begin(*args)
                if change == "scope":
                    state.scope["labels"] = {"role": "revoked"}
                elif change == "session":
                    state.orch.active_session_id = "other"
                else:
                    expiry = approval_binding.current_approval_grant.get().expires_at
                    monkeypatch.setattr(
                        approval_binding, "time", SimpleNamespace(time=lambda: expiry + 1)
                    )
                return action

            monkeypatch.setattr(durable, "begin_action", save_then_change)
        with pytest.raises(ResumeError):
            await resume_paused_run(
                state.orch, state.paused, ApprovalDecision(True), action_store=durable
            )
        assert state.effects == []
        state.orch.turn_executor.execute_turn.assert_not_awaited()
        action = durable.get(run_id).action
        assert action["state"] == ("pending" if change == "prior_intent" else "unknown")
        assert not durable.mark_resumed(run_id)
    finally:
        reset_paused_run_store()
