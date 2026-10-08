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


async def paused_runtime(store, *, argument=None, empty=False, effect_error=None, recovery=None):
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
    registry.get("submit_record").action_recovery = recovery
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


@pytest.mark.parametrize("changed_owner", [None, "active_session_id", "agent_id"])
@pytest.mark.parametrize("persistent", [False, True], ids=["custom-store", "project-db"])
async def test_continuation_loops_and_chained_pause_parks_again(
    store, tmp_path, monkeypatch, persistent, changed_owner
):
    import threading
    from victor.agent.paused_run_store import ProjectDbPausedRunStore

    if persistent:
        store = ProjectDbPausedRunStore(tmp_path / "chained.db")
        set_paused_run_store(store)
    state = await paused_runtime(store)
    caller_thread = threading.get_ident()
    save = store.save
    save_threads = []

    def observed_save(**kwargs):
        save_threads.append(threading.get_ident())
        run_id = save(**kwargs)
        if changed_owner:
            setattr(state.orch, changed_owner, "other")
        return run_id

    monkeypatch.setattr(store, "save", observed_save)
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
    if changed_owner:
        with pytest.raises(PermissionError, match="ownership changed"):
            await resume_paused_run(state.orch, state.paused, ApprovalDecision(True))
        chained = [run for run in store.list_pending() if run.run_id != state.paused.run_id]
        assert len(chained) == 1
        assert chained[0].session_id == state.paused.session_id
        assert chained[0].metadata["chained_from"] == state.paused.run_id
    else:
        out = await resume_paused_run(state.orch, state.paused, ApprovalDecision(True))
        assert out.continuation_turns == 1 and out.awaiting_run_id
        assert store.get(out.awaiting_run_id).metadata["chained_from"] == state.paused.run_id
    assert state.effects == ["original"]
    assert state.orch.turn_executor.execute_turn.await_count == 2
    assert len(save_threads) == 1
    assert (save_threads[0] != caller_thread) is persistent
    assert not current_durable_pause_enabled.get()


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


async def test_second_cancel_during_interruption_writes_unknown_detached(tmp_path):
    """A second cancellation during the shielded interrupted_async await must
    still propagate cancellation AND complete the durable 'unknown' write
    detached — the marker must never be skipped by a racing cancel."""
    import asyncio
    import threading

    from victor.agent.action_observation import ActionJournal
    from victor.agent.paused_run_store import ProjectDbPausedRunStore

    durable = ProjectDbPausedRunStore(tmp_path / "state.db")
    set_paused_run_store(durable)
    binding = {"session_id": "s", "tool_name": "t"}
    run_id = durable.save(
        session_id="s",
        agent_id="a",
        approval_request={},
        pending_tool={"tool_name": "t", "arguments": {}, "binding": binding},
    )
    assert durable.mark_resumed(run_id)

    started = threading.Event()
    gate = threading.Event()
    real_observe = durable.observe_action

    def gated(*args, **kwargs):
        started.set()
        assert gate.wait(10), "gate never released"
        return real_observe(*args, **kwargs)

    durable.observe_action = gated  # instance attribute: gates only this test
    journal = ActionJournal(durable, run_id, binding)
    journal.begin()

    task = asyncio.create_task(journal.interrupted_async(RuntimeError("cancelled")))
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()

    gate.set()
    for _ in range(100):
        action = durable.get(run_id).action
        if action is not None and action["state"] == "unknown":
            break
        await asyncio.sleep(0.05)
    assert durable.get(run_id).action["state"] == "unknown"


class ReceiptBackend:
    """Test-only backend: business effect and exact receipt share one SQLite commit."""

    def __init__(self, path):
        from victor.framework.action_recovery import RecoveryIdentity

        import hashlib

        self.path = path
        self.identity = RecoveryIdentity(
            "sqlite-test-v1", "test-account:" + hashlib.sha256(str(path).encode()).hexdigest()
        )
        self.lookups = 0

    def commit(self, request, payload):
        import sqlite3

        with sqlite3.connect(self.path) as db:
            assert db.execute("PRAGMA synchronous").fetchone()[0] == 2
            db.execute(
                "CREATE TABLE IF NOT EXISTS effects (action_id TEXT PRIMARY KEY, "
                "binding_digest TEXT, payload TEXT)"
            )
            db.execute(
                "INSERT INTO effects VALUES (?, ?, ?)",
                (request.action_id, request.binding_digest, payload),
            )

    async def lookup(self, request):
        import sqlite3
        from victor.framework.action_recovery import BackendReceipt

        self.lookups += 1
        with sqlite3.connect(self.path) as db:
            row = db.execute(
                "SELECT binding_digest FROM effects WHERE action_id = ?", (request.action_id,)
            ).fetchone()
        if row is None:
            return None
        return BackendReceipt(
            request.action_id, row[0], self.identity, "receipt:" + request.action_id
        )


def receipt_client(orch):
    from victor.framework.client import VictorClient

    client = VictorClient.__new__(VictorClient)
    client._initialized = True
    client._context = object()
    client._agent = SimpleNamespace(_orchestrator=orch)
    return client


async def committed_receipt_runtime(tmp_path, monkeypatch):
    from victor.agent.paused_run_store import ProjectDbPausedRunStore

    durable = ProjectDbPausedRunStore(tmp_path / "actions.db")
    set_paused_run_store(durable)
    backend = ReceiptBackend(tmp_path / "backend.db")
    state = await paused_runtime(durable, recovery=backend)

    async def commit_then_lose_response(payload, _exec_ctx):
        request = _exec_ctx["durable_action"]
        assert request.action_id == _exec_ctx["durable_action_id"]
        backend.commit(request, payload)
        state.effects.append(payload)
        raise TimeoutError("backend committed; response lost")

    monkeypatch.setattr(
        state.pipeline.executor.tools.get("submit_record"), "execute", commit_then_lose_response
    )
    assert durable.mark_resumed(state.paused.run_id)
    with pytest.raises(ResumeError):
        await resume_paused_run(
            state.orch, state.paused, ApprovalDecision(True), action_store=durable
        )
    return durable, backend, state


async def test_receipt_lookup_after_lost_response_and_store_restart_never_reexecutes(
    tmp_path, monkeypatch, store
):
    from victor.agent.paused_run_store import ProjectDbPausedRunStore

    durable, backend, state = await committed_receipt_runtime(tmp_path, monkeypatch)
    run_id = state.paused.run_id
    action = durable.get(run_id).action
    assert action["version"] == 2
    assert action["state"] == "unknown" and action["backend_receipt"] is None
    reopened = ProjectDbPausedRunStore(tmp_path / "actions.db")
    set_paused_run_store(reopened)
    # Recreate the adapter too; no in-memory execution observation is recovery proof.
    backend = ReceiptBackend(tmp_path / "backend.db")
    state.pipeline.executor.tools.get("submit_record").action_recovery = backend
    client = receipt_client(state.orch)
    original_transcript = deepcopy(state.controller.appended)
    result = await client.reconcile_action(run_id)
    assert result["status"] == "verified"
    assert result["action"]["state"] == "unknown"  # Invocation evidence remains honest.
    assert result["action"]["backend_receipt"]["outcome"] == "committed"
    assert client.get_action_status(run_id)["action"] == result["action"]
    assert not reopened.mark_resumed(run_id)
    assert reopened.purge(before=10**12) == 0
    assert (await client.reconcile_action(run_id))["action"] == result["action"]
    assert backend.lookups == 1
    assert state.effects == ["original"]
    assert state.controller.appended == original_transcript
    state.orch.turn_executor.execute_turn.assert_not_awaited()


@pytest.mark.parametrize("change", ["session", "authority", "backend", "rbac"])
@pytest.mark.parametrize("recorded", [False, True])
async def test_receipt_lookup_and_status_reject_changed_authority(
    tmp_path, monkeypatch, store, change, recorded
):
    from victor.framework.action_recovery import RecoveryIdentity

    durable, backend, state = await committed_receipt_runtime(tmp_path, monkeypatch)
    client = receipt_client(state.orch)
    if recorded:
        await client.reconcile_action(state.paused.run_id)
    before = durable.get(state.paused.run_id).action
    if change == "session":
        state.orch.active_session_id = "other"
    elif change == "authority":
        state.pipeline.executor.current_user = "other"
    elif change == "backend":
        backend.identity = RecoveryIdentity("sqlite-test-v1", "different-account")
    else:
        state.pipeline.executor.rbac_manager = SimpleNamespace(check_tool_access=lambda **kw: False)
    with pytest.raises(PermissionError):
        await client.reconcile_action(state.paused.run_id)
    with pytest.raises(PermissionError):
        client.get_action_status(state.paused.run_id)
    assert backend.lookups == int(recorded)
    assert durable.get(state.paused.run_id).action == before


@pytest.mark.parametrize(
    "failure",
    [
        "absent",
        "error",
        "timeout",
        "cancel",
        "action",
        "binding",
        "identity",
        "untyped",
        "extended",
    ],
)
async def test_unproven_receipts_cannot_change_the_action(tmp_path, monkeypatch, store, failure):
    import asyncio
    from dataclasses import replace
    from victor.agent.action_observation import ActionStateError
    from victor.framework.action_recovery import RecoveryIdentity

    durable, backend, state = await committed_receipt_runtime(tmp_path, monkeypatch)
    original = backend.lookup

    async def lookup(request):
        if failure == "absent":
            return None
        if failure == "error":
            raise OSError("secret credential must not escape")
        if failure == "timeout":
            await asyncio.Event().wait()
        if failure == "cancel":
            raise asyncio.CancelledError()
        receipt = await original(request)
        if failure == "action":
            return replace(receipt, action_id="0" * 64)
        if failure == "binding":
            return replace(receipt, binding_digest="0" * 64)
        if failure == "identity":
            return replace(receipt, identity=RecoveryIdentity("sqlite-test-v1", "other"))
        if failure == "extended":
            from dataclasses import make_dataclass
            from victor.framework.action_recovery import BackendReceipt

            extended = make_dataclass(
                "ExtendedReceipt",
                [("raw_response", str, "secret")],
                bases=(BackendReceipt,),
                frozen=True,
            )
            return extended(
                receipt.action_id, receipt.binding_digest, receipt.identity, receipt.receipt_id
            )
        return receipt.to_dict()

    monkeypatch.setattr(backend, "lookup", lookup)
    client = receipt_client(state.orch)
    run_id = state.paused.run_id
    before = durable.get(run_id).action
    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await client.reconcile_action(run_id)
    elif failure in {"action", "binding", "identity", "untyped", "extended"}:
        with pytest.raises((ActionStateError, PermissionError)):
            await client.reconcile_action(run_id)
    else:
        result = await client.reconcile_action(run_id, timeout_seconds=0.01)
        assert result["status"] == "unknown"
        assert (
            result["reason"]
            == {"absent": "receipt_absent", "error": "lookup_failed", "timeout": "lookup_timeout"}[
                failure
            ]
        )
        assert "secret" not in str(result)
    assert durable.get(run_id).action == before
    assert state.effects == ["original"]
    state.orch.turn_executor.execute_turn.assert_not_awaited()


@pytest.mark.parametrize("change", ["session", "authority", "capability", "registry", "binding"])
async def test_receipt_access_is_rechecked_after_lookup(tmp_path, monkeypatch, store, change):
    import json
    import sqlite3
    from copy import copy
    from victor.agent.action_observation import ActionStateError

    durable, backend, state = await committed_receipt_runtime(tmp_path, monkeypatch)
    original = backend.lookup
    tool = state.pipeline.executor.tools.get("submit_record")

    async def lookup(request):
        receipt = await original(request)
        if change == "session":
            state.orch.active_session_id = "other"
        elif change == "authority":
            state.pipeline.executor.current_user = "other"
        elif change == "capability":
            tool.action_recovery = ReceiptBackend(backend.path)
        elif change == "registry":
            replacement = copy(tool)
            monkeypatch.setattr(state.pipeline.executor.tools, "get", lambda name: replacement)
        else:
            pending = deepcopy(state.paused.pending_tool)
            pending["binding"]["payload"] = "0" * 64
            with sqlite3.connect(durable.db_path) as db:
                db.execute(
                    "UPDATE paused_run SET pending_tool = ? WHERE run_id = ?",
                    (json.dumps(pending), state.paused.run_id),
                )
        return receipt

    monkeypatch.setattr(backend, "lookup", lookup)
    with pytest.raises((PermissionError, ActionStateError)):
        await receipt_client(state.orch).reconcile_action(state.paused.run_id)
    assert durable.get(state.paused.run_id).action["backend_receipt"] is None
    assert state.effects == ["original"]


@pytest.mark.parametrize("conflict", [False, True])
async def test_concurrent_receipt_lookup_retains_one_immutable_result(
    tmp_path, monkeypatch, store, conflict
):
    import asyncio
    from dataclasses import replace
    from victor.agent.action_observation import ActionStateError

    durable, backend, state = await committed_receipt_runtime(tmp_path, monkeypatch)
    barrier = asyncio.Barrier(2)
    original = backend.lookup

    async def lookup(request):
        receipt = await original(request)
        index = await asyncio.wait_for(barrier.wait(), timeout=2)
        return replace(receipt, receipt_id=f"receipt:{index}") if conflict else receipt

    monkeypatch.setattr(backend, "lookup", lookup)
    client = receipt_client(state.orch)
    results = await asyncio.gather(
        client.reconcile_action(state.paused.run_id),
        client.reconcile_action(state.paused.run_id),
        return_exceptions=True,
    )
    assert sum(isinstance(result, ActionStateError) for result in results) == int(conflict)
    receipts = [
        result["action"]["backend_receipt"] for result in results if isinstance(result, dict)
    ]
    assert len(receipts) == 2 - int(conflict)
    assert all(
        receipt == durable.get(state.paused.run_id).action["backend_receipt"]
        for receipt in receipts
    )
    assert state.effects == ["original"] and not durable.mark_resumed(state.paused.run_id)


@pytest.mark.parametrize("failure", ["before", "after"])
async def test_receipt_storage_failure_never_replays_effect(tmp_path, monkeypatch, store, failure):
    durable, backend, state = await committed_receipt_runtime(tmp_path, monkeypatch)
    retain = durable.retain_receipt

    def fail(*args):
        if failure == "after":
            retain(*args)
        raise OSError("storage acknowledgement unavailable")

    monkeypatch.setattr(durable, "retain_receipt", fail)
    client = receipt_client(state.orch)
    with pytest.raises(OSError):
        await client.reconcile_action(state.paused.run_id)
    action = durable.get(state.paused.run_id).action
    assert (action["backend_receipt"] is not None) == (failure == "after")
    monkeypatch.setattr(durable, "retain_receipt", retain)
    assert (await client.reconcile_action(state.paused.run_id))["status"] == "verified"
    assert state.effects == ["original"]
    state.orch.turn_executor.execute_turn.assert_not_awaited()


async def test_legacy_action_has_no_recovery_capability(tmp_path, store):
    from victor.agent.paused_run_store import ProjectDbPausedRunStore

    durable = ProjectDbPausedRunStore(tmp_path / "actions.db")
    set_paused_run_store(durable)
    state = await paused_runtime(durable, effect_error=TimeoutError())
    assert durable.mark_resumed(state.paused.run_id)
    with pytest.raises(ResumeError):
        await resume_paused_run(
            state.orch, state.paused, ApprovalDecision(True), action_store=durable
        )
    result = await receipt_client(state.orch).reconcile_action(state.paused.run_id)
    assert result["status"] == "unsupported"
    assert result["action"]["version"] == 1 and "recovery" not in result["action"]
    assert state.effects == ["original"]


async def test_cancelled_receipt_write_can_commit_without_dispatch_or_disclosure(
    tmp_path, monkeypatch, store
):
    import asyncio
    import threading

    durable, backend, state = await committed_receipt_runtime(tmp_path, monkeypatch)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    retain = durable.retain_receipt

    def delayed(*args):
        entered.set()
        try:
            assert release.wait(3), "receipt worker release timed out"
            return retain(*args)
        finally:
            finished.set()

    monkeypatch.setattr(durable, "retain_receipt", delayed)
    client = receipt_client(state.orch)
    task = asyncio.create_task(client.reconcile_action(state.paused.run_id))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        assert await asyncio.to_thread(finished.wait, 2)
    assert durable.get(state.paused.run_id).action["backend_receipt"] is not None
    assert (await client.reconcile_action(state.paused.run_id))["status"] == "verified"
    assert backend.lookups == 1 and state.effects == ["original"]
    state.orch.turn_executor.execute_turn.assert_not_awaited()


@pytest.mark.parametrize(
    "corruption",
    ["legacy", "future", "boolean", "receipt_action", "receipt_binding", "receipt_backend"],
)
async def test_status_and_reconciliation_reject_corrupt_receipt_records(
    tmp_path, monkeypatch, store, corruption
):
    import json
    import sqlite3
    from victor.agent.action_observation import ActionStateError

    durable, backend, state = await committed_receipt_runtime(tmp_path, monkeypatch)
    client = receipt_client(state.orch)
    await client.reconcile_action(state.paused.run_id)
    action = durable.get(state.paused.run_id).action
    if corruption in {"legacy", "future", "boolean"}:
        action["version"] = {"legacy": 1, "future": 3, "boolean": True}[corruption]
    elif corruption == "receipt_action":
        action["backend_receipt"]["action_id"] = "0" * 64
    elif corruption == "receipt_binding":
        action["backend_receipt"]["binding_digest"] = "0" * 64
    else:
        action["backend_receipt"]["identity"]["backend_id"] = "wrong"
    with sqlite3.connect(durable.db_path) as db:
        db.execute(
            "UPDATE paused_run SET action_record = ? WHERE run_id = ?",
            (json.dumps(action), state.paused.run_id),
        )
    with pytest.raises((PermissionError, ActionStateError)):
        client.get_action_status(state.paused.run_id)
    with pytest.raises((PermissionError, ActionStateError)):
        await client.reconcile_action(state.paused.run_id)
    assert backend.lookups == 1


async def test_recovery_backend_change_after_approval_blocks_dispatch(tmp_path, store):
    from victor.agent.paused_run_store import ProjectDbPausedRunStore
    from victor.framework.action_recovery import RecoveryIdentity

    durable = ProjectDbPausedRunStore(tmp_path / "actions.db")
    set_paused_run_store(durable)
    backend = ReceiptBackend(tmp_path / "backend.db")
    state = await paused_runtime(durable, recovery=backend)
    backend.identity = RecoveryIdentity("sqlite-test-v1", "different-account")
    assert durable.mark_resumed(state.paused.run_id)
    with pytest.raises(ResumeError):
        await resume_paused_run(
            state.orch, state.paused, ApprovalDecision(True), action_store=durable
        )
    assert state.effects == []
    assert durable.get(state.paused.run_id).action is None
