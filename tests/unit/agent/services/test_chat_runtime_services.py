"""The first chat-runtime capability mutates its existing session owner."""

from dataclasses import fields, FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.factory.chat_runtime_bindings import bind_chat_runtime_services
from victor.agent.factory.runtime_builders import RuntimeBuildersMixin
from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.services.chat_runtime_services import (
    ChatCompletion,
    ChatConversation,
    ChatFeedback,
    ChatGovernance,
    ChatRuntimeServices,
    ChatStreamLifecycle,
    ChatToolCalls,
)
from victor.agent.services.chat_stream_executor import StreamingChatExecutor
from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter
from victor.agent.services.streaming_act_adapter import StreamingActAdapter
from victor.agent.services.task_guidance_runtime import TaskGuidanceRuntime
from victor.agent.services.tool_selection_runtime import ToolSelectionRuntime
from victor.agent.session_state_accessor import SessionStateAccessor
from victor.agent.session_state_manager import SessionStateManager


def owner():
    # Exercise the real facade's existing session properties without provider setup.
    result = object.__new__(AgentOrchestrator)
    result._session_accessor = SessionStateAccessor(SessionStateManager())
    result._tool_planner = SimpleNamespace()
    return result


@pytest.mark.parametrize("through_protocol_adapter", [False, True])
async def test_factory_view_updates_existing_requirement_owner(
    through_protocol_adapter, monkeypatch
):
    orchestrator = owner()
    accessor = orchestrator._session_accessor
    orchestrator._required_files = ["old.py"]
    orchestrator._required_outputs = ["old output"]
    orchestrator._read_files_session.add("old.py")
    orchestrator._all_files_read_nudge_sent = True
    original_read_set = orchestrator._read_files_session
    supplied = (
        OrchestratorProtocolAdapter(orchestrator) if through_protocol_adapter else orchestrator
    )
    runtime = RuntimeBuildersMixin().create_streaming_chat_adapter(supplied)
    executor = StreamingChatExecutor(runtime)
    assert executor.services is runtime.services
    assert isinstance(executor.services.planning.guidance, TaskGuidanceRuntime)
    assert isinstance(executor.services.planning.selection, ToolSelectionRuntime)
    assert executor.services.planning.planner is orchestrator._tool_planner
    session = executor.services.session
    assert session.required_files is accessor.required_files
    assert session.read_files is original_read_set
    bus = SimpleNamespace(emit=AsyncMock())
    monkeypatch.setattr("victor.core.events.get_observability_bus", lambda: bus)

    await executor._extract_task_requirements(session, "Read ./src/main.py and ./src/main.py")

    assert orchestrator._required_files == ["./src/main.py"]
    assert session.required_files is orchestrator._required_files
    assert orchestrator._required_outputs == []
    assert original_read_set == set()
    assert orchestrator._all_files_read_nudge_sent is False
    bus.emit.assert_awaited_once_with(
        topic="state.task.requirements_extracted",
        data={
            "required_files": ["./src/main.py"],
            "required_outputs": [],
            "file_count": 1,
            "output_count": 0,
            "category": "state",
        },
    )


async def test_view_tracks_restored_and_reset_state_without_sharing_owners(monkeypatch):
    first, second = owner(), owner()
    view = bind_chat_runtime_services(first)
    same_owner_view = bind_chat_runtime_services(first)
    other = bind_chat_runtime_services(second)
    assert same_owner_view.stream_turn_lock is view.stream_turn_lock
    assert other.stream_turn_lock is not view.stream_turn_lock
    manager = first._session_accessor.session_state
    other.session.required_files = ["unrelated.py"]
    other.session.read_files.add("unrelated.py")
    other.session.all_files_read_nudge_sent = True
    first._required_files = ["saved.py"]
    first._read_files_session.add("saved.py")
    first._all_files_read_nudge_sent = True
    checkpoint = manager.get_checkpoint_state()
    old_files = view.session.required_files
    old_reads = view.session.read_files
    manager.reset_for_new_turn()
    manager.apply_checkpoint_state(checkpoint)
    assert view.session.required_files == ["saved.py"]
    assert view.session.required_files is first._required_files
    assert view.session.required_files is not old_files
    assert view.session.read_files is first._read_files_session
    assert view.session.read_files is not old_reads
    assert view.session.all_files_read_nudge_sent is True
    bus = SimpleNamespace(emit=AsyncMock())
    monkeypatch.setattr("victor.core.events.get_observability_bus", lambda: bus)

    await StreamingChatExecutor._extract_task_requirements(view.session, "Read ../next.py")
    assert first._required_files == ["../next.py"]
    assert first._read_files_session == set()
    assert first._all_files_read_nudge_sent is False
    await StreamingChatExecutor._extract_task_requirements(view.session, "plain next turn")
    assert first._required_files == []
    assert first._required_outputs == []
    assert bus.emit.await_count == 1  # no event for empty requirements
    assert other.session.required_files == ["unrelated.py"]
    assert other.session.read_files == {"unrelated.py"}
    assert other.session.all_files_read_nudge_sent is True


async def test_act_prepare_uses_bound_session_capability(monkeypatch):
    orchestrator = owner()
    orchestrator._recovery_service = None
    orchestrator._recovery_coordinator = object()
    runtime = RuntimeBuildersMixin().create_streaming_chat_adapter(orchestrator)
    executor = StreamingChatExecutor(runtime)
    ctx = SimpleNamespace(max_exploration_iterations=4)
    monkeypatch.setattr(runtime, "_create_stream_context", AsyncMock(return_value=ctx))
    monkeypatch.setattr(executor, "_reset_streaming_turn_state", lambda _: None)
    monkeypatch.setattr(executor, "_apply_run_guidance", lambda *args: None)
    monkeypatch.setattr(executor, "_initialize_task_intent", lambda *args: [])
    bus = SimpleNamespace(emit=AsyncMock())
    monkeypatch.setattr("victor.core.events.get_observability_bus", lambda: bus)

    adapter = await StreamingActAdapter.prepare(executor, "Inspect ./live.py")

    assert adapter.session.stream_ctx is ctx
    assert orchestrator._required_files == ["./live.py"]
    assert executor.services.session.required_files is orchestrator._required_files
    bus.emit.assert_awaited_once()


def test_view_is_enumerated_and_does_not_retain_or_forward_facade():
    orchestrator = owner()
    view = bind_chat_runtime_services(orchestrator)
    assert [field.name for field in fields(view)] == [
        "session",
        "stream_lifecycle",
        "stream_turn_lock",
        "delivery",
        "planning",
        "governance",
        "completion",
        "conversation",
        "tool_calls",
        "feedback",
        "recovery",
    ]
    assert not hasattr(view, "__dict__")
    assert not hasattr(view.session, "__dict__")
    for name in ("orchestrator", "_orchestrator", "runtime_owner", "state_host", "get", "settings"):
        assert not hasattr(view, name)
        assert not hasattr(view.session, name)
    assert view.stream_turn_lock is orchestrator._session_accessor.stream_turn_lock
    assert isinstance(view.stream_lifecycle, ChatStreamLifecycle)
    assert view.session._accessor is orchestrator._session_accessor
    with pytest.raises(FrozenInstanceError):
        view.session = object()
    with pytest.raises(AttributeError):
        view.session._orchestrator = orchestrator


def test_missing_session_owner_requires_explicit_binding():
    with pytest.raises(TypeError, match="SessionStateAccessor"):
        bind_chat_runtime_services(SimpleNamespace())


class _CompletionDetector:
    def __init__(self, confidence, summary=""):
        self.confidence = confidence
        self.state = SimpleNamespace(last_summary=summary)
        self.clear_active_signal = MagicMock()
        self.reset = MagicMock()

    def analyze_response(self, content):
        self.content = content

    def get_completion_confidence(self):
        return self.confidence

    def get_state(self):
        return self.state


def test_completion_and_conversation_capabilities_persist_sanitized_summary():
    from victor.agent.task_completion import CompletionConfidence

    detector = _CompletionDetector(
        CompletionConfidence.HIGH,
        "VICTOR_SUMMARY:: changed app.py",
    )
    runtime = SimpleNamespace(persist_terminal_summary=MagicMock())
    completion = ChatCompletion(detector=detector)
    conversation = ChatConversation(runtime=runtime)

    assert completion.detect_high_confidence("done", has_pending_tools=False) is True
    assert completion.terminal_summary() == "changed app.py"
    conversation.persist_terminal_summary(completion.terminal_summary())
    runtime.persist_terminal_summary.assert_called_once_with("changed app.py")


def test_completion_capability_defers_pending_tools_without_persisting():
    from victor.agent.task_completion import CompletionConfidence

    detector = _CompletionDetector(CompletionConfidence.HIGH, "VICTOR_SUMMARY:: premature")
    completion = ChatCompletion(detector=detector)

    assert completion.detect_high_confidence("done", has_pending_tools=True) is False
    detector.clear_active_signal.assert_called_once_with()


def test_conversation_capability_delegates_history_and_usage():
    runtime = SimpleNamespace(
        messages=MagicMock(return_value=["first", "second"]),
        record_actual_usage=MagicMock(),
    )
    conversation = ChatConversation(runtime=runtime)

    assert conversation.messages() == ["first", "second"]
    conversation.record_actual_usage(17)

    runtime.messages.assert_called_once_with()
    runtime.record_actual_usage.assert_called_once_with(17)


@pytest.mark.parametrize("invalid_result", [None, object()], ids=["none", "malformed"])
@pytest.mark.parametrize("method_name", ["check_request", "check_response"])
async def test_governance_rejects_invalid_result_from_configured_gate(method_name, invalid_result):
    gate = SimpleNamespace(
        gate_request=AsyncMock(return_value=invalid_result),
        gate_response=AsyncMock(return_value=invalid_result),
    )

    with pytest.raises(TypeError, match=f"invalid {method_name.removeprefix('check_')} result"):
        await getattr(ChatGovernance(gate=gate), method_name)("sensitive content")


def test_tool_call_capability_fails_closed_without_runtime():
    with pytest.raises(TypeError, match="requires a runtime"):
        ChatToolCalls().parse_and_validate(None, "tool content")


def test_feedback_capability_delegates_only_declared_outcome_fields():
    recorder = SimpleNamespace(record_outcome=MagicMock())
    feedback = ChatFeedback(recorder=recorder)

    feedback.record_outcome(
        success=False,
        quality_score=0.3,
        user_satisfied=False,
        completed=False,
    )

    recorder.record_outcome.assert_called_once_with(
        success=False,
        quality_score=0.3,
        user_satisfied=False,
        completed=False,
    )
