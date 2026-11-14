"""Run-policy admission and durable pause evidence (FEP-0044 Phase 1)."""

from types import SimpleNamespace

import pytest

from victor.agent.paused_run_store import (
    InMemoryPausedRunStore,
    reset_paused_run_store,
    set_paused_run_store,
)
from victor.config.sandbox_settings import SandboxSettings
from victor.framework.approval_pause import ApprovalPause
from victor.runtime.policy_snapshot import RunPolicySnapshot, current_run_policy_snapshot


class FakeExecutionContext:
    def __init__(self, chat):
        self.services = SimpleNamespace(chat=chat)
        self.run_policy_snapshot = None
        self.admitted_policy = None

    def with_run_policy(self, snapshot: RunPolicySnapshot):
        self.admitted_policy = snapshot
        cloned = FakeExecutionContext(self.services.chat)
        cloned.run_policy_snapshot = snapshot
        return cloned


class ApprovalRequestFixture:
    def __init__(self):
        self.context = {
            "tool_name": "write",
            "arguments": {"path": "evidence.txt"},
            "action_binding": None,
        }

    def to_dict(self):
        return {
            "id": "approval-1",
            "created_at": 1.0,
            "timeout_seconds": 60.0,
            "context": self.context,
        }


@pytest.fixture
def store():
    value = InMemoryPausedRunStore()
    set_paused_run_store(value)
    yield value
    reset_paused_run_store()


def _orchestrator(policy: str = "required"):
    return SimpleNamespace(
        active_session_id="session-1",
        agent_id="agent-1",
        model="test-model",
        current_stage=SimpleNamespace(value="ACT"),
        settings=SimpleNamespace(
            governance=SimpleNamespace(durable=True),
            sandbox=SandboxSettings(
                sandbox_policy=policy,
                sandbox_network_isolation=True,
                sandbox_allowed_mounts=["/workspace"],
            ),
        ),
    )


@pytest.mark.asyncio
async def test_buffered_pause_persists_admitted_policy_snapshot(store) -> None:
    from victor.framework.message_execution import execute_message

    async def chat(message, **kwargs):
        raise ApprovalPause(ApprovalRequestFixture())

    context = FakeExecutionContext(SimpleNamespace(chat=chat))
    result = await execute_message(
        orchestrator=_orchestrator(),
        execution_context=context,
        user_message="write evidence",
    )

    assert result.status == "awaiting_approval"
    snapshot = RunPolicySnapshot.from_settings(_orchestrator().settings)
    assert context.admitted_policy == snapshot
    paused = store.get(result.run_id)
    assert paused is not None
    assert paused.metadata["run_policy"] == snapshot.to_metadata()


@pytest.mark.asyncio
async def test_stream_pause_persists_admitted_policy_snapshot(store) -> None:
    from victor.framework.message_execution import stream_message_events

    request = ApprovalRequestFixture()

    class StreamChat:
        async def stream_chat(self, message, **kwargs):
            raise ApprovalPause(request)
            yield  # pragma: no cover - async-generator shape

    context = FakeExecutionContext(SimpleNamespace(stream_chat=StreamChat().stream_chat))
    events = [
        event
        async for event in stream_message_events(
            orchestrator=_orchestrator(),
            execution_context=context,
            user_message="write evidence",
        )
    ]

    awaiting = [event for event in events if event.event_type == "awaiting_approval"]
    assert len(awaiting) == 1
    paused = store.get(awaiting[0].metadata["run_id"])
    assert paused is not None
    assert paused.metadata["run_policy"]["effective_policy"] == "required"


@pytest.mark.asyncio
async def test_buffered_execution_context_carries_admitted_policy() -> None:
    from victor.framework.message_execution import execute_message

    captured = {}

    async def chat(message, **kwargs):
        captured["policy"] = current_run_policy_snapshot()
        raise ApprovalPause(ApprovalRequestFixture())

    await execute_message(
        orchestrator=_orchestrator(),
        execution_context=FakeExecutionContext(SimpleNamespace(chat=chat)),
        user_message="write evidence",
    )

    assert isinstance(captured["policy"], RunPolicySnapshot)
    assert captured["policy"].effective_policy == "required"


@pytest.mark.asyncio
async def test_stream_execution_context_carries_admitted_policy(store) -> None:
    from victor.framework.message_execution import stream_message_events

    request = ApprovalRequestFixture()
    captured = {}

    class StreamChat:
        async def stream_chat(self, message, **kwargs):
            captured["policy"] = current_run_policy_snapshot()
            raise ApprovalPause(request)
            yield  # pragma: no cover - async-generator shape

    context = FakeExecutionContext(SimpleNamespace(stream_chat=StreamChat().stream_chat))
    [
        _
        async for _ in stream_message_events(
            orchestrator=_orchestrator(),
            execution_context=context,
            user_message="write evidence",
        )
    ]

    assert isinstance(captured["policy"], RunPolicySnapshot)
    assert captured["policy"].effective_policy == "required"


@pytest.mark.asyncio
async def test_unreadable_policy_fails_admission_before_model_work(store) -> None:
    from victor.framework.message_execution import execute_message

    class ExplodingSettings:
        @property
        def governance(self):
            return SimpleNamespace(durable=True)

        @property
        def sandbox(self):
            raise OSError("settings unavailable")

    orchestrator = SimpleNamespace(
        active_session_id="session-1",
        agent_id="agent-1",
        model="test-model",
        current_stage=SimpleNamespace(value="ACT"),
        settings=ExplodingSettings(),
    )
    context = FakeExecutionContext(
        SimpleNamespace(chat=AssertionError("chat runtime must not be resolved"))
    )
    result = await execute_message(
        orchestrator=orchestrator,
        execution_context=context,
        user_message="write evidence",
    )

    assert not result.success
    assert "run policy" in result.error.lower()
    assert list(store.list_pending()) == []
