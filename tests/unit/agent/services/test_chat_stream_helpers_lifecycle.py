from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.chat_stream_helpers import ChatStreamHelperMixin
from victor.agent.services.chat_planning import ChatPlanning
from victor.agent.unified_task_tracker import TrackerTaskType


class _Helper(ChatStreamHelperMixin):
    def __init__(self, orchestrator):
        self._orchestrator = orchestrator


@pytest.mark.asyncio
async def test_stream_preparation_delegates_task_preparation_to_planning(monkeypatch):
    from victor.agent.services import chat_stream_helpers

    monkeypatch.setattr(
        chat_stream_helpers,
        "extract_prompt_requirements",
        lambda _message: SimpleNamespace(has_explicit_requirements=lambda: False),
    )
    monkeypatch.setattr(
        chat_stream_helpers,
        "classify_direct_response_prompt",
        lambda _message: SimpleNamespace(is_direct_response=True),
    )
    classification = object()
    guidance = SimpleNamespace(
        prepare_task=MagicMock(return_value=(classification, 17)),
    )
    task_state = MagicMock()
    task_state.max_total_iterations.return_value = 50
    task_state.detect_task_type.return_value = TrackerTaskType.EDIT
    task_state.max_exploration_iterations.return_value = 8
    orch = SimpleNamespace(
        cancelled=False,
        add_message=MagicMock(),
        _record_runtime_intelligence_outcome=MagicMock(),
    )
    helper = _Helper(orch)
    helper.services = SimpleNamespace(
        stream_lifecycle=SimpleNamespace(begin=MagicMock()),
        metrics=SimpleNamespace(begin=MagicMock(return_value=SimpleNamespace(start_time=1.0))),
        conversation=SimpleNamespace(ensure_system_prompt=MagicMock()),
        task_state=task_state,
        context_lifecycle=SimpleNamespace(start_background_compaction=AsyncMock()),
        planning=ChatPlanning(guidance=guidance),
        intelligence=SimpleNamespace(prepare_request=AsyncMock()),
    )
    helper._get_runtime_capability_value = MagicMock(return_value=None)
    helper._has_runtime_capability = MagicMock(return_value=False)
    helper._resolve_continuation_task_context = MagicMock(return_value=None)

    prepared = await helper._prepare_stream("update app.py")

    assert prepared[9:] == (classification, 17)
    guidance.prepare_task.assert_called_once_with("update app.py", TrackerTaskType.EDIT)
