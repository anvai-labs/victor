from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.agent.services.task_guidance_runtime import TaskGuidanceRuntime


def test_task_guidance_runtime_prepare_task_wires_reminder_manager_and_delegates():
    task_coordinator = MagicMock()
    task_coordinator._reminder_manager = None
    task_coordinator.prepare_task.return_value = ("classification", 7)
    runtime_host = SimpleNamespace(
        task_coordinator=task_coordinator,
        reminder_manager=MagicMock(name="reminder_manager"),
        conversation_controller=MagicMock(name="conversation_controller"),
    )

    runtime = TaskGuidanceRuntime(runtime_host)
    result = runtime.prepare_task("plan this", "analysis")

    assert result == ("classification", 7)
    task_coordinator.set_reminder_manager.assert_called_once_with(runtime_host.reminder_manager)
    task_coordinator.prepare_task.assert_called_once_with(
        "plan this",
        "analysis",
        runtime_host.conversation_controller,
    )


def test_task_guidance_runtime_apply_intent_guard_syncs_runtime_state():
    task_coordinator = MagicMock()
    task_coordinator.current_intent = "read_only"
    runtime_host = SimpleNamespace(
        task_coordinator=task_coordinator,
        conversation_controller=MagicMock(name="conversation_controller"),
        _current_intent=None,
        _current_user_message=None,
    )

    runtime = TaskGuidanceRuntime(runtime_host)
    runtime.apply_intent_guard("read this file")

    task_coordinator.apply_intent_guard.assert_called_once_with(
        "read this file",
        runtime_host.conversation_controller,
    )
    assert runtime_host._current_intent == "read_only"
    assert runtime_host._current_user_message == "read this file"


def test_task_guidance_runtime_classifies_with_pipeline_before_analyzer():
    pipeline = MagicMock()
    pipeline.classify_task_keywords.return_value = {"task_type": "analysis"}
    analyzer = MagicMock()
    runtime = TaskGuidanceRuntime(
        SimpleNamespace(_prompt_pipeline=pipeline, _task_analyzer=analyzer)
    )

    assert runtime.classify_task_keywords("inspect app.py") == {"task_type": "analysis"}
    pipeline.classify_task_keywords.assert_called_once_with("inspect app.py")
    analyzer.classify_task_keywords.assert_not_called()


def test_task_guidance_runtime_classification_fallback_is_explicit():
    runtime = TaskGuidanceRuntime(SimpleNamespace(_prompt_pipeline=None, _task_analyzer=None))

    assert runtime.classify_task_keywords("hello") == {
        "task_type": "default",
        "confidence": 0.0,
    }


def test_task_guidance_runtime_apply_intent_guard_preserves_explicit_sqlite_request_message():
    task_coordinator = MagicMock()
    task_coordinator.current_intent = "display_only"
    runtime_host = SimpleNamespace(
        task_coordinator=task_coordinator,
        conversation_controller=MagicMock(name="conversation_controller"),
        _current_intent=None,
        _current_user_message=None,
    )

    runtime = TaskGuidanceRuntime(runtime_host)
    runtime.apply_intent_guard("inspect the sqllite db directly")

    assert runtime_host._current_intent == "display_only"
    assert runtime_host._current_user_message == "inspect the sqllite db directly"


def test_task_guidance_runtime_apply_task_guidance_syncs_temperature_and_budget():
    task_coordinator = MagicMock()
    task_coordinator.temperature = 0.1
    task_coordinator.tool_budget = 5

    def _mutate_guidance(**kwargs):
        assert kwargs["conversation_controller"] is runtime_host.conversation_controller
        task_coordinator.temperature = 0.6
        task_coordinator.tool_budget = 25

    task_coordinator.apply_task_guidance.side_effect = _mutate_guidance

    runtime_host = SimpleNamespace(
        task_coordinator=task_coordinator,
        conversation_controller=MagicMock(name="conversation_controller"),
        temperature=0.3,
        tool_budget=10,
    )

    runtime = TaskGuidanceRuntime(runtime_host)
    runtime.apply_task_guidance(
        user_message="analyze architecture",
        unified_task_type="analysis",
        is_analysis_task=True,
        is_action_task=False,
        needs_execution=False,
        max_exploration_iterations=8,
    )

    assert task_coordinator.temperature == 0.6
    assert task_coordinator.tool_budget == 25
    assert runtime_host.temperature == 0.6
    assert runtime_host.tool_budget == 25
