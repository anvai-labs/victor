from types import SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

from victor.agent.orchestrator import AgentOrchestrator


def test_initialize_interaction_runtime_uses_resolved_runtime_service_bundle():
    interaction_runtime = SimpleNamespace(
        chat_service=MagicMock(name="chat_service"),
        tool_service=MagicMock(name="tool_service"),
        session_service=MagicMock(name="session_service"),
        context_service=MagicMock(name="context_service"),
        recovery_service=MagicMock(name="recovery_service"),
    )
    orchestrator = SimpleNamespace(
        _factory=MagicMock(),
        _enabled_tools=["shell"],
        _tool_pipeline=MagicMock(),
        tools=MagicMock(),
        tool_executor=MagicMock(),
        tool_cache=MagicMock(),
        tool_budget=12,
        tool_selector=MagicMock(),
        _tool_access_controller=MagicMock(),
        mode_controller=MagicMock(),
        argument_normalizer=MagicMock(),
        _session_state=MagicMock(),
        _lifecycle_manager=MagicMock(),
        memory_manager=MagicMock(),
        _memory_session_id="session-1",
        _checkpoint_manager=MagicMock(),
        _session_cost_tracker=MagicMock(),
        _conversation_controller=MagicMock(),
        _streaming_controller=MagicMock(),
        settings=sentinel.settings,
    )

    with patch(
        "victor.runtime.context.resolve_runtime_services",
        return_value=sentinel.runtime_services,
    ) as mock_resolve:
        with patch(
            "victor.agent.runtime.interaction_runtime.create_interaction_runtime_components",
            return_value=interaction_runtime,
        ) as mock_create:
            AgentOrchestrator._initialize_interaction_runtime(orchestrator)

    mock_resolve.assert_called_once_with(orchestrator)
    assert mock_create.call_args.kwargs["runtime_services"] is sentinel.runtime_services
    assert mock_create.call_args.kwargs["settings"] is sentinel.settings
    assert orchestrator._interaction_runtime is interaction_runtime
    assert orchestrator._chat_service is interaction_runtime.chat_service
    assert orchestrator._tool_service is interaction_runtime.tool_service
    assert orchestrator._session_service is interaction_runtime.session_service
    assert orchestrator._context_service is interaction_runtime.context_service
    assert orchestrator._recovery_service is interaction_runtime.recovery_service
