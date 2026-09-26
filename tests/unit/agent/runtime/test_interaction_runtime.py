from unittest.mock import MagicMock

from types import SimpleNamespace

from victor.agent.runtime.interaction_runtime import (
    create_interaction_runtime_components,
)
from victor.runtime.context import ResolvedRuntimeServices


def _runtime_kwargs():
    return {
        "enabled_tools": ["shell"],
        "factory": MagicMock(),
        "context_compactor": MagicMock(),
        "tool_pipeline": MagicMock(),
        "tool_registry": MagicMock(),
        "tool_executor": MagicMock(),
        "tool_cache": MagicMock(),
        "tool_budget": 12,
        "tool_selector": MagicMock(),
        "tool_access_controller": MagicMock(),
        "mode_controller": MagicMock(),
        "argument_normalizer": MagicMock(),
        "session_state_manager": MagicMock(),
        "lifecycle_manager": MagicMock(),
        "memory_manager": MagicMock(),
        "memory_session_id": "session-1",
        "checkpoint_manager": MagicMock(),
        "cost_tracker": MagicMock(),
        "conversation_controller": MagicMock(),
        "streaming_coordinator": MagicMock(),
        "settings": MagicMock(name="settings"),
    }


def test_create_interaction_runtime_components_prefers_runtime_service_bundle():
    tool_service = MagicMock(name="tool_service")
    session_service = MagicMock(name="session_service")
    context_service = MagicMock(name="context_service")
    recovery_service = MagicMock(name="recovery_service")
    provider_service = MagicMock(name="provider_service")
    chat_service = MagicMock(name="chat_service")

    components = create_interaction_runtime_components(
        runtime_services=ResolvedRuntimeServices(
            chat=chat_service,
            tool=tool_service,
            session=session_service,
            context=context_service,
            provider=provider_service,
            recovery=recovery_service,
        ),
        **_runtime_kwargs(),
    )

    assert components.chat_service is chat_service
    assert components.tool_service is tool_service
    assert components.session_service is session_service
    assert components.context_service is context_service
    assert components.recovery_service is recovery_service
    tool_service.bind_runtime_components.assert_called_once()
    tool_service.set_enabled_tools.assert_called_once_with(["shell"])
    session_service.bind_runtime_components.assert_called_once()


def test_resolved_tool_service_keeps_owner_settings_and_fixed_binder_contract():
    from victor.agent.services.tool_service import ToolService, ToolServiceConfig

    owner_settings = SimpleNamespace(tools=SimpleNamespace(tool_selection_enabled=True))
    later_settings = SimpleNamespace(tools=SimpleNamespace(tool_selection_enabled=False))
    service = ToolService(
        ToolServiceConfig(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        settings=owner_settings,
    )

    first_kwargs = _runtime_kwargs()
    first_kwargs["settings"] = owner_settings
    first = create_interaction_runtime_components(
        runtime_services=ResolvedRuntimeServices(tool=service),
        **first_kwargs,
    )
    second_kwargs = _runtime_kwargs()
    second_kwargs["settings"] = later_settings
    second = create_interaction_runtime_components(
        runtime_services=ResolvedRuntimeServices(tool=service),
        **second_kwargs,
    )

    assert first.tool_service is second.tool_service is service
    assert service._settings is owner_settings

    class FixedBinderService:
        def bind_runtime_components(
            self,
            *,
            tool_registry=None,
            tool_pipeline=None,
            tool_cache=None,
            mode_controller=None,
            argument_normalizer=None,
        ):
            self.bound = True

    fixed_service = FixedBinderService()
    create_interaction_runtime_components(
        runtime_services=ResolvedRuntimeServices(tool=fixed_service),
        **_runtime_kwargs(),
    )
    assert fixed_service.bound is True


def test_create_interaction_runtime_components_uses_context_adapter_fallback():
    from victor.agent.services.adapters.context_adapter import ContextServiceAdapter

    kwargs = _runtime_kwargs()
    components = create_interaction_runtime_components(
        runtime_services=ResolvedRuntimeServices(),
        **kwargs,
    )

    assert isinstance(components.context_service, ContextServiceAdapter)
    assert components.tool_service._settings is kwargs["settings"]


def test_create_interaction_runtime_components_passes_context_compactor_to_adapter():
    from victor.agent.services.adapters.context_adapter import ContextServiceAdapter

    kwargs = _runtime_kwargs()
    compactor = kwargs["context_compactor"]
    components = create_interaction_runtime_components(
        runtime_services=ResolvedRuntimeServices(),
        **kwargs,
    )

    assert isinstance(components.context_service, ContextServiceAdapter)
    assert components.context_service._context_compactor is compactor
