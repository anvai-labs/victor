# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from unittest.mock import MagicMock

from victor.agent.conversation.controller import ConversationController
from victor.agent.conversation.state_machine import ConversationStage
from victor.agent.service_provider import OrchestratorServiceProvider
from victor.agent.services.protocols import StateRuntimeProtocol
from victor.core.container import ServiceContainer, ServiceLifetime


def _mock_settings():
    settings = MagicMock()
    settings.search.unified_embedding_model = "test-model"
    settings.enable_observability = True
    settings.max_conversation_history = 50
    return settings


def test_state_runtime_adapter_uses_real_conversation_state():
    from victor.agent.services.state_runtime import StateRuntimeAdapter

    controller = ConversationController()
    adapter = StateRuntimeAdapter(conversation_controller=controller)

    assert isinstance(adapter, StateRuntimeProtocol)
    assert adapter.get_current_stage() == ConversationStage.INITIAL
    assert adapter.transition_to(ConversationStage.PLANNING, reason="planning")
    assert adapter.get_current_stage() == ConversationStage.PLANNING

    controller.add_message("system", "system")
    controller.add_message("user", "inspect file")
    controller.add_message("assistant", "working")

    assert adapter.is_in_exploration_phase() is True
    assert adapter.is_in_execution_phase() is False
    assert len(adapter.get_message_history()) == 3
    assert len(adapter.get_recent_messages(limit=2, include_system=False)) == 2


def test_state_runtime_protocol_resolves_to_canonical_adapter_when_controller_registered():
    from victor.agent.protocols import ConversationControllerProtocol
    from victor.agent.services.state_runtime import StateRuntimeAdapter

    container = ServiceContainer()
    provider = OrchestratorServiceProvider(_mock_settings())
    provider.register_singleton_services(container)

    controller = ConversationController()
    container.register(
        ConversationControllerProtocol,
        lambda c: controller,
        ServiceLifetime.SINGLETON,
    )

    with container.create_scope() as scope:
        runtime = scope.get(StateRuntimeProtocol)

    assert isinstance(runtime, StateRuntimeAdapter)
    assert runtime.get_current_stage() == ConversationStage.INITIAL
