# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for OrchestrationFacade canonical runtime surface."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.agent.facades.orchestration_facade import OrchestrationFacade
from victor.agent.facades.protocols import OrchestrationFacadeProtocol

REMOVED_COMPAT_PROPERTIES = (
    "chat_coordinator",
    "tool_coordinator",
    "session_coordinator",
    "sync_chat_coordinator",
    "streaming_chat_coordinator",
    "unified_chat_coordinator",
)


class TestOrchestrationFacadeInit:
    """Tests for OrchestrationFacade initialization."""

    def test_init_with_all_supported_components(self):
        adapter = MagicMock()
        analyzer = MagicMock()

        facade = OrchestrationFacade(
            interaction_runtime=MagicMock(),
            chat_service=MagicMock(),
            chat_stream_adapter=MagicMock(),
            tool_service=MagicMock(),
            session_service=MagicMock(),
            context_service=MagicMock(),
            provider_service=MagicMock(),
            recovery_service=MagicMock(),
            turn_executor=MagicMock(),
            protocol_adapter=adapter,
            streaming_handler=MagicMock(),
            streaming_controller=MagicMock(),
            streaming_coordinator=MagicMock(),
            iteration_coordinator=MagicMock(),
            task_analyzer=analyzer,
            exploration_state_passed=MagicMock(),
            system_prompt_state_passed=MagicMock(),
            safety_state_passed=MagicMock(),
            coordination_state_passed=MagicMock(),
            presentation=MagicMock(),
            vertical_integration_adapter=MagicMock(),
            vertical_context=MagicMock(),
            observability=MagicMock(),
            execution_tracer=MagicMock(),
            tool_call_tracer=MagicMock(),
            runtime_intelligence_integration=MagicMock(),
            subagent_orchestrator=MagicMock(),
        )

        assert facade.protocol_adapter is adapter
        assert facade.task_analyzer is analyzer

    def test_init_with_minimal_components(self):
        facade = OrchestrationFacade()

        assert facade.interaction_runtime is None
        assert facade.chat_service is None
        assert facade.chat_stream_adapter is None
        assert facade.tool_service is None
        assert facade.session_service is None
        assert facade.context_service is None
        assert facade.provider_service is None
        assert facade.recovery_service is None
        assert facade.turn_executor is None
        assert facade.protocol_adapter is None
        assert facade.streaming_handler is None
        assert facade.streaming_controller is None
        assert facade.streaming_coordinator is None
        assert facade.iteration_coordinator is None
        assert facade.task_analyzer is None
        assert facade.exploration_state_passed is None
        assert facade.system_prompt_state_passed is None
        assert facade.safety_state_passed is None
        assert facade.coordination_state_passed is None
        assert facade.presentation is None
        assert facade.vertical_integration_adapter is None
        assert facade.vertical_context is None
        assert facade.observability is None
        assert facade.execution_tracer is None
        assert facade.tool_call_tracer is None
        assert facade.runtime_intelligence_integration is None
        assert facade.subagent_orchestrator is None

    def test_removed_compatibility_properties_are_not_exposed(self):
        facade = OrchestrationFacade()

        for attr in REMOVED_COMPAT_PROPERTIES:
            assert hasattr(facade, attr) is False


class TestOrchestrationFacadeProperties:
    """Tests for canonical OrchestrationFacade properties."""

    def test_canonical_property_access(self):
        facade = OrchestrationFacade(
            interaction_runtime=MagicMock(name="interaction"),
            chat_service=MagicMock(name="chat_service"),
            chat_stream_adapter=MagicMock(name="chat_stream_adapter"),
            tool_service=MagicMock(name="tool_service"),
            session_service=MagicMock(name="session_service"),
            context_service=MagicMock(name="context_service"),
            provider_service=MagicMock(name="provider_service"),
            recovery_service=MagicMock(name="recovery_service"),
            turn_executor=MagicMock(name="turn_executor"),
            protocol_adapter=MagicMock(name="protocol_adapter"),
            streaming_handler=MagicMock(name="streaming_handler"),
            streaming_controller=MagicMock(name="streaming_controller"),
            streaming_coordinator=MagicMock(name="streaming_coordinator"),
            iteration_coordinator=MagicMock(name="iteration_coordinator"),
            task_analyzer=MagicMock(name="task_analyzer"),
            exploration_state_passed=MagicMock(name="exploration_state_passed"),
            system_prompt_state_passed=MagicMock(name="system_prompt_state_passed"),
            safety_state_passed=MagicMock(name="safety_state_passed"),
            coordination_state_passed=MagicMock(name="coordination_state_passed"),
            presentation=MagicMock(name="presentation"),
            vertical_integration_adapter=MagicMock(name="vertical_integration_adapter"),
            vertical_context=MagicMock(name="vertical_context"),
            observability=MagicMock(name="observability"),
            execution_tracer=MagicMock(name="execution_tracer"),
            tool_call_tracer=MagicMock(name="tool_call_tracer"),
            runtime_intelligence_integration=MagicMock(name="runtime_intelligence"),
            subagent_orchestrator=MagicMock(name="subagent_orchestrator"),
        )

        assert facade.interaction_runtime._mock_name == "interaction"
        assert facade.chat_service._mock_name == "chat_service"
        assert facade.chat_stream_adapter._mock_name == "chat_stream_adapter"
        assert facade.tool_service._mock_name == "tool_service"
        assert facade.session_service._mock_name == "session_service"
        assert facade.context_service._mock_name == "context_service"
        assert facade.provider_service._mock_name == "provider_service"
        assert facade.recovery_service._mock_name == "recovery_service"
        assert facade.turn_executor._mock_name == "turn_executor"
        assert facade.protocol_adapter._mock_name == "protocol_adapter"
        assert facade.streaming_handler._mock_name == "streaming_handler"
        assert facade.streaming_controller._mock_name == "streaming_controller"
        assert facade.streaming_coordinator._mock_name == "streaming_coordinator"
        assert facade.iteration_coordinator._mock_name == "iteration_coordinator"
        assert facade.task_analyzer._mock_name == "task_analyzer"
        assert facade.exploration_state_passed._mock_name == "exploration_state_passed"
        assert facade.system_prompt_state_passed._mock_name == "system_prompt_state_passed"
        assert facade.safety_state_passed._mock_name == "safety_state_passed"
        assert facade.coordination_state_passed._mock_name == "coordination_state_passed"
        assert facade.presentation._mock_name == "presentation"
        assert facade.vertical_integration_adapter._mock_name == "vertical_integration_adapter"
        assert facade.vertical_context._mock_name == "vertical_context"
        assert facade.observability._mock_name == "observability"
        assert facade.execution_tracer._mock_name == "execution_tracer"
        assert facade.tool_call_tracer._mock_name == "tool_call_tracer"
        assert facade.runtime_intelligence_integration._mock_name == "runtime_intelligence"
        assert facade.subagent_orchestrator._mock_name == "subagent_orchestrator"

    def test_lazy_getters_resolve_supported_runtime_surfaces(self):
        adapter = MagicMock(name="chat_stream_adapter")
        runtime_intelligence = MagicMock(name="runtime_intelligence")
        subagent = MagicMock(name="subagent")
        facade = OrchestrationFacade(
            get_chat_stream_adapter=lambda: adapter,
            get_runtime_intelligence_integration=lambda: runtime_intelligence,
            get_subagent_orchestrator=lambda: subagent,
        )

        assert facade.chat_stream_adapter is adapter
        assert facade.runtime_intelligence_integration is runtime_intelligence
        assert facade.subagent_orchestrator is subagent

    def test_supported_setters_update_surface(self):
        facade = OrchestrationFacade()
        protocol_adapter = MagicMock(name="protocol_adapter")
        turn_executor = MagicMock(name="turn_executor")
        iteration_coordinator = MagicMock(name="iteration_coordinator")
        observability = MagicMock(name="observability")
        runtime_intelligence = MagicMock(name="runtime_intelligence")
        subagent = MagicMock(name="subagent")

        facade.protocol_adapter = protocol_adapter
        facade.turn_executor = turn_executor
        facade.iteration_coordinator = iteration_coordinator
        facade.observability = observability
        facade.runtime_intelligence_integration = runtime_intelligence
        facade.subagent_orchestrator = subagent

        assert facade.protocol_adapter is protocol_adapter
        assert facade.turn_executor is turn_executor
        assert facade.iteration_coordinator is iteration_coordinator
        assert facade.observability is observability
        assert facade.runtime_intelligence_integration is runtime_intelligence
        assert facade.subagent_orchestrator is subagent


class TestRuntimeStateHostIntegration:
    """Tests that mutable orchestration state reads/writes through the host."""

    def test_runtime_state_host_keeps_supported_runtime_state_live(self):
        runtime_state_host = SimpleNamespace(
            _chat_stream_adapter=MagicMock(name="runtime_adapter"),
            _turn_executor=MagicMock(name="runtime_turn_executor"),
            _protocol_adapter=MagicMock(name="runtime_protocol_adapter"),
            _iteration_coordinator=MagicMock(name="runtime_iteration"),
            _observability=MagicMock(name="runtime_observability"),
            _runtime_intelligence_integration=MagicMock(name="runtime_intelligence"),
            _subagent_orchestrator=MagicMock(name="runtime_subagent"),
        )
        facade = OrchestrationFacade(
            chat_stream_adapter=MagicMock(name="stale_adapter"),
            turn_executor=MagicMock(name="stale_turn_executor"),
            protocol_adapter=MagicMock(name="stale_protocol_adapter"),
            iteration_coordinator=MagicMock(name="stale_iteration"),
            observability=MagicMock(name="stale_observability"),
            runtime_intelligence_integration=MagicMock(name="stale_intelligence"),
            subagent_orchestrator=MagicMock(name="stale_subagent"),
            runtime_state_host=runtime_state_host,
        )

        assert facade.chat_stream_adapter is runtime_state_host._chat_stream_adapter
        assert facade.turn_executor is runtime_state_host._turn_executor
        assert facade.protocol_adapter is runtime_state_host._protocol_adapter
        assert facade.iteration_coordinator is runtime_state_host._iteration_coordinator
        assert facade.observability is runtime_state_host._observability
        assert (
            facade.runtime_intelligence_integration
            is runtime_state_host._runtime_intelligence_integration
        )
        assert facade.subagent_orchestrator is runtime_state_host._subagent_orchestrator

        runtime_state_host._chat_stream_adapter = MagicMock(name="updated_adapter")
        runtime_state_host._turn_executor = MagicMock(name="updated_turn_executor")
        runtime_state_host._protocol_adapter = MagicMock(name="updated_protocol_adapter")
        runtime_state_host._iteration_coordinator = MagicMock(name="updated_iteration")
        runtime_state_host._observability = MagicMock(name="updated_observability")
        runtime_state_host._runtime_intelligence_integration = MagicMock(
            name="updated_intelligence"
        )
        runtime_state_host._subagent_orchestrator = MagicMock(name="updated_subagent")

        assert facade.chat_stream_adapter is runtime_state_host._chat_stream_adapter
        assert facade.turn_executor is runtime_state_host._turn_executor
        assert facade.protocol_adapter is runtime_state_host._protocol_adapter
        assert facade.iteration_coordinator is runtime_state_host._iteration_coordinator
        assert facade.observability is runtime_state_host._observability
        assert (
            facade.runtime_intelligence_integration
            is runtime_state_host._runtime_intelligence_integration
        )
        assert facade.subagent_orchestrator is runtime_state_host._subagent_orchestrator

    def test_runtime_state_host_setters_update_canonical_orchestration_state(self):
        runtime_state_host = SimpleNamespace(
            _turn_executor=MagicMock(name="runtime_turn_executor"),
            _protocol_adapter=MagicMock(name="runtime_protocol_adapter"),
            _iteration_coordinator=MagicMock(name="runtime_iteration"),
            _observability=MagicMock(name="runtime_observability"),
            _runtime_intelligence_integration=MagicMock(name="runtime_intelligence"),
            _subagent_orchestrator=MagicMock(name="runtime_subagent"),
        )
        facade = OrchestrationFacade(runtime_state_host=runtime_state_host)
        new_turn_executor = MagicMock(name="new_turn_executor")
        new_protocol_adapter = MagicMock(name="new_protocol_adapter")
        new_iteration = MagicMock(name="new_iteration")
        new_observability = MagicMock(name="new_observability")
        new_intelligence = MagicMock(name="new_intelligence")
        new_subagent = MagicMock(name="new_subagent")

        facade.turn_executor = new_turn_executor
        facade.protocol_adapter = new_protocol_adapter
        facade.iteration_coordinator = new_iteration
        facade.observability = new_observability
        facade.runtime_intelligence_integration = new_intelligence
        facade.subagent_orchestrator = new_subagent

        assert runtime_state_host._turn_executor is new_turn_executor
        assert runtime_state_host._protocol_adapter is new_protocol_adapter
        assert runtime_state_host._iteration_coordinator is new_iteration
        assert runtime_state_host._observability is new_observability
        assert runtime_state_host._runtime_intelligence_integration is new_intelligence
        assert runtime_state_host._subagent_orchestrator is new_subagent


class TestOrchestrationFacadeProtocolConformance:
    """Tests that OrchestrationFacade satisfies OrchestrationFacadeProtocol."""

    def test_satisfies_protocol(self):
        assert isinstance(OrchestrationFacade(), OrchestrationFacadeProtocol)

    def test_protocol_properties_present(self):
        required = [
            "protocol_adapter",
            "chat_stream_adapter",
            "task_analyzer",
            "exploration_state_passed",
            "system_prompt_state_passed",
            "safety_state_passed",
            "coordination_state_passed",
        ]
        facade = OrchestrationFacade()
        for prop in required:
            assert hasattr(facade, prop), f"Missing protocol property: {prop}"
