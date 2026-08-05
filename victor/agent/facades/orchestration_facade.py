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

"""Orchestration domain facade for orchestrator decomposition.

Groups coordinators, protocol adapters, streaming handlers, intelligent
pipeline integration, and subagent orchestration components behind a
single interface.

This facade wraps already-initialized components from the orchestrator,
providing a coherent grouping without changing initialization ordering.
The orchestrator delegates property access through this facade. It does not own
runtime behavior; it only groups canonical services and state-passed surfaces.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class OrchestrationFacade:
    """Groups top-level orchestration coordinators and integration components.

    Satisfies ``OrchestrationFacadeProtocol`` structurally.  The orchestrator
    creates this facade after all orchestration-domain components are
    initialized, passing references to the already-created instances.

    Components managed:
        - interaction_runtime: Interaction runtime boundary components
        - chat_service: Canonical chat service
        - chat_stream_adapter: Canonical service-owned chat-stream adapter
        - tool_service: Canonical tool service
        - session_service: Canonical session service
        - context_service: Canonical context service
        - provider_service: Canonical provider service
        - recovery_service: Canonical recovery service
        - turn_executor: Execution coordinator for agentic loop
        - protocol_adapter: Protocol adapter for coordinator communication
        - streaming_handler: StreamingChatHandler for streaming loop logic
        - streaming_controller: StreamingController for session/metrics
        - streaming_coordinator: StreamingCoordinator for response processing
        - iteration_coordinator: IterationCoordinator for loop control
        - task_analyzer: Unified task analysis facade
        - exploration_state_passed: State-passed exploration coordinator
        - system_prompt_state_passed: State-passed system prompt coordinator
        - safety_state_passed: State-passed safety coordinator
        - coordination_state_passed: State-passed coordination suggestion coordinator
        - presentation: Presentation adapter for icon/emoji rendering
        - vertical_integration_adapter: VerticalIntegrationAdapter
        - vertical_context: VerticalContext for unified vertical state
        - observability: ObservabilityIntegration for unified event bus
        - execution_tracer: Execution tracing (if available)
        - tool_call_tracer: Tool call tracing (if available)
        - runtime_state_host: Canonical runtime owner for mutable orchestration state
        - runtime_intelligence_integration: Runtime-intelligence integration
        - subagent_orchestrator: Sub-agent orchestration
    """

    def __init__(
        self,
        *,
        interaction_runtime: Optional[Any] = None,
        chat_service: Optional[Any] = None,
        chat_stream_adapter: Optional[Any] = None,
        get_chat_stream_adapter: Optional[Callable[[], Optional[Any]]] = None,
        tool_service: Optional[Any] = None,
        session_service: Optional[Any] = None,
        context_service: Optional[Any] = None,
        provider_service: Optional[Any] = None,
        recovery_service: Optional[Any] = None,
        turn_executor: Optional[Any] = None,
        protocol_adapter: Optional[Any] = None,
        streaming_handler: Optional[Any] = None,
        streaming_controller: Optional[Any] = None,
        streaming_coordinator: Optional[Any] = None,
        iteration_coordinator: Optional[Any] = None,
        task_analyzer: Optional[Any] = None,
        exploration_state_passed: Optional[Any] = None,
        system_prompt_state_passed: Optional[Any] = None,
        safety_state_passed: Optional[Any] = None,
        coordination_state_passed: Optional[Any] = None,
        presentation: Optional[Any] = None,
        vertical_integration_adapter: Optional[Any] = None,
        vertical_context: Optional[Any] = None,
        observability: Optional[Any] = None,
        execution_tracer: Optional[Any] = None,
        tool_call_tracer: Optional[Any] = None,
        runtime_state_host: Optional[Any] = None,
        runtime_intelligence_integration: Optional[Any] = None,
        get_runtime_intelligence_integration: Optional[Callable[[], Optional[Any]]] = None,
        subagent_orchestrator: Optional[Any] = None,
        get_subagent_orchestrator: Optional[Callable[[], Optional[Any]]] = None,
    ) -> None:
        self._interaction_runtime = interaction_runtime
        self._chat_service = chat_service
        self._chat_stream_adapter = chat_stream_adapter
        self._get_chat_stream_adapter = get_chat_stream_adapter
        self._tool_service = tool_service
        self._session_service = session_service
        self._context_service = context_service
        self._provider_service = provider_service
        self._recovery_service = recovery_service
        self._turn_executor = turn_executor
        self._protocol_adapter = protocol_adapter
        self._streaming_handler = streaming_handler
        self._streaming_controller = streaming_controller
        self._streaming_coordinator = streaming_coordinator
        self._iteration_coordinator = iteration_coordinator
        self._task_analyzer = task_analyzer
        self._exploration_state_passed = exploration_state_passed
        self._system_prompt_state_passed = system_prompt_state_passed
        self._safety_state_passed = safety_state_passed
        self._coordination_state_passed = coordination_state_passed
        self._presentation = presentation
        self._vertical_integration_adapter = vertical_integration_adapter
        self._vertical_context = vertical_context
        self._observability = observability
        self._execution_tracer = execution_tracer
        self._tool_call_tracer = tool_call_tracer
        self._runtime_state_host = runtime_state_host
        self._runtime_intelligence_integration = runtime_intelligence_integration
        self._get_runtime_intelligence_integration = get_runtime_intelligence_integration
        self._subagent_orchestrator = subagent_orchestrator
        self._get_subagent_orchestrator = get_subagent_orchestrator

        logger.debug(
            "OrchestrationFacade initialized (protocol_adapter=%s, observability=%s, "
            "subagent=%s)",
            protocol_adapter is not None,
            observability is not None,
            subagent_orchestrator is not None,
        )

    # ------------------------------------------------------------------
    # Properties (satisfy OrchestrationFacadeProtocol)
    # ------------------------------------------------------------------

    @property
    def interaction_runtime(self) -> Optional[Any]:
        """Interaction runtime boundary components."""
        return self._interaction_runtime

    @property
    def chat_service(self) -> Optional[Any]:
        """Canonical chat service."""
        return self._chat_service

    @property
    def chat_stream_adapter(self) -> Optional[Any]:
        """Canonical service-owned chat-stream adapter."""
        if self._runtime_state_host is not None:
            runtime_state = getattr(self._runtime_state_host, "__dict__", {})
            if "_chat_stream_adapter" in runtime_state:
                runtime_adapter = runtime_state["_chat_stream_adapter"]
                if runtime_adapter is not None:
                    self._chat_stream_adapter = runtime_adapter
                    return runtime_adapter
                if self._get_chat_stream_adapter is None:
                    self._chat_stream_adapter = None
                    return None
        if self._chat_stream_adapter is None and self._get_chat_stream_adapter is not None:
            self._chat_stream_adapter = self._get_chat_stream_adapter()
            if self._runtime_state_host is not None:
                self._runtime_state_host._chat_stream_adapter = self._chat_stream_adapter
        return self._chat_stream_adapter

    @property
    def tool_service(self) -> Optional[Any]:
        """Canonical tool service."""
        return self._tool_service

    @property
    def session_service(self) -> Optional[Any]:
        """Canonical session service."""
        return self._session_service

    @property
    def context_service(self) -> Optional[Any]:
        """Canonical context service."""
        return self._context_service

    @property
    def provider_service(self) -> Optional[Any]:
        """Canonical provider service."""
        return self._provider_service

    @property
    def recovery_service(self) -> Optional[Any]:
        """Canonical recovery service."""
        return self._recovery_service

    @property
    def turn_executor(self) -> Optional[Any]:
        """Execution coordinator for agentic loop."""
        if self._runtime_state_host is not None:
            runtime_state = getattr(self._runtime_state_host, "__dict__", {})
            if "_turn_executor" in runtime_state:
                return runtime_state["_turn_executor"]
        return self._turn_executor

    @turn_executor.setter
    def turn_executor(self, value: Any) -> None:
        """Update the execution coordinator."""
        if self._runtime_state_host is not None:
            self._runtime_state_host._turn_executor = value
        self._turn_executor = value

    @property
    def protocol_adapter(self) -> Optional[Any]:
        """Protocol adapter for coordinator communication."""
        if self._runtime_state_host is not None:
            runtime_state = getattr(self._runtime_state_host, "__dict__", {})
            if "_protocol_adapter" in runtime_state:
                return runtime_state["_protocol_adapter"]
        return self._protocol_adapter

    @protocol_adapter.setter
    def protocol_adapter(self, value: Any) -> None:
        """Update the protocol adapter."""
        if self._runtime_state_host is not None:
            self._runtime_state_host._protocol_adapter = value
        self._protocol_adapter = value

    @property
    def streaming_handler(self) -> Optional[Any]:
        """StreamingChatHandler for streaming loop logic."""
        return self._streaming_handler

    @property
    def streaming_controller(self) -> Optional[Any]:
        """StreamingController for session/metrics management."""
        return self._streaming_controller

    @property
    def streaming_coordinator(self) -> Optional[Any]:
        """StreamingCoordinator for response processing."""
        return self._streaming_coordinator

    @property
    def iteration_coordinator(self) -> Optional[Any]:
        """IterationCoordinator for loop control."""
        if self._runtime_state_host is not None:
            runtime_state = getattr(self._runtime_state_host, "__dict__", {})
            if "_iteration_coordinator" in runtime_state:
                return runtime_state["_iteration_coordinator"]
        return self._iteration_coordinator

    @iteration_coordinator.setter
    def iteration_coordinator(self, value: Any) -> None:
        """Update the iteration coordinator."""
        if self._runtime_state_host is not None:
            self._runtime_state_host._iteration_coordinator = value
        self._iteration_coordinator = value

    @property
    def task_analyzer(self) -> Optional[Any]:
        """Unified task analysis facade."""
        return self._task_analyzer

    @property
    def exploration_state_passed(self) -> Optional[Any]:
        """State-passed exploration coordinator."""
        return self._exploration_state_passed

    @property
    def system_prompt_state_passed(self) -> Optional[Any]:
        """State-passed system prompt coordinator."""
        return self._system_prompt_state_passed

    @property
    def safety_state_passed(self) -> Optional[Any]:
        """State-passed safety coordinator."""
        return self._safety_state_passed

    @property
    def coordination_state_passed(self) -> Optional[Any]:
        """State-passed coordination suggestion coordinator."""
        return self._coordination_state_passed

    @property
    def presentation(self) -> Optional[Any]:
        """Presentation adapter for icon/emoji rendering."""
        return self._presentation

    @property
    def vertical_integration_adapter(self) -> Optional[Any]:
        """VerticalIntegrationAdapter for single-source vertical methods."""
        return self._vertical_integration_adapter

    @property
    def vertical_context(self) -> Optional[Any]:
        """VerticalContext for unified vertical state management."""
        return self._vertical_context

    @property
    def observability(self) -> Optional[Any]:
        """ObservabilityIntegration for unified event bus."""
        if self._runtime_state_host is not None:
            runtime_state = getattr(self._runtime_state_host, "__dict__", {})
            if "_observability" in runtime_state:
                return runtime_state["_observability"]
        return self._observability

    @observability.setter
    def observability(self, value: Any) -> None:
        """Update the observability integration."""
        if self._runtime_state_host is not None:
            self._runtime_state_host._observability = value
        self._observability = value

    @property
    def execution_tracer(self) -> Optional[Any]:
        """Execution tracing (if available)."""
        return self._execution_tracer

    @property
    def tool_call_tracer(self) -> Optional[Any]:
        """Tool call tracing (if available)."""
        return self._tool_call_tracer

    @property
    def runtime_intelligence_integration(self) -> Optional[Any]:
        """Runtime-intelligence integration."""
        if self._runtime_state_host is not None:
            runtime_state = getattr(self._runtime_state_host, "__dict__", {})
            if "_runtime_intelligence_integration" in runtime_state:
                runtime_integration = runtime_state["_runtime_intelligence_integration"]
                if runtime_integration is not None:
                    self._runtime_intelligence_integration = runtime_integration
                    return runtime_integration
                if self._get_runtime_intelligence_integration is None:
                    self._runtime_intelligence_integration = None
                    return None
        if (
            self._runtime_intelligence_integration is None
            and self._get_runtime_intelligence_integration is not None
        ):
            self._runtime_intelligence_integration = self._get_runtime_intelligence_integration()
            if self._runtime_state_host is not None:
                self._runtime_state_host._runtime_intelligence_integration = (
                    self._runtime_intelligence_integration
                )
        return self._runtime_intelligence_integration

    @runtime_intelligence_integration.setter
    def runtime_intelligence_integration(self, value: Any) -> None:
        """Update the runtime-intelligence integration."""
        if self._runtime_state_host is not None:
            self._runtime_state_host._runtime_intelligence_integration = value
        self._runtime_intelligence_integration = value

    @property
    def subagent_orchestrator(self) -> Optional[Any]:
        """Sub-agent orchestration."""
        if self._runtime_state_host is not None:
            runtime_state = getattr(self._runtime_state_host, "__dict__", {})
            if "_subagent_orchestrator" in runtime_state:
                runtime_subagent = runtime_state["_subagent_orchestrator"]
                if runtime_subagent is not None:
                    self._subagent_orchestrator = runtime_subagent
                    return runtime_subagent
                if self._get_subagent_orchestrator is None:
                    self._subagent_orchestrator = None
                    return None
        if self._subagent_orchestrator is None and self._get_subagent_orchestrator is not None:
            self._subagent_orchestrator = self._get_subagent_orchestrator()
            if self._runtime_state_host is not None:
                self._runtime_state_host._subagent_orchestrator = self._subagent_orchestrator
        return self._subagent_orchestrator

    @subagent_orchestrator.setter
    def subagent_orchestrator(self, value: Any) -> None:
        """Update the subagent orchestrator."""
        if self._runtime_state_host is not None:
            self._runtime_state_host._subagent_orchestrator = value
        self._subagent_orchestrator = value
