# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from unittest.mock import MagicMock, patch

from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.runtime.coordination_runtime import (
    create_coordination_runtime_components,
)


def test_create_coordination_runtime_components_lazy_materialization():
    factory = MagicMock()
    recovery_coordinator = MagicMock()
    chunk_generator = MagicMock()
    tool_planner = MagicMock()
    task_coordinator = MagicMock()
    coordination_advisor_runtime = MagicMock()

    factory.create_recovery_coordinator.return_value = recovery_coordinator
    factory.create_chunk_generator.return_value = chunk_generator
    factory.create_tool_planner.return_value = tool_planner
    factory.create_task_coordinator.return_value = task_coordinator
    factory.create_coordination_advisor_runtime.return_value = coordination_advisor_runtime

    runtime = create_coordination_runtime_components(factory=factory)

    assert runtime.recovery_coordinator.initialized is False
    assert runtime.chunk_generator.initialized is False
    assert runtime.tool_planner.initialized is False
    assert runtime.task_coordinator.initialized is False
    assert runtime.coordination_advisor_runtime.initialized is False

    assert runtime.recovery_coordinator.get_instance() is recovery_coordinator
    assert runtime.chunk_generator.get_instance() is chunk_generator
    assert runtime.tool_planner.get_instance() is tool_planner
    assert runtime.task_coordinator.get_instance() is task_coordinator
    assert runtime.coordination_advisor_runtime.get_instance() is coordination_advisor_runtime

    assert runtime.recovery_coordinator.initialized is True
    assert runtime.chunk_generator.initialized is True
    assert runtime.tool_planner.initialized is True
    assert runtime.task_coordinator.initialized is True
    assert runtime.coordination_advisor_runtime.initialized is True

    factory.create_recovery_coordinator.assert_called_once_with()
    factory.create_chunk_generator.assert_called_once_with()
    factory.create_tool_planner.assert_called_once_with()
    factory.create_task_coordinator.assert_called_once_with()
    factory.create_coordination_advisor_runtime.assert_called_once_with()


def test_create_coordination_runtime_components_binds_recovery_service_on_materialization():
    factory = MagicMock()
    recovery_coordinator = MagicMock()
    recovery_service = MagicMock()
    factory.create_recovery_coordinator.return_value = recovery_coordinator

    runtime = create_coordination_runtime_components(
        factory=factory,
        get_recovery_service=lambda: recovery_service,
    )

    assert runtime.recovery_coordinator.get_instance() is recovery_coordinator
    recovery_coordinator.bind_recovery_service.assert_called_once_with(recovery_service)


def test_initialize_coordination_runtime_binds_task_analyzer_to_coordination_runtime():
    orchestrator = object.__new__(AgentOrchestrator)
    factory = MagicMock()
    coordination_runtime = create_coordination_runtime_components(factory=factory)
    analyzer = MagicMock()

    orchestrator._factory = factory
    orchestrator._task_analyzer = analyzer

    with patch(
        "victor.agent.runtime.coordination_runtime.create_coordination_runtime_components",
        return_value=coordination_runtime,
    ):
        AgentOrchestrator._initialize_coordination_runtime(orchestrator)

    assert (
        orchestrator._coordination_advisor_runtime
        is coordination_runtime.coordination_advisor_runtime
    )
    analyzer.set_coordination_runtime.assert_called_once_with(
        coordination_runtime.coordination_advisor_runtime
    )
