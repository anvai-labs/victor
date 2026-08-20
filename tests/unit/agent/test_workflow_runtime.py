# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from unittest.mock import MagicMock, patch

from victor.agent.runtime.workflow_runtime import create_workflow_runtime_components


def test_create_workflow_runtime_components_lazy_registry():
    factory = MagicMock()
    workflow_registry = MagicMock()
    factory.create_workflow_registry.return_value = workflow_registry

    with patch("victor.workflows.discovery.register_builtin_workflows") as register_builtin:
        runtime = create_workflow_runtime_components(factory=factory)

        assert runtime.workflow_registry.initialized is False

        resolved = runtime.workflow_registry.get_instance()
        resolved_again = runtime.workflow_registry.get_instance()

    assert resolved is workflow_registry
    assert resolved_again is workflow_registry
    assert runtime.workflow_registry.initialized is True
    factory.create_workflow_registry.assert_called_once_with()
    register_builtin.assert_called_once_with(workflow_registry)
