# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from unittest.mock import MagicMock

from victor.agent.runtime.resilience_runtime import create_resilience_runtime_components


def test_create_resilience_runtime_components_recovery_handler_lazy():
    factory = MagicMock()
    context_compactor = MagicMock()
    handler = MagicMock()
    factory.create_recovery_handler.return_value = handler

    runtime = create_resilience_runtime_components(
        factory=factory,
        context_compactor=context_compactor,
    )

    assert runtime.recovery_handler.initialized is False
    resolved = runtime.recovery_handler.get_instance()
    assert resolved is handler
    assert runtime.recovery_handler.initialized is True
    handler.set_context_compactor.assert_called_once_with(context_compactor)


def test_create_resilience_runtime_components_recovery_integration_lazy():
    factory = MagicMock()
    context_compactor = MagicMock()
    handler = MagicMock()
    integration = MagicMock()
    factory.create_recovery_handler.return_value = handler
    factory.create_recovery_integration.return_value = integration

    runtime = create_resilience_runtime_components(
        factory=factory,
        context_compactor=context_compactor,
    )

    assert runtime.recovery_handler.initialized is False
    assert runtime.recovery_integration.initialized is False

    resolved = runtime.recovery_integration.get_instance()
    assert resolved is integration
    assert runtime.recovery_integration.initialized is True
    assert runtime.recovery_handler.initialized is True
    factory.create_recovery_integration.assert_called_once_with(handler)
