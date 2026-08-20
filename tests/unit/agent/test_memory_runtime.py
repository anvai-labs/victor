# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from unittest.mock import MagicMock

from victor.agent.runtime.memory_runtime import create_memory_runtime_components


def test_create_memory_runtime_components_preserves_factory_diagnostics():
    factory = MagicMock()
    memory_manager = MagicMock()
    factory.create_memory_components.return_value = (memory_manager, "session-1")
    factory.get_memory_initialization_diagnostics.return_value = {
        "status": "initialized",
        "recovered_from_lock": True,
        "lock_retries": 2,
    }

    runtime = create_memory_runtime_components(
        factory=factory,
        provider_name="zai",
        native_tool_calls=True,
    )

    assert runtime.memory_manager is memory_manager
    assert runtime.memory_session_id == "session-1"
    assert runtime.memory_initialization_diagnostics == {
        "status": "initialized",
        "recovered_from_lock": True,
        "lock_retries": 2,
    }
