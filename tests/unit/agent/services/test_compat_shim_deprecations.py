# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for removed compatibility shims.

Note: tool_compat, recovery_compat, prompt_compat, and state_compat were
removed as part of the service-first architecture migration. This test file
verifies that the removed modules are no longer importable.
"""

from __future__ import annotations

import pytest


# Removed compatibility shims - verify they cannot be imported
def test_prompt_compat_module_removed():
    """Verify that prompt_compat module has been removed."""
    with pytest.raises(ImportError, match="prompt_compat"):
        from victor.agent.services import prompt_compat  # noqa: F401


def test_state_compat_module_removed():
    """Verify that state_compat module has been removed."""
    with pytest.raises(ImportError, match="state_compat"):
        from victor.agent.services import state_compat  # noqa: F401


def test_state_coordinator_module_removed():
    """Verify that the concrete state_coordinator shim has been removed."""
    with pytest.raises(ImportError, match="state_coordinator"):
        import victor.agent.state_coordinator  # noqa: F401


def test_recovery_compat_module_removed():
    """Verify that recovery_compat module has been removed."""
    with pytest.raises(ImportError, match="recovery_compat"):
        from victor.agent.services import recovery_compat  # noqa: F401


def test_prompt_runtime_support_module_removed():
    """Verify that prompt_runtime_support module has been removed."""
    with pytest.raises(ImportError, match="prompt_runtime_support"):
        import victor.agent.services.prompt_runtime_support  # noqa: F401


def test_system_prompt_runtime_module_removed():
    """Verify that system_prompt_runtime module has been removed."""
    with pytest.raises(ImportError, match="system_prompt_runtime"):
        import victor.agent.services.system_prompt_runtime  # noqa: F401


def test_chat_compat_telemetry_module_removed():
    """Verify that chat_compat_telemetry module has been removed."""
    with pytest.raises(ImportError, match="chat_compat_telemetry"):
        import victor.agent.services.chat_compat_telemetry  # noqa: F401


def test_mode_workflow_team_coordinator_module_removed():
    """Verify that mode_workflow_team_coordinator module has been removed."""
    with pytest.raises(ImportError, match="mode_workflow_team_coordinator"):
        import victor.agent.mode_workflow_team_coordinator  # noqa: F401


# Verify canonical imports still work
def test_canonical_prompt_runtime_imports():
    """Verify canonical prompt runtime imports work."""
    from victor.agent.services.prompt_runtime import (
        PromptRuntimeAdapter,
        PromptRuntimeConfig,
        PromptRuntimeContext,
    )

    assert PromptRuntimeAdapter is not None
    assert PromptRuntimeConfig is not None
    assert PromptRuntimeContext is not None


def test_canonical_system_prompt_state_passed_imports():
    """Verify canonical state-passed system prompt imports work."""
    from victor.agent.coordinators.system_prompt_state_passed import (
        SystemPromptStatePassedCoordinator,
    )

    assert SystemPromptStatePassedCoordinator is not None


def test_canonical_state_runtime_imports():
    """Verify canonical state runtime imports work."""
    from victor.agent.services.state_runtime import StateRuntimeAdapter

    assert StateRuntimeAdapter is not None


def test_canonical_recovery_service_imports():
    """Verify canonical recovery service imports work."""
    from victor.agent.services.recovery_service import (
        RecoveryService,
        StreamingRecoveryContext,
    )

    assert RecoveryService is not None
    assert StreamingRecoveryContext is not None


# Verify deprecated names are no longer exported from services/__init__.py
def test_prompt_coordinator_not_exported_from_services():
    """Verify PromptCoordinator is no longer exported from services."""
    from victor.agent import services

    assert "PromptCoordinator" not in dir(services)
    assert "create_prompt_coordinator" not in dir(services)
    assert "SystemPromptCoordinator" not in dir(services)


def test_state_coordinator_not_exported_from_services():
    """Verify StateCoordinator is no longer exported from services."""
    from victor.agent import services

    assert "StateCoordinator" not in dir(services)
    assert "create_state_coordinator" not in dir(services)


def test_chat_compat_runtime_protocol_not_exported_from_service_protocol_package():
    """Verify deprecated chat-compat runtime protocol is not re-exported from services.protocols."""
    from victor.agent.services import protocols

    assert "ChatCompatRuntimeProtocol" not in dir(protocols)


def test_chat_compat_runtime_protocol_removed_from_chat_runtime_module():
    """Verify deprecated chat-compat runtime protocol is removed from the module surface."""
    with pytest.raises(ImportError, match="ChatCompatRuntimeProtocol"):
        from victor.agent.services.protocols.chat_runtime import (
            ChatCompatRuntimeProtocol,
        )  # noqa: F401


# Verify deprecated names are no longer exported from coordinators/__init__.py
def test_state_coordinator_not_exported_from_coordinators():
    """Verify StateCoordinator is no longer exported from coordinators."""
    from victor.agent import coordinators

    # StateCoordinator should not be importable
    with pytest.raises(AttributeError, match="has no attribute"):
        coordinators.StateCoordinator()

    with pytest.raises(AttributeError, match="has no attribute"):
        coordinators.StateCoordinatorConfig()

    with pytest.raises(AttributeError, match="has no attribute"):
        coordinators.create_state_coordinator()
