# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for FrameworkIntegrationRegistryService."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from victor.framework.framework_integration_registry_service import (
    FrameworkIntegrationRegistryService,
    resolve_framework_integration_registry_service,
)


def test_register_workflows_namespaces_without_mutating_original() -> None:
    """Workflow registration should namespace workflow names and keep input unchanged."""
    service = FrameworkIntegrationRegistryService()
    workflow = SimpleNamespace(name="original")
    mock_registry = MagicMock()

    with patch(
        "victor.workflows.registry.get_workflow_registry",
        return_value=mock_registry,
        create=True,
    ):
        count = service.register_workflows("coding", {"review": workflow}, replace=True)

    assert count == 1
    assert workflow.name == "original"
    registered_workflow = mock_registry.register.call_args[0][0]
    assert registered_workflow.name == "coding:review"
    assert mock_registry.register.call_args.kwargs["replace"] is True


def test_resolve_service_prefers_container_instance() -> None:
    """Resolver should return DI container-provided service when available."""
    custom_service = FrameworkIntegrationRegistryService()
    container = MagicMock()
    container.get_optional.return_value = custom_service

    orchestrator = MagicMock()
    orchestrator.get_service_container.return_value = container

    resolved = resolve_framework_integration_registry_service(orchestrator)

    assert resolved is custom_service
    container.get_optional.assert_called_once()


def test_register_handlers_uses_vertical_registry() -> None:
    """Handler registration should use framework handler registry when present."""
    service = FrameworkIntegrationRegistryService()
    mock_registry = MagicMock()
    mock_registry.register_vertical = MagicMock()
    handler = object()

    with patch(
        "victor.framework.handler_registry.get_handler_registry",
        return_value=mock_registry,
    ):
        count = service.register_handlers("coding", {"lint": handler}, replace=True)

    assert count == 1
    mock_registry.register_vertical.assert_called_once_with("coding", {"lint": handler})


def test_register_workflows_skips_replay_and_tracks_metrics() -> None:
    """Workflow replay with unchanged payload should be skipped and metered."""
    service = FrameworkIntegrationRegistryService()
    mock_registry = MagicMock()

    class _Workflow:
        def __init__(self, name: str):
            self.name = name

        def to_dict(self):
            return {"name": self.name, "nodes": []}

    workflow = _Workflow("original")
    with patch(
        "victor.workflows.registry.get_workflow_registry",
        return_value=mock_registry,
        create=True,
    ):
        first = service.register_workflows("coding", {"review": workflow}, replace=True)
        second = service.register_workflows("coding", {"review": workflow}, replace=True)

    assert first == 1
    assert second == 0
    mock_registry.register.assert_called_once()

    metrics = service.snapshot_metrics()["workflows"]
    assert metrics["attempted"] == 2
    assert metrics["applied"] == 1
    assert metrics["skipped"] == 1
    assert metrics["failed"] == 0


def test_register_workflows_reapplies_when_registration_version_changes() -> None:
    """Version token change should force re-application for same payload."""
    service = FrameworkIntegrationRegistryService()
    mock_registry = MagicMock()

    workflow = SimpleNamespace(name="original")
    with patch(
        "victor.workflows.registry.get_workflow_registry",
        return_value=mock_registry,
        create=True,
    ):
        first = service.register_workflows(
            "coding",
            {"review": workflow},
            replace=True,
            registration_version="1.0.0",
        )
        second = service.register_workflows(
            "coding",
            {"review": workflow},
            replace=True,
            registration_version="1.0.0",
        )
        third = service.register_workflows(
            "coding",
            {"review": workflow},
            replace=True,
            registration_version="1.0.1",
        )

    assert first == 1
    assert second == 0
    assert third == 1
    assert mock_registry.register.call_count == 2

    metrics = service.snapshot_metrics()["workflows"]
    assert metrics["attempted"] == 3
    assert metrics["applied"] == 2
    assert metrics["skipped"] == 1
