# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for DevOps capability config storage behavior."""

from victor.framework.capability_config_service import CapabilityConfigService
from victor_devops.capabilities import (
    configure_container_settings,
    configure_deployment_safety,
    configure_infrastructure_settings,
    get_container_settings,
)


class _StubContainer:
    def __init__(self, service: CapabilityConfigService | None = None) -> None:
        self._service = service

    def get_optional(self, service_type):
        if self._service is None:
            return None
        if isinstance(self._service, service_type):
            return self._service
        return None


class _ServiceBackedOrchestrator:
    def __init__(self, service: CapabilityConfigService) -> None:
        self._container = _StubContainer(service)

    def get_service_container(self):
        return self._container


class _LegacyOrchestrator:
    def __init__(self) -> None:
        self.container_config = {}
        self.infra_config = {}
        self.safety_config = {}


class TestDevOpsCapabilityConfigStorage:
    """Validate DevOps capability config storage migration path."""

    def test_container_settings_store_and_read_from_framework_service(self):
        service = CapabilityConfigService()
        orchestrator = _ServiceBackedOrchestrator(service)

        configure_container_settings(orchestrator, runtime="podman", max_image_size_mb=500)

        assert service.get_config("container_config") == {
            "runtime": "podman",
            "default_registry": None,
            "security_scan_enabled": True,
            "max_image_size_mb": 500,
        }
        assert get_container_settings(orchestrator)["runtime"] == "podman"

    def test_legacy_fallback_preserves_infrastructure_attr_behavior(self):
        orchestrator = _LegacyOrchestrator()

        configure_infrastructure_settings(orchestrator, iac_tool="opentofu", state_backend="s3")

        assert orchestrator.infra_config == {
            "iac_tool": "opentofu",
            "auto_approve_non_destructive": False,
            "require_plan_before_apply": True,
            "state_backend": "s3",
        }

    def test_legacy_fallback_preserves_deployment_safety_nested_behavior(self):
        orchestrator = _LegacyOrchestrator()

        configure_deployment_safety(orchestrator, require_backup_before_deploy=False)

        assert orchestrator.safety_config["deployment"] == {
            "require_approval_for_production": True,
            "require_backup_before_deploy": False,
            "enable_rollback": True,
            "protected_environments": ["production", "staging"],
        }
