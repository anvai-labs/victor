# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for Research capability config storage behavior."""

from victor.framework.capability_config_service import CapabilityConfigService
from victor_research.capabilities import (
    configure_source_verification,
    get_source_verification,
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
        self.source_verification_config = {}


class TestResearchCapabilityConfigStorage:
    """Validate Research capability config storage migration path."""

    def test_configure_source_verification_stores_in_framework_service(self):
        service = CapabilityConfigService()
        orchestrator = _ServiceBackedOrchestrator(service)

        configure_source_verification(
            orchestrator,
            min_credibility=0.85,
            min_source_count=4,
        )

        assert service.get_config("source_verification_config") == {
            "min_credibility": 0.85,
            "min_source_count": 4,
            "require_diverse_sources": True,
            "validate_urls": True,
        }

    def test_get_source_verification_reads_framework_service_first(self):
        service = CapabilityConfigService()
        service.set_config(
            "source_verification_config",
            {
                "min_credibility": 0.9,
                "min_source_count": 5,
                "require_diverse_sources": False,
                "validate_urls": False,
            },
        )
        orchestrator = _ServiceBackedOrchestrator(service)

        assert get_source_verification(orchestrator) == {
            "min_credibility": 0.9,
            "min_source_count": 5,
            "require_diverse_sources": False,
            "validate_urls": False,
        }

    def test_legacy_fallback_preserves_attribute_behavior(self):
        orchestrator = _LegacyOrchestrator()

        configure_source_verification(
            orchestrator,
            min_credibility=0.75,
            min_source_count=2,
            require_diverse_sources=False,
            validate_urls=False,
        )

        assert orchestrator.source_verification_config == {
            "min_credibility": 0.75,
            "min_source_count": 2,
            "require_diverse_sources": False,
            "validate_urls": False,
        }
