# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for Coding capability config storage behavior."""

from victor.framework.capability_config_service import CapabilityConfigService
from victor_coding.capabilities import (
    configure_code_style,
    get_code_style,
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
        self.code_style = {}


class TestCodingCapabilityConfigStorage:
    """Validate Coding capability config storage migration path."""

    def test_configure_code_style_stores_in_framework_service(self):
        service = CapabilityConfigService()
        orchestrator = _ServiceBackedOrchestrator(service)

        configure_code_style(orchestrator, formatter="black", linter="ruff", max_line_length=88)

        assert service.get_config("code_style") == {
            "formatter": "black",
            "linter": "ruff",
            "max_line_length": 88,
            "enforce_type_hints": True,
        }

    def test_get_code_style_reads_framework_service_first(self):
        service = CapabilityConfigService()
        service.set_config(
            "code_style",
            {
                "formatter": "yapf",
                "linter": "pylint",
                "max_line_length": 120,
                "enforce_type_hints": False,
            },
        )
        orchestrator = _ServiceBackedOrchestrator(service)

        assert get_code_style(orchestrator) == {
            "formatter": "yapf",
            "linter": "pylint",
            "max_line_length": 120,
            "enforce_type_hints": False,
        }

    def test_legacy_fallback_preserves_attribute_behavior(self):
        orchestrator = _LegacyOrchestrator()

        configure_code_style(
            orchestrator,
            formatter="yapf",
            linter="pylint",
            max_line_length=120,
            enforce_type_hints=False,
        )

        assert orchestrator.code_style == {
            "formatter": "yapf",
            "linter": "pylint",
            "max_line_length": 120,
            "enforce_type_hints": False,
        }
