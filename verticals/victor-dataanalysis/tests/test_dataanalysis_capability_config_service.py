# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for DataAnalysis capability config storage behavior."""

from victor.framework.capability_config_service import CapabilityConfigService
from victor_dataanalysis.capabilities import (
    configure_data_quality,
    configure_visualization_style,
    get_data_quality,
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
        self.data_quality_config = {}
        self.visualization_config = {}


class TestDataAnalysisCapabilityConfigStorage:
    """Validate DataAnalysis capability config storage migration path."""

    def test_data_quality_store_and_read_from_framework_service(self):
        service = CapabilityConfigService()
        orchestrator = _ServiceBackedOrchestrator(service)

        configure_data_quality(orchestrator, min_completeness=0.95, handle_missing="drop")

        assert service.get_config("data_quality_config") == {
            "min_completeness": 0.95,
            "max_outlier_ratio": 0.05,
            "require_type_validation": True,
            "handle_missing": "drop",
        }
        assert get_data_quality(orchestrator)["min_completeness"] == 0.95

    def test_data_quality_getter_supports_legacy_service_key_alias(self):
        service = CapabilityConfigService()
        service.set_config(
            "data_quality",
            {
                "min_completeness": 0.92,
                "max_outlier_ratio": 0.07,
                "require_type_validation": False,
                "handle_missing": "flag",
            },
        )
        orchestrator = _ServiceBackedOrchestrator(service)

        assert get_data_quality(orchestrator) == {
            "min_completeness": 0.92,
            "max_outlier_ratio": 0.07,
            "require_type_validation": False,
            "handle_missing": "flag",
        }

    def test_legacy_fallback_preserves_attribute_behavior(self):
        orchestrator = _LegacyOrchestrator()

        configure_visualization_style(
            orchestrator,
            default_backend="plotly",
            theme="plotly_white",
        )

        assert orchestrator.visualization_config == {
            "backend": "plotly",
            "theme": "plotly_white",
            "figure_size": (10, 6),
            "dpi": 100,
            "save_format": "png",
        }
