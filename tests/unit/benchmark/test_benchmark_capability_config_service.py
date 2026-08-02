# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for Benchmark capability config storage behavior."""

from victor.framework.capability_config_service import CapabilityConfigService
from victor.benchmark.capabilities import (
    configure_passk_evaluation,
    configure_swe_bench_execution,
    configure_metrics_collection,
    get_passk_config,
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
        self.benchmark_config = {}
        self.metrics_config = {}


class TestBenchmarkCapabilityConfigStorage:
    """Validate Benchmark capability config storage migration path."""

    def test_passk_store_and_read_from_framework_service(self):
        service = CapabilityConfigService()
        orchestrator = _ServiceBackedOrchestrator(service)

        configure_passk_evaluation(orchestrator, k_value=20, max_samples=250)

        assert get_passk_config(orchestrator) == {
            "k_value": 20,
            "temperature_range": [0.2, 0.4, 0.6, 0.8],
            "max_samples": 250,
            "enable_early_stopping": True,
            "stop_after_n_success": 5,
        }

    def test_benchmark_sections_merge_in_service(self):
        service = CapabilityConfigService()
        orchestrator = _ServiceBackedOrchestrator(service)

        configure_swe_bench_execution(orchestrator, max_iterations=7)
        configure_passk_evaluation(orchestrator, k_value=15)

        benchmark_config = service.get_config("benchmark_config")
        assert benchmark_config["swe_bench"]["max_iterations"] == 7
        assert benchmark_config["passk"]["k_value"] == 15

    def test_legacy_fallback_preserves_metrics_attribute_behavior(self):
        orchestrator = _LegacyOrchestrator()

        configure_metrics_collection(orchestrator, output_format="csv")

        assert orchestrator.metrics_config == {
            "track_token_usage": True,
            "track_execution_time": True,
            "track_tool_calls": True,
            "track_test_results": True,
            "enable_detailed_tracing": False,
            "output_format": "csv",
        }
