# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from victor.agent.runtime.metrics_runtime import create_metrics_runtime_components


def test_legacy_metrics_module_reexports_service_runtime():
    """Legacy coordinator path should re-export service-owned metrics runtime."""
    from victor.agent.services.metrics_service import (
        AgentMetricsService as legacy_metrics_coordinator,
        create_agent_metrics_service as legacy_create_agent_metrics_service,
    )
    from victor.agent.services.metrics_service import (
        AgentMetricsService as service_metrics_coordinator,
        create_agent_metrics_service as service_create_agent_metrics_service,
    )

    assert legacy_metrics_coordinator is service_metrics_coordinator
    assert legacy_create_agent_metrics_service is service_create_agent_metrics_service


def test_create_metrics_runtime_components_lazy_materialization():
    factory = MagicMock()
    usage_logger = MagicMock()
    streaming_collector = MagicMock()
    debug_logger = MagicMock()
    metrics_collector = MagicMock()
    metrics_coordinator = MagicMock()

    factory.create_usage_logger.return_value = usage_logger
    factory.create_streaming_metrics_collector.return_value = streaming_collector
    factory.create_metrics_collector.return_value = metrics_collector

    provider = SimpleNamespace(name="ollama")
    cumulative_token_usage = {"input_tokens": 0, "output_tokens": 0}

    runtime = create_metrics_runtime_components(
        factory=factory,
        provider=provider,
        model="qwen3-coder:30b",
        debug_logger=debug_logger,
        cumulative_token_usage=cumulative_token_usage,
        tool_cost_lookup=lambda tool_name: f"tier:{tool_name}",
    )

    assert runtime.usage_logger is usage_logger
    assert runtime.streaming_metrics_collector is streaming_collector
    assert runtime.metrics_collector.initialized is False
    assert runtime.session_cost_tracker.initialized is False
    assert runtime.metrics_coordinator.initialized is False

    with patch("victor.agent.session_cost_tracker.SessionCostTracker") as session_cost_tracker_cls:
        with patch(
            "victor.agent.services.metrics_service.AgentMetricsService"
        ) as metrics_coordinator_cls:
            session_cost_tracker = MagicMock()
            session_cost_tracker_cls.return_value = session_cost_tracker
            metrics_coordinator_cls.return_value = metrics_coordinator

            resolved = runtime.metrics_coordinator.get_instance()
            resolved_again = runtime.metrics_coordinator.get_instance()

    assert resolved is metrics_coordinator
    assert resolved_again is metrics_coordinator
    assert runtime.metrics_collector.initialized is True
    assert runtime.session_cost_tracker.initialized is True
    assert runtime.metrics_coordinator.initialized is True

    factory.create_metrics_collector.assert_called_once()
    metrics_collector_kwargs = factory.create_metrics_collector.call_args.kwargs
    assert metrics_collector_kwargs["streaming_metrics_collector"] is streaming_collector
    assert metrics_collector_kwargs["usage_logger"] is usage_logger
    assert metrics_collector_kwargs["debug_logger"] is debug_logger
    assert metrics_collector_kwargs["tool_cost_lookup"]("read_file") == "tier:read_file"

    session_cost_tracker_cls.assert_called_once_with(provider="ollama", model="qwen3-coder:30b")
    metrics_coordinator_cls.assert_called_once_with(
        metrics_collector=metrics_collector,
        session_cost_tracker=session_cost_tracker,
        cumulative_token_usage=cumulative_token_usage,
    )
