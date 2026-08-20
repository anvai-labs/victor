# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Stress/integration coverage for P3 queue pressure and extension saturation."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from victor.core.events import BackendConfig, InMemoryEventBackend, MessagingEvent
from victor.core.verticals.extension_loader import VerticalExtensionLoader


class _SaturationVertical(VerticalExtensionLoader):
    """Synthetic vertical that adds latency to extension loaders."""

    name = "saturation_vertical"

    @staticmethod
    def _pause() -> None:
        time.sleep(0.01)

    @classmethod
    def get_middleware(cls):
        cls._pause()
        return []

    @classmethod
    def get_safety_extension(cls):
        cls._pause()
        return None

    @classmethod
    def get_prompt_contributor(cls):
        cls._pause()
        return None

    @classmethod
    def get_mode_config_provider(cls):
        cls._pause()
        return None

    @classmethod
    def get_tool_dependency_provider(cls):
        cls._pause()
        return None

    @classmethod
    def get_workflow_provider(cls):
        cls._pause()
        return None

    @classmethod
    def get_service_provider(cls):
        cls._pause()
        return None

    @classmethod
    def get_rl_config_provider(cls):
        cls._pause()
        return None

    @classmethod
    def get_team_spec_provider(cls):
        cls._pause()
        return None

    @classmethod
    def get_enrichment_strategy(cls):
        cls._pause()
        return None

    @classmethod
    def get_tiered_tool_config(cls):
        cls._pause()
        return None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_queue_overflow_block_timeout_default_under_sustained_pressure():
    """Default block timeout should fail fast when queue remains saturated."""
    backend = InMemoryEventBackend(
        config=BackendConfig(extra={"queue_overflow_policy": "block_with_timeout"}),
        queue_maxsize=1,
    )
    backend._is_connected = True

    assert await backend.publish(MessagingEvent(topic="stress.full", data={"i": 1})) is True
    t0 = time.perf_counter()
    result = await backend.publish(MessagingEvent(topic="stress.timeout", data={"i": 2}))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert result is False
    # Default timeout is 50ms. Allow small scheduling jitter.
    assert elapsed_ms >= 35.0
    stats = backend.get_queue_pressure_stats()["stats"]
    assert stats["blocked_timeout"] == 1
    assert stats["max_queue_depth"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_queue_overflow_topic_defaults_apply_blocking_only_to_critical_topics():
    """Critical topics should block-with-timeout while default telemetry uses drop-newest."""
    backend = InMemoryEventBackend(
        config=BackendConfig(
            extra={
                "queue_overflow_policy": "drop_newest",
                "queue_overflow_topic_policies": {"vertical.applied": "block_with_timeout"},
                "queue_overflow_topic_block_timeout_ms": {"vertical.applied": 30.0},
            }
        ),
        queue_maxsize=1,
    )
    backend._is_connected = True

    assert await backend.publish(MessagingEvent(topic="metric.latency", data={"i": 1})) is True

    t0 = time.perf_counter()
    critical_result = await backend.publish(MessagingEvent(topic="vertical.applied", data={"i": 2}))
    critical_elapsed_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    telemetry_result = await backend.publish(MessagingEvent(topic="metric.latency", data={"i": 3}))
    telemetry_elapsed_ms = (time.perf_counter() - t1) * 1000.0

    assert critical_result is False
    assert telemetry_result is False
    assert critical_elapsed_ms >= 20.0
    assert telemetry_elapsed_ms < 15.0

    stats = backend.get_queue_pressure_stats()["stats"]
    assert stats["blocked_timeout"] == 1
    assert stats["dropped_newest"] == 1
    assert stats["topic_policy_override_hits"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extension_loader_saturation_surfaces_pressure_with_default_thresholds():
    """Concurrent async extension loads should trigger queue wait/pressure telemetry."""
    pm = VerticalExtensionLoader._pressure_monitor
    original: dict[str, Any] = {
        "warn_queue": pm.warn_queue_threshold,
        "error_queue": pm.error_queue_threshold,
        "warn_in_flight": pm.warn_in_flight_threshold,
        "error_in_flight": pm.error_in_flight_threshold,
        "cooldown": pm.pressure_cooldown_seconds,
        "emit_events": pm.emit_pressure_events,
    }

    try:
        VerticalExtensionLoader.reset_extension_loader_metrics()
        _SaturationVertical.clear_extension_cache(clear_all=True)

        tasks = [
            _SaturationVertical.get_extensions_async(use_cache=False, strict=False)
            for _ in range(12)
        ]
        await asyncio.gather(*tasks)

        metrics = VerticalExtensionLoader.get_extension_loader_metrics()
        assert metrics["queue_waits"] > 0
        assert metrics["max_queued"] >= metrics["warn_queue_threshold"]
        assert metrics["pressure_warnings"] + metrics["pressure_errors"] >= 1
    finally:
        VerticalExtensionLoader.configure_extension_loader_pressure(
            warn_queue_threshold=original["warn_queue"],
            error_queue_threshold=original["error_queue"],
            warn_in_flight_threshold=original["warn_in_flight"],
            error_in_flight_threshold=original["error_in_flight"],
            cooldown_seconds=original["cooldown"],
            emit_events=original["emit_events"],
        )
        VerticalExtensionLoader.reset_extension_loader_metrics()
        _SaturationVertical.clear_extension_cache(clear_all=True)
