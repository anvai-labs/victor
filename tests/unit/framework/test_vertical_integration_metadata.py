"""Tests for vertical metadata propagation in integration telemetry."""

from __future__ import annotations

import asyncio
from typing import List
from unittest.mock import patch

from victor.core.verticals.base import VerticalBase, VerticalRegistry
from victor.framework.vertical_integration import (
    IntegrationResult,
    VerticalIntegrationPipeline,
)
from victor_contracts.verticals.manifest import ExtensionManifest, ExtensionType


def test_vertical_applied_payload_includes_manifest_metadata() -> None:
    """Observability payloads should carry manifest version and namespace."""

    class _TelemetryVertical(VerticalBase):
        name = "telemetry_vertical"
        description = "Telemetry test vertical"
        version = "9.8.7"
        _victor_manifest = ExtensionManifest(
            name="telemetry_vertical",
            version="9.8.7",
            api_version=1,
            provides={ExtensionType.TOOLS},
            plugin_namespace="analytics",
        )

        @classmethod
        def get_tools(cls) -> List[str]:
            return ["read"]

        @classmethod
        def get_system_prompt(cls) -> str:
            return "telemetry prompt"

    VerticalRegistry.unregister(_TelemetryVertical.name)
    VerticalRegistry.register(_TelemetryVertical)

    pipeline = VerticalIntegrationPipeline()
    result = IntegrationResult(vertical_name="telemetry_vertical")

    payload = pipeline._build_vertical_applied_payload(object(), result, cache_hit=False)

    assert payload["vertical"] == "telemetry_vertical"
    assert payload["vertical_manifest_version"] == "9.8.7"
    assert payload["vertical_plugin_namespace"] == "analytics"

    VerticalRegistry.unregister(_TelemetryVertical.name)


def test_step_handler_warning_includes_vertical_namespace() -> None:
    """Step-handler failures should carry vertical and namespace metadata."""

    class _TelemetryVertical(VerticalBase):
        name = "telemetry_warning_vertical"
        description = "Telemetry warning test vertical"
        version = "1.2.3"
        _victor_manifest = ExtensionManifest(
            name="telemetry_warning_vertical",
            version="1.2.3",
            api_version=1,
            provides={ExtensionType.TOOLS},
            plugin_namespace="isolated",
        )

        @classmethod
        def get_tools(cls) -> List[str]:
            return ["read"]

        @classmethod
        def get_system_prompt(cls) -> str:
            return "telemetry prompt"

    class _FailingHandler:
        name = "failing"

        def apply(self, *args, **kwargs) -> None:
            raise RuntimeError("boom")

    class _Registry:
        def get_ordered_handlers(self):
            return [_FailingHandler()]

    pipeline = VerticalIntegrationPipeline(step_registry=_Registry())
    result = IntegrationResult(vertical_name=_TelemetryVertical.name)

    pipeline._apply_with_step_handlers(
        object(),
        _TelemetryVertical,
        object(),  # type: ignore[arg-type]
        result,
    )

    assert result.warnings == [
        "Step handler 'failing' failed for vertical "
        "'telemetry_warning_vertical' [namespace=isolated]: boom"
    ]


def test_async_handler_uses_namespace_executor_pool() -> None:
    """Sync handlers offloaded from async integration should use namespace executors."""

    class _TelemetryVertical(VerticalBase):
        name = "telemetry_async_vertical"
        description = "Telemetry async test vertical"
        version = "4.5.6"
        _victor_manifest = ExtensionManifest(
            name="telemetry_async_vertical",
            version="4.5.6",
            api_version=1,
            provides={ExtensionType.TOOLS},
            plugin_namespace="async-isolated",
        )

        @classmethod
        def get_tools(cls) -> List[str]:
            return ["read"]

        @classmethod
        def get_system_prompt(cls) -> str:
            return "telemetry prompt"

    class _Handler:
        name = "sync-handler"

        def __init__(self) -> None:
            self.called = False

        def apply(self, *args, **kwargs) -> None:
            self.called = True

    class _FakeLoop:
        def __init__(self) -> None:
            self.executor = None

        async def run_in_executor(self, executor, func):
            self.executor = executor
            func()
            return None

    handler = _Handler()
    pipeline = VerticalIntegrationPipeline()
    result = IntegrationResult(vertical_name=_TelemetryVertical.name)
    fake_loop = _FakeLoop()
    sentinel_executor = object()

    with patch(
        "asyncio.get_running_loop",
        return_value=fake_loop,
    ):
        with patch(
            "victor.framework.vertical_integration.get_namespace_executor_pool",
        ) as pool_mock:
            pool_mock.return_value.get_executor.return_value = sentinel_executor
            asyncio.run(
                pipeline._run_handler_async(
                    handler,
                    object(),
                    _TelemetryVertical,
                    object(),  # type: ignore[arg-type]
                    result,
                )
            )

    assert handler.called is True
    assert fake_loop.executor is sentinel_executor
