# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from victor.agent.runtime.provider_runtime import (
    LazyRuntimeProxy,
    create_provider_runtime_components,
)


def test_lazy_runtime_proxy_initializes_once():
    calls = {"count": 0}
    target = SimpleNamespace(value=42)

    def _factory():
        calls["count"] += 1
        return target

    proxy = LazyRuntimeProxy(factory=_factory, name="test_component")
    assert proxy.initialized is False

    assert proxy.value == 42
    assert proxy.initialized is True
    assert calls["count"] == 1

    assert proxy.value == 42
    assert calls["count"] == 1


def test_lazy_runtime_proxy_setattr_delegates_to_instance():
    target = SimpleNamespace(value=1)

    proxy = LazyRuntimeProxy(factory=lambda: target, name="test_component")
    assert proxy.initialized is False

    proxy.value = 7

    assert proxy.initialized is True
    assert target.value == 7
    assert proxy.value == 7


def test_create_provider_runtime_components_provider_coordinator_removed():
    """Test that provider_coordinator was removed (migration 2026-05-01)."""
    manager = MagicMock()
    manager._provider_switcher = object()
    manager._health_monitor = object()
    settings = SimpleNamespace(max_rate_limit_retries=7, provider_health_checks=False)

    runtime = create_provider_runtime_components(
        settings=settings,
        provider_manager=manager,
    )

    # ProviderCoordinator was removed - use ProviderService instead
    assert not hasattr(runtime, "provider_coordinator")


def test_create_provider_runtime_components_switch_coordinator_removed():
    """Test that provider_switch_coordinator was removed (migration 2026-05-01)."""
    manager = MagicMock()
    manager._provider_switcher = object()
    manager._health_monitor = object()
    settings = SimpleNamespace(max_rate_limit_retries=3, provider_health_checks=True)

    runtime = create_provider_runtime_components(
        settings=settings,
        provider_manager=manager,
    )

    # ProviderSwitchCoordinator was removed - use ProviderService instead
    assert not hasattr(runtime, "provider_switch_coordinator")


def test_create_provider_runtime_components_accepts_legacy_get_provider_service_kwarg():
    """Older callers passing get_provider_service should not fail."""
    manager = MagicMock()
    settings = SimpleNamespace(feature_flags=SimpleNamespace())

    runtime = create_provider_runtime_components(
        settings=settings,
        provider_manager=manager,
        get_provider_service=lambda: object(),
    )

    assert runtime.pool is None
