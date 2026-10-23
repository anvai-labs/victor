"""The core RL coordinator port and its composition-root binding."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest import mock

import pytest

import victor.framework.rl.coordinator as framework_rl
from victor.core.rl import coordinator_port as port


def test_first_use_binds_the_default_provider():
    port.set_rl_coordinator_provider(None)
    sentinel = mock.Mock()
    with mock.patch.object(framework_rl, "get_rl_coordinator", return_value=sentinel) as get:
        assert port.get_rl_coordinator() is sentinel
        assert port._provider is not None
        assert port.get_rl_coordinator() is sentinel
    assert get.call_count == 2


def test_port_returns_the_framework_singleton():
    port.set_rl_coordinator_provider(None)
    shared = mock.Mock(is_closed=False)
    framework_rl._rl_coordinator = shared
    try:
        assert port.get_rl_coordinator() is framework_rl.get_rl_coordinator()
        assert port.get_rl_coordinator() is shared
    finally:
        framework_rl._rl_coordinator = None


def test_reset_and_reopen_after_close_flow_through_the_port():
    port.set_rl_coordinator_provider(None)
    first, second, third = (mock.Mock(is_closed=False) for _ in range(3))
    with mock.patch.object(framework_rl, "RLCoordinator", side_effect=[first, second, third]):
        assert port.get_rl_coordinator() is first
        framework_rl.reset_rl_coordinator()
        assert port.get_rl_coordinator() is second
        second.is_closed = True
        assert port.get_rl_coordinator() is third
    framework_rl.reset_rl_coordinator()


def test_old_path_patch_reaches_infrastructure_callers():
    """Existing tests patch the framework accessor; infra now calls the port."""
    from victor.storage.cache.tool_cache import ToolCache

    coordinator = mock.Mock()
    coordinator.get_learner.return_value = None
    with mock.patch("victor.framework.rl.coordinator.get_rl_coordinator", return_value=coordinator):
        ToolCache(ttl=60)
    coordinator.get_learner.assert_called_once_with("cache_eviction")


def test_custom_provider_wins_until_unbound():
    custom = mock.Mock()
    port.set_rl_coordinator_provider(lambda: custom)
    try:
        assert port.get_rl_coordinator() is custom
    finally:
        port.set_rl_coordinator_provider(None)
    assert port._provider is None


def test_failed_binding_raises_instead_of_returning_none():
    port.set_rl_coordinator_provider(None)
    with mock.patch("victor.core.bootstrap.bind_default_rl_coordinator"):
        with pytest.raises(RuntimeError, match="no RL coordinator provider"):
            port.get_rl_coordinator()


def test_infrastructure_callers_tolerate_an_unavailable_coordinator():
    from victor.storage.cache.tool_cache import ToolCache

    port.set_rl_coordinator_provider(mock.Mock(side_effect=RuntimeError("db down")))
    try:
        ToolCache(ttl=60)  # graceful non-RL fallback, as before
    finally:
        port.set_rl_coordinator_provider(None)


def test_framework_coordinator_satisfies_the_port():
    members = ["get_learner", "record_outcome"]
    for name in members:
        port_params = list(inspect.signature(getattr(port.RLCoordinatorProtocol, name)).parameters)
        impl_params = list(inspect.signature(getattr(framework_rl.RLCoordinator, name)).parameters)
        assert impl_params[: len(port_params)] == port_params, name
    assert isinstance(framework_rl.RLCoordinator.is_closed, property)


@pytest.mark.parametrize(
    "module",
    [
        "victor/storage/cache/tool_cache.py",
        "victor/tools/hybrid_tool_selector.py",
        "victor/core/rl/hooks.py",
        "victor/core/rl/coordinator_port.py",
    ],
)
def test_infrastructure_no_longer_imports_the_framework_coordinator(module):
    root = Path(port.__file__).resolve().parents[3]
    tree = ast.parse((root / module).read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "victor.framework.rl.coordinator" not in imported
