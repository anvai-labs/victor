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

"""Provider runtime boundaries for AgentOrchestrator.

This module extracts provider runtime wiring from orchestrator construction.

Migration Notes (2026-05-01):
- ProviderCoordinator removed from ProviderRuntimeComponents (unused internally)
- ProviderSwitchCoordinator removed from ProviderRuntimeComponents (unused internally)
- ProviderService is the canonical owner for provider operations
- Removed coordinator root shims stay absent in v1.0.0+
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LazyRuntimeProxy(Generic[T]):
    """Thread-safe lazy proxy for runtime components."""

    _INTERNAL_ATTRS = {"_factory", "_name", "_instance", "_lock"}

    def __init__(self, *, factory: Callable[[], T], name: str) -> None:
        self._factory = factory
        self._name = name
        self._instance: Optional[T] = None
        self._lock = threading.Lock()

    def get_instance(self) -> T:
        """Get the underlying instance, creating it on first access."""
        instance = self._instance
        if instance is not None:
            return instance

        with self._lock:
            if self._instance is None:
                self._instance = self._factory()
                logger.debug("Lazily initialized runtime component: %s", self._name)
            return self._instance

    @property
    def initialized(self) -> bool:
        """Whether the wrapped instance has been created."""
        return self._instance is not None

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.get_instance(), attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr in self._INTERNAL_ATTRS:
            object.__setattr__(self, attr, value)
            return
        setattr(self.get_instance(), attr, value)


@dataclass(frozen=True)
class ProviderRuntimeComponents:
    """Provider runtime handles exposed to the orchestrator facade.

    Migration Notes (2026-05-01):
    - provider_coordinator removed: use ProviderService instead
    - provider_switch_coordinator removed: use ProviderService instead
    - Removed coordinator shims must not be reintroduced
    """

    pool: Optional[Any] = None


def create_provider_runtime_components(
    *,
    settings: Any,
    provider_manager: Any,
    pool: Optional[Any] = None,
    get_provider_service: Optional[Callable[[], Any]] = None,
) -> ProviderRuntimeComponents:
    """Create lazy provider runtime components for orchestrator wiring.

    ``get_provider_service`` is accepted as a no-op compatibility kwarg so
    mixed-version environments do not fail during the coordinator-removal
    migration. The canonical ProviderService is now initialized directly by the
    orchestrator.
    """

    if get_provider_service is not None:
        logger.debug(
            "Ignoring deprecated get_provider_service compatibility kwarg in "
            "create_provider_runtime_components()"
        )

    return ProviderRuntimeComponents(
        pool=pool,
    )
