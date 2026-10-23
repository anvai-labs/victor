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


"""Port to the RL coordinator for code below the framework.

The coordinator (learners, SQLite outcome store, experiments) is a framework
subsystem. Infrastructure that feeds it -- the tool cache, the hybrid tool
selector, the RL hook registry -- depends on this protocol and accessor
instead of the framework module. The concrete binding is owned by the
composition root (:func:`victor.core.bootstrap.bind_default_rl_coordinator`),
which this accessor asks for on first use, so callers that run before or
without a bootstrap behave exactly as before: they reach the one framework
singleton through ``victor.framework.rl.coordinator.get_rl_coordinator``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, runtime_checkable

__all__ = [
    "RLCoordinatorProtocol",
    "get_rl_coordinator",
    "set_rl_coordinator_provider",
]


@runtime_checkable
class RLCoordinatorProtocol(Protocol):
    """The coordinator surface used below the framework."""

    @property
    def is_closed(self) -> bool: ...

    def get_learner(self, name: str) -> Optional[Any]: ...

    def record_outcome(self, learner_name: str, outcome: Any, vertical: str = ...) -> None: ...


_provider: Optional[Callable[[], RLCoordinatorProtocol]] = None


def set_rl_coordinator_provider(
    provider: Optional[Callable[[], RLCoordinatorProtocol]],
) -> None:
    """Bind (or with ``None``, unbind) the callable that yields the coordinator."""
    global _provider
    _provider = provider


def get_rl_coordinator() -> RLCoordinatorProtocol:
    """Return the RL coordinator, binding the default provider on first use."""
    provider = _provider
    if provider is None:
        from victor.core.bootstrap import bind_default_rl_coordinator

        bind_default_rl_coordinator()
        provider = _provider
    if provider is None:
        raise RuntimeError("no RL coordinator provider is bound")
    return provider()
