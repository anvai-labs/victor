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

"""DEPRECATED: circuit breaker moved to :mod:`victor.core.circuit_breaker`.

FOU-C1 consolidated Victor's duplicated circuit-breaker implementations onto a
single canonical state machine. ``CircuitBreaker``, ``CircuitBreakerRegistry``,
and the shared ``CircuitState``/``CircuitBreakerConfig``/``CircuitBreakerError``
types now live in ``victor.core.circuit_breaker`` (importable by every layer,
including agent and workflows, which must not import ``victor.providers``).

This module is a pure re-export shim kept so the historical import path
(``from victor.providers.circuit_breaker import CircuitBreaker``) keeps working
for external consumers. New code should import from
``victor.core.circuit_breaker`` directly.

Layer-specific wrappers that delegate to the canonical implementation:
- victor.providers.resilience: ProviderCircuitBreaker adapter for ResilientProvider
- victor.agent.resilience: MultiCircuitBreaker for many named circuits
"""

from victor.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState,
)

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitBreakerRegistry",
    "CircuitState",
]
