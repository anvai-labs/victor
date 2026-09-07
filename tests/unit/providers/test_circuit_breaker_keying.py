# Copyright 2025 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""Circuit-breaker registry keying on BaseProvider.

The breaker key is `provider_{class}_{base_url}` — stable across instances so
breaker state aggregates per provider class + endpoint and the registry stays
bounded. Previously the key embedded `id(self)`, fragmenting state per
instance and leaking one registry entry per object ever created
(co-design review U3-F3).
"""

from __future__ import annotations

from victor.providers.circuit_breaker import CircuitBreakerRegistry
from victor.providers.base import BaseProvider


class _BreakerProbeProvider(BaseProvider):
    """Concrete minimal provider exercising BaseProvider.__init__ wiring."""

    name = "breaker_probe"

    async def chat(self, messages, model=None, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def stream(self, messages, model=None, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def close(self):  # pragma: no cover
        return None


class TestBreakerKeying:
    def setup_method(self):
        CircuitBreakerRegistry.reset_all()

    def teardown_method(self):
        CircuitBreakerRegistry.reset_all()

    def test_same_class_and_url_share_breaker(self):
        a = _BreakerProbeProvider(base_url="http://one:1", circuit_breaker_failure_threshold=1)
        b = _BreakerProbeProvider(base_url="http://one:1", circuit_breaker_failure_threshold=1)
        assert a.circuit_breaker is b.circuit_breaker

    def test_different_url_independent_breakers(self):
        a = _BreakerProbeProvider(base_url="http://one:1")
        c = _BreakerProbeProvider(base_url="http://two:2")
        assert a.circuit_breaker is not c.circuit_breaker

    def test_registry_bounded_across_instances(self):
        # Unique url for this test: registry dict entries persist across
        # reset_all() (which resets state, not membership).
        before = len(CircuitBreakerRegistry._breakers)
        for _ in range(25):
            _BreakerProbeProvider(base_url="http://bounded:9").circuit_breaker
        after = len(CircuitBreakerRegistry._breakers)
        # One entry per (class, url) — not one per instance ever created.
        assert after - before == 1

    def test_trip_via_one_instance_visible_to_the_next(self):
        a = _BreakerProbeProvider(base_url="http://trip:3", circuit_breaker_failure_threshold=1)
        a.circuit_breaker.record_failure(RuntimeError("probe failure"))

        b = _BreakerProbeProvider(base_url="http://trip:3")
        assert b.circuit_breaker.is_open


class TestLazyBreakerKeying:
    def test_post_init_base_url_resolution_shares_state(self):
        """LlamaCpp/VLLM resolve their default base_url AFTER super().__init__
        — the lazy key must reflect the final value, so a default-constructed
        provider that then resolves its URL shares state with one constructed
        with the explicit URL (adversarial-review finding)."""
        CircuitBreakerRegistry.reset_all()
        a = _BreakerProbeProvider()  # base_url=None at init (like llamacpp)
        a.base_url = "http://localhost:8080"  # post-init resolution
        b = _BreakerProbeProvider(base_url="http://localhost:8080")
        assert a.circuit_breaker is b.circuit_breaker

    def test_lazy_creation_deferred_until_first_access(self):
        CircuitBreakerRegistry.reset_all()
        provider = _BreakerProbeProvider(base_url="http://lazy:1")
        assert provider._circuit_breaker is None  # not yet created
        assert provider.circuit_breaker is not None  # created on access
