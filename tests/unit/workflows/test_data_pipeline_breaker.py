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

"""FOU-C1 selector mirror: the data_pipeline CircuitBreaker adapter.

The pipeline breaker is an API adapter over the canonical
``victor.core.circuit_breaker.CircuitBreaker`` (legacy call()/is_open()
interface preserved). Pins: delegation identity, legacy parameter
mapping, success/failure recording through the canonical state machine,
and the open-circuit fast-fail — positive and negative.
"""

import pytest

from victor.core.circuit_breaker import CircuitBreaker as CanonicalCircuitBreaker
from victor.workflows.data_pipeline import CircuitBreaker as PipelineCircuitBreaker


class TestAdapterDelegation:
    def test_is_canonical_breaker_underneath(self):
        adapter = PipelineCircuitBreaker(max_failures=3, reset_timeout=30)
        assert isinstance(adapter._breaker, CanonicalCircuitBreaker)
        assert adapter._breaker.name == "data_pipeline"

    def test_legacy_parameters_map_to_canonical_config(self):
        adapter = PipelineCircuitBreaker(max_failures=3, reset_timeout=30)
        # from_config flattens CircuitBreakerConfig onto the breaker's own
        # attributes; the config's timeout_seconds bridges to the breaker's
        # recovery_timeout via the alias property (a non-default value must
        # survive the mapping, not silently fall back to the 30s default).
        assert adapter._breaker.failure_threshold == 3
        assert adapter._breaker.recovery_timeout == 30.0
        nondefault = PipelineCircuitBreaker(max_failures=7, reset_timeout=120)
        assert nondefault._breaker.recovery_timeout == 120.0
        assert nondefault._breaker.failure_threshold == 7


class TestLegacyCallSemantics:
    def test_successful_call_records_success_and_returns_result(self):
        adapter = PipelineCircuitBreaker()
        assert adapter.call(lambda: "payload") == "payload"
        assert not adapter.is_open()

    def test_failures_propagate_and_record(self):
        adapter = PipelineCircuitBreaker(max_failures=5)

        def _boom():
            raise RuntimeError("upstream down")

        for _ in range(4):  # below threshold
            with pytest.raises(RuntimeError):
                adapter.call(_boom)
        assert not adapter.is_open()  # still closed below the threshold

    def test_threshold_failures_open_the_circuit(self):
        adapter = PipelineCircuitBreaker(max_failures=2)

        def _boom():
            raise RuntimeError("upstream down")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                adapter.call(_boom)
        assert adapter.is_open()

    def test_open_circuit_fails_fast_without_calling_the_function(self):
        adapter = PipelineCircuitBreaker(max_failures=1)
        calls = {"count": 0}

        def _boom():
            calls["count"] += 1
            raise RuntimeError("upstream down")

        with pytest.raises(RuntimeError):
            adapter.call(_boom)
        assert adapter.is_open()
        # Fast-fail: the wrapped function must NOT be invoked once open.
        with pytest.raises(Exception, match="Circuit breaker open"):
            adapter.call(_boom)
        assert calls["count"] == 1
