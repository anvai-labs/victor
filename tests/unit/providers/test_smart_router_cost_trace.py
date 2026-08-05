# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""L4: cost/latency-aware routing — the cost factor honors the C0 per-turn cost trace.

When the recent per-turn record (TaskExecutionReport) shows turns are expensive or slow, the
cost score biases toward cheaper/faster local providers; without a trace the static profile
preference is used unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

from victor.providers.smart_router import (
    _COST_TRACE_LATENCY_THRESHOLD_S,
    _COST_TRACE_USD_THRESHOLD,
    RoutingDecisionEngine,
)


def _engine(cost_pref: str = "normal") -> RoutingDecisionEngine:
    # _score_cost only reads self.profile.cost_preference + the cost_trace arg, so we bypass the
    # heavy constructor and stub the profile.
    eng = RoutingDecisionEngine.__new__(RoutingDecisionEngine)
    eng.profile = SimpleNamespace(cost_preference=cost_pref)
    return eng


async def test_no_trace_uses_static_preference():
    eng = _engine("normal")
    assert await eng._score_cost("ollama") == 0.7
    assert await eng._score_cost("anthropic") == 0.7


async def test_over_budget_trace_biases_to_local():
    eng = _engine("high")  # static pref would favor cloud; trace must override
    trace = {
        "total_cost_usd": _COST_TRACE_USD_THRESHOLD + 0.01,
        "duration_seconds": 1.0,
    }
    assert await eng._score_cost("ollama", cost_trace=trace) == 1.0
    assert await eng._score_cost("anthropic", cost_trace=trace) == 0.2


async def test_slow_trace_biases_to_local():
    eng = _engine("high")
    trace = {
        "total_cost_usd": 0.0,
        "duration_seconds": _COST_TRACE_LATENCY_THRESHOLD_S + 5,
    }
    assert await eng._score_cost("vllm", cost_trace=trace) == 1.0
    assert await eng._score_cost("openai", cost_trace=trace) == 0.2


async def test_under_threshold_trace_does_not_override():
    eng = _engine("normal")
    trace = {"total_cost_usd": 0.001, "duration_seconds": 2.0}  # cheap + fast
    # Falls through to static preference (normal -> 0.7 for both).
    assert await eng._score_cost("ollama", cost_trace=trace) == 0.7
    assert await eng._score_cost("anthropic", cost_trace=trace) == 0.7


async def test_empty_trace_is_ignored():
    eng = _engine("low")
    assert await eng._score_cost("ollama", cost_trace={}) == 1.0  # static low -> local 1.0
    assert await eng._score_cost("anthropic", cost_trace={}) == 0.3
