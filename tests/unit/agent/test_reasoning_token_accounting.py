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

"""G2b (sandhi typed-integration gap ledger): reasoning tokens reach cost accounting.

Sandhi's typed response (``UsageV2``) reports ``reasoning_tokens`` where the
provider does; Victor's usage dict dropped them, silently under-counting
reasoning models. The folding invariant mirrors sandhi's ``billable()``:
providers either fold reasoning into ``completion_tokens`` (OpenAI, Anthropic)
or report it separately (Gemini ``thoughtsTokenCount``). Folded reasoning must
never be double-priced; unfolded reasoning is billed output that
``completion_tokens`` does not contain.
"""

from __future__ import annotations

from victor.agent.stream_handler import StreamMetrics
from victor.config.metrics_capabilities import ProviderMetricsCapabilities
from victor.providers.usage_parsing import usage_dict_from_neutral


def _capabilities() -> ProviderMetricsCapabilities:
    return ProviderMetricsCapabilities(
        provider="gemini",
        model="test-model",
        cost_enabled=True,
        input_cost_per_mtok=1.0,
        output_cost_per_mtok=10.0,
    )


class TestNeutralMapping:
    def test_reasoning_tokens_surfaced_from_neutral_usage(self):
        usage = usage_dict_from_neutral(
            {"tokens_in": 100, "tokens_out": 50, "reasoning_tokens": 25},
            None,
            slug="gemini",
        )
        assert usage is not None
        assert usage["reasoning_tokens"] == 25

    def test_absent_reasoning_leaves_dict_unchanged(self):
        usage = usage_dict_from_neutral({"tokens_in": 100, "tokens_out": 50}, None, slug="openai")
        assert usage is not None
        assert "reasoning_tokens" not in usage


class TestStreamMetricsAccounting:
    def test_record_usage_accumulates_reasoning(self):
        metrics = StreamMetrics()
        metrics.record_usage({"prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 7})
        metrics.record_usage({"prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 3})
        assert metrics.reasoning_tokens == 10

    def test_folded_reasoning_is_never_double_priced(self):
        """OpenAI-style: reasoning is already inside completion_tokens."""
        with_reasoning = StreamMetrics(prompt_tokens=1000, completion_tokens=500)
        with_reasoning.reasoning_tokens = 300  # 300 <= 500 → folded
        with_reasoning.calculate_cost(_capabilities())

        without = StreamMetrics(prompt_tokens=1000, completion_tokens=500)
        without.calculate_cost(_capabilities())

        assert with_reasoning.total_cost == without.total_cost

    def test_unfolded_reasoning_is_priced_at_output_rate(self):
        """Gemini-style: thoughts tokens are billed output outside candidates count."""
        metrics = StreamMetrics(prompt_tokens=0, completion_tokens=100)
        metrics.reasoning_tokens = 400  # 400 > 100 → unfolded
        metrics.calculate_cost(_capabilities())

        expected_output_cost = ((100 + 400) / 1_000_000) * 10.0
        assert metrics.output_cost == expected_output_cost
        assert metrics.cost_calculated


class TestCumulativeUsageKey:
    def test_streaming_context_accumulates_reasoning(self):
        from victor.agent.streaming.context import StreamingChatContext

        ctx = StreamingChatContext(user_message="hi")
        assert "reasoning_tokens" in ctx.cumulative_usage
        # The accumulation loop iterates the context's keys — simulate it.
        raw_usage = {"prompt_tokens": 5, "completion_tokens": 2, "reasoning_tokens": 9}
        for key in ctx.cumulative_usage:
            ctx.cumulative_usage[key] += raw_usage.get(key, 0)
        assert ctx.cumulative_usage["reasoning_tokens"] == 9


class TestStreamCompletedRecord:
    def test_stream_completed_carries_reasoning_tokens(self):
        from victor.agent.metrics_collector import MetricsCollector, MetricsCollectorConfig

        events = []

        class _Logger:
            def log_event(self, name, data):
                events.append((name, data))

        collector = MetricsCollector(
            config=MetricsCollectorConfig(model="gemini-3-pro", provider="google"),
            usage_logger=_Logger(),
        )
        collector.init_stream_metrics()
        collector.finalize_stream_metrics(
            {"prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 12}
        )

        record = [d for n, d in events if n == "stream_completed"][-1]
        assert record["reasoning_tokens"] == 12

    def test_zero_reasoning_stays_absent(self):
        from victor.agent.metrics_collector import MetricsCollector, MetricsCollectorConfig

        events = []

        class _Logger:
            def log_event(self, name, data):
                events.append((name, data))

        collector = MetricsCollector(
            config=MetricsCollectorConfig(model="glm-5.2", provider="zai"),
            usage_logger=_Logger(),
        )
        collector.init_stream_metrics()
        collector.finalize_stream_metrics({"prompt_tokens": 10, "completion_tokens": 5})

        record = [d for n, d in events if n == "stream_completed"][-1]
        assert "reasoning_tokens" not in record
