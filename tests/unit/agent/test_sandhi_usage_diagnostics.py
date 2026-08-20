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

"""G4 (sandhi typed-integration gap ledger): consume ``sandhi_usage`` diagnostics.

The Sandhi transport attaches ``metadata["sandhi_usage"]`` — retry attempts,
usage completeness, outcome, upstream request id — to every completion and the
final stream chunk. Before this change nothing in Victor read it: the typed
retry/completeness signal was recorded by the transport and dropped.

Now: chunks fold diagnostics into ``StreamingChatContext.provider_diagnostics``
(aggregated per turn), and finalize surfaces them on the canonical
``stream_completed`` usage record plus ``StreamMetrics.metadata``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

from victor.agent.metrics_collector import MetricsCollector, MetricsCollectorConfig
from victor.agent.streaming.context import StreamingChatContext


class _FakeUsageLogger:
    def __init__(self) -> None:
        self.events: List[Tuple[str, Dict[str, Any]]] = []

    def log_event(self, name: str, data: Dict[str, Any]) -> None:
        self.events.append((name, data))


def _collector() -> Tuple[MetricsCollector, _FakeUsageLogger]:
    logger = _FakeUsageLogger()
    collector = MetricsCollector(
        config=MetricsCollectorConfig(model="glm-5.2", provider="zai"),
        usage_logger=logger,
    )
    return collector, logger


def _stream_completed(logger: _FakeUsageLogger) -> Dict[str, Any]:
    completed = [data for name, data in logger.events if name == "stream_completed"]
    assert completed, "no stream_completed event was emitted"
    return completed[-1]


class TestContextAggregation:
    def test_single_call_diagnostics_recorded(self):
        ctx = StreamingChatContext(user_message="hi")
        ctx.record_provider_diagnostics(
            {
                "attempts": 2,
                "completeness": "complete",
                "outcome": "ok",
                "upstream_request_id": "req-1",
            }
        )

        assert ctx.provider_diagnostics == {
            "provider_calls": 1,
            "attempts_total": 2,
            "last_outcome": "ok",
            "last_upstream_request_id": "req-1",
        }

    def test_multi_call_aggregation_sums_and_keeps_last(self):
        ctx = StreamingChatContext(user_message="hi")
        ctx.record_provider_diagnostics(
            {"attempts": 1, "completeness": "complete", "upstream_request_id": "req-1"}
        )
        ctx.record_provider_diagnostics(
            {
                "attempts": 3,
                "completeness": "partial",
                "outcome": "ok",
                "upstream_request_id": "req-2",
            }
        )

        diag = ctx.provider_diagnostics
        assert diag["provider_calls"] == 2
        assert diag["attempts_total"] == 4
        assert diag["incomplete_usage_count"] == 1
        assert diag["last_upstream_request_id"] == "req-2"

    def test_malformed_attempts_defaults_to_one(self):
        ctx = StreamingChatContext(user_message="hi")
        ctx.record_provider_diagnostics({"attempts": "garbage"})
        ctx.record_provider_diagnostics({})

        assert ctx.provider_diagnostics["attempts_total"] == 2
        assert ctx.provider_diagnostics["provider_calls"] == 2


class TestFinalizeSurfacesDiagnostics:
    def test_stream_completed_record_carries_sandhi_usage(self):
        collector, logger = _collector()
        collector.init_stream_metrics()

        diagnostics = {
            "provider_calls": 3,
            "attempts_total": 5,
            "incomplete_usage_count": 1,
            "last_outcome": "ok",
            "last_upstream_request_id": "req-9",
        }
        metrics = collector.finalize_stream_metrics(
            {"prompt_tokens": 100, "completion_tokens": 10},
            provider_diagnostics=diagnostics,
        )

        record = _stream_completed(logger)
        assert record["sandhi_usage"] == diagnostics
        assert metrics.metadata["sandhi_usage"] == diagnostics
        # Token fields unaffected (no regression on C0)
        assert record["prompt_tokens"] == 100

    def test_no_diagnostics_leaves_record_unchanged(self):
        collector, logger = _collector()
        collector.init_stream_metrics()

        metrics = collector.finalize_stream_metrics({"prompt_tokens": 10, "completion_tokens": 1})

        record = _stream_completed(logger)
        assert "sandhi_usage" not in record
        assert "sandhi_usage" not in metrics.metadata


class TestServicePassthrough:
    def test_metrics_service_forwards_diagnostics(self):
        from victor.agent.services.metrics_service import AgentMetricsService

        collector = MagicMock()
        collector.finalize_stream_metrics.return_value = None
        service = AgentMetricsService(
            metrics_collector=collector,
            session_cost_tracker=MagicMock(),
            cumulative_token_usage={},
        )

        service.finalize_stream_metrics(
            {"prompt_tokens": 1}, provider_diagnostics={"attempts_total": 2}
        )

        collector.finalize_stream_metrics.assert_called_once_with(
            {"prompt_tokens": 1}, provider_diagnostics={"attempts_total": 2}
        )
