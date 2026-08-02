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

"""C0 completion: the canonical cost record carries the dominant cost term (tokens).

Trace-first measurement (capturing a live `victor ui` run) revealed that the
``stream_completed`` usage record was emitted with timing only — no token fields — so the
dominant cost term (tokens per turn) was invisible in the sink the RL trace-miner and cost
analysis read. This pins the fix: finalized stream metrics emit prompt/completion/total tokens.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from victor.agent.metrics_collector import MetricsCollector, MetricsCollectorConfig


class _FakeUsageLogger:
    """Captures log_event calls so we can assert on the emitted record."""

    def __init__(self) -> None:
        self.events: List[Tuple[str, Dict[str, Any]]] = []

    def log_event(self, name: str, data: Dict[str, Any]) -> None:
        self.events.append((name, data))


def _collector() -> Tuple[MetricsCollector, _FakeUsageLogger]:
    logger = _FakeUsageLogger()
    collector = MetricsCollector(
        config=MetricsCollectorConfig(model="qwen3.5:4b", provider="ollama"),
        usage_logger=logger,
    )
    return collector, logger


def _stream_completed(logger: _FakeUsageLogger) -> Dict[str, Any]:
    completed = [data for name, data in logger.events if name == "stream_completed"]
    assert completed, "no stream_completed event was emitted"
    return completed[-1]


def test_stream_completed_record_carries_token_fields():
    collector, logger = _collector()
    collector.init_stream_metrics()

    collector.finalize_stream_metrics(
        {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500}
    )

    record = _stream_completed(logger)
    assert record["prompt_tokens"] == 1200
    assert record["completion_tokens"] == 300
    # total is derived from prompt+completion so the record is self-consistent
    assert record["total_tokens"] == 1500
    # timing fields preserved (no regression)
    assert "ttft" in record and "total_duration" in record


def test_stream_completed_record_includes_cache_tokens():
    collector, logger = _collector()
    collector.init_stream_metrics()

    collector.finalize_stream_metrics(
        {
            "prompt_tokens": 800,
            "completion_tokens": 200,
            "cache_read_input_tokens": 640,
            "cache_creation_input_tokens": 160,
        }
    )

    record = _stream_completed(logger)
    assert record["cache_read_tokens"] == 640
    assert record["cache_write_tokens"] == 160


def test_zero_usage_still_emits_explicit_token_fields():
    # Even with no usage, the record must carry explicit zero token fields rather than
    # omitting them — downstream readers should never have to guess whether 0 means
    # "no tokens" or "field absent".
    collector, logger = _collector()
    collector.init_stream_metrics()

    collector.finalize_stream_metrics(None)

    record = _stream_completed(logger)
    assert record["prompt_tokens"] == 0
    assert record["completion_tokens"] == 0
    assert record["total_tokens"] == 0


def _metrics_service():
    """Wire a real AgentMetricsService over a real SessionCostTracker + collector."""
    from victor.agent.services.metrics_service import AgentMetricsService
    from victor.agent.session_cost_tracker import SessionCostTracker

    collector, _logger = _collector()
    tracker = SessionCostTracker(provider="ollama", model="qwen3.5:4b")
    # The legacy cumulative dict that previously (and wrongly) sourced the task report.
    cumulative = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    service = AgentMetricsService(
        metrics_collector=collector,
        session_cost_tracker=tracker,
        cumulative_token_usage=cumulative,
    )
    return service, collector


def test_task_report_tokens_come_from_session_cost_tracker():
    # The regression: the task report sourced tokens from `_cumulative_token_usage`, which has
    # no writer on the service path, so every report read total_tokens=0 even though the turn's
    # usage was correctly recorded to the SessionCostTracker via finalize_stream_metrics. The
    # snapshot must read the SAME authoritative tracker the cost/cache fields already use.
    service, collector = _metrics_service()

    service.start_task_report("explain the codebase", task_type="general")
    # Simulate one finalized streaming turn (records to SessionCostTracker).
    collector.init_stream_metrics()
    service.finalize_stream_metrics(
        {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200}
    )
    report = service.finish_task_report(True)

    assert report["api_prompt_tokens"] == 1000
    assert report["api_completion_tokens"] == 200
    assert report["api_total_tokens"] == 1200


def test_task_report_tokens_not_double_counted():
    # Guard against over-counting: a single finalized turn must produce a delta equal to that
    # turn's usage, not a multiple of it.
    service, collector = _metrics_service()

    service.start_task_report("task", task_type="general")
    collector.init_stream_metrics()
    service.finalize_stream_metrics({"prompt_tokens": 500, "completion_tokens": 50})
    report = service.finish_task_report(True)

    assert report["api_total_tokens"] == 550
