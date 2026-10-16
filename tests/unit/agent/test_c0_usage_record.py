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

from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from victor.agent.metrics_collector import MetricsCollector, MetricsCollectorConfig
from victor.providers.base import BaseProvider, StreamChunk


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
    """Wire the real streaming metrics capability, accumulator, tracker and collector."""
    from victor.agent.factory.chat_runtime_bindings import _ChatStreamMetricsView
    from victor.agent.services.chat_runtime_services import ChatStreamMetrics
    from victor.agent.services.metrics_service import AgentMetricsService
    from victor.agent.session_cost_tracker import SessionCostTracker

    collector, _logger = _collector()
    tracker = SessionCostTracker(provider="ollama", model="qwen3.5:4b")
    # This is the same dictionary shared by the production stream and task-report paths.
    cumulative = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    service = AgentMetricsService(
        metrics_collector=collector,
        session_cost_tracker=tracker,
        cumulative_token_usage=cumulative,
    )
    owner = MagicMock(
        _metrics_collector=collector,
        _metrics_coordinator=service,
        _cumulative_token_usage=cumulative,
    )
    return service, ChatStreamMetrics(_ChatStreamMetricsView(owner)), owner


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
        {"prompt_tokens": 500, "completion_tokens": 50},
    ],
    ids=["reported-total", "derived-total"],
)
async def test_task_report_counts_stream_usage_once_through_presentation(usage):
    service, stream_metrics, _owner = _metrics_service()

    async def stream(**kwargs: Any) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(usage=usage)
        yield StreamChunk(content="done", usage=usage, is_final=True)

    provider = MagicMock(spec=BaseProvider)
    provider.name = "ollama"
    provider.stream = MagicMock(side_effect=stream)
    service.start_task_report("explain the codebase", task_type="general")
    stream_metrics.begin()
    # Drive the same capability/view/invocation-owner path as production.
    # Repeated usage snapshots are one invocation, not additional token charges.
    async with aclosing(stream_metrics.stream(provider, messages=[], model="qwen3.5:4b")) as chunks:
        assert [chunk.content async for chunk in chunks] == ["", "done"]
    # Presentation finalization must not settle that invocation a second time,
    # even if a consumer asks for the final metrics again.
    stream_metrics.finalize(usage)
    stream_metrics.finalize(usage)
    report = service.finish_task_report(True)

    provider.stream.assert_called_once_with(messages=[], model="qwen3.5:4b")
    assert report["request_count"] == 1
    assert report["api_prompt_tokens"] == usage["prompt_tokens"]
    assert report["api_completion_tokens"] == usage["completion_tokens"]
    assert report["api_total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
