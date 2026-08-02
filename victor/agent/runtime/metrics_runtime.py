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

"""Metrics/analytics runtime boundaries for AgentOrchestrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from victor.agent.runtime.provider_runtime import LazyRuntimeProxy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricsRuntimeComponents:
    """Metrics runtime handles exposed to the orchestrator facade."""

    usage_logger: Any
    streaming_metrics_collector: Optional[Any]
    metrics_collector: LazyRuntimeProxy[Any]
    session_cost_tracker: LazyRuntimeProxy[Any]
    metrics_coordinator: LazyRuntimeProxy[Any]


def _maybe_attach_sandhi_meter(tracker: Any, settings: Optional[Any]) -> None:
    """Attach a SandhiMeter to a freshly built cost tracker (FEP-0020 Phase 2).

    Default-off and byte-identical when disabled: with no settings or
    ``usage_gateway.enabled`` False, this returns before importing sandhi or
    constructing anything. Identity (subject/group) is filled only-when-None so
    the API-server auth seam keeps precedence.
    """
    ug = getattr(settings, "usage_gateway", None) if settings is not None else None
    if ug is None or not getattr(ug, "enabled", False):
        return  # default-off: no sandhi import, no construction

    from victor.observability.sandhi_meter import SandhiMeter, sandhi_available

    if not sandhi_available():
        logger.info(
            "usage_gateway.enabled but sandhi-gateway is not installed "
            "(pip install 'victor-ai[sandhi]'); attribution disabled"
        )
        return

    try:
        sink = getattr(ug, "sink_path", None)
        if not sink:
            from victor.config.settings import get_project_paths

            sink = get_project_paths().global_logs_dir / "usage_events.jsonl"
        tracker._sandhi = SandhiMeter(sink_path=str(sink))
        if tracker.subject_id is None:
            tracker.subject_id = getattr(ug, "subject_id", None)
        if tracker.group_id is None:
            tracker.group_id = getattr(ug, "group_id", None)
    except Exception as exc:
        logger.debug("sandhi meter attach failed (ignored): %s", exc)


def create_metrics_runtime_components(
    *,
    factory: Any,
    provider: Any,
    model: str,
    debug_logger: Any,
    cumulative_token_usage: Dict[str, int],
    tool_cost_lookup: Callable[[str], Any],
    settings: Optional[Any] = None,
) -> MetricsRuntimeComponents:
    """Create metrics runtime components with lazy collector/coordinator wiring."""
    usage_logger = factory.create_usage_logger()
    streaming_metrics_collector = factory.create_streaming_metrics_collector()

    def _build_metrics_collector() -> Any:
        return factory.create_metrics_collector(
            streaming_metrics_collector=streaming_metrics_collector,
            usage_logger=usage_logger,
            debug_logger=debug_logger,
            tool_cost_lookup=tool_cost_lookup,
        )

    metrics_collector = LazyRuntimeProxy(
        factory=_build_metrics_collector,
        name="metrics_collector",
    )

    def _build_session_cost_tracker() -> Any:
        from victor.agent.session_cost_tracker import SessionCostTracker

        tracker = SessionCostTracker(
            provider=provider.name,
            model=model,
        )
        _maybe_attach_sandhi_meter(tracker, settings)
        return tracker

    session_cost_tracker = LazyRuntimeProxy(
        factory=_build_session_cost_tracker,
        name="session_cost_tracker",
    )

    def _build_metrics_coordinator() -> Any:
        from victor.agent.services.metrics_service import AgentMetricsService

        return AgentMetricsService(
            metrics_collector=metrics_collector.get_instance(),
            session_cost_tracker=session_cost_tracker.get_instance(),
            cumulative_token_usage=cumulative_token_usage,
        )

    metrics_coordinator = LazyRuntimeProxy(
        factory=_build_metrics_coordinator,
        name="metrics_coordinator",
    )

    return MetricsRuntimeComponents(
        usage_logger=usage_logger,
        streaming_metrics_collector=streaming_metrics_collector,
        metrics_collector=metrics_collector,
        session_cost_tracker=session_cost_tracker,
        metrics_coordinator=metrics_coordinator,
    )
