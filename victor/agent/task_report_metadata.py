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

"""Pure task-report metadata builders (ADR-019 orchestrator decomposition).

These functions were extracted verbatim from ``AgentOrchestrator`` as the first,
lowest-risk slice of the god-object decomposition (TD-14). They are *pure* over
their inputs — a stream context, the context service, the unified tracker, and
task-type candidates — so they carry no orchestrator coupling and are directly
unit-testable. The orchestrator now delegates to them at the task-report seams.

Behaviour is preserved exactly: every ``getattr`` default, bound, and ordering
matches the original orchestrator methods.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

logger = logging.getLogger(__name__)


def resolve_task_type(
    stream_ctx: Any,
    unified_tracker: Any,
    candidate_task_types: Iterable[Any],
) -> str:
    """Resolve the most specific available task type for task-level metrics.

    Precedence: stream context's unified type → coarse type → tracker task type →
    the first truthy value in ``candidate_task_types`` → ``"default"``.
    """
    if stream_ctx is not None:
        unified_task_type = getattr(getattr(stream_ctx, "unified_task_type", None), "value", None)
        if unified_task_type:
            return str(unified_task_type)
        coarse_task_type = getattr(stream_ctx, "coarse_task_type", None)
        if coarse_task_type:
            return str(coarse_task_type)

    tracker_task_type = getattr(unified_tracker, "task_type", None)
    if tracker_task_type:
        return str(tracker_task_type)

    for value in candidate_task_types:
        if value:
            return str(value)

    return "default"


def build_compaction_metadata(stream_ctx: Any, context_service: Any) -> Dict[str, Any]:
    """Collect compaction continuity signals for the current task report."""
    perf_metrics: Dict[str, Any] = {}
    if context_service is not None and hasattr(context_service, "get_performance_metrics"):
        try:
            perf_metrics = context_service.get_performance_metrics() or {}
        except Exception as exc:  # noqa: BLE001 - metadata must never break the report
            logger.debug(
                "Failed to read context performance metrics for task report: %s",
                exc,
            )

    summary = ""
    occurred = False
    messages_removed = 0
    if stream_ctx is not None:
        summary = str(getattr(stream_ctx, "compaction_summary", "") or "")
        occurred = bool(
            getattr(stream_ctx, "compaction_occurred", False)
            or getattr(stream_ctx, "last_compaction_turn", -1) >= 0
        )
        messages_removed = int(getattr(stream_ctx, "compaction_message_removed_count", 0) or 0)

    saved_tokens = int(perf_metrics.get("last_compaction_saved_tokens", 0) or 0)
    return {
        "occurred": occurred or bool(summary) or messages_removed > 0 or saved_tokens > 0,
        "summary": summary,
        "messages_removed": messages_removed,
        "saved_tokens": saved_tokens,
        "strategy": str(getattr(stream_ctx, "last_compaction_strategy", "") or ""),
        "reason": str(getattr(stream_ctx, "last_compaction_reason", "") or ""),
        "policy_reason": str(getattr(stream_ctx, "last_compaction_policy_reason", "") or ""),
    }


def build_continuation_metadata(stream_ctx: Any) -> Dict[str, Any]:
    """Collect bounded continuation-ledger state for reporting/export paths."""
    if stream_ctx is None:
        return {}

    metadata: Dict[str, Any] = {}
    task_intent = str(getattr(stream_ctx, "task_intent", "") or "").strip()
    if task_intent:
        metadata["task_intent"] = task_intent

    plan_steps = [
        str(item).strip()
        for item in (getattr(stream_ctx, "plan_steps", []) or [])
        if str(item).strip()
    ][:6]
    if plan_steps:
        metadata["plan_steps"] = plan_steps

    intent_log = [
        dict(item)
        for item in (getattr(stream_ctx, "intent_log", []) or [])
        if isinstance(item, dict)
    ][-6:]
    if intent_log:
        metadata["intent_log"] = intent_log

    resume_summary = str(getattr(stream_ctx, "resume_summary", "") or "").strip()
    if resume_summary:
        metadata["resume_summary"] = resume_summary

    if bool(getattr(stream_ctx, "degraded_resume_state", False)):
        metadata["degraded_resume_state"] = True

    build_ledger = getattr(stream_ctx, "build_continuation_ledger", None)
    if callable(build_ledger):
        try:
            continuation_ledger = str(
                build_ledger(max_events=4, max_plan_steps=4, max_chars=500) or ""
            ).strip()
        except TypeError:
            continuation_ledger = str(build_ledger() or "").strip()
        if continuation_ledger:
            metadata["continuation_ledger"] = continuation_ledger

    return metadata
