# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pure helpers for collecting and shaping execution traces.

Extracted from ``prompt_optimizer.py`` (whose learner delegates to these) as
the first step of pulling trace collection out of the god-class. Everything
here is a free function operating on plain data — session dicts, event dicts,
and ``ExecutionTrace`` lists — with no learner or DB state. Provider/service
imports (used only by credit enrichment) stay lazy so the module is
import-cycle-free.

Contents:
- ``categorize_failure`` — map a tool error string to a FAILURE_HINTS key.
- ``score_session`` — grade a session (harness verdict or tool-failure proxy).
- ``normalize_provider_label`` / ``absorb_session_identity`` /
  ``absorb_run_kind`` — reconcile the identity a JSONL session reports.
- ``scope_traces_to_provider`` — restrict a trace pool to one provider.
- ``merge_traces`` — dedupe trace lists by session, preferring richer detail.
- ``enrich_traces_with_credit`` — attach per-tool credit signals when available.
"""

from __future__ import annotations

import gzip
import logging
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from victor.core.json_utils import json_loads
from victor.framework.rl.learners.trace_analysis import (
    ExecutionTrace,
    HarnessVerdict,
    ToolCallTrace,
)

logger = logging.getLogger(__name__)

# A zero-arg callable returning {session_id: HarnessVerdict} for graded sessions.
HarnessVerdictLookup = Callable[[], Dict[str, HarnessVerdict]]


def categorize_failure(error: str) -> str:
    """Categorize a tool failure error message into a structured category.

    The order of checks matters: more specific patterns come before
    more generic ones to ensure correct classification.

    Returns one of the 13 FAILURE_HINTS keys.
    """
    lower = error.lower()
    # Filesystem errors
    if "not found" in lower and ("file" in lower or "path" in lower):
        return "file_not_found"
    if "directory" in lower and ("read" in lower or "cannot" in lower):
        return "read_directory"
    if "permission denied" in lower or "access denied" in lower:
        return "permission_denied"
    # Edit errors
    if "old_str" in lower and "not found" in lower:
        return "edit_mismatch"
    if "ambiguous" in lower or ("match" in lower and "found" in lower and "times" in lower):
        return "edit_ambiguous"
    if "syntax error" in lower and ("edit" in lower or "after" in lower):
        return "edit_syntax"
    # Tool errors
    if "tool" in lower and "not found" in lower:
        return "tool_not_found"
    if "timeout" in lower or "timed out" in lower:
        return "timeout"
    # Search errors
    if "no results" in lower or "no matches" in lower:
        return "search_no_results"
    # Test failures
    if "test failed" in lower or ("assertion" in lower and "fail" in lower):
        return "test_failure"
    # Shell command errors
    if "command" in lower and ("fail" in lower or "error" in lower):
        return "shell_error"
    # Generic tool errors (after all specific checks)
    if "error" in lower and "tool" in lower:
        return "tool_error"
    return "other"


def score_session(
    verdict: Optional[HarnessVerdict],
    failure_rate: float,
) -> Tuple[float, bool, str]:
    """Score one session: harness verdict when it exists, proxy otherwise.

    Returns ``(completion_score, success, score_source)``.
    """
    if verdict is not None:
        return verdict.completion_score, verdict.success, "harness"
    # No verdict exists for interactive sessions — nothing graded them — so
    # the tool-failure proxy stands, but it is labelled as inference.
    return max(0.0, 1.0 - failure_rate * 1.5), failure_rate < 0.3, "tool_failure_proxy"


def normalize_provider_label(raw: str) -> str:
    """Map a runtime provider class name onto the candidate ``provider`` scope.

    Candidates are stored under short scopes (``moonshot``, ``zai``,
    ``ollama``); the JSONL logs the class name (``MoonshotProvider``,
    ``SandhiOllamaProvider``, ``MoonshotCompatProvider``). Without this
    mapping the two namespaces never meet.
    """
    label = str(raw or "").strip()
    if not label:
        return ""
    for suffix in ("Provider", "Compat"):
        while label.endswith(suffix):
            label = label[: -len(suffix)]
    # Gateway-fronted variants (SandhiOllama → ollama) share the upstream
    # provider's prompt scope; the gateway is transport, not a dialect.
    if label.startswith("Sandhi") and len(label) > len("Sandhi"):
        label = label[len("Sandhi") :]
    return label.lower()


def absorb_run_kind(session: Dict[str, Any], event: Dict[str, Any]) -> None:
    """Record the run kind the emitter stamped on this event.

    Sits beside ``session_id`` on the event rather than inside ``data``,
    because it describes the run rather than the thing that happened. First
    non-empty value wins: a session does not change kind partway through.

    Events written before the emitter tagged them carry nothing, and those
    sessions stay ``unknown`` — deliberately, rather than being guessed from
    prompt text, which is the inference that conflated delegate work with
    benchmark runs in the first place.
    """
    if not isinstance(event, dict) or session.get("run_kind"):
        return
    kind = str(event.get("run_kind") or "").strip().lower()
    if kind:
        session["run_kind"] = kind


def absorb_session_identity(session: Dict[str, Any], data: Dict[str, Any]) -> None:
    """Fill a session's provider/model from any event that carries them.

    ``provider``/``model`` were initialised to ``""`` and never assigned, so
    every collected trace reported ``provider="unknown"`` even though
    ``session_start`` and ``stream_completed`` events carry the real values.
    Evolution therefore reflected over a provider-blind trace pool while
    labelling the resulting candidate with the *current* session's provider.
    First non-empty value wins — a session does not change provider mid-run.
    """
    if not isinstance(data, dict):
        return
    if not session.get("provider"):
        provider = normalize_provider_label(data.get("provider", ""))
        if provider:
            session["provider"] = provider
    if not session.get("model"):
        model = str(data.get("model") or "").strip()
        if model:
            session["model"] = model


def scope_traces_to_provider(
    traces: List[ExecutionTrace],
    provider: str,
    min_traces: int,
) -> List[ExecutionTrace]:
    """Keep only the traces that belong to the provider being evolved.

    Candidates are persisted per ``(section, provider)``, but the trace pool
    is the *global* ``~/.victor/logs/usage.jsonl`` — every project and every
    provider the operator has ever run. Reflecting a ``moonshot`` candidate
    over ZAI/Ollama/DeepSeek failures attributes another model's mistakes to
    Moonshot's prompt.

    Falls back to the unscoped pool when the provider's own traces are too
    few to evolve from; a narrower-but-empty pool would just stall the loop,
    and the caller logs the degraded provenance.
    """
    if not provider or provider == "default":
        return traces
    wanted = normalize_provider_label(provider) or provider.lower()
    scoped = [t for t in traces if normalize_provider_label(t.provider) == wanted]
    if len(scoped) < min_traces:
        logger.info(
            "Provider-scoped traces for '%s' insufficient (%d < %d); "
            "falling back to the unscoped pool (%d traces, mixed provenance).",
            wanted,
            len(scoped),
            min_traces,
            len(traces),
        )
        return traces
    logger.info(
        "Scoped evolution traces to provider '%s': %d of %d.",
        wanted,
        len(scoped),
        len(traces),
    )
    return scoped


def merge_traces(*trace_lists: List[ExecutionTrace]) -> List[ExecutionTrace]:
    """Merge multiple trace lists, deduplicating by session_id.

    When the same session_id appears in multiple sources, the
    version with more tool_call_details wins (richer data).
    """
    by_id: Dict[str, ExecutionTrace] = {}
    for traces in trace_lists:
        for t in traces:
            existing = by_id.get(t.session_id)
            if existing is None:
                by_id[t.session_id] = t
            elif len(t.tool_call_details) > len(existing.tool_call_details):
                # Prefer the richer trace
                by_id[t.session_id] = t
    merged = list(by_id.values())
    merged.sort(key=lambda t: -t.completion_score)
    return merged


def enrich_traces_with_credit(traces: List[ExecutionTrace]) -> None:
    """Enrich execution traces with credit assignment signals.

    Pulls recent credit signals from CreditTrackingService (if available
    via DI container) and attaches per-tool credit data to traces.
    This gives GEPA concrete per-tool value attribution for targeted
    prompt mutations.
    """
    try:
        from victor.core import get_container
        from victor.framework.rl.credit_tracking_service import (
            CreditTrackingService,
        )

        container = get_container()
        service = container.get_optional(CreditTrackingService)
        if service is None:
            return

        tool_summary_cache: Dict[Optional[str], Dict[str, Dict[str, float]]] = {}
        agent_guidance_cache: Dict[Optional[str], Optional[str]] = {}

        for trace in traces:
            session_id = getattr(trace, "session_id", None)
            if session_id not in tool_summary_cache:
                try:
                    tool_summary_cache[session_id] = service.get_tool_credit_summary(
                        session_id=session_id
                    )
                except TypeError:
                    tool_summary_cache[session_id] = service.get_tool_credit_summary()
            if session_id not in agent_guidance_cache:
                try:
                    agent_guidance_cache[session_id] = service.generate_agent_guidance(
                        session_id=session_id
                    )
                except TypeError:
                    agent_guidance_cache[session_id] = service.generate_agent_guidance()

            tool_summary = tool_summary_cache[session_id]
            agent_guidance = agent_guidance_cache[session_id]
            credit_data = []
            for detail in trace.tool_call_details:
                tool_name = getattr(detail, "tool_name", "")
                if tool_name in tool_summary:
                    credit_data.append(
                        {
                            "tool_name": tool_name,
                            "credit": tool_summary[tool_name]["avg_credit"],
                            "total_credit": tool_summary[tool_name]["total_credit"],
                            "call_count": tool_summary[tool_name]["call_count"],
                        }
                    )
            if credit_data:
                trace.credit_signals = credit_data
            if agent_guidance:
                trace.agent_guidance = agent_guidance
    except Exception as exc:
        # Best-effort: traces stay usable without credit signals. The broad
        # catch, however, also hid a missing DI container / unimported
        # credit service — failures that degrade every reflection but look
        # like healthy-but-uninformative traces. Log at debug so the cause
        # is recoverable without spamming the common no-signals case.
        logger.debug("Credit-signal enrichment skipped (best-effort): %s", exc)


class TraceCollector:
    """Builds ``ExecutionTrace`` lists from usage JSONL logs and the store.

    Extracted verbatim from ``PromptOptimizerLearner``; the only learner state
    it needs is the harness-verdict lookup, which is injected as a zero-arg
    callable so this class stays free of DB/learner coupling. Grading and
    identity reconciliation reuse the pure helpers above, so every collector
    scores a session the same way.
    """

    def __init__(self, harness_verdicts: HarnessVerdictLookup):
        self._harness_verdicts = harness_verdicts

    def collect_v1(self, limit: int = 50) -> List[ExecutionTrace]:
        """Collect execution traces from usage.jsonl files."""
        traces: List[ExecutionTrace] = []

        try:
            from victor.config.settings import get_project_paths

            logs_dir = get_project_paths().global_logs_dir
        except Exception:
            logs_dir = Path.home() / ".victor" / "logs"

        # Read from all usage.jsonl files (current + rotated .gz)
        jsonl_files = sorted(logs_dir.glob("usage.*.jsonl.gz")) + [logs_dir / "usage.jsonl"]

        sessions: Dict[str, Dict[str, Any]] = {}
        for jsonl_path in jsonl_files:
            if not jsonl_path.exists():
                continue
            try:
                opener = gzip.open if jsonl_path.suffix == ".gz" else open
                mode = "rt" if jsonl_path.suffix == ".gz" else "r"
                with opener(jsonl_path, mode) as f:
                    for line in f:
                        try:
                            event = json_loads(line.strip())
                            sid = event.get("session_id", "")
                            etype = event.get("event_type", "")
                            data = event.get("data", {})

                            if sid not in sessions:
                                sessions[sid] = {
                                    "tool_calls": 0,
                                    "failures": {},
                                    "provider": "",
                                    "model": "",
                                    "task_type": "default",
                                    "tokens": 0,
                                    "run_kind": "",
                                }

                            absorb_session_identity(sessions[sid], data)
                            absorb_run_kind(sessions[sid], event)

                            if etype == "tool_result":
                                # tool_result is the event actually emitted per
                                # tool invocation (a paired tool_call event is
                                # rarely/never logged); count the call here so
                                # sessions aren't all dropped by the <2-calls filter.
                                sessions[sid]["tool_calls"] += 1
                                if not data.get("success", True):
                                    error = str(
                                        data.get("error") or data.get("result", {}).get("error", "")
                                    )
                                    cat = categorize_failure(error)
                                    sessions[sid]["failures"][cat] = (
                                        sessions[sid]["failures"].get(cat, 0) + 1
                                    )
                            elif etype == "task_classification":
                                sessions[sid]["task_type"] = data.get("task_type", "default")
                        except (JSONDecodeError, KeyError):
                            continue
            except Exception:
                continue

        # Convert to ExecutionTrace objects with quality scoring
        # Quality filter: skip sessions with < 2 tool calls (likely API errors)
        verdicts = self._harness_verdicts()
        for sid, data in list(sessions.items())[-limit:]:
            if data["tool_calls"] < 2:
                continue  # Skip trivially broken sessions

            total_failures = sum(data["failures"].values())
            total_calls = data["tool_calls"]
            failure_rate = total_failures / max(total_calls, 1)

            completion_score, success, score_source = score_session(verdicts.get(sid), failure_rate)

            traces.append(
                ExecutionTrace(
                    session_id=sid,
                    task_type=data["task_type"],
                    provider=data.get("provider") or "unknown",
                    model=data.get("model") or "unknown",
                    tool_calls=total_calls,
                    tool_failures=data["failures"],
                    success=success,
                    completion_score=completion_score,
                    tokens_used=data.get("tokens", 0),
                    score_source=score_source,
                    run_kind=data.get("run_kind") or "unknown",
                )
            )

        # Sort by quality — high-quality traces first for GEPA reflection
        traces.sort(key=lambda t: -t.completion_score)
        return traces

    def collect_v2(self, limit: int = 50) -> List[ExecutionTrace]:
        """Collect enriched execution traces (GEPA v2 with ASI detail).

        Reads the enriched JSONL events which include reasoning_before_call,
        result_summary, error_detail, and duration_ms per tool call.
        Falls back to v1 collection if enriched fields are absent.
        """
        traces: List[ExecutionTrace] = []

        try:
            from victor.config.settings import get_project_paths

            logs_dir = get_project_paths().global_logs_dir
        except Exception:
            logs_dir = Path.home() / ".victor" / "logs"

        jsonl_files = sorted(logs_dir.glob("usage.*.jsonl.gz")) + [logs_dir / "usage.jsonl"]

        sessions: Dict[str, Dict[str, Any]] = {}
        for jsonl_path in jsonl_files:
            if not jsonl_path.exists():
                continue
            try:
                opener = gzip.open if jsonl_path.suffix == ".gz" else open
                mode = "rt" if jsonl_path.suffix == ".gz" else "r"
                with opener(jsonl_path, mode) as f:
                    for line in f:
                        try:
                            event = json_loads(line.strip())
                            sid = event.get("session_id", "")
                            etype = event.get("event_type", "")
                            data = event.get("data", {})

                            if sid not in sessions:
                                sessions[sid] = {
                                    "tool_calls": 0,
                                    "failures": {},
                                    "provider": "",
                                    "model": "",
                                    "task_type": "default",
                                    "tokens": 0,
                                    "run_kind": "",
                                    "details": [],  # v2: per-call details
                                }

                            absorb_session_identity(sessions[sid], data)
                            absorb_run_kind(sessions[sid], event)

                            if etype == "tool_call":
                                # Create a pending detail (reasoning enrichment).
                                # Counting happens on tool_result (the reliably-
                                # emitted event) to avoid double-counting.
                                detail = ToolCallTrace(
                                    tool_name=data.get("tool_name", ""),
                                    arguments_summary=str(data.get("arguments_sanitized", ""))[
                                        :200
                                    ],
                                    reasoning_before=str(data.get("reasoning_before_call", ""))[
                                        :500
                                    ],
                                )
                                sessions[sid]["details"].append(detail)

                            elif etype == "tool_result":
                                success = data.get("success", True)
                                sessions[sid]["tool_calls"] += 1
                                # Fill a pending tool_call detail if one is open;
                                # otherwise build one from the result (the emitter
                                # logs tool_result directly, with no paired tool_call).
                                details = sessions[sid]["details"]
                                if details and not (
                                    getattr(details[-1], "result_summary", "")
                                    or getattr(details[-1], "error_detail", "")
                                ):
                                    last = details[-1]
                                else:
                                    last = ToolCallTrace(
                                        tool_name=data.get("tool_name", ""),
                                        arguments_summary="",
                                        reasoning_before="",
                                    )
                                    details.append(last)
                                last.success = success
                                last.duration_ms = data.get("duration_ms", 0)
                                last.result_summary = str(
                                    data.get("result_summary") or data.get("result") or ""
                                )[:500]
                                last.error_detail = str(
                                    data.get("error_detail") or data.get("error") or ""
                                )[:500]
                                if not last.tool_name and data.get("tool_name"):
                                    last.tool_name = data.get("tool_name", "")

                                if not success:
                                    error = str(
                                        data.get("error_detail")
                                        or data.get("error")
                                        or data.get("result", {}).get("error", "")
                                    )
                                    cat = categorize_failure(error)
                                    sessions[sid]["failures"][cat] = (
                                        sessions[sid]["failures"].get(cat, 0) + 1
                                    )

                            elif etype == "task_classification":
                                sessions[sid]["task_type"] = data.get("task_type", "default")
                        except (JSONDecodeError, KeyError):
                            continue
            except Exception:
                continue

        verdicts = self._harness_verdicts()
        for sid, data in list(sessions.items())[-limit:]:
            if data["tool_calls"] > 0:
                # Was a flat 0.5-if-any-failure / 0.8-otherwise, a third scoring
                # rule alongside v1's and the conversation collector's. All three
                # now go through score_session, so a session is graded the same
                # way regardless of which collector observed it.
                failure_rate = sum(data["failures"].values()) / max(data["tool_calls"], 1)
                completion_score, success, score_source = score_session(
                    verdicts.get(sid), failure_rate
                )
                traces.append(
                    ExecutionTrace(
                        session_id=sid,
                        task_type=data["task_type"],
                        provider=data.get("provider") or "unknown",
                        model=data.get("model") or "unknown",
                        tool_calls=data["tool_calls"],
                        tool_failures=data["failures"],
                        success=success,
                        completion_score=completion_score,
                        tokens_used=data.get("tokens", 0),
                        tool_call_details=data.get("details", []),
                        score_source=score_source,
                        run_kind=data.get("run_kind") or "unknown",
                    )
                )

        return traces

    def collect_from_conversations(self, limit: int = 50) -> List[ExecutionTrace]:
        """Collect execution traces from ConversationStore SQLite DB.

        Converts normalized session+message data into ExecutionTrace
        objects that all prompt optimization strategies can consume.
        This supplements JSONL-based traces with richer historical data
        (provider metadata, model family, message counts, duration).
        """
        traces: List[ExecutionTrace] = []
        try:
            from victor.agent.conversation.store import ConversationStore
            from victor.agent.conversation.types import MessageRole
        except ImportError:
            return traces

        try:
            store = ConversationStore()
        except Exception:
            logger.debug("ConversationStore unavailable for trace collection")
            return traces

        try:
            # Get sessions with enough messages to be meaningful
            sessions = store.get_rl_training_data(limit=limit, min_messages=3)
        except Exception as e:
            logger.debug("Failed to query RL training data: %s", e)
            return traces

        verdicts = self._harness_verdicts()
        for sess in sessions:
            session_id = sess.get("session_id", "")
            provider = sess.get("provider") or "unknown"
            model = sess.get("model") or "unknown"
            tool_msg_count = sess.get("tool_messages") or 0

            # Skip sessions with no tool usage
            if tool_msg_count < 2:
                continue

            # Build tool call details from individual messages
            details: List[ToolCallTrace] = []
            failures: Dict[str, int] = {}
            try:
                session_obj = store.get_session(session_id)
                if session_obj:
                    pending_by_call_id: Dict[str, ToolCallTrace] = {}
                    pending_without_id: List[ToolCallTrace] = []
                    for msg in session_obj.messages:
                        if msg.role == MessageRole.TOOL_CALL:
                            detail = ToolCallTrace(
                                tool_name=msg.tool_name or "",
                                arguments_summary=msg.content[:200],
                                reasoning_before="",
                            )
                            details.append(detail)
                            if msg.tool_call_id:
                                pending_by_call_id[msg.tool_call_id] = detail
                            else:
                                pending_without_id.append(detail)
                        elif msg.role == MessageRole.TOOL:
                            is_error = "error" in msg.content.lower()[:200]
                            matched = None
                            if msg.tool_call_id:
                                matched = pending_by_call_id.pop(msg.tool_call_id, None)
                            if matched is None and pending_without_id:
                                matched = pending_without_id.pop(0)
                            if matched is None:
                                matched = ToolCallTrace(
                                    tool_name=msg.tool_name or "",
                                    arguments_summary="",
                                    reasoning_before="",
                                )
                                details.append(matched)

                            if msg.tool_name and not matched.tool_name:
                                matched.tool_name = msg.tool_name
                            matched.success = not is_error
                            matched.result_summary = msg.content[:500]
                            if is_error:
                                matched.error_detail = msg.content[:500]
                            if is_error:
                                cat = categorize_failure(msg.content[:300])
                                failures[cat] = failures.get(cat, 0) + 1
            except Exception:
                pass  # Fall back to aggregate-only

            total_tool_calls = max(len(details), int(tool_msg_count), 1)
            total_failures = sum(failures.values())
            failure_rate = total_failures / max(total_tool_calls, 1)

            completion_score, success, score_source = score_session(
                verdicts.get(session_id), failure_rate
            )

            traces.append(
                ExecutionTrace(
                    session_id=session_id,
                    task_type="default",
                    provider=provider,
                    model=model,
                    tool_calls=total_tool_calls,
                    tool_failures=failures,
                    success=success,
                    completion_score=completion_score,
                    tokens_used=0,
                    tool_call_details=details,
                    score_source=score_source,
                )
            )

        traces.sort(key=lambda t: -t.completion_score)
        return traces
