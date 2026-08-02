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

import logging
from typing import Any, Dict, List, Optional, Tuple

from victor.framework.rl.learners.trace_analysis import ExecutionTrace, HarnessVerdict

logger = logging.getLogger(__name__)


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
