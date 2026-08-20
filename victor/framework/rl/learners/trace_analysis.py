# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Execution-trace data model and pure analysis helpers for prompt evolution.

Extracted from ``prompt_optimizer.py`` (which re-exports every name here for
backward compatibility) so the trace data model and the stateless analysis
functions live apart from the learner. Everything in this module is pure and
depends only on the standard library — no learner, provider, or DB coupling —
which makes it independently testable and safe to import from anywhere in the
RL layer.

Contents:
- Data model: ``ToolCallTrace``, ``ExecutionTrace``, ``HarnessVerdict``.
- Semantic zones (PRIME, arXiv:2604.07645): ``TraceZone``, ``classify_trace_zone``.
- Quality scoring (MemReader, arXiv:2604.07877): ``score_trace_quality``.
- Capability gaps (TRACE, arXiv:2604.05336): ``CapabilityGap``,
  ``FAILURE_TO_CAPABILITY``, ``analyze_capability_gaps``.
- Failure taxonomy (arXiv:2601.08884): ``FAILURE_HINTS``, ``get_failure_hint``.
- Reflection input: ``format_failing_exemplars`` and its ``MAX_EXEMPLAR_*`` caps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Semantic Trace Zones (inspired by arXiv:2604.07645 — PRIME)
# ---------------------------------------------------------------------------


class TraceZone(str, Enum):
    """Semantic zones for GEPA trace organization."""

    SUCCESS = "successful_strategies"
    FAILURE = "failure_patterns"
    RECOVERY = "recovery_patterns"


def classify_trace_zone(trace) -> TraceZone:
    """Classify an execution trace into a semantic zone.

    - RECOVERY: successful despite having tool failures (retry worked)
    - FAILURE: score < 0.5 or not successful
    - SUCCESS: everything else (score >= 0.5, no failures)
    """
    has_failures = bool(getattr(trace, "tool_failures", None))
    is_success = getattr(trace, "success", False)
    score = getattr(trace, "completion_score", 0.0)

    if is_success and has_failures:
        return TraceZone.RECOVERY
    if not is_success or score < 0.5:
        return TraceZone.FAILURE
    return TraceZone.SUCCESS


# ---------------------------------------------------------------------------
# Trace Quality Scoring (inspired by arXiv:2604.07877 — MemReader)
# ---------------------------------------------------------------------------

TRACE_QUALITY_THRESHOLD = 0.3


def score_trace_quality(trace) -> float:
    """Score trace quality for GEPA reflection value (MemReader-inspired).

    Returns 0.0-1.0. Traces below TRACE_QUALITY_THRESHOLD are noise.
    Criteria: substance, completeness, richness, coherence.
    """
    score = 0.0

    tool_calls = getattr(trace, "tool_calls", 0)
    if isinstance(tool_calls, int):
        if tool_calls >= 5:
            score += 0.3
        elif tool_calls >= 3:
            score += 0.2
        elif tool_calls >= 1:
            score += 0.1

    details = getattr(trace, "tool_call_details", [])
    if details:
        populated = sum(
            1 for d in details if getattr(d, "result_summary", "") or getattr(d, "error_detail", "")
        )
        completeness = populated / max(len(details), 1)
        score += 0.3 * completeness

    reasoning_count = sum(1 for d in details if getattr(d, "reasoning_before", ""))
    if details and reasoning_count / max(len(details), 1) > 0.5:
        score += 0.2

    failures = getattr(trace, "tool_failures", {})
    if failures:
        categorized = sum(v for k, v in failures.items() if k != "other")
        total = sum(failures.values())
        if total > 0 and categorized / total > 0.5:
            score += 0.2
        else:
            score += 0.1
    elif getattr(trace, "success", False):
        score += 0.15

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Capability Gap Analysis (inspired by arXiv:2604.05336 — TRACE)
# ---------------------------------------------------------------------------


@dataclass
class CapabilityGap:
    """A specific capability deficiency identified from trace contrast."""

    capability: str
    failure_rate: float
    failure_count: int
    example_errors: List[str]


FAILURE_TO_CAPABILITY: Dict[str, str] = {
    "edit_mismatch": "edit_precision",
    "edit_ambiguous": "edit_precision",
    "edit_syntax": "edit_precision",
    "file_not_found": "path_resolution",
    "read_directory": "path_resolution",
    "permission_denied": "path_resolution",
    "search_no_results": "search_strategy",
    "tool_not_found": "tool_knowledge",
    "tool_error": "tool_knowledge",
    "timeout": "execution_efficiency",
    "shell_error": "execution_efficiency",
    "test_failure": "verification",
    "verbosity": "conciseness",
    "other": "other",
}


def analyze_capability_gaps(traces) -> List[CapabilityGap]:
    """Contrast success vs failure zones to find dominant gaps (TRACE-inspired)."""
    capability_failures: Dict[str, int] = {}
    capability_errors: Dict[str, List[str]] = {}
    total_failures = 0

    for trace in traces:
        zone = classify_trace_zone(trace)
        if zone != TraceZone.FAILURE:
            continue
        for cat, count in getattr(trace, "tool_failures", {}).items():
            capability = FAILURE_TO_CAPABILITY.get(cat, "other")
            capability_failures[capability] = capability_failures.get(capability, 0) + count
            total_failures += count
            for detail in getattr(trace, "tool_call_details", []):
                if not getattr(detail, "success", True) and getattr(detail, "error_detail", ""):
                    errors = capability_errors.setdefault(capability, [])
                    if len(errors) < 3:
                        errors.append(getattr(detail, "error_detail", "")[:200])

    if not total_failures:
        return []

    gaps = []
    for cap, count in sorted(capability_failures.items(), key=lambda x: -x[1]):
        gaps.append(
            CapabilityGap(
                capability=cap,
                failure_rate=count / total_failures,
                failure_count=count,
                example_errors=capability_errors.get(cap, []),
            )
        )
    return gaps[:3]


# ---------------------------------------------------------------------------
# Structured Failure Taxonomy (inspired by arXiv:2601.08884)
# ---------------------------------------------------------------------------
# Each failure category maps to a corrective "Prompt Hint" that feeds into
# GEPA's reflection step, giving the mutation LLM actionable guidance
# instead of raw category names. Add new categories by adding entries.

FAILURE_HINTS: Dict[str, str] = {
    "file_not_found": (
        "Verify file paths with ls() before reading. "
        "Use code_search to find files by name or content."
    ),
    "read_directory": (
        "Use ls() for directories, read() only for files. "
        "Check the path is a file, not a directory."
    ),
    "permission_denied": (
        "Check file permissions. Avoid writing to read-only paths. "
        "Use a working directory the agent has write access to."
    ),
    "edit_mismatch": (
        "Read the complete file before editing. Copy old_str exactly from "
        "tool output — character for character, including whitespace and indentation."
    ),
    "edit_ambiguous": (
        "Include 3+ surrounding context lines in old_str to make the match unique. "
        "If the string appears multiple times, add distinguishing context."
    ),
    "edit_syntax": (
        "Validate that new_str preserves correct syntax. Check indentation matches "
        "the surrounding code. Run a linter after editing if available."
    ),
    "tool_not_found": (
        "Use only tools listed in the available tools. Check tool name spelling. "
        "Use ls or code_search as universal fallbacks."
    ),
    "timeout": (
        "Keep tool calls focused. Avoid reading entire large directories. "
        "Use targeted searches instead of broad scans. Limit shell command duration."
    ),
    "tool_error": (
        "Check tool arguments match the expected schema. Review the error message "
        "and adjust arguments before retrying."
    ),
    "search_no_results": (
        "Broaden the search query. Try alternative keywords, partial names, "
        "or regex patterns. Fall back to ls() + grep for manual search."
    ),
    "shell_error": (
        "Check command syntax. Ensure required tools (git, npm, etc.) are "
        "installed. Use absolute paths for reliability."
    ),
    "test_failure": (
        "Read the test output carefully. Identify which assertion failed and why. "
        "Fix the root cause, not the symptom."
    ),
    "verbosity": (
        "Keep responses concise and direct. Avoid unnecessary preamble, summaries, "
        "or explanations. Skip 'I'll' and 'Let me' phrases. For code: show the code "
        "with minimal commentary. For questions: answer directly, then stop."
    ),
    "other": (
        "Read the error message carefully. Diagnose the root cause before "
        "retrying. Avoid repeating the same failing operation."
    ),
}


def get_failure_hint(category: str) -> str:
    """Get the corrective prompt hint for a failure category."""
    return FAILURE_HINTS.get(category, "")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ToolCallTrace:
    """Individual tool call within a session (ASI detail)."""

    tool_name: str
    arguments_summary: str = ""
    reasoning_before: str = ""
    success: bool = True
    result_summary: str = ""
    error_detail: str = ""
    duration_ms: float = 0.0


@dataclass
class ExecutionTrace:
    """Summary of one agent session's execution for prompt evolution.

    When GEPA v2 trace enrichment is enabled, tool_call_details contains
    per-call ASI data. Otherwise, tool_calls is an int count (v1 compat).
    """

    session_id: str
    task_type: str
    provider: str
    model: str
    tool_calls: int
    tool_failures: Dict[str, int]  # category → count
    success: bool
    completion_score: float
    tokens_used: int
    # Per-section token counts for efficiency tracking
    section_tokens: Dict[str, int] = field(default_factory=dict)
    # GEPA v2: detailed per-call traces (ASI)
    tool_call_details: List["ToolCallTrace"] = field(default_factory=list)
    # Credit assignment signals (FEP-0001 Phase 3)
    # Per-tool credit values from CreditTrackingService
    credit_signals: List[Dict[str, Any]] = field(default_factory=list)
    # Optional agent-level summary for multi-agent team runs
    agent_guidance: Optional[str] = None
    # Where completion_score came from: "harness" (a benchmark graded this
    # session) or "tool_failure_proxy" (no verdict exists — interactive work).
    # Kept on the trace so reflection and audits can tell evidence from inference.
    score_source: str = "tool_failure_proxy"
    # What kind of run produced this session — read from the event's own
    # ``run_kind`` tag rather than inferred from prompt text, which conflated
    # delegate workers with benchmark runs.
    run_kind: str = "unknown"


@dataclass(frozen=True)
class HarnessVerdict:
    """Ground-truth outcome for one evaluated session."""

    completion_score: float
    success: bool
    task_id: str
    benchmark: str


# Shaping is done by the structural caps below, not by the character budget: at
# most 3 traces x 4 calls, each call bounded by its own field caps (120 args /
# 160 intent / 240 error) to roughly 570 chars — so a full exemplar set lands
# near 6.8k. The character budget is deliberately set above that as a safety
# valve for pathological error blobs, rather than a second limiter that silently
# drops whole traces from a well-formed set. Context is not the constraint: the
# default reflect tier is gpt-4.1-mini, and 8k of exemplars plus the largest
# section (2934 chars) is roughly 3k tokens.
MAX_EXEMPLAR_CHARS = 8192
MAX_EXEMPLAR_TRACES = 3
MAX_EXEMPLAR_CALLS_PER_TRACE = 4


def format_failing_exemplars(
    traces: List["ExecutionTrace"],
    *,
    max_traces: int = MAX_EXEMPLAR_TRACES,
    max_chars: int = MAX_EXEMPLAR_CHARS,
) -> str:
    """Render concrete failing tool calls for the reflection prompt.

    Reflection used to receive only aggregate counts — "edit_mismatch: 7" — from
    which no rewrite can be derived: the model is told a category is frequent but
    never what the agent actually did or what the tool said back. The detail has
    been collected all along on ``ToolCallTrace`` (``error_detail``,
    ``arguments_summary``, ``reasoning_before``, populated by
    ``_collect_traces_v2``) and simply never reached the prompt.

    Worst traces first, so the budget is spent on the most informative failures.
    Returns "" when nothing failed, so a clean run adds no noise.
    """
    failing = [
        trace
        for trace in traces
        if any(not call.success for call in (trace.tool_call_details or []))
    ]
    if not failing:
        return ""
    failing.sort(key=lambda t: (t.completion_score, -len(t.tool_call_details)))

    blocks: List[str] = ["- Failing exemplars (what actually went wrong):"]
    budget = max_chars
    for trace in failing[:max_traces]:
        bad_calls = [c for c in trace.tool_call_details if not c.success]
        if not bad_calls:
            continue
        lines = [f"  session {trace.session_id[:12]} (score {trace.completion_score:.2f}):"]
        for call in bad_calls[:MAX_EXEMPLAR_CALLS_PER_TRACE]:
            lines.append(f"    - {call.tool_name}({call.arguments_summary[:120]})")
            if call.reasoning_before:
                lines.append(f"      intent: {call.reasoning_before[:160]}")
            if call.error_detail:
                lines.append(f"      error:  {call.error_detail[:240]}")
        block = "\n".join(lines)
        if len(block) > budget:
            break
        blocks.append(block)
        budget -= len(block)
    return "\n".join(blocks) if len(blocks) > 1 else ""
