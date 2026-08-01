# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Chain-of-Thought transfer strategy for prompt optimization.

This is a prompt-layer adaptation of CoT distillation, not student-model
training. It mines a strong execution trace from a better-performing provider
and converts it into a compact reasoning scaffold that can be injected into a
weaker target-provider prompt.

Faithfulness note: the scaffold is derived from the *actual observed
trajectory* of the source trace — the real ordered sequence of successful tool
calls and the reasoning the model recorded before each — whenever ASI
tool-call detail is present (``tool_call_details``). It only falls back to a
generic discover/read/plan/edit/verify scaffold for older traces captured
before detail collection existed. A previous version emitted that generic
scaffold unconditionally and never read the trace, so "distilled from a
94%-scoring trace" was decoration; this version makes the claim true.
"""

import logging
from typing import Any, Dict, List

from victor.core.completion_markers import TASK_DONE_MARKER

logger = logging.getLogger(__name__)


# Verb by tool family, used to phrase a distilled step. Matched by substring so
# grouped/aliased names ("semantic_code_search", "run_tests") still resolve.
# Ordered: the first needle found in the (lowercased) tool name wins.
_TOOL_VERBS = (
    ("search", "DISCOVER"),
    ("grep", "DISCOVER"),
    ("find", "DISCOVER"),
    ("overview", "DISCOVER"),
    ("graph", "ANALYZE"),
    ("architecture", "ANALYZE"),
    ("read", "READ"),
    ("ls", "READ"),
    ("cat", "READ"),
    ("edit", "EDIT"),
    ("write", "EDIT"),
    ("apply", "EDIT"),
    ("patch", "EDIT"),
    ("test", "VERIFY"),
    ("shell", "RUN"),
    ("bash", "RUN"),
    ("git", "RUN"),
)


def _verb_for_tool(tool_name: str) -> str:
    """Map a tool name to a phase verb for the reasoning scaffold."""
    lowered = tool_name.lower()
    for needle, verb in _TOOL_VERBS:
        if needle in lowered:
            return verb
    return "ACT"


class CoTDistillationStrategy:
    """Distill a reasoning scaffold from a strong trace into prompt guidance.

    Implements the PromptOptimizationStrategy protocol. The scaffold reflects
    the source trace's real tool trajectory (see the module docstring); it is
    not a fixed template.
    """

    def __init__(
        self,
        min_source_score: float = 0.7,
        max_steps: int = 5,
        min_score_gap: float = 0.15,
        llm_service: Any = None,
    ):
        self._min_score = min_source_score
        self._max_steps = max_steps
        self._min_gap = min_score_gap
        self._llm = llm_service

    def reflect(
        self,
        traces: List[Any],
        section_name: str,
        current_text: str,
        **kwargs: Any,
    ) -> str:
        """Extract provider-aware reasoning patterns from strong traces.

        When a target provider is supplied, only distill when another provider
        materially outperforms it. This keeps us from re-injecting guidance
        into the already-best provider and matches the current prompt-transfer
        use case more closely than unconditional trace copying.
        """
        del section_name, current_text
        if not traces:
            return ""

        target_provider = kwargs.get("target_provider") or kwargs.get("provider")

        # Find high-scoring successful traces
        strong = [
            t
            for t in traces
            if t.success
            and t.completion_score >= self._min_score
            and t.tool_calls >= 3  # Need enough steps for a meaningful chain
        ]

        if not strong:
            logger.debug(
                "CoT: No strong traces found (need score >= %.1f, tools >= 3)",
                self._min_score,
            )
            return ""

        if target_provider:
            target_traces = [t for t in strong if getattr(t, "provider", None) == target_provider]
            source_traces = [t for t in strong if getattr(t, "provider", None) != target_provider]

            if not source_traces:
                logger.debug("CoT: no stronger source provider available for %s", target_provider)
                return ""

            best_source = max(source_traces, key=lambda t: t.completion_score)
            best_target_score = max(
                (getattr(t, "completion_score", 0.0) for t in target_traces),
                default=0.0,
            )

            if best_target_score >= getattr(best_source, "completion_score", 0.0) - self._min_gap:
                logger.debug(
                    "CoT: target provider %s already within %.2f of best source",
                    target_provider,
                    self._min_gap,
                )
                return ""

            return self._distill_reasoning(
                best_source,
                source_provider=getattr(best_source, "provider", "unknown"),
                target_provider=target_provider,
            )

        best = max(strong, key=lambda t: t.completion_score)
        return self._distill_reasoning(best)

    def mutate(self, current_text: str, reflection: str, section_name: str) -> str:
        """Append distilled reasoning template to the prompt."""
        if not reflection:
            return current_text
        return f"{current_text}\n\n{reflection}"

    def _distill_reasoning(
        self,
        trace: Any,
        *,
        source_provider: str = "",
        target_provider: str = "",
    ) -> str:
        """Convert a successful trace into a step-by-step reasoning scaffold.

        Prefers the trace's real tool trajectory (``tool_call_details``);
        falls back to a generic scaffold only when a trace predates ASI
        detail capture.
        """
        score = getattr(trace, "completion_score", 0.0)
        failures = getattr(trace, "tool_failures", {}) or {}
        details = getattr(trace, "tool_call_details", None) or []

        if details:
            steps = self._steps_from_trace(details, failures)
            basis = "an observed trace"
        else:
            steps = self._generic_scaffold(trace, failures)
            basis = "the success profile"

        steps = steps[: self._max_steps]
        if not steps:
            return ""

        scope = ""
        if source_provider and target_provider:
            scope = f" from {source_provider} to {target_provider}"
        header = f"STEP-BY-STEP APPROACH{scope} (distilled from {basis}, score {score:.0%}):"
        return f"{header}\n" + "\n".join(steps)

    def _steps_from_trace(
        self,
        details: List[Any],
        failures: Dict[str, int],
    ) -> List[str]:
        """Build a scaffold from the real ordered sequence of successful calls.

        Consecutive calls to the same tool (e.g. several reads in a row) are
        collapsed into one step so the scaffold reads as a plan rather than a
        transcript. A recovery step is appended only for failures the trace
        actually hit — the guidance is grounded in what went wrong, not a
        blanket warning.
        """
        phrases: List[str] = []
        prev_tool = None
        for detail in details:
            tool = (getattr(detail, "tool_name", "") or "").strip()
            if not tool or not getattr(detail, "success", True):
                continue
            if tool == prev_tool:
                continue
            prev_tool = tool
            verb = _verb_for_tool(tool)
            reason = (getattr(detail, "reasoning_before", "") or "").strip()
            if reason:
                clause = reason.split(". ")[0].strip().rstrip(".")
                clause = clause[:120].rstrip()
                phrases.append(f"{verb} with {tool}: {clause}.")
            else:
                phrases.append(f"{verb} with {tool}.")
            if len(phrases) >= self._max_steps:
                break

        if "edit_mismatch" in failures and len(phrases) < self._max_steps:
            phrases.append(
                "If an edit fails to apply, re-read the file and copy old_str "
                "exactly — do not guess the surrounding context."
            )

        return [f"{i}. {phrase}" for i, phrase in enumerate(phrases, 1)]

    def _generic_scaffold(self, trace: Any, failures: Dict[str, int]) -> List[str]:
        """Fallback scaffold for traces captured before ASI detail existed.

        Behaviour-preserving copy of the original template. Used only when a
        trace carries no ``tool_call_details`` to distil from.
        """
        tools = getattr(trace, "tool_calls", 0)
        steps = [
            "1. DISCOVER: Use code_search(query='relevant term') to find the "
            "file(s) related to the issue. Do NOT guess file paths.",
            "2. READ: Read the identified file(s) to understand the current "
            "implementation. Use search= parameter for large files.",
            "3. PLAN: Before editing, identify exactly which lines need to "
            "change and what the fix should be. State your plan.",
        ]
        if failures and "edit_mismatch" in failures:
            steps.append(
                "4. EDIT: Copy old_str EXACTLY from the file (use 3+ lines of "
                "surrounding context). If the edit fails, re-read the file and "
                "adjust the old_str — do NOT guess."
            )
        else:
            steps.append(
                "4. EDIT: Apply the fix using edit() with exact old_str "
                "copied from the file. Include sufficient context for uniqueness."
            )
        if isinstance(tools, int) and tools > 5:
            steps.append(
                "5. VERIFY: Read the modified file to confirm the edit was "
                f"applied correctly. Signal {TASK_DONE_MARKER} when verified."
            )
        return steps
