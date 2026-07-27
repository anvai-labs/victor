"""Adapts the async GEPAService to the sync PromptOptimizationStrategy protocol.

The PromptOptimizerLearner expects a strategy with sync reflect() and mutate()
methods. GEPAService provides these as sync methods already (using
run_sync_in_thread internally), so this adapter primarily:

1. Converts List[ExecutionTrace] → rich ASI text summary
2. Delegates to GEPATierManager.get_service() for the current tier
3. Tracks evolution deltas for auto-tier switching
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from victor.framework.tool_naming import ToolNames, get_canonical_name
from victor.framework.rl.gepa_tier_manager import GEPATierManager
from victor.tools.core_tool_aliases import canonicalize_core_tool_name

logger = logging.getLogger(__name__)

# Tool name → category mapping for per-tool-type GEPA evolution
TOOL_CATEGORY_MAP: Dict[str, str] = {
    ToolNames.READ: "exploration",
    ToolNames.LS: "exploration",
    "code_search": "exploration",
    "semantic_code_search": "exploration",
    "overview": "exploration",
    "graph": "analysis",
    "architecture_summary": "analysis",
    ToolNames.EDIT: "mutation",
    ToolNames.WRITE: "mutation",
    ToolNames.SHELL: "execution",
    ToolNames.GIT: "execution",
    "run_tests": "execution",
}


def categorize_tool(tool_name: str) -> str:
    """Map a tool name to its GEPA category."""
    canonical_tool_name = get_canonical_name(canonicalize_core_tool_name(tool_name))
    return TOOL_CATEGORY_MAP.get(canonical_tool_name, "general")


class GEPAServiceStrategy:
    """Adapts GEPAService to the PromptOptimizationStrategy protocol.

    This is the strategy passed to PromptOptimizerLearner when
    USE_GEPA_V2 is enabled.
    """

    def __init__(self, tier_manager: GEPATierManager):
        self._tier_manager = tier_manager

    def set_main_model_spec(self, spec: Any) -> None:
        """Update the active model spec and invalidate cached services."""
        self._tier_manager.set_main_model_spec(spec)

    def set_mutator_rotation(self, rotation: Any) -> None:
        """Supply providers to fail over to when the active mutator is throttled."""
        self._tier_manager.set_mutator_rotation(rotation)

    def reflect(
        self,
        traces: List[Any],
        section_name: str,
        current_text: str,
        **kwargs: Any,
    ) -> str:
        """Convert traces to ASI summary and call GEPA reflection."""
        del kwargs
        from victor.framework.rl.learners.prompt_optimizer import (
            analyze_capability_gaps,
        )

        service = self._tier_manager.get_service()
        traces_summary = self._format_traces_as_asi(traces)
        heuristic = self._build_heuristic_summary(traces)

        # TRACE-inspired: prepend capability gap analysis for focused reflection
        gaps = analyze_capability_gaps(traces)
        gap_report = self._format_gap_report(gaps)

        # MIPROv2-inspired: data distribution profile for data-aware reflection
        data_profile = self._build_data_profile(traces)

        full_summary = f"{data_profile}\n{gap_report}\n{heuristic}\n\n{traces_summary}"
        return service.reflect(full_summary, section_name, current_text)

    @staticmethod
    def _build_data_profile(traces: List[Any]) -> str:
        """Data distribution profile for data-aware reflection (MIPROv2-inspired)."""
        if not traces:
            return ""
        from collections import Counter

        total = len(traces)
        task_types = Counter(getattr(t, "task_type", "default") for t in traces)
        providers = Counter(getattr(t, "provider", "unknown") for t in traces)
        scores = [getattr(t, "completion_score", 0.0) for t in traces]
        avg_score = sum(scores) / max(len(scores), 1)
        tool_counts = [getattr(t, "tool_calls", 0) for t in traces]
        avg_tools = sum(tc for tc in tool_counts if isinstance(tc, (int, float))) / max(
            len(tool_counts), 1
        )
        all_tools: Counter = Counter()
        for t in traces:
            for d in getattr(t, "tool_call_details", []):
                name = getattr(d, "tool_name", "")
                if name:
                    canonical_tool_name = get_canonical_name(canonicalize_core_tool_name(name))
                    all_tools[canonical_tool_name] += 1
        top_tools = all_tools.most_common(5)
        lines = [
            "=== DATA DISTRIBUTION PROFILE ===",
            f"  Total sessions: {total}",
            f"  Avg completion score: {avg_score:.2f}",
            f"  Avg tool calls/session: {avg_tools:.1f}",
            f"  Task types: {dict(task_types.most_common(3))}",
            f"  Providers: {dict(providers.most_common(3))}",
            f"  Top tools: {', '.join(f'{t}({c})' for t, c in top_tools)}",
            "Optimize the prompt for THIS distribution of usage.\n",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_gap_report(gaps) -> str:
        """Format capability gaps as structured report for GEPA reflection."""
        if not gaps:
            return ""
        lines = ["=== CAPABILITY GAP ANALYSIS (TRACE-inspired) ==="]
        for i, gap in enumerate(gaps, 1):
            pct = int(gap.failure_rate * 100)
            lines.append(f"  #{i} {gap.capability}: {gap.failure_count} failures ({pct}% of total)")
            for err in gap.example_errors[:2]:
                lines.append(f"      Example: {err}")
        lines.append("Focus reflection on the PRIMARY gap above.\n")
        return "\n".join(lines)

    def mutate(
        self,
        current_text: str,
        reflection: str,
        section_name: str,
    ) -> str:
        """Call GEPA mutation via current tier service."""
        service = self._tier_manager.get_service()
        return service.mutate(current_text, reflection, section_name)

    def merge(
        self,
        candidate_a: str,
        candidate_b: str,
        section_name: str,
        max_chars: int = 1500,
    ) -> str:
        """Call GEPA merge via the current tier service."""
        service = self._tier_manager.get_service()
        return service.merge(candidate_a, candidate_b, section_name, max_chars=max_chars)

    @staticmethod
    def _format_single_trace(trace) -> List[str]:
        """Format a single execution trace into ASI text lines."""
        lines: List[str] = []
        sid = getattr(trace, "session_id", "?")
        header = (
            f"Session {str(sid)[:8]} "
            f"({getattr(trace, 'task_type', '?')}, "
            f"{getattr(trace, 'provider', '?')}/{getattr(trace, 'model', '?')}):"
        )
        lines.append(header)

        tool_calls = getattr(trace, "tool_call_details", None)
        if tool_calls and isinstance(tool_calls, list):
            for i, tc in enumerate(tool_calls[:10], 1):
                tool_name = getattr(tc, "tool_name", "?")
                success = getattr(tc, "success", True)
                duration = getattr(tc, "duration_ms", 0)
                status = "success" if success else "FAILED"
                args_summary = getattr(tc, "arguments_summary", "")

                line = f"  [{i}] {tool_name}({args_summary}) → {status}"
                if duration:
                    line += f" ({duration:.0f}ms)"
                lines.append(line)

                reasoning = getattr(tc, "reasoning_before", "")
                if reasoning:
                    lines.append(f'      Reasoning: "{reasoning[:200]}"')

                error = getattr(tc, "error_detail", "")
                if error and not success:
                    lines.append(f"      Error: {error[:300]}")

                result_summary = getattr(tc, "result_summary", "")
                if result_summary and success:
                    lines.append(f"      Result: {result_summary[:200]}")
        else:
            tc_count = getattr(trace, "tool_calls", 0)
            if isinstance(tc_count, int):
                lines.append(f"  Tool calls: {tc_count}")
            failures = getattr(trace, "tool_failures", {})
            if failures:
                for cat, count in failures.items():
                    lines.append(f"  Failures: {cat} x{count}")

        outcome = "SUCCESS" if trace.success else "FAILED"
        score = getattr(trace, "completion_score", 0)
        lines.append(f"  Outcome: {outcome} (score={score:.2f})")
        lines.append("")
        return lines

    @staticmethod
    def _format_traces_as_asi(traces: List[Any]) -> str:
        """Convert ExecutionTrace list to ASI text with semantic zone grouping.

        Organizes traces into 3 zones (PRIME-inspired) before formatting:
        - SUCCESSFUL STRATEGIES: high-scoring, no failures
        - FAILURE PATTERNS: low-scoring or failed
        - RECOVERY PATTERNS: succeeded despite tool failures
        """
        from victor.framework.rl.learners.prompt_optimizer import (
            TraceZone,
            classify_trace_zone,
        )

        zones: Dict[str, List[Any]] = {z.value: [] for z in TraceZone}
        for trace in traces[:20]:
            zone = classify_trace_zone(trace)
            zones[zone.value].append(trace)

        zone_labels = {
            TraceZone.SUCCESS.value: "SUCCESSFUL STRATEGIES",
            TraceZone.FAILURE.value: "FAILURE PATTERNS",
            TraceZone.RECOVERY.value: "RECOVERY PATTERNS",
        }

        parts: List[str] = []
        for zone in TraceZone:
            zone_traces = zones[zone.value]
            if not zone_traces:
                continue
            label = zone_labels[zone.value]
            parts.append(f"\n=== {label} ({len(zone_traces)} sessions) ===")
            for trace in zone_traces:
                parts.extend(GEPAServiceStrategy._format_single_trace(trace))

        return "\n".join(parts) if parts else "No trace data available."

    @staticmethod
    def _build_heuristic_summary(traces: List[Any]) -> str:
        """Build aggregated stats header (same as old GEPAStrategy.reflect)."""
        total = len(traces)
        successes = sum(1 for t in traces if t.success)
        all_failures: Dict[str, int] = {}
        total_tool_calls = 0
        total_tokens = 0

        for trace in traces:
            tc = getattr(trace, "tool_calls", 0)
            if isinstance(tc, int):
                total_tool_calls += tc
            elif isinstance(tc, list):
                total_tool_calls += len(tc)
            total_tokens += getattr(trace, "tokens_used", 0)
            failures = getattr(trace, "tool_failures", {})
            if isinstance(failures, dict):
                for category, count in failures.items():
                    all_failures[category] = all_failures.get(category, 0) + count

        lines = [
            f"Aggregate ({total} sessions):",
            f"- Success rate: {successes}/{total} ({100*successes/max(total,1):.0f}%)",
            f"- Avg tool calls: {total_tool_calls/max(total,1):.1f}",
            f"- Avg tokens: {total_tokens/max(total,1):.0f}",
        ]
        if all_failures:
            from victor.framework.rl.learners.prompt_optimizer import FAILURE_HINTS

            lines.append("- Top failures:")
            for cat, count in sorted(all_failures.items(), key=lambda x: -x[1])[:5]:
                hint = FAILURE_HINTS.get(cat, "")
                if hint:
                    lines.append(f"  {cat}: {count} → Hint: {hint}")
                else:
                    lines.append(f"  {cat}: {count}")
        return "\n".join(lines)
