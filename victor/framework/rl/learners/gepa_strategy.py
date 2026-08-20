# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Prompt-optimization strategy protocol and the built-in GEPA strategy.

Extracted from ``prompt_optimizer.py`` (which re-exports both names for
backward compatibility) so the strategy contract and the default reflect →
mutate → merge implementation live apart from the learner that drives them.

``GEPAStrategy`` is LLM-driven when a provider is reachable (Ollama by default,
or an injected decision service) and falls back to deterministic heuristics
otherwise, so it never hard-requires a model. Its only module-level
dependencies are the trace data model and the failing-exemplar formatter from
``trace_analysis``; every provider/service import is lazy and in-method, which
keeps this module free of import cycles.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol

from victor.framework.rl.learners.trace_analysis import ExecutionTrace, format_failing_exemplars

logger = logging.getLogger(__name__)


class PromptOptimizationStrategy(Protocol):
    """Strategy interface for prompt evolution approaches."""

    def reflect(
        self,
        traces: List[ExecutionTrace],
        section_name: str,
        current_text: str,
        **kwargs: Any,
    ) -> str:
        """Analyze traces and produce a reflection/diagnosis."""
        ...

    def mutate(self, current_text: str, reflection: str, section_name: str) -> str:
        """Generate mutated prompt section text."""
        ...


class GEPAStrategy:
    """GEPA-inspired: reflect on execution traces, then mutate prompt text.

    Uses LLM for reflection + mutation when available. Falls back to
    heuristic reflection (failure frequency analysis) when LLM unavailable.

    LLM sources (in order of preference):
    1. Explicit llm_service (decision service)
    2. Ollama local model (free, fast — default: qwen3.5:2b)
    3. Heuristic fallback (no LLM needed)
    """

    def __init__(
        self,
        llm_service: Any = None,
        ollama_model: str = "qwen3.5:2b",
        ollama_url: str = "http://localhost:11434",
    ):
        self._llm = llm_service
        self._provider_name = "ollama"
        self._model = ollama_model
        self._provider = None  # Lazy-loaded

    def _get_provider(self) -> Any:
        """Get or create provider via Victor's provider abstraction.

        Uses the configured provider name to instantiate. Defaults to
        ollama (free, local). Set _provider_name to None to disable.
        """
        if self._provider is not None:
            return self._provider
        if not self._provider_name:
            return None
        try:
            from importlib import import_module

            from victor.providers.sandhi_transport import resolve_transport_class

            mod = import_module(f"victor.providers.{self._provider_name}_provider")
            native_cls = getattr(mod, f"{self._provider_name.title()}Provider")
            # Sandhi owns transport: resolve the typed variant so chat()/stream()
            # run via the FFI binding rather than the (removed) raw wire path.
            self._provider = resolve_transport_class(self._provider_name, native_cls, {})()
            return self._provider
        except Exception as e:
            logger.debug("Failed to create %s provider: %s", self._provider_name, e)
            return None

    def _call_llm(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """Call LLM via Victor's provider abstraction (free local or cloud).

        Prepends /no_think for Qwen models to suppress verbose reasoning.
        """
        provider = self._get_provider()
        if provider is None:
            return None
        try:
            from victor.core.async_utils import run_sync_in_thread
            from victor.providers.base import Message

            # Suppress thinking for Qwen models
            effective_prompt = prompt
            if "qwen" in self._model.lower():
                effective_prompt = f"/no_think\n{prompt}"

            messages = [Message(role="user", content=effective_prompt)]
            response = run_sync_in_thread(
                provider.chat(
                    messages=messages,
                    model=self._model,
                    max_tokens=max_tokens,
                    temperature=0.7,
                ),
                timeout=30.0,
            )
            content = response.content if response else ""
            # Strip thinking artifacts (Qwen3, DeepSeek R1)
            import re

            if "<think>" in content:
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            # Strip "Thinking Process:" preamble
            if "Thinking Process:" in content:
                parts = re.split(r"\n(?=TOOL EFFECTIVENESS|[A-Z]{3,}[:\s])", content)
                for part in reversed(parts):
                    if "Thinking Process" not in part and len(part.strip()) > 50:
                        content = part.strip()
                        break
            if content and len(content) > 20:
                return content.strip()
        except Exception as e:
            logger.debug("LLM call via %s failed: %s", self._provider_name, e)
        return None

    def reflect(
        self,
        traces: List[ExecutionTrace],
        section_name: str,
        current_text: str,
        **kwargs: Any,
    ) -> str:
        """Analyze traces and produce natural language reflection."""
        del kwargs
        # Aggregate failure patterns
        total = len(traces)
        successes = sum(1 for t in traces if t.success)
        all_failures: Dict[str, int] = {}
        total_tool_calls = 0
        total_tokens = 0

        for trace in traces:
            total_tool_calls += trace.tool_calls
            total_tokens += trace.tokens_used
            for category, count in trace.tool_failures.items():
                all_failures[category] = all_failures.get(category, 0) + count

        # Build heuristic reflection from failure frequencies
        lines = [
            f"Analysis of {total} execution traces:",
            f"- Success rate: {successes}/{total} ({100*successes/max(total,1):.0f}%)",
            f"- Avg tool calls: {total_tool_calls/max(total,1):.1f}",
            f"- Avg tokens: {total_tokens/max(total,1):.0f}",
        ]
        if all_failures:
            lines.append("- Top failure categories:")
            for cat, count in sorted(all_failures.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  - {cat}: {count}")

        # Enrich with credit assignment data (FEP-0001 Phase 3)
        credit_traces = [t for t in traces if t.credit_signals]
        if credit_traces:
            tool_credits: Dict[str, List[float]] = {}
            for trace in credit_traces:
                for cs in trace.credit_signals:
                    tname = cs.get("tool_name", "unknown")
                    tool_credits.setdefault(tname, []).append(cs.get("credit", 0.0))
            if tool_credits:
                lines.append("- Tool credit attribution:")
                for tool, credits in sorted(
                    tool_credits.items(), key=lambda x: sum(x[1]), reverse=True
                )[:5]:
                    avg = sum(credits) / len(credits)
                    lines.append(f"  - {tool}: avg_credit={avg:+.2f} ({len(credits)} calls)")

        agent_guidance_blocks = []
        for trace in traces:
            guidance = getattr(trace, "agent_guidance", None)
            if guidance and guidance not in agent_guidance_blocks:
                agent_guidance_blocks.append(guidance)
        if agent_guidance_blocks:
            lines.append("- Agent execution credit:")
            lines.extend(agent_guidance_blocks[:2])

        # Counts give the shape of the problem; exemplars give the specifics a
        # rewrite can act on. Without these the model receives only a histogram
        # ("edit_mismatch: 7") and cannot tell what to write differently.
        exemplars = format_failing_exemplars(traces)
        if exemplars:
            lines.append("")
            lines.append(exemplars)

        reflection = "\n".join(lines)

        # Enhance with LLM-driven reflection (provider → decision service → skip)
        # The section is passed in full: this prompt truncated it to 500 chars,
        # so for most evolvable sections the model was asked to improve guidance
        # whose majority it never saw.
        llm_prompt = (
            f"You are analyzing execution traces for an AI coding agent.\n\n"
            f"{reflection}\n\n"
            f"Current prompt section '{section_name}':\n{current_text}\n\n"
            f"What 3 specific, actionable changes to this prompt guidance would "
            f"reduce the failure patterns above? Be concise — bullet points only."
        )

        # Try provider abstraction first (Ollama by default, free + local)
        llm_result = self._call_llm(llm_prompt)
        if llm_result:
            reflection += f"\n\nLLM Reflection ({self._provider_name}/{self._model}):\n{llm_result}"
            return reflection

        # Try decision service if available
        if self._llm is not None:
            try:
                from victor.agent.services.protocols.decision_service import (
                    DecisionType,
                )

                llm_reflection = self._llm.decide_sync(
                    DecisionType.TASK_TYPE_CLASSIFICATION,
                    {
                        "message_excerpt": llm_prompt,
                    },
                )
                if llm_reflection.source != "timeout_fallback":
                    reflection += f"\n\nLLM Reflection:\n{llm_reflection.result}"
            except Exception as exc:
                # Best-effort: the heuristic reflection above already stands on
                # its own, so a failed LLM augmentation must not abort. But
                # swallowing it silently made an offline decision service or a
                # throttled provider look like "the strategy had nothing to
                # add" — log it so the two are distinguishable.
                logger.debug("GEPA LLM reflection augmentation failed (best-effort): %s", exc)

        return reflection

    def mutate(self, current_text: str, reflection: str, section_name: str) -> str:
        """Generate mutated prompt text based on reflection.

        Uses provider abstraction for LLM mutation, falls back to
        heuristic mutations based on failure patterns.
        """
        mutation_prompt = (
            f"Improve this prompt section for an AI coding agent based on "
            f"the execution analysis below.\n\n"
            f"Current '{section_name}':\n{current_text}\n\n"
            f"Reflection on failures:\n{reflection}\n\n"
            f"Generate an improved version. Requirements:\n"
            f"- Keep same length (±20%)\n"
            f"- Be specific and actionable\n"
            f"- Address the failure patterns from the reflection\n"
            f"- Output ONLY the improved prompt text, no explanation\n\n"
            f"Improved version:"
        )

        # Try provider abstraction (Ollama by default)
        llm_result = self._call_llm(mutation_prompt, max_tokens=800)
        if llm_result and len(llm_result) > 50:
            return llm_result

        # Heuristic mutation: append failure-specific guidance
        mutations = []
        if "file_not_found" in reflection.lower():
            mutations.append("- Verify file paths with ls() before reading them.")
        if "edit" in reflection.lower() and "mismatch" in reflection.lower():
            mutations.append("- When editing, read the file first and copy old_str exactly.")
        if "timeout" in reflection.lower():
            mutations.append("- Keep tool calls focused. Avoid redundant reads of the same file.")

        if mutations:
            return current_text + "\n" + "\n".join(mutations)
        return current_text

    def merge(
        self,
        candidate_a: str,
        candidate_b: str,
        section_name: str,
        max_chars: int = 1500,
    ) -> str:
        """Combine complementary prompt variants when Pareto merge is requested."""
        merge_prompt = (
            f"Merge these two prompt variants for '{section_name}'.\n\n"
            f"Candidate A:\n{candidate_a}\n\n"
            f"Candidate B:\n{candidate_b}\n\n"
            f"Preserve the strongest guidance from both, remove duplication, "
            f"and keep the result under {max_chars} characters.\n"
            f"Output only the merged prompt text."
        )
        llm_result = self._call_llm(merge_prompt, max_tokens=800)
        if llm_result:
            return llm_result[:max_chars]

        merged_lines: List[str] = []
        seen = set()
        for line in candidate_a.splitlines() + candidate_b.splitlines():
            normalized = line.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged_lines.append(line)
        merged = "\n".join(merged_lines).strip()
        return merged[:max_chars] if merged else candidate_a[:max_chars]
