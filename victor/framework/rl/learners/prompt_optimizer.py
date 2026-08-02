# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""GEPA-Inspired Prompt Optimizer.

Evolves system prompt sections using execution trace analysis and
LLM-driven reflection, following the GEPA methodology (ICLR 2026):
  1. Collect execution traces (tool calls, failures, outcomes)
  2. Reflect on failure patterns using LLM
  3. Mutate prompt sections based on reflection
  4. Select best candidates via Thompson Sampling

Uses strategy pattern — GEPAStrategy is the default, but can be
swapped for alternatives (random mutation, manual, etc.).

Current implementation notes:
- Candidates are provider-scoped and persist full layered strategy-chain metadata.
- Pareto frontiers are tracked per `(section, provider)` key.
- Pareto instance scores come from real runtime outcomes or evaluation artifacts that
  name a specific prompt candidate hash; aggregate section scores are not reused as
  fake per-instance evidence.
- Candidate-bound benchmark suites can be synced back into prompt candidates as
  benchmark evidence; by default only the suite winner can satisfy benchmark gating.
- When reflection/mutation yields no novel text, Pareto merge can synthesize a new
  candidate from complementary frontier members.
- Credit enrichment is session-aligned when runtime credit metadata is available.

Usage:
    from victor.framework.rl.learners.prompt_optimizer import (
        PromptOptimizerLearner,
    )

    learner = PromptOptimizerLearner("prompt_optimizer", db)
    candidate = learner.evolve("ASI_TOOL_EFFECTIVENESS_GUIDANCE", current_text)
    recommendation = learner.get_recommendation("ollama", "qwen3", "action",
                                                 section_name="ASI_TOOL_EFFECTIVENESS_GUIDANCE")
"""

import gzip
import hashlib
import json
from victor.core.json_utils import json_dumps, json_loads
from json import JSONDecodeError
import logging
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation

logger = logging.getLogger(__name__)

# Trace data model and the pure trace-analysis helpers were extracted to
# trace_analysis.py. They are re-exported here so existing imports of the
# form ``from ...prompt_optimizer import ExecutionTrace`` keep working.
from victor.framework.rl.learners.trace_analysis import (
    FAILURE_HINTS,
    FAILURE_TO_CAPABILITY,
    MAX_EXEMPLAR_CALLS_PER_TRACE,
    MAX_EXEMPLAR_CHARS,
    MAX_EXEMPLAR_TRACES,
    TRACE_QUALITY_THRESHOLD,
    CapabilityGap,
    ExecutionTrace,
    HarnessVerdict,
    ToolCallTrace,
    TraceZone,
    analyze_capability_gaps,
    classify_trace_zone,
    format_failing_exemplars,
    get_failure_hint,
    score_trace_quality,
)

# The prompt-candidate data model was extracted to candidate.py. Re-exported
# here so existing ``from ...prompt_optimizer import PromptCandidate`` works.
from victor.framework.rl.learners.candidate import (
    PromptCandidate,
    PromptCandidateBenchmarkDecision,
    PromptCandidateBenchmarkSyncResult,
)


@dataclass
class Objective:
    """An optimization objective with weight."""

    name: str
    weight: float
    direction: str = "maximize"


# ---------------------------------------------------------------------------
# Strategy Protocol + GEPA Default
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Core Learner
# ---------------------------------------------------------------------------

# Minimum traces required before evolution
MIN_TRACES_FOR_EVOLUTION = 5

# Share of an arm's tasks that may be rate-limited before the arm counts as
# aborted rather than measured. Set low: a handful of throttled tasks already
# drop out of the paired contrast, so this only has to catch the case where a
# provider quit partway and the remainder is not a sample of anything.
MAX_THROTTLED_TASK_SHARE = 0.25
# Evaluation artifacts number in the thousands while trace collection only
# looks at the last ~50 sessions, so the verdict scan is bounded by mtime.
MAX_EVAL_ARTIFACTS_SCANNED = 400
MIN_SAMPLES_FOR_CONFIDENCE = 3

# Confidence floor + non-baseline requirement that historically gated live
# serving of an evolved candidate. Exposed as a constant so the serve decision
# and its tests share one source of truth.
SERVE_CONFIDENCE_FLOOR = 0.6
# Pending-benchmark candidates explore at a fraction of the normal rate: they
# must be able to earn evidence, but a candidate no suite has validated should
# reach live traffic more rarely than one that merely lacks samples.
PENDING_BENCHMARK_EXPLORATION_FACTOR = 0.5


def should_serve_candidate(
    rec: Optional["RLRecommendation"],
    *,
    exploration_enabled: bool,
    exploration_epsilon: float,
    rng: Optional[random.Random] = None,
) -> bool:
    """Decide whether to serve an evolved recommendation this turn.

    Serve when the candidate is:
      - **proven** (live confidence above the floor with enough samples), or
      - **benchmark-approved** (validated by a suite, even at sample_count=0), or
      - **explored** — with probability ``exploration_epsilon`` when exploration
        is enabled, so a fresh candidate can bootstrap its Thompson posterior.

    The exploration path is bounded upstream by the persist structural gate
    (growth_exceeded/repeated_trigrams), so explored candidates are never
    corrupt — at worst mediocre, which the reward signal then corrects.
    """
    if rec is None:
        return False
    metadata = getattr(rec, "metadata", None) or {}
    approved = bool(metadata.get("benchmark_passed"))
    pending_benchmark = bool(metadata.get("requires_benchmark")) and not approved

    if not pending_benchmark:
        proven = rec.confidence > SERVE_CONFIDENCE_FLOOR and not rec.is_baseline
        if proven or approved:
            return True

    # A candidate still awaiting its benchmark is reachable *only* here. It is
    # never trusted on merit — no proven path, no approved path — but it can
    # accumulate the live evidence that a benchmark would otherwise have to
    # supply. Without this the gate has no exit and the candidate is inert
    # forever; see _servable_candidates.
    epsilon = exploration_epsilon
    if pending_benchmark:
        epsilon *= PENDING_BENCHMARK_EXPLORATION_FACTOR
    if exploration_enabled and epsilon > 0.0:
        draw = rng if rng is not None else _SERVE_RNG
        return draw.random() < epsilon
    return False


_SERVE_RNG = random.Random()

_DEFAULT_EVOLVABLE_SECTIONS = [
    "ASI_TOOL_EFFECTIVENESS_GUIDANCE",
    "GROUNDING_RULES",
    "COMPLETION_GUIDANCE",
    "CONCISE_MODE_GUIDANCE",
    "PARALLEL_READ_GUIDANCE",
    "LARGE_FILE_PAGINATION_GUIDANCE",
    "GROUNDING_RULES_EXTENDED",
    "FEW_SHOT_EXAMPLES",
    "INIT_SYNTHESIS_RULES",  # Only the RULES section, frame stays fixed
]


def get_registered_evolvable_sections() -> List[str]:
    """Return evolvable prompt sections from the shared registry in priority order."""
    try:
        from victor.agent.prompt_section_registry import get_section_registry

        registry = get_section_registry()
        sections = sorted(
            (section for section in registry.get_all() if section.evolvable),
            key=lambda section: (section.priority, section.name),
        )
        if sections:
            return [section.name for section in sections]
    except Exception:
        logger.debug("Falling back to default evolvable prompt sections", exc_info=True)
    return list(_DEFAULT_EVOLVABLE_SECTIONS)


class PromptOptimizerLearner(BaseLearner):
    """Evolves system prompt sections using GEPA-inspired trace analysis.

    Registered as 'prompt_optimizer' in the RL coordinator. Opt-in only:
    prompt evolution must be triggered explicitly via evolve(), and
    candidates are only used when confidence exceeds threshold.
    """

    EVOLVABLE_SECTIONS = list(_DEFAULT_EVOLVABLE_SECTIONS)

    DEFAULT_OBJECTIVES = [
        Objective("completion_score", weight=0.5),
        Objective("tool_effectiveness", weight=0.3),
        Objective("token_efficiency", weight=0.2),
    ]

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = 0.1,
        provider_adapter: Any = None,
        strategy: Optional[PromptOptimizationStrategy] = None,
        use_pareto: bool = False,
        max_prompt_chars: int = 1500,
    ):
        self._strategy: PromptOptimizationStrategy = strategy or GEPAStrategy()
        self._extra_strategies: Dict[str, List["PromptOptimizationStrategy"]] = {}
        # Section-specific strategy overrides (e.g., FEW_SHOT_EXAMPLES → MIPROv2)
        self._candidates: Dict[str, List[PromptCandidate]] = {}
        # Last candidate actually injected into the live prompt per
        # (section, provider) — lets record_outcome attribute an outcome to a
        # brand-new candidate (is_active=False, sample_count=0) so it can earn
        # its first reward. Set by record_served() at injection time.
        self._last_served: Dict[str, str] = {}
        # Lazily loaded session_id -> ground-truth outcome from benchmark runs.
        self._harness_verdict_cache: Optional[Dict[str, HarnessVerdict]] = None
        self._use_pareto = use_pareto
        self._max_prompt_chars = max_prompt_chars
        self._pareto_frontiers: Dict[str, Any] = {}  # section → ParetoFrontier
        super().__init__(name, db_connection, learning_rate, provider_adapter)
        self._load_candidates()
        if self._use_pareto:
            self._init_pareto_frontiers()
        self._init_section_strategies()

    @classmethod
    def get_evolvable_sections(cls) -> List[str]:
        """Return the current evolvable section list.

        The shared prompt section registry is the source of truth. The legacy
        ``EVOLVABLE_SECTIONS`` attribute is still honored when tests or callers
        intentionally override it with a custom list.
        """
        configured_sections = getattr(cls, "EVOLVABLE_SECTIONS", None)
        if (
            isinstance(configured_sections, list)
            and configured_sections != _DEFAULT_EVOLVABLE_SECTIONS
        ):
            return list(configured_sections)
        return get_registered_evolvable_sections()

    @staticmethod
    def _load_prompt_optimization_settings() -> Any:
        """Load prompt-optimization settings, tolerating bootstrap-time failures."""
        try:
            from victor.config.settings import get_settings

            settings = get_settings()
            return getattr(settings, "prompt_optimization", None)
        except Exception:
            return None

    def _init_section_strategies(self) -> None:
        """Initialize section-specific strategies from config."""
        try:
            from victor.config.prompt_optimization_settings import (
                PromptOptimizationSettings,
            )
            from victor.framework.rl.learners.strategy_registry import (
                build_prompt_strategy,
            )

            po_settings = self._load_prompt_optimization_settings()
            if po_settings is None:
                po_settings = PromptOptimizationSettings(enabled=True)

            self._extra_strategies = {}
            for section_name in self.get_evolvable_sections():
                strategy_names = po_settings.get_strategies_for_section(section_name)
                strategies = []
                for strategy_name in strategy_names:
                    strategy = build_prompt_strategy(
                        strategy_name,
                        settings=po_settings,
                        gepa_strategy=self._strategy,
                    )
                    if strategy is not None:
                        strategies.append(strategy)
                self._extra_strategies[section_name] = strategies

            logger.debug(
                "Section strategies initialized: %s",
                {k: [type(s).__name__ for s in v] for k, v in self._extra_strategies.items()},
            )
        except ImportError:
            logger.debug("Strategy imports failed, using default GEPA only")

    def set_main_model_spec(
        self,
        provider: str,
        model: str,
        # Matches GEPAModelSpec's default: a reasoning model rewriting a
        # 1500-char section runs one to three minutes.
        timeout_s: float = 180.0,
        base_url: str = "",
    ) -> None:
        """Push the active session's provider/model into the GEPA tier manager.

        Called by the /prompt-optimize command so reflections and mutations use
        the same LLM the user is actively chatting with, not the settings default.
        Includes base_url so provider-specific endpoints (e.g. ZAI coding plan) are preserved.
        """
        if hasattr(self._strategy, "set_main_model_spec"):
            from victor.config.gepa_settings import GEPAModelSpec

            spec = GEPAModelSpec(
                provider=provider, model=model, timeout_s=timeout_s, base_url=base_url
            )
            self._strategy.set_main_model_spec(spec)
            logger.info(
                "GEPA main model updated: %s/%s endpoint=%s (timeout=%.0fs)",
                provider,
                model,
                base_url or "default",
                timeout_s,
            )

    def set_mutator_rotation(self, rotation: Any) -> None:
        """Give GEPA somewhere to go when the active mutator returns a 429.

        Without this the throttle is invisible above the transport: `_call_llm`
        logs and returns None, `mutate()` hands back its input, and the caller
        cannot tell "no improvement found" from "never asked".
        """
        if hasattr(self._strategy, "set_mutator_rotation"):
            self._strategy.set_mutator_rotation(rotation)
            logger.info("GEPA mutator failover armed: %s", rotation.summary())

    def _strategies_for_section(self, section_name: str) -> List["PromptOptimizationStrategy"]:
        """Return the configured strategy chain for a section."""
        if section_name in self._extra_strategies:
            return list(self._extra_strategies[section_name])
        return [self._strategy]

    def _trace_cache_ttl_seconds(self) -> float:
        """TTL (seconds) for caching the learning-trace merge; 0 disables (memoized read)."""
        ttl = getattr(self, "_traces_cache_ttl_cached", None)
        if ttl is None:
            try:
                from victor.config.settings import load_settings

                ttl = float(
                    getattr(
                        load_settings().prompt_optimization,
                        "cache_traces_ttl_seconds",
                        0.0,
                    )
                )
            except Exception:  # settings unavailable -> disabled (current behavior)
                ttl = 0.0
            self._traces_cache_ttl_cached = ttl
        return ttl

    def _collect_learning_traces(self, limit: int = 50) -> List[ExecutionTrace]:
        """Collect and merge traces for prompt optimization.

        Within a task the JSONL + conversation traces are stable across iterations, so when
        ``prompt_optimization.cache_traces_ttl_seconds`` > 0 the merged result is memoized for
        that window — avoiding the repeated read+merge every iteration with an identical result.
        """
        ttl = self._trace_cache_ttl_seconds()
        if ttl > 0:
            cached = getattr(self, "_traces_cache", None)
            if cached is not None and cached[0] == limit and time.monotonic() < cached[2]:
                return cached[1]

        if self._use_pareto:
            jsonl_traces = self._collect_traces_v2(limit=limit)
        else:
            jsonl_traces = self._collect_traces(limit=limit)

        conv_traces = self._collect_traces_from_conversations(limit=limit)
        traces = self._merge_traces(jsonl_traces, conv_traces)

        # Ground the synthetic completion_score/task_type in real linked outcomes
        # (quality_weights response-quality, prompt_optimizer completion) so
        # reflection and the live reward share one signal.
        self._apply_real_outcomes(traces)

        if conv_traces:
            logger.info(
                "Unified traces: %d from JSONL + %d from conversations = %d unique",
                len(jsonl_traces),
                len(conv_traces),
                len(traces),
            )

        if ttl > 0:
            self._traces_cache = (limit, traces, time.monotonic() + ttl)
        return traces

    def _apply_real_outcomes(self, traces: List[ExecutionTrace]) -> None:
        """Override synthetic completion_score/success/task_type with real outcomes.

        A trace's ``completion_score`` is otherwise the ``1 - 1.5*failure_rate``
        proxy and ``task_type`` is ``'default'``. Where the RL correlation spine
        linked genuine outcomes to the session — ``quality_weights``
        response-quality scores and ``prompt_optimizer`` completion scores, both
        carrying a real ``quality_score`` + ``task_type`` + ``session_id`` — use
        them so reflection sees the same signal the live reward trains on.
        Silently falls back to the proxy when no outcome is linked.
        """
        if not traces:
            return
        try:
            from collections import Counter

            from victor.core.schema import Tables

            table = Tables.RL_OUTCOME
        except Exception:
            return
        for trace in traces:
            # Traces may be bare session-ID strings (from _merge_traces when
            # JSONL/conversation sources only yield IDs); skip those — they
            # can't be enriched (no .session_id/.completion_score to mutate).
            if not hasattr(trace, "session_id"):
                continue
            if not trace.session_id:
                continue
            try:
                rows = self.db.execute(
                    f"SELECT quality_score, task_type FROM {table} "
                    f"WHERE session_id = ? "
                    f"AND learner_id IN ('quality_weights', 'prompt_optimizer') "
                    f"AND quality_score IS NOT NULL",
                    (trace.session_id,),
                ).fetchall()
            except Exception:
                continue
            if not rows:
                continue
            scores = [row[0] for row in rows if row[0] is not None]
            if scores:
                trace.completion_score = sum(scores) / len(scores)
                trace.success = trace.completion_score >= 0.5
            tasks = [row[1] for row in rows if row[1]]
            if tasks:
                trace.task_type = Counter(tasks).most_common(1)[0][0]

    def _apply_section_strategies(
        self,
        section_name: str,
        current_text: str,
        traces: List[ExecutionTrace],
        *,
        provider: str = "default",
        query: Optional[str] = None,
        on_phase: Optional[Callable[..., None]] = None,
    ) -> str:
        """Apply the configured strategy chain for a section."""
        new_text = current_text
        for strat in self._strategies_for_section(section_name):
            strat_name = type(strat).__name__
            if on_phase:
                on_phase(section_name, "reflect", strat_name)
            reflection = strat.reflect(
                traces,
                section_name,
                new_text,
                query=query,
                provider=provider,
                target_provider=provider,
            )
            if reflection:
                preview = reflection.lstrip() if section_name == "FEW_SHOT_EXAMPLES" else reflection
                logger.info(
                    "%s reflection for '%s':\n%s",
                    strat_name,
                    section_name,
                    preview[:400],
                )
                if on_phase:
                    on_phase(section_name, "mutate", strat_name)
                new_text = strat.mutate(new_text, reflection, section_name)
            else:
                # An empty reflection is indistinguishable from a skipped or
                # internally-failed strategy unless it says so. This is the
                # common "nothing to propose" path, so debug, not info.
                logger.debug("%s proposed no change for '%s'", strat_name, section_name)
        return new_text

    def get_query_aware_few_shots(self, query: str) -> Optional[str]:
        """Render MIPROv2 few-shot examples tailored to the current query."""
        if not query or not query.strip():
            return None
        if not self._strategies_for_section("FEW_SHOT_EXAMPLES"):
            return None

        traces = self._collect_learning_traces(limit=50)
        if not traces:
            return None

        few_shots = self._apply_section_strategies(
            "FEW_SHOT_EXAMPLES",
            "",
            traces,
            query=query,
        ).strip()
        return few_shots or None

    def _ensure_tables(self) -> None:
        """Create the prompt candidate table and GEPA v2 extensions."""
        from victor.core.schema import Schema

        try:
            self.db.executescript(Schema.AGENT_PROMPT_CANDIDATE)
            self.db.executescript(Schema.AGENT_PROMPT_PARETO_INSTANCE)
            # Migrate: add v2 columns to existing table
            for col_def, default in [
                ("instance_scores TEXT", "'{}'"),
                ("coverage_count INTEGER", "0"),
                ("is_on_frontier INTEGER", "1"),
                ("char_length INTEGER", "0"),
                ("benchmark_score REAL", "0.0"),
                ("benchmark_runs INTEGER", "0"),
                ("benchmark_passed INTEGER", "0"),
                ("strategy_name TEXT", "'gepa'"),
                ("strategy_chain TEXT", "'gepa'"),
                ("requires_benchmark INTEGER", "0"),
            ]:
                try:
                    self.db.execute(
                        f"ALTER TABLE agent_prompt_candidate "
                        f"ADD COLUMN {col_def} DEFAULT {default}"
                    )
                except Exception:
                    pass  # Column already exists
            self.db.commit()
            logger.debug("Prompt optimizer tables ensured (v2)")
        except Exception as e:
            logger.warning("Failed to create prompt optimizer tables: %s", e)

    def _load_candidates(self) -> None:
        """Load candidates from DB into memory.

        Candidates are keyed by (section_name, provider) for provider-aware
        prompt evolution. The dict key is "section_name::provider".
        """
        from victor.core.schema import Tables

        try:
            cursor = self.db.execute(
                f"SELECT section_name, provider, text_hash, text, generation, parent_hash, "
                f"completion_score, token_efficiency, tool_effectiveness, "
                f"alpha, beta, sample_count, instance_scores, coverage_count, "
                f"is_on_frontier, char_length, benchmark_score, benchmark_runs, "
                f"benchmark_passed, is_active, strategy_name, strategy_chain, "
                f"requires_benchmark "
                f"FROM {Tables.AGENT_PROMPT_CANDIDATE}"
            )
            for row in cursor.fetchall():
                try:
                    instance_scores = json_loads(row[12] or "{}")
                except Exception:
                    instance_scores = {}
                candidate = PromptCandidate(
                    section_name=row[0],
                    provider=row[1] or "default",
                    text_hash=row[2],
                    text=row[3],
                    generation=row[4],
                    parent_hash=row[5] or "",
                    scores={
                        "completion_score": row[6],
                        "token_efficiency": row[7],
                        "tool_effectiveness": row[8],
                    },
                    alpha=row[9],
                    beta_val=row[10],
                    sample_count=row[11],
                    instance_scores=instance_scores,
                    coverage_count=row[13] or 0,
                    is_on_frontier=bool(row[14]),
                    char_length=row[15] or 0,
                    benchmark_score=row[16] or 0.0,
                    benchmark_runs=row[17] or 0,
                    benchmark_passed=bool(row[18]),
                    is_active=bool(row[19]),
                    strategy_name=row[20] or "gepa",
                    strategy_chain=row[21] or row[20] or "gepa",
                    requires_benchmark=bool(row[22]),
                )
                key = self._candidate_key(row[0], row[1] or "default")
                self._candidates.setdefault(key, []).append(candidate)
            total = sum(len(v) for v in self._candidates.values())
            if total:
                logger.info("Loaded %d prompt candidates from database", total)
        except Exception as e:
            logger.debug("Failed to load prompt candidates: %s", e)

    @staticmethod
    def _candidate_key(section_name: str, provider: str = "default") -> str:
        """Build the dict key for a (section, provider) pair."""
        return f"{section_name}::{provider}"

    @staticmethod
    def _normalize_strategy_class_name(strategy: "PromptOptimizationStrategy") -> str:
        """Convert a strategy class name to a stable config-style identifier."""
        name = type(strategy).__name__
        if name.endswith("Strategy"):
            name = name[: -len("Strategy")]
        known_names = {
            "GEPA": "gepa",
            "MIPROv2": "miprov2",
            "CoTDistillation": "cot_distillation",
            "PrefPO": "prefpo",
        }
        if name in known_names:
            return known_names[name]
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    @staticmethod
    def _strategy_name_for_candidate(
        strategies: List["PromptOptimizationStrategy"],
    ) -> str:
        """Derive a stable strategy name for persisted candidate metadata."""
        if not strategies:
            return "gepa"
        return PromptOptimizerLearner._normalize_strategy_class_name(strategies[0])

    @staticmethod
    def _strategy_chain_for_candidate(
        strategies: List["PromptOptimizationStrategy"],
    ) -> str:
        """Derive a stable layered strategy-chain label for observability."""
        if not strategies:
            return "gepa"
        parts = [
            PromptOptimizerLearner._normalize_strategy_class_name(strategy)
            for strategy in strategies
        ]
        return "+".join(parts) or "gepa"

    @staticmethod
    def _requires_benchmark_for_candidate(
        strategies: List["PromptOptimizationStrategy"],
    ) -> bool:
        """Whether a strategy chain should remain shadow-only until benchmark approval."""
        return any(
            bool(getattr(strategy, "requires_benchmark_gate", False)) for strategy in strategies
        )

    @staticmethod
    def _servable_candidates(
        candidates: List[PromptCandidate],
    ) -> List[PromptCandidate]:
        """Candidates that may be served on merit — benchmark gate satisfied."""
        return [
            candidate
            for candidate in candidates
            if not candidate.requires_benchmark or candidate.benchmark_passed
        ]

    @staticmethod
    def _pending_benchmark_candidates(
        candidates: List[PromptCandidate],
    ) -> List[PromptCandidate]:
        """Candidates still awaiting the benchmark that would validate them.

        Excluding these outright made the gate a wall rather than a gate:
        nothing in the runtime benchmarks a pending candidate — the only exit is
        an operator running ``victor benchmark --prompt-candidate-hash …`` by
        hand. A PrefPO candidate was therefore never served, never rewarded and
        never promoted for the life of the install. On a default config that is
        three of nine evolvable sections (CONCISE_MODE_GUIDANCE,
        COMPLETION_GUIDANCE, GROUNDING_RULES) unable to enter the loop at all.

        They are offered only when nothing servable exists, and even then
        ``should_serve_candidate`` reaches them through the exploration path
        alone — never proven, never approved. So a validated candidate always
        wins, an unvalidated one is never trusted on merit, and the dead end
        becomes a slow path instead. Corruption is still excluded upstream by
        the persist gate (truncated_tail / redundant_additions / growth).
        """
        return [
            candidate
            for candidate in candidates
            if candidate.requires_benchmark and not candidate.benchmark_passed
        ]

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Update posteriors for the candidate that was served.

        A freshly-evolved candidate has ``is_active=False`` and
        ``sample_count=0``, so a pure active/sample_count lookup could never
        find it (chicken-and-egg: unrewardable until already rewarded). We
        therefore resolve the served candidate by hash first — from the
        outcome metadata, then the last-served ledger — before falling back to
        active / sample_count>0.
        """
        section = outcome.metadata.get("prompt_section")
        if not section:
            return

        provider = outcome.provider or "default"
        served_hash = outcome.metadata.get("prompt_candidate_hash")

        # Try provider-specific first, then default
        for key in [
            self._candidate_key(section, provider),
            self._candidate_key(section, "default"),
        ]:
            candidate = self._resolve_reward_candidate(key, served_hash)
            if candidate is None:
                continue
            success = outcome.success and outcome.quality_score >= 0.5
            candidate.update(success)
            candidate.scores["completion_score"] = (
                candidate.scores.get("completion_score", 0.0) * 0.9 + outcome.quality_score * 0.1
            )
            self._record_pareto_outcome(
                key=key,
                candidate=candidate,
                outcome=outcome,
            )
            self._save_candidate(candidate)
            return

    def _resolve_reward_candidate(
        self,
        key: str,
        served_hash: Optional[str],
    ) -> Optional[PromptCandidate]:
        """Find the candidate to reward for a (section, provider) key.

        Preference order:
          1. exact hash match — from the outcome metadata or the last-served
             ledger; this unblocks a brand-new ``sample_count=0`` candidate;
          2. the active candidate;
          3. the most-recently rewarded candidate (``sample_count > 0``).
        """
        candidates = self._candidates.get(key, [])
        if not candidates:
            return None
        wanted = served_hash or self._last_served.get(key)
        if wanted:
            for candidate in candidates:
                if candidate.text_hash == wanted:
                    return candidate
        active = [c for c in candidates if c.is_active]
        if active:
            return active[-1]
        sampled = [c for c in candidates if c.sample_count > 0]
        if sampled:
            return sampled[-1]
        return None

    def record_served(
        self,
        section_name: str,
        provider: str,
        text_hash: str,
    ) -> None:
        """Record the candidate most recently served for a (section, provider).

        Called by the optimization injector when a candidate is actually
        injected into the live prompt, so a subsequent ``record_outcome`` can
        attribute the turn's outcome to it even before it has any rewards
        (``sample_count=0``).
        """
        if not text_hash:
            return
        self._last_served[self._candidate_key(section_name, provider or "default")] = text_hash

    def get_recommendation(
        self,
        provider: str,
        model: str,
        task_type: str,
        section_name: Optional[str] = None,
    ) -> Optional[RLRecommendation]:
        """Thompson Sampling over candidates for a section.

        Looks up provider-specific candidates first, then falls back to
        'default' provider candidates. This enables per-provider prompt
        evolution while sharing globally evolved prompts as baseline.
        """
        if not section_name:
            return None

        provider_pool = self._candidates.get(
            self._candidate_key(section_name, provider or "default"), []
        )
        default_pool = self._candidates.get(self._candidate_key(section_name, "default"), [])

        provider_candidates = self._servable_candidates(provider_pool)
        default_candidates = self._servable_candidates(default_pool)

        # Only when nothing is servable anywhere do pending candidates get a
        # look — so an approved candidate is never displaced by an unvalidated
        # one, while a section whose every candidate is benchmark-gated still
        # has a way into the loop.
        if not provider_candidates and not default_candidates:
            provider_candidates = self._pending_benchmark_candidates(provider_pool)
            default_candidates = self._pending_benchmark_candidates(default_pool)

        candidates = provider_candidates or default_candidates
        if not candidates:
            return None

        active_candidates = [c for c in candidates if c.is_active]
        approved_candidates = [c for c in candidates if c.benchmark_passed]

        if active_candidates:
            candidates = [c for c in active_candidates if c.benchmark_passed] or active_candidates
        elif approved_candidates:
            candidates = approved_candidates

        # Hybrid: if Pareto enabled, restrict Thompson to frontier candidates
        if self._use_pareto:
            frontier_key = self._candidate_key(
                section_name,
                candidates[0].provider if candidates else (provider or "default"),
            )
            frontier = self._pareto_frontiers.get(frontier_key)
            if frontier:
                frontier_hashes = {
                    e.text_hash
                    for e in frontier.get_frontier()
                    if getattr(e, "instance_scores", {})
                }
                if frontier_hashes:
                    frontier_candidates = [c for c in candidates if c.text_hash in frontier_hashes]
                    if frontier_candidates:
                        candidates = frontier_candidates
                        logger.debug(
                            "Pareto frontier filtered %d → %d candidates for %s",
                            len(candidates) + len(frontier_candidates),
                            len(frontier_candidates),
                            section_name,
                        )

        # Thompson Sampling: sample from (frontier) candidates' Beta distributions
        best = max(candidates, key=lambda c: c.sample())
        evidence_count = max(best.sample_count, best.benchmark_runs)
        confidence = min(evidence_count / (MIN_SAMPLES_FOR_CONFIDENCE * 2), 1.0)

        strategy_label = best.strategy_chain or best.strategy_name or "gepa"
        reason_parts = [
            f"{strategy_label} gen-{best.generation} (α={best.alpha:.1f}, β={best.beta_val:.1f})"
        ]
        if best.is_active:
            reason_parts.append("active")
        if best.benchmark_passed:
            reason_parts.append(
                f"bench={best.benchmark_score:.2f}/{best.benchmark_runs}"
                if best.benchmark_runs
                else "bench-approved"
            )

        return RLRecommendation(
            value=best.text,
            confidence=confidence,
            reason=", ".join(reason_parts),
            sample_size=evidence_count,
            is_baseline=best.sample_count < MIN_SAMPLES_FOR_CONFIDENCE,
            metadata={
                "strategy_name": best.strategy_name,
                "strategy_chain": best.strategy_chain,
                "provider": best.provider,
                "generation": best.generation,
                "prompt_candidate_hash": best.text_hash,
                "section_name": best.section_name,
                "prompt_section_name": best.section_name,
                "benchmark_passed": bool(best.benchmark_passed),
                # Read by should_serve_candidate: a pending candidate is
                # reachable through exploration only, never on merit.
                "requires_benchmark": bool(best.requires_benchmark),
            },
        )

    def evolve(
        self,
        section_name: str,
        current_text: str,
        provider: str = "default",
        query: Optional[str] = None,
        on_phase: Optional[Callable[..., None]] = None,
    ) -> Optional[PromptCandidate]:
        """Run one GEPA evolution cycle for a section.

        Args:
            section_name: Which prompt section to evolve
            current_text: Current text of the section
            provider: Provider scope (e.g., "xai", "ollama", "default")

        Steps:
        1. Collect execution traces from usage.jsonl + evaluation results
        2. Reflect on failure patterns (via strategy)
        3. Mutate prompt text (via strategy)
        4. Store new candidate

        Returns:
            New PromptCandidate, or None if insufficient data
        """
        traces = self._collect_learning_traces(limit=50)
        traces = self._scope_traces_to_provider(traces, provider)

        # Enrich traces with credit signals (FEP-0001 Phase 3)
        self._enrich_traces_with_credit(traces)

        if len(traces) < MIN_TRACES_FOR_EVOLUTION:
            logger.info(
                "Not enough traces for evolution (%d < %d)",
                len(traces),
                MIN_TRACES_FOR_EVOLUTION,
            )
            return None

        # GEPA/ProTeGi require a real scalar reward to make selection meaningful.
        # Without observed outcomes the loop would mutate blindly and never close:
        # candidates would stay at alpha=beta=1.0, sample_count=0, is_active=0
        # (the inert-record state observed in prior sessions). Require at least one
        # rewarded trace so evolution is grounded in evidence, not noise.
        rewarded = [
            t
            for t in traces
            if getattr(t, "reward", None) is not None
            or getattr(t, "quality_score", None) is not None
            or getattr(t, "completion_score", 0.0) > 0.0
        ]
        if not rewarded:
            logger.info(
                "Skipping evolution for '%s': no rewarded traces available "
                "(%d traces, 0 with reward/quality signals).",
                section_name,
                len(traces),
            )
            return None

        # Apply strategies sequentially (layered composition)
        new_text = self._apply_section_strategies(
            section_name,
            current_text,
            traces,
            provider=provider,
            query=query,
            on_phase=on_phase,
        )
        if new_text == current_text:
            merged_candidate = self._attempt_pareto_merge_candidate(
                section_name=section_name,
                provider=provider,
                current_text=current_text,
            )
            if merged_candidate is not None:
                return merged_candidate
            logger.info("Mutation produced no change for '%s'", section_name)
            return None

        # Prompt bloat control: boundary-aware truncation (never mid-instruction).
        # Strategies already sanitize, but this is a final safety net before persist.
        #
        # The cap is a *bloat* control, so it can never sit below the seed it is
        # measuring growth against. COMPLETION_GUIDANCE ships at 1551 chars while
        # max_prompt_chars defaults to 1500, so every mutation of it was amputated
        # to 1496 regardless of quality — the cap, not the strategy, decided the
        # candidate. Floor the effective cap at the seed length so truncation only
        # ever trims genuine growth.
        effective_cap = (
            max(self._max_prompt_chars, len(current_text)) if self._max_prompt_chars else 0
        )
        if effective_cap and len(new_text) > effective_cap:
            from victor.framework.rl.prompt_hygiene import boundary_aware_truncate

            new_text, _ = boundary_aware_truncate(new_text, effective_cap)
            logger.info(
                "GEPA bloat control: truncated '%s' to %d chars (boundary-aware, cap=%d)",
                section_name,
                len(new_text),
                effective_cap,
            )

        # Reject if over 2x the limit (likely garbage output)
        if effective_cap and len(new_text) > 2 * effective_cap:
            logger.warning(
                "GEPA rejected mutation for '%s': %d chars > 2x limit",
                section_name,
                len(new_text),
            )
            return None

        # Persist hygiene gate: enforce only STRUCTURAL degradation signals
        # (runaway growth, repetitive trigrams) — mirroring the GEPA-service
        # strategy gate (gepa_service.py). Seed-similarity and unsupported-
        # addition violations are intentionally NOT enforced here: a mutation
        # is by definition a rewrite, which legitimately rephrases seed lines
        # (→ unsupported_additions) and, for few-shot / distillation strategies,
        # has low seed overlap (→ seed_similarity_too_low). Strategy layers that
        # care about additive constraints (PrefPO) apply their own
        # allowed_additions gate (prefpo_strategy.py); this net catches only
        # corruption. Garbage collapses (e.g. 44-char output) are already
        # rejected upstream in gepa_service.mutate() before reaching here.
        #
        # ``truncated_tail`` and ``redundant_additions`` are corruption too, and
        # both were observed reaching the DB: a COMPLETION_GUIDANCE candidate
        # persisted ending "- Read error messages carefully and", and a
        # CONCISE_MODE_GUIDANCE candidate persisted with "Read the error message
        # carefully." appended directly beneath the line it restates. Neither is
        # a legitimate rewrite, so both join the structural set.
        from victor.framework.rl.prompt_hygiene import evaluate_prompt_candidate

        report = evaluate_prompt_candidate(current_text, new_text)
        structural = {
            "growth_exceeded",
            "repeated_trigrams",
            "truncated_tail",
            "redundant_additions",
        }
        triggered = structural & set(report.violations)
        if triggered:
            # Warning, not info: this is the last gate before storage, so it is
            # the one that turns a completed mutation into a bare "no change"
            # row. Every other discard on this path already announces itself.
            logger.warning(
                "GEPA rejected the '%s' candidate at the persist gate (%s); "
                "%d chars offered against a %d-char seed.",
                section_name,
                ",".join(sorted(triggered)),
                len(new_text),
                len(current_text),
            )
            return None

        # Create candidate
        text_hash = hashlib.md5(new_text.encode()).hexdigest()[:12]
        parent_hash = hashlib.md5(current_text.encode()).hexdigest()[:12]
        key = self._candidate_key(section_name, provider)
        generation = self._get_max_generation(key) + 1
        strategies = self._strategies_for_section(section_name)

        candidate = PromptCandidate(
            section_name=section_name,
            provider=provider,
            text=new_text,
            text_hash=text_hash,
            generation=generation,
            parent_hash=parent_hash,
            strategy_name=self._strategy_name_for_candidate(strategies),
            strategy_chain=self._strategy_chain_for_candidate(strategies),
            char_length=len(new_text),
            requires_benchmark=self._requires_benchmark_for_candidate(strategies),
        )

        self._candidates.setdefault(key, []).append(candidate)
        self._save_candidate(candidate)

        # Prune: keep only top N candidates per section (by mean score)
        MAX_CANDIDATES_PER_SECTION = 10
        section_candidates = self._candidates.get(key, [])
        if len(section_candidates) > MAX_CANDIDATES_PER_SECTION:
            # Keep the highest-mean candidates
            section_candidates.sort(key=lambda c: -c.mean)
            pruned = section_candidates[MAX_CANDIDATES_PER_SECTION:]
            self._candidates[key] = section_candidates[:MAX_CANDIDATES_PER_SECTION]
            # Remove pruned from DB
            from victor.core.schema import Tables

            for p_candidate in pruned:
                try:
                    self.db.execute(
                        f"DELETE FROM {Tables.AGENT_PROMPT_CANDIDATE} WHERE text_hash = ?",
                        (p_candidate.text_hash,),
                    )
                except Exception:
                    pass
            self.db.commit()
            logger.info(
                "Pruned %d candidates from %s (kept top %d)",
                len(pruned),
                key,
                MAX_CANDIDATES_PER_SECTION,
            )

        # GEPA v2: Add to Pareto frontier
        if self._use_pareto:
            if key not in self._pareto_frontiers:
                from victor.framework.rl.pareto import ParetoFrontier

                self._pareto_frontiers[key] = ParetoFrontier(max_candidates=20)
            self._pareto_frontiers[key].add_candidate(
                text_hash=text_hash,
                text=new_text,
                generation=generation,
            )
            self._sync_pareto_state(key)

        logger.info(
            "GEPA evolved '%s' to gen-%d (hash=%s, %d chars%s)",
            section_name,
            generation,
            text_hash,
            len(new_text),
            ", pareto" if self._use_pareto else "",
        )
        return candidate

    def _get_max_generation(self, section_name: str) -> int:
        """Get the highest generation number for a section."""
        candidates = self._candidates.get(section_name, [])
        if not candidates:
            return 0
        return max(c.generation for c in candidates)

    def _save_candidate(self, candidate: PromptCandidate) -> None:
        """Persist a candidate to the database."""
        from victor.core.schema import Tables

        try:
            self.db.execute(
                f"INSERT OR REPLACE INTO {Tables.AGENT_PROMPT_CANDIDATE} "
                f"(section_name, provider, text_hash, text, generation, parent_hash, "
                f"completion_score, token_efficiency, tool_effectiveness, "
                f"alpha, beta, sample_count, instance_scores, coverage_count, "
                f"is_on_frontier, char_length, benchmark_score, benchmark_runs, "
                f"benchmark_passed, is_active, strategy_name, strategy_chain, "
                f"requires_benchmark) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    candidate.section_name,
                    candidate.provider,
                    candidate.text_hash,
                    candidate.text,
                    candidate.generation,
                    candidate.parent_hash,
                    candidate.scores.get("completion_score", 0.0),
                    candidate.scores.get("token_efficiency", 0.0),
                    candidate.scores.get("tool_effectiveness", 0.0),
                    candidate.alpha,
                    candidate.beta_val,
                    candidate.sample_count,
                    json_dumps(candidate.instance_scores or {}),
                    candidate.coverage_count,
                    int(candidate.is_on_frontier),
                    candidate.char_length or len(candidate.text),
                    candidate.benchmark_score,
                    candidate.benchmark_runs,
                    int(candidate.benchmark_passed),
                    int(candidate.is_active),
                    candidate.strategy_name,
                    candidate.strategy_chain,
                    int(candidate.requires_benchmark),
                ),
            )
            self.db.commit()
        except Exception as e:
            logger.warning("Failed to save prompt candidate: %s", e)

    def _find_candidate(
        self,
        section_name: str,
        provider: str,
        text_hash: str,
    ) -> Optional[PromptCandidate]:
        """Find a specific candidate by section/provider/hash."""
        candidates = self._candidates.get(self._candidate_key(section_name, provider), [])
        for candidate in candidates:
            if candidate.text_hash == text_hash:
                return candidate
        return None

    def get_candidate(
        self,
        *,
        section_name: str,
        provider: str,
        text_hash: str,
    ) -> Optional[PromptCandidate]:
        """Return one exact prompt candidate for targeted evaluation/runtime binding."""
        return self._find_candidate(section_name, provider, text_hash)

    def find_candidate_any_provider(
        self,
        *,
        section_name: str,
        text_hash: str,
    ) -> Optional[PromptCandidate]:
        """Find a candidate by section+hash across ALL providers.

        Fallback for binding lookups where the requested provider doesn't match
        the candidate's stored provider (e.g. a default-profile run binding a
        zai-scoped candidate). Used only after the provider-specific lookup misses.
        """
        prefix = f"{section_name}::"
        for key, candidates in self._candidates.items():
            if not key.startswith(prefix):
                continue
            for candidate in candidates:
                if candidate.text_hash == text_hash:
                    return candidate
        return None

    def get_candidates(
        self,
        *,
        section_name: str,
        provider: str = "default",
    ) -> List[PromptCandidate]:
        """Return all candidates for one section/provider ordered by creation."""
        candidates = self._candidates.get(self._candidate_key(section_name, provider), [])
        return sorted(candidates, key=lambda c: (c.generation, c.text_hash))

    def resolve_candidate(
        self,
        *,
        section_name: str,
        provider: str = "default",
        selector: str,
    ) -> Optional[PromptCandidate]:
        """Resolve a candidate by hash prefix or ordinal string."""
        normalized = selector.strip()
        if not normalized:
            return None
        candidates = self.get_candidates(section_name=section_name, provider=provider)
        if normalized.isdigit():
            generation = int(normalized)
            matches = [candidate for candidate in candidates if candidate.generation == generation]
            if matches:
                return matches[-1]
            return None
        matches = [
            candidate
            for candidate in candidates
            if candidate.text_hash == normalized or candidate.text_hash.startswith(normalized)
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def record_benchmark_result(
        self,
        section_name: str,
        provider: str,
        text_hash: str,
        score: float,
        passed: bool,
    ) -> Optional[PromptCandidate]:
        """Record a benchmark result for a candidate.

        Uses a running average for benchmark score and remembers whether
        the candidate has ever passed its gating benchmark.
        """
        candidate = self._find_candidate(section_name, provider, text_hash)
        if candidate is None:
            # A candidate served cross-provider — evolved under one provider,
            # measured under another — must still be recordable. Serving resolves
            # it via ``find_candidate_any_provider`` (optimization_injector), so
            # the candidate's text is injected correctly; recording has to agree
            # or the measurement is silently discarded. ``recorded=False`` forces
            # ``decision.passed`` false however strongly the candidate won, so
            # every cross-provider benchmark (all of them, in practice) reported
            # "no candidate met the threshold" on results that cleared the gate.
            candidate = self.find_candidate_any_provider(
                section_name=section_name, text_hash=text_hash
            )
        if candidate is None:
            return None

        previous_runs = candidate.benchmark_runs
        cumulative_score = candidate.benchmark_score * previous_runs
        candidate.benchmark_runs += 1
        candidate.benchmark_score = (cumulative_score + score) / candidate.benchmark_runs
        candidate.benchmark_passed = candidate.benchmark_passed or bool(passed)
        self._save_candidate(candidate)
        return candidate

    @staticmethod
    def _throttled_task_share(result: Any) -> tuple[int, int]:
        """(throttled, total) tasks for one arm."""
        from victor.evaluation.harness import was_throttled

        tasks = list(getattr(result, "task_results", []) or [])
        return sum(1 for task in tasks if was_throttled(task)), len(tasks)

    @staticmethod
    def _baseline_run(runs: List[Any]) -> Optional[Any]:
        """The seed arm, if the suite ran one."""
        from victor.agent.optimization_injector import BASELINE_CANDIDATE_HASH

        for run in runs:
            config = getattr(run, "config", None)
            spec = getattr(run, "spec", None)
            candidate_hash = getattr(config, "prompt_candidate_hash", None) or getattr(
                spec, "prompt_candidate_hash", None
            )
            if candidate_hash == BASELINE_CANDIDATE_HASH:
                return run
        return None

    @staticmethod
    def _paired_contrast(baseline_run: Optional[Any], run: Any) -> Optional[Any]:
        """Pair a candidate arm against the baseline, or None when impossible."""
        if baseline_run is None:
            return None
        from victor.evaluation.harness import PairedContrast

        try:
            return PairedContrast.from_results(
                getattr(baseline_run, "result", None),
                getattr(run, "result", None),
            )
        except (ValueError, AttributeError, TypeError) as exc:
            # A failed pairing must not silently become "no difference".
            logger.warning("Could not pair this arm against the baseline: %s", exc)
            return None

    def sync_evaluation_suite(
        self,
        suite: Any,
        *,
        min_pass_rate: float = 0.5,
        min_discordant: int = 8,
        promote_best: bool = False,
    ) -> PromptCandidateBenchmarkSyncResult:
        """Write a candidate-bound benchmark suite back into prompt-candidate state.

        This path is intentionally conservative:
        - every suite run contributes benchmark score history
        - only the suite winner can satisfy benchmark gating by default
        - promotion remains opt-in and only happens after the winner passes the gate

        When the suite carries a baseline arm the gate is **comparative**: the
        winner must beat the seed on the same tasks, by enough disagreements to
        mean anything. An absolute pass rate cannot answer the question being
        asked — it approves a candidate that clears 50% while being worse than
        the prompt it replaces, and rejects one that beats the seed on a hard
        benchmark. ``min_discordant`` is the floor on evidence: fewer than this
        many tasks where the arms disagree is not a result, whichever way it
        leans.

        The p-value is reported, never enforced. At these sample sizes it is
        advisory — a decision aid for the human reading the row, not a gate.

        Without a baseline arm this falls back to the old absolute threshold and
        says so, because a suite with nothing to compare against is an
        observation, not an experiment.
        """
        if min_pass_rate < 0.0 or min_pass_rate > 1.0:
            raise ValueError("min_pass_rate must be between 0.0 and 1.0")
        if min_discordant < 0:
            raise ValueError("min_discordant must be non-negative")

        runs = list(getattr(suite, "runs", []) or [])
        if not runs:
            return PromptCandidateBenchmarkSyncResult()

        baseline_run = self._baseline_run(runs)
        # The baseline is the referent and can never be a candidate: ranking it
        # alongside the arms it measures lets the seed "win" its own experiment
        # and blocks every real candidate behind it.
        runs = [run for run in runs if run is not baseline_run]
        if not runs:
            logger.warning("Suite contained only a baseline arm; nothing to evaluate.")
            return PromptCandidateBenchmarkSyncResult()
        if baseline_run is None:
            logger.warning(
                "Suite has no baseline arm, so candidates cannot be compared to the "
                "seed; falling back to the absolute pass-rate gate (>= %.2f). Re-run "
                "with --include-baseline for a real contrast.",
                min_pass_rate,
            )

        def _run_sort_key(run: Any) -> tuple[float, int, float]:
            result = getattr(run, "result", None)
            pass_rate = float(getattr(result, "pass_rate", 0.0) or 0.0)
            total_tasks = int(getattr(result, "total_tasks", 0) or 0)
            duration_seconds = float(getattr(result, "duration_seconds", 0.0) or 0.0)
            return (pass_rate, total_tasks, -duration_seconds)

        ranked_runs = sorted(runs, key=_run_sort_key, reverse=True)
        best_run = ranked_runs[0]
        best_hash = getattr(getattr(best_run, "config", None), "prompt_candidate_hash", None)

        sync_result = PromptCandidateBenchmarkSyncResult(best_prompt_candidate_hash=best_hash)
        best_decision: Optional[PromptCandidateBenchmarkDecision] = None

        for rank, run in enumerate(ranked_runs, start=1):
            config = getattr(run, "config", None)
            spec = getattr(run, "spec", None)
            result = getattr(run, "result", None)

            section_name = (
                getattr(config, "prompt_section_name", None)
                or getattr(spec, "section_name", None)
                or ""
            )
            provider = (
                getattr(config, "provider", None)
                or getattr(spec, "provider", None)
                or getattr(getattr(suite, "base_config", None), "provider", None)
                or "default"
            )
            prompt_candidate_hash = (
                getattr(config, "prompt_candidate_hash", None)
                or getattr(spec, "prompt_candidate_hash", None)
                or ""
            )
            score = float(getattr(result, "pass_rate", 0.0) or 0.0)

            # An arm the provider throttled was never evaluated. Recording it is
            # worse than recording nothing: a candidate nobody tested acquires a
            # "failed benchmark (0.00 over 1 runs)" that every later audit,
            # promotion check, and Thompson draw treats as real. Observed live —
            # three arms back to back exhausted one provider and the last one
            # posted 0/24 in 128 seconds against a prompt that never ran.
            throttled, total = self._throttled_task_share(result)
            if total and throttled / total > MAX_THROTTLED_TASK_SHARE:
                logger.warning(
                    "Skipping %s: %d of %d tasks were rate-limited, so this arm was "
                    "never evaluated. Re-run it with fresh quota — recording it would "
                    "mark an untested candidate as failed.",
                    prompt_candidate_hash[:12] or "an arm",
                    throttled,
                    total,
                )
                continue

            contrast = self._paired_contrast(baseline_run, run)
            is_winner = prompt_candidate_hash == best_hash
            if contrast is not None:
                # Enough disagreements *and* a lead bigger than chance would
                # produce on that many. The first condition alone approved a
                # candidate at 8 versus 6 — 14 disagreements is plenty of
                # evidence, and an effect of 2 against a 3.7 noise floor is
                # still a coin flip (the exact test said p=0.79). Volume of
                # disagreement is not the same question as asymmetry of it.
                passed = (
                    is_winner and contrast.discordant >= min_discordant and contrast.beats_noise()
                )
                if is_winner and not passed:
                    logger.info(
                        "Candidate %s did not clear the comparative gate: %s "
                        "(needs %d+ discordant tasks and an effect above the "
                        "%.1f noise floor).",
                        prompt_candidate_hash[:12],
                        contrast.summary(),
                        min_discordant,
                        contrast.noise_floor,
                    )
            else:
                passed = is_winner and score > 0.0 and score >= min_pass_rate

            candidate = None
            recorded = bool(section_name and prompt_candidate_hash)
            if recorded:
                candidate = self.record_benchmark_result(
                    section_name=section_name,
                    provider=provider,
                    text_hash=prompt_candidate_hash,
                    score=score,
                    passed=passed,
                )
                recorded = candidate is not None

            decision = PromptCandidateBenchmarkDecision(
                prompt_candidate_hash=prompt_candidate_hash,
                section_name=section_name,
                provider=provider,
                score=score,
                passed=passed and recorded,
                recorded=recorded,
                rank=rank,
                benchmark_score=(candidate.benchmark_score if candidate is not None else 0.0),
                benchmark_runs=(candidate.benchmark_runs if candidate is not None else 0),
                paired_contrast=contrast,
            )
            sync_result.decisions.append(decision)

            if decision.passed:
                sync_result.approved_prompt_candidate_hash = prompt_candidate_hash
                best_decision = decision

        if promote_best and best_decision is not None:
            promoted = self.promote_candidate(
                section_name=best_decision.section_name,
                provider=best_decision.provider,
                text_hash=best_decision.prompt_candidate_hash,
            )
            if promoted is not None:
                best_decision.promoted = True
                sync_result.promoted_prompt_candidate_hash = promoted.text_hash

        return sync_result

    def promote_candidate(
        self,
        section_name: str,
        provider: str,
        text_hash: str,
    ) -> Optional[PromptCandidate]:
        """Promote a candidate to active status for its section/provider."""
        key = self._candidate_key(section_name, provider)
        candidates = self._candidates.get(key, [])
        if not candidates:
            return None

        target: Optional[PromptCandidate] = None
        for candidate in candidates:
            if candidate.text_hash == text_hash:
                target = candidate
                break

        if target is None:
            return None
        if target.requires_benchmark and not target.benchmark_passed:
            raise ValueError("cannot promote candidate before benchmark gating passes")
        if target.benchmark_runs > 0 and not target.benchmark_passed:
            raise ValueError("cannot promote candidate that has failed benchmark gating")

        for candidate in candidates:
            candidate.is_active = candidate.text_hash == text_hash
            self._save_candidate(candidate)
        return target

    def build_rollout_experiment_config(
        self,
        section_name: str,
        provider: str,
        treatment_hash: str,
        *,
        control_hash: Optional[str] = None,
        traffic_split: float = 0.1,
        min_samples_per_variant: int = 100,
    ) -> Any:
        """Build an A/B experiment config for safely rolling out a prompt candidate."""
        from victor.framework.rl.experiment_coordinator import (
            ExperimentConfig,
            Variant,
            VariantType,
        )

        key = self._candidate_key(section_name, provider)
        candidates = self._candidates.get(key, [])
        if not candidates:
            raise ValueError(f"no candidates found for {section_name}/{provider}")

        treatment = next((c for c in candidates if c.text_hash == treatment_hash), None)
        if treatment is None:
            raise ValueError(f"unknown treatment candidate: {treatment_hash}")
        if treatment.requires_benchmark and not treatment.benchmark_passed:
            raise ValueError("cannot create rollout experiment before benchmark gating passes")

        if control_hash:
            control = next(
                (
                    c
                    for c in candidates
                    if c.text_hash == control_hash and c.text_hash != treatment_hash
                ),
                None,
            )
        else:
            approved_controls = [
                candidate
                for candidate in candidates
                if candidate.text_hash != treatment_hash and candidate.benchmark_passed
            ]
            active_controls = [candidate for candidate in approved_controls if candidate.is_active]
            control = active_controls[0] if active_controls else None
            if control is None and approved_controls:
                control = max(
                    approved_controls,
                    key=lambda c: (
                        c.benchmark_score,
                        c.benchmark_runs,
                        c.sample_count,
                        c.generation,
                    ),
                )
        if control is None:
            raise ValueError("no approved control candidate available for rollout")

        experiment_id = (
            f"prompt_optimizer_{section_name.lower()}_{provider or 'default'}_{treatment_hash}"
        )
        return ExperimentConfig(
            experiment_id=experiment_id,
            name=f"Prompt rollout for {section_name}",
            description=(
                f"Roll out prompt candidate {treatment_hash} against control {control.text_hash} "
                f"for section {section_name} on provider {provider or 'default'}."
            ),
            control=Variant(
                name=control.text_hash,
                type=VariantType.CONTROL,
                config={
                    "learner": "prompt_optimizer",
                    "section_name": section_name,
                    "provider": provider,
                    "text_hash": control.text_hash,
                    "strategy_name": control.strategy_name,
                },
                description=f"Approved control prompt ({control.strategy_name})",
            ),
            treatment=Variant(
                name=treatment.text_hash,
                type=VariantType.TREATMENT,
                config={
                    "learner": "prompt_optimizer",
                    "section_name": section_name,
                    "provider": provider,
                    "text_hash": treatment.text_hash,
                    "strategy_name": treatment.strategy_name,
                },
                description=f"Candidate rollout prompt ({treatment.strategy_name})",
            ),
            traffic_split=traffic_split,
            min_samples_per_variant=min_samples_per_variant,
        )

    def create_rollout_experiment(
        self,
        coordinator: Any,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
        control_hash: Optional[str] = None,
        traffic_split: float = 0.1,
        min_samples_per_variant: int = 100,
    ) -> str:
        """Create and start a rollout experiment for an approved candidate."""
        config = self.build_rollout_experiment_config(
            section_name=section_name,
            provider=provider,
            treatment_hash=treatment_hash,
            control_hash=control_hash,
            traffic_split=traffic_split,
            min_samples_per_variant=min_samples_per_variant,
        )
        if not coordinator.create_experiment(config):
            raise ValueError(f"experiment already exists: {config.experiment_id}")
        if not coordinator.start_experiment(config.experiment_id):
            raise ValueError(f"failed to start experiment: {config.experiment_id}")
        return config.experiment_id

    def rollback_active_candidate(
        self,
        section_name: str,
        provider: str,
        failed_text_hash: Optional[str] = None,
    ) -> Optional[PromptCandidate]:
        """Rollback the active candidate to the best prior approved candidate."""
        key = self._candidate_key(section_name, provider)
        candidates = self._candidates.get(key, [])
        if not candidates:
            return None

        failed_hash = failed_text_hash
        if failed_hash is None:
            active = next((c for c in candidates if c.is_active), None)
            failed_hash = active.text_hash if active else None

        for candidate in candidates:
            if failed_hash and candidate.text_hash == failed_hash:
                candidate.is_active = False
                self._save_candidate(candidate)

        fallback_candidates = [
            c for c in candidates if c.text_hash != failed_hash and c.benchmark_passed
        ]
        if not fallback_candidates:
            return None

        fallback = max(
            fallback_candidates,
            key=lambda c: (
                c.benchmark_score,
                c.benchmark_runs,
                c.sample_count,
                c.generation,
            ),
        )
        for candidate in candidates:
            candidate.is_active = candidate.text_hash == fallback.text_hash
            self._save_candidate(candidate)
        return fallback

    def _record_pareto_outcome(
        self,
        *,
        key: str,
        candidate: PromptCandidate,
        outcome: RLOutcome,
    ) -> None:
        """Update provider-scoped Pareto evidence from a concrete runtime outcome."""
        if not self._use_pareto:
            return

        instance_id = self._runtime_instance_id(outcome)
        if not instance_id:
            return

        frontier = self._pareto_frontiers.get(key)
        if frontier is None:
            from victor.framework.rl.pareto import ParetoFrontier

            frontier = ParetoFrontier(max_candidates=20)
            self._pareto_frontiers[key] = frontier

        frontier.add_candidate(
            text_hash=candidate.text_hash,
            text=candidate.text,
            generation=candidate.generation,
            instance_scores=candidate.instance_scores,
        )
        frontier.update_instance_score(
            candidate.text_hash, instance_id, self._compute_reward(outcome)
        )
        self._sync_pareto_state(key)

    @staticmethod
    def _runtime_instance_id(outcome: RLOutcome) -> Optional[str]:
        """Build a stable runtime instance identifier for Pareto tracking."""
        metadata = outcome.metadata or {}
        raw_instance = (
            metadata.get("task_id") or metadata.get("instance_id") or metadata.get("session_id")
        )
        if not raw_instance:
            return None
        return f"{raw_instance}::{outcome.provider or 'default'}"

    def _sync_pareto_state(self, key: str) -> None:
        """Persist frontier-derived metadata back onto candidates and instance table."""
        if not self._use_pareto:
            return

        from victor.core.schema import Tables

        frontier = self._pareto_frontiers.get(key)
        if frontier is None:
            return

        candidates = self._candidates.get(key, [])
        entry_by_hash = {entry.text_hash: entry for entry in frontier.get_frontier()}
        frontier_hashes = set(entry_by_hash)

        for candidate in candidates:
            entry = entry_by_hash.get(candidate.text_hash)
            if entry is not None:
                candidate.instance_scores = dict(entry.instance_scores)
                candidate.coverage_count = entry.coverage_count
                candidate.is_on_frontier = True
                candidate.char_length = entry.char_length or len(candidate.text)
            else:
                candidate.coverage_count = 0
                candidate.is_on_frontier = candidate.text_hash in frontier_hashes
                candidate.char_length = candidate.char_length or len(candidate.text)
            self._save_candidate(candidate)

        section_name, provider = key.split("::", 1)
        try:
            self.db.execute(
                f"DELETE FROM {Tables.AGENT_PROMPT_PARETO_INSTANCE} "
                f"WHERE section_name = ? AND provider = ?",
                (section_name, provider),
            )
            for instance_id, (
                best_hash,
                best_score,
            ) in frontier.get_instance_winners().items():
                winner = next((c for c in candidates if c.text_hash == best_hash), None)
                sample_count = winner.sample_count if winner is not None else 0
                self.db.execute(
                    f"INSERT OR REPLACE INTO {Tables.AGENT_PROMPT_PARETO_INSTANCE} "
                    f"(section_name, provider, instance_id, best_candidate_hash, best_score, sample_count) "
                    f"VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        section_name,
                        provider,
                        instance_id,
                        best_hash,
                        best_score,
                        sample_count,
                    ),
                )
            self.db.commit()
        except Exception as e:
            logger.debug("Failed to persist Pareto instance state for %s: %s", key, e)

    def _attempt_pareto_merge_candidate(
        self,
        *,
        section_name: str,
        provider: str,
        current_text: str,
    ) -> Optional[PromptCandidate]:
        """Fallback to GEPA merge when reflection/mutation yields no novel candidate."""
        if not self._use_pareto:
            return None

        key = self._candidate_key(section_name, provider)
        frontier = self._pareto_frontiers.get(key)
        if frontier is None or frontier.size < 2:
            return None

        merge_strategy = next(
            (
                strategy
                for strategy in self._strategies_for_section(section_name)
                if callable(getattr(strategy, "merge", None))
            ),
            None,
        )
        if merge_strategy is None:
            return None

        merged_entry = frontier.attempt_merge(merge_strategy, section_name=section_name)
        if merged_entry is None:
            return None

        existing = self._find_candidate(section_name, provider, merged_entry.text_hash)
        if existing is not None:
            return existing

        strategies = self._strategies_for_section(section_name)
        candidate = PromptCandidate(
            section_name=section_name,
            provider=provider,
            text=merged_entry.text,
            text_hash=merged_entry.text_hash,
            generation=max(merged_entry.generation, self._get_max_generation(key) + 1),
            parent_hash=hashlib.md5(current_text.encode()).hexdigest()[:12],
            strategy_name=self._strategy_name_for_candidate(strategies),
            strategy_chain=f"{self._strategy_chain_for_candidate(strategies)}+merge",
            instance_scores=dict(merged_entry.instance_scores),
            coverage_count=merged_entry.coverage_count,
            is_on_frontier=True,
            char_length=merged_entry.char_length or len(merged_entry.text),
            requires_benchmark=self._requires_benchmark_for_candidate(strategies),
        )

        self._candidates.setdefault(key, []).append(candidate)
        frontier.add_candidate(
            text_hash=merged_entry.text_hash,
            text=merged_entry.text,
            generation=candidate.generation,
            instance_scores=merged_entry.instance_scores,
        )
        self._sync_pareto_state(key)
        return candidate

    @staticmethod
    def _artifact_text_value(value: Any) -> Optional[str]:
        """Normalize serialized evaluation artifact identity values."""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @classmethod
    def _artifact_feedback_metadata(cls, payload: Any) -> Dict[str, Any]:
        """Extract nested runtime-feedback metadata from a saved artifact."""
        if not isinstance(payload, dict):
            return {}
        runtime_feedback = payload.get("runtime_evaluation_feedback")
        if not isinstance(runtime_feedback, dict):
            return {}
        metadata = runtime_feedback.get("metadata")
        return dict(metadata) if isinstance(metadata, dict) else {}

    @classmethod
    def _artifact_identities(
        cls,
        payload: Dict[str, Any],
    ) -> List[tuple[str, str, str, str]]:
        """Every complete prompt identity an artifact can be attributed to.

        Prefers the explicit identity (a targeted ``--prompt-candidate-hash``
        A/B), then falls back to ``observed_prompt_identities`` — what the
        runtime actually served during the run. The fallback is what makes
        ordinary benchmark runs usable: before it existed, only hand-run A/Bs
        carried identity, so nearly every eval artifact on disk was skipped and
        the Pareto frontier stayed empty despite thousands of scored tasks.

        Only fully-identified entries are returned, so callers need no further
        validation.
        """
        candidate_hash, section_name, provider, model = cls._artifact_identity(payload)
        if candidate_hash and section_name:
            return [(candidate_hash, section_name, provider, model)]

        identities: List[tuple[str, str, str, str]] = []
        seen: set = set()
        observed = payload.get("observed_prompt_identities")
        if not isinstance(observed, list):
            return identities
        for entry in observed:
            if not isinstance(entry, dict):
                continue
            entry_hash = cls._artifact_text_value(entry.get("prompt_candidate_hash"))
            entry_section = cls._artifact_text_value(
                entry.get("prompt_section_name") or entry.get("section_name")
            )
            if not entry_hash or not entry_section:
                continue
            entry_provider = cls._artifact_text_value(entry.get("provider")) or provider
            key = (entry_hash, entry_section, entry_provider)
            if key in seen:
                continue
            seen.add(key)
            identities.append((entry_hash, entry_section, entry_provider, model))
        return identities

    @classmethod
    def _artifact_identity(
        cls,
        payload: Dict[str, Any],
    ) -> tuple[Optional[str], Optional[str], str, str]:
        """Return canonical prompt identity for a benchmark/session artifact."""
        config = payload.get("config")
        config_dict = dict(config) if isinstance(config, dict) else {}
        feedback_metadata = cls._artifact_feedback_metadata(payload)
        feedback_scope = feedback_metadata.get("scope")
        scope_dict = dict(feedback_scope) if isinstance(feedback_scope, dict) else {}

        candidate_hash = (
            cls._artifact_text_value(payload.get("prompt_candidate_hash"))
            or cls._artifact_text_value(config_dict.get("prompt_candidate_hash"))
            or cls._artifact_text_value(config_dict.get("text_hash"))
            or cls._artifact_text_value(config_dict.get("candidate_hash"))
            or cls._artifact_text_value(feedback_metadata.get("prompt_candidate_hash"))
        )
        section_name = (
            cls._artifact_text_value(
                payload.get("section_name") or payload.get("prompt_section_name")
            )
            or cls._artifact_text_value(
                config_dict.get("section_name") or config_dict.get("prompt_section_name")
            )
            or cls._artifact_text_value(
                feedback_metadata.get("section_name")
                or feedback_metadata.get("prompt_section_name")
            )
        )
        provider = (
            cls._artifact_text_value(payload.get("provider"))
            or cls._artifact_text_value(config_dict.get("provider"))
            or cls._artifact_text_value(feedback_metadata.get("provider"))
            or cls._artifact_text_value(scope_dict.get("provider"))
            or "default"
        )
        model = (
            cls._artifact_text_value(payload.get("model"))
            or cls._artifact_text_value(config_dict.get("model"))
            or cls._artifact_text_value(feedback_metadata.get("model"))
            or cls._artifact_text_value(scope_dict.get("model"))
            or "unknown"
        )
        return candidate_hash, section_name, provider, model

    @classmethod
    def _validated_artifact_score(cls, payload: Dict[str, Any]) -> Optional[float]:
        """Return a conservative per-instance score for validated session artifacts."""
        score_payload = payload.get("score")
        if isinstance(score_payload, dict):
            overall_score = score_payload.get("overall_score")
            if isinstance(overall_score, (int, float)):
                return max(0.0, min(1.0, float(overall_score)))
        elif isinstance(score_payload, (int, float)):
            return max(0.0, min(1.0, float(score_payload)))

        validation_result = payload.get("validation_result")
        if isinstance(validation_result, dict):
            validation_score = validation_result.get("score")
            if isinstance(validation_score, (int, float)):
                return max(0.0, min(1.0, float(validation_score)))

        status = cls._artifact_text_value(payload.get("status"))
        if status is None:
            return None
        return 1.0 if status.lower() == "passed" else 0.0

    @classmethod
    def _artifact_instance_scores(
        cls,
        payload: Dict[str, Any],
        *,
        model: str,
    ) -> List[tuple[str, float]]:
        """Extract concrete per-instance outcome scores from saved artifacts."""
        tasks = payload.get("tasks")
        if isinstance(tasks, list):
            scored_instances: List[tuple[str, float]] = []
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                task_id = cls._artifact_text_value(task.get("task_id") or task.get("instance_id"))
                status = cls._artifact_text_value(task.get("status"))
                if task_id is None or status is None:
                    continue
                score = 1.0 if status.lower() == "passed" else 0.0
                scored_instances.append((f"{task_id}::{model}", score))
            return scored_instances

        task_id = cls._artifact_text_value(payload.get("task_id") or payload.get("instance_id"))
        if task_id is None:
            return []

        score = cls._validated_artifact_score(payload)
        if score is None:
            return []
        return [(f"{task_id}::{model}", score)]

    def seed_from_evaluations(self, eval_dir: Optional[Path] = None) -> int:
        """Load evaluation results and update Pareto instance scores.

        Reads canonical ``eval_*.json`` benchmark/session artifacts and updates
        each frontier candidate's per-instance scores for multi-objective
        selection. Full benchmark result files contribute pass/fail evidence,
        while validated session-truth artifacts can contribute richer explicit
        validation scores when available.

        Returns:
            Number of instance scores updated
        """
        if not self._use_pareto:
            return 0

        if eval_dir is None:
            eval_dir = Path.home() / ".victor" / "evaluations"

        updated = 0
        for eval_file in sorted(Path(eval_dir).glob("eval_*.json")):
            try:
                with open(eval_file) as f:
                    data = json.load(f)
                for candidate_hash, section_name, provider, model in self._artifact_identities(
                    data
                ):
                    key = self._candidate_key(section_name, provider)
                    frontier = self._pareto_frontiers.get(key)
                    if frontier is None:
                        continue

                    for instance_id, score in self._artifact_instance_scores(data, model=model):
                        frontier.update_instance_score(candidate_hash, instance_id, score)
                        updated += 1
                    self._sync_pareto_state(key)
            except Exception:
                continue

        if updated:
            logger.info("Seeded %d Pareto instance scores from evaluations", updated)
        return updated

    def _select_challenging_traces(
        self, traces: List[ExecutionTrace], max_traces: int = 20
    ) -> List[ExecutionTrace]:
        """SIMBA-inspired: bias selection toward challenging examples.

        Scores each trace by challenge value. Recovery patterns are most
        valuable, followed by high-failure, borderline scores, detailed errors.
        Returns 70/30 mix of challenging/easy traces for contrast.
        """
        if len(traces) < max_traces:
            return traces

        scored = []
        for trace in traces:
            challenge = 0.0
            zone = classify_trace_zone(trace)
            if zone == TraceZone.RECOVERY:
                challenge += 0.4
            total_failures = sum(getattr(trace, "tool_failures", {}).values())
            challenge += 0.3 * min(total_failures / 5.0, 1.0)
            score = getattr(trace, "completion_score", 0.0)
            if 0.1 < score < 0.7:
                challenge += 0.2 * (1.0 - score)
            details = getattr(trace, "tool_call_details", [])
            has_errors = any(getattr(d, "error_detail", "") for d in details)
            if has_errors:
                challenge += 0.1
            scored.append((trace, challenge))

        scored.sort(key=lambda x: -x[1])
        n_challenging = int(max_traces * 0.7)
        n_easy = max_traces - n_challenging
        challenging = [t for t, _ in scored[:n_challenging]]
        easy = [t for t in traces if classify_trace_zone(t) == TraceZone.SUCCESS][:n_easy]
        return challenging + easy

    def _collect_traces(self, limit: int = 50) -> List[ExecutionTrace]:
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

                            self._absorb_session_identity(sessions[sid], data)
                            self._absorb_run_kind(sessions[sid], event)

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
                                    cat = self._categorize_failure(error)
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

            completion_score, success, score_source = self._score_session(
                verdicts.get(sid), failure_rate
            )

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

    @staticmethod
    def _verdict_from_task(task: Dict[str, Any], benchmark: str) -> Optional[HarnessVerdict]:
        """Build a verdict from one serialized task, or None if it cannot grade.

        Derived from hard signals only — ``status`` and the test counts — and
        deliberately *not* from the artifact's own ``completion_score`` field.
        That field is the proxy this change exists to replace, and it is
        observably unreliable: an artifact on disk carries
        ``"status": "failed"`` alongside ``"completion_score": "1.0"``. Reading
        it back would reimport the defect under a new name.

        Test counts give partial credit where they exist, so a run that fixed
        8 of 10 tests is better evidence than one that fixed none, without
        trusting a soft score that can contradict the verdict.
        """
        session_id = str(task.get("session_id") or "").strip()
        if not session_id:
            return None
        status = str(task.get("status") or "").strip().lower()
        if not status:
            return None

        success = status == "passed"
        if success:
            score = 1.0
        else:
            try:
                tests_total = int(task.get("tests_total") or 0)
                tests_passed = int(task.get("tests_passed") or 0)
            except (TypeError, ValueError):
                tests_total = tests_passed = 0
            # A non-passing run never scores 1.0, however many tests it passed.
            score = min(tests_passed / tests_total, 0.99) if tests_total > 0 else 0.0
            score = max(score, 0.0)

        return HarnessVerdict(
            completion_score=score,
            success=success,
            task_id=str(task.get("task_id") or ""),
            benchmark=benchmark,
        )

    def _harness_verdicts(self, eval_dir: Optional[Path] = None) -> Dict[str, HarnessVerdict]:
        """Map session_id → ground-truth outcome, from benchmark artifacts.

        Evaluation runs are the only sessions that carry a real verdict: a
        harness decided whether the task was actually solved. Traces from those
        sessions must be scored by that verdict rather than by how tidily the
        agent called its tools — an artifact on disk shows the two disagreeing
        outright (``"status": "failed"`` alongside ``"completion_score": 1.0``
        computed from the proxy).

        Bounded to the most recent artifacts by mtime and memoized per instance:
        the corpus runs to thousands of files, while trace collection only ever
        looks at the last ~50 sessions.
        """
        if self._harness_verdict_cache is not None:
            return self._harness_verdict_cache

        if eval_dir is None:
            try:
                from victor.config.settings import get_project_paths

                eval_dir = Path(get_project_paths().global_victor_dir) / "evaluations"
            except Exception:
                eval_dir = Path.home() / ".victor" / "evaluations"

        verdicts: Dict[str, HarnessVerdict] = {}
        try:
            artifacts = sorted(
                Path(eval_dir).glob("eval_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:MAX_EVAL_ARTIFACTS_SCANNED]
        except OSError:
            artifacts = []

        for artifact in artifacts:
            try:
                with open(artifact) as handle:
                    payload = json_loads(handle.read())
            except (OSError, JSONDecodeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            config = payload.get("config")
            benchmark = str(
                payload.get("benchmark")
                or (config.get("benchmark") if isinstance(config, dict) else "")
                or ""
            )
            tasks = payload.get("tasks")
            if not isinstance(tasks, list):
                continue
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                verdict = self._verdict_from_task(task, benchmark)
                if verdict is None:
                    continue
                # Artifacts are walked newest-first, so setdefault means a
                # re-run's grade supersedes the earlier one for that session.
                verdicts.setdefault(str(task.get("session_id")).strip(), verdict)

        if verdicts:
            logger.info(
                "Loaded %d harness verdicts from %d evaluation artifacts",
                len(verdicts),
                len(artifacts),
            )
        self._harness_verdict_cache = verdicts
        return verdicts

    @staticmethod
    def _score_session(
        verdict: Optional[HarnessVerdict],
        failure_rate: float,
    ) -> tuple[float, bool, str]:
        """Score one session: harness verdict when it exists, proxy otherwise.

        Returns ``(completion_score, success, score_source)``.
        """
        if verdict is not None:
            return verdict.completion_score, verdict.success, "harness"
        # No verdict exists for interactive sessions — nothing graded them — so
        # the tool-failure proxy stands, but it is labelled as inference.
        return max(0.0, 1.0 - failure_rate * 1.5), failure_rate < 0.3, "tool_failure_proxy"

    @classmethod
    def _scope_traces_to_provider(
        cls,
        traces: List[ExecutionTrace],
        provider: str,
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
        wanted = cls._normalize_provider_label(provider) or provider.lower()
        scoped = [t for t in traces if cls._normalize_provider_label(t.provider) == wanted]
        if len(scoped) < MIN_TRACES_FOR_EVOLUTION:
            logger.info(
                "Provider-scoped traces for '%s' insufficient (%d < %d); "
                "falling back to the unscoped pool (%d traces, mixed provenance).",
                wanted,
                len(scoped),
                MIN_TRACES_FOR_EVOLUTION,
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

    @staticmethod
    def _normalize_provider_label(raw: str) -> str:
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

    @staticmethod
    def _absorb_run_kind(session: Dict[str, Any], event: Dict[str, Any]) -> None:
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

    @classmethod
    def _absorb_session_identity(cls, session: Dict[str, Any], data: Dict[str, Any]) -> None:
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
            provider = cls._normalize_provider_label(data.get("provider", ""))
            if provider:
                session["provider"] = provider
        if not session.get("model"):
            model = str(data.get("model") or "").strip()
            if model:
                session["model"] = model

    @staticmethod
    def _categorize_failure(error: str) -> str:
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

    # ------------------------------------------------------------------
    # GEPA v2: Pareto support
    # ------------------------------------------------------------------

    def _init_pareto_frontiers(self) -> None:
        """Initialize Pareto frontiers from existing candidates."""
        try:
            from victor.framework.rl.pareto import ParetoFrontier
        except ImportError:
            logger.warning("Pareto module not available, disabling Pareto mode")
            self._use_pareto = False
            return

        for key, candidates in self._candidates.items():
            if key not in self._pareto_frontiers:
                self._pareto_frontiers[key] = ParetoFrontier(max_candidates=20)
            frontier = self._pareto_frontiers[key]
            for c in candidates:
                frontier.add_candidate(
                    text_hash=c.text_hash,
                    text=c.text,
                    generation=c.generation,
                    instance_scores=c.instance_scores,
                )
            self._sync_pareto_state(key)

    def get_pareto_frontier(self, section_name: str, provider: str = "default") -> Optional[Any]:
        """Get the provider-scoped Pareto frontier for a section."""
        return self._pareto_frontiers.get(self._candidate_key(section_name, provider))

    def _collect_traces_v2(self, limit: int = 50) -> List[ExecutionTrace]:
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

                            self._absorb_session_identity(sessions[sid], data)
                            self._absorb_run_kind(sessions[sid], event)

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
                                    cat = self._categorize_failure(error)
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
                # now go through _score_session, so a session is graded the same
                # way regardless of which collector observed it.
                failure_rate = sum(data["failures"].values()) / max(data["tool_calls"], 1)
                completion_score, success, score_source = self._score_session(
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

    def _collect_traces_from_conversations(self, limit: int = 50) -> List[ExecutionTrace]:
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
                                cat = self._categorize_failure(msg.content[:300])
                                failures[cat] = failures.get(cat, 0) + 1
            except Exception:
                pass  # Fall back to aggregate-only

            total_tool_calls = max(len(details), int(tool_msg_count), 1)
            total_failures = sum(failures.values())
            failure_rate = total_failures / max(total_tool_calls, 1)

            completion_score, success, score_source = self._score_session(
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

    @staticmethod
    def _merge_traces(
        *trace_lists: List[ExecutionTrace],
    ) -> List[ExecutionTrace]:
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

    def _enrich_traces_with_credit(self, traces: List[ExecutionTrace]) -> None:
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

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward from outcome."""
        return (
            0.4 * (1.0 if outcome.success else 0.0)
            + 0.4 * outcome.quality_score
            + 0.2 * outcome.metadata.get("tool_effectiveness", 0.5)
        )

    def export_metrics(self) -> Dict[str, Any]:
        """Export optimizer metrics."""
        pareto_info = {}
        for section, frontier in self._pareto_frontiers.items():
            pareto_info[section] = {
                "frontier_size": frontier.size,
                "candidates": [
                    {
                        "hash": e.text_hash,
                        "gen": e.generation,
                        "coverage": e.coverage_count,
                        "chars": e.char_length,
                    }
                    for e in frontier.get_frontier()
                ],
            }

        return {
            "total_candidates": sum(len(v) for v in self._candidates.values()),
            "sections": {name: len(candidates) for name, candidates in self._candidates.items()},
            "max_generation": max(
                (
                    max((c.generation for c in cands), default=0)
                    for cands in self._candidates.values()
                ),
                default=0,
            ),
            "use_pareto": self._use_pareto,
            "pareto": pareto_info,
        }

    def export_candidate_rows(self) -> List[Dict[str, Any]]:
        """Export flat candidate rows for status/reporting UIs."""
        rows: List[Dict[str, Any]] = []
        for key, candidates in self._candidates.items():
            for candidate in sorted(candidates, key=lambda c: (-c.generation, c.text_hash)):
                rows.append(
                    {
                        "key": key,
                        "section": candidate.section_name,
                        "provider": candidate.provider,
                        "ordinal": candidate.generation,
                        "parent_hash": candidate.parent_hash,
                        "text_hash": candidate.text_hash,
                        "active": candidate.is_active,
                        "benchmark_passed": candidate.benchmark_passed,
                        "benchmark_runs": candidate.benchmark_runs,
                        "sample_count": candidate.sample_count,
                        "alpha": candidate.alpha,
                        "beta": candidate.beta_val,
                        "mean": candidate.mean,
                        "strategy": candidate.strategy_chain or candidate.strategy_name,
                        "chars": candidate.char_length or len(candidate.text),
                        "preview": (
                            candidate.text[:80] + "..."
                            if len(candidate.text) > 80
                            else candidate.text
                        ),
                    }
                )
        rows.sort(key=lambda row: (row["section"], row["provider"], -row["ordinal"]))
        return rows
