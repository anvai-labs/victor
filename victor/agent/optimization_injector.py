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

"""Optimization injector for prompt composition.

Consolidates all prompt optimization strategy outputs (GEPA, MIPROv2, CoT,
failure hints) into a single interface consumed by PromptComposer.

Key capability: Real-time failure hint injection. After a tool rollback,
the injector maps the error to one of 13 GEPA failure categories and
provides corrective guidance in the NEXT turn's user prefix.

Research basis:
- arXiv:2507.19457 — GEPA Pareto frontier + Thompson Sampling (ICLR 2026)
- arXiv:2406.11695 — MIPROv2 KNN few-shot demonstration mining
- arXiv:2601.08884 — Structured failure taxonomy with corrective hints
- arXiv:2604.07645 — PRIME semantic trace zones (SUCCESS/FAILURE/RECOVERY)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Sentinel hash for binding a section's SEED (baseline) prompt in a candidate
# suite — lets ``run-prompt-suite`` measure evolved-vs-seed without storing the
# seed as a real candidate (no Thompson pollution). Resolved to the registry
# default text by ``_resolve_bound_candidate_payload``.
BASELINE_CANDIDATE_HASH = "__baseline__"

_DEFAULT_TURN_PREFIX_EVOLVABLE_SECTIONS = (
    "ASI_TOOL_EFFECTIVENESS_GUIDANCE",
    "GROUNDING_RULES",
    "COMPLETION_GUIDANCE",
)

# Failure category → corrective hint mapping (from GEPA arXiv:2601.08884).
# These are injected into user messages in real-time after tool failures,
# not just during offline evolution.
FAILURE_HINTS: Dict[str, str] = {
    "file_not_found": (
        "The file was not found. Use ls() to check the directory, "
        "or code_search(query='...') to find the correct path."
    ),
    "read_directory": (
        "You tried to read a directory. Use ls() for directories, read() for files."
    ),
    "permission_denied": ("Permission denied. Check file permissions or use a different path."),
    "edit_mismatch": (
        "Your edit failed because old_str did not match the file content. "
        "RE-READ the file at the exact location and COPY the text "
        "character-by-character from the read output. Do NOT type from memory."
    ),
    "edit_ambiguous": (
        "Your edit matched multiple locations. Include 3+ surrounding lines "
        "of context in old_str to make the match unique."
    ),
    "edit_syntax": (
        "The edit produced invalid syntax. Check indentation and ensure "
        "new_str preserves correct Python/language syntax."
    ),
    "tool_not_found": (
        "Tool not found. Use only tools listed in the available set. "
        "Check spelling. Use code_search or read as fallbacks."
    ),
    "timeout": (
        "The operation timed out. Use more targeted searches and avoid "
        "reading entire large directories."
    ),
    "tool_error": (
        "Tool call error. Check the arguments match the expected schema "
        "and review the error message before retrying."
    ),
    "search_no_results": (
        "Search returned no results. Broaden the query, try alternative "
        "keywords or partial names, or use ls() to browse manually."
    ),
    "shell_error": (
        "Shell command failed. Check syntax and ensure required tools "
        "are available. Use absolute paths for reliability."
    ),
    "test_failure": (
        "Tests failed. Read the output carefully, identify which assertion "
        "failed and why, then fix the root cause."
    ),
    "other": (
        "An error occurred. Read the error message carefully and diagnose "
        "the root cause before retrying."
    ),
}

# Patterns for categorizing failures from error messages.
_FAILURE_PATTERNS: List[tuple] = [
    ("edit_mismatch", r"old_str not found|transaction rolled back|Failed to commit"),
    ("edit_ambiguous", r"ambiguous|multiple matches|appears \d+ times"),
    ("file_not_found", r"file not found|no such file|FileNotFoundError"),
    ("read_directory", r"is a directory|IsADirectoryError"),
    ("permission_denied", r"permission denied|PermissionError"),
    ("edit_syntax", r"syntax error|SyntaxError|IndentationError"),
    ("tool_not_found", r"tool.*not found|unknown tool"),
    ("timeout", r"timed? ?out|TimeoutError"),
    ("search_no_results", r"no results|0 results|no matches"),
    ("shell_error", r"command not found|exit code [1-9]"),
    ("test_failure", r"FAILED|AssertionError|test.*fail"),
    ("tool_error", r"missing required|invalid argument|parameter"),
]


def categorize_failure(error: str) -> str:
    """Categorize a failure error message into one of 13 GEPA categories.

    Args:
        error: The error message string from a failed tool call.

    Returns:
        Failure category string (e.g., "edit_mismatch", "file_not_found").
    """
    if not error:
        return "other"
    error_lower = error.lower()
    for category, pattern in _FAILURE_PATTERNS:
        if re.search(pattern, error_lower, re.IGNORECASE):
            return category
    return "other"


class OptimizationInjector:
    """Consolidates all prompt optimization strategy outputs.

    Provides a single interface for PromptComposer to collect:
    - GEPA-evolved sections (per-session, Thompson Sampling)
    - MIPROv2 few-shot examples (per-query, KNN)
    - CoT distillation hints (per-session)
    - Real-time failure hints (per-turn, after errors)

    Usage:
        injector = OptimizationInjector()
        evolved = injector.get_evolved_sections("deepseek", "deepseek-chat", "edit")
        hint = injector.get_failure_hint("edit_mismatch", "old_str not found")
    """

    def __init__(self) -> None:
        self._section_cache: Dict[Tuple[str, str, str, str], Optional[str]] = {}
        self._section_payload_cache: Dict[Tuple[str, str, str, str], Optional[Dict[str, Any]]] = {}
        self._few_shot_cache: Dict[Tuple[str, str, str, str], Optional[str]] = {}
        self._few_shot_payload_cache: Dict[Tuple[str, str, str, str], Optional[Dict[str, Any]]] = {}
        self._bound_candidates: Dict[str, Dict[str, Any]] = {}
        self._last_failure_category: Optional[str] = None
        self._last_failure_error: Optional[str] = None

    @staticmethod
    def _section_cache_key(
        section_name: str,
        provider: str,
        model: str,
        task_type: str,
    ) -> Tuple[str, str, str, str]:
        """Build a cache key scoped to section + prompt identity."""
        return (
            str(section_name or "").strip(),
            str(provider or "").strip(),
            str(model or "").strip(),
            str(task_type or "default").strip() or "default",
        )

    @staticmethod
    def _few_shot_cache_key(
        query: str,
        provider: str,
        model: str,
        task_type: str,
    ) -> Tuple[str, str, str, str]:
        """Build a cache key scoped to query + prompt identity."""
        normalized_query = (query or "").strip() or "__empty_query__"
        return (
            normalized_query,
            str(provider or "").strip(),
            str(model or "").strip(),
            str(task_type or "default").strip() or "default",
        )

    def clear_session_cache(self) -> None:
        """Clear per-session cache (called on workspace switch)."""
        self._section_cache.clear()
        self._section_payload_cache.clear()
        self._few_shot_cache.clear()
        self._few_shot_payload_cache.clear()

    def clear_prompt_bindings(self) -> None:
        """Remove explicit candidate bindings and clear any cached prompt payloads."""
        self._bound_candidates.clear()
        self.clear_session_cache()

    @staticmethod
    def _get_turn_prefix_section_names() -> List[str]:
        """Return the canonical turn-prefix evolvable sections.

        The shared prompt section registry is the source of truth. We keep a
        legacy fallback to preserve bootstrap behavior if the registry cannot be
        initialized for any reason.
        """
        try:
            from victor.agent.prompt_section_registry import (
                get_required_evolvable_section_names,
            )

            section_names = get_required_evolvable_section_names()
            if section_names:
                return list(section_names)
        except Exception:
            logger.debug(
                "Falling back to legacy turn-prefix evolvable section bundle",
                exc_info=True,
            )
        return list(_DEFAULT_TURN_PREFIX_EVOLVABLE_SECTIONS)

    def bind_prompt_candidate(
        self,
        *,
        section_name: str,
        prompt_candidate_hash: str,
        provider: Optional[str] = None,
        strict: bool = True,
    ) -> None:
        """Bind one exact prompt candidate for targeted evaluation runs.

        Bound candidates bypass Thompson sampling and active-candidate selection.
        This keeps the runtime prompt content aligned with benchmark metadata.
        """
        normalized_section = str(section_name or "").strip()
        normalized_hash = str(prompt_candidate_hash or "").strip()
        normalized_provider = str(provider or "").strip()

        if not normalized_section:
            raise ValueError("section_name is required for prompt candidate binding")
        if not normalized_hash:
            raise ValueError("prompt_candidate_hash is required for prompt candidate binding")

        self._bound_candidates[normalized_section] = {
            "prompt_candidate_hash": normalized_hash,
            "provider": normalized_provider,
            "strict": bool(strict),
        }
        self.clear_session_cache()
        if strict and normalized_provider:
            self._resolve_bound_candidate_payload(normalized_section, normalized_provider)

    def get_evolved_section_payloads(
        self,
        provider: str = "",
        model: str = "",
        task_type: str = "default",
    ) -> List[Dict[str, Any]]:
        """Get evolved sections plus canonical prompt-identity metadata."""
        results: List[Dict[str, Any]] = []

        for section_name in self._get_turn_prefix_section_names():
            payload = self.get_section_payload(section_name, provider, model, task_type)
            if payload is None and section_name == "ASI_TOOL_EFFECTIVENESS_GUIDANCE":
                payload = self._build_static_fallback_payload(section_name, provider)

            if payload and payload.get("text"):
                results.append(dict(payload))

        if results:
            evolved_count = sum(1 for payload in results if payload.get("prompt_candidate_hash"))
            if evolved_count > 0:
                logger.info(
                    "[OptimizationInjector] Serving %d evolved + %d static sections for %s/%s",
                    evolved_count,
                    len(results) - evolved_count,
                    provider or "default",
                    model or "default",
                )

        return results

    @staticmethod
    def _build_static_fallback_payload(
        section_name: str,
        provider: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Build one registry-backed static fallback payload for a prompt section."""
        try:
            from victor.agent.prompt_section_registry import build_fallback_map

            fallback_text = build_fallback_map([section_name]).get(section_name)
        except Exception:
            logger.debug(
                "Failed to resolve static fallback payload for %s",
                section_name,
                exc_info=True,
            )
            fallback_text = None

        if not fallback_text:
            return None

        return {
            "text": fallback_text,
            "provider": provider or "",
            "prompt_candidate_hash": None,
            "section_name": section_name,
            "prompt_section_name": section_name,
            "strategy_name": None,
            "strategy_chain": None,
            "source": "static_fallback",
        }

    def get_section_payload(
        self,
        section_name: str,
        provider: str = "",
        model: str = "",
        task_type: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Resolve one prompt section without widening the turn-prefix bundle.

        This is used by scoped prompt builders such as init synthesis or
        optional system-prompt sections that should evolve only where they
        are actually rendered.
        """
        normalized_section = str(section_name or "").strip()
        if not normalized_section:
            return None

        cache_key = self._section_cache_key(normalized_section, provider, model, task_type)
        if cache_key in self._section_payload_cache:
            cached = self._section_payload_cache[cache_key]
            return dict(cached) if cached else None

        payload = self._sample_evolved_payload(normalized_section, provider, model, task_type)
        self._section_payload_cache[cache_key] = payload
        return dict(payload) if payload else None

    def get_evolved_sections(
        self,
        provider: str = "",
        model: str = "",
        task_type: str = "default",
    ) -> List[str]:
        """Get GEPA-evolved sections for user prefix injection.

        Returns evolved versions when confidence > threshold,
        static fallback for ASI_TOOL_EFFECTIVENESS otherwise.

        Args:
            provider: Provider name for per-provider evolution.
            model: Model name.
            task_type: Task type for context.

        Returns:
            List of evolved section texts to include in user prefix.
        """
        payloads = self.get_evolved_section_payloads(provider, model, task_type)
        return [str(payload.get("text", "")).strip() for payload in payloads if payload.get("text")]

    def get_few_shot_payload(
        self,
        query: str,
        provider: str = "",
        model: str = "",
        task_type: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Get MIPROv2 few-shot examples plus canonical prompt-identity metadata.

        Unlike evolved sections (per-session), few-shots can be
        per-query to match the current task context.

        Args:
            query: Current user message for KNN similarity matching.

        Returns:
            Few-shot payload dictionary or None.
        """
        normalized_query = (query or "").strip()
        cache_key = self._few_shot_cache_key(normalized_query, provider, model, task_type)
        if cache_key in self._few_shot_payload_cache:
            cached_payload = self._few_shot_payload_cache[cache_key]
            return dict(cached_payload) if cached_payload else None

        if "FEW_SHOT_EXAMPLES" in self._bound_candidates:
            payload = self._sample_evolved_payload("FEW_SHOT_EXAMPLES", provider, model, task_type)
            self._few_shot_payload_cache[cache_key] = payload
            self._few_shot_cache[cache_key] = (
                str(payload.get("text")).strip() if payload and payload.get("text") else None
            )
            return dict(payload) if payload else None

        try:
            from victor.config.settings import get_settings

            po = getattr(get_settings(), "prompt_optimization", None)
            if po is not None:
                strategies = po.get_strategies_for_section("FEW_SHOT_EXAMPLES")
                if not strategies:
                    self._few_shot_cache[cache_key] = None
                    self._few_shot_payload_cache[cache_key] = None
                    return None
        except Exception:
            pass

        try:
            from victor.agent.services.rl_runtime import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("prompt_optimizer")
            if learner is not None and hasattr(learner, "get_query_aware_few_shots"):
                few_shots = learner.get_query_aware_few_shots(normalized_query)
                if few_shots:
                    payload = {
                        "text": few_shots,
                        "provider": provider or "",
                        "prompt_candidate_hash": None,
                        "section_name": "FEW_SHOT_EXAMPLES",
                        "prompt_section_name": "FEW_SHOT_EXAMPLES",
                        "strategy_name": "miprov2",
                        "strategy_chain": "miprov2",
                        "source": "query_aware_strategy",
                    }
                    self._few_shot_cache[cache_key] = few_shots
                    self._few_shot_payload_cache[cache_key] = payload
                    return dict(payload)
        except Exception:
            pass

        payload = self._sample_evolved_payload("FEW_SHOT_EXAMPLES", provider, model, task_type)
        self._few_shot_payload_cache[cache_key] = payload
        self._few_shot_cache[cache_key] = (
            str(payload.get("text")).strip() if payload and payload.get("text") else None
        )
        return dict(payload) if payload else None

    def get_few_shots(
        self,
        query: str,
        provider: str = "",
        model: str = "",
        task_type: str = "default",
    ) -> Optional[str]:
        """Get MIPROv2 KNN-selected few-shot examples."""
        payload = self.get_few_shot_payload(
            query,
            provider=provider,
            model=model,
            task_type=task_type,
        )
        if not payload:
            return None
        text = str(payload.get("text", "")).strip()
        return text or None

    def get_failure_hint(
        self,
        category: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Optional[str]:
        """Get a corrective hint for a specific failure category.

        Called by PromptComposer after a tool failure to inject
        actionable guidance into the NEXT turn's user prefix.

        Args:
            category: Pre-categorized failure type (e.g., "edit_mismatch").
            error: Raw error message (will be categorized if category is None).

        Returns:
            Corrective hint string or None.
        """
        if not category and error:
            category = categorize_failure(error)
        if not category:
            return None

        hint = FAILURE_HINTS.get(category)
        if not hint:
            return None

        return f"PREVIOUS ERROR: {hint}"

    def record_failure(self, tool_name: str, error: str) -> str:
        """Record a tool failure for hint injection on next turn.

        Called by the execution coordinator when a tool call fails.

        Args:
            tool_name: Name of the failed tool.
            error: Error message.

        Returns:
            The categorized failure type.
        """
        category = categorize_failure(error)
        self._last_failure_category = category
        self._last_failure_error = error
        logger.info(
            "[OptimizationInjector] Recorded failure: tool=%s category=%s",
            tool_name,
            category,
        )
        return category

    def _sample_evolved(
        self,
        section_name: str,
        provider: str,
        model: str,
        task_type: str,
    ) -> Optional[str]:
        """Sample an evolved section from GEPA via Thompson Sampling.

        Gated by prompt_optimization.enabled setting. Returns None if
        no candidates exist or confidence is below threshold.
        """
        payload = self._sample_evolved_payload(section_name, provider, model, task_type)
        if not payload:
            return None
        text = str(payload.get("text", "")).strip()
        return text or None

    def _sample_evolved_payload(
        self,
        section_name: str,
        provider: str,
        model: str,
        task_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Sample an evolved section and preserve candidate identity when present."""
        bound_payload = self._resolve_bound_candidate_payload(section_name, provider or "")
        if bound_payload is not None:
            return bound_payload

        try:
            from victor.config.settings import get_settings

            po = getattr(get_settings(), "prompt_optimization", None)
            if po is None or not po.enabled:
                return None
            strategies = po.get_strategies_for_section(section_name)
            if not strategies:
                return None
        except Exception:
            return None

        try:
            from victor.agent.services.rl_runtime import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("prompt_optimizer")
            if learner is None:
                return None

            rec = learner.get_recommendation(
                provider or "",
                model or "",
                task_type,
                section_name=section_name,
            )
            from victor.framework.rl.learners.prompt_optimizer import (
                should_serve_candidate,
            )

            if should_serve_candidate(
                rec,
                exploration_enabled=getattr(po, "exploration_enabled", True),
                exploration_epsilon=getattr(po, "exploration_epsilon", 0.1),
            ):
                rec_metadata = dict(getattr(rec, "metadata", {}) or {})
                resolved_provider = str(rec_metadata.get("provider") or provider or "")
                resolved_section = str(
                    rec_metadata.get("section_name")
                    or rec_metadata.get("prompt_section_name")
                    or section_name
                )
                learner.record_served(
                    resolved_section,
                    resolved_provider,
                    str(rec_metadata.get("prompt_candidate_hash") or ""),
                )
                logger.info(
                    "[OptimizationInjector] Using evolved '%s' (gen=%s, conf=%.2f)",
                    section_name,
                    rec.reason,
                    rec.confidence,
                )
                return {
                    "text": rec.value,
                    "provider": resolved_provider,
                    "prompt_candidate_hash": rec_metadata.get("prompt_candidate_hash"),
                    "section_name": resolved_section,
                    "prompt_section_name": resolved_section,
                    "strategy_name": rec_metadata.get("strategy_name"),
                    "strategy_chain": rec_metadata.get("strategy_chain"),
                    "source": "candidate",
                }
        except Exception:
            pass

        return None

    def _resolve_bound_candidate_payload(
        self,
        section_name: str,
        provider: str,
    ) -> Optional[Dict[str, Any]]:
        """Resolve a bound candidate into the canonical prompt payload shape."""
        binding = self._bound_candidates.get(section_name)
        if not binding:
            return None

        resolved_provider = str(binding.get("provider") or provider or "").strip()
        prompt_candidate_hash = str(binding.get("prompt_candidate_hash") or "").strip()
        strict = bool(binding.get("strict", True))

        # Baseline sentinel: serve the section's seed text (no stored candidate,
        # no Thompson pollution) so a suite can measure evolved-vs-seed.
        if prompt_candidate_hash == BASELINE_CANDIDATE_HASH:
            seed = self._section_seed_text(section_name)
            if seed is None:
                if strict:
                    raise ValueError(f"baseline seed not found for section {section_name}")
                return None
            logger.info(
                "[OptimizationInjector] Using bound BASELINE (seed) for %s",
                section_name,
            )
            return {
                "text": seed,
                "provider": resolved_provider,
                "prompt_candidate_hash": BASELINE_CANDIDATE_HASH,
                "section_name": section_name,
                "prompt_section_name": section_name,
                "strategy_name": "baseline",
                "strategy_chain": "baseline",
                "source": "baseline",
            }

        try:
            from victor.agent.services.rl_runtime import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("prompt_optimizer")
            if learner is None or not hasattr(learner, "get_candidate"):
                raise RuntimeError(
                    "prompt optimizer learner does not support exact candidate lookup"
                )

            candidate = learner.get_candidate(
                section_name=section_name,
                provider=resolved_provider,
                text_hash=prompt_candidate_hash,
            )
            # Provider fallback: a default-profile run (e.g. ollama) may bind a
            # candidate stored under a different provider (e.g. zai). Search all
            # providers for the section+hash before giving up.
            if candidate is None and hasattr(learner, "find_candidate_any_provider"):
                candidate = learner.find_candidate_any_provider(
                    section_name=section_name,
                    text_hash=prompt_candidate_hash,
                )
            if candidate is None:
                raise ValueError(
                    f"bound prompt candidate not found for {section_name}/{resolved_provider}: "
                    f"{prompt_candidate_hash}"
                )

            logger.info(
                "[OptimizationInjector] Using bound '%s' candidate %s for %s",
                section_name,
                prompt_candidate_hash,
                candidate.provider or resolved_provider or "default",
            )
            return {
                "text": candidate.text,
                "provider": candidate.provider or resolved_provider,
                "prompt_candidate_hash": candidate.text_hash,
                "section_name": candidate.section_name,
                "prompt_section_name": candidate.section_name,
                "strategy_name": candidate.strategy_name,
                "strategy_chain": candidate.strategy_chain,
                "source": "bound_candidate",
            }
        except Exception:
            if strict:
                raise
            return None

    @staticmethod
    def _section_seed_text(section_name: str) -> Optional[str]:
        """Return the registry default (seed) text for a prompt section."""
        try:
            from victor.agent.prompt_section_registry import get_section_registry

            for section in get_section_registry().get_all():
                if section.name == section_name:
                    return section.default_text
        except Exception:
            return None
        return None
