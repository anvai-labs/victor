# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pairwise preference prompt optimization strategy.

This is a surgical, deterministic adaptation of PrefPO for prompt sections
that benefit from criteria-based refinement. It stays offline and additive:
it proposes a challenger prompt, judges it against the current prompt using
trace-derived failure pressure, and only emits a new candidate when the
challenger wins.
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps, json_loads
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from victor.framework.rl.learners.prompt_optimizer import get_failure_hint
from victor.framework.rl.prompt_hygiene import evaluate_prompt_candidate

logger = logging.getLogger(__name__)


JudgeFn = Callable[[str, str, List[Any], str], Tuple[str, str]]
RewriteFn = Callable[[str, str, str], str]
ChallengerFactoryFn = Callable[[str, List[Any], str], str]


class PrefPOStrategy:
    """Deterministic pairwise prompt optimizer for targeted sections."""

    TARGET_SECTIONS = {
        "GROUNDING_RULES",
        "COMPLETION_GUIDANCE",
        "CONCISE_MODE_GUIDANCE",
    }
    requires_benchmark_gate = True

    # Which failure categories each target section may legitimately address.
    #
    # Without this, PrefPO ranked failures globally and appended the hint for
    # whatever failed most — so an output-style section such as
    # CONCISE_MODE_GUIDANCE could receive tool-discipline guidance
    # ("copy old_str exactly") for an ``edit_mismatch`` that has nothing to do
    # with verbosity. The 2026-07-27 FEP-0025 checkpoint recorded exactly this
    # cross-section contamination ("tool-discipline guidance landed in the
    # output-style section"). Scoping additions to the section's own concern
    # keeps a section on-topic and stops it accreting unrelated rules.
    #
    # A section absent from this map is left unscoped (all categories eligible),
    # so custom sections keep the prior behaviour; the three TARGET_SECTIONS
    # that actually reach these helpers are all mapped.
    SECTION_RELEVANT_FAILURES: Dict[str, frozenset] = {
        "GROUNDING_RULES": frozenset(
            {
                "file_not_found",
                "read_directory",
                "permission_denied",
                "edit_mismatch",
                "edit_ambiguous",
                "edit_syntax",
                "tool_not_found",
                "tool_error",
                "search_no_results",
                "shell_error",
                "other",
            }
        ),
        "COMPLETION_GUIDANCE": frozenset(
            {
                "timeout",
                "test_failure",
                "tool_error",
                "shell_error",
                "other",
            }
        ),
        "CONCISE_MODE_GUIDANCE": frozenset({"verbosity"}),
    }

    def _relevant_categories(self, section_name: str) -> Optional[frozenset]:
        """Categories eligible for ``section_name``; None means unscoped."""
        return self.SECTION_RELEVANT_FAILURES.get(section_name)

    def __init__(
        self,
        *,
        max_guidance_items: int = 2,
        min_failure_count: int = 1,
        max_prompt_growth_chars: int = 240,
        challenger_factory: Optional[ChallengerFactoryFn] = None,
        judge: Optional[JudgeFn] = None,
        optimizer: Optional[RewriteFn] = None,
    ):
        self._max_guidance_items = max(1, max_guidance_items)
        self._min_failure_count = max(1, min_failure_count)
        self._max_prompt_growth_chars = max(0, max_prompt_growth_chars)
        self._challenger_factory = challenger_factory or self._build_challenger
        self._judge = judge or self._judge_pair
        self._optimizer = optimizer or self._rewrite_loser

    def reflect(
        self,
        traces: List[Any],
        section_name: str,
        current_text: str,
        **kwargs: Any,
    ) -> str:
        """Return a serialized winning rewrite when the challenger wins."""
        del kwargs
        if not traces or section_name not in self.TARGET_SECTIONS:
            return ""

        challenger_text = self._challenger_factory(current_text, traces, section_name).strip()
        if not challenger_text or challenger_text == current_text.strip():
            return ""

        winner, feedback = self._judge(current_text, challenger_text, traces, section_name)
        if winner != "challenger" or not feedback.strip():
            return ""

        candidate_text = self._optimizer(current_text, feedback, section_name).strip()
        candidate_text = self._cap_prompt_growth(current_text, candidate_text)
        if not candidate_text or candidate_text == current_text.strip():
            return ""
        report = evaluate_prompt_candidate(
            current_text,
            candidate_text,
            allowed_additions=self._dominant_guidance_lines(traces, section_name),
            max_growth_chars=self._max_prompt_growth_chars,
        )
        if not report.accepted:
            # Warning, not info: the strategy chain reads an empty reflection as
            # "nothing to propose", so a rejection here is indistinguishable from
            # having found no improvement unless it says so.
            logger.warning(
                "PrefPO rejected the candidate for %s on hygiene (%s); proposing nothing.",
                section_name,
                ",".join(report.violations),
            )
            return ""

        return json_dumps(
            {
                "winner": winner,
                "feedback": feedback,
                "candidate_text": candidate_text,
            }
        )

    def mutate(self, current_text: str, reflection: str, section_name: str) -> str:
        """Return the optimized candidate text from a reflection payload."""
        del section_name
        if not reflection:
            return current_text
        try:
            payload = json_loads(reflection)
        except Exception:
            logger.debug("PrefPO: invalid reflection payload")
            return current_text

        candidate_text = str(payload.get("candidate_text", "")).strip()
        return candidate_text or current_text

    def _build_challenger(self, current_text: str, traces: List[Any], section_name: str) -> str:
        """Propose a minimally edited challenger from dominant failures."""
        guidance_lines = self._guidance_lines(traces, current_text, section_name)
        if not guidance_lines:
            return current_text
        base = current_text.rstrip()
        suffix = "\n" if base else ""
        return f"{base}{suffix}" + "\n".join(guidance_lines)

    def _judge_pair(
        self,
        current_text: str,
        challenger_text: str,
        traces: List[Any],
        section_name: str,
    ) -> Tuple[str, str]:
        """Prefer the prompt that better covers dominant failures with less bloat."""
        failure_counts = self._failure_counts(traces)
        dominant_lines = self._dominant_guidance_lines(traces, section_name)
        current_lines = dominant_lines
        challenger_lines = dominant_lines

        current_score = self._score_prompt(current_text, current_lines, failure_counts)
        challenger_score = self._score_prompt(challenger_text, challenger_lines, failure_counts)

        if challenger_score <= current_score:
            return ("current", "Existing prompt already covers dominant failures.")

        feedback_lines = [
            line for line in dominant_lines if line.strip() and line.strip() not in current_text
        ]
        feedback = "Prefer challenger because it adds:\n" + "\n".join(feedback_lines)
        return ("challenger", feedback)

    def _rewrite_loser(self, losing_text: str, feedback: str, section_name: str) -> str:
        """Rewrite the losing prompt by merging the judge's missing guidance."""
        del section_name
        additions = [
            line.strip() for line in feedback.splitlines() if line.strip().startswith("- ")
        ]
        if not additions:
            return losing_text

        # Only append. Rebuilding the text from non-blank lines discarded every
        # blank line in the section as a side effect of the join — and when all
        # additions were already present it discarded them for no gain at all,
        # producing a "new candidate" whose entire diff was six removed blank
        # lines. COMPLETION_GUIDANCE went 1551 -> 1545 chars with zero semantic
        # change, three generations running, on two different models. Merging
        # guidance is not a licence to reformat the prompt.
        merged = losing_text.rstrip()
        new_additions = [addition for addition in additions if addition not in merged]
        if not new_additions:
            return losing_text
        for addition in new_additions:
            merged = f"{merged}\n{addition}" if merged else addition
        return merged

    def _guidance_lines(self, traces: List[Any], current_text: str, section_name: str) -> List[str]:
        """Return the top missing guidance lines for the dominant failures."""
        current_lower = current_text.lower()
        guidance_lines = [
            line
            for line in self._dominant_guidance_lines(traces, section_name)
            if line[2:].strip().lower() not in current_lower
        ]
        return guidance_lines[: self._max_guidance_items]

    def _dominant_guidance_lines(self, traces: List[Any], section_name: str) -> List[str]:
        """Return guidance lines for the highest-pressure failures."""
        guidance_lines = []
        for category, _count in self._dominant_failures(traces, section_name):
            hint = get_failure_hint(category).strip()
            if not hint:
                continue
            first_sentence = hint.split(". ")[0].strip()
            if not first_sentence.endswith("."):
                first_sentence += "."
            guidance_lines.append(f"- {first_sentence}")
        return guidance_lines[: self._max_guidance_items]

    def _dominant_failures(self, traces: List[Any], section_name: str) -> List[Tuple[str, int]]:
        """Return the highest-pressure failure categories relevant to the section.

        Categories outside the section's concern are dropped before ranking, so
        a section only ever accretes guidance about the failures it can address
        (see ``SECTION_RELEVANT_FAILURES``).
        """
        counts = self._failure_counts(traces)
        relevant = self._relevant_categories(section_name)
        if relevant is not None:
            counts = {cat: n for cat, n in counts.items() if cat in relevant}
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [item for item in ranked if item[1] >= self._min_failure_count][
            : self._max_guidance_items
        ]

    @staticmethod
    def _failure_counts(traces: List[Any]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for trace in traces:
            for category, count in getattr(trace, "tool_failures", {}).items():
                counts[category] = counts.get(category, 0) + int(count)
        return counts

    @staticmethod
    def _score_prompt(
        prompt_text: str,
        guidance_lines: List[str],
        failure_counts: Dict[str, int],
    ) -> float:
        """Heuristic judge score balancing guidance coverage and prompt growth."""
        prompt_lower = prompt_text.lower()
        coverage = 0.0
        for line in guidance_lines:
            content = line[2:].strip().lower()
            if content and content in prompt_lower:
                coverage += 1.0

        pressure = sum(failure_counts.values()) or 1
        density_bonus = coverage * 10.0
        length_penalty = len(prompt_text) / max(pressure * 15.0, 1.0)
        return density_bonus - length_penalty

    def _cap_prompt_growth(self, current_text: str, candidate_text: str) -> str:
        """Enforce minimal-change growth budget."""
        if not candidate_text or self._max_prompt_growth_chars <= 0:
            return candidate_text

        max_length = len(current_text) + self._max_prompt_growth_chars
        if len(candidate_text) <= max_length:
            return candidate_text
        return candidate_text[:max_length].rstrip()
