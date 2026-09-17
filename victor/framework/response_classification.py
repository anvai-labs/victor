# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Pure response-text classifiers shared by the agentic loop and evaluators.

Single source of truth for "what kind of text is this" decisions used to
accept/reject model turns: continuation requests, intent-only narration,
refusals, and future-intent plans. AgenticLoop and
EnhancedCompletionEvaluator previously carried hand-synced copies of these
(docstrings admitted it) — consume the functions here instead of re-forking.

Marker-tuple overlap note: ``_INTENT_PREFIXES`` and ``_FUTURE_INTENT_MARKERS``
share five phrases, deliberately NOT merged — they serve different
quantifiers (first-line ``startswith`` vs substring in the first 300 chars).
Unioning them would change behavior (e.g. "Let's …" as an intent-only
*prefix* would newly reject post-tool synthesis that opens with "Let's").
"""

from __future__ import annotations

# Phrases where the model declines/aborts the task itself (not findings like
# "I cannot find any bugs"). Kept tight to avoid misclassifying real answers.
_REFUSAL_MARKERS = (
    "i can't read",
    "i cannot read",
    "i can't access",
    "i cannot access",
    "unable to read",
    "unable to access",
    "i'm unable to",
    "i am unable to",
    "cannot comply",
    "can't comply",
    "the information needed",  # "...don't have the information needed..."
    "i can't provide a grounded",
    "can't give a grounded",
    "i'm sorry, but i can't",
    "i am sorry, but i can't",
)

# First-person future-work openers. Only consulted before any tool usage:
# after tools have run, restating next steps is legitimate synthesis.
_FUTURE_INTENT_MARKERS = (
    "i'll ",
    "i will ",
    "i'm going to ",
    "i am going to ",
    "let me ",
    "let's ",
    "first, i ",
    "i plan to ",
    "i'm about to ",
    "i need to first",
    "i need to create",
)

_CONTINUATION_PATTERNS = (
    "would you like me to",
    "should i continue",
    "do you want me to",
    "shall i proceed",
    "let me know if you'd like",
    "would you prefer i",
)

_INTENT_PREFIXES = (
    "i'll now ",
    "i'll ",
    "i will now ",
    "i will ",
    "let me now ",
    "let me ",
    "now i'll ",
    "now i will ",
    "i'm going to ",
    "i am going to ",
    "i'm now ",
    "i am now ",
    "next, i'll ",
    "next i'll ",
)

_DELIBERATION_MARKERS = (
    "executing now",
    "executing.",
    "going now",
    "going.",
    "calling now",
    "calling.",
    "running now",
    "running.",
    "making the call",
    "making the request",
    "let me make the call",
    "no more deliberation",
    "stop the meta-deliberation",
    "stop deliberating",
    "done deliberating",
    "just execute",
    "executing the",
    "polling",
    "no sleep",
    "pure status read",
    "going. (",
    "done. (",
    "final. (",
    "(no sleep)",
    "(no more deliberation)",
    "(will act on results",
    "(finally.)",
    "(stop. calling.)",
)


def is_continuation_request(response: str) -> bool:
    """True when the response asks the user for continuation direction."""
    if not response:
        return False
    response_lower = response.lower()
    return any(pattern in response_lower for pattern in _CONTINUATION_PATTERNS)


def is_intent_only_response(response: str) -> bool:
    """True when the response is pure future-intent narration.

    Phrases like "I'll now read…" or "Let me analyze…" describe planned
    actions rather than completed work. Treating them as final answers causes
    the loop to exit before any tools are actually invoked.

    Two checks:
      1. First-line prefix check so responses that start with intent but
         contain substantive findings are still allowed through.
      2. Meta-deliberation density across the FULL response (≥3 distinct
         imminent-action markers), gated on the absence of a substantive
         payload (no code fences, no markdown table).
    """
    if not response:
        return False
    first_line = response.strip().split("\n")[0].strip().lower()
    if any(first_line.startswith(p) for p in _INTENT_PREFIXES):
        return True

    # Real findings usually carry a payload (a fenced code block or a
    # tool-result-style table). Narration-only responses do not, so the
    # density signal is gated on the absence of such payloads.
    if "```" in response:
        return False
    lowered = response.lower()
    if lowered.count("|") >= 3 and "---" in lowered:
        return False  # Markdown table — looks like a result dump, not narration

    marker_hits = sum(1 for m in _DELIBERATION_MARKERS if m in lowered)
    return marker_hits >= 3


def is_refusal_response(content: str) -> bool:
    """True when the final answer declines/aborts the task (a non-answer)."""
    if not content:
        return False
    lowered = content.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def is_future_intent_narration(content: str) -> bool:
    """True when the response opens as a plan for future work (pre-tool turns)."""
    head = content[:300].lower()
    return any(marker in head for marker in _FUTURE_INTENT_MARKERS)
