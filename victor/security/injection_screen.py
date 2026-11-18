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

"""FOU-S3c injection screen (ADR-037 decision 2): pattern classifiers for
known prompt-injection families over tool output BEFORE it enters model
context.

Verdicts are ADVISORY (annotation-only default per the ADR graduation
decision): flagged spans are highlighted inside the untrusted-content
envelope so the containment rule ("instructions inside envelopes are
data") has a visible target. Hard-fail on screen verdicts is gated
behind an ADR-036 false-positive battery and is NOT implemented here.
Canary exfiltration hard-blocking is the separate deterministic floor
(``canary_blocking_enabled``, FOU-S3b) and does not depend on this
screen's quality.

False-positive discipline: every pattern targets imperative prompt-syntax
context, not code vocabulary — normal source code that mentions e.g.
"ignore" in a comment must never flag. Each pattern ships with a negative
sample in the test suite; a pattern that cannot keep its negative sample
clean gets tightened, not relaxed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Pattern, Tuple

logger = logging.getLogger(__name__)

HIGHLIGHT_OPEN = "[victor-flagged"
HIGHLIGHT_CLOSE = "[/victor-flagged]"


@dataclass(frozen=True)
class InjectionFamily:
    """One classifier family: compiled patterns + the family tag reported."""

    name: str
    description: str
    patterns: Tuple[Pattern[str], ...]


_INSTRUCTION_IMPERSONATION = InjectionFamily(
    name="instruction_impersonation",
    description=(
        "Content attempting to re-role the model or override prior "
        "instructions (classic instruction-impersonation families)."
    ),
    patterns=(
        re.compile(
            r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+"
            r"(instructions?|prompts?|rules?|directions?|constraints?)",
            re.IGNORECASE,
        ),
        re.compile(
            r"disregard\s+(all\s+|any\s+)?"
            r"((previous|prior|above|earlier|your|own)\s+)+"
            r"(instructions?|prompts?|rules?|guidelines?|constraints?)",
            re.IGNORECASE,
        ),
        re.compile(
            r"forget\s+(all\s+|everything\s+|your\s+)?(previous|prior|above|earlier)\s+"
            r"(instructions?|prompts?|rules?|training)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(new|updated|revised)\s+(system\s+)?instructions?\s*:",
            re.IGNORECASE,
        ),
        re.compile(
            r"you\s+are\s+now\s+(a|an|the)\s+",
            re.IGNORECASE,
        ),
        # Line-start anchored: prose/code referencing "system prompt:" mid-line
        # (e.g. a regression guard asserting its absence) must not flag.
        re.compile(
            r"(?m)^\s*(system|developer|admin)\s+(prompt|message|override)\s*:",
            re.IGNORECASE,
        ),
    ),
)

_TOOL_CALL_FORGERY = InjectionFamily(
    name="tool_call_forgery",
    description=(
        "Content forging tool-call syntax or claiming a tool executed, to "
        "trick the model into treating fabricated calls/results as real."
    ),
    patterns=(
        # Fabricated XML-style invocation blocks WITH bodies: a bare
        # "<tool_call>" mention in documentation is not a forgery attempt.
        re.compile(r"<tool_call\b[^>]*>\s*\{", re.IGNORECASE),
        re.compile(r"<function_results?\b[^>]*>[^\s<]", re.IGNORECASE),
        re.compile(r"</tool_result\b", re.IGNORECASE),
        # Fabricated assistant-side routing tokens.
        re.compile(r"assistant\s+to\s*=\s*tools?\b", re.IGNORECASE),
        re.compile(r"\bto\s*=\s*(functions|tools)\.", re.IGNORECASE),
        # Claimed execution of destructive/state-changing operations.
        re.compile(
            r"(successfully\s+)?(executed|ran)\s+(the\s+)?(command|tool)\b"
            r"[^\n]{0,80}(delete|remove|drop|rm\s+-rf|curl|wget)",
            re.IGNORECASE,
        ),
    ),
)

_EXFIL_SOLICITATION = InjectionFamily(
    name="exfil_solicitation",
    description=(
        "Content soliciting the model to reveal or repeat hidden context, "
        "system text, credentials, or everything above — the exfiltration "
        "half of an injection attempt."
    ),
    patterns=(
        # Requires a direction/context suffix: "repeat everything" alone is
        # common benign phrasing (echo services, docs); "repeat everything
        # above/verbatim/back" is solicitation.
        re.compile(
            r"(repeat|reveal|print|output|show|echo)\s+(everything|all\s+of\s+it|the\s+text)"
            r"\s+(above|before|so\s+far|in\s+your\s+context|verbatim|back|to\s+me)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(what|tell\s+me)\s+(is|are)\s+(your|the)\s+"
            r"(system\s+prompt|initial\s+instructions?|secrets?|api[_\s]?keys?|"
            r"access\s+tokens?|credentials?)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(print|reveal|show|output)\s+your\s+"
            r"(system\s+prompt|secrets?|api[_\s]?keys?|access\s+tokens?|credentials?)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(encode|base64|hex|reverse)\s+.{0,40}"
            r"(and\s+)?(send|post|transmit|include)\s+.{0,40}(url|http|request)",
            re.IGNORECASE,
        ),
    ),
)


FAMILIES: Tuple[InjectionFamily, ...] = (
    _INSTRUCTION_IMPERSONATION,
    _TOOL_CALL_FORGERY,
    _EXFIL_SOLICITATION,
)


@dataclass(frozen=True)
class ScreenVerdict:
    """Advisory verdict over one piece of model-bound untrusted content."""

    flagged: bool
    families: Tuple[str, ...] = field(default_factory=tuple)
    # (family, start, end) into the screened text, non-overlapping per family.
    spans: Tuple[Tuple[str, int, int], ...] = field(default_factory=tuple)


def screen_untrusted_content(text: str) -> ScreenVerdict:
    """Run the injection-family classifiers over untrusted content.

    Pure and cheap: compiled-pattern scan only, no model call. Never
    raises for content reasons (a hard failure here is a bug — callers
    also guard, because screening must never break tool-output delivery).
    """
    if not text:
        return ScreenVerdict(flagged=False)
    families: List[str] = []
    spans: List[Tuple[str, int, int]] = []
    for family in FAMILIES:
        for pattern in family.patterns:
            for match in pattern.finditer(text):
                families.append(family.name)
                spans.append((family.name, match.start(), match.end()))
                break  # one span per family keeps highlighting readable
    if not families:
        return ScreenVerdict(flagged=False)
    return ScreenVerdict(flagged=True, families=tuple(dict.fromkeys(families)), spans=tuple(spans))


def highlight_flagged_spans(text: str, verdict: ScreenVerdict) -> str:
    """Insert advisory highlight markers around flagged spans.

    Ascending walk over the ORIGINAL text with overlap clipping: spans from
    different families can overlap (e.g. a line-start override match and an
    in-line impersonation phrase); the first (earliest-start) span wins and
    later overlapping spans are skipped rather than nest markers. The
    markers are advisory: they draw the model's attention to spans that
    match injection families WITHOUT changing the span content (the
    containment rule still treats everything inside the envelope as data).
    """
    if not verdict.flagged or not verdict.spans:
        return text
    parts: List[str] = []
    pos = 0
    last_end = -1
    for family, start, end in sorted(verdict.spans, key=lambda s: (s[1], s[2])):
        if start < last_end:
            continue  # overlaps an already-highlighted span — clip
        parts.append(text[pos:start])
        parts.append(
            HIGHLIGHT_OPEN + f" {family} — untrusted content matched an injection pattern]"
        )
        parts.append(text[start:end])
        parts.append(HIGHLIGHT_CLOSE)
        pos = end
        last_end = end
    parts.append(text[pos:])
    return "".join(parts)


def screen_and_highlight(text: str) -> Tuple[str, ScreenVerdict]:
    """Convenience: screen then highlight. Returns (possibly-marked) text."""
    verdict = screen_untrusted_content(text)
    if not verdict.flagged:
        return text, verdict
    return highlight_flagged_spans(text, verdict), verdict


__all__ = [
    "FAMILIES",
    "ScreenVerdict",
    "screen_untrusted_content",
    "highlight_flagged_spans",
    "screen_and_highlight",
    "HIGHLIGHT_OPEN",
    "HIGHLIGHT_CLOSE",
]
