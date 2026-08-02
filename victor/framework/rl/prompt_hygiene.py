# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Prompt candidate hygiene checks for optimization strategies."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, List, Optional


@dataclass
class PromptHygieneReport:
    """Validation report for an optimized prompt candidate."""

    accepted: bool
    growth_chars: int
    seed_similarity: float
    repeated_trigrams: int
    unsupported_additions: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    redundant_additions: List[str] = field(default_factory=list)
    truncated_tail: bool = False


# Repeated trigrams tolerated before a candidate counts as repetitive garbage.
#
# This was zero, which is not a bound on garbage but a bound on parallel
# structure — and prompt sections are largely parallel structure. Measured over
# the shipped, hand-reviewed sections: LARGE_FILE_PAGINATION_GUIDANCE scores 12
# and PARALLEL_READ_GUIDANCE scores 4, so at zero tolerance two of our own
# prompts would be rejected as garbage if a mutation proposed them. A genuine
# 1421-char rewrite from kimi-k3 died on this gate.
#
# Set to twice the highest count any shipped section reaches. The failure mode it
# guards against is not subtle: a line repeated forty times scores 154, six times
# this cap.
MAX_REPEATED_TRIGRAMS = 24


def evaluate_prompt_candidate(
    seed_text: str,
    candidate_text: str,
    *,
    allowed_additions: Iterable[str] = (),
    max_growth_chars: int = 240,
    min_seed_similarity: float = 0.6,
    max_repeated_trigrams: int = MAX_REPEATED_TRIGRAMS,
) -> PromptHygieneReport:
    """Validate an optimized prompt against basic safety and quality constraints."""
    growth_chars = len(candidate_text) - len(seed_text)
    seed_similarity = _seed_similarity(seed_text, candidate_text)
    repeated_trigrams = _repeated_trigram_count(candidate_text)
    unsupported_additions = _find_unsupported_additions(
        seed_text,
        candidate_text,
        allowed_additions,
    )
    redundant_additions = find_redundant_additions(seed_text, candidate_text)
    truncated_tail = has_truncated_tail(candidate_text)

    violations = []
    if growth_chars > max_growth_chars:
        violations.append("growth_exceeded")
    if seed_similarity < min_seed_similarity:
        violations.append("seed_similarity_too_low")
    if repeated_trigrams > max_repeated_trigrams:
        violations.append("repeated_trigrams")
    if unsupported_additions:
        violations.append("unsupported_additions")
    if redundant_additions:
        violations.append("redundant_additions")
    if truncated_tail:
        violations.append("truncated_tail")

    return PromptHygieneReport(
        accepted=not violations,
        growth_chars=growth_chars,
        seed_similarity=seed_similarity,
        repeated_trigrams=repeated_trigrams,
        unsupported_additions=unsupported_additions,
        violations=violations,
        redundant_additions=redundant_additions,
        truncated_tail=truncated_tail,
    )


# Newly added lines this similar to an existing line are restating, not adding.
REDUNDANT_LINE_SIMILARITY = 0.6


# Smallest share of the truncation limit a boundary cut may retain before it
# counts as a collapse rather than a clean cut. Matches the quarter-of-the-seed
# collapse guard callers apply, so truncation never returns a stump they will
# immediately reject.
MIN_BOUNDARY_RETENTION = 0.25

# Words that carry no instruction, dropped before comparing two lines.
_STOPWORDS = frozenset(
    {"a", "an", "the", "is", "are", "be", "it", "its", "this", "that", "your", "you"}
)

# A final line ending on one of these is a fragment, not an instruction.
_DANGLING_WORDS = frozenset(
    {
        # conjunctions
        "and",
        "or",
        "but",
        "so",
        "because",
        "while",
        "than",
        "that",
        "which",
        # determiners
        "the",
        "a",
        "an",
        # prepositions
        "to",
        "of",
        "for",
        "with",
        "from",
        "into",
        "over",
        "under",
        "by",
        "on",
        "in",
        "at",
        "as",
        # temporal / conditional connectives
        "then",
        "before",
        "after",
        "if",
        "when",
    }
)

_TERMINATORS = (".", "!", "?", ":", ";", ")", '"', "'", "`", "—")


def has_truncated_tail(text: str) -> bool:
    """True when the candidate's final instruction is a fragment.

    A live candidate was persisted ending ``"- Read error messages carefully
    and"``: the bloat cap cut at the last space, which fell mid-sentence. The
    last line of a prompt is an instruction, so a half-written one is a defect
    the persist gate must catch, not merely a cosmetic issue.
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return True
    last = lines[-1].rstrip()
    if last.endswith(_TERMINATORS):
        return False
    words = last.split()
    return words[-1].lower() in _DANGLING_WORDS if words else True


def find_redundant_additions(seed_text: str, candidate_text: str) -> List[str]:
    """Added lines that merely restate guidance already in the seed.

    Distinct from ``_find_unsupported_additions``: a rewrite legitimately adds
    and rephrases lines, so *new* content is not itself a violation. What is a
    violation is a mutation that appends a near-copy of a line already present
    — a live candidate grew ``CONCISE_MODE_GUIDANCE`` by appending "Read the
    error message carefully." directly beneath "Read error messages carefully
    before retrying." Degenerate growth like that dilutes the section without
    adding an instruction.

    ``dilutes`` is the operative word, and it is what separates this from an
    ordinary rewrite. Redundancy needs *both* copies present: the corruption
    above appended its near-copy directly beneath the line it restated. When a
    line is reworded **in place** the original is gone, so the only thing left to
    compare against is the seed line it replaced — and by that measure every
    tightening of existing wording looked like corruption. Measured against the
    shipped COMPLETION_GUIDANCE, changing one word tripped this, as did merging
    two bullets, while a pure append passed; the gate was rejecting exactly the
    improvements worth having and admitting the growth it meant to stop. So a
    near-duplicate only counts when the line it restates survives in the
    candidate.
    """
    seed_lines = [line for line in seed_text.splitlines() if line.strip()]
    seed_word_sets = [_content_words(line) for line in seed_lines]
    seed_norm = {_normalize_line(line) for line in seed_lines}
    candidate_norm = {_normalize_line(line) for line in candidate_text.splitlines() if line.strip()}

    redundant: List[str] = []
    for raw_line in candidate_text.splitlines():
        if not raw_line.strip() or _normalize_line(raw_line) in seed_norm:
            continue
        added = _content_words(raw_line)
        if not added:
            continue
        for seed_line, existing in zip(seed_lines, seed_word_sets):
            union = added | existing
            if not union:
                continue
            if len(added & existing) / len(union) < REDUNDANT_LINE_SIMILARITY:
                continue
            if _normalize_line(seed_line) not in candidate_norm:
                # The line this restates was replaced, not duplicated. Keep
                # scanning: it may also resemble a line that did survive.
                continue
            redundant.append(raw_line.strip())
            break
    return redundant


def _content_words(line: str) -> set:
    """Instruction-bearing words of a line: bullet-stripped, de-pluralized."""
    stripped = line.strip().lstrip("-*0123456789. ").lower()
    words = {word.strip(".,:;()\"'") for word in stripped.split()}
    return {
        word[:-1] if word.endswith("s") and len(word) > 3 else word
        for word in words
        if word and word not in _STOPWORDS
    }


def _seed_similarity(seed_text: str, candidate_text: str) -> float:
    seed_tokens = set(_tokens(seed_text))
    candidate_tokens = set(_tokens(candidate_text))
    if not seed_tokens or not candidate_tokens:
        return 0.0
    return len(seed_tokens & candidate_tokens) / len(seed_tokens)


def _repeated_trigram_count(text: str) -> int:
    tokens = _tokens(text)
    if len(tokens) < 3:
        return 0
    trigrams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    counts = Counter(trigrams)
    return sum(count - 1 for count in counts.values() if count > 1)


def _find_unsupported_additions(
    seed_text: str,
    candidate_text: str,
    allowed_additions: Iterable[str],
) -> List[str]:
    seed_lines = {_normalize_line(line) for line in seed_text.splitlines() if line.strip()}
    allowed_lines = [_normalize_line(line) for line in allowed_additions if str(line).strip()]
    unsupported = []

    for raw_line in candidate_text.splitlines():
        line = _normalize_line(raw_line)
        if not line or line in seed_lines:
            continue
        if any(line.startswith(allowed) or allowed.startswith(line) for allowed in allowed_lines):
            continue
        unsupported.append(raw_line.strip())
    return unsupported


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lower())


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_()]+", text.lower())


# ── Sanitization (transform) ───────────────────────────────────────────────
# evaluate_prompt_candidate() above is a *validation* gate (report-only). The
# two helpers below are the matching *transform* used by GEPA/PrefPO mutation
# paths to clean a candidate before it is stored. They were referenced by
# gepa_service.mutate / prompt_optimizer but never defined, which surfaced as
# ImportError on those code paths.


def boundary_aware_truncate(text: str, limit: int) -> tuple[str, bool]:
    """Truncate ``text`` to ``limit`` chars without severing an instruction.

    The cut is taken at the strongest boundary available at or before
    ``limit``, in descending order of preference:

      1. a **line** boundary — drops the trailing partial line whole
      2. a **sentence** boundary (``.``/``!``/``?`` followed by space)
      3. a **word** boundary (last whitespace)
      4. a hard cut at ``limit`` as a last resort

    Word-boundary cutting alone (the original behaviour) is not sufficient for
    prompts: it keeps a dangling clause as the final instruction. A real
    candidate was persisted ending ``"- Read error messages carefully and"``,
    because the last space before the cap fell mid-sentence. A prompt's last
    line is an instruction, so a partial one is worse than no line at all.

    Returns ``(truncated_text, was_truncated)``.
    """
    if limit <= 0 or len(text) <= limit:
        return text, False

    head = text[:limit]

    # A boundary this early is not a boundary, it is a collapse: the strongest
    # cut available can still discard nearly all of the budget we were allowed
    # to use. A real candidate arrived as a short heading followed by one very
    # long line, so the last newline inside a 1500-char limit sat at 118 — the
    # line cut threw away 92% of the allowance, and the caller then rejected the
    # 118-char stump as collapsed, wasting the whole mutation. The floor matches
    # the caller's own collapse guard (a quarter), so this never hands back
    # something that is about to be thrown away.
    floor = int(limit * MIN_BOUNDARY_RETENTION)

    line_cut = head.rfind("\n")
    if line_cut > floor:
        return text[:line_cut].rstrip(), True

    sentence_cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if sentence_cut > floor:
        return text[: sentence_cut + 1].rstrip(), True

    word_cut = head.rfind(" ")
    return text[: word_cut if word_cut > floor else limit].rstrip(), True


def sanitize_prompt_candidate(
    result: str,
    limit: int = 0,
    seed_text: str = "",
) -> str:
    """Clean a mutated prompt before storage.

    Applies, in order:
      1. code-fence stripping — drop ```` ``` ```` delimiters, keep inner text
      2. consecutive-line dedupe — collapse immediately-repeated lines
      3. boundary-aware truncation to ``limit`` chars (when ``limit > 0``)

    ``seed_text`` is accepted for API symmetry with ``evaluate_prompt_candidate``
    and for future similarity-preserving strategies; it does not alter the
    transform today.
    """
    text = _strip_code_fences(result)
    text = _dedupe_consecutive_lines(text)
    if limit > 0 and len(text) > limit:
        text, _ = boundary_aware_truncate(text, limit)
    return text


def _strip_code_fences(text: str) -> str:
    """Remove ```` ``` ```` fence delimiters, preserving the fenced content."""
    out: List[str] = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            continue
        out.append(line)
    return "\n".join(out)


def _dedupe_consecutive_lines(text: str) -> str:
    """Collapse runs of immediately-repeated identical lines to a single line."""
    out: List[str] = []
    prev: Optional[str] = None
    for line in text.splitlines():
        if line != prev:
            out.append(line)
        prev = line
    return "\n".join(out)
