# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared heuristics for prompt scope and clarification checks."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Iterable, Mapping, Optional, Sequence

_DIRECT_AMBIGUOUS_REFERENCE_PATTERN = re.compile(
    r"\b(?:it|this|them|these|those|same thing|same one|above|below)\b",
    flags=re.IGNORECASE,
)
_THAT_PATTERN = re.compile(r"\bthat\b", flags=re.IGNORECASE)
_RELATIVE_CLAUSE_THAT_PATTERN = re.compile(
    r"\bthat\s+(?:should|must|can|could|would|will|may|might|"
    r"needs?|need|needed|is|are|was|were|has|have|had)\b",
    flags=re.IGNORECASE,
)
_FILE_PATH_PATTERN = re.compile(
    r"(?:^|\s|[\"'\-`])((?:\.{0,2}/)?[\w./-]+/[\w.-]+\.[a-z]{1,10})" r"(?:\s|[\"'`]|$|[,;:.\)]|\Z)",
    flags=re.IGNORECASE,
)
_BARE_FILENAME_PATTERN = re.compile(
    r"(?:^|\s|[\"'`])([\w.-]+\.[a-z]{1,10})(?:\s|[\"'`]|$|[,;:.\)]|\Z)",
    flags=re.IGNORECASE,
)
_SCOPE_HINT_PATTERN = re.compile(r"`([^`]{3,80})`")
_TARGET_KEYS = ("file", "file_path", "path", "target", "component", "symbol")
_REMEDIATION_ACTION_PATTERN = re.compile(
    r"\b(?:address|handle|tackle|resolve|fix|implement|apply|finish|complete|"
    r"update|modify|edit|refactor|consolidate|merge|deduplicate|remove|reduce)\b",
    flags=re.IGNORECASE,
)
_FOLLOW_UP_SCOPE_PATTERN = re.compile(
    r"\b(?:remaining|rest|last|final|findings|issues|concerns|problems|tasks|"
    r"feedback|changes|fixes|recommendations|optimizations)\b",
    flags=re.IGNORECASE,
)


@lru_cache(maxsize=None)
def _compile_marker_pattern(markers: tuple[str, ...]) -> re.Pattern[str]:
    escaped = "|".join(re.escape(marker) for marker in markers)
    return re.compile(rf"\b(?:{escaped})\b", flags=re.IGNORECASE)


def contains_keyword_marker(text: str, markers: Sequence[str]) -> bool:
    """Return whether text contains a whole-word marker from the provided set."""
    if not text or not markers:
        return False
    return bool(_compile_marker_pattern(tuple(markers)).search(text))


def has_ambiguous_target_reference(text: str) -> bool:
    """Return whether text contains an unresolved target reference."""
    if not text:
        return False
    if _DIRECT_AMBIGUOUS_REFERENCE_PATTERN.search(text):
        return True
    for match in _THAT_PATTERN.finditer(text):
        if _RELATIVE_CLAUSE_THAT_PATTERN.match(text, match.start()):
            continue
        return True
    return False


def is_ambiguous_write_followup_request(text: str) -> bool:
    """Return whether text looks like a write request that depends on prior context."""
    if not text:
        return False
    if not _REMEDIATION_ACTION_PATTERN.search(text):
        return False
    return has_ambiguous_target_reference(text) or bool(_FOLLOW_UP_SCOPE_PATTERN.search(text))


def content_has_explicit_target(text: str) -> bool:
    """Return whether text contains an explicit file or scoped identifier target."""
    if not text:
        return False
    return bool(
        _FILE_PATH_PATTERN.search(text)
        or _BARE_FILENAME_PATTERN.search(text)
        or _SCOPE_HINT_PATTERN.search(text)
    )


def conversation_history_has_explicit_target(
    conversation_history: Optional[Iterable[Any]],
    *,
    max_messages: int = 6,
) -> bool:
    """Return whether recent conversation history already established a target."""
    if not conversation_history:
        return False

    messages = list(conversation_history)
    for message in reversed(messages[-max_messages:]):
        payload = _message_payload(message)
        if payload is None:
            continue

        if any(bool(payload.get(key)) for key in _TARGET_KEYS):
            return True

        content = payload.get("content")
        if isinstance(content, str) and content_has_explicit_target(content):
            return True

    return False


def _message_payload(message: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(message, Mapping):
        return message
    if hasattr(message, "model_dump"):
        payload = message.model_dump()
        if isinstance(payload, Mapping):
            return payload
    role = getattr(message, "role", None)
    content = getattr(message, "content", None)
    if role is None and content is None:
        return None
    return {"role": role, "content": content}
