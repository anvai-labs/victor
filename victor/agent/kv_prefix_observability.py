"""KV prefix-stability observability.

Provider prefix caching (OpenAI-compatible automatic caching, KV-cache tiers)
only pays off when consecutive provider calls extend each other byte-for-byte:
the payload of call N must be a strict prefix of call N+1 within a user turn.
This module provides the canonical serialization used to check that invariant
plus a tiny per-orchestrator tracker that classifies every provider call:

- ``prefix_extended`` — did this call's payload extend the previous call's
  bytes (cache-friendly append-only growth)?
- ``divergence_offset`` — first differing byte versus the previous call, the
  smoking-gun locator for cache-busting mutations (e.g. a re-rendered ledger
  message at index 1 diverges immediately after the system prompt).
- ``boundary`` — one of the sanctioned single-miss boundary labels (user turn,
  compaction, tool-set change, system refresh, model switch, resume, trim) or
  ``none``.

The serialization covers exactly the wire-relevant Message fields; ``metadata``
is never sent to providers and ``images`` are excluded from the fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_DEBUG_ENV = "VICTOR_KV_PREFIX_DEBUG"

#: Message model fields excluded from the wire fingerprint. ``metadata`` never
#: reaches providers; ``images`` (rare multimodal payloads) are excluded so the
#: fingerprint stays cheap and text-stable.
_EXCLUDED_FIELDS = {"metadata", "images"}


def canonical_message_bytes(message: Any) -> bytes:
    """Deterministic wire-relevant serialization of one message."""
    payload = message.model_dump(exclude=_EXCLUDED_FIELDS)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def serialize_payload(messages: Sequence[Any]) -> bytes:
    """Length-prefixed concat so message boundaries cannot alias."""
    parts: List[bytes] = []
    for message in messages:
        chunk = canonical_message_bytes(message)
        parts.append(f"{len(chunk)}:".encode("ascii") + chunk)
    return b"".join(parts)


def tools_fingerprint(tools: Optional[Sequence[Any]]) -> Optional[str]:
    """Order-insensitive fingerprint of the tool set (names)."""
    if tools is None:
        return None
    names = sorted(str(tool) for tool in tools)
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:12]


def first_divergence(previous: bytes, current: bytes) -> Optional[int]:
    """First differing byte offset, or ``None`` when ``current`` extends ``previous``."""
    if current.startswith(previous):
        return None
    limit = min(len(previous), len(current))
    for offset in range(limit):
        if previous[offset] != current[offset]:
            return offset
    return limit


class PrefixCallRecord:
    """Classification of one provider call's payload versus the previous call."""

    __slots__ = (
        "call_index",
        "prefix_hash",
        "bytes_len",
        "tools_hash",
        "prefix_extended",
        "divergence_offset",
        "boundary",
        "message_count",
    )

    def __init__(
        self,
        call_index: int,
        prefix_hash: str,
        bytes_len: int,
        tools_hash: Optional[str],
        prefix_extended: Optional[bool],
        divergence_offset: Optional[int],
        boundary: str,
        message_count: int,
    ) -> None:
        self.call_index = call_index
        self.prefix_hash = prefix_hash
        self.bytes_len = bytes_len
        self.tools_hash = tools_hash
        self.prefix_extended = prefix_extended
        self.divergence_offset = divergence_offset
        self.boundary = boundary
        self.message_count = message_count

    def log_fields(self) -> Dict[str, Any]:
        fields: Dict[str, Any] = {
            "call": self.call_index,
            "prefix": self.prefix_hash,
            "extended": (
                str(self.prefix_extended).lower() if self.prefix_extended is not None else "first"
            ),
            "boundary": self.boundary,
            "bytes": self.bytes_len,
            "messages": self.message_count,
        }
        if self.tools_hash is not None:
            fields["tools"] = self.tools_hash
        if self.divergence_offset is not None:
            fields["divergence"] = self.divergence_offset
        return fields

    def log_line(self) -> str:
        return "[kv-cache] " + " ".join(
            f"{key}={value}" for key, value in self.log_fields().items()
        )


class PrefixCallTracker:
    """Remembers the previous call's payload; classifies each call.

    The orchestrator records every assembled provider-bound payload here.
    Sanctioned boundary events (user turn, compaction, tool-set change, …) are
    announced via :meth:`note_boundary` and consumed by the next
    :meth:`record`; anything that diverges without an announced boundary logs
    at INFO with its ``divergence=`` offset so the mutating message can be
    located from logs alone.
    """

    def __init__(self) -> None:
        self._previous_bytes: Optional[bytes] = None
        self._previous_tools: Optional[str] = None
        self._previous_count: int = 0
        self._call_index: int = 0
        self._pending_boundary: Optional[str] = None

    def note_boundary(self, kind: str) -> None:
        """Announce a sanctioned boundary for the next recorded call."""
        self._pending_boundary = kind

    def record(
        self,
        messages: Sequence[Any],
        tools: Optional[Sequence[Any]] = None,
    ) -> PrefixCallRecord:
        payload = serialize_payload(messages)
        prefix_hash = hashlib.sha256(payload).hexdigest()[:12]
        tools_hash = tools_fingerprint(tools)

        self._call_index += 1
        boundary = self._pending_boundary
        self._pending_boundary = None

        extended: Optional[bool]
        divergence: Optional[int]
        if self._previous_bytes is None:
            extended = None
            divergence = None
            if boundary is None:
                boundary = "session_start"
        else:
            divergence = first_divergence(self._previous_bytes, payload)
            extended = divergence is None
            if boundary is None:
                if tools_hash is not None and self._previous_tools is not None:
                    if tools_hash != self._previous_tools:
                        boundary = "tool_set"
                if boundary is None and len(messages) < self._previous_count:
                    boundary = "compaction_or_trim"

        record = PrefixCallRecord(
            call_index=self._call_index,
            prefix_hash=prefix_hash,
            bytes_len=len(payload),
            tools_hash=tools_hash,
            prefix_extended=extended,
            divergence_offset=divergence,
            boundary=boundary or "none",
            message_count=len(messages),
        )

        notable = record.boundary != "none" or extended is False
        (logger.info if notable else logger.debug)("%s", record.log_line())

        if os.getenv(_DEBUG_ENV):
            for index, message in enumerate(messages):
                chunk = canonical_message_bytes(message)
                logger.debug(
                    "[kv-cache] call=%d msg=%d role=%s len=%d sha=%s",
                    record.call_index,
                    index,
                    getattr(message, "role", "?"),
                    len(chunk),
                    hashlib.sha256(chunk).hexdigest()[:8],
                )

        self._previous_bytes = payload
        self._previous_tools = tools_hash
        self._previous_count = len(messages)
        return record

    def reset(self) -> None:
        """Forget history (e.g. after session resume or model switch)."""
        self.__init__()  # type: ignore[misc]
