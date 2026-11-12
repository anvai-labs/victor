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

"""Transcript format detection for session interop.

Reads only a bounded prefix of the file: detection never parses the whole
transcript. All limits are guards against hostile or accidentally huge
inputs; nothing here evaluates anything but ``json.loads``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, List

# Resource guards (checked again by readers while streaming).
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_LINE_BYTES = 16 * 1024 * 1024
MAX_MESSAGES = 100_000
MAX_MESSAGE_CHARS = 200_000

# How much of the file detection inspects.
_DETECT_PREFIX_BYTES = 64 * 1024


class TranscriptFormat(str, Enum):
    """Supported transcript formats (readable; subset writable)."""

    CLAUDE_CODE = "claude"  # ~/.claude/projects/<slug>/<uuid>.jsonl
    CODEX_ROLLOUT = "codex"  # ~/.codex/sessions/**/rollout-*.jsonl
    CHATGPT_EXPORT = "chatgpt"  # ChatGPT conversations.json (tree or array)
    VICTOR_JSON = "victor"  # victor-native flat dict (or legacy export array)
    ANTHROPIC_JSON = "anthropic"  # Anthropic Messages API request shape
    GENERIC_JSON = "json"  # {"messages": [...]} fallback


class UnsupportedFormatError(ValueError):
    """Raised when a transcript cannot be detected or does not match a flag."""

    def __init__(self, message: str, detection: "DetectionResult | None" = None):
        super().__init__(message)
        self.detection = detection


@dataclass
class DetectionResult:
    """Outcome of format sniffing."""

    format: TranscriptFormat
    confidence: float
    reasons: List[str] = field(default_factory=list)


def _first_jsonl_records(path: Path, max_records: int = 5) -> List[Any]:
    """Parse up to ``max_records`` JSONL records from the file prefix."""
    records: List[Any] = []
    with open(path, "rb") as f:
        raw = f.read(_DETECT_PREFIX_BYTES)
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) > MAX_LINE_BYTES:
            break
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # The bounded prefix can end inside JSON or a UTF-8 character.
            # Keep complete records already read for format identification.
            break
        if len(records) >= max_records:
            break
    return records


def _looks_like_claude(records: List[Any]) -> bool:
    # Current Claude Code transcripts may begin with bookkeeping records such
    # as mode, permission-mode, atis-latch, and file-history-snapshot. Detect
    # the first actual conversation record instead of requiring every sampled
    # record to have the message/uuid shape.
    return any(
        isinstance(record, dict)
        and record.get("type") in ("user", "assistant")
        and isinstance(record.get("message"), dict)
        and isinstance(record.get("uuid"), str)
        for record in records
    )


def _looks_like_codex(records: List[Any]) -> bool:
    return bool(records) and any(
        isinstance(r, dict) and r.get("type") in ("session_meta", "response_item") for r in records
    )


def _json_head(path: Path) -> Any:
    """Parse a single JSON document from the file prefix (or None)."""
    with open(path, "rb") as f:
        raw = f.read(_DETECT_PREFIX_BYTES)
    # Array-or-document: try the whole prefix first; for arrays the prefix may
    # be a truncation, so fall back to a first-element probe.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    text = raw.decode("utf-8", errors="replace").lstrip()
    if text.startswith("["):
        # Parse just the first array element.
        try:
            dec = json.JSONDecoder()
            start = text.index("[") + 1
            while start < len(text) and text[start] in " \t\r\n":
                start += 1
            obj, _ = dec.raw_decode(text, start)
            return [obj]
        except (ValueError, json.JSONDecodeError):
            return None
    return None


def detect_format(path: Path) -> DetectionResult:
    """Sniff the transcript format from a bounded file prefix.

    Raises:
        UnsupportedFormatError: file missing, oversized, or unrecognizable.
    """
    path = Path(path)
    if not path.is_file():
        raise UnsupportedFormatError(f"File not found: {path}")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise UnsupportedFormatError(
            f"File exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB guard: {size} bytes"
        )
    if size == 0:
        raise UnsupportedFormatError("File is empty")

    # JSONL shapes first (claude/codex are the common real transcripts).
    records = _first_jsonl_records(path)
    if _looks_like_claude(records):
        return DetectionResult(
            TranscriptFormat.CLAUDE_CODE, 0.95, ["JSONL entries with message+uuid"]
        )
    if _looks_like_codex(records):
        return DetectionResult(
            TranscriptFormat.CODEX_ROLLOUT, 0.95, ["JSONL session_meta/response_item"]
        )

    # Single-document JSON shapes.
    doc = _json_head(path)
    if isinstance(doc, dict):
        if "mapping" in doc:
            return DetectionResult(TranscriptFormat.CHATGPT_EXPORT, 0.9, ["mapping tree"])
        if isinstance(doc.get("messages"), list) and "session_id" in doc:
            return DetectionResult(TranscriptFormat.VICTOR_JSON, 0.9, ["session_id + messages"])
        if "conversation" in doc and isinstance(doc.get("conversation"), dict):
            return DetectionResult(
                TranscriptFormat.VICTOR_JSON, 0.8, ["legacy export payload shape"]
            )
        if "messages" in doc and "model" in doc:
            return DetectionResult(TranscriptFormat.ANTHROPIC_JSON, 0.7, ["messages + model"])
        if isinstance(doc.get("messages"), list):
            return DetectionResult(TranscriptFormat.GENERIC_JSON, 0.5, ["messages array"])
    elif isinstance(doc, list) and doc:
        first = doc[0]
        if isinstance(first, dict):
            if "mapping" in first:
                return DetectionResult(
                    TranscriptFormat.CHATGPT_EXPORT, 0.9, ["array of mapping trees"]
                )
            if "metadata" in first and "conversation" in first:
                return DetectionResult(
                    TranscriptFormat.VICTOR_JSON, 0.85, ["legacy victor export array"]
                )

    raise UnsupportedFormatError(
        f"Could not detect transcript format for {path.name}",
        detection=DetectionResult(TranscriptFormat.GENERIC_JSON, 0.0, ["no signature matched"]),
    )
