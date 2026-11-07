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

"""Locate sessions in the claude/codex local stores for address-based import.

Lets users import a *specific* session without knowing file paths:

    /import claude:<session-uuid>
    /import codex:<session-uuid>
    victor data session import --list

Safety: session ids are strictly validated (no path characters), every
resolved path is containment-checked against the store root, and listing
parses only a bounded prefix of each transcript.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

_SOURCE_ROOTS = {
    "claude": lambda home: home / ".claude" / "projects",
    "codex": lambda home: home / ".codex" / "sessions",
}

# Session ids are filesystem-safe tokens only (uuids and rollout ids).
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,79}$")
_ADDRESS_RE = re.compile(r"^(claude|codex):([A-Za-z0-9][A-Za-z0-9_-]{7,79})$")

# Listing reads at most this many lines per transcript.
_LIST_SCAN_LINES = 40


class SessionAddressError(ValueError):
    """Invalid session address, or the session was not found in the store."""


@dataclass
class StoreSessionInfo:
    """One discoverable session in an external tool's store."""

    source: str  # "claude" | "codex"
    session_id: str
    title: str
    modified: datetime
    path: Path


def _store_root(source: str) -> Path:
    home = Path.home()
    root = _SOURCE_ROOTS[source](home)
    if not root.is_dir():
        raise SessionAddressError(f"{source} store not found at {root} — is {source} installed?")
    return root


def _claude_slug(project_cwd: Optional[Path]) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", str(project_cwd or Path.cwd()))


def find_transcript(source: str, session_id: str, project_cwd: Optional[Path] = None) -> Path:
    """Return the transcript file for a session id in the source store.

    Raises:
        SessionAddressError: bad id, unknown store, or session not found.
    """
    if source not in _SOURCE_ROOTS:
        raise SessionAddressError(f"Unknown source {source!r} (claude|codex)")
    if not _SESSION_ID_RE.fullmatch(session_id):
        raise SessionAddressError(f"Invalid session id {session_id!r} (expected a uuid-like token)")
    root = _store_root(source)

    if source == "claude":
        # Current project's directory first, then a bounded global search
        # (session uuids are unique, so an unambiguous hit is safe).
        candidates = [root / _claude_slug(project_cwd) / f"{session_id}.jsonl"]
        candidates.extend(sorted(root.glob(f"*/{session_id}.jsonl")))
    else:
        # Codex embeds the session id in the rollout filename.
        candidates = sorted(root.glob(f"*/*/*/*{session_id}*.jsonl"))

    for candidate in candidates:
        if candidate.is_file() and candidate.resolve().is_relative_to(root.resolve()):
            return candidate
    raise SessionAddressError(f"No {source} session {session_id} found in {root}")


def resolve_import_target(spec: str) -> Tuple[Path, str]:
    """Resolve an import spec: ``claude:<id>`` / ``codex:<id>`` or a file path.

    Returns (path, source_label); the label is the store source for addresses
    or "" for plain paths.
    """
    spec = spec.strip()
    if spec.startswith(("claude:", "codex:")):
        match = _ADDRESS_RE.fullmatch(spec)
        if not match:
            raise SessionAddressError(
                f"Invalid session address {spec!r} — expected claude:<uuid> or codex:<uuid>"
            )
        source, session_id = match.groups()
        return find_transcript(source, session_id), source
    return Path(spec).expanduser(), ""


def _first_user_snippet(path: Path, source: str) -> str:
    """Bounded scan for the first real user text (listing preview)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for _ in range(_LIST_SCAN_LINES):
                line = handle.readline()
                if not line:
                    break
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if source == "claude":
                    if (
                        entry.get("type") == "user"
                        and not entry.get("isSidechain")
                        and not entry.get("isMeta")
                    ):
                        content = (entry.get("message") or {}).get("content")
                        text = content if isinstance(content, str) else ""
                        if text and not text.startswith(("<system-reminder", "<")):
                            return " ".join(text.split())[:70]
                else:
                    payload = entry.get("payload") or {}
                    if (
                        entry.get("type") == "response_item"
                        and payload.get("type") == "message"
                        and payload.get("role") == "user"
                    ):
                        for item in payload.get("content", []):
                            if isinstance(item, dict) and item.get("type") == "input_text":
                                text = item.get("text") or ""
                                if text and not text.startswith(("<", "#")):
                                    return " ".join(text.split())[:70]
    except OSError:
        pass
    return "(no preview)"


def list_store_sessions(
    source: str, limit: int = 15, project_cwd: Optional[Path] = None
) -> List[StoreSessionInfo]:
    """List recent sessions in an external store (newest first, bounded)."""
    root = _store_root(source)
    if source == "claude":
        dirs = [root / _claude_slug(project_cwd)]
        if not dirs[0].is_dir():
            dirs = [d for d in root.iterdir() if d.is_dir()]
        files: List[Path] = []
        for directory in dirs:
            files.extend(f for f in directory.glob("*.jsonl") if f.is_file())
    else:
        files = list(root.glob("*/*/*/*.jsonl"))

    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    results: List[StoreSessionInfo] = []
    for path in files[: max(limit, 1) * 2]:
        session_id = path.stem if source == "claude" else _codex_id_from_filename(path.name)
        if not session_id:
            continue
        results.append(
            StoreSessionInfo(
                source=source,
                session_id=session_id,
                title=_first_user_snippet(path, source),
                modified=datetime.fromtimestamp(path.stat().st_mtime),
                path=path,
            )
        )
        if len(results) >= limit:
            break
    return results


def _codex_id_from_filename(name: str) -> str:
    """Extract the trailing uuid from a rollout-<ts>-<uuid>.jsonl name."""
    match = re.search(
        r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-" r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})", name
    )
    return match.group(1) if match else ""
