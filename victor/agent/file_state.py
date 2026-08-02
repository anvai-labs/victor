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

"""Helpers for capturing lightweight filesystem state for read deduplication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class FileStateSnapshot:
    """Cheap fingerprint for detecting on-disk file changes."""

    resolved_path: str
    mtime_ns: int
    size_bytes: int


def normalize_file_path(path: str | None) -> Optional[str]:
    """Normalize a filesystem path without requiring it to exist."""
    if not path:
        return None
    try:
        return str(Path(path).expanduser().resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return path


def capture_file_state(path: str | None) -> Optional[FileStateSnapshot]:
    """Capture a file's current state for change detection.

    Returns ``None`` when the path is missing or cannot be stat-ed. Callers
    should treat that as "state unknown" and avoid blocking rereads.
    """
    normalized_path = normalize_file_path(path)
    if not normalized_path:
        return None

    try:
        stat_result = Path(normalized_path).stat()
    except (OSError, RuntimeError, ValueError):
        return None

    return FileStateSnapshot(
        resolved_path=normalized_path,
        mtime_ns=stat_result.st_mtime_ns,
        size_bytes=stat_result.st_size,
    )
