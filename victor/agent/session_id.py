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

"""Session ID generation and utilities.

Generates unique session IDs in the format:
    {project_root}-{base62_timestamp}-{random_suffix}

Example: abc123-9Kx7Z2-aB3dEf_g1w==
- project_root: First 6 chars of project directory name (base62-encoded if needed)
- base62_timestamp: Millisecond timestamp in base62 (sortable, compact)
- random_suffix: Cryptographic entropy via secrets.token_urlsafe(8)

Legacy 2-part format ({project_root}-{base62_timestamp}) is still accepted
when parsing/validating.

This provides:
- Project context in the session ID
- Lexicographically sortable by timestamp
- Compact representation (13 chars vs 15 for timestamp format)
- Collision-resistant across projects and concurrent generation
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Base62 alphabet (0-9, A-Z, a-z)
BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BASE62_LEN = len(BASE62_ALPHABET)


def encode_base62(num: int) -> str:
    """Encode a number to base62 string.

    Args:
        num: Integer to encode

    Returns:
        Base62 encoded string

    Example:
        >>> encode_base62(123456789)
        '8M0KX'
    """
    if num == 0:
        return BASE62_ALPHABET[0]

    encoded = []
    while num > 0:
        num, remainder = divmod(num, BASE62_LEN)
        encoded.append(BASE62_ALPHABET[remainder])

    return "".join(reversed(encoded))


def decode_base62(encoded: str) -> int:
    """Decode a base62 string to number.

    Args:
        encoded: Base62 encoded string

    Returns:
        Decoded integer

    Example:
        >>> decode_base62('8M0KX')
        123456789
    """
    num = 0
    for char in encoded:
        num = num * BASE62_LEN + BASE62_ALPHABET.index(char)
    return num


def get_project_root_hash(project_root: Path) -> str:
    """Generate 6-character project root identifier.

    Uses the first 6 characters of the directory name, encoded to base62
    if the name contains special characters or is too long.

    Args:
        project_root: Path to project root directory

    Returns:
        6-character identifier for the project

    Examples:
        >>> get_project_root_hash(Path("/home/user/myproject"))
        'myproj'
        >>> get_project_root_hash(Path("/home/user/victor-ai"))
        'victo'
    """
    # Get directory name
    dirname = project_root.name

    # If directory name is >= 6 chars and alphanumeric, use first 6
    if len(dirname) >= 6 and dirname.isalnum():
        return dirname[:6].lower()

    # Otherwise, hash the directory name and encode to base62
    hash_bytes = hashlib.md5(dirname.encode()).digest()
    hash_num = int.from_bytes(hash_bytes[:4], byteorder="big")
    return encode_base62(hash_num).zfill(6)[:6].lower()


def generate_session_id(project_root: Optional[Path] = None) -> str:
    """Generate a unique session ID.

    Format: {project_root}-{base62_timestamp}-{random_suffix}

    Args:
        project_root: Path to project root (auto-detected if None)

    Returns:
        Session ID string (e.g., "abc123-9Kx7Z2-aB3dEf_g1w==")
    """
    from victor.config.settings import get_project_paths

    # Auto-detect project root if not provided
    if project_root is None:
        project_root = get_project_paths().project_root

    # Get project root hash (6 chars)
    root_hash = get_project_root_hash(project_root)

    # Get current timestamp in milliseconds
    timestamp_ms = int(time.time() * 1000)

    # Encode timestamp to base62
    base62_timestamp = encode_base62(timestamp_ms)

    # Add cryptographic entropy suffix for collision resistance
    random_suffix = secrets.token_urlsafe(8)

    # Combine: {root_hash}-{base62_timestamp}-{random_suffix}
    session_id = f"{root_hash}-{base62_timestamp}-{random_suffix}"

    logger.debug(f"Generated session ID: {session_id} for project: {project_root}")
    return session_id


def parse_session_id(session_id: str) -> dict[str, str | int]:
    """Parse a session ID into components.

    Supports both legacy 2-part format ({root}-{timestamp}) and new
    3-part format ({root}-{timestamp}-{entropy}).

    Args:
        session_id: Session ID string (e.g., "abc123-9Kx7Z2-aB3dEf_g1w==")

    Returns:
        Dictionary with components:
        - project_root: 6-char project identifier
        - base62_timestamp: Base62 encoded timestamp
        - timestamp_ms: Decoded timestamp in milliseconds
        - timestamp_iso: ISO format timestamp
        - entropy: Random suffix (empty string for legacy IDs)
    """
    try:
        parts = session_id.split("-", maxsplit=2)

        if len(parts) < 2 or len(parts) > 3:
            raise ValueError(f"Invalid session ID format: {session_id}")

        project_root = parts[0]
        base62_timestamp = parts[1]
        entropy = parts[2] if len(parts) == 3 else ""

        # Decode timestamp
        timestamp_ms = decode_base62(base62_timestamp)

        # Convert to ISO format
        from datetime import datetime, timezone

        timestamp_iso = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()

        return {
            "project_root": project_root,
            "base62_timestamp": base62_timestamp,
            "timestamp_ms": timestamp_ms,
            "timestamp_iso": timestamp_iso,
            "entropy": entropy,
        }
    except Exception as e:
        logger.error(f"Failed to parse session ID {session_id}: {e}")
        raise ValueError(f"Invalid session ID: {session_id}") from e


def validate_session_id(session_id: str) -> bool:
    """Validate session ID format.

    Args:
        session_id: Session ID string to validate

    Returns:
        True if valid format

    Example:
        >>> validate_session_id("abc123-9Kx7Z2")
        True
        >>> validate_session_id("invalid")
        False
    """
    try:
        parse_session_id(session_id)
        return True
    except (ValueError, IndexError):
        return False


__all__ = [
    "encode_base62",
    "decode_base62",
    "get_project_root_hash",
    "generate_session_id",
    "parse_session_id",
    "validate_session_id",
]
