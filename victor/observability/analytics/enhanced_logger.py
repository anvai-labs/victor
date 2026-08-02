# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Enhanced Usage Logger with rotation, encryption, and PII scrubbing.

This module provides an enterprise-grade logging solution:
- Log rotation (size and time-based)
- Optional encryption at rest (Fernet symmetric encryption)
- PII scrubbing (email, API keys, paths, etc.)
- Structured JSONL format with compression

Design Principles:
- Drop-in replacement for UsageLogger
- Configurable via Settings or constructor
- Thread-safe for concurrent access
- Minimal performance overhead
"""

from __future__ import annotations

import gzip
import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Set
from uuid import UUID

from victor.observability.run_kind import current_run_kind

logger = logging.getLogger(__name__)


class PIIScrubber:
    """Scrubs personally identifiable information from log data."""

    # Patterns for common PII
    PATTERNS: List[tuple[str, Pattern, str]] = [
        # Email addresses
        (
            "email",
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "[EMAIL]",
        ),
        # API keys (common formats)
        ("api_key", re.compile(r"\b(sk-[a-zA-Z0-9]{20,})\b"), "[API_KEY]"),
        ("api_key", re.compile(r"\b(xai-[a-zA-Z0-9]{20,})\b"), "[API_KEY]"),
        ("api_key", re.compile(r"\b(AIza[a-zA-Z0-9_-]{35})\b"), "[API_KEY]"),
        # Bearer tokens
        (
            "token",
            re.compile(r"\b(Bearer\s+[a-zA-Z0-9._-]+)\b", re.I),
            "[BEARER_TOKEN]",
        ),
        # Home directory paths
        ("path", re.compile(r"/Users/[a-zA-Z0-9_-]+"), "/Users/[USER]"),
        ("path", re.compile(r"/home/[a-zA-Z0-9_-]+"), "/home/[USER]"),
        ("path", re.compile(r"C:\\Users\\[a-zA-Z0-9_-]+", re.I), r"C:\\Users\\[USER]"),
        # IP addresses (optional)
        ("ip", re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "[IP_ADDR]"),
        # Credit card numbers (basic pattern)
        ("cc", re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"), "[CREDIT_CARD]"),
    ]

    def __init__(
        self,
        scrub_emails: bool = True,
        scrub_api_keys: bool = True,
        scrub_tokens: bool = True,
        scrub_paths: bool = True,
        scrub_ips: bool = False,  # Often useful for debugging
        scrub_credit_cards: bool = True,
        custom_patterns: Optional[List[tuple[str, Pattern, str]]] = None,
    ):
        """Initialize PII scrubber with configurable scrubbing options.

        Args:
            scrub_emails: Scrub email addresses
            scrub_api_keys: Scrub API keys
            scrub_tokens: Scrub bearer tokens
            scrub_paths: Scrub home directory paths
            scrub_ips: Scrub IP addresses
            scrub_credit_cards: Scrub credit card numbers
            custom_patterns: Additional custom patterns to scrub
        """
        self._patterns = []

        for name, pattern, replacement in self.PATTERNS:
            if name == "email" and scrub_emails:
                self._patterns.append((pattern, replacement))
            elif name == "api_key" and scrub_api_keys:
                self._patterns.append((pattern, replacement))
            elif name == "token" and scrub_tokens:
                self._patterns.append((pattern, replacement))
            elif name == "path" and scrub_paths:
                self._patterns.append((pattern, replacement))
            elif name == "ip" and scrub_ips:
                self._patterns.append((pattern, replacement))
            elif name == "cc" and scrub_credit_cards:
                self._patterns.append((pattern, replacement))

        if custom_patterns:
            for _, pattern, replacement in custom_patterns:
                self._patterns.append((pattern, replacement))

    def scrub(self, text: str) -> str:
        """Scrub PII from text.

        Args:
            text: Text to scrub

        Returns:
            Scrubbed text with PII replaced
        """
        for pattern, replacement in self._patterns:
            text = pattern.sub(replacement, text)
        return text

    def scrub_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively scrub PII from a dictionary.

        Args:
            data: Dictionary to scrub

        Returns:
            Scrubbed dictionary
        """
        return self._scrub_mapping(data, seen=set())

    def _scrub_mapping(self, data: Dict[str, Any], seen: Set[int]) -> Dict[str, Any]:
        """Scrub a mapping while breaking recursive references."""
        obj_id = id(data)
        if obj_id in seen:
            return {"__recursive_ref__": "dict"}

        seen.add(obj_id)
        try:
            return {key: self._scrub_value(value, seen) for key, value in data.items()}
        finally:
            seen.discard(obj_id)

    def _scrub_sequence(self, data: List[Any], seen: Set[int]) -> List[Any]:
        """Scrub a sequence while breaking recursive references."""
        obj_id = id(data)
        if obj_id in seen:
            return ["<recursive_ref:list>"]

        seen.add(obj_id)
        try:
            return [self._scrub_value(value, seen) for value in data]
        finally:
            seen.discard(obj_id)

    def _scrub_value(self, value: Any, seen: Set[int]) -> Any:
        """Scrub an arbitrary nested value."""
        if isinstance(value, str):
            return self.scrub(value)
        if isinstance(value, dict):
            return self._scrub_mapping(value, seen)
        if isinstance(value, list):
            return self._scrub_sequence(value, seen)
        return value


class LogRotator:
    """Handles log file rotation by size and time."""

    def __init__(
        self,
        log_file: Path,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        compress: bool = True,
    ):
        """Initialize log rotator.

        Args:
            log_file: Path to log file
            max_bytes: Maximum file size before rotation (default 10MB)
            backup_count: Number of backup files to keep
            compress: Whether to compress rotated files
        """
        self.log_file = log_file
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.compress = compress
        self._lock = threading.Lock()

    def should_rotate(self) -> bool:
        """Check if log file should be rotated.

        Returns:
            True if rotation needed
        """
        if not self.log_file.exists():
            return False
        return self.log_file.stat().st_size >= self.max_bytes

    def rotate(self) -> None:
        """Perform log rotation."""
        with self._lock:
            if not self.should_rotate():
                return

            # Remove oldest backup if at limit
            oldest = self.log_file.with_suffix(f".{self.backup_count}.jsonl")
            if self.compress:
                oldest = oldest.with_suffix(".jsonl.gz")
            if oldest.exists():
                oldest.unlink()

            # Shift existing backups
            for i in range(self.backup_count - 1, 0, -1):
                old_path = self.log_file.with_suffix(f".{i}.jsonl")
                if self.compress:
                    old_path = old_path.with_suffix(".jsonl.gz")
                new_path = self.log_file.with_suffix(f".{i + 1}.jsonl")
                if self.compress:
                    new_path = new_path.with_suffix(".jsonl.gz")
                if old_path.exists():
                    old_path.rename(new_path)

            # Rotate current file
            backup_path = self.log_file.with_suffix(".1.jsonl")
            if self.compress:
                # Compress to .gz
                gz_path = backup_path.with_suffix(".jsonl.gz")
                with open(self.log_file, "rb") as f_in:
                    with gzip.open(gz_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                self.log_file.unlink()
            else:
                self.log_file.rename(backup_path)

            logger.debug(f"Rotated log file: {self.log_file}")


class LogEncryptor:
    """Handles log encryption at rest using Fernet symmetric encryption."""

    def __init__(self, key: Optional[bytes] = None, key_file: Optional[Path] = None):
        """Initialize encryptor.

        Args:
            key: Encryption key (32-byte base64-encoded Fernet key)
            key_file: Path to file containing encryption key
        """
        self._fernet = None

        try:
            from cryptography.fernet import Fernet

            if key:
                self._fernet = Fernet(key)
            elif key_file and key_file.exists():
                with open(key_file, "rb") as f:
                    self._fernet = Fernet(f.read().strip())
            else:
                # Generate and save new key
                if key_file:
                    key = Fernet.generate_key()
                    key_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(key_file, "wb") as f:
                        f.write(key)
                    os.chmod(key_file, 0o600)  # Restrict permissions
                    self._fernet = Fernet(key)
                    logger.info(f"Generated new encryption key: {key_file}")

        except ImportError:
            logger.warning("cryptography package not installed, encryption disabled")

    @property
    def enabled(self) -> bool:
        """Check if encryption is enabled."""
        return self._fernet is not None

    def encrypt(self, data: str) -> bytes:
        """Encrypt data.

        Args:
            data: String to encrypt

        Returns:
            Encrypted bytes
        """
        if not self._fernet:
            return data.encode("utf-8")
        return self._fernet.encrypt(data.encode("utf-8"))

    def decrypt(self, data: bytes) -> str:
        """Decrypt data.

        Args:
            data: Encrypted bytes

        Returns:
            Decrypted string
        """
        if not self._fernet:
            return data.decode("utf-8")
        return self._fernet.decrypt(data).decode("utf-8")


class EnhancedUsageLogger:
    """Enhanced usage logger with rotation, encryption, and PII scrubbing.

    Drop-in replacement for UsageLogger with additional features:
    - Log rotation (size-based with compression)
    - Optional encryption at rest
    - PII scrubbing for privacy compliance
    - Thread-safe concurrent access

    Example:
        logger = EnhancedUsageLogger(
            log_file=Path("~/.victor/logs/usage.jsonl"),
            enabled=True,
            scrub_pii=True,
            encrypt=True,
        )

        logger.log_event("tool_call", {"tool": "read", "file": "/path/to/file"})
    """

    def __init__(
        self,
        log_file: Path,
        enabled: bool = True,
        scrub_pii: bool = True,
        encrypt: bool = False,
        encryption_key_file: Optional[Path] = None,
        max_log_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        compress_rotated: bool = True,
        sampling_filter: Optional[Any] = None,
    ):
        """Initialize enhanced usage logger.

        Args:
            log_file: Path to log file
            enabled: Whether logging is enabled
            scrub_pii: Whether to scrub PII from logs
            encrypt: Whether to encrypt logs at rest
            encryption_key_file: Path to encryption key file
            max_log_size: Maximum log file size before rotation
            backup_count: Number of backup files to keep
            compress_rotated: Whether to compress rotated files
            sampling_filter: Optional SemanticSamplingFilter for noise reduction.
        """
        self._enabled = enabled
        self._log_file = Path(log_file).expanduser()
        self.session_id = str(uuid.uuid4())
        self._lock = threading.Lock()
        self._sampling_filter = sampling_filter
        # Backward compat: UsageLogger used self._logger
        self._logger = logger

        # Initialize components
        self._scrubber = PIIScrubber() if scrub_pii else None

        self._encryptor = None
        if encrypt:
            key_file = encryption_key_file or self._log_file.parent / ".log_key"
            self._encryptor = LogEncryptor(key_file=key_file)

        self._rotator = LogRotator(
            log_file=self._log_file,
            max_bytes=max_log_size,
            backup_count=backup_count,
            compress=compress_rotated,
        )

        if self._enabled:
            self._prepare_log_file()

    def _prepare_log_file(self) -> None:
        """Ensure log directory and file exist."""
        try:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_file.touch(exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create log file {self._log_file}: {e}")
            self._enabled = False

    def is_enabled(self) -> bool:
        """Returns True if logging is enabled."""
        return self._enabled

    def _sanitize_log_data(self, data: Any) -> Any:
        """Sanitize log data to remove non-serializable objects.

        Args:
            data: Data to sanitize (can be dict, list, or primitive)

        Returns:
            Sanitized data safe for JSON serialization
        """
        return self._sanitize_log_data_inner(data, seen=set(), depth=0)

    def _sanitize_log_data_inner(self, data: Any, seen: Set[int], depth: int) -> Any:
        """Recursively sanitize log data while breaking cycles safely."""
        from datetime import date, time
        from types import CoroutineType, MappingProxyType

        if data is None or isinstance(data, (str, int, float, bool)):
            return data

        if isinstance(data, (Path, UUID)):
            return str(data)

        if isinstance(data, (datetime, date, time)):
            return data.isoformat()

        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")

        if isinstance(data, Enum):
            return self._sanitize_log_data_inner(data.value, seen, depth + 1)

        if isinstance(data, CoroutineType):
            return "<coroutine>"
        if inspect.isgenerator(data):
            return "<generator>"
        if inspect.isasyncgen(data):
            return "<async_generator>"
        if inspect.isroutine(data):
            return f"<callable:{getattr(data, '__name__', type(data).__name__)}>"
        if inspect.ismodule(data):
            return f"<module:{getattr(data, '__name__', type(data).__name__)}>"

        if depth >= 20:
            return f"<max_depth:{type(data).__name__}>"

        track_identity = isinstance(
            data, (dict, MappingProxyType, list, tuple, set, frozenset)
        ) or hasattr(data, "__dict__")
        obj_id = id(data)
        if track_identity:
            if obj_id in seen:
                return f"<recursive_ref:{type(data).__name__}>"
            seen.add(obj_id)

        try:
            if isinstance(data, dict) or isinstance(data, MappingProxyType):
                sanitized: Dict[Any, Any] = {}
                for key, value in dict(data).items():
                    safe_key = (
                        key if isinstance(key, (str, int, float, bool)) or key is None else str(key)
                    )
                    sanitized[safe_key] = self._sanitize_log_data_inner(value, seen, depth + 1)
                return sanitized

            if isinstance(data, (list, tuple)):
                return [self._sanitize_log_data_inner(item, seen, depth + 1) for item in data]

            if isinstance(data, (set, frozenset)):
                return [self._sanitize_log_data_inner(item, seen, depth + 1) for item in data]

            if hasattr(data, "model_dump") and callable(data.model_dump):
                return self._sanitize_log_data_inner(data.model_dump(), seen, depth + 1)

            if hasattr(data, "dict") and callable(data.dict):
                return self._sanitize_log_data_inner(data.dict(), seen, depth + 1)

            if hasattr(data, "__dict__"):
                attrs = {
                    key: self._sanitize_log_data_inner(value, seen, depth + 1)
                    for key, value in vars(data).items()
                    if not key.startswith("__")
                }
                if attrs:
                    return {"__class__": type(data).__name__, **attrs}
                return repr(data)

            return repr(data)
        finally:
            if track_identity:
                seen.discard(obj_id)

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a usage event.

        Args:
            event_type: Type of event (e.g., 'tool_call', 'user_prompt')
            data: Event-specific data dictionary
        """
        if not self._enabled:
            return

        # Semantic sampling: drop noise events before disk I/O
        if self._sampling_filter and not self._sampling_filter.should_emit(event_type, data):
            return

        try:
            # Check for rotation
            self._rotator.rotate()

            # Scrub PII if enabled
            if self._scrubber:
                data = self._scrubber.scrub_dict(data)

            # Sanitize data to remove non-serializable objects (coroutines, generators, etc.)
            sanitized_data = self._sanitize_log_data(data)

            log_entry = {
                "session_id": self.session_id,
                # A property of the run, not of the event, so it sits beside
                # session_id rather than inside the payload. Recorded here
                # because it cannot be recovered later: the turn-budget prompt
                # that consumers used to classify on is emitted by the shared
                # agentic loop, so delegate work read as benchmark runs.
                "run_kind": current_run_kind(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "data": sanitized_data,
            }

            log_line = json.dumps(log_entry)

            # Encrypt if enabled
            if self._encryptor and self._encryptor.enabled:
                log_line = self._encryptor.encrypt(log_line).decode("utf-8")

            with self._lock:
                with open(self._log_file, "a") as f:
                    f.write(log_line + "\n")

        except TypeError as e:
            logger.error(f"Failed to serialize log entry: {e}")
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")

    def read_logs(
        self,
        limit: int = 100,
        event_types: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Read logs with optional filtering.

        Args:
            limit: Maximum number of entries to return
            event_types: Filter by event types
            session_id: Filter by session ID

        Returns:
            List of log entries (newest first)
        """
        if not self._log_file.exists():
            return []

        entries = []
        try:
            with open(self._log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Decrypt if needed
                    if self._encryptor and self._encryptor.enabled:
                        line = self._encryptor.decrypt(line.encode("utf-8"))

                    try:
                        entry = json.loads(line)

                        # Apply filters
                        if event_types and entry.get("event_type") not in event_types:
                            continue
                        if session_id and entry.get("session_id") != session_id:
                            continue

                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue

            # Return newest first, limited
            return list(reversed(entries[-limit:]))

        except Exception as e:
            logger.error(f"Failed to read logs: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics.

        Returns:
            Dictionary with log statistics
        """
        stats = {
            "enabled": self._enabled,
            "session_id": self.session_id,
            "log_file": str(self._log_file),
            "encryption_enabled": self._encryptor.enabled if self._encryptor else False,
            "pii_scrubbing_enabled": self._scrubber is not None,
        }

        if self._log_file.exists():
            stats["log_size_bytes"] = self._log_file.stat().st_size
            stats["log_size_mb"] = round(stats["log_size_bytes"] / (1024 * 1024), 2)

        return stats


# Factory function for backwards compatibility
def create_usage_logger(
    log_file: Path,
    enabled: bool = True,
    enhanced: bool = True,
    **kwargs,
) -> EnhancedUsageLogger:
    """Create a usage logger (enhanced or basic).

    Args:
        log_file: Path to log file
        enabled: Whether logging is enabled
        enhanced: Use enhanced logger with all features
        **kwargs: Additional arguments for EnhancedUsageLogger

    Returns:
        Logger instance
    """
    # EnhancedUsageLogger is the canonical implementation (UsageLogger is now an alias)
    return EnhancedUsageLogger(log_file=log_file, enabled=enabled, **kwargs)
