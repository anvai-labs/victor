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
# See the the License for the specific language governing permissions and
# limitations under the License.

"""Shell command result cache for reducing redundant CI/CD and system queries.

This module provides intelligent caching for shell commands to address:
1. Multiple redundant API calls (e.g., repeated 'gh run view' commands)
2. No caching of command results
3. Spin detection from deduplication blocking legitimate queries

The cache is:
- Platform-agnostic (works with gh, az, gitlab, kubectl, etc.)
- Time-based (TTL per command type)
- Context-aware (respects working directory)
- Safe (only caches read-only commands by default)
"""

import hashlib
from victor.core.json_utils import json_dumps
import logging
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ShellCommandCache:
    """Thread-safe cache for shell command results with TTL."""

    def __init__(self, default_ttl_minutes: int = 5):
        """Initialize shell command cache.

        Args:
            default_ttl_minutes: Default time-to-live for cache entries
        """
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._lock = threading.RLock()
        self._default_ttl = timedelta(minutes=default_ttl_minutes)

        # Command-specific TTLs (in minutes)
        self._command_ttls: Dict[str, timedelta] = {
            # CI/CD queries - cache longer (results stable)
            "gh": timedelta(minutes=10),
            "az": timedelta(minutes=10),
            "gitlab": timedelta(minutes=10),
            "kubectl": timedelta(minutes=5),
            # Git commands - medium cache
            "git": timedelta(minutes=5),
            # System info - longer cache
            "uname": timedelta(minutes=30),
            "whoami": timedelta(minutes=30),
            "hostname": timedelta(minutes=30),
            # File listings - short cache (may change)
            "ls": timedelta(minutes=1),
            "find": timedelta(minutes=2),
        }

    def _get_ttl_for_command(self, command: str) -> timedelta:
        """Get TTL for a specific command type.

        Args:
            command: The command string

        Returns:
            TTL duration for this command type
        """
        # Extract base command
        parts = command.strip().split()
        if not parts:
            return self._default_ttl

        base_cmd = parts[0].lower()

        # Check for command-specific TTL
        for cmd_prefix, ttl in self._command_ttls.items():
            if base_cmd == cmd_prefix or base_cmd.startswith(cmd_prefix):
                return ttl

        return self._default_ttl

    def _generate_key(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate cache key from command, cwd, and environment.

        Args:
            command: The command string
            cwd: Current working directory
            env: Environment variables (only a subset matters)

        Returns:
            Cache key hash
        """
        # Only include environment variables that affect command output
        relevant_env = {}
        if env:
            for key in ["PATH", "HOME", "USER", "LANG"]:
                if key in env:
                    relevant_env[key] = env[key]

        key_data = {
            "command": command,
            "cwd": cwd or "",
            "env": relevant_env,
        }

        key_str = json_dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, command: str, cwd: Optional[str] = None) -> Optional[Tuple[int, str, str]]:
        """Get cached command result if available and not expired.

        Args:
            command: The command string
            cwd: Current working directory

        Returns:
            Tuple of (returncode, stdout, stderr) or None if not found/expired
        """
        key = self._generate_key(command, cwd)

        with self._lock:
            if key in self._cache:
                result, timestamp = self._cache[key]
                ttl = self._get_ttl_for_command(command)

                if datetime.now() - timestamp < ttl:
                    logger.debug(f"Cache HIT: {command[:60]}...")
                    return result
                else:
                    # Expired, remove
                    del self._cache[key]
                    logger.debug(f"Cache EXPIRED: {command[:60]}...")

        return None

    def set(self, command: str, result: Tuple[int, str, str], cwd: Optional[str] = None) -> None:
        """Cache a command result.

        Args:
            command: The command string
            result: Tuple of (returncode, stdout, stderr)
            cwd: Current working directory
        """
        key = self._generate_key(command, cwd)

        with self._lock:
            self._cache[key] = (result, datetime.now())
            logger.debug(f"Cached: {command[:60]}...")

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            logger.debug("Shell command cache cleared")

    def cleanup_expired(self) -> int:
        """Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        now = datetime.now()
        removed = 0

        with self._lock:
            expired_keys = []
            for key, (_, timestamp) in self._cache.items():
                # Use default TTL for cleanup (conservative)
                if now - timestamp >= self._default_ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                removed += 1

        return removed

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache stats
        """
        with self._lock:
            return {
                "entries": len(self._cache),
                "default_ttl_minutes": int(self._default_ttl.total_seconds() / 60),
                "command_ttls": {
                    cmd: int(ttl.total_seconds() / 60) for cmd, ttl in self._command_ttls.items()
                },
            }


# Global cache instance
_shell_cache: Optional[ShellCommandCache] = None
_cache_lock = threading.Lock()


def get_shell_cache() -> ShellCommandCache:
    """Get the global shell command cache instance.

    Returns:
        Global cache instance (singleton)
    """
    global _shell_cache

    if _shell_cache is None:
        with _cache_lock:
            if _shell_cache is None:
                _shell_cache = ShellCommandCache(default_ttl_minutes=5)

    return _shell_cache


def execute_with_cache(
    command: str,
    cwd: Optional[str] = None,
    shell: bool = True,
    timeout: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
    use_cache: bool = True,
) -> Tuple[int, str, str]:
    """Execute shell command with caching.

    Args:
        command: Command to execute
        cwd: Working directory
        shell: Whether to use shell
        timeout: Command timeout in seconds
        env: Environment variables
        use_cache: Whether to use cache (default: True)

    Returns:
        Tuple of (returncode, stdout, stderr)
    """
    cache = get_shell_cache()

    # Check cache first
    if use_cache:
        cached = cache.get(command, cwd)
        if cached is not None:
            return cached

    # Execute command
    try:
        result = subprocess.run(
            command,
            shell=shell,  # nosec B602 — shell param is caller-controlled, not user input
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        output = (result.returncode, result.stdout, result.stderr)

        # Cache successful results
        if use_cache and result.returncode == 0:
            cache.set(command, output, cwd)

        return output

    except subprocess.TimeoutExpired:
        error_msg = f"Command timed out after {timeout}s"
        logger.error(f"{error_msg}: {command[:60]}...")
        return (1, "", error_msg)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Command execution failed: {error_msg}")
        return (1, "", error_msg)


def clear_shell_cache() -> None:
    """Clear the shell command cache."""
    cache = get_shell_cache()
    cache.clear()


def get_shell_cache_stats() -> Dict[str, Any]:
    """Get shell command cache statistics.

    Returns:
        Cache statistics
    """
    cache = get_shell_cache()
    return cache.get_stats()


# Export for use in other modules
__all__ = [
    "ShellCommandCache",
    "get_shell_cache",
    "execute_with_cache",
    "clear_shell_cache",
    "get_shell_cache_stats",
]
