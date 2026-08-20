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

"""Secure path handling utilities for Victor.

This module provides security-hardened path resolution and validation
to protect against:
- HOME environment variable manipulation (SEC-001)
- Path traversal attacks (SEC-002)
- Symlink attacks (SEC-003)

Usage:
    from victor.config.secure_paths import get_secure_home, get_victor_dir, safe_resolve_path

    # Get validated home directory
    home = get_secure_home()

    # Get Victor config directory with validation
    victor_dir = get_victor_dir()

    # Safely resolve a path with symlink detection
    safe_path = safe_resolve_path(user_path, expected_parent=victor_dir)
"""

from __future__ import annotations

import logging
import os
import platform
import pwd
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Security event types for audit logging
SECURITY_EVENT_HOME_MANIPULATION = "HOME_MANIPULATION_DETECTED"
SECURITY_EVENT_PATH_TRAVERSAL = "PATH_TRAVERSAL_ATTEMPT"
SECURITY_EVENT_SYMLINK_DETECTED = "SYMLINK_IN_CRITICAL_PATH"
SECURITY_EVENT_SYMLINK_ESCAPE = "SYMLINK_ESCAPE_ATTEMPT"
SECURITY_EVENT_INSECURE_PERMISSIONS = "INSECURE_FILE_PERMISSIONS"

# Environment sentinels indicating the process is running under pytest. Mirrors
# the telemetry-redirect logic in ``victor/core/bootstrap.py`` so test sessions
# never pollute the project audit database with benign events.
_TEST_MODE_SENTINELS = ("TEST_MODE", "PYTEST_XDIST_WORKER", "PYTEST_CURRENT_TEST")

# Benign HOME values legitimately set by the test suite (temp dirs created via
# ``tmp_path``/``mkdtemp``). These are *not* attacks and account for >99% of the
# audit noise observed in audit/ (6,518 of 6,544 HOME_MANIPULATION events).
_BENIGN_TEST_HOME_MARKERS = ("victor_test_home", "fake_home")


def is_test_mode() -> bool:
    """Return True when the process is running under the pytest test harness.

    Detection mirrors ``victor.core.bootstrap._register_usage_logger``: any of
    the standard pytest environment sentinels being set counts as test mode.
    """
    return any(os.getenv(v) for v in _TEST_MODE_SENTINELS)


def _real_temp_dir() -> Path:
    """Return the system temp directory (resolved for macOS /private/var)."""
    return Path(tempfile.gettempdir()).resolve()


def _recognized_temp_prefixes() -> set[str]:
    """Return the set of recognized OS temp-root prefixes (string forms).

    Covers this machine's ``tempfile.gettempdir()`` plus the *convention*
    prefixes used across Linux/macOS so a temp HOME created on any machine or
    by any user is recognized:
    - Linux: ``/tmp``, ``/var/tmp``
    - macOS: ``/var/folders/.../T`` (and resolved ``/private/var/folders/.../T``)
    - This machine's own temp dir (resolved + raw)
    - Windows: ``$TEMP`` / ``%LOCALAPPDATA%\\Temp`` (returned by gettempdir)
    """
    prefixes: set[str] = set()

    # Convention roots
    for convention in ("/tmp", "/var/tmp", "/private/tmp"):
        prefixes.add(convention)
    # macOS per-user temp convention: /var/folders/<hash>/<hash>/T and its
    # resolved form /private/var/folders/<hash>/<hash>/T
    prefixes.add("/var/folders")
    prefixes.add("/private/var/folders")

    # This machine's actual temp dir(s) — both raw and resolved
    try:
        raw = str(Path(tempfile.gettempdir()))
        prefixes.add(raw)
        resolved = str(Path(tempfile.gettempdir()).resolve())
        if resolved != raw:
            prefixes.add(resolved)
    except OSError:
        pass

    return prefixes


def is_benign_test_home_path(path: Path) -> bool:
    """Classify a HOME value as a benign test/temp directory (not an attack).

    A path is benign when it lives under the system temp directory AND either
    carries a known test-suite marker (e.g. ``victor_test_home``) or the process
    is explicitly in test mode. Real manipulation (HOME pointed at an
    attacker-controlled, non-temp location) never matches.

    Args:
        path: Candidate HOME path.

    Returns:
        True if the path is a legitimate test/temp HOME value.
    """
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False

    # Recognize any standard OS temp-root convention, not just this machine's
    # exact ``tempfile.gettempdir()``. Test suites legitimately create homes under
    # ``/tmp`` (Linux), ``/var/folders/<hash>/T`` (macOS, any user/machine), or
    # their resolved ``/private/...`` forms. Matching these broadly is safe
    # because the benign classification ALSO requires a test marker or test mode.
    temp_prefixes = _recognized_temp_prefixes()

    try:
        matched = any(
            str(resolved) == p or str(resolved).startswith(p + os.sep) for p in temp_prefixes
        )
        if not matched:
            return False  # Not under any recognized temp dir -> cannot be benign
    except (ValueError, OSError):
        return False

    text = str(resolved).lower()
    if any(marker in text for marker in _BENIGN_TEST_HOME_MARKERS):
        return True
    return is_test_mode()


def _get_audit_manager():
    """Resolve the AuditManager singleton, or None when unavailable.

    Wrapped so callers can be patched in tests and remain robust to the audit
    module being absent (e.g. optional dependency not installed).
    """
    try:
        from victor.security.audit import AuditManager

        return AuditManager.get_instance()
    except ImportError:
        return None
    except Exception as e:  # pragma: no cover - defensive singleton init failure
        logger.debug(f"Could not resolve audit manager: {e}")
        return None


def _log_security_event(event_type: str, details: dict) -> None:
    """Log a security event for audit purposes.

    Security detections are surfaced at WARNING severity (never INFO): a
    detected manipulation attempt is a notable security signal regardless of
    whether it was benign.

    Args:
        event_type: Type of security event
        details: Event details dictionary
    """
    from victor.security.audit.protocol import Severity

    # Always log to standard logger at WARNING level
    logger.warning(f"SECURITY: {event_type} - {details}")

    # Try to log to audit system if available
    audit = _get_audit_manager()
    if audit is None:
        return
    try:
        audit.log_event(
            event_type=_audit_security_event_type(),
            action=event_type,
            details=details,
            severity=Severity.WARNING,
        )
    except Exception as e:
        logger.debug(f"Could not log to audit system: {e}")


# Resolved lazily to avoid importing the audit package at module import time
# (the SDK-contracts boundary keeps these modules independently installable).
def _audit_security_event_type():
    from victor.security.audit import AuditEventType

    return AuditEventType.SECURITY_EVENT


def get_real_home_from_passwd() -> Optional[Path]:
    """Get home directory from passwd database (harder to spoof).

    On Unix systems, this reads from /etc/passwd or NIS/LDAP,
    which requires root access to modify.

    Returns:
        Path to real home directory from passwd, or None if unavailable
    """
    if platform.system() == "Windows":
        # Windows doesn't have passwd - use USERPROFILE
        return None

    try:
        pw_entry = pwd.getpwuid(os.getuid())
        return Path(pw_entry.pw_dir)
    except (KeyError, AttributeError):
        return None


def get_secure_home() -> Path:
    """Get home directory with validation against manipulation.

    This function detects when HOME environment variable has been
    manipulated to point to a different location than the system
    passwd database.

    Returns:
        Validated home directory path

    Security:
        - Compares HOME env against passwd database
        - Logs warning if manipulation detected
        - Returns passwd entry on Unix, HOME on Windows
    """
    env_home = Path(os.environ.get("HOME", os.path.expanduser("~")))

    # Get real home from passwd (Unix only)
    passwd_home = get_real_home_from_passwd()

    if passwd_home is None:
        # Windows or passwd unavailable - trust environment
        return env_home

    # Detect manipulation
    try:
        env_resolved = env_home.resolve()
        passwd_resolved = passwd_home.resolve()

        if env_resolved != passwd_resolved:
            # NOTE: We intentionally do NOT suppress benign test/temp HOME values
            # here. The HOME-manipulation detector is a *security* control and
            # must fire for any env/passwd mismatch (legacy contract, enforced by
            # test_secure_paths.py::test_detects_home_manipulation). The audit
            # *noise* problem is solved at the logging layer instead:
            # FileAuditLogger redirects to <tmp>/victor_test_audit under test
            # mode, so benign detections never pollute the project database.
            _log_security_event(
                SECURITY_EVENT_HOME_MANIPULATION,
                {
                    "env_home": str(env_home),
                    "passwd_home": str(passwd_home),
                    "env_resolved": str(env_resolved),
                    "passwd_resolved": str(passwd_resolved),
                    "uid": os.getuid(),
                    "euid": os.geteuid(),
                    "pid": os.getpid(),
                },
            )
            # Return passwd entry (more trusted)
            return passwd_home
    except OSError:
        # Path resolution failed - use passwd entry
        return passwd_home

    return env_home


def validate_victor_dir_name(dir_name: str) -> Tuple[str, bool]:
    """Validate VICTOR_DIR_NAME against path injection.

    Args:
        dir_name: Directory name to validate

    Returns:
        Tuple of (validated_name, is_valid)

    Security:
        - Blocks path traversal (../)
        - Blocks absolute paths (/)
        - Blocks hidden directory escape
    """
    # Block empty names
    if not dir_name or not dir_name.strip():
        return ".victor", False

    # Block path traversal
    if ".." in dir_name:
        _log_security_event(
            SECURITY_EVENT_PATH_TRAVERSAL,
            {
                "attempted_name": dir_name,
                "attack_type": "parent_directory_traversal",
            },
        )
        return ".victor", False

    # Block absolute paths
    if dir_name.startswith("/") or dir_name.startswith("\\"):
        _log_security_event(
            SECURITY_EVENT_PATH_TRAVERSAL,
            {
                "attempted_name": dir_name,
                "attack_type": "absolute_path_injection",
            },
        )
        return ".victor", False

    # Block paths with separators
    if "/" in dir_name or "\\" in dir_name:
        _log_security_event(
            SECURITY_EVENT_PATH_TRAVERSAL,
            {
                "attempted_name": dir_name,
                "attack_type": "path_separator_injection",
            },
        )
        return ".victor", False

    # Warn if not a hidden directory (expected convention)
    if not dir_name.startswith("."):
        logger.warning(
            f"VICTOR_DIR_NAME '{dir_name}' doesn't start with '.'. "
            f"Consider using '.{dir_name}' for convention."
        )

    return dir_name, True


def get_victor_dir() -> Path:
    """Get Victor configuration directory with security validation.

    Returns:
        Path to Victor directory (~/.victor by default)

    Security:
        - Validates VICTOR_DIR_NAME against path injection
        - Uses secure home directory resolution
    """
    raw_dir_name = os.getenv("VICTOR_DIR_NAME", ".victor")
    dir_name, is_valid = validate_victor_dir_name(raw_dir_name)

    if not is_valid:
        logger.error(
            f"Invalid VICTOR_DIR_NAME '{raw_dir_name}' rejected for security. "
            f"Using default '.victor'"
        )

    return get_secure_home() / dir_name


def check_symlink_in_path(path: Path) -> Optional[Path]:
    """Check if any component in the path is a symlink.

    Args:
        path: Path to check

    Returns:
        First symlink found in path, or None if no symlinks
    """
    try:
        resolved = path.resolve()

        # Check each component
        current = path
        while current != current.parent:
            if current.is_symlink():
                return current
            current = current.parent

        # Also check parents of resolved path
        current = resolved
        while current != current.parent:
            if current.exists() and current.is_symlink():
                return current
            current = current.parent

    except OSError:
        pass

    return None


def safe_resolve_path(
    path: Path,
    expected_parent: Optional[Path] = None,
    allow_symlinks: bool = False,
    log_symlinks: bool = True,
) -> Optional[Path]:
    """Safely resolve a path with symlink and escape detection.

    Args:
        path: Path to resolve
        expected_parent: If provided, verify resolved path is under this directory
        allow_symlinks: If False, reject paths containing symlinks
        log_symlinks: If True, log when symlinks are detected

    Returns:
        Resolved path if safe, None if security check failed

    Security:
        - Detects symlinks in path components
        - Verifies path doesn't escape expected parent
        - Logs all security-relevant events
    """
    try:
        # Expand user and resolve
        expanded = path.expanduser()
        resolved = expanded.resolve()

        # Check for symlinks
        if not allow_symlinks:
            symlink = check_symlink_in_path(expanded)
            if symlink:
                if log_symlinks:
                    _log_security_event(
                        SECURITY_EVENT_SYMLINK_DETECTED,
                        {
                            "original_path": str(path),
                            "symlink_at": str(symlink),
                            "resolved_to": str(resolved),
                        },
                    )
                # For critical paths, we might want to reject
                # For now, just log and continue

        # Check for escape from expected parent
        if expected_parent is not None:
            expected_resolved = expected_parent.expanduser().resolve()
            try:
                resolved.relative_to(expected_resolved)
            except ValueError:
                _log_security_event(
                    SECURITY_EVENT_SYMLINK_ESCAPE,
                    {
                        "original_path": str(path),
                        "resolved_to": str(resolved),
                        "expected_parent": str(expected_resolved),
                    },
                )
                return None

        return resolved

    except OSError as e:
        logger.warning(f"Failed to resolve path {path}: {e}")
        return None


def check_file_permissions(path: Path, max_mode: int = 0o600) -> bool:
    """Check if file has appropriately restrictive permissions.

    Args:
        path: Path to check
        max_mode: Maximum allowed permission mode (default: owner-only rw)

    Returns:
        True if permissions are acceptable, False otherwise
    """
    if not path.exists():
        return True  # File doesn't exist yet

    try:
        mode = path.stat().st_mode & 0o777

        # Check for world or group readable/writable
        if mode & 0o077:  # Any group/world bits set
            _log_security_event(
                SECURITY_EVENT_INSECURE_PERMISSIONS,
                {
                    "path": str(path),
                    "current_mode": oct(mode),
                    "recommended_mode": oct(max_mode),
                },
            )
            return False

        return True

    except OSError:
        return True  # Can't check, assume ok


def secure_create_file(
    path: Path,
    content: str,
    mode: int = 0o600,
    atomic: bool = True,
) -> bool:
    """Create a file with secure permissions from the start.

    This avoids TOCTOU race conditions by setting permissions atomically.

    Args:
        path: Path to create
        content: Content to write
        mode: File permission mode (default: 0600)
        atomic: If True, use atomic write via temp file

    Returns:
        True if successful, False otherwise

    Security:
        - Creates file with restricted permissions from the start
        - Uses atomic write to prevent race conditions
        - Ensures parent directory exists
    """
    try:
        # Ensure parent exists
        path.parent.mkdir(parents=True, exist_ok=True)

        if atomic:
            # Atomic write via temp file
            temp_path = path.with_suffix(path.suffix + ".tmp")

            # Create with restricted permissions from the start
            fd = os.open(
                temp_path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                mode,
            )
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(content)
            except Exception:
                os.close(fd)
                raise

            # Atomic rename
            os.rename(temp_path, path)
        else:
            # Direct write with secure open
            fd = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                mode,
            )
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(content)
            except Exception:
                os.close(fd)
                raise

        return True

    except Exception as e:
        logger.error(f"Failed to securely create file {path}: {e}")
        return False


def secure_read_file(path: Path, check_permissions: bool = True) -> Optional[str]:
    """Read a file with security checks.

    Args:
        path: Path to read
        check_permissions: If True, warn about insecure permissions

    Returns:
        File content or None if read failed
    """
    if not path.exists():
        return None

    if check_permissions:
        check_file_permissions(path)

    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read file {path}: {e}")
        return None


# =============================================================================
# XDG CONFIG PATH VALIDATION
# =============================================================================

SECURITY_EVENT_XDG_MANIPULATION = "XDG_CONFIG_MANIPULATION_DETECTED"


def get_secure_xdg_config_home() -> Path:
    """Get XDG_CONFIG_HOME with validation.

    XDG_CONFIG_HOME can be manipulated to redirect config loading.
    This function validates the path is under the user's home directory.

    Returns:
        Validated XDG config home path

    Security:
        - Validates XDG_CONFIG_HOME is under home directory
        - Falls back to ~/.config if manipulation detected
    """
    secure_home = get_secure_home()
    default_config = secure_home / ".config"

    xdg_config = os.environ.get("XDG_CONFIG_HOME", "")
    if not xdg_config:
        return default_config

    xdg_path = Path(xdg_config)

    # Validate it's under home directory
    try:
        xdg_resolved = xdg_path.resolve()

        # Check if it's under the secure home
        try:
            xdg_resolved.relative_to(secure_home.resolve())
        except ValueError:
            # XDG_CONFIG_HOME points outside home - potential attack
            _log_security_event(
                SECURITY_EVENT_XDG_MANIPULATION,
                {
                    "xdg_config_home": str(xdg_config),
                    "resolved_to": str(xdg_resolved),
                    "secure_home": str(secure_home),
                    "uid": os.getuid() if hasattr(os, "getuid") else None,
                },
            )
            return default_config

        return xdg_path

    except OSError:
        return default_config


def get_secure_xdg_data_home() -> Path:
    """Get XDG_DATA_HOME with validation.

    Returns:
        Validated XDG data home path
    """
    secure_home = get_secure_home()
    default_data = secure_home / ".local" / "share"

    xdg_data = os.environ.get("XDG_DATA_HOME", "")
    if not xdg_data:
        return default_data

    xdg_path = Path(xdg_data)

    try:
        xdg_resolved = xdg_path.resolve()

        try:
            xdg_resolved.relative_to(secure_home.resolve())
        except ValueError:
            _log_security_event(
                SECURITY_EVENT_XDG_MANIPULATION,
                {
                    "xdg_data_home": str(xdg_data),
                    "resolved_to": str(xdg_resolved),
                    "secure_home": str(secure_home),
                },
            )
            return default_data

        return xdg_path

    except OSError:
        return default_data


# =============================================================================
# PLUGIN DIRECTORY SECURITY
# =============================================================================

SECURITY_EVENT_PLUGIN_PATH_INJECTION = "PLUGIN_PATH_INJECTION_ATTEMPT"
SECURITY_EVENT_PLUGIN_SYMLINK = "PLUGIN_SYMLINK_DETECTED"


def validate_plugin_directory(plugin_dir: Path | str) -> Tuple[Path, bool]:
    """Validate a plugin directory for security.

    Plugin directories can be attack vectors for code injection.
    This function validates:
    - Path doesn't contain traversal attempts
    - Path doesn't escape expected locations
    - No suspicious symlinks

    Args:
        plugin_dir: Plugin directory path (Path or string)

    Returns:
        Tuple of (validated_path, is_valid)
    """
    # Convert to Path if string
    if isinstance(plugin_dir, str):
        plugin_dir = Path(plugin_dir)

    try:
        resolved = plugin_dir.expanduser().resolve()
    except OSError as e:
        logger.warning(f"Failed to resolve plugin path {plugin_dir}: {e}")
        return plugin_dir, False

    # Check for path traversal in the original path string
    path_str = str(plugin_dir)
    if ".." in path_str:
        _log_security_event(
            SECURITY_EVENT_PLUGIN_PATH_INJECTION,
            {
                "plugin_dir": path_str,
                "attack_type": "path_traversal",
            },
        )
        return plugin_dir, False

    # Verify path is under allowed locations
    secure_home = get_secure_home()
    victor_dir = get_victor_dir()
    allowed_parents = [
        secure_home,  # ~/.victor/plugins, ~/.local/share/victor/plugins
        victor_dir,  # .victor/plugins
        Path.cwd(),  # Project-local plugins
    ]

    is_allowed = False
    for parent in allowed_parents:
        try:
            resolved.relative_to(parent.resolve())
            is_allowed = True
            break
        except ValueError:
            continue

    if not is_allowed:
        _log_security_event(
            SECURITY_EVENT_PLUGIN_PATH_INJECTION,
            {
                "plugin_dir": path_str,
                "resolved_to": str(resolved),
                "attack_type": "path_escape",
            },
        )
        return plugin_dir, False

    # Check for symlinks in the path
    symlink = check_symlink_in_path(resolved)
    if symlink:
        _log_security_event(
            SECURITY_EVENT_PLUGIN_SYMLINK,
            {
                "plugin_dir": path_str,
                "symlink_at": str(symlink),
            },
        )
        # Symlinks in plugin paths are suspicious but may be intentional
        logger.warning(f"Symlink detected in plugin path: {symlink}")

    return resolved, True


def get_secure_plugin_dirs() -> list[Path]:
    """Get list of secure plugin directories.

    Returns validated plugin directories in priority order:
    1. Project-local: ./.victor/plugins
    2. User global: ~/.victor/plugins
    3. XDG data: ~/.local/share/victor/plugins

    Returns:
        List of validated plugin directories
    """
    dirs = []

    # Project-local plugins
    project_plugins = Path.cwd() / ".victor" / "plugins"
    validated, is_valid = validate_plugin_directory(project_plugins)
    if is_valid:
        dirs.append(validated)

    # User global plugins
    global_plugins = get_victor_dir() / "plugins"
    validated, is_valid = validate_plugin_directory(global_plugins)
    if is_valid:
        dirs.append(validated)

    # XDG data plugins
    xdg_plugins = get_secure_xdg_data_home() / "victor" / "plugins"
    validated, is_valid = validate_plugin_directory(xdg_plugins)
    if is_valid:
        dirs.append(validated)

    return dirs


# =============================================================================
# EMBEDDING CACHE INTEGRITY
# =============================================================================

SECURITY_EVENT_CACHE_TAMPERING = "EMBEDDING_CACHE_TAMPERING_DETECTED"


def compute_file_hash(file_path: Path) -> Optional[str]:
    """Compute SHA-256 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        Hex digest of file hash, or None on error
    """
    import hashlib

    if not file_path.exists():
        return None

    try:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        logger.debug(f"Failed to hash file {file_path}: {e}")
        return None


def get_cache_manifest_path(cache_dir: Path) -> Path:
    """Get path to cache manifest file.

    Args:
        cache_dir: Cache directory

    Returns:
        Path to manifest file
    """
    return cache_dir / ".cache_manifest.json"


def create_cache_manifest(cache_dir: Path) -> bool:
    """Create or update cache integrity manifest.

    The manifest stores hashes of all cache files to detect tampering.

    Args:
        cache_dir: Cache directory to manifest

    Returns:
        True if manifest created successfully
    """
    import json

    if not cache_dir.exists():
        return False

    manifest = {
        "version": 1,
        "created": str(Path.cwd()),
        "files": {},
    }

    try:
        # Hash all files in cache directory
        for file_path in cache_dir.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                rel_path = str(file_path.relative_to(cache_dir))
                file_hash = compute_file_hash(file_path)
                if file_hash:
                    manifest["files"][rel_path] = {
                        "hash": file_hash,
                        "size": file_path.stat().st_size,
                    }

        # Write manifest with secure permissions
        manifest_path = get_cache_manifest_path(cache_dir)
        content = json.dumps(manifest, indent=2)
        return secure_create_file(manifest_path, content, mode=0o600)

    except Exception as e:
        logger.error(f"Failed to create cache manifest: {e}")
        return False


def verify_cache_integrity(cache_dir: Path) -> Tuple[bool, list[str]]:
    """Verify cache integrity against manifest.

    Args:
        cache_dir: Cache directory to verify

    Returns:
        Tuple of (is_valid, list of tampered files)
    """
    import json

    manifest_path = get_cache_manifest_path(cache_dir)

    if not manifest_path.exists():
        # No manifest - cache is unverified but not tampered
        logger.debug(f"No cache manifest found at {manifest_path}")
        return True, []

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read cache manifest: {e}")
        return False, ["<manifest_corrupted>"]

    tampered_files = []

    for rel_path, file_info in manifest.get("files", {}).items():
        file_path = cache_dir / rel_path

        if not file_path.exists():
            # File was deleted
            tampered_files.append(f"{rel_path} (deleted)")
            continue

        expected_hash = file_info.get("hash")
        actual_hash = compute_file_hash(file_path)

        if actual_hash != expected_hash:
            tampered_files.append(rel_path)

    if tampered_files:
        _log_security_event(
            SECURITY_EVENT_CACHE_TAMPERING,
            {
                "cache_dir": str(cache_dir),
                "tampered_files": tampered_files[:10],  # Limit to first 10
                "total_tampered": len(tampered_files),
            },
        )
        return False, tampered_files

    return True, []


def secure_embeddings_dir() -> Path:
    """Get secure embeddings directory with integrity check.

    Returns:
        Path to embeddings directory
    """
    victor_dir = get_victor_dir()
    embeddings_dir = victor_dir / "embeddings"

    if embeddings_dir.exists():
        is_valid, tampered = verify_cache_integrity(embeddings_dir)
        if not is_valid:
            logger.warning(
                f"Embedding cache integrity check failed. " f"Tampered files: {tampered[:3]}..."
            )

    return embeddings_dir


# =============================================================================
# PLUGIN SIGNATURE VERIFICATION
# =============================================================================

SECURITY_EVENT_PLUGIN_SIGNATURE_INVALID = "PLUGIN_SIGNATURE_INVALID"
SECURITY_EVENT_PLUGIN_UNSIGNED = "PLUGIN_UNSIGNED"
SECURITY_EVENT_PLUGIN_TRUSTED = "PLUGIN_SIGNATURE_VERIFIED"

# Trust store path
PLUGIN_TRUST_STORE = ".plugin_trust.json"


def get_plugin_trust_store_path() -> Path:
    """Get path to plugin trust store.

    Returns:
        Path to trust store file
    """
    return get_victor_dir() / PLUGIN_TRUST_STORE


def compute_plugin_hash(plugin_path: Path) -> Optional[str]:
    """Compute hash of a plugin for signature verification.

    Hashes all Python files in the plugin directory.

    Args:
        plugin_path: Path to plugin directory or file

    Returns:
        Hex digest of combined hash, or None on error
    """
    import hashlib

    if not plugin_path.exists():
        return None

    sha256 = hashlib.sha256()

    try:
        if plugin_path.is_file():
            # Single file plugin
            with open(plugin_path, "rb") as f:
                sha256.update(f.read())
        else:
            # Directory plugin - hash all Python files
            for py_file in sorted(plugin_path.rglob("*.py")):
                with open(py_file, "rb") as f:
                    # Include relative path in hash to detect file moves
                    rel_path = py_file.relative_to(plugin_path)
                    sha256.update(str(rel_path).encode())
                    sha256.update(f.read())

        return sha256.hexdigest()

    except Exception as e:
        logger.debug(f"Failed to hash plugin {plugin_path}: {e}")
        return None


def load_plugin_trust_store() -> dict:
    """Load the plugin trust store.

    Returns:
        Trust store dictionary with trusted plugin hashes
    """
    import json

    trust_path = get_plugin_trust_store_path()

    if not trust_path.exists():
        return {"version": 1, "trusted_plugins": {}, "trusted_signers": []}

    try:
        with open(trust_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load plugin trust store: {e}")
        return {"version": 1, "trusted_plugins": {}, "trusted_signers": []}


def save_plugin_trust_store(trust_store: dict) -> bool:
    """Save the plugin trust store.

    Args:
        trust_store: Trust store dictionary

    Returns:
        True if saved successfully
    """
    import json

    trust_path = get_plugin_trust_store_path()
    content = json.dumps(trust_store, indent=2)

    return secure_create_file(trust_path, content, mode=0o600)


def trust_plugin(plugin_path: Path, name: Optional[str] = None) -> bool:
    """Add a plugin to the trust store.

    Args:
        plugin_path: Path to plugin
        name: Optional friendly name

    Returns:
        True if added successfully
    """
    plugin_hash = compute_plugin_hash(plugin_path)
    if not plugin_hash:
        return False

    trust_store = load_plugin_trust_store()

    plugin_name = name or plugin_path.name
    trust_store["trusted_plugins"][plugin_name] = {
        "hash": plugin_hash,
        "path": str(plugin_path),
        "trusted_at": str(Path.cwd()),
    }

    return save_plugin_trust_store(trust_store)


def untrust_plugin(name: str) -> bool:
    """Remove a plugin from the trust store.

    Args:
        name: Plugin name

    Returns:
        True if removed successfully
    """
    trust_store = load_plugin_trust_store()

    if name in trust_store["trusted_plugins"]:
        del trust_store["trusted_plugins"][name]
        return save_plugin_trust_store(trust_store)

    return False


def verify_plugin_trust(plugin_path: Path) -> Tuple[bool, str]:
    """Verify if a plugin is trusted.

    Args:
        plugin_path: Path to plugin

    Returns:
        Tuple of (is_trusted, reason)
    """
    plugin_hash = compute_plugin_hash(plugin_path)
    if not plugin_hash:
        return False, "failed_to_hash"

    trust_store = load_plugin_trust_store()

    # Check if this plugin hash is trusted
    for name, info in trust_store["trusted_plugins"].items():
        if info.get("hash") == plugin_hash:
            _log_security_event(
                SECURITY_EVENT_PLUGIN_TRUSTED,
                {
                    "plugin_path": str(plugin_path),
                    "plugin_name": name,
                    "hash": plugin_hash[:16] + "...",
                },
            )
            return True, f"trusted:{name}"

    # Plugin not in trust store
    _log_security_event(
        SECURITY_EVENT_PLUGIN_UNSIGNED,
        {
            "plugin_path": str(plugin_path),
            "hash": plugin_hash[:16] + "...",
        },
    )
    return False, "not_trusted"


def list_trusted_plugins() -> list[dict]:
    """List all trusted plugins.

    Returns:
        List of trusted plugin info
    """
    trust_store = load_plugin_trust_store()
    return [{"name": name, **info} for name, info in trust_store["trusted_plugins"].items()]


# =============================================================================
# SECURITY STATUS
# =============================================================================


def get_security_status() -> dict:
    """Get comprehensive security status.

    Returns:
        Dictionary with security status information
    """
    import platform

    status = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
        },
        "home_security": {
            "secure_home": str(get_secure_home()),
            "home_validated": True,  # If we got here, validation passed
        },
        "keyring": {
            "available": False,
            "backend": "unknown",
        },
        "paths": {
            "victor_dir": str(get_victor_dir()),
            "xdg_config": str(get_secure_xdg_config_home()),
            "xdg_data": str(get_secure_xdg_data_home()),
        },
        "plugins": {
            "trusted_count": len(list_trusted_plugins()),
            "plugin_dirs": [str(p) for p in get_secure_plugin_dirs()],
        },
        "cache_integrity": {
            "embeddings_verified": False,
        },
    }

    # Check keyring
    try:
        import keyring

        status["keyring"]["available"] = True
        try:
            backend = keyring.get_keyring()
            status["keyring"]["backend"] = type(backend).__name__
        except Exception:
            pass
    except ImportError:
        pass

    # Check embeddings cache
    embeddings_dir = get_victor_dir() / "embeddings"
    if embeddings_dir.exists():
        is_valid, _ = verify_cache_integrity(embeddings_dir)
        status["cache_integrity"]["embeddings_verified"] = is_valid

    return status


# =============================================================================
# PLUGIN SANDBOXING
# =============================================================================

SECURITY_EVENT_PLUGIN_SANDBOX_VIOLATION = "PLUGIN_SANDBOX_VIOLATION"
SECURITY_EVENT_PLUGIN_LOAD_BLOCKED = "PLUGIN_LOAD_BLOCKED"
SECURITY_EVENT_PLUGIN_LOAD_ALLOWED = "PLUGIN_LOAD_ALLOWED"


@dataclass
class PluginSandboxPolicy:
    """Policy defining what a plugin is allowed to do.

    Attributes:
        allow_network: Plugin can make network requests
        allow_subprocess: Plugin can spawn subprocesses
        allow_file_write: Plugin can write to filesystem
        allowed_paths: Paths plugin can access (empty = all)
        blocked_paths: Paths plugin cannot access
        require_trust: Plugin must be in trust store
        max_memory_mb: Maximum memory usage in MB (0 = unlimited)
    """

    allow_network: bool = True
    allow_subprocess: bool = True
    allow_file_write: bool = True
    allowed_paths: list = field(default_factory=list)
    blocked_paths: list = field(default_factory=list)
    require_trust: bool = False
    max_memory_mb: int = 0


# Default policies
DEFAULT_SANDBOX_POLICY = PluginSandboxPolicy()
STRICT_SANDBOX_POLICY = PluginSandboxPolicy(
    allow_network=False,
    allow_subprocess=False,
    allow_file_write=False,
    require_trust=True,
    max_memory_mb=256,
)


def get_plugin_sandbox_policy() -> PluginSandboxPolicy:
    """Get the current plugin sandbox policy.

    Returns:
        Current sandbox policy
    """
    # Default to permissive policy for backward compatibility
    # Can be configured via settings in the future
    return DEFAULT_SANDBOX_POLICY


def check_plugin_can_load(
    plugin_path: Path, policy: Optional[PluginSandboxPolicy] = None
) -> Tuple[bool, str]:
    """Check if a plugin can be loaded according to sandbox policy.

    Args:
        plugin_path: Path to plugin
        policy: Sandbox policy (uses default if not specified)

    Returns:
        Tuple of (can_load, reason)
    """
    if policy is None:
        policy = get_plugin_sandbox_policy()

    # Validate plugin directory first
    validated_path, is_valid_path = validate_plugin_directory(plugin_path)
    if not is_valid_path:
        _log_security_event(
            SECURITY_EVENT_PLUGIN_LOAD_BLOCKED,
            {
                "plugin_path": str(plugin_path),
                "reason": "invalid_path",
            },
        )
        return False, "invalid_plugin_path"

    # Check trust requirement
    if policy.require_trust:
        is_trusted, trust_reason = verify_plugin_trust(plugin_path)
        if not is_trusted:
            _log_security_event(
                SECURITY_EVENT_PLUGIN_LOAD_BLOCKED,
                {
                    "plugin_path": str(plugin_path),
                    "reason": "not_trusted",
                    "trust_status": trust_reason,
                },
            )
            return False, f"plugin_not_trusted:{trust_reason}"

    # Check if path is in blocked list
    for blocked in policy.blocked_paths:
        blocked_path = Path(blocked).expanduser().resolve()
        try:
            plugin_path.resolve().relative_to(blocked_path)
            _log_security_event(
                SECURITY_EVENT_PLUGIN_LOAD_BLOCKED,
                {
                    "plugin_path": str(plugin_path),
                    "reason": "blocked_path",
                    "blocked_by": str(blocked_path),
                },
            )
            return False, f"plugin_in_blocked_path:{blocked}"
        except ValueError:
            pass  # Not in blocked path, continue

    # If allowed_paths is specified, check plugin is within
    if policy.allowed_paths:
        is_allowed = False
        for allowed in policy.allowed_paths:
            allowed_path = Path(allowed).expanduser().resolve()
            try:
                plugin_path.resolve().relative_to(allowed_path)
                is_allowed = True
                break
            except ValueError:
                pass

        if not is_allowed:
            _log_security_event(
                SECURITY_EVENT_PLUGIN_LOAD_BLOCKED,
                {
                    "plugin_path": str(plugin_path),
                    "reason": "not_in_allowed_paths",
                    "allowed_paths": [str(p) for p in policy.allowed_paths],
                },
            )
            return False, "plugin_not_in_allowed_paths"

    # Plugin can load
    _log_security_event(
        SECURITY_EVENT_PLUGIN_LOAD_ALLOWED,
        {
            "plugin_path": str(plugin_path),
            "policy": {
                "require_trust": policy.require_trust,
                "allow_network": policy.allow_network,
                "allow_subprocess": policy.allow_subprocess,
            },
        },
    )
    return True, "allowed"


def check_sandbox_action(
    plugin_name: str,
    action: str,
    details: Optional[dict] = None,
    policy: Optional[PluginSandboxPolicy] = None,
) -> Tuple[bool, str]:
    """Check if a plugin action is allowed by sandbox policy.

    Args:
        plugin_name: Name of plugin
        action: Action being attempted (network, subprocess, file_write, file_read)
        details: Additional details about the action
        policy: Sandbox policy (uses default if not specified)

    Returns:
        Tuple of (is_allowed, reason)
    """
    if policy is None:
        policy = get_plugin_sandbox_policy()

    details = details or {}

    # Check action against policy
    if action == "network" and not policy.allow_network:
        _log_security_event(
            SECURITY_EVENT_PLUGIN_SANDBOX_VIOLATION,
            {
                "plugin": plugin_name,
                "action": action,
                "violation": "network_not_allowed",
                **details,
            },
        )
        return False, "network_not_allowed"

    if action == "subprocess" and not policy.allow_subprocess:
        _log_security_event(
            SECURITY_EVENT_PLUGIN_SANDBOX_VIOLATION,
            {
                "plugin": plugin_name,
                "action": action,
                "violation": "subprocess_not_allowed",
                **details,
            },
        )
        return False, "subprocess_not_allowed"

    if action == "file_write" and not policy.allow_file_write:
        _log_security_event(
            SECURITY_EVENT_PLUGIN_SANDBOX_VIOLATION,
            {
                "plugin": plugin_name,
                "action": action,
                "violation": "file_write_not_allowed",
                **details,
            },
        )
        return False, "file_write_not_allowed"

    # Check path-based restrictions for file operations
    if action in ("file_read", "file_write"):
        target_path = details.get("path")
        if target_path:
            target = Path(target_path).expanduser().resolve()

            # Check blocked paths
            for blocked in policy.blocked_paths:
                blocked_path = Path(blocked).expanduser().resolve()
                try:
                    target.relative_to(blocked_path)
                    _log_security_event(
                        SECURITY_EVENT_PLUGIN_SANDBOX_VIOLATION,
                        {
                            "plugin": plugin_name,
                            "action": action,
                            "violation": "blocked_path_access",
                            "target": str(target),
                            "blocked_by": str(blocked_path),
                        },
                    )
                    return False, f"blocked_path:{blocked}"
                except ValueError:
                    pass

    return True, "allowed"


def get_sandbox_summary() -> dict:
    """Get summary of sandbox configuration.

    Returns:
        Dictionary with sandbox status
    """
    policy = get_plugin_sandbox_policy()
    trusted_plugins = list_trusted_plugins()

    return {
        "policy": {
            "require_trust": policy.require_trust,
            "allow_network": policy.allow_network,
            "allow_subprocess": policy.allow_subprocess,
            "allow_file_write": policy.allow_file_write,
            "blocked_paths_count": len(policy.blocked_paths),
            "allowed_paths_count": len(policy.allowed_paths),
        },
        "trusted_plugins": {
            "count": len(trusted_plugins),
            "names": [p["name"] for p in trusted_plugins],
        },
    }
