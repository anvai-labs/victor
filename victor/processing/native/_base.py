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

"""Shared state for native extension modules.

This module holds the native extension availability flag and reference,
shared by all submodules in victor.processing.native.
"""

from __future__ import annotations

import logging
import os
from types import ModuleType
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import the native extension
_NATIVE_AVAILABLE = False
_native: ModuleType | None = None

try:
    import victor_native as _victor_native

    _native = _victor_native
    _NATIVE_AVAILABLE = True
    version = getattr(_native, "__version__", None)
    if isinstance(version, str):
        logger.info("Native extensions loaded (version %s)", version)
    else:
        logger.info("Native extensions loaded")
except ImportError:
    logger.debug("Native extensions not available, using pure Python fallback")
    # One-time hint: suggest installing pre-built wheels for 2-50x speedups.
    # Only shown at INFO level (visible in console), not on every import.
    import sys

    if not getattr(sys, "_victor_native_hint_shown", False):
        sys._victor_native_hint_shown = True
        logger.info(
            "Tip: install Rust native extensions for 2-50x faster tokenization, "
            "similarity, and pattern matching — pip install 'victor-ai[native]'"
        )


# Operations where Python is faster than Rust based on benchmarks.
# "Python" here means NumPy+BLAS, which is no longer a core dependency: these
# operations prefer numpy ONLY when it is importable (see
# numpy_accelerator_available); without numpy they dispatch to the native
# engine like everything else. See
# victor/processing/native/accelerator.py for benchmark data.
_PYTHON_PREFERRED_OPERATIONS = frozenset(
    {
        "batch_cosine_similarity",  # NumPy+BLAS is ~6x faster for batch operations
        "similarity_matrix",  # NumPy matmul is ~4x faster
    }
)

_NUMPY_ACCELERATOR_CACHE: bool | None = None


def numpy_accelerator_available() -> bool:
    """True when numpy is importable — WITHOUT importing it.

    A cached ``find_spec`` probe: the accelerator decision runs per dispatch
    call, so the check must stay import-free and O(1) after the first call.
    """
    global _NUMPY_ACCELERATOR_CACHE
    if _NUMPY_ACCELERATOR_CACHE is None:
        import importlib.util

        try:
            # find_spec raises ValueError when sys.modules["numpy"] is None
            # (the bare-install simulation used by tests) — treat as absent.
            _NUMPY_ACCELERATOR_CACHE = importlib.util.find_spec("numpy") is not None
        except (ImportError, ValueError):
            _NUMPY_ACCELERATOR_CACHE = False
    return _NUMPY_ACCELERATOR_CACHE


def reset_numpy_probe_for_tests() -> None:
    """Clear the cached numpy probe (tests install/uninstall numpy fakes)."""
    global _NUMPY_ACCELERATOR_CACHE
    _NUMPY_ACCELERATOR_CACHE = None


def is_native_available() -> bool:
    """Check if native Rust extensions are available."""
    return _NATIVE_AVAILABLE


def should_use_native_for_operation(operation_name: str) -> bool:
    """Check if native Rust should be used for a given operation.

    Returns False if:
    - Native extensions are not available, OR
    - The operation is in PYTHON_PREFERRED_OPERATIONS (Python is faster)

    This allows module-level dispatch to respect benchmark data without
    the overhead of importing the full accelerator module.

    Args:
        operation_name: Name of the operation (e.g., "batch_cosine_similarity")

    Returns:
        True if native Rust should be used, False otherwise
    """
    if not _NATIVE_AVAILABLE:
        return False
    if operation_name in _PYTHON_PREFERRED_OPERATIONS:
        # NumPy+BLAS wins these only while numpy is importable; a bare install
        # (no numpy) dispatches batches to the native engine instead.
        return not numpy_accelerator_available()
    return True


def get_native_version() -> Optional[str]:
    """Get the version of the native extension, if available."""
    if _native is not None:
        version = getattr(_native, "__version__", None)
        if isinstance(version, str):
            return version
        if version is not None:
            return str(version)
    return None


def _major_minor(version: object) -> Optional[tuple[int, int]]:
    """Return (major, minor) for a dotted version string, or None if unparseable."""
    if not isinstance(version, str):
        return None
    parts = version.split(".")
    if len(parts) < 2:
        return None
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        return None


def evaluate_native_version(
    native_version: object, package_version: object, *, skip_check: bool = False
) -> str:
    """Classify a loaded native extension against the installed victor-ai version.

    The native wheel releases in lockstep with victor-ai (same version line).
    A stale ``victor_native`` wheel silently executing under newer Python code
    is a recurring distribution hazard: on 2026-11-06 a venv carrying
    victor_native 0.15.0 under victor-ai 0.26.x pinned a chat session at 100%
    CPU inside a native dedup livelock that current Rust had already fixed.

    Returns one of: ``"ok"`` (identical versions), ``"patch-drift"`` (same
    major.minor, different patch), ``"stale"`` (native older), ``"newer"``
    (native newer), or ``"unknown"`` (either version missing/unparseable).
    The native extension stays enabled in every case: several modules import
    ``victor_native`` directly, so a partial disable here would split behavior
    across code paths. Import-time logging below surfaces ``"stale"`` loudly
    so drift is seen instead of silently riding an old binary.

    Args:
        native_version: Version reported by the loaded ``victor_native``.
        package_version: Installed ``victor-ai`` version (may be None).
        skip_check: Escape hatch for debugging (VICTOR_NATIVE_SKIP_VERSION_CHECK).
    """
    if skip_check:
        return "skipped"
    native_mm = _major_minor(native_version)
    package_mm = _major_minor(package_version)
    if native_mm is None or package_mm is None:
        return "unknown"
    if native_mm == package_mm:
        return "ok" if native_version == package_version else "patch-drift"
    return "stale" if native_mm < package_mm else "newer"


def _installed_package_version() -> Optional[str]:
    """Best-effort installed victor-ai version, without importing victor."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("victor-ai")
        except PackageNotFoundError:
            return None
    except Exception:  # pragma: no cover - defensive: metadata itself broken
        return None


def get_native_version_status() -> dict:
    """Return the recorded native-vs-package version comparison.

    Keys: ``state`` (``ok`` | ``patch-drift`` | ``stale`` | ``newer`` |
    ``unknown`` | ``skipped``), ``native_version``, ``package_version``.
    Intended for diagnostics surfaced by a future ``victor doctor`` command.
    """
    return dict(_NATIVE_VERSION_STATUS)


# Import-time drift check. Recorded for get_native_version_status(); logged so
# a stale native wheel is impossible to miss (see evaluate_native_version).
_NATIVE_VERSION_STATUS: dict = {"state": "unchecked"}
if _NATIVE_AVAILABLE:
    _pkg_version = _installed_package_version()
    _NATIVE_VERSION_STATUS = {
        "state": evaluate_native_version(
            get_native_version(),
            _pkg_version,
            skip_check=os.environ.get("VICTOR_NATIVE_SKIP_VERSION_CHECK") == "1",
        ),
        "native_version": get_native_version(),
        "package_version": _pkg_version,
    }
    if _NATIVE_VERSION_STATUS["state"] == "stale":
        logger.warning(
            "victor_native %s is older than victor-ai %s — a stale native wheel is "
            "in use. Upgrade it: pip install -U 'victor-ai[native]' "
            "(or build from source: cd rust && maturin develop --release). "
            "Set VICTOR_NATIVE_SKIP_VERSION_CHECK=1 to silence this warning.",
            _NATIVE_VERSION_STATUS["native_version"],
            _pkg_version,
        )
    elif _NATIVE_VERSION_STATUS["state"] == "newer":
        logger.info(
            "victor_native %s is newer than victor-ai %s (editable/source build?); "
            "keeping native extensions",
            _NATIVE_VERSION_STATUS["native_version"],
            _pkg_version,
        )
