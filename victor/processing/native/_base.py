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
# These operations should use Python fallback even when native is available.
# See victor/processing/native/accelerator.py for benchmark data.
_PYTHON_PREFERRED_OPERATIONS = frozenset(
    {
        "batch_cosine_similarity",  # NumPy+BLAS is ~6x faster for batch operations
        "similarity_matrix",  # NumPy matmul is ~4x faster
    }
)


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
    return operation_name not in _PYTHON_PREFERRED_OPERATIONS


def get_native_version() -> Optional[str]:
    """Get the version of the native extension, if available."""
    if _native is not None:
        version = getattr(_native, "__version__", None)
        if isinstance(version, str):
            return version
        if version is not None:
            return str(version)
    return None
