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

"""Native performance extensions (Rust via PyO3).

This module provides optional native (Rust) implementations of performance-critical
operations. When the Rust extensions are not installed, this module provides
fallback stubs that raise ImportError with helpful messages.

NOTE: Tree-sitter extraction is NOT included in native extensions because Python's
tree-sitter bindings already provide efficient access to the C library. Adding Rust
would introduce FFI overhead (Python → Rust → C) without performance benefits.

The native extensions provide:
- Fast code chunking with zero-copy string operations
- Fast regex processing using Rust's regex crate

The native extensions can be installed via:
    pip install victor-ai[native]

Or built from source:
    maturin develop --release

Usage:
    try:
        from victor_coding.native import FastChunker, FastRegexProcessor
        # Use native implementations
    except ImportError:
        # Fall back to Python implementations
        from victor_coding.codebase.chunker import CodeChunker
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Try to import the native Rust module
NATIVE_AVAILABLE = False

try:
    # Try importing as a submodule first (for package builds)
    from . import _native  # type: ignore

    FastChunker = _native.FastChunker
    FastRegexProcessor = _native.FastRegexProcessor
    detect_language = _native.detect_language
    NATIVE_AVAILABLE = True
    logger.info("Native performance extensions loaded (submodule)")
except (ImportError, ModuleNotFoundError):
    try:
        # Try importing as a standalone package (for maturin develop)
        from _native import FastChunker, FastRegexProcessor, detect_language  # type: ignore

        NATIVE_AVAILABLE = True
        logger.info("Native performance extensions loaded (standalone)")
    except (ImportError, ModuleNotFoundError):
        NATIVE_AVAILABLE = False

if not NATIVE_AVAILABLE:
    NATIVE_AVAILABLE = False

    # Create placeholder classes that raise ImportError
    class _NativePlaceholder:
        """Placeholder for native classes when extensions are not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Native performance extensions are not installed. "
                "Install them with: pip install victor-ai[native] "
                "or build from source with: maturin develop --release"
            )

    # Use placeholders
    FastChunker = _NativePlaceholder  # type: ignore
    FastRegexProcessor = _NativePlaceholder  # type: ignore

    # Placeholder for detect_language function
    def detect_language(file_path: str) -> str:  # type: ignore
        """Placeholder for detect_language when native extensions are not installed."""
        raise ImportError(
            "Native performance extensions are not installed. "
            "Install them with: pip install victor-ai[native] "
            "or build from source with: maturin develop --release"
        )

    logger.debug("Native performance extensions not available")


def check_native_available() -> bool:
    """Check if native extensions are available.

    Returns:
        True if native extensions are installed and available
    """
    return NATIVE_AVAILABLE


def get_native_info() -> dict:
    """Get information about native extensions.

    Returns:
        Dictionary with native extension information
    """
    return {
        "available": NATIVE_AVAILABLE,
        "has_chunker": hasattr(FastChunker, "__bases__"),
        "has_regex": hasattr(FastRegexProcessor, "__bases__"),
        "note": "Tree-sitter uses Python bindings directly (no Rust wrapper needed)",
    }


__all__ = [
    "NATIVE_AVAILABLE",
    "FastChunker",
    "FastRegexProcessor",
    "detect_language",
    "check_native_available",
    "get_native_info",
]
