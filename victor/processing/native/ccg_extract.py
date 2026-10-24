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

"""Native CCG statement extraction — the engine's vertical slice.

The full-tree statement walk in ``CodeContextGraphBuilder`` touches every
tree-sitter node through the GIL-bound py-tree-sitter binding — the
measured hot core of the index parse phase (native-acceleration-strategy
backlog item 7). The native path parses and walks under
``Python::detach`` with rayon, record-for-record byte-identical to the
Python walk (differential parity is pinned by tests; the Python
implementation STAYS as oracle and fallback).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from victor.processing.native._base import _NATIVE_AVAILABLE, _native

# Languages with a native grammar in the Rust crate. Others (go, ...)
# fall back to the Python walk until their crate lands.
NATIVE_STATEMENT_LANGUAGES = frozenset({"python", "javascript", "typescript", "tsx"})


def native_statements_available(language: str) -> bool:
    """Whether native extraction can serve this language right now."""
    return (
        _NATIVE_AVAILABLE
        and language in NATIVE_STATEMENT_LANGUAGES
        and hasattr(_native, "extract_statements_batch")
    )


def extract_statements(language: str, file_path: str, source: bytes) -> Optional[List[Any]]:
    """Native statement records for one file, or None (caller falls back).

    Returns a list of record objects with ``node_id``, ``node_type``,
    ``statement_type``, ``start_line``, ``end_line``, ``content_preview``,
    ``visibility`` and ``scope_id`` — the exact field set the Python walk
    produces.
    """
    if not native_statements_available(language):
        return None
    try:
        results = _native.extract_statements_batch([(language, file_path, source)])
    except Exception:
        # Native failure must degrade to the Python walk, never break the
        # index run (bounded failure semantics — no side effects replay).
        return None
    return results[0] if results else []


def extract_statements_batch(
    files: List[Tuple[str, str, bytes]],
) -> Optional[List[List[Any]]]:
    """Batch variant — one FFI crossing for many files (rayon inside)."""
    if not _NATIVE_AVAILABLE or not hasattr(_native, "extract_statements_batch"):
        return None
    if not files or not all(lang in NATIVE_STATEMENT_LANGUAGES for lang, _, _ in files):
        return None
    try:
        return _native.extract_statements_batch(files)
    except Exception:
        return None
