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

"""Native plugin-shape statement extraction (Option B phase 1).

Serves the victor-coding enhanced CCG builder's statement walk: the same
statement set and per-statement enrichment the plugin's Python walk
produces, minus the ~37 ms/file interpreter floor (phase-level profile in
``scripts/benchmark_ccg_builder_phases.py``). The engine is GENERIC —
every language-syntax decision arrives as data: the caller (victor-coding,
the knowledge owner) passes its classification/identifier tables and the
Python-side symbol hints; this module never imports a vertical. The
plugin's Python walk remains the typed reference implementation, and the
golden corpus at
``verticals/victor-coding/tests/codebase/goldens/`` is the parity oracle
(read there as pure data by the differential test).

Not yet wired into the production consult path — phase 2 flips
``PluginBackedCCGBuilder`` onto it behind the existing builder-resolution
knob once parity is pinned across languages.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from victor.processing.native._base import _NATIVE_AVAILABLE, _native

# Same grammar set as the builtin engine slice; the plugin tables make the
# WALK language-agnostic — only the parser is per-language.
# Same grammar set as the builtin engine slice; the plugin tables make the
# WALK language-agnostic — only the parser is per-language.
PLUGIN_NATIVE_LANGUAGES = frozenset(
    {"python", "javascript", "typescript", "tsx", "rust", "go", "java", "ruby"}
)

HintRecord = Tuple[int, int, str, str, Optional[str]]
PluginFile = Tuple[str, str, bytes]  # (language, path, source)


def plugin_statements_available(language: str) -> bool:
    """Whether the native plugin walk can serve this language right now."""
    return (
        _NATIVE_AVAILABLE
        and language in PLUGIN_NATIVE_LANGUAGES
        and hasattr(_native, "extract_plugin_statements_batch")
    )


def extract_plugin_statements(
    files: Sequence[PluginFile],
    tables: Mapping[str, Any],
    hints_by_file: Sequence[Sequence[HintRecord]],
) -> Optional[List[List[Any]]]:
    """Native plugin-shape statements for a batch, or None (caller falls
    back to the Python walk).

    ``tables`` is the victor-coding language-syntax table set (the JSON
    shape exported by the golden generator); ``hints_by_file`` mirrors
    ``files`` order with (start_byte, end_byte, node_type, name,
    symbol_type) tuples from the Python-side hint pass. Records carry
    ``node_id``, ``node_type``, ``statement_type``, ``name``,
    ``start_line``, ``end_line``, ``visibility``, ``scope_id``,
    ``symbol_name``, ``symbol_type``, ``definitions``, ``uses``, ``calls``
    and ``content_preview`` — the exact field set the plugin walk feeds
    GraphNode construction.

    Bounded failure semantics: any FFI failure returns None so the caller
    degrades to the Python walk; no side effects replay after a failure.
    """
    if not _NATIVE_AVAILABLE or not hasattr(_native, "extract_plugin_statements_batch"):
        return None
    if not files:
        return []
    if not all(lang in PLUGIN_NATIVE_LANGUAGES for lang, _, _ in files):
        return None
    kwargs = _tables_kwargs(tables)
    if kwargs is None:
        return None
    try:  # pragma: no cover - extension-only path (differential CI job)
        return _native.extract_plugin_statements_batch(
            list(files), **kwargs, hints_by_file=list(hints_by_file)
        )
    except Exception:  # pragma: no cover - bounded failure semantics
        return None


def _tables_kwargs(tables: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Unpack the table set into the FFI signature; None on a shape miss."""
    required = (
        "classify",
        "identifier_types",
        "skip_descent",
        "scope_types",
        "definition_types",
        "keywords",
        "function_def",
        "class_def",
        "call_type",
        "def_fields",
        "name_fields",
        "param_fields",
        "call_fields",
    )
    if any(key not in tables for key in required):
        return None
    kwargs = {key: tables[key] for key in required}
    # The JSON export carries sorted lists; the FFI takes sets.
    for key in ("identifier_types", "skip_descent", "scope_types", "definition_types", "keywords"):
        value = kwargs[key]
        if isinstance(value, list):
            kwargs[key] = set(value)
    return kwargs
