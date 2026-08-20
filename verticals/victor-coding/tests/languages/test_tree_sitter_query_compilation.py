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

"""Query compilation harness.

Discovers every registered language plugin and verifies that every non-empty
tree-sitter query string it ships compiles against its grammar. Mirrors the
spirit of VS Code Copilot's centralized ``allKnownQueries`` registration.

A failure here usually means a grammar wheel updated and the plugin's queries
need a corresponding update, or a plugin shipped an invalid query.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import pytest

pytest.importorskip("victor_coding.languages.registry")

from victor_coding.codebase.tree_sitter_service import (
    LANGUAGE_MODULES,
    _reset_for_tests,
    get_tree_sitter_service,
)
from victor_coding.languages.base import TreeSitterQueries
from victor_coding.languages.registry import get_language_registry


@pytest.fixture(autouse=True)
def _isolate_service():
    _reset_for_tests()
    yield
    _reset_for_tests()


def _iter_query_kinds(q: TreeSitterQueries) -> Iterable[Tuple[str, str]]:
    """Yield (kind, source) pairs for every non-empty query a plugin owns."""
    if q.symbols:
        for pattern in q.symbols:
            if pattern.query:
                yield f"symbols:{pattern.symbol_type}", pattern.query
    for attr in ("calls", "references", "inheritance", "implements", "composition"):
        source = getattr(q, attr, None)
        if source:
            yield attr, source


def _discover_plugins() -> List[Tuple[str, object]]:
    registry = get_language_registry()
    if not registry._plugins:
        registry.discover_plugins()
    out: List[Tuple[str, object]] = []
    # LanguageRegistry exposes the plugin factories by name; instantiate
    # each so we can inspect queries.
    for name in sorted(registry._plugins.keys()):
        try:
            plugin = registry.get(name)
        except Exception:
            continue
        out.append((name, plugin))
    return out


# Baseline: plugins whose tree_sitter_queries do NOT all compile against the
# currently-installed grammar wheels. Each entry should be tracked as a query
# fix-up ticket in the language plugin. The set is intentionally allow-listed
# (rather than auto-skipped) so that once a plugin is fixed the harness flips
# to XPASS and forces the entry to be removed.
#
# Empty by design: as of the TSA plugin-query-fix sweep every registered
# plugin's queries compile against its current grammar wheel.
_KNOWN_BROKEN_PLUGINS: set[str] = set()


@pytest.mark.parametrize(
    "language,plugin",
    _discover_plugins(),
    ids=lambda x: x if isinstance(x, str) else "plugin",
)
def test_plugin_queries_compile(language: str, plugin, request) -> None:
    """Every non-empty query a plugin defines must compile against its grammar."""
    service = get_tree_sitter_service()
    canonical = service.normalize_language(language)
    if canonical not in LANGUAGE_MODULES:
        pytest.skip(f"language {language} ({canonical}) not in LANGUAGE_MODULES")
    if not service.supports_language(language):
        pytest.skip(f"grammar wheel for {language} ({canonical}) not installed")

    queries = plugin.tree_sitter_queries
    pairs = list(_iter_query_kinds(queries))
    if not pairs:
        pytest.skip(f"plugin {language} defines no queries")

    failures: List[Tuple[str, str]] = []
    for kind, source in pairs:
        compiled = service.get_query(language, kind, source)
        if compiled is None:
            failures.append((kind, source.strip()[:80]))

    if language in _KNOWN_BROKEN_PLUGINS:
        # Allow-list of plugins with known query/grammar drift. If the
        # plugin is actually fixed (failures == []), XPASS flips this to a
        # test failure so the entry gets removed.
        if not failures:
            pytest.fail(
                f"{language} now compiles all queries cleanly; "
                f"remove it from _KNOWN_BROKEN_PLUGINS in this file."
            )
        pytest.xfail(f"{language}: known query/grammar drift ({len(failures)} failing)")

    assert not failures, f"{language}: {len(failures)} query/queries failed to compile: {failures}"
