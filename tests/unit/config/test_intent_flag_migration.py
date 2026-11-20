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

"""FEP-0049 intent-flag spelling migration (Codex finding 21).

The intent switch is effective at exactly ONE field —
``search.graph.codebase_intent_indexing_enabled`` — the field
``graph_flag`` reads at runtime and the graph CLI resolves ``--intent``
from. Every supported spelling (flat legacy key/env, ``search.``,
``codebase.`` and ``embedding.`` mirrors) must reach it; conflicting
spellings are rejected, never silently resolved.
"""

from __future__ import annotations

import pytest

from victor.config.settings import Settings, graph_flag

ENV_VARS = (
    "VICTOR_CODEBASE_INTENT_INDEXING_ENABLED",
    "VICTOR_SEARCH__CODEBASE_INTENT_INDEXING_ENABLED",
    "VICTOR_CODEBASE__CODEBASE_INTENT_INDEXING_ENABLED",
    "VICTOR_EMBEDDING__CODEBASE_INTENT_INDEXING_ENABLED",
    "VICTOR_SEARCH__GRAPH__CODEBASE_INTENT_INDEXING_ENABLED",
)

EFFECTIVE = "codebase_intent_indexing_enabled"


@pytest.fixture(autouse=True)
def _clean_intent_env(monkeypatch):
    """Keep the developer's real environment out of every case."""
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _effective(settings: Settings) -> bool:
    """The runtime read: exactly what the query tool and graph CLI do."""
    return graph_flag(settings, EFFECTIVE, False)


class TestSupportedSpellingsReachTheEffectiveField:
    """Every advertised spelling must activate the ONE effective field."""

    def test_canonical_field_is_effective(self):
        settings = Settings(search={"graph": {EFFECTIVE: True}})
        assert _effective(settings) is True
        assert settings.search.graph.codebase_intent_indexing_enabled is True

    def test_flat_legacy_key_migrates(self):
        # The pre-FEP-0049 flat yaml key / VICTOR_CODEBASE_INTENT_INDEXING_ENABLED
        # route. Used to land on search.codebase_intent_indexing_enabled — a
        # dead field the runtime never read.
        settings = Settings(**{EFFECTIVE: True})
        assert _effective(settings) is True

    def test_legacy_env_flat_spelling_migrates(self, monkeypatch):
        monkeypatch.setenv("VICTOR_CODEBASE_INTENT_INDEXING_ENABLED", "true")
        assert _effective(Settings()) is True

    def test_search_group_mirror_migrates(self):
        settings = Settings(search={EFFECTIVE: True})
        assert _effective(settings) is True

    def test_codebase_group_mirror_migrates(self):
        # The spelling the graph CLI help used to advertise.
        settings = Settings(codebase={EFFECTIVE: True})
        assert _effective(settings) is True

    def test_embedding_group_mirror_migrates(self):
        settings = Settings(embedding={EFFECTIVE: True})
        assert _effective(settings) is True

    def test_new_style_group_env_migrates(self, monkeypatch):
        monkeypatch.setenv("VICTOR_CODEBASE__CODEBASE_INTENT_INDEXING_ENABLED", "true")
        assert _effective(Settings()) is True

    def test_default_stays_off(self):
        assert _effective(Settings()) is False


class TestConflictRejection:
    """Two spellings that disagree are a configuration error, not a winner."""

    def test_mirror_conflict_rejected(self):
        with pytest.raises(Exception, match="Conflicting"):
            Settings(
                codebase={EFFECTIVE: True},
                embedding={EFFECTIVE: False},
            )

    def test_mirror_conflicting_with_effective_field_rejected(self):
        with pytest.raises(Exception, match="Conflicting"):
            Settings(
                search={"graph": {EFFECTIVE: True}},
                codebase={EFFECTIVE: False},
            )

    def test_env_conflict_rejected(self, monkeypatch):
        monkeypatch.setenv("VICTOR_SEARCH__GRAPH__CODEBASE_INTENT_INDEXING_ENABLED", "true")
        monkeypatch.setenv("VICTOR_CODEBASE__CODEBASE_INTENT_INDEXING_ENABLED", "false")
        with pytest.raises(Exception, match="Conflicting"):
            Settings()

    def test_agreeing_mirrors_accepted(self):
        settings = Settings(
            codebase={EFFECTIVE: True},
            embedding={EFFECTIVE: True},
            search={EFFECTIVE: True},
        )
        assert _effective(settings) is True

    def test_non_boolean_spelling_rejected(self):
        with pytest.raises(Exception, match="boolean"):
            Settings(codebase={EFFECTIVE: "yes-please"})
