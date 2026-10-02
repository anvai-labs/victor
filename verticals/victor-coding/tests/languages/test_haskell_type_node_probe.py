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

"""TD/ratchet follow-up: the haskell class-declaration node spelling is
grammar-era dependent (``type_synomym`` typo before the upstream rename,
``type_synonym`` after). The haskell plugin must probe the installed grammar
and emit the spelling that compiles — a hard-coded spelling fails query
compilation on the other era's grammar wheel (observed in the promotion
battery when the grammar wheel updated)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("tree_sitter")

MODULE_PATH = Path(__file__).resolve().parents[2] / (
    "victor_coding/languages/plugins/additional.py"
)


def _load_additional_module():
    """Load additional.py by path so the test exercises the checked-out file
    rather than whatever the editable install maps."""
    spec = importlib.util.spec_from_file_location(
        "_additional_under_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_additional_under_test", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def additional_module():
    try:
        return _load_additional_module()
    except Exception as exc:  # pragma: no cover - import env issue
        pytest.fail(f"could not load additional.py: {exc}")


class _StubQuery:
    """Stub tree_sitter.Query: accepts only the pattern it is told to."""

    accepted = "type_synonym"

    def __init__(self, language, pattern):
        if self.accepted not in pattern:
            raise ValueError(f"invalid node type: {pattern}")


def test_probe_prefers_fixed_spelling_when_grammar_supports_it(
    additional_module, monkeypatch
):
    import tree_sitter

    monkeypatch.setattr(tree_sitter, "Query", _StubQuery)
    assert additional_module._haskell_type_class_node(object()) == "type_synonym"


def test_probe_falls_back_to_legacy_spelling(additional_module, monkeypatch):
    import tree_sitter

    class _LegacyOnlyQuery:
        def __init__(self, language, pattern):
            if "type_synonym" in pattern:
                raise ValueError("invalid node type: type_synonym")

    monkeypatch.setattr(tree_sitter, "Query", _LegacyOnlyQuery)
    assert additional_module._haskell_type_class_node(object()) == "type_synomym"


def test_probe_survives_broken_query_type(additional_module, monkeypatch):
    import tree_sitter

    def _explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(tree_sitter, "Query", _explode)
    assert additional_module._haskell_type_class_node(object()) == "type_synomym"


def test_real_grammar_spelling_compiles_if_wheel_present():
    tree_sitter_haskell = pytest.importorskip("tree_sitter_haskell")
    from tree_sitter import Language, Query

    language = Language(tree_sitter_haskell.language())
    module = _load_additional_module()
    spelling = module._haskell_type_class_node(language)
    # Whatever spelling the probe chose must compile against the real grammar.
    Query(language, f"({spelling} (name) @name)")
