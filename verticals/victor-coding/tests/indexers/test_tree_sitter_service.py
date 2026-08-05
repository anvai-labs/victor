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

"""Tests for TreeSitterService canonical parsing service."""

from __future__ import annotations

import pytest

pytest.importorskip("victor_coding.codebase.tree_sitter_service")

from victor_coding.codebase.tree_sitter_service import (
    LANGUAGE_MODULES,
    ParsedSource,
    TreeSitterService,
    _reset_for_tests,
    get_tree_sitter_service,
)


@pytest.fixture(autouse=True)
def _isolate_service():
    _reset_for_tests()
    yield
    _reset_for_tests()


class TestNormalizeLanguage:
    """Alias normalization."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("python", "python"),
            ("Python", "python"),
            ("PY", "python"),
            ("py", "python"),
            ("js", "javascript"),
            ("node", "javascript"),
            ("typescriptreact", "tsx"),
            ("csharp", "c_sharp"),
            ("cs", "c_sharp"),
            ("c++", "cpp"),
            ("cxx", "cpp"),
            ("cc", "cpp"),
            ("rs", "rust"),
            ("kt", "kotlin"),
            ("rb", "ruby"),
            ("unknown_lang", "unknown_lang"),
        ],
    )
    def test_normalization(self, raw, expected):
        svc = TreeSitterService()
        assert svc.normalize_language(raw) == expected

    def test_empty_or_none(self):
        svc = TreeSitterService()
        assert svc.normalize_language("") == ""
        assert svc.normalize_language(None) == ""


class TestSupportsLanguage:
    """``supports_language`` is the cheap gate for callers."""

    def test_unknown_language_is_false(self):
        svc = TreeSitterService()
        assert svc.supports_language("definitely_not_a_language") is False

    def test_known_language_returns_bool(self):
        svc = TreeSitterService()
        # True iff the grammar wheel is installed; never raises.
        assert isinstance(svc.supports_language("python"), bool)


class TestParse:
    """End-to-end parse against the real Python grammar."""

    def test_parse_python_returns_parsed_source(self):
        svc = TreeSitterService()
        if not svc.supports_language("python"):
            pytest.skip("tree-sitter-python not installed")
        parsed = svc.parse(b"def foo():\n    pass\n", "python", file_path="a.py")
        assert isinstance(parsed, ParsedSource)
        assert parsed.language == "python"
        assert parsed.file_path == "a.py"
        assert parsed.root_node is not None

    def test_parse_unknown_language_returns_none(self):
        svc = TreeSitterService()
        assert svc.parse(b"whatever", "definitely_not_a_language") is None

    def test_parse_normalizes_alias(self):
        svc = TreeSitterService()
        if not svc.supports_language("python"):
            pytest.skip("tree-sitter-python not installed")
        parsed = svc.parse(b"x = 1\n", "PY")
        assert parsed is not None
        assert parsed.language == "python"


class TestGetParserPerThread:
    """Per-thread parser cache invariants."""

    def test_same_thread_returns_same_parser(self):
        svc = TreeSitterService()
        if not svc.supports_language("python"):
            pytest.skip("tree-sitter-python not installed")
        a = svc.get_parser("python")
        b = svc.get_parser("python")
        assert a is b

    def test_unsupported_returns_none(self):
        svc = TreeSitterService()
        assert svc.get_parser("definitely_not_a_language") is None


class TestQueryCache:
    """Compiled queries are cached by ``(language, kind)``."""

    def test_repeated_get_query_same_kind_returns_same_object(self):
        svc = TreeSitterService()
        if not svc.supports_language("python"):
            pytest.skip("tree-sitter-python not installed")
        src = "(function_definition name: (identifier) @name)"
        first = svc.get_query("python", "symbols", src)
        second = svc.get_query("python", "symbols", src)
        assert first is not None
        assert first is second

    def test_different_kinds_compile_independently(self):
        svc = TreeSitterService()
        if not svc.supports_language("python"):
            pytest.skip("tree-sitter-python not installed")
        a = svc.get_query("python", "symbols", "(function_definition) @x")
        b = svc.get_query("python", "calls", "(call) @x")
        assert a is not None
        assert b is not None
        assert a is not b

    def test_run_query_returns_captures(self):
        svc = TreeSitterService()
        if not svc.supports_language("python"):
            pytest.skip("tree-sitter-python not installed")
        parsed = svc.parse(b"def foo():\n    pass\n", "python")
        assert parsed is not None
        captures = svc.run_query(
            parsed,
            "symbols",
            "(function_definition name: (identifier) @name)",
        )
        assert "name" in captures
        assert len(captures["name"]) == 1


class TestSingleton:
    """``get_tree_sitter_service`` returns a process-wide singleton."""

    def test_singleton_identity(self):
        a = get_tree_sitter_service()
        b = get_tree_sitter_service()
        assert a is b

    def test_reset_for_tests_clears_singleton(self):
        a = get_tree_sitter_service()
        _reset_for_tests()
        b = get_tree_sitter_service()
        assert a is not b


class TestLanguageModules:
    """Sanity checks on the grammar map."""

    def test_module_format(self):
        for lang, info in LANGUAGE_MODULES.items():
            assert isinstance(info, tuple), lang
            assert len(info) == 2, lang
            module_name, func_name = info
            assert module_name.startswith("tree_sitter_"), lang
            assert isinstance(func_name, str), lang
