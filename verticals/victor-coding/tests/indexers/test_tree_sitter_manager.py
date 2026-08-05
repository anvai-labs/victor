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

"""Tests for tree-sitter manager - achieving 70%+ coverage."""

import pytest
from unittest.mock import patch, MagicMock

pytest.importorskip("victor_coding.codebase.tree_sitter_manager")

from victor_coding.codebase.tree_sitter_manager import (
    LANGUAGE_MODULES,
    get_language,
    get_parser,
    run_query,
)
from victor_coding.codebase.tree_sitter_service import _reset_for_tests


@pytest.fixture(autouse=True)
def _reset_service():
    """Each test sees a fresh TreeSitterService singleton."""
    _reset_for_tests()
    yield
    _reset_for_tests()


class TestLanguageModules:
    """Tests for LANGUAGE_MODULES constant."""

    def test_language_modules_exists(self):
        """Test LANGUAGE_MODULES dict exists."""
        assert isinstance(LANGUAGE_MODULES, dict)
        assert len(LANGUAGE_MODULES) > 0

    def test_common_languages_present(self):
        """Test common languages are in mapping."""
        expected = ["python", "javascript", "typescript", "java", "go", "rust"]
        for lang in expected:
            assert lang in LANGUAGE_MODULES, f"{lang} not in LANGUAGE_MODULES"

    def test_language_module_format(self):
        """Test each entry has (module_name, function_name) format."""
        for _lang, info in LANGUAGE_MODULES.items():
            assert isinstance(info, tuple)
            assert len(info) == 2
            module_name, func_name = info
            assert isinstance(module_name, str)
            assert isinstance(func_name, str)
            assert module_name.startswith("tree_sitter_")

    def test_web_languages_present(self):
        """Test web languages are present."""
        web_langs = ["html", "css", "json", "yaml"]
        for lang in web_langs:
            assert lang in LANGUAGE_MODULES


class TestGetLanguage:
    """Tests for get_language function."""

    def test_unsupported_language_raises_error(self):
        """Test unsupported language raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported language"):
            get_language("nonexistent_language_xyz")

    def test_cached_language_returned(self):
        """Calling get_language twice returns the same Language instance."""
        try:
            first = get_language("python")
        except ImportError:
            pytest.skip("tree-sitter-python not installed")
        second = get_language("python")
        assert first is second

    def test_import_error_message(self):
        """Test ImportError includes install instructions."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tree_sitter_python":
                raise ImportError("Module not found")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="pip install"):
                get_language("python")

    def test_attribute_error_message(self):
        """Test AttributeError includes helpful message."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tree_sitter_python":
                mock_module = MagicMock()
                # Configure getattr to raise AttributeError for 'language'
                mock_module.configure_mock(
                    **{"language": MagicMock(side_effect=AttributeError("no attribute"))}
                )
                del mock_module.language
                return mock_module
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with pytest.raises(AttributeError, match="does not have function"):
                get_language("python")


class TestGetParser:
    """Tests for get_parser function."""

    def test_repeat_calls_in_same_thread_return_same_parser(self):
        """Per-thread cache: same thread, same language -> same Parser."""
        try:
            first = get_parser("python")
        except ImportError:
            pytest.skip("tree-sitter-python not installed")
        second = get_parser("python")
        assert first is second


class TestRunQuery:
    """Tests for run_query function."""

    @patch("victor_coding.codebase.tree_sitter_manager.get_language")
    @patch("victor_coding.codebase.tree_sitter_manager.Query")
    @patch("victor_coding.codebase.tree_sitter_manager.QueryCursor")
    def test_run_query_basic(self, mock_cursor_class, mock_query_class, mock_get_lang):
        """Test basic query execution."""
        mock_lang = MagicMock()
        mock_get_lang.return_value = mock_lang

        mock_query = MagicMock()
        mock_query_class.return_value = mock_query

        mock_cursor = MagicMock()
        mock_cursor.captures.return_value = {"name": [MagicMock()]}
        mock_cursor_class.return_value = mock_cursor

        mock_tree = MagicMock()
        mock_tree.root_node = MagicMock()

        result = run_query(mock_tree, "(function_definition) @func", "python")

        mock_get_lang.assert_called_once_with("python")
        mock_query_class.assert_called_once_with(mock_lang, "(function_definition) @func")
        mock_cursor_class.assert_called_once_with(mock_query)
        mock_cursor.captures.assert_called_once_with(mock_tree.root_node)
        assert result == {"name": [mock_cursor.captures.return_value["name"][0]]}

    @patch("victor_coding.codebase.tree_sitter_manager.get_language")
    @patch("victor_coding.codebase.tree_sitter_manager.Query")
    @patch("victor_coding.codebase.tree_sitter_manager.QueryCursor")
    def test_run_query_empty_result(self, mock_cursor_class, mock_query_class, mock_get_lang):
        """Test query with no matches."""
        mock_lang = MagicMock()
        mock_get_lang.return_value = mock_lang

        mock_query = MagicMock()
        mock_query_class.return_value = mock_query

        mock_cursor = MagicMock()
        mock_cursor.captures.return_value = {}
        mock_cursor_class.return_value = mock_cursor

        mock_tree = MagicMock()

        result = run_query(mock_tree, "(nonexistent) @x", "python")

        assert result == {}


class TestSpecialLanguageCases:
    """Tests for special language configurations."""

    def test_typescript_has_special_function(self):
        """Test TypeScript has special function name."""
        assert LANGUAGE_MODULES["typescript"][1] == "language_typescript"

    def test_tsx_has_special_function(self):
        """Test TSX has special function name."""
        assert LANGUAGE_MODULES["tsx"][1] == "language_tsx"

    def test_php_may_have_special_function(self):
        """Test PHP may have special function name."""
        assert "php" in LANGUAGE_MODULES
        # PHP uses language_php
        assert LANGUAGE_MODULES["php"][1] == "language_php"

    def test_standard_languages_use_language_function(self):
        """Test standard languages use 'language' function."""
        standard_langs = ["python", "java", "go", "c", "cpp", "ruby"]
        for lang in standard_langs:
            if lang in LANGUAGE_MODULES:
                assert LANGUAGE_MODULES[lang][1] == "language"


class TestRealTreeSitterIntegration:
    """Integration tests with real tree-sitter (if installed)."""

    def test_python_parsing_if_available(self):
        """Test Python parsing works if tree-sitter-python installed."""
        try:
            parser = get_parser("python")
            tree = parser.parse(b"def foo():\n    pass")
            assert tree is not None
            assert tree.root_node is not None
        except ImportError:
            pytest.skip("tree-sitter-python not installed")

    def test_query_python_if_available(self):
        """Test Python query works if tree-sitter-python installed."""
        try:
            parser = get_parser("python")
            tree = parser.parse(b"def foo():\n    pass\ndef bar():\n    return 42")
            captures = run_query(tree, "(function_definition name: (identifier) @name)", "python")
            # Should find function names
            assert "name" in captures
            assert len(captures["name"]) >= 2
        except ImportError:
            pytest.skip("tree-sitter-python not installed")

    def test_javascript_parsing_if_available(self):
        """Test JavaScript parsing works if installed."""
        try:
            parser = get_parser("javascript")
            tree = parser.parse(b"function hello() { return 'world'; }")
            assert tree is not None
        except ImportError:
            pytest.skip("tree-sitter-javascript not installed")


class TestEdgeCases:
    """Edge case tests."""

    def test_language_name_is_case_insensitive(self):
        """Language names are normalized to lowercase by the service."""
        try:
            from_lower = get_language("python")
        except ImportError:
            pytest.skip("tree-sitter-python not installed")
        from_upper = get_language("Python")
        assert from_lower is from_upper

    def test_language_name_with_hyphens(self):
        """Test language names don't use hyphens."""
        # C# is "c_sharp", not "c-sharp"
        assert "c_sharp" in LANGUAGE_MODULES
        assert "c-sharp" not in LANGUAGE_MODULES

    def test_all_modules_follow_naming(self):
        """Test all module names start with tree_sitter_."""
        for lang, (module, _func) in LANGUAGE_MODULES.items():
            assert module.startswith("tree_sitter_"), f"{lang} module doesn't follow naming"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
