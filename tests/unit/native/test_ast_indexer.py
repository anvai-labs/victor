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

"""TDD Tests for AST indexer acceleration.

These tests validate BOTH Python and Rust implementations of AST indexer
hot paths. Tests are written BEFORE implementation (TDD approach).

Target hot paths from graph analysis (indexer.py has 218 degree centrality):
1. is_stdlib_module() - Called per import during indexing
2. extract_identifiers() - Regex fallback for reference extraction
3. batch_is_stdlib_modules() - Batch version for efficiency
"""

import pytest

from victor.native.protocols import AstIndexerProtocol


class TestStdlibModuleDetection:
    """Tests for stdlib module detection.

    This is a hot path called for every import during indexing.
    Optimized with perfect hash in Rust for O(1) lookup.
    """

    def test_core_stdlib_modules(self, ast_indexer):
        """Test detection of core Python stdlib modules."""
        core_modules = [
            "os",
            "sys",
            "json",
            "re",
            "typing",
            "collections",
            "functools",
            "itertools",
            "pathlib",
            "datetime",
            "asyncio",
            "logging",
            "unittest",
            "dataclasses",
        ]
        for module in core_modules:
            assert ast_indexer.is_stdlib_module(module) is True, f"{module} should be stdlib"

    def test_stdlib_submodules(self, ast_indexer):
        """Test detection of stdlib submodules (e.g., os.path)."""
        submodules = [
            "os.path",
            "collections.abc",
            "typing.Optional",
            "unittest.mock",
            "urllib.parse",
            "xml.etree",
        ]
        for module in submodules:
            assert ast_indexer.is_stdlib_module(module) is True, f"{module} should be stdlib"

    def test_common_third_party_modules(self, ast_indexer):
        """Test detection of common third-party modules (excluded from graph)."""
        # These are included in the stdlib set to avoid inflating PageRank
        third_party = [
            "numpy",
            "pandas",
            "requests",
            "pydantic",
            "pytest",
        ]
        for module in third_party:
            assert (
                ast_indexer.is_stdlib_module(module) is True
            ), f"{module} should be in allowed set"

    def test_project_modules_not_stdlib(self, ast_indexer):
        """Test that project-specific modules are NOT detected as stdlib."""
        project_modules = [
            "victor",
            "victor.agent",
            "myproject.utils",
            "custom_module",
            "my_app.models",
        ]
        for module in project_modules:
            assert ast_indexer.is_stdlib_module(module) is False, f"{module} should NOT be stdlib"

    def test_empty_module_name(self, ast_indexer):
        """Test handling of empty module name."""
        assert ast_indexer.is_stdlib_module("") is False

    def test_batch_is_stdlib_modules(self, ast_indexer):
        """Test batch stdlib detection for efficiency."""
        modules = ["os", "victor", "json", "myapp", "typing"]
        results = ast_indexer.batch_is_stdlib_modules(modules)

        assert len(results) == 5
        assert results[0] is True  # os
        assert results[1] is False  # victor
        assert results[2] is True  # json
        assert results[3] is False  # myapp
        assert results[4] is True  # typing

    def test_batch_empty_list(self, ast_indexer):
        """Test batch with empty list."""
        assert ast_indexer.batch_is_stdlib_modules([]) == []


class TestIdentifierExtraction:
    """Tests for identifier extraction.

    This is the regex fallback when tree-sitter misses references.
    Pattern: [A-Za-z_][A-Za-z0-9_]*
    Optimized with SIMD in Rust.
    """

    def test_extract_simple_identifiers(self, ast_indexer):
        """Test extraction of simple identifiers."""
        source = "x = foo + bar * baz"
        identifiers = ast_indexer.extract_identifiers(source)

        assert "x" in identifiers
        assert "foo" in identifiers
        assert "bar" in identifiers
        assert "baz" in identifiers

    def test_extract_function_names(self, ast_indexer):
        """Test extraction of function names."""
        source = """
def hello_world():
    return greet_user(get_name())
"""
        identifiers = ast_indexer.extract_identifiers(source)

        assert "def" in identifiers  # keyword is an identifier pattern
        assert "hello_world" in identifiers
        assert "return" in identifiers
        assert "greet_user" in identifiers
        assert "get_name" in identifiers

    def test_extract_class_names(self, ast_indexer):
        """Test extraction of class and method names."""
        source = """
class MyClass:
    def __init__(self, value):
        self.value = value
"""
        identifiers = ast_indexer.extract_identifiers(source)

        assert "class" in identifiers
        assert "MyClass" in identifiers
        assert "def" in identifiers
        assert "__init__" in identifiers
        assert "self" in identifiers
        assert "value" in identifiers

    def test_extract_with_numbers(self, ast_indexer):
        """Test identifiers containing numbers."""
        source = "var1 = foo2bar + baz3"
        identifiers = ast_indexer.extract_identifiers(source)

        assert "var1" in identifiers
        assert "foo2bar" in identifiers
        assert "baz3" in identifiers
        # Numbers alone are not identifiers
        assert "1" not in identifiers
        assert "2" not in identifiers
        assert "3" not in identifiers

    def test_extract_underscore_identifiers(self, ast_indexer):
        """Test identifiers with underscores."""
        source = "_private = __dunder__ + _leading"
        identifiers = ast_indexer.extract_identifiers(source)

        assert "_private" in identifiers
        assert "__dunder__" in identifiers
        assert "_leading" in identifiers

    def test_extract_returns_unique(self, ast_indexer):
        """Test that duplicates are removed."""
        source = "x = x + x * x"
        identifiers = ast_indexer.extract_identifiers(source)

        # Should be unique
        assert identifiers.count("x") == 1

    def test_extract_empty_source(self, ast_indexer):
        """Test extraction from empty source."""
        assert ast_indexer.extract_identifiers("") == []

    def test_extract_no_identifiers(self, ast_indexer):
        """Test source with no valid identifiers."""
        source = "123 + 456 * 789"
        assert ast_indexer.extract_identifiers(source) == []

    def test_extract_with_strings(self, ast_indexer):
        """Test that identifiers inside strings are extracted too."""
        # Note: This is a simple regex-based extraction, not AST-aware
        source = '"hello" + world'
        identifiers = ast_indexer.extract_identifiers(source)

        # Both are extracted (simple regex doesn't parse strings)
        assert "hello" in identifiers
        assert "world" in identifiers


class TestIdentifierExtractionWithPositions:
    """Tests for identifier extraction with position information."""

    def test_extract_with_positions_basic(self, ast_indexer):
        """Test extraction with start/end positions."""
        source = "x = y"
        results = ast_indexer.extract_identifiers_with_positions(source)

        # Should have x and y with positions
        identifiers = [r[0] for r in results]
        assert "x" in identifiers
        assert "y" in identifiers

        # Check position for x (should be at offset 0)
        x_entry = next(r for r in results if r[0] == "x")
        assert x_entry[1] == 0  # start
        assert x_entry[2] == 1  # end

    def test_extract_with_positions_multiline(self, ast_indexer):
        """Test positions across multiple lines."""
        source = "foo\nbar"
        results = ast_indexer.extract_identifiers_with_positions(source)

        foo_entry = next(r for r in results if r[0] == "foo")
        bar_entry = next(r for r in results if r[0] == "bar")

        assert foo_entry[1] == 0  # foo starts at 0
        assert foo_entry[2] == 3  # foo ends at 3
        assert bar_entry[1] == 4  # bar starts at 4 (after \n)
        assert bar_entry[2] == 7  # bar ends at 7


class TestFilterStdlibImports:
    """Tests for partitioning imports into stdlib and non-stdlib."""

    def test_filter_mixed_imports(self, ast_indexer):
        """Test filtering a mix of stdlib and non-stdlib imports."""
        imports = ["os", "victor", "json", "myapp.utils", "typing", "custom"]
        stdlib, non_stdlib = ast_indexer.filter_stdlib_imports(imports)

        assert set(stdlib) == {"os", "json", "typing"}
        assert set(non_stdlib) == {"victor", "myapp.utils", "custom"}

    def test_filter_all_stdlib(self, ast_indexer):
        """Test when all imports are stdlib."""
        imports = ["os", "sys", "json"]
        stdlib, non_stdlib = ast_indexer.filter_stdlib_imports(imports)

        assert stdlib == ["os", "sys", "json"]
        assert non_stdlib == []

    def test_filter_no_stdlib(self, ast_indexer):
        """Test when no imports are stdlib."""
        imports = ["victor", "myapp", "custom"]
        stdlib, non_stdlib = ast_indexer.filter_stdlib_imports(imports)

        assert stdlib == []
        assert set(non_stdlib) == {"victor", "myapp", "custom"}

    def test_filter_empty(self, ast_indexer):
        """Test with empty list."""
        stdlib, non_stdlib = ast_indexer.filter_stdlib_imports([])
        assert stdlib == []
        assert non_stdlib == []


class TestObservabilityIntegration:
    """Tests for observability integration in AST indexer."""

    def test_indexer_records_metrics(self, ast_indexer, native_metrics):
        """Test that AST indexer records metrics."""
        # Perform some operations
        ast_indexer.is_stdlib_module("os")
        ast_indexer.extract_identifiers("x = y")

        stats = ast_indexer.get_metrics()
        assert stats["calls_total"] >= 2

    def test_batch_operation_metrics(self, ast_indexer, native_metrics):
        """Test metrics for batch operations."""
        modules = ["os", "sys", "json"] * 100  # 300 modules
        ast_indexer.batch_is_stdlib_modules(modules)

        stats = ast_indexer.get_metrics()
        assert stats["calls_total"] >= 1


class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_implements_protocol(self, ast_indexer):
        """Test that implementation satisfies the protocol."""
        assert isinstance(ast_indexer, AstIndexerProtocol)

    def test_is_available(self, ast_indexer):
        """Test is_available method."""
        # Python fallback is always available
        assert ast_indexer.is_available() is True

    def test_get_version(self, ast_indexer):
        """Test get_version method."""
        version = ast_indexer.get_version()
        assert version is None or isinstance(version, str)
