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

"""Tests for TreeSitterAnalysisProvider."""

from __future__ import annotations

import pytest

pytest.importorskip("victor_coding.codebase.tree_sitter_analysis")
pytest.importorskip("tree_sitter_python")

from victor_coding.codebase.tree_sitter_analysis import TreeSitterAnalysisProvider
from victor_coding.codebase.tree_sitter_service import _reset_for_tests


@pytest.fixture(autouse=True)
def _isolate_service():
    _reset_for_tests()
    yield
    _reset_for_tests()


@pytest.fixture
def provider() -> TreeSitterAnalysisProvider:
    return TreeSitterAnalysisProvider()


class TestSupportsLanguage:
    def test_python_supported(self, provider):
        assert provider.supports_language("python") is True

    def test_unknown_language_not_supported(self, provider):
        assert provider.supports_language("definitely_not_a_language") is False

    def test_alias_normalization(self, provider):
        # "py" -> "python" should be supported when python is.
        assert provider.supports_language("py") is True


class TestExtractSymbols:
    def test_python_returns_standard_dict_shape(self, provider):
        source = b"""
class Foo:
    def bar(self):
        return 1

def top_level():
    return 2
""".lstrip()
        symbols = provider.extract_symbols(source, "python", file_path="a.py")
        names = {s["name"] for s in symbols}
        assert "Foo" in names
        # Top-level function captured; the method itself is captured by the
        # function pattern as well (Python query doesn't distinguish).
        assert "top_level" in names

        for sym in symbols:
            assert set(sym.keys()) >= {
                "name",
                "symbol_type",
                "file_path",
                "line_start",
                "line_end",
            }
            assert sym["file_path"] == "a.py"
            assert sym["line_start"] >= 1
            assert sym["line_end"] >= sym["line_start"]

    def test_unknown_language_returns_empty(self, provider):
        symbols = provider.extract_symbols(b"whatever", "fortran77", file_path="x.f")
        assert symbols == []


class TestExtractEdges:
    def test_python_calls_distinguish_method_vs_function(self, provider):
        source = b"""
def caller():
    foo()
    obj.bar()
""".lstrip()
        edges = provider.extract_edges(source, "python", file_path="a.py")
        targets_by_method = {e["target"]: e["is_method_call"] for e in edges}
        assert targets_by_method.get("foo") is False
        assert targets_by_method.get("bar") is True

    def test_python_inheritance_edge(self, provider):
        source = b"""
class Parent:
    pass

class Child(Parent):
    pass
""".lstrip()
        edges = provider.extract_edges(source, "python", file_path="a.py")
        inherits = [e for e in edges if e["edge_type"] == "INHERITS"]
        assert any(e["source"] == "Child" and e["target"] == "Parent" for e in inherits)

    def test_edge_dict_has_required_keys(self, provider):
        source = b"def caller():\n    foo()\n"
        edges = provider.extract_edges(source, "python", file_path="a.py")
        for edge in edges:
            assert set(edge.keys()) >= {
                "source",
                "target",
                "edge_type",
                "file_path",
                "line_number",
            }


class TestParse:
    def test_parse_python_returns_truthy(self, provider):
        parsed = provider.parse(b"x = 1\n", "python", file_path="a.py")
        assert parsed is not None
        assert parsed.root_node is not None

    def test_parse_unknown_returns_none(self, provider):
        assert provider.parse(b"whatever", "fortran77") is None


class TestExtractImports:
    def test_python_imports(self, provider):
        source = b"""
import os
from typing import List, Dict
""".lstrip()
        imports = provider.extract_imports(source, "python", file_path="a.py")
        joined = " ".join(imports)
        assert "import os" in joined
        assert "from typing" in joined

    def test_unsupported_language_returns_empty(self, provider):
        # Bash isn't in _IMPORT_NODE_TYPES even when its grammar is installed.
        # The provider should return [] rather than raise.
        assert provider.extract_imports(b"echo hi", "bash") == []

    def test_scala_imports_extracted(self, provider):
        source = b"""
import com.acme.core.Engine
import com.acme.core.{Config, Log => L}

class Routes
""".lstrip()
        imports = provider.extract_imports(source, "scala", file_path="Routes.scala")
        joined = " ".join(imports)
        assert "com.acme.core.Engine" in joined
        assert "Config" in joined
        assert "class Routes" not in joined

    def test_typescript_imports_include_reexports_not_plain_exports(self, provider):
        source = b"""
import { graph } from './utils/graph';
export * from './components';
export { Button } from './Button';
export class Widget { render() { return 1; } }
export const N = 42;
""".lstrip()
        imports = provider.extract_imports(source, "typescript", file_path="a.ts")
        joined = " ".join(imports)
        # Real imports and re-exports are captured...
        assert "./utils/graph" in joined
        assert "./components" in joined
        assert "./Button" in joined
        # ...plain export declarations are not (they'd dump whole bodies
        # into the raw-import buffer).
        assert "Widget" not in joined
        assert "N = 42" not in joined


class TestBuildChunkContext:
    def test_returns_object_with_root_node_and_content(self, provider):
        source = "def foo():\n    pass\n"
        ctx = provider.build_chunk_context(source, "python", file_path="a.py")
        assert ctx is not None
        assert ctx.root_node is not None
        assert ctx.content == source
        assert ctx.language == "python"

    def test_unknown_language_returns_none(self, provider):
        assert provider.build_chunk_context("x", "fortran77") is None


class TestParseReuse:
    def test_extract_from_parsed_runs_both(self, provider):
        parsed = provider.parse(
            b"def caller():\n    foo()\n",
            "python",
            file_path="a.py",
        )
        assert parsed is not None
        symbols, edges = provider.extract_from_parsed(parsed, file_path="a.py")
        assert any(s["name"] == "caller" for s in symbols)
        assert any(e["target"] == "foo" for e in edges)
