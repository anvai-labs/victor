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

"""Tests for documentation_tool module."""

import pytest
import ast
import tempfile
from pathlib import Path

pytest.importorskip("victor_coding.languages.base")

from victor.tools.documentation_tool import (
    _generate_function_docstring,
    _generate_class_docstring,
    _extract_api_info,
    _build_markdown_docs,
    _has_doc_comment_before,
    _extract_doc_comment_text,
    docs,
    docs_coverage,
)
from victor_coding.languages.base import DocCommentPattern


class TestGenerateFunctionDocstring:
    """Tests for _generate_function_docstring function."""

    def test_simple_function(self):
        """Test docstring for simple function."""
        code = "def hello(): pass"
        tree = ast.parse(code)
        node = tree.body[0]
        docstring = _generate_function_docstring(node, "google")
        assert "Hello" in docstring or "hello" in docstring

    def test_function_with_args(self):
        """Test docstring for function with arguments."""
        code = "def greet(name, age): pass"
        tree = ast.parse(code)
        node = tree.body[0]
        docstring = _generate_function_docstring(node, "google")
        assert "Args:" in docstring
        assert "name" in docstring
        assert "age" in docstring

    def test_function_with_return(self):
        """Test docstring for function with return."""
        code = "def get_value():\n    return 42"
        tree = ast.parse(code)
        node = tree.body[0]
        docstring = _generate_function_docstring(node, "google")
        assert "Returns:" in docstring

    def test_function_with_raise(self):
        """Test docstring for function with raise."""
        code = "def risky():\n    raise ValueError('error')"
        tree = ast.parse(code)
        node = tree.body[0]
        docstring = _generate_function_docstring(node, "google")
        assert "Raises:" in docstring


class TestGenerateClassDocstring:
    """Tests for _generate_class_docstring function."""

    def test_simple_class(self):
        """Test docstring for simple class."""
        code = "class MyClass:\n    pass"
        tree = ast.parse(code)
        node = tree.body[0]
        docstring = _generate_class_docstring(node, "google")
        assert "MyClass" in docstring

    def test_class_with_init(self):
        """Test docstring for class with __init__."""
        code = """
class MyClass:
    def __init__(self):
        self.name = "test"
        self.value = 42
"""
        tree = ast.parse(code)
        node = tree.body[0]
        docstring = _generate_class_docstring(node, "google")
        # Should detect attributes
        assert "MyClass" in docstring


class TestExtractApiInfo:
    """Tests for _extract_api_info function."""

    def test_extract_api_info(self):
        """Test extracting API info from AST."""
        code = """
def hello():
    pass

class MyClass:
    def method(self):
        pass
"""
        tree = ast.parse(code)
        info = _extract_api_info(tree, "test_module")
        assert "module" in info  # Returns "module", not "module_name"
        assert "functions" in info
        assert "classes" in info


class TestBuildMarkdownDocs:
    """Tests for _build_markdown_docs function."""

    def test_build_markdown_docs(self):
        """Test building markdown from API info."""
        api_info = {
            "module": "test",  # Uses "module", not "module_name"
            "functions": [{"name": "hello", "args": [], "docstring": "A test function"}],
            "classes": [],
        }
        md = _build_markdown_docs(api_info)
        assert "test" in md or "hello" in md


class TestGenerateDocs:
    """Tests for generate_docs function."""

    @pytest.mark.asyncio
    async def test_generate_docs_file(self):
        """Test generating docs for a single file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
def hello():
    print("hello")

class MyClass:
    def method(self):
        pass
""")
            temp_path = f.name

        try:
            result = await docs(path=temp_path)
            assert result["success"] is True
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_generate_docs_nonexistent_file(self):
        """Test generating docs for nonexistent file."""
        result = await docs(path="/nonexistent/file.py")
        assert result["success"] is False


class TestAnalyzeDocs:
    """Tests for analyze_docs function."""

    @pytest.mark.asyncio
    async def test_analyze_docs(self):
        """Test analyzing documentation coverage."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('''
def documented():
    """This function is documented."""
    pass

def undocumented():
    pass
''')
            temp_path = f.name

        try:
            result = await docs_coverage(path=temp_path)
            assert result["success"] is True
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_analyze_docs_nonexistent(self):
        """Test analyzing nonexistent file."""
        result = await docs_coverage(path="/nonexistent/file.py")
        assert result["success"] is False


class TestMultiLanguageDocsCoverage:
    """Tests for multi-language documentation coverage."""

    def test_rust_doc_coverage(self):
        """Test detection of Rust /// doc comments."""
        rust_pattern = DocCommentPattern(line_prefixes=["///", "//!"])
        source = """\
/// Documented function.
/// With a second line.
pub fn documented_func() {
}

pub fn undocumented_func() {
}

/// A documented struct.
pub struct MyStruct {
}
"""
        lines = source.split("\n")
        # documented_func is on line 3
        assert _has_doc_comment_before(lines, 3, rust_pattern) is True
        # undocumented_func is on line 6
        assert _has_doc_comment_before(lines, 6, rust_pattern) is False
        # MyStruct is on line 10
        assert _has_doc_comment_before(lines, 10, rust_pattern) is True

    def test_rust_inner_doc_comments(self):
        """Test detection of Rust //! inner doc comments."""
        rust_pattern = DocCommentPattern(line_prefixes=["///", "//!"])
        source = """\
//! Module-level documentation.
//! This describes the module.

/// Function docs.
pub fn my_func() {
}
"""
        lines = source.split("\n")
        # my_func is on line 5
        assert _has_doc_comment_before(lines, 5, rust_pattern) is True

    @pytest.mark.asyncio
    async def test_python_still_works(self):
        """Regression test: Python docstrings still detected via ast."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('''\
def documented():
    """This function is documented."""
    pass

def undocumented():
    pass

class MyClass:
    """A documented class."""
    pass
''')
            temp_path = f.name

        try:
            result = await docs_coverage(path=temp_path)
            assert result["success"] is True
            assert result["total_items"] == 3  # 2 functions + 1 class
            assert result["documented_items"] == 2  # documented() + MyClass
            assert result["missing_count"] == 1
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_go_doc_coverage(self):
        """Test detection of Go // doc comments."""
        go_pattern = DocCommentPattern(line_prefixes=["//"])
        source = """\
// Documented is a helper function.
func Documented() {
}

func Undocumented() {
}

// MyStruct represents a thing.
type MyStruct struct {
}
"""
        lines = source.split("\n")
        # Documented is on line 2
        assert _has_doc_comment_before(lines, 2, go_pattern) is True
        # Undocumented is on line 5
        assert _has_doc_comment_before(lines, 5, go_pattern) is False
        # MyStruct is on line 9
        assert _has_doc_comment_before(lines, 9, go_pattern) is True

    def test_js_jsdoc_coverage(self):
        """Test detection of JS /** */ JSDoc blocks."""
        js_pattern = DocCommentPattern(line_prefixes=["///"], block_start="/**", block_end="*/")
        source = """\
/**
 * A documented function.
 * @param {string} name - The name.
 */
function documented(name) {
}

function undocumented() {
}
"""
        lines = source.split("\n")
        # documented is on line 5
        assert _has_doc_comment_before(lines, 5, js_pattern) is True
        # undocumented is on line 8
        assert _has_doc_comment_before(lines, 8, js_pattern) is False

    @pytest.mark.asyncio
    async def test_unsupported_extension_skipped(self):
        """Test that unsupported file extensions are gracefully skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Some random text content\n")
            temp_path = f.name

        try:
            result = await docs_coverage(path=temp_path)
            assert result["success"] is True
            assert result["total_items"] == 0
            assert result["documented_items"] == 0
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_extract_doc_comment_text_rust(self):
        """Test extracting doc comment text for quality checks."""
        rust_pattern = DocCommentPattern(line_prefixes=["///", "//!"])
        source = """\
/// Short.
pub fn my_func() {
}
"""
        lines = source.split("\n")
        text = _extract_doc_comment_text(lines, 2, rust_pattern)
        assert text == "Short."

    def test_extract_doc_comment_text_jsdoc(self):
        """Test extracting JSDoc block text."""
        js_pattern = DocCommentPattern(line_prefixes=["///"], block_start="/**", block_end="*/")
        source = """\
/**
 * Greet the user with a nice message.
 */
function greet() {
}
"""
        lines = source.split("\n")
        text = _extract_doc_comment_text(lines, 4, js_pattern)
        assert "Greet the user" in text

    def test_decorator_skipping(self):
        """Test that decorators/attributes above doc comments are skipped."""
        rust_pattern = DocCommentPattern(line_prefixes=["///", "//!"])
        source = """\
/// Documented despite attribute below.
#[derive(Debug)]
pub struct MyStruct {
}
"""
        lines = source.split("\n")
        # MyStruct is on line 3
        assert _has_doc_comment_before(lines, 3, rust_pattern) is True
