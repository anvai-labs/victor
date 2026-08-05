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

"""Unit tests for victor.contrib.codebase package."""

import inspect

import pytest
from pathlib import Path

from victor.contrib.codebase import BasicCodebaseAnalyzer
from victor.contrib.codebase import analyzer as analyzer_module


@pytest.fixture
def sample_codebase(tmp_path: Path) -> Path:
    """Create a sample codebase for testing."""
    # Create Python files
    (tmp_path / "module1.py").write_text("""
'''Module one docstring.'''

def function_one(param1, param2):
    '''Function one docstring.'''
    return param1 + param2

class ClassOne:
    '''Class one docstring.'''
    def method_one(self):
        '''Method one docstring.'''
        pass
""")

    (tmp_path / "module2.py").write_text("""
from module1 import function_one

def function_two():
    return function_one(1, 2)
""")

    # Create subdirectory
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "module3.py").write_text("""
def function_three():
    pass
""")

    # Create a text file
    (tmp_path / "README.md").write_text("# Test Project\n\nThis is a test.")

    return tmp_path


class TestBasicCodebaseAnalyzer:
    """Test BasicCodebaseAnalyzer implementation."""

    def test_analyzer_info(self) -> None:
        """Test analyzer metadata retrieval."""
        analyzer = BasicCodebaseAnalyzer()
        info = analyzer.get_analyzer_info()

        assert info["name"] == "BasicCodebaseAnalyzer"
        assert info["version"] == "1.0.0"
        assert "capabilities" in info
        assert "file_discovery" in info["capabilities"]

    @pytest.mark.asyncio
    async def test_analyze_codebase_python_only(self, sample_codebase: Path) -> None:
        """Test analyzing a Python codebase."""
        analyzer = BasicCodebaseAnalyzer()
        result = await analyzer.analyze_codebase(
            root_path=sample_codebase,
            include_patterns=["**/*.py"],
        )

        assert result.root_path == sample_codebase
        assert result.total_files == 3  # module1.py, module2.py, module3.py
        assert len(result.files) == 3
        assert "python" in result.languages
        assert result.languages["python"] == 3
        assert result.total_lines > 0

    @pytest.mark.asyncio
    async def test_analyze_codebase_with_exclude(self, sample_codebase: Path) -> None:
        """Test analyzing codebase with exclusion patterns."""
        analyzer = BasicCodebaseAnalyzer()
        result = await analyzer.analyze_codebase(
            root_path=sample_codebase,
            include_patterns=["**/*.py"],
            exclude_patterns=["**/subdir/*.py"],
        )

        assert result.total_files == 2  # Only module1.py and module2.py
        assert len(result.files) == 2

    @pytest.mark.asyncio
    async def test_analyze_codebase_multiple_languages(self, sample_codebase: Path) -> None:
        """Test analyzing codebase with multiple file types."""
        analyzer = BasicCodebaseAnalyzer()
        result = await analyzer.analyze_codebase(
            root_path=sample_codebase,
            include_patterns=["**/*"],
        )

        # Should find .py and .md files
        assert result.total_files >= 4
        assert "python" in result.languages
        assert result.total_lines > 0

    @pytest.mark.asyncio
    async def test_parse_python_file(self, sample_codebase: Path) -> None:
        """Test parsing a Python file."""
        analyzer = BasicCodebaseAnalyzer()
        result = await analyzer.parse_file(sample_codebase / "module1.py")

        assert result.file_path == sample_codebase / "module1.py"
        assert result.language == "python"
        assert result.lines > 0
        assert len(result.classes) == 1
        assert result.classes[0].name == "ClassOne"
        # Basic analyzer only finds top-level functions in the parsed file
        assert len(result.functions) == 1
        assert result.functions[0].name == "function_one"
        # Note: method_one is a class method, not extracted separately
        assert len(result.imports) == 0  # module1.py has no imports

    @pytest.mark.asyncio
    async def test_parse_file_language_detection(self, sample_codebase: Path) -> None:
        """Test language detection from file extension."""
        analyzer = BasicCodebaseAnalyzer()

        # Python file
        py_result = await analyzer.parse_file(sample_codebase / "module1.py")
        assert py_result.language == "python"

        # Markdown file
        md_result = await analyzer.parse_file(sample_codebase / "README.md")
        assert md_result.language == "markdown"

    @pytest.mark.asyncio
    async def test_get_dependencies(self, sample_codebase: Path) -> None:
        """Test getting file dependencies."""
        analyzer = BasicCodebaseAnalyzer()
        result = await analyzer.get_dependencies(sample_codebase / "module2.py")

        assert result.file_path == sample_codebase / "module2.py"
        # module2.py imports from module1
        assert any(imp.module == "module1" for imp in result.imports)

    @pytest.mark.asyncio
    async def test_parse_nonexistent_file(self) -> None:
        """Test parsing a file that doesn't exist."""
        analyzer = BasicCodebaseAnalyzer()
        result = await analyzer.parse_file(Path("/nonexistent/file.py"))

        assert result.errors
        assert len(result.errors) > 0
        assert "Cannot read file" in result.errors[0]


class TestPythonParsing:
    """Test Python-specific parsing patterns."""

    @pytest.mark.asyncio
    async def test_extract_class_with_bases(self, tmp_path: Path) -> None:
        """Test extracting class with base classes."""
        (tmp_path / "test.py").write_text("""
class MyClass(BaseClass):
    pass

class InheritedClass(Parent1, Parent2):
    pass
""")

        analyzer = BasicCodebaseAnalyzer()
        result = await analyzer.parse_file(tmp_path / "test.py")

        assert len(result.classes) == 2
        assert result.classes[0].name == "MyClass"
        assert result.classes[0].bases == ["BaseClass"]
        assert result.classes[1].bases == ["Parent1", "Parent2"]

    @pytest.mark.asyncio
    async def test_extract_function_with_params(self, tmp_path: Path) -> None:
        """Test extracting function with parameters."""
        (tmp_path / "test.py").write_text("""
def function_with_params(param1, param2, param3=None):
    pass

async def async_function(param1: str, param2: int) -> None:
    pass
""")

        analyzer = BasicCodebaseAnalyzer()
        result = await analyzer.parse_file(tmp_path / "test.py")

        assert len(result.functions) == 2
        func1 = result.functions[0]
        assert func1.name == "function_with_params"
        assert len(func1.parameters) == 3

        func2 = result.functions[1]
        assert func2.name == "async_function"
        assert func2.is_async

    @pytest.mark.asyncio
    async def test_extract_imports(self, tmp_path: Path) -> None:
        """Test extracting import statements."""
        (tmp_path / "test.py").write_text("""
import os
import sys
from pathlib import Path
from collections import defaultdict
""")

        analyzer = BasicCodebaseAnalyzer()
        result = await analyzer.parse_file(tmp_path / "test.py")

        assert len(result.imports) == 4
        assert result.imports[0].module == "os"
        assert result.imports[0].is_from_import is False

        assert result.imports[2].module == "pathlib"
        assert result.imports[2].names == ["Path"]
        assert result.imports[2].is_from_import is True

    @pytest.mark.asyncio
    async def test_extract_decorators(self, tmp_path: Path) -> None:
        """Test extracting decorators."""
        (tmp_path / "test.py").write_text("""
@dataclass
class MyClass:
    pass

@property
def my_property(self):
    return 1
""")

        analyzer = BasicCodebaseAnalyzer()
        result = await analyzer.parse_file(tmp_path / "test.py")

        # Implementation extracts decorators for classes
        assert result.classes[0].decorators == ["dataclass"]
        # And functions
        assert result.functions[0].decorators == ["property"]

    @pytest.mark.asyncio
    async def test_extract_method(self, tmp_path: Path) -> None:
        """Test extracting methods in classes."""
        (tmp_path / "test.py").write_text("""
class MyClass:
    def method_one(self):
        pass

    @staticmethod
    def static_method():
        pass
""")

        analyzer = BasicCodebaseAnalyzer()
        result = await analyzer.parse_file(tmp_path / "test.py")

        # The current implementation only extracts top-level functions
        # Methods inside classes are not extracted separately
        assert len(result.functions) == 0  # No top-level functions
        assert len(result.classes) == 1
        assert result.classes[0].name == "MyClass"
        # Note: method_one and static_method are not extracted
        # as separate functions - they're part of the class

    def test_analyzer_has_no_parallel_regex_parser(self) -> None:
        source = inspect.getsource(analyzer_module)
        assert "re.compile" not in source
        assert "CLASS_PATTERN" not in source
        assert "FUNCTION_PATTERN" not in source
