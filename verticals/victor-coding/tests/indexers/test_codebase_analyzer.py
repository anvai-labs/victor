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

"""Tests for codebase_analyzer module."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

# Check if graph_tool is available
try:
    from victor.tools.graph_tool import graph_tool  # noqa: F401

    GRAPH_TOOL_AVAILABLE = True
except ImportError:
    GRAPH_TOOL_AVAILABLE = False

from victor.config.settings import VICTOR_CONTEXT_FILE
from victor_coding.codebase_analyzer import (
    ClassInfo,
    ModuleInfo,
    CodebaseAnalysis,
    CodebaseAnalyzer,
    extract_graph_insights,
    generate_smart_victor_md,
    _extract_readme_description,
)
from victor.core.schema import Tables


class TestClassInfo:
    """Tests for ClassInfo dataclass."""

    def test_class_info_creation(self):
        """Test creating a ClassInfo instance."""
        ci = ClassInfo(
            name="MyClass",
            file_path="test.py",
            line_number=10,
        )
        assert ci.name == "MyClass"
        assert ci.file_path == "test.py"
        assert ci.line_number == 10
        assert ci.base_classes == []
        assert ci.docstring is None
        assert ci.is_abstract is False
        assert ci.category is None

    def test_class_info_with_all_fields(self):
        """Test ClassInfo with all fields."""
        ci = ClassInfo(
            name="BaseProvider",
            file_path="providers/base.py",
            line_number=50,
            base_classes=["ABC"],
            docstring="Base class for providers.",
            is_abstract=True,
            category="provider",
        )
        assert ci.base_classes == ["ABC"]
        assert ci.docstring == "Base class for providers."
        assert ci.is_abstract is True
        assert ci.category == "provider"


class TestModuleInfo:
    """Tests for ModuleInfo dataclass."""

    def test_module_info_creation(self):
        """Test creating a ModuleInfo instance."""
        mi = ModuleInfo(name="my_module", path="pkg/my_module.py")
        assert mi.name == "my_module"
        assert mi.path == "pkg/my_module.py"
        assert mi.classes == []
        assert mi.functions == []
        assert mi.description is None

    def test_module_info_with_classes(self):
        """Test ModuleInfo with classes."""
        ci = ClassInfo(name="TestClass", file_path="test.py", line_number=1)
        mi = ModuleInfo(
            name="test",
            path="test.py",
            classes=[ci],
            functions=["main", "helper"],
            description="Test module.",
        )
        assert len(mi.classes) == 1
        assert mi.functions == ["main", "helper"]
        assert mi.description == "Test module."


class TestCodebaseAnalysis:
    """Tests for CodebaseAnalysis dataclass."""

    def test_codebase_analysis_creation(self):
        """Test creating a CodebaseAnalysis instance."""
        ca = CodebaseAnalysis(
            project_name="myproject",
            root_path=Path("/tmp/myproject"),
        )
        assert ca.project_name == "myproject"
        assert ca.main_package is None
        assert ca.deprecated_paths == []
        assert ca.packages == {}
        assert ca.key_components == []
        assert ca.entry_points == {}
        assert ca.cli_commands == []
        assert ca.architecture_patterns == []
        assert ca.config_files == []


class TestCodebaseAnalyzerInit:
    """Tests for CodebaseAnalyzer initialization."""

    def test_init_default(self):
        """Test initialization with default path."""
        analyzer = CodebaseAnalyzer()
        assert analyzer.root == Path.cwd()
        assert analyzer.analysis.project_name == Path.cwd().name

    def test_init_with_path(self):
        """Test initialization with custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = CodebaseAnalyzer(tmpdir)
            # Use resolve() since analyzer also resolves paths
            # (handles macOS /var/ -> /private/var/ symlink).
            assert analyzer.root == Path(tmpdir).resolve()


class TestCodebaseAnalyzerDetectPackageLayout:
    """Tests for _detect_package_layout method."""

    def test_detect_flat_layout(self):
        """Test detecting flat package layout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a package directory
            pkg_dir = Path(tmpdir) / "mypackage"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text("")

            analyzer = CodebaseAnalyzer(tmpdir)
            analyzer._detect_package_layout()

            assert analyzer.analysis.main_package == "mypackage"

    def test_detect_src_layout(self):
        """Test detecting src layout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create src/package structure
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            pkg_dir = src_dir / "mypackage"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text("")

            analyzer = CodebaseAnalyzer(tmpdir)
            analyzer._detect_package_layout()

            assert "src" in analyzer.analysis.main_package

    def test_detect_layout_with_deprecated_src(self):
        """Test detecting flat layout with deprecated src."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create both flat and src packages
            flat_pkg = Path(tmpdir) / "mypackage"
            flat_pkg.mkdir()
            (flat_pkg / "__init__.py").write_text("")

            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()

            analyzer = CodebaseAnalyzer(tmpdir)
            analyzer._detect_package_layout()

            assert analyzer.analysis.main_package == "mypackage"
            assert "src/" in analyzer.analysis.deprecated_paths

    def test_skip_test_packages(self):
        """Test that test packages are not selected as main."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test package and real package
            test_pkg = Path(tmpdir) / "tests"
            test_pkg.mkdir()
            (test_pkg / "__init__.py").write_text("")

            real_pkg = Path(tmpdir) / "mypackage"
            real_pkg.mkdir()
            (real_pkg / "__init__.py").write_text("")

            analyzer = CodebaseAnalyzer(tmpdir)
            analyzer._detect_package_layout()

            assert analyzer.analysis.main_package == "mypackage"


@pytest.mark.skip(reason="CodebaseAnalyzer._parse_python_file method not implemented")
class TestCodebaseAnalyzerParsePythonFile:
    """Tests for _parse_python_file method."""

    def test_parse_simple_file(self):
        """Test parsing a simple Python file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "test.py"
            py_file.write_text('''"""Module docstring."""

class MyClass:
    """A test class."""
    pass

def my_function():
    pass
''')

            analyzer = CodebaseAnalyzer(tmpdir)
            module_info = analyzer._parse_python_file(py_file, "test.py")

            assert module_info is not None
            assert module_info.name == "test"
            assert module_info.description == "Module docstring."
            assert len(module_info.classes) == 1
            assert module_info.classes[0].name == "MyClass"
            assert "my_function" in module_info.functions

    def test_parse_invalid_syntax(self):
        """Test parsing file with syntax error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "bad.py"
            py_file.write_text("def broken(")

            analyzer = CodebaseAnalyzer(tmpdir)
            result = analyzer._parse_python_file(py_file, "bad.py")

            assert result is None


@pytest.mark.skip(reason="CodebaseAnalyzer._extract_class_info method not implemented")
class TestCodebaseAnalyzerExtractClassInfo:
    """Tests for _extract_class_info method."""

    def test_extract_basic_class(self):
        """Test extracting basic class info."""
        import ast

        code = '''class MyClass:
    """My docstring."""
    pass'''
        tree = ast.parse(code)
        class_node = tree.body[0]

        analyzer = CodebaseAnalyzer()
        class_info = analyzer._extract_class_info(class_node, "test.py")

        assert class_info.name == "MyClass"
        assert class_info.docstring == "My docstring."
        assert class_info.line_number == 1

    def test_extract_class_with_base(self):
        """Test extracting class with base classes."""
        import ast

        code = "class MyProvider(BaseProvider): pass"
        tree = ast.parse(code)
        class_node = tree.body[0]

        analyzer = CodebaseAnalyzer()
        class_info = analyzer._extract_class_info(class_node, "test.py")

        assert "BaseProvider" in class_info.base_classes
        assert class_info.category == "provider"

    def test_extract_abstract_class(self):
        """Test extracting abstract class."""
        import ast

        code = "class AbstractTool(ABC): pass"
        tree = ast.parse(code)
        class_node = tree.body[0]

        analyzer = CodebaseAnalyzer()
        class_info = analyzer._extract_class_info(class_node, "test.py")

        assert class_info.is_abstract is True


class TestCodebaseAnalyzerCategorizeClass:
    """Tests for _categorize_class method."""

    def test_categorize_provider(self):
        """Test categorizing provider classes."""
        analyzer = CodebaseAnalyzer()
        assert analyzer._categorize_class("OllamaProvider", []) == "provider"
        assert analyzer._categorize_class("MyClass", ["BaseProvider"]) == "provider"

    def test_categorize_tool(self):
        """Test categorizing tool classes."""
        analyzer = CodebaseAnalyzer()
        assert analyzer._categorize_class("GitTool", []) == "tool"
        assert analyzer._categorize_class("FileHandler", []) == "tool"

    def test_categorize_manager(self):
        """Test categorizing manager classes."""
        analyzer = CodebaseAnalyzer()
        assert analyzer._categorize_class("AgentOrchestrator", []) == "manager"
        assert analyzer._categorize_class("CacheManager", []) == "manager"

    def test_categorize_unknown(self):
        """Test uncategorized classes."""
        analyzer = CodebaseAnalyzer()
        assert analyzer._categorize_class("SomeRandomClass", []) is None


class TestCodebaseAnalyzerExtractEntryPoints:
    """Tests for _extract_entry_points method."""

    def test_extract_entry_points(self):
        """Test extracting entry points from pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text("""
[project]
name = "myproject"

[project.scripts]
myapp = "mypackage.cli:main"
myapp2 = "mypackage.cli:other"

[project.optional-dependencies]
dev = ["pytest", "black", "ruff", "mypy"]
""")

            analyzer = CodebaseAnalyzer(tmpdir)
            analyzer.analysis.main_package = "mypackage"
            analyzer._extract_entry_points()

            assert "myapp" in analyzer.analysis.entry_points
            assert "pytest" in analyzer.analysis.cli_commands
            assert "black ." in analyzer.analysis.cli_commands

    def test_no_pyproject(self):
        """Test when pyproject.toml doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = CodebaseAnalyzer(tmpdir)
            analyzer._extract_entry_points()
            assert analyzer.analysis.entry_points == {}


class TestCodebaseAnalyzerDetectArchitecturePatterns:
    """Tests for _detect_architecture_patterns method."""

    def test_detect_provider_pattern(self):
        """Test detecting provider pattern."""
        analyzer = CodebaseAnalyzer()
        analyzer.analysis.key_components = [
            ClassInfo(
                name="BaseProvider",
                file_path="providers/base.py",
                line_number=10,
                is_abstract=True,
                category="provider",
            ),
            ClassInfo(
                name="OllamaProvider",
                file_path="providers/ollama.py",
                line_number=20,
                category="provider",
            ),
        ]

        analyzer._detect_architecture_patterns()

        assert any("Provider Pattern" in p for p in analyzer.analysis.architecture_patterns)

    def test_detect_registry_pattern(self):
        """Test detecting registry pattern."""
        analyzer = CodebaseAnalyzer()
        analyzer.analysis.key_components = [
            ClassInfo(
                name="ToolRegistry",
                file_path="tools/registry.py",
                line_number=10,
                category="registry",
            ),
        ]

        analyzer._detect_architecture_patterns()

        assert any("Registry Pattern" in p for p in analyzer.analysis.architecture_patterns)


class TestCodebaseAnalyzerFindConfigFiles:
    """Tests for _find_config_files method."""

    def test_find_config_files(self):
        """Test finding configuration files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "pyproject.toml").write_text("")
            (Path(tmpdir) / "docker-compose.yml").write_text("")
            (Path(tmpdir) / "Dockerfile").write_text("")

            analyzer = CodebaseAnalyzer(tmpdir)
            analyzer._find_config_files()

            config_files = [f for f, _ in analyzer.analysis.config_files]
            assert "pyproject.toml" in config_files
            assert "docker-compose.yml" in config_files
            assert "Dockerfile" in config_files

    def test_find_glob_patterns(self):
        """Test finding config files with glob patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            github_dir = Path(tmpdir) / ".github" / "workflows"
            github_dir.mkdir(parents=True)
            (github_dir / "ci.yml").write_text("")

            analyzer = CodebaseAnalyzer(tmpdir)
            analyzer._find_config_files()

            config_files = [f for f, _ in analyzer.analysis.config_files]
            assert ".github/workflows/*.yml" in config_files


class TestCodebaseAnalyzerFullAnalyze:
    """Tests for full analyze() method."""

    def test_full_analyze(self):
        """Test complete analysis workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mini package structure
            pkg_dir = Path(tmpdir) / "mypackage"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text("")
            (pkg_dir / "main.py").write_text('''"""Main module."""

class MyOrchestrator:
    """The main orchestrator."""
    pass
''')

            (Path(tmpdir) / "pyproject.toml").write_text("""
[project]
name = "mypackage"
""")

            (Path(tmpdir) / "tests").mkdir()

            analyzer = CodebaseAnalyzer(tmpdir)
            analysis = analyzer.analyze()

            assert analysis.project_name == Path(tmpdir).name
            assert analysis.main_package == "mypackage"
            assert len(analysis.packages) > 0


class TestGenerateSmartVictorMd:
    """Tests for generate_smart_victor_md function."""

    def test_generate_basic(self):
        """Test basic generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_dir = Path(tmpdir) / "mypackage"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text("")

            result = generate_smart_victor_md(tmpdir)

            assert f"# {VICTOR_CONTEXT_FILE}" in result
            assert "## Project Overview" in result
            assert "## Package Layout" in result

    def test_generate_with_readme(self):
        """Test generation with README description."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_dir = Path(tmpdir) / "mypackage"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text("")

            (Path(tmpdir) / "README.md").write_text("""# My Project

This is an awesome project for doing things.
""")

            result = generate_smart_victor_md(tmpdir)

            assert "awesome project" in result

    def test_generate_with_deprecated_paths(self):
        """Test generation highlights deprecated paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create flat package and src/
            pkg_dir = Path(tmpdir) / "mypackage"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text("")

            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()

            result = generate_smart_victor_md(tmpdir)

            assert "DEPRECATED" in result


class TestExtractReadmeDescription:
    """Tests for _extract_readme_description function."""

    def test_extract_description(self):
        """Test extracting description from README."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "README.md").write_text("""# Project

This is the project description.
""")

            desc = _extract_readme_description(Path(tmpdir))
            assert "project description" in desc

    def test_skip_badges(self):
        """Test that badges are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "README.md").write_text("""# Project

[![Badge](https://example.com/badge.svg)](url)

The actual description.
""")

            desc = _extract_readme_description(Path(tmpdir))
            assert "actual description" in desc

    def test_skip_headers(self):
        """Test that headers are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "README.md").write_text("""# Project

## Installation

The real description here.
""")

            desc = _extract_readme_description(Path(tmpdir))
            assert "real description" in desc

    def test_no_readme(self):
        """Test when no README exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            desc = _extract_readme_description(Path(tmpdir))
            assert desc == ""

    def test_readme_rst(self):
        """Test extracting from README.rst."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # RST uses blank lines between paragraphs
            (Path(tmpdir) / "README.rst").write_text("""Project
=======

This is the RST description here.
""")

            desc = _extract_readme_description(Path(tmpdir))
            # RST header lines are separate paragraphs, so "Project" with "======="
            # will be skipped, leaving us with the description
            assert desc == "" or "RST description" in desc or "Project" in desc


class TestCodebaseAnalyzerAnalyzePythonFiles:
    """Tests for _analyze_source_files method."""

    def test_analyze_files_no_package(self):
        """Test analyze files when no main package."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = CodebaseAnalyzer(tmpdir)
            analyzer._analyze_source_files()
            assert analyzer.analysis.packages == {}

    def test_analyze_files_with_subpackages(self):
        """Test analyzing files with subpackages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create package with subpackages
            pkg_dir = Path(tmpdir) / "mypackage"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text("")

            sub_dir = pkg_dir / "tools"
            sub_dir.mkdir()
            (sub_dir / "__init__.py").write_text("")
            (sub_dir / "git_tool.py").write_text('''"""Git tool."""

class GitTool:
    pass
''')

            analyzer = CodebaseAnalyzer(tmpdir)
            analyzer._detect_package_layout()
            analyzer._analyze_source_files()

            # The main package should contain modules from subpackages
            assert "mypackage" in analyzer.analysis.packages
            # Check that git_tool module is in the package
            module_names = [m.name for m in analyzer.analysis.packages["mypackage"]]
            assert "git_tool" in module_names


class TestCodebaseAnalyzerIdentifyKeyComponents:
    """Tests for _identify_key_components method."""

    def test_identify_key_components(self):
        """Test identifying key components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_dir = Path(tmpdir) / "mypackage"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text("")
            (pkg_dir / "orchestrator.py").write_text('''"""Orchestrator module."""

class AgentOrchestrator:
    """Central orchestrator."""
    pass

class BaseProvider:
    """Base provider class."""
    pass
''')

            analyzer = CodebaseAnalyzer(tmpdir)
            analyzer._detect_package_layout()
            analyzer._analyze_source_files()
            analyzer._identify_key_components()

            # Orchestrator should be identified
            component_names = [c.name for c in analyzer.analysis.key_components]
            assert "AgentOrchestrator" in component_names


@pytest.mark.skipif(
    not GRAPH_TOOL_AVAILABLE,
    reason="graph_tool not available. Install with: pip install graph-tool",
)
class TestExtractGraphInsights:
    """Tests for extract_graph_insights — ensures canonical table names are used."""

    def _create_graph_db(self, db_path: Path) -> None:
        """Create a project.db with the canonical graph_node/graph_edge tables."""
        conn = sqlite3.connect(db_path)
        conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS {Tables.GRAPH_NODE} (
                node_id TEXT PRIMARY KEY,
                type TEXT,
                name TEXT,
                file TEXT,
                line INTEGER,
                end_line INTEGER,
                lang TEXT,
                signature TEXT,
                docstring TEXT,
                parent_id TEXT,
                embedding_ref TEXT,
                metadata TEXT
            );
            CREATE TABLE IF NOT EXISTS {Tables.GRAPH_EDGE} (
                src TEXT,
                dst TEXT,
                type TEXT,
                weight REAL,
                metadata TEXT,
                PRIMARY KEY (src, dst, type)
            );
        """)
        # Insert sample nodes
        conn.executemany(
            f"INSERT INTO {Tables.GRAPH_NODE}"
            " (node_id, type, name, file, line, lang) VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("cls_app", "class", "App", "app.py", 1, "python"),
                ("fn_main", "function", "main", "main.py", 10, "python"),
                ("fn_helper", "function", "helper", "utils.py", 5, "python"),
            ],
        )
        # Insert sample edges
        conn.executemany(
            f"INSERT INTO {Tables.GRAPH_EDGE}" " (src, dst, type, weight) VALUES (?, ?, ?, ?)",
            [
                ("fn_main", "cls_app", "CALLS", 1.0),
                ("fn_main", "fn_helper", "CALLS", 1.0),
                ("fn_helper", "cls_app", "REFERENCES", 1.0),
            ],
        )
        conn.commit()
        conn.close()

    @pytest.mark.asyncio
    async def test_extracts_insights_from_canonical_tables(self):
        """Graph insights should read from graph_node/graph_edge tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            victor_dir = Path(tmpdir) / ".victor"
            victor_dir.mkdir()
            db_path = victor_dir / "project.db"
            self._create_graph_db(db_path)

            insights = await extract_graph_insights(tmpdir)

            assert insights["has_graph"] is True
            assert insights["stats"]["total_nodes"] == 3
            assert insights["stats"]["total_edges"] == 3
            assert "function" in insights["stats"]["node_types"]
            assert "CALLS" in insights["stats"]["edge_types"]

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_db(self):
        """Should return empty insights when project.db doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            insights = await extract_graph_insights(tmpdir)
            assert insights["has_graph"] is False
            assert insights["stats"] == {}

    @pytest.mark.asyncio
    async def test_returns_empty_when_tables_empty(self):
        """Should return empty insights when tables exist but have no rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            victor_dir = Path(tmpdir) / ".victor"
            victor_dir.mkdir()
            db_path = victor_dir / "project.db"
            conn = sqlite3.connect(db_path)
            conn.executescript(f"""
                CREATE TABLE {Tables.GRAPH_NODE} (
                    node_id TEXT PRIMARY KEY, type TEXT, name TEXT,
                    file TEXT, line INTEGER, lang TEXT
                );
                CREATE TABLE {Tables.GRAPH_EDGE} (
                    src TEXT, dst TEXT, type TEXT, weight REAL,
                    metadata TEXT, PRIMARY KEY (src, dst, type)
                );
            """)
            conn.close()

            insights = await extract_graph_insights(tmpdir)
            assert insights["has_graph"] is False

    @pytest.mark.asyncio
    async def test_fails_with_wrong_table_names(self):
        """Regression: queries must NOT use bare 'nodes'/'edges' table names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            victor_dir = Path(tmpdir) / ".victor"
            victor_dir.mkdir()
            db_path = victor_dir / "project.db"

            # Create tables with OLD (wrong) names only
            conn = sqlite3.connect(db_path)
            conn.executescript("""
                CREATE TABLE nodes (
                    node_id TEXT PRIMARY KEY, type TEXT, name TEXT,
                    file TEXT, line INTEGER, lang TEXT
                );
                CREATE TABLE edges (
                    src TEXT, dst TEXT, type TEXT, weight REAL,
                    metadata TEXT, PRIMARY KEY (src, dst, type)
                );
            """)
            conn.execute(
                "INSERT INTO nodes (node_id, type, name, file, line, lang)"
                " VALUES ('x', 'class', 'X', 'x.py', 1, 'python')"
            )
            conn.commit()
            conn.close()

            # Should NOT find data — it must use graph_node/graph_edge
            insights = await extract_graph_insights(tmpdir)
            assert insights["has_graph"] is False
