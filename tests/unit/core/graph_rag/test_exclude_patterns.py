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

"""Tests for universal build artifact exclusion patterns."""

import tempfile
from pathlib import Path
from unittest import mock

import pytest

from victor.core.graph_rag.exclude_patterns import (
    UNIVERSAL_EXCLUDE_PATTERNS,
    detect_language_excludes,
    get_exclusion_patterns,
    is_path_excluded,
    parse_gitignore,
)


class TestUniversalExcludes:
    """Test universal exclusion patterns."""

    def test_universal_patterns_include_common_build_dirs(self):
        """Verify universal patterns include all major build directories."""
        patterns = UNIVERSAL_EXCLUDE_PATTERNS

        # Python
        assert any("pyc" in p for p in patterns)
        assert any("__pycache__" in p for p in patterns)
        assert any(".pytest_cache" in p for p in patterns)

        # Node.js
        assert any("node_modules" in p for p in patterns)
        assert any(".next" in p for p in patterns)

        # Rust
        assert any("target" in p for p in patterns)

        # Java
        assert any("target" in p for p in patterns)

        # IDE
        assert any(".idea" in p for p in patterns)
        assert any(".vscode" in p for p in patterns)

        # Version control
        assert any(".git" in p for p in patterns)

        # VS Code extension test fixtures
        assert any(".vscode-test" in p for p in patterns)

    def test_universal_patterns_count(self):
        """Verify we have a comprehensive list of patterns."""
        # Should have 100+ patterns covering all major languages
        assert len(UNIVERSAL_EXCLUDE_PATTERNS) >= 100


class TestGitignoreParsing:
    """Test .gitignore parsing."""

    def test_parse_gitignore_missing_file(self):
        """Test that missing .gitignore returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            patterns = parse_gitignore(root_path)
            assert patterns == []

    def test_parse_gitignore_basic_patterns(self):
        """Test parsing basic .gitignore patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            gitignore = root_path / ".gitignore"

            gitignore.write_text("""# Comment
node_modules/
dist/
*.log
.env
""")

            patterns = parse_gitignore(root_path)
            assert len(patterns) == 4
            assert any("node_modules" in p for p in patterns)
            assert any("dist" in p for p in patterns)
            assert any("*.log" in p for p in patterns)
            assert any(".env" in p for p in patterns)

    def test_parse_gitignore_ignores_comments_and_empty(self):
        """Test that comments and empty lines are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            gitignore = root_path / ".gitignore"

            gitignore.write_text("""
# This is a comment

node_modules/

# Another comment
dist/
""")

            patterns = parse_gitignore(root_path)
            assert len(patterns) == 2

    def test_parse_gitignore_ignores_negation(self):
        """Test that negation patterns (starting with !) are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            gitignore = root_path / ".gitignore"

            gitignore.write_text("""*.log
!important.log
""")

            patterns = parse_gitignore(root_path)
            assert len(patterns) == 1
            assert any("*.log" in p for p in patterns)


class TestLanguageDetection:
    """Test language detection and build artifact exclusion."""

    def test_detect_rust_project(self):
        """Test detection of Rust project via Cargo.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            (root_path / "Cargo.toml").write_text("[package]\nname = 'test'")

            patterns = detect_language_excludes(root_path)
            assert len(patterns) > 0
            assert any("target" in p for p in patterns)

    def test_detect_node_project(self):
        """Test detection of Node.js project via package.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            (root_path / "package.json").write_text('{"name": "test"}')

            patterns = detect_language_excludes(root_path)
            assert len(patterns) > 0
            assert any("node_modules" in p for p in patterns)

    def test_detect_python_project(self):
        """Test detection of Python project via pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            (root_path / "pyproject.toml").write_text("[project]\nname = 'test'")

            patterns = detect_language_excludes(root_path)
            assert len(patterns) > 0
            assert any("__pycache__" in p or "build" in p for p in patterns)

    def test_detect_java_maven_project(self):
        """Test detection of Java Maven project via pom.xml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            (root_path / "pom.xml").write_text("<project></project>")

            patterns = detect_language_excludes(root_path)
            assert len(patterns) > 0
            assert any("target" in p for p in patterns)


class TestGetExclusionPatterns:
    """Test the main get_exclusion_patterns function."""

    def test_get_exclusion_patterns_includes_universal(self):
        """Test that universal patterns are always included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            patterns = get_exclusion_patterns(root_path)

            # After deduplication, should still have most universal patterns
            # Allow some reduction due to deduplication
            assert len(patterns) >= len(UNIVERSAL_EXCLUDE_PATTERNS) * 0.8

    def test_get_exclusion_patterns_with_gitignore(self):
        """Test that .gitignore patterns are included when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            (root_path / ".gitignore").write_text("custom_build/\n")

            patterns = get_exclusion_patterns(root_path, respect_gitignore=True)
            assert any("custom_build" in p for p in patterns)

    def test_get_exclusion_patterns_without_gitignore(self):
        """Test that .gitignore is ignored when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            (root_path / ".gitignore").write_text("custom_build/\n")

            patterns = get_exclusion_patterns(root_path, respect_gitignore=False)
            # Should not contain custom_build pattern
            assert not any("custom_build" in p for p in patterns)

    def test_get_exclusion_patterns_with_language_detection(self):
        """Test that language detection adds patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            (root_path / "Cargo.toml").write_text("[package]")

            patterns_with_detection = get_exclusion_patterns(root_path, detect_languages=True)
            patterns_without_detection = get_exclusion_patterns(root_path, detect_languages=False)

            # With detection should have more specific patterns
            assert len(patterns_with_detection) >= len(patterns_without_detection)

    def test_get_exclusion_patterns_with_custom(self):
        """Test that custom patterns are added."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)

            patterns = get_exclusion_patterns(root_path, custom_patterns=["**/my_custom_dir/**"])
            assert any("my_custom_dir" in p for p in patterns)

    def test_get_exclusion_patterns_deduplicates(self):
        """Test that duplicate patterns are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)

            patterns = get_exclusion_patterns(
                root_path, custom_patterns=["**/target/**", "**/target/**"]
            )

            # Count occurrences of target patterns
            target_count = sum(1 for p in patterns if "target" in p)
            # Should have at most one exact duplicate
            assert target_count == patterns.count(patterns[0]) or target_count <= 2


class TestIsPathExcluded:
    """Test path exclusion checking."""

    def test_excluded_build_directory(self):
        """Test that build directories are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            patterns = ["**/target/**", "**/node_modules/**"]

            # Rust build dir
            assert is_path_excluded(Path(tmpdir) / "target" / "debug" / "main", root_path, patterns)

            # Node modules
            assert is_path_excluded(Path(tmpdir) / "node_modules" / "lodash", root_path, patterns)

    def test_not_excluded_source_file(self):
        """Test that source files are not excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            patterns = ["**/target/**", "**/node_modules/**"]

            # Source file
            assert not is_path_excluded(Path(tmpdir) / "src" / "main.py", root_path, patterns)

    def test_excluded_patterns_with_wildcards(self):
        """Test that wildcard patterns work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            patterns = ["**/*.pyc", "**/*.min.js"]

            # Python bytecode
            assert is_path_excluded(Path(tmpdir) / "module.pyc", root_path, patterns)

            # Minified JS
            assert is_path_excluded(Path(tmpdir) / "app.min.js", root_path, patterns)

            # Regular source (should not be excluded)
            assert not is_path_excluded(Path(tmpdir) / "app.js", root_path, patterns)

    def test_root_level_coverage_temp_file_is_excluded(self):
        """Root-level coverage temp files should be excluded like gitignore expects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            patterns = ["**/.coverage.*"]

            assert is_path_excluded(
                Path(tmpdir) / ".coverage.Vijays-MacBook-Pro.local.69421.XNBbPQdx.c",
                root_path,
                patterns,
            )

    def test_excluded_vscode_test_fixtures(self):
        """Test that VS Code test fixtures are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            patterns = ["**/.vscode-test/**"]

            assert is_path_excluded(
                Path(tmpdir) / ".vscode-test" / "vscode-darwin" / "VSCode.app" / "Contents",
                root_path,
                patterns,
            )

    def test_universal_patterns_exclude_generated_doc_and_coverage_outputs(self):
        """Generated documentation and coverage outputs should not enter graph indexes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            patterns = UNIVERSAL_EXCLUDE_PATTERNS

            assert is_path_excluded(
                Path(tmpdir) / "docs" / "_build" / "html" / "api.html",
                root_path,
                patterns,
            )
            assert is_path_excluded(
                Path(tmpdir) / "htmlcov" / "index.html",
                root_path,
                patterns,
            )

    def test_suffixed_venv_directories_are_excluded(self):
        """Named virtualenvs (.venv-vllm, venv310) must be pruned — only the
        exact `.venv`/`venv` names were matched before, letting 65k
        site-packages files into a real project's index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            patterns = UNIVERSAL_EXCLUDE_PATTERNS
            for venv_dir in (".venv-vllm", ".venv-sglang", "venv310", "venv-py312"):
                assert is_path_excluded(
                    root_path / venv_dir / "lib" / "python3.12" / "site-packages" / "x.py",
                    root_path,
                    patterns,
                ), venv_dir

    def test_third_party_is_excluded_but_external_is_not_universal(self):
        """third_party/ is unambiguous vendoring; external/ can hold
        first-party integration code, so it is only excluded for CMake
        projects (LANGUAGE_CONFIG_MAP)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            patterns = UNIVERSAL_EXCLUDE_PATTERNS
            assert is_path_excluded(
                root_path / "third_party" / "lib" / "code.py", root_path, patterns
            )
            assert not is_path_excluded(
                root_path / "external" / "integrations.py", root_path, patterns
            )

    def test_cmake_projects_exclude_build_flavors_and_external(self):
        """CMake out-of-source builds (build-cuda/, build_common/) and
        FetchContent vendoring (external/) are excluded when CMakeLists.txt
        is present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)
            (root_path / "CMakeLists.txt").write_text("project(x)\n")
            patterns = get_exclusion_patterns(root_path, respect_gitignore=False)
            for path in (
                root_path / "build-cuda" / "gen.cpp",
                root_path / "build_common" / "gen.cpp",
                root_path / "external" / "llama.cpp" / "ggml.c",
            ):
                assert is_path_excluded(path, root_path, patterns), path
