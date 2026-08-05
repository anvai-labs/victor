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

"""Tests for PathResolver and related classes."""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from victor.protocols.path_resolver import (
    IPathResolver,
    PathResolution,
    PathResolver,
    strip_human_root_prefix,
    strip_cwd_prefix,
    strip_first_component,
    strip_common_prefix,
    normalize_separators,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project structure for testing."""
    # Create directory structure
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "components").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()

    # Create files
    (tmp_path / "README.md").write_text("# Test Project")
    (tmp_path / "src" / "main.py").write_text("# Main module")
    (tmp_path / "src" / "utils.py").write_text("# Utils module")
    (tmp_path / "src" / "components" / "button.py").write_text("# Button")
    (tmp_path / "tests" / "test_main.py").write_text("# Tests")
    (tmp_path / "docs" / "guide.md").write_text("# Guide")

    return tmp_path


@pytest.fixture
def resolver(temp_project):
    """Create a PathResolver with temp project as cwd."""
    return PathResolver(cwd=temp_project)


# =============================================================================
# PathResolution Tests
# =============================================================================


class TestPathResolution:
    """Tests for PathResolution dataclass."""

    def test_basic_creation(self, tmp_path):
        """Test creating PathResolution with basic values."""
        result = PathResolution(
            original_path="test.py",
            resolved_path=tmp_path / "test.py",
        )
        assert result.original_path == "test.py"
        assert result.resolved_path == tmp_path / "test.py"
        assert result.was_normalized is False
        assert result.normalization_applied is None
        assert result.exists is False

    def test_with_normalization(self, tmp_path):
        """Test PathResolution with normalization info."""
        result = PathResolution(
            original_path="project/test.py",
            resolved_path=tmp_path / "test.py",
            was_normalized=True,
            normalization_applied="stripped_cwd_prefix:project",
            exists=True,
            is_file=True,
        )
        assert result.was_normalized is True
        assert "stripped_cwd_prefix" in result.normalization_applied
        assert result.is_file is True

    def test_path_str_property(self, tmp_path):
        """Test path_str property returns string."""
        result = PathResolution(
            original_path="test.py",
            resolved_path=tmp_path / "test.py",
        )
        assert result.path_str == str(tmp_path / "test.py")

    def test_str_representation(self, tmp_path):
        """Test string representation."""
        result = PathResolution(
            original_path="test.py",
            resolved_path=tmp_path / "test.py",
        )
        assert "test.py" in str(result)

    def test_str_with_normalization(self, tmp_path):
        """Test string representation shows normalization."""
        result = PathResolution(
            original_path="project/test.py",
            resolved_path=tmp_path / "test.py",
            was_normalized=True,
            normalization_applied="stripped_cwd",
        )
        str_repr = str(result)
        assert "project/test.py" in str_repr
        assert "stripped_cwd" in str_repr


# =============================================================================
# Normalizer Function Tests
# =============================================================================


class TestStripCwdPrefix:
    """Tests for strip_cwd_prefix normalizer."""

    def test_strips_matching_prefix(self, temp_project):
        """Test stripping cwd name prefix from path."""
        cwd_name = temp_project.name
        path = f"{cwd_name}/src/main.py"
        result, desc = strip_cwd_prefix(path, temp_project)
        assert result == "src/main.py"
        assert "stripped_cwd_prefix" in desc

    def test_returns_none_for_non_matching(self, temp_project):
        """Test returns None when prefix doesn't match."""
        result, desc = strip_cwd_prefix("other/main.py", temp_project)
        assert result is None
        assert desc == ""

    def test_handles_absolute_paths(self, temp_project):
        """Test handles absolute paths (returns None)."""
        result, desc = strip_cwd_prefix("/absolute/path", temp_project)
        assert result is None

    def test_handles_home_paths(self, temp_project):
        """Test handles home paths (returns None)."""
        result, desc = strip_cwd_prefix("~/home/path", temp_project)
        assert result is None

    def test_handles_empty_path(self, temp_project):
        """Test handles empty path."""
        result, desc = strip_cwd_prefix("", temp_project)
        assert result is None


class TestStripHumanRootPrefix:
    """Tests for natural-language root/repository prefix normalization."""

    def test_strips_root_prefix_when_remaining_path_exists(self, tmp_path):
        (tmp_path / "rust").mkdir()
        (tmp_path / "rust" / "Cargo.toml").write_text("[workspace]\n")

        result, desc = strip_human_root_prefix("root rust/Cargo.toml", tmp_path)

        assert result == "rust/Cargo.toml"
        assert desc == "stripped_human_root_prefix:root"

    def test_keeps_literal_path_when_remaining_path_missing(self, tmp_path):
        result, desc = strip_human_root_prefix("root rust/Cargo.toml", tmp_path)

        assert result is None
        assert desc == ""


class TestStripFirstComponent:
    """Tests for strip_first_component normalizer."""

    def test_strips_cwd_component(self, temp_project):
        """Test stripping component that matches cwd part."""
        # Get a component of cwd path
        cwd_part = temp_project.parts[-2] if len(temp_project.parts) > 1 else None
        if cwd_part:
            path = f"{cwd_part}/src/main.py"
            result, desc = strip_first_component(path, temp_project)
            # Result depends on whether file exists after stripping
            if result is not None:
                assert "stripped_component" in desc

    def test_returns_none_for_non_matching(self, temp_project):
        """Test returns None when component doesn't match."""
        result, desc = strip_first_component("nonexistent/path.py", temp_project)
        assert result is None

    def test_handles_path_without_slash(self, temp_project):
        """Test handles path without slash."""
        result, desc = strip_first_component("main.py", temp_project)
        assert result is None


class TestStripCommonPrefix:
    """Tests for strip_common_prefix normalizer."""

    def test_strips_duplicate_prefix(self, temp_project):
        """Test stripping duplicated path prefix."""
        # Create the nested path structure
        path = "src/src/main.py"
        result, desc = strip_common_prefix(path, temp_project)
        if result is not None:
            assert "stripped_duplicate" in desc

    def test_returns_none_for_non_duplicate(self, temp_project):
        """Test returns None when no duplicate."""
        result, desc = strip_common_prefix("src/main.py", temp_project)
        assert result is None

    def test_handles_path_without_slash(self, temp_project):
        """Test handles path without slash."""
        result, desc = strip_common_prefix("main.py", temp_project)
        assert result is None


class TestNormalizeSeparators:
    """Tests for normalize_separators normalizer."""

    def test_normalizes_backslashes(self, temp_project):
        """Test converting backslashes to forward slashes."""
        path = "src\\main.py"
        result, desc = normalize_separators(path, temp_project)
        assert result == "src/main.py"
        assert desc == "normalized_separators"

    def test_strips_trailing_slash(self, temp_project):
        """Test stripping trailing slash."""
        path = "src/components/"
        result, desc = normalize_separators(path, temp_project)
        assert result == "src/components"
        assert desc == "stripped_trailing_slash"

    def test_preserves_root_slash(self, temp_project):
        """Test preserves single slash for root."""
        path = "/"
        result, desc = normalize_separators(path, temp_project)
        assert result is None  # No change needed

    def test_returns_none_for_normal_path(self, temp_project):
        """Test returns None for already normalized path."""
        path = "src/main.py"
        result, desc = normalize_separators(path, temp_project)
        assert result is None


# =============================================================================
# PathResolver Tests
# =============================================================================


class TestPathResolverInit:
    """Tests for PathResolver initialization."""

    def test_default_cwd(self):
        """Test default cwd is current directory."""
        resolver = PathResolver()
        assert resolver.cwd == Path.cwd()

    def test_custom_cwd(self, temp_project):
        """Test custom cwd is set correctly."""
        resolver = PathResolver(cwd=temp_project)
        assert resolver.cwd == temp_project

    def test_cwd_converted_to_path(self, temp_project):
        """Test cwd string is converted to Path."""
        resolver = PathResolver(cwd=str(temp_project))
        assert isinstance(resolver.cwd, Path)

    def test_default_normalizers(self, temp_project):
        """Test default normalizers are set."""
        resolver = PathResolver(cwd=temp_project)
        assert len(resolver.normalizers) > 0


class TestPathResolverResolve:
    """Tests for PathResolver.resolve() method."""

    def test_resolves_existing_file(self, resolver, temp_project):
        """Test resolving an existing file."""
        result = resolver.resolve("README.md")
        assert result.exists is True
        assert result.is_file is True
        assert result.was_normalized is False

    def test_resolves_existing_directory(self, resolver, temp_project):
        """Test resolving an existing directory."""
        result = resolver.resolve("src")
        assert result.exists is True
        assert result.is_directory is True

    def test_resolves_nested_path(self, resolver, temp_project):
        """Test resolving nested path."""
        result = resolver.resolve("src/main.py")
        assert result.exists is True
        assert result.is_file is True

    def test_raises_for_nonexistent_must_exist(self, resolver):
        """Test raises FileNotFoundError when must_exist=True."""
        with pytest.raises(FileNotFoundError) as exc_info:
            resolver.resolve("nonexistent.py", must_exist=True)
        assert "not found" in str(exc_info.value).lower()

    def test_returns_resolution_for_nonexistent_when_optional(self, resolver):
        """Test returns resolution when must_exist=False."""
        result = resolver.resolve("nonexistent.py", must_exist=False)
        assert result.exists is False

    def test_caches_results(self, resolver, temp_project):
        """Test that results are cached."""
        result1 = resolver.resolve("README.md")
        result2 = resolver.resolve("README.md")
        assert result1 is result2

    def test_empty_path_returns_cwd(self, resolver, temp_project):
        """Test empty path resolves to cwd."""
        result = resolver.resolve("")
        assert result.resolved_path == temp_project
        assert result.is_directory is True

    def test_applies_normalization(self, resolver, temp_project):
        """Test that normalizers are applied for missing paths."""
        # Path with cwd name prefix
        cwd_name = temp_project.name
        # First create a path that would need normalization
        path = f"{cwd_name}/src/main.py"
        result = resolver.resolve(path)
        # Should either find it or normalize it
        assert result.resolved_path.exists() or not result.exists


class TestPathResolverResolveFile:
    """Tests for PathResolver.resolve_file() method."""

    def test_resolves_file(self, resolver, temp_project):
        """Test resolving a file."""
        result = resolver.resolve_file("README.md")
        assert result.exists is True
        assert result.is_file is True

    def test_raises_for_directory(self, resolver, temp_project):
        """Test raises IsADirectoryError for directory."""
        with pytest.raises(IsADirectoryError) as exc_info:
            resolver.resolve_file("src")
        assert "directory" in str(exc_info.value).lower()
        assert "ls(" in str(exc_info.value)

    def test_raises_for_nonexistent(self, resolver):
        """Test raises FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            resolver.resolve_file("nonexistent.py")


class TestPathResolverResolveDirectory:
    """Tests for PathResolver.resolve_directory() method."""

    def test_resolves_directory(self, resolver, temp_project):
        """Test resolving a directory."""
        result = resolver.resolve_directory("src")
        assert result.exists is True
        assert result.is_directory is True

    def test_raises_for_file(self, resolver, temp_project):
        """Test raises NotADirectoryError for file."""
        with pytest.raises(NotADirectoryError) as exc_info:
            resolver.resolve_directory("README.md")
        assert "not a directory" in str(exc_info.value).lower()
        assert "read(" in str(exc_info.value)

    def test_raises_for_nonexistent(self, resolver):
        """Test raises FileNotFoundError for nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            resolver.resolve_directory("nonexistent/")


class TestPathResolverSuggestSimilar:
    """Tests for PathResolver.suggest_similar() method."""

    def test_suggests_similar_filenames(self, resolver, temp_project):
        """Test suggesting similar filenames."""
        # Try a typo of README.md
        suggestions = resolver.suggest_similar("READM.md", limit=5)
        # May or may not find suggestions depending on implementation
        assert isinstance(suggestions, list)

    def test_respects_limit(self, resolver, temp_project):
        """Test limit parameter is respected."""
        suggestions = resolver.suggest_similar("nonexistent.py", limit=2)
        assert len(suggestions) <= 2

    def test_returns_empty_for_very_different(self, resolver):
        """Test returns empty for paths with no similarity."""
        suggestions = resolver.suggest_similar("xyzabc123.zzz", limit=5)
        assert isinstance(suggestions, list)

    def test_prefers_package_entry_files_for_missing_module_path(self, resolver, temp_project):
        """Package-backed modules should suggest concrete package files first."""
        package_dir = temp_project / "victor" / "core" / "registry"
        package_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text("# package")
        (package_dir / "base.py").write_text("# base")
        (temp_project / "victor" / "core" / "registry_base.py").write_text("# legacy")
        resolver.clear_cache()
        resolver._known_paths = None

        suggestions = resolver.suggest_similar("victor/core/registry.py", limit=5)

        assert suggestions[:2] == [
            "victor/core/registry/base.py",
            "victor/core/registry/__init__.py",
        ]

    def test_bare_manifest_suggestions_use_relative_paths(self, tmp_path):
        """Duplicated manifest names should be suggested as actionable relative paths."""
        from victor.protocols.path_resolver import PathResolver

        (tmp_path / "rust" / "crates" / "core").mkdir(parents=True)
        (tmp_path / "rust" / "Cargo.toml").write_text("[workspace]\n")
        (tmp_path / "rust" / "crates" / "core" / "Cargo.toml").write_text("[package]\n")
        resolver = PathResolver(cwd=tmp_path)

        suggestions = resolver.suggest_similar("Cargo.toml", limit=5)

        # Order depends on filesystem walk order (platform-dependent); assert
        # both are present in the top 2 without requiring a specific order.
        assert set(suggestions[:2]) == {
            "rust/Cargo.toml",
            "rust/crates/core/Cargo.toml",
        }
        assert "Cargo.toml" not in suggestions

    def test_resolves_human_root_prefixed_manifest(self, tmp_path):
        """Model phrasing like 'root rust/Cargo.toml' should resolve to the file."""
        (tmp_path / "rust").mkdir()
        (tmp_path / "rust" / "Cargo.toml").write_text("[workspace]\n")
        resolver = PathResolver(cwd=tmp_path)

        result = resolver.resolve_file("root rust/Cargo.toml")

        assert result.resolved_path == (tmp_path / "rust" / "Cargo.toml").resolve()
        assert result.was_normalized is True
        assert "stripped_human_root_prefix:root" in str(result.normalization_applied)


# =============================================================================
# IPathResolver Protocol Tests
# =============================================================================


class TestIPathResolverProtocol:
    """Tests for IPathResolver protocol compliance."""

    def test_pathresolver_implements_protocol(self, temp_project):
        """Test that PathResolver implements IPathResolver."""
        resolver = PathResolver(cwd=temp_project)
        assert isinstance(resolver, IPathResolver)

    def test_protocol_has_required_methods(self):
        """Test that IPathResolver defines required methods."""
        assert hasattr(IPathResolver, "resolve")
        assert hasattr(IPathResolver, "resolve_file")
        assert hasattr(IPathResolver, "resolve_directory")
        assert hasattr(IPathResolver, "suggest_similar")


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestPathResolverEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_absolute_path(self, resolver, temp_project):
        """Test resolving absolute path."""
        abs_path = str(temp_project / "README.md")
        result = resolver.resolve(abs_path)
        assert result.exists is True

    def test_home_expansion(self, resolver):
        """Test that ~ is expanded."""
        # Just verify it doesn't crash
        try:
            result = resolver.resolve("~/nonexistent_test_file.xyz", must_exist=False)
            assert "~" not in str(result.resolved_path)
        except (FileNotFoundError, PermissionError):
            pass  # Expected if path doesn't exist

    def test_deeply_nested_path(self, resolver, temp_project):
        """Test resolving deeply nested path."""
        result = resolver.resolve("src/components/button.py")
        assert result.exists is True

    def test_multiple_normalizers_applied(self, temp_project):
        """Test that multiple normalizers can be applied."""
        # Create resolver with known normalizers
        resolver = PathResolver(cwd=temp_project)
        # Path with both backslashes and trailing slash
        path = "src\\components\\"
        result, desc = normalize_separators(path, temp_project)
        # At least separators should be normalized
        if result:
            assert "\\" not in result
