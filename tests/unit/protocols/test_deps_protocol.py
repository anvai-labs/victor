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

"""Tests for deps protocol module - achieving 70%+ coverage."""

import pytest
from pathlib import Path

from victor.deps.protocol import (
    DependencyType,
    PackageManager,
    VersionConstraint,
    Version,
    Dependency,
    DependencyConflict,
    DependencyVulnerability,
    DependencyUpdate,
    DependencyGraph,
    LockFile,
    DependencyAnalysis,
    DepsConfig,
)


class TestDependencyType:
    """Tests for DependencyType enum."""

    def test_runtime_value(self):
        """Test RUNTIME value."""
        assert DependencyType.RUNTIME.value == "runtime"

    def test_dev_value(self):
        """Test DEV value."""
        assert DependencyType.DEV.value == "dev"

    def test_build_value(self):
        """Test BUILD value."""
        assert DependencyType.BUILD.value == "build"

    def test_optional_value(self):
        """Test OPTIONAL value."""
        assert DependencyType.OPTIONAL.value == "optional"

    def test_peer_value(self):
        """Test PEER value."""
        assert DependencyType.PEER.value == "peer"

    def test_all_types_defined(self):
        """Test all expected types are defined."""
        expected = {"runtime", "dev", "build", "optional", "peer"}
        actual = {t.value for t in DependencyType}
        assert actual == expected


class TestPackageManager:
    """Tests for PackageManager enum."""

    def test_pip_value(self):
        """Test PIP value."""
        assert PackageManager.PIP.value == "pip"

    def test_poetry_value(self):
        """Test POETRY value."""
        assert PackageManager.POETRY.value == "poetry"

    def test_npm_value(self):
        """Test NPM value."""
        assert PackageManager.NPM.value == "npm"

    def test_cargo_value(self):
        """Test CARGO value."""
        assert PackageManager.CARGO.value == "cargo"

    def test_go_value(self):
        """Test GO value."""
        assert PackageManager.GO.value == "go"

    def test_all_managers_defined(self):
        """Test all expected managers are defined."""
        expected = {
            "pip",
            "poetry",
            "conda",
            "npm",
            "yarn",
            "pnpm",
            "cargo",
            "go",
            "maven",
            "gradle",
        }
        actual = {m.value for m in PackageManager}
        assert actual == expected


class TestVersionConstraint:
    """Tests for VersionConstraint enum."""

    def test_exact_value(self):
        """Test EXACT value."""
        assert VersionConstraint.EXACT.value == "exact"

    def test_greater_value(self):
        """Test GREATER value."""
        assert VersionConstraint.GREATER.value == "greater"

    def test_compatible_value(self):
        """Test COMPATIBLE value."""
        assert VersionConstraint.COMPATIBLE.value == "compatible"

    def test_range_value(self):
        """Test RANGE value."""
        assert VersionConstraint.RANGE.value == "range"

    def test_any_value(self):
        """Test ANY value."""
        assert VersionConstraint.ANY.value == "any"


class TestVersion:
    """Tests for Version dataclass."""

    def test_default_values(self):
        """Test default values."""
        v = Version()
        assert v.major == 0
        assert v.minor == 0
        assert v.patch == 0
        assert v.prerelease == ""
        assert v.build == ""

    def test_parse_full_version(self):
        """Test parsing full semantic version."""
        v = Version.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_parse_version_with_v_prefix(self):
        """Test parsing version with v prefix."""
        v = Version.parse("v2.0.1")
        assert v.major == 2
        assert v.minor == 0
        assert v.patch == 1

    def test_parse_major_only(self):
        """Test parsing major version only."""
        v = Version.parse("5")
        assert v.major == 5
        assert v.minor == 0
        assert v.patch == 0

    def test_parse_major_minor_only(self):
        """Test parsing major.minor version."""
        v = Version.parse("3.2")
        assert v.major == 3
        assert v.minor == 2
        assert v.patch == 0

    def test_parse_with_prerelease(self):
        """Test parsing version with prerelease."""
        v = Version.parse("1.0.0-alpha.1")
        assert v.major == 1
        assert v.minor == 0
        assert v.patch == 0
        assert v.prerelease == "alpha.1"

    def test_parse_with_build(self):
        """Test parsing version with build metadata."""
        v = Version.parse("1.0.0+build.123")
        assert v.major == 1
        assert v.build == "build.123"

    def test_parse_with_prerelease_and_build(self):
        """Test parsing version with both prerelease and build."""
        v = Version.parse("1.0.0-rc.1+build.456")
        assert v.prerelease == "rc.1"
        assert v.build == "build.456"

    def test_parse_invalid_version(self):
        """Test parsing invalid version returns 0.0.0."""
        v = Version.parse("invalid")
        assert v.major == 0
        assert v.minor == 0
        assert v.patch == 0

    def test_str_basic(self):
        """Test __str__ for basic version."""
        v = Version(1, 2, 3)
        assert str(v) == "1.2.3"

    def test_str_with_prerelease(self):
        """Test __str__ with prerelease."""
        v = Version(1, 0, 0, prerelease="alpha")
        assert str(v) == "1.0.0-alpha"

    def test_str_with_build(self):
        """Test __str__ with build."""
        v = Version(1, 0, 0, build="build.1")
        assert str(v) == "1.0.0+build.1"

    def test_str_with_prerelease_and_build(self):
        """Test __str__ with both prerelease and build."""
        v = Version(1, 0, 0, prerelease="rc.1", build="123")
        assert str(v) == "1.0.0-rc.1+123"

    def test_lt_comparison(self):
        """Test less than comparison."""
        v1 = Version(1, 0, 0)
        v2 = Version(2, 0, 0)
        assert v1 < v2

    def test_lt_minor_comparison(self):
        """Test less than comparison on minor version."""
        v1 = Version(1, 1, 0)
        v2 = Version(1, 2, 0)
        assert v1 < v2

    def test_lt_patch_comparison(self):
        """Test less than comparison on patch version."""
        v1 = Version(1, 1, 1)
        v2 = Version(1, 1, 2)
        assert v1 < v2

    def test_le_comparison(self):
        """Test less than or equal comparison."""
        v1 = Version(1, 0, 0)
        v2 = Version(1, 0, 0)
        assert v1 <= v2

    def test_gt_comparison(self):
        """Test greater than comparison."""
        v1 = Version(2, 0, 0)
        v2 = Version(1, 0, 0)
        assert v1 > v2

    def test_ge_comparison(self):
        """Test greater than or equal comparison."""
        v1 = Version(2, 0, 0)
        v2 = Version(2, 0, 0)
        assert v1 >= v2

    def test_eq_comparison(self):
        """Test equality comparison."""
        v1 = Version(1, 2, 3)
        v2 = Version(1, 2, 3)
        assert v1 == v2

    def test_eq_comparison_with_non_version(self):
        """Test equality comparison with non-Version object."""
        v = Version(1, 0, 0)
        assert v != "1.0.0"


class TestDependency:
    """Tests for Dependency dataclass."""

    def test_default_values(self):
        """Test default values."""
        dep = Dependency(name="test")
        assert dep.name == "test"
        assert dep.version_spec == ""
        assert dep.installed_version is None
        assert dep.latest_version is None
        assert dep.dependency_type == DependencyType.RUNTIME
        assert dep.source == ""
        assert dep.extras == []
        assert dep.repository is None
        assert dep.is_direct is True

    def test_is_outdated_true(self):
        """Test is_outdated when package is outdated."""
        dep = Dependency(name="test", installed_version="1.0.0", latest_version="2.0.0")
        assert dep.is_outdated is True

    def test_is_outdated_false(self):
        """Test is_outdated when package is current."""
        dep = Dependency(name="test", installed_version="2.0.0", latest_version="2.0.0")
        assert dep.is_outdated is False

    def test_is_outdated_no_installed(self):
        """Test is_outdated when no installed version."""
        dep = Dependency(name="test", latest_version="2.0.0")
        assert dep.is_outdated is False

    def test_is_outdated_no_latest(self):
        """Test is_outdated when no latest version."""
        dep = Dependency(name="test", installed_version="1.0.0")
        assert dep.is_outdated is False

    def test_update_available_when_outdated(self):
        """Test update_available when outdated."""
        dep = Dependency(name="test", installed_version="1.0.0", latest_version="2.0.0")
        assert dep.update_available == "2.0.0"

    def test_update_available_when_current(self):
        """Test update_available when current."""
        dep = Dependency(name="test", installed_version="2.0.0", latest_version="2.0.0")
        assert dep.update_available is None


class TestDependencyConflict:
    """Tests for DependencyConflict dataclass."""

    def test_default_values(self):
        """Test default values."""
        conflict = DependencyConflict(
            package="test", required_by=[("pkg1", ">=1.0"), ("pkg2", "<2.0")]
        )
        assert conflict.package == "test"
        assert len(conflict.required_by) == 2
        assert conflict.message == ""
        assert conflict.severity == "warning"

    def test_with_custom_values(self):
        """Test with custom values."""
        conflict = DependencyConflict(
            package="test", required_by=[], message="Version conflict", severity="error"
        )
        assert conflict.message == "Version conflict"
        assert conflict.severity == "error"


class TestDependencyVulnerability:
    """Tests for DependencyVulnerability dataclass."""

    def test_required_fields(self):
        """Test required fields."""
        vuln = DependencyVulnerability(
            package="test",
            installed_version="1.0.0",
            vulnerability_id="CVE-2024-1234",
            severity="high",
        )
        assert vuln.package == "test"
        assert vuln.installed_version == "1.0.0"
        assert vuln.vulnerability_id == "CVE-2024-1234"
        assert vuln.severity == "high"

    def test_default_values(self):
        """Test default values."""
        vuln = DependencyVulnerability(
            package="test",
            installed_version="1.0.0",
            vulnerability_id="CVE-2024-1234",
            severity="medium",
        )
        assert vuln.title == ""
        assert vuln.description == ""
        assert vuln.fixed_version is None
        assert vuln.url == ""


class TestDependencyUpdate:
    """Tests for DependencyUpdate dataclass."""

    def test_required_fields(self):
        """Test required fields."""
        update = DependencyUpdate(
            package="test",
            current_version="1.0.0",
            new_version="2.0.0",
            change_type="major",
        )
        assert update.package == "test"
        assert update.current_version == "1.0.0"
        assert update.new_version == "2.0.0"
        assert update.change_type == "major"

    def test_default_values(self):
        """Test default values."""
        update = DependencyUpdate(
            package="test",
            current_version="1.0.0",
            new_version="1.1.0",
            change_type="minor",
        )
        assert update.breaking is False
        assert update.changelog_url is None
        assert update.risk_score == 0.0


class TestDependencyGraph:
    """Tests for DependencyGraph dataclass."""

    def test_default_values(self):
        """Test default values."""
        graph = DependencyGraph()
        assert graph.root_packages == []
        assert graph.all_packages == {}
        assert graph.edges == {}

    def test_get_dependents(self):
        """Test get_dependents method."""
        graph = DependencyGraph(
            edges={
                "pkg-a": ["pkg-b", "pkg-c"],
                "pkg-d": ["pkg-b"],
            }
        )
        dependents = graph.get_dependents("pkg-b")
        assert set(dependents) == {"pkg-a", "pkg-d"}

    def test_get_dependents_none(self):
        """Test get_dependents with no dependents."""
        graph = DependencyGraph(edges={"pkg-a": ["pkg-b"]})
        dependents = graph.get_dependents("pkg-a")
        assert dependents == []

    def test_get_dependencies(self):
        """Test get_dependencies method."""
        graph = DependencyGraph(edges={"pkg-a": ["pkg-b", "pkg-c"]})
        deps = graph.get_dependencies("pkg-a")
        assert deps == ["pkg-b", "pkg-c"]

    def test_get_dependencies_none(self):
        """Test get_dependencies with no dependencies."""
        graph = DependencyGraph()
        deps = graph.get_dependencies("pkg-a")
        assert deps == []

    def test_get_transitive_dependencies(self):
        """Test get_transitive_dependencies method."""
        graph = DependencyGraph(
            edges={
                "pkg-a": ["pkg-b"],
                "pkg-b": ["pkg-c"],
                "pkg-c": ["pkg-d"],
            }
        )
        transitive = graph.get_transitive_dependencies("pkg-a")
        assert transitive == {"pkg-b", "pkg-c", "pkg-d"}

    def test_get_transitive_dependencies_with_cycle(self):
        """Test get_transitive_dependencies with cycle."""
        graph = DependencyGraph(
            edges={
                "pkg-a": ["pkg-b"],
                "pkg-b": ["pkg-c"],
                "pkg-c": ["pkg-a"],  # Cycle
            }
        )
        # Should handle cycles without infinite loop
        transitive = graph.get_transitive_dependencies("pkg-a")
        assert transitive == {"pkg-b", "pkg-c"}


class TestLockFile:
    """Tests for LockFile dataclass."""

    def test_default_values(self):
        """Test default values."""
        lock = LockFile(path=Path("poetry.lock"))
        assert lock.path == Path("poetry.lock")
        assert lock.packages == {}
        assert lock.hash_algorithm == "sha256"
        assert lock.hashes == {}


class TestDependencyAnalysis:
    """Tests for DependencyAnalysis dataclass."""

    def test_default_values(self):
        """Test default values."""
        analysis = DependencyAnalysis()
        assert analysis.dependencies == []
        assert analysis.dev_dependencies == []
        assert analysis.graph is None
        assert analysis.conflicts == []
        assert analysis.vulnerabilities == []
        assert analysis.updates_available == []
        assert analysis.total_packages == 0
        assert analysis.direct_packages == 0
        assert analysis.outdated_packages == 0
        assert analysis.vulnerable_packages == 0


class TestDepsConfig:
    """Tests for DepsConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = DepsConfig()
        assert config.package_manager is None
        assert config.check_vulnerabilities is True
        assert config.check_updates is True
        assert config.include_dev is True
        assert config.include_transitive is True
        assert config.cache_ttl == 3600
        assert config.pypi_url == "https://pypi.org/pypi"
        assert config.npm_registry == "https://registry.npmjs.org"

    def test_custom_values(self):
        """Test custom values."""
        config = DepsConfig(
            package_manager=PackageManager.POETRY,
            check_vulnerabilities=False,
            cache_ttl=7200,
        )
        assert config.package_manager == PackageManager.POETRY
        assert config.check_vulnerabilities is False
        assert config.cache_ttl == 7200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
