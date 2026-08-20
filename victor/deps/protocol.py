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

"""Dependency management protocol types.

Defines data structures for dependency analysis and management.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class DependencyType(Enum):
    """Types of dependencies."""

    RUNTIME = "runtime"  # Required for runtime
    DEV = "dev"  # Development only
    BUILD = "build"  # Build-time only
    OPTIONAL = "optional"  # Optional feature
    PEER = "peer"  # Peer dependency (npm)


class PackageManager(Enum):
    """Package managers."""

    PIP = "pip"
    POETRY = "poetry"
    CONDA = "conda"
    NPM = "npm"
    YARN = "yarn"
    PNPM = "pnpm"
    CARGO = "cargo"
    GO = "go"
    MAVEN = "maven"
    GRADLE = "gradle"


class VersionConstraint(Enum):
    """Version constraint types."""

    EXACT = "exact"  # ==1.0.0
    GREATER = "greater"  # >1.0.0
    GREATER_EQUAL = "greater_equal"  # >=1.0.0
    LESS = "less"  # <2.0.0
    LESS_EQUAL = "less_equal"  # <=2.0.0
    COMPATIBLE = "compatible"  # ~=1.0.0 or ^1.0.0
    RANGE = "range"  # >=1.0.0,<2.0.0
    ANY = "any"  # *


@dataclass
class Version:
    """A semantic version."""

    major: int = 0
    minor: int = 0
    patch: int = 0
    prerelease: str = ""
    build: str = ""

    @classmethod
    def parse(cls, version_str: str) -> "Version":
        """Parse a version string."""
        import re

        # Clean up the version string
        version_str = version_str.strip().lstrip("v")

        # Match semantic version pattern
        pattern = r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:-([a-zA-Z0-9.]+))?(?:\+([a-zA-Z0-9.]+))?$"
        match = re.match(pattern, version_str)

        if not match:
            return cls(0, 0, 0)

        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2) or 0),
            patch=int(match.group(3) or 0),
            prerelease=match.group(4) or "",
            build=match.group(5) or "",
        )

    def __str__(self) -> str:
        """Convert to string."""
        result = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            result += f"-{self.prerelease}"
        if self.build:
            result += f"+{self.build}"
        return result

    def __lt__(self, other: "Version") -> bool:
        """Compare versions."""
        return (self.major, self.minor, self.patch) < (
            other.major,
            other.minor,
            other.patch,
        )

    def __le__(self, other: "Version") -> bool:
        return self == other or self < other

    def __gt__(self, other: "Version") -> bool:
        return not self <= other

    def __ge__(self, other: "Version") -> bool:
        return not self < other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return False
        return (self.major, self.minor, self.patch) == (
            other.major,
            other.minor,
            other.patch,
        )


@dataclass
class PackageDependency:
    """A package dependency for package management.

    Renamed from Dependency to be semantically distinct:
    - PackageDependency (here): Package management with version tracking
    - SecurityDependency (victor.security.protocol): Security scanning with CVE info
    """

    name: str
    version_spec: str = ""  # Original version specification
    installed_version: Optional[str] = None
    latest_version: Optional[str] = None
    dependency_type: DependencyType = DependencyType.RUNTIME
    source: str = ""  # File where dependency was found
    extras: list[str] = field(default_factory=list)  # pip extras
    repository: Optional[str] = None
    is_direct: bool = True  # Direct vs transitive

    @property
    def is_outdated(self) -> bool:
        """Check if dependency is outdated."""
        if not self.installed_version or not self.latest_version:
            return False
        installed = Version.parse(self.installed_version)
        latest = Version.parse(self.latest_version)
        return installed < latest

    @property
    def update_available(self) -> Optional[str]:
        """Get available update version."""
        if self.is_outdated:
            return self.latest_version
        return None


# Backward compatibility alias
Dependency = PackageDependency


@dataclass
class DependencyConflict:
    """A dependency version conflict."""

    package: str
    required_by: list[tuple[str, str]]  # (package, version_spec)
    message: str = ""
    severity: str = "warning"  # warning, error


@dataclass
class DependencyVulnerability:
    """A security vulnerability in a dependency."""

    package: str
    installed_version: str
    vulnerability_id: str  # CVE or GHSA
    severity: str  # low, medium, high, critical
    title: str = ""
    description: str = ""
    fixed_version: Optional[str] = None
    url: str = ""


@dataclass
class DependencyUpdate:
    """A proposed dependency update."""

    package: str
    current_version: str
    new_version: str
    change_type: str  # major, minor, patch
    breaking: bool = False
    changelog_url: Optional[str] = None
    risk_score: float = 0.0  # 0-1


@dataclass
class DependencyGraph:
    """Graph of dependencies and their relationships."""

    root_packages: list[Dependency] = field(default_factory=list)
    all_packages: dict[str, Dependency] = field(default_factory=dict)
    edges: dict[str, list[str]] = field(default_factory=dict)  # package -> [dependencies]

    def get_dependents(self, package: str) -> list[str]:
        """Get packages that depend on the given package."""
        dependents = []
        for pkg, deps in self.edges.items():
            if package in deps:
                dependents.append(pkg)
        return dependents

    def get_dependencies(self, package: str) -> list[str]:
        """Get dependencies of a package."""
        return self.edges.get(package, [])

    def get_transitive_dependencies(self, package: str) -> set[str]:
        """Get all transitive dependencies."""
        visited = set()
        to_visit = [package]

        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)
            to_visit.extend(self.edges.get(current, []))

        visited.discard(package)  # Remove the package itself
        return visited


@dataclass
class LockFile:
    """A parsed lock file."""

    path: Path
    packages: dict[str, Dependency] = field(default_factory=dict)
    hash_algorithm: str = "sha256"
    hashes: dict[str, str] = field(default_factory=dict)


@dataclass
class DependencyAnalysis:
    """Result of dependency analysis."""

    dependencies: list[Dependency] = field(default_factory=list)
    dev_dependencies: list[Dependency] = field(default_factory=list)
    graph: Optional[DependencyGraph] = None
    conflicts: list[DependencyConflict] = field(default_factory=list)
    vulnerabilities: list[DependencyVulnerability] = field(default_factory=list)
    updates_available: list[DependencyUpdate] = field(default_factory=list)
    total_packages: int = 0
    direct_packages: int = 0
    outdated_packages: int = 0
    vulnerable_packages: int = 0


@dataclass
class DepsConfig:
    """Configuration for dependency management."""

    package_manager: Optional[PackageManager] = None  # Auto-detect if None
    check_vulnerabilities: bool = True
    check_updates: bool = True
    include_dev: bool = True
    include_transitive: bool = True
    cache_ttl: int = 3600  # Cache TTL in seconds
    pypi_url: str = "https://pypi.org/pypi"
    npm_registry: str = "https://registry.npmjs.org"
