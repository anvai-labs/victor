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

"""Security scanner for detecting vulnerabilities in dependencies.

Parses dependency files and checks against CVE databases.
"""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from victor.security.cve_database import BaseCVEDatabase, get_cve_database
from victor.security.protocol import (
    Dependency,
    SecurityScanResult,
    Vulnerability,
    VulnerabilityStatus,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class DependencyParser(Protocol):
    """Protocol for dependency file parsers."""

    @property
    def ecosystem(self) -> str:
        """Package ecosystem name."""
        ...

    @property
    def file_patterns(self) -> list[str]:
        """Glob patterns for files this parser handles."""
        ...

    def parse(self, file_path: Path) -> list[Dependency]:
        """Parse a dependency file."""
        ...


class BaseSecurityDependencyParser(ABC):
    """Abstract base class for security dependency parsers.

    Renamed from BaseDependencyParser to be semantically distinct:
    - BaseSecurityDependencyParser (here): Security scanning with ecosystem property
    - BasePackageDependencyParser (victor.deps.parsers): Package management with PackageManager
    """

    @property
    @abstractmethod
    def ecosystem(self) -> str:
        """Package ecosystem name."""
        ...

    @property
    @abstractmethod
    def file_patterns(self) -> list[str]:
        """Glob patterns for files this parser handles."""
        ...

    @abstractmethod
    def parse(self, file_path: Path) -> list[Dependency]:
        """Parse a dependency file."""
        ...


# Backward compatibility alias
BaseDependencyParser = BaseSecurityDependencyParser


class PythonDependencyParser(BaseDependencyParser):
    """Parser for Python dependency files."""

    @property
    def ecosystem(self) -> str:
        return "pypi"

    @property
    def file_patterns(self) -> list[str]:
        return [
            "requirements.txt",
            "requirements*.txt",
            "Pipfile.lock",
            "poetry.lock",
            "setup.py",
            "pyproject.toml",
        ]

    def parse(self, file_path: Path) -> list[Dependency]:
        """Parse Python dependency files."""
        name = file_path.name

        if name == "requirements.txt" or name.startswith("requirements"):
            return self._parse_requirements(file_path)
        elif name == "Pipfile.lock":
            return self._parse_pipfile_lock(file_path)
        elif name == "poetry.lock":
            return self._parse_poetry_lock(file_path)
        elif name == "pyproject.toml":
            return self._parse_pyproject(file_path)

        return []

    def _parse_requirements(self, file_path: Path) -> list[Dependency]:
        """Parse requirements.txt format."""
        deps = []
        try:
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("-"):
                        continue

                    # Parse package==version, package>=version, etc.
                    match = re.match(r"^([a-zA-Z0-9_.-]+)([<>=!~]+)(.+)$", line)
                    if match:
                        name, op, version = match.groups()
                        # Clean version (remove extras like [security])
                        version = version.split("[")[0].split(";")[0].strip()
                        deps.append(
                            Dependency(
                                name=name.lower(),
                                version=version,
                                ecosystem=self.ecosystem,
                                source_file=file_path,
                            )
                        )
                    else:
                        # Package without version
                        name = line.split("[")[0].split(";")[0].strip()
                        if name:
                            deps.append(
                                Dependency(
                                    name=name.lower(),
                                    version="*",
                                    ecosystem=self.ecosystem,
                                    source_file=file_path,
                                )
                            )
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")

        return deps

    def _parse_pipfile_lock(self, file_path: Path) -> list[Dependency]:
        """Parse Pipfile.lock format."""
        deps = []
        try:
            with open(file_path) as f:
                data = json.load(f)

            for section in ["default", "develop"]:
                packages = data.get(section, {})
                for name, info in packages.items():
                    version = info.get("version", "").lstrip("=")
                    deps.append(
                        Dependency(
                            name=name.lower(),
                            version=version,
                            ecosystem=self.ecosystem,
                            source_file=file_path,
                            is_direct=section == "default",
                        )
                    )
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")

        return deps

    def _parse_poetry_lock(self, file_path: Path) -> list[Dependency]:
        """Parse poetry.lock format."""
        deps = []
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                logger.warning("tomllib/tomli not available for poetry.lock parsing")
                return deps

        try:
            with open(file_path, "rb") as f:
                data = tomllib.load(f)

            for package in data.get("package", []):
                name = package.get("name", "")
                version = package.get("version", "")
                deps.append(
                    Dependency(
                        name=name.lower(),
                        version=version,
                        ecosystem=self.ecosystem,
                        source_file=file_path,
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")

        return deps

    def _parse_pyproject(self, file_path: Path) -> list[Dependency]:
        """Parse pyproject.toml for dependencies."""
        deps = []
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                return deps

        try:
            with open(file_path, "rb") as f:
                data = tomllib.load(f)

            # PEP 621 format
            project = data.get("project", {})
            for dep_str in project.get("dependencies", []):
                match = re.match(r"^([a-zA-Z0-9_.-]+)([<>=!~]+)?(.+)?$", dep_str)
                if match:
                    name = match.group(1)
                    version = match.group(3) or "*"
                    deps.append(
                        Dependency(
                            name=name.lower(),
                            version=version.split(";")[0].strip(),
                            ecosystem=self.ecosystem,
                            source_file=file_path,
                        )
                    )

            # Poetry format
            poetry = data.get("tool", {}).get("poetry", {})
            for name, spec in poetry.get("dependencies", {}).items():
                if name.lower() == "python":
                    continue
                version = spec if isinstance(spec, str) else spec.get("version", "*")
                deps.append(
                    Dependency(
                        name=name.lower(),
                        version=version.lstrip("^~"),
                        ecosystem=self.ecosystem,
                        source_file=file_path,
                    )
                )

        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")

        return deps


class NodeDependencyParser(BaseDependencyParser):
    """Parser for Node.js dependency files."""

    @property
    def ecosystem(self) -> str:
        return "npm"

    @property
    def file_patterns(self) -> list[str]:
        return ["package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"]

    def parse(self, file_path: Path) -> list[Dependency]:
        """Parse Node.js dependency files."""
        name = file_path.name

        if name == "package-lock.json":
            return self._parse_package_lock(file_path)
        elif name == "package.json":
            return self._parse_package_json(file_path)

        return []

    def _parse_package_json(self, file_path: Path) -> list[Dependency]:
        """Parse package.json."""
        deps = []
        try:
            with open(file_path) as f:
                data = json.load(f)

            for dep_type in ["dependencies", "devDependencies"]:
                is_direct = dep_type == "dependencies"
                for name, version in data.get(dep_type, {}).items():
                    # Clean version specifier
                    version = version.lstrip("^~>=<")
                    deps.append(
                        Dependency(
                            name=name,
                            version=version,
                            ecosystem=self.ecosystem,
                            source_file=file_path,
                            is_direct=is_direct,
                        )
                    )
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")

        return deps

    def _parse_package_lock(self, file_path: Path) -> list[Dependency]:
        """Parse package-lock.json."""
        deps = []
        try:
            with open(file_path) as f:
                data = json.load(f)

            # npm v2 format
            packages = data.get("packages", {})
            for pkg_path, info in packages.items():
                if pkg_path == "" or not info.get("version"):
                    continue
                # Extract package name from path
                name = pkg_path.split("node_modules/")[-1]
                deps.append(
                    Dependency(
                        name=name,
                        version=info["version"],
                        ecosystem=self.ecosystem,
                        source_file=file_path,
                        is_direct=info.get("dev", False) is False,
                    )
                )

            # npm v1 format
            if not packages:
                for name, info in data.get("dependencies", {}).items():
                    deps.append(
                        Dependency(
                            name=name,
                            version=info.get("version", ""),
                            ecosystem=self.ecosystem,
                            source_file=file_path,
                        )
                    )

        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")

        return deps


class RustDependencyParser(BaseDependencyParser):
    """Parser for Rust dependency files."""

    @property
    def ecosystem(self) -> str:
        return "cargo"

    @property
    def file_patterns(self) -> list[str]:
        return ["Cargo.toml", "Cargo.lock"]

    def parse(self, file_path: Path) -> list[Dependency]:
        """Parse Rust dependency files."""
        if file_path.name == "Cargo.lock":
            return self._parse_cargo_lock(file_path)
        return self._parse_cargo_toml(file_path)

    def _parse_cargo_toml(self, file_path: Path) -> list[Dependency]:
        """Parse Cargo.toml."""
        deps = []
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                return deps

        try:
            with open(file_path, "rb") as f:
                data = tomllib.load(f)

            for dep_type in ["dependencies", "dev-dependencies", "build-dependencies"]:
                is_direct = dep_type == "dependencies"
                for name, spec in data.get(dep_type, {}).items():
                    if isinstance(spec, str):
                        version = spec
                    elif isinstance(spec, dict):
                        version = spec.get("version", "*")
                    else:
                        version = "*"

                    deps.append(
                        Dependency(
                            name=name,
                            version=version.lstrip("^~"),
                            ecosystem=self.ecosystem,
                            source_file=file_path,
                            is_direct=is_direct,
                        )
                    )

        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")

        return deps

    def _parse_cargo_lock(self, file_path: Path) -> list[Dependency]:
        """Parse Cargo.lock."""
        deps = []
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                return deps

        try:
            with open(file_path, "rb") as f:
                data = tomllib.load(f)

            for package in data.get("package", []):
                name = package.get("name", "")
                version = package.get("version", "")
                deps.append(
                    Dependency(
                        name=name,
                        version=version,
                        ecosystem=self.ecosystem,
                        source_file=file_path,
                    )
                )

        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")

        return deps


class GoDependencyParser(BaseDependencyParser):
    """Parser for Go dependency files."""

    @property
    def ecosystem(self) -> str:
        return "go"

    @property
    def file_patterns(self) -> list[str]:
        return ["go.mod", "go.sum"]

    def parse(self, file_path: Path) -> list[Dependency]:
        """Parse Go dependency files."""
        if file_path.name == "go.sum":
            return self._parse_go_sum(file_path)
        return self._parse_go_mod(file_path)

    def _parse_go_mod(self, file_path: Path) -> list[Dependency]:
        """Parse go.mod."""
        deps = []
        in_require = False

        try:
            with open(file_path) as f:
                for line in f:
                    line = line.strip()

                    if line.startswith("require ("):
                        in_require = True
                        continue
                    elif line == ")":
                        in_require = False
                        continue
                    elif line.startswith("require "):
                        # Single-line require
                        parts = line[8:].strip().split()
                        if len(parts) >= 2:
                            deps.append(
                                Dependency(
                                    name=parts[0],
                                    version=parts[1].lstrip("v"),
                                    ecosystem=self.ecosystem,
                                    source_file=file_path,
                                )
                            )

                    if in_require:
                        parts = line.split()
                        if len(parts) >= 2 and not parts[0].startswith("//"):
                            deps.append(
                                Dependency(
                                    name=parts[0],
                                    version=parts[1].lstrip("v"),
                                    ecosystem=self.ecosystem,
                                    source_file=file_path,
                                )
                            )

        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")

        return deps

    def _parse_go_sum(self, file_path: Path) -> list[Dependency]:
        """Parse go.sum."""
        deps = []
        seen = set()

        try:
            with open(file_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        name = parts[0]
                        version = parts[1].split("/")[0].lstrip("v")

                        key = (name, version)
                        if key not in seen:
                            seen.add(key)
                            deps.append(
                                Dependency(
                                    name=name,
                                    version=version,
                                    ecosystem=self.ecosystem,
                                    source_file=file_path,
                                )
                            )

        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")

        return deps


# Registry of parsers
DEPENDENCY_PARSERS: list[type[BaseDependencyParser]] = [
    PythonDependencyParser,
    NodeDependencyParser,
    RustDependencyParser,
    GoDependencyParser,
]


class SecurityScanner:
    """Scans dependencies for security vulnerabilities."""

    def __init__(
        self,
        cve_db: Optional[BaseCVEDatabase] = None,
        parsers: Optional[list[BaseDependencyParser]] = None,
        offline: bool = False,
    ):
        """Initialize the scanner.

        Args:
            cve_db: CVE database client
            parsers: List of dependency parsers
            offline: Whether to use offline mode
        """
        self._cve_db = cve_db or get_cve_database(offline=offline)
        self._parsers = parsers or [cls() for cls in DEPENDENCY_PARSERS]

    async def scan(
        self,
        project_root: Path,
        include_dev: bool = True,
    ) -> SecurityScanResult:
        """Scan a project for vulnerabilities.

        Args:
            project_root: Root directory of the project
            include_dev: Whether to include dev dependencies

        Returns:
            SecurityScanResult with findings
        """
        start_time = time.time()
        result = SecurityScanResult()

        # Find and parse dependency files
        for parser in self._parsers:
            for pattern in parser.file_patterns:
                for file_path in project_root.rglob(pattern):
                    # Skip node_modules and other vendor directories
                    if any(
                        part in file_path.parts
                        for part in ["node_modules", "vendor", ".git", "venv"]
                    ):
                        continue

                    try:
                        deps = parser.parse(file_path)
                        for dep in deps:
                            if not include_dev and not dep.is_direct:
                                continue
                            result.dependencies.append(dep)
                    except Exception as e:
                        result.errors.append(f"Failed to parse {file_path}: {e}")

        # Check for vulnerabilities
        for dep in result.dependencies:
            try:
                cves = await self._cve_db.search_by_package(dep.name, dep.ecosystem, dep.version)
                for cve in cves:
                    vuln = Vulnerability(
                        cve=cve,
                        dependency=dep,
                        status=VulnerabilityStatus.OPEN,
                    )
                    result.vulnerabilities.append(vuln)
            except Exception as e:
                result.errors.append(f"Failed to check {dep.name}: {e}")

        result.scan_duration_ms = (time.time() - start_time) * 1000
        return result

    async def scan_file(self, file_path: Path) -> SecurityScanResult:
        """Scan a single dependency file.

        Args:
            file_path: Path to dependency file

        Returns:
            SecurityScanResult
        """
        start_time = time.time()
        result = SecurityScanResult()

        # Find appropriate parser
        parser = None
        for p in self._parsers:
            import fnmatch

            if any(fnmatch.fnmatch(file_path.name, pat) for pat in p.file_patterns):
                parser = p
                break

        if parser is None:
            result.errors.append(f"No parser found for {file_path}")
            return result

        # Parse dependencies
        try:
            result.dependencies = parser.parse(file_path)
        except Exception as e:
            result.errors.append(f"Parse error: {e}")
            return result

        # Check vulnerabilities
        for dep in result.dependencies:
            try:
                cves = await self._cve_db.search_by_package(dep.name, dep.ecosystem, dep.version)
                for cve in cves:
                    vuln = Vulnerability(
                        cve=cve,
                        dependency=dep,
                        status=VulnerabilityStatus.OPEN,
                    )
                    result.vulnerabilities.append(vuln)
            except Exception as e:
                logger.debug(f"CVE check failed for {dep.name}: {e}")

        result.scan_duration_ms = (time.time() - start_time) * 1000
        return result


def get_scanner(offline: bool = False) -> SecurityScanner:
    """Get a configured security scanner.

    Args:
        offline: Whether to use offline mode

    Returns:
        SecurityScanner instance
    """
    return SecurityScanner(offline=offline)
