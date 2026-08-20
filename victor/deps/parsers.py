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

"""Dependency file parsers for various package managers.

Supports requirements.txt, pyproject.toml, package.json, Cargo.toml, go.mod.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from victor.deps.protocol import (
    Dependency,
    DependencyType,
    PackageManager,
)

logger = logging.getLogger(__name__)


class BasePackageDependencyParser(ABC):
    """Abstract base for package dependency parsers.

    Renamed from BaseDependencyParser to be semantically distinct:
    - BasePackageDependencyParser (here): Package management with PackageManager
    - BaseSecurityDependencyParser (victor.security.scanner): Security scanning with ecosystem

    Implements Strategy pattern for different file formats.
    """

    @property
    @abstractmethod
    def package_manager(self) -> PackageManager:
        """Get the package manager this parser handles."""
        pass

    @property
    @abstractmethod
    def file_patterns(self) -> list[str]:
        """Get file patterns this parser handles."""
        pass

    @abstractmethod
    def parse(self, path: Path) -> tuple[list[Dependency], list[Dependency]]:
        """Parse a dependency file.

        Args:
            path: Path to the dependency file

        Returns:
            Tuple of (runtime_dependencies, dev_dependencies)
        """
        pass

    def can_parse(self, path: Path) -> bool:
        """Check if this parser can handle the file."""
        from fnmatch import fnmatch

        return any(fnmatch(path.name, pattern) for pattern in self.file_patterns)


# Backward compatibility alias
BaseDependencyParser = BasePackageDependencyParser


class RequirementsTxtParser(BaseDependencyParser):
    """Parser for requirements.txt files."""

    @property
    def package_manager(self) -> PackageManager:
        return PackageManager.PIP

    @property
    def file_patterns(self) -> list[str]:
        return ["requirements*.txt", "requirements/*.txt"]

    # Pattern for parsing pip requirements
    REQ_PATTERN = re.compile(
        r"^\s*"
        r"(?P<name>[a-zA-Z0-9][-a-zA-Z0-9._]*)"
        r"(?:\[(?P<extras>[^\]]+)\])?"
        r"(?P<version_spec>[^#;\n]*)"
        r"(?:;(?P<markers>[^#\n]*))?"
        r"(?:#(?P<comment>.*))?$"
    )

    def parse(self, path: Path) -> tuple[list[Dependency], list[Dependency]]:
        """Parse requirements.txt file."""
        dependencies = []

        try:
            content = path.read_text()
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")
            return [], []

        # Determine dependency type from filename
        dep_type = DependencyType.DEV if "dev" in path.name.lower() else DependencyType.RUNTIME

        for line in content.split("\n"):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Skip options like -r, -e, --extra-index-url
            if line.startswith("-"):
                continue

            # Parse the requirement
            match = self.REQ_PATTERN.match(line)
            if match:
                name = match.group("name")
                version_spec = (match.group("version_spec") or "").strip()
                extras = []
                if match.group("extras"):
                    extras = [e.strip() for e in match.group("extras").split(",")]

                dependencies.append(
                    Dependency(
                        name=name,
                        version_spec=version_spec,
                        dependency_type=dep_type,
                        source=str(path),
                        extras=extras,
                    )
                )

        if dep_type == DependencyType.DEV:
            return [], dependencies
        return dependencies, []


class PyprojectParser(BaseDependencyParser):
    """Parser for pyproject.toml files."""

    @property
    def package_manager(self) -> PackageManager:
        return PackageManager.POETRY

    @property
    def file_patterns(self) -> list[str]:
        return ["pyproject.toml"]

    def parse(self, path: Path) -> tuple[list[Dependency], list[Dependency]]:
        """Parse pyproject.toml file."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                logger.error("Neither tomllib nor tomli available")
                return [], []

        try:
            content = path.read_text()
            data = tomllib.loads(content)
        except Exception as e:
            logger.error(f"Failed to parse {path}: {e}")
            return [], []

        runtime_deps = []
        dev_deps = []

        # PEP 621 format (project.dependencies)
        if "project" in data:
            project = data["project"]

            # Main dependencies
            for dep_str in project.get("dependencies", []):
                dep = self._parse_pep508(dep_str, str(path))
                if dep:
                    runtime_deps.append(dep)

            # Optional dependencies
            for group, deps in project.get("optional-dependencies", {}).items():
                dep_type = DependencyType.DEV if "dev" in group.lower() else DependencyType.OPTIONAL
                for dep_str in deps:
                    dep = self._parse_pep508(dep_str, str(path))
                    if dep:
                        dep.dependency_type = dep_type
                        if dep_type == DependencyType.DEV:
                            dev_deps.append(dep)
                        else:
                            runtime_deps.append(dep)

        # Poetry format (tool.poetry.dependencies)
        if "tool" in data and "poetry" in data["tool"]:
            poetry = data["tool"]["poetry"]

            # Main dependencies
            for name, spec in poetry.get("dependencies", {}).items():
                if name.lower() == "python":
                    continue
                dep = self._parse_poetry_dep(name, spec, str(path))
                if dep:
                    runtime_deps.append(dep)

            # Dev dependencies (old format)
            for name, spec in poetry.get("dev-dependencies", {}).items():
                dep = self._parse_poetry_dep(name, spec, str(path), DependencyType.DEV)
                if dep:
                    dev_deps.append(dep)

            # Dev dependencies (new format with groups)
            for group, group_data in poetry.get("group", {}).items():
                dep_type = DependencyType.DEV if "dev" in group.lower() else DependencyType.OPTIONAL
                for name, spec in group_data.get("dependencies", {}).items():
                    dep = self._parse_poetry_dep(name, spec, str(path), dep_type)
                    if dep:
                        if dep_type == DependencyType.DEV:
                            dev_deps.append(dep)
                        else:
                            runtime_deps.append(dep)

        return runtime_deps, dev_deps

    def _parse_pep508(self, dep_str: str, source: str) -> Optional[Dependency]:
        """Parse a PEP 508 dependency string."""
        # Simple pattern for PEP 508
        pattern = r"^([a-zA-Z0-9][-a-zA-Z0-9._]*)(?:\[([^\]]+)\])?\s*(.*)$"
        match = re.match(pattern, dep_str.strip())

        if not match:
            return None

        name = match.group(1)
        extras = match.group(2).split(",") if match.group(2) else []
        version_spec = match.group(3).strip()

        return Dependency(
            name=name,
            version_spec=version_spec,
            source=source,
            extras=[e.strip() for e in extras],
        )

    def _parse_poetry_dep(
        self,
        name: str,
        spec: any,
        source: str,
        dep_type: DependencyType = DependencyType.RUNTIME,
    ) -> Optional[Dependency]:
        """Parse a Poetry dependency specification."""
        if isinstance(spec, str):
            return Dependency(
                name=name,
                version_spec=spec,
                dependency_type=dep_type,
                source=source,
            )
        elif isinstance(spec, dict):
            version = spec.get("version", "")
            extras = spec.get("extras", [])
            optional = spec.get("optional", False)

            return Dependency(
                name=name,
                version_spec=version,
                dependency_type=DependencyType.OPTIONAL if optional else dep_type,
                source=source,
                extras=extras,
            )

        return None


class PackageJsonParser(BaseDependencyParser):
    """Parser for package.json files."""

    @property
    def package_manager(self) -> PackageManager:
        return PackageManager.NPM

    @property
    def file_patterns(self) -> list[str]:
        return ["package.json"]

    def parse(self, path: Path) -> tuple[list[Dependency], list[Dependency]]:
        """Parse package.json file."""
        try:
            content = path.read_text()
            data = json.loads(content)
        except Exception as e:
            logger.error(f"Failed to parse {path}: {e}")
            return [], []

        runtime_deps = []
        dev_deps = []

        # Main dependencies
        for name, version in data.get("dependencies", {}).items():
            runtime_deps.append(
                Dependency(
                    name=name,
                    version_spec=version,
                    dependency_type=DependencyType.RUNTIME,
                    source=str(path),
                )
            )

        # Dev dependencies
        for name, version in data.get("devDependencies", {}).items():
            dev_deps.append(
                Dependency(
                    name=name,
                    version_spec=version,
                    dependency_type=DependencyType.DEV,
                    source=str(path),
                )
            )

        # Peer dependencies
        for name, version in data.get("peerDependencies", {}).items():
            runtime_deps.append(
                Dependency(
                    name=name,
                    version_spec=version,
                    dependency_type=DependencyType.PEER,
                    source=str(path),
                )
            )

        # Optional dependencies
        for name, version in data.get("optionalDependencies", {}).items():
            runtime_deps.append(
                Dependency(
                    name=name,
                    version_spec=version,
                    dependency_type=DependencyType.OPTIONAL,
                    source=str(path),
                )
            )

        return runtime_deps, dev_deps


class CargoTomlParser(BaseDependencyParser):
    """Parser for Cargo.toml files."""

    @property
    def package_manager(self) -> PackageManager:
        return PackageManager.CARGO

    @property
    def file_patterns(self) -> list[str]:
        return ["Cargo.toml"]

    def parse(self, path: Path) -> tuple[list[Dependency], list[Dependency]]:
        """Parse Cargo.toml file."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                logger.error("Neither tomllib nor tomli available")
                return [], []

        try:
            content = path.read_text()
            data = tomllib.loads(content)
        except Exception as e:
            logger.error(f"Failed to parse {path}: {e}")
            return [], []

        runtime_deps = []
        dev_deps = []

        # Main dependencies
        for name, spec in data.get("dependencies", {}).items():
            dep = self._parse_cargo_dep(name, spec, str(path))
            if dep:
                runtime_deps.append(dep)

        # Dev dependencies
        for name, spec in data.get("dev-dependencies", {}).items():
            dep = self._parse_cargo_dep(name, spec, str(path), DependencyType.DEV)
            if dep:
                dev_deps.append(dep)

        # Build dependencies
        for name, spec in data.get("build-dependencies", {}).items():
            dep = self._parse_cargo_dep(name, spec, str(path), DependencyType.BUILD)
            if dep:
                runtime_deps.append(dep)

        return runtime_deps, dev_deps

    def _parse_cargo_dep(
        self,
        name: str,
        spec: any,
        source: str,
        dep_type: DependencyType = DependencyType.RUNTIME,
    ) -> Optional[Dependency]:
        """Parse a Cargo dependency specification."""
        if isinstance(spec, str):
            return Dependency(
                name=name,
                version_spec=spec,
                dependency_type=dep_type,
                source=source,
            )
        elif isinstance(spec, dict):
            version = spec.get("version", "")
            features = spec.get("features", [])

            return Dependency(
                name=name,
                version_spec=version,
                dependency_type=dep_type,
                source=source,
                extras=features,
            )

        return None


class GoModParser(BaseDependencyParser):
    """Parser for go.mod files."""

    @property
    def package_manager(self) -> PackageManager:
        return PackageManager.GO

    @property
    def file_patterns(self) -> list[str]:
        return ["go.mod"]

    REQUIRE_PATTERN = re.compile(r"^\s*([^\s]+)\s+v([^\s]+)\s*(?://.*)?$")

    def parse(self, path: Path) -> tuple[list[Dependency], list[Dependency]]:
        """Parse go.mod file."""
        try:
            content = path.read_text()
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")
            return [], []

        dependencies = []
        in_require_block = False

        for line in content.split("\n"):
            line = line.strip()

            # Check for require block
            if line.startswith("require ("):
                in_require_block = True
                continue
            elif line == ")" and in_require_block:
                in_require_block = False
                continue

            # Parse single require or block entry
            if line.startswith("require ") or in_require_block:
                req_line = line
                if line.startswith("require "):
                    req_line = line[8:].strip()

                match = self.REQUIRE_PATTERN.match(req_line)
                if match:
                    dependencies.append(
                        Dependency(
                            name=match.group(1),
                            version_spec=f"v{match.group(2)}",
                            dependency_type=DependencyType.RUNTIME,
                            source=str(path),
                        )
                    )

        return dependencies, []


# Parser registry
PARSERS: list[type[BaseDependencyParser]] = [
    RequirementsTxtParser,
    PyprojectParser,
    PackageJsonParser,
    CargoTomlParser,
    GoModParser,
]


def get_parser(path: Path) -> Optional[BaseDependencyParser]:
    """Get a parser for the given file.

    Args:
        path: Path to dependency file

    Returns:
        Parser instance or None
    """
    for parser_class in PARSERS:
        parser = parser_class()
        if parser.can_parse(path):
            return parser
    return None


def detect_package_manager(directory: Path) -> Optional[PackageManager]:
    """Detect the package manager used in a directory.

    Args:
        directory: Directory to check

    Returns:
        Detected package manager or None
    """
    file_to_manager = {
        "pyproject.toml": PackageManager.POETRY,
        "requirements.txt": PackageManager.PIP,
        "package.json": PackageManager.NPM,
        "yarn.lock": PackageManager.YARN,
        "pnpm-lock.yaml": PackageManager.PNPM,
        "Cargo.toml": PackageManager.CARGO,
        "go.mod": PackageManager.GO,
        "pom.xml": PackageManager.MAVEN,
        "build.gradle": PackageManager.GRADLE,
    }

    for filename, manager in file_to_manager.items():
        if (directory / filename).exists():
            return manager

    return None
