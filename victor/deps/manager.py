from __future__ import annotations

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

"""Dependency management manager.

Provides high-level API for dependency analysis and management.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from victor.deps.parsers import (
    detect_package_manager,
    get_parser,
)
from victor.deps.protocol import (
    Dependency,
    DependencyAnalysis,
    DependencyConflict,
    DependencyGraph,
    DependencyUpdate,
    DepsConfig,
    PackageManager,
    Version,
)

logger = logging.getLogger(__name__)


class DepsManager:
    """High-level manager for dependency operations.

    Orchestrates parsing, analysis, and management of dependencies.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        config: Optional[DepsConfig] = None,
    ):
        """Initialize the manager.

        Args:
            project_root: Root directory of the project
            config: Configuration
        """
        self.project_root = project_root or Path.cwd()
        self.config = config or DepsConfig()

        # Cache for version lookups
        self._version_cache: dict[str, str] = {}

    def analyze(
        self,
        directory: Optional[Path] = None,
    ) -> DependencyAnalysis:
        """Analyze dependencies in a directory.

        Args:
            directory: Directory to analyze (defaults to project root)

        Returns:
            DependencyAnalysis with findings
        """
        directory = directory or self.project_root
        analysis = DependencyAnalysis()

        # Detect package manager if not set
        pkg_manager = self.config.package_manager or detect_package_manager(directory)
        if not pkg_manager:
            logger.warning("Could not detect package manager")
            return analysis

        # Find and parse dependency files
        all_runtime = []
        all_dev = []

        for dep_file in self._find_dependency_files(directory):
            parser = get_parser(dep_file)
            if parser:
                runtime, dev = parser.parse(dep_file)
                all_runtime.extend(runtime)
                all_dev.extend(dev)

        # Remove duplicates
        analysis.dependencies = self._dedupe_dependencies(all_runtime)
        analysis.dev_dependencies = self._dedupe_dependencies(all_dev)

        analysis.total_packages = len(analysis.dependencies) + len(analysis.dev_dependencies)
        analysis.direct_packages = analysis.total_packages

        # Get installed versions
        if pkg_manager == PackageManager.PIP:
            self._get_installed_pip_versions(analysis.dependencies)
            self._get_installed_pip_versions(analysis.dev_dependencies)
        elif pkg_manager == PackageManager.NPM:
            self._get_installed_npm_versions(analysis.dependencies, directory)
            self._get_installed_npm_versions(analysis.dev_dependencies, directory)

        # Check for updates
        if self.config.check_updates:
            analysis.updates_available = self._check_updates(
                analysis.dependencies + analysis.dev_dependencies,
                pkg_manager,
            )
            analysis.outdated_packages = len(analysis.updates_available)

        # Check for conflicts
        all_deps = analysis.dependencies + analysis.dev_dependencies
        analysis.conflicts = self._detect_conflicts(all_deps)

        # Build dependency graph
        analysis.graph = self._build_graph(all_deps)

        return analysis

    def get_outdated(
        self,
        directory: Optional[Path] = None,
    ) -> list[DependencyUpdate]:
        """Get list of outdated dependencies.

        Args:
            directory: Directory to analyze

        Returns:
            List of available updates
        """
        analysis = self.analyze(directory)
        return analysis.updates_available

    def get_conflicts(
        self,
        directory: Optional[Path] = None,
    ) -> list[DependencyConflict]:
        """Get dependency conflicts.

        Args:
            directory: Directory to analyze

        Returns:
            List of conflicts
        """
        analysis = self.analyze(directory)
        return analysis.conflicts

    def get_dependency_tree(
        self,
        package: str,
        directory: Optional[Path] = None,
    ) -> dict:
        """Get dependency tree for a package.

        Args:
            package: Package name
            directory: Project directory

        Returns:
            Dependency tree as nested dict
        """
        analysis = self.analyze(directory)
        if not analysis.graph:
            return {}

        def build_tree(pkg_name: str, visited: set) -> dict:
            if pkg_name in visited:
                return {"name": pkg_name, "circular": True}
            visited.add(pkg_name)

            dep = analysis.graph.all_packages.get(pkg_name)
            if not dep:
                return {"name": pkg_name}

            children = []
            for child in analysis.graph.get_dependencies(pkg_name):
                children.append(build_tree(child, visited.copy()))

            return {
                "name": pkg_name,
                "version": dep.installed_version or dep.version_spec,
                "dependencies": children,
            }

        return build_tree(package, set())

    def suggest_updates(
        self,
        conservative: bool = True,
        directory: Optional[Path] = None,
    ) -> list[DependencyUpdate]:
        """Suggest dependency updates.

        Args:
            conservative: Only suggest minor/patch updates
            directory: Project directory

        Returns:
            List of suggested updates
        """
        updates = self.get_outdated(directory)

        if conservative:
            # Filter to only minor/patch updates
            updates = [u for u in updates if u.change_type in ("minor", "patch")]

        # Sort by risk score
        return sorted(updates, key=lambda u: u.risk_score)

    def format_report(
        self,
        analysis: DependencyAnalysis,
        format: str = "text",
    ) -> str:
        """Format analysis as report.

        Args:
            analysis: Dependency analysis
            format: Output format (text, json, markdown)

        Returns:
            Formatted report
        """
        if format == "json":
            return self._format_json(analysis)
        elif format == "markdown":
            return self._format_markdown(analysis)
        else:
            return self._format_text(analysis)

    def _format_text(self, analysis: DependencyAnalysis) -> str:
        """Format as plain text."""
        lines = []
        lines.append("=" * 60)
        lines.append("DEPENDENCY ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append(f"Total packages: {analysis.total_packages}")
        lines.append(f"Direct dependencies: {analysis.direct_packages}")
        lines.append(f"Outdated packages: {analysis.outdated_packages}")
        lines.append(f"Vulnerable packages: {analysis.vulnerable_packages}")

        # Runtime dependencies
        if analysis.dependencies:
            lines.append("")
            lines.append("Runtime Dependencies:")
            lines.append("-" * 40)
            for dep in analysis.dependencies[:20]:
                version = dep.installed_version or dep.version_spec or "?"
                lines.append(f"  {dep.name}: {version}")

        # Dev dependencies
        if analysis.dev_dependencies:
            lines.append("")
            lines.append("Development Dependencies:")
            lines.append("-" * 40)
            for dep in analysis.dev_dependencies[:20]:
                version = dep.installed_version or dep.version_spec or "?"
                lines.append(f"  {dep.name}: {version}")

        # Updates
        if analysis.updates_available:
            lines.append("")
            lines.append("Available Updates:")
            lines.append("-" * 40)
            for update in analysis.updates_available[:10]:
                lines.append(
                    f"  {update.package}: {update.current_version} -> "
                    f"{update.new_version} ({update.change_type})"
                )

        # Conflicts
        if analysis.conflicts:
            lines.append("")
            lines.append("Conflicts:")
            lines.append("-" * 40)
            for conflict in analysis.conflicts:
                lines.append(f"  {conflict.package}: {conflict.message}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _format_markdown(self, analysis: DependencyAnalysis) -> str:
        """Format as Markdown."""
        lines = []
        lines.append("# Dependency Analysis Report")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total packages:** {analysis.total_packages}")
        lines.append(f"- **Direct dependencies:** {analysis.direct_packages}")
        lines.append(f"- **Outdated:** {analysis.outdated_packages}")
        lines.append(f"- **Vulnerable:** {analysis.vulnerable_packages}")
        lines.append("")

        # Dependencies table
        if analysis.dependencies:
            lines.append("## Runtime Dependencies")
            lines.append("")
            lines.append("| Package | Version | Latest |")
            lines.append("|---------|---------|--------|")
            for dep in analysis.dependencies[:30]:
                version = dep.installed_version or dep.version_spec or "?"
                latest = dep.latest_version or "-"
                lines.append(f"| {dep.name} | {version} | {latest} |")
            lines.append("")

        # Updates
        if analysis.updates_available:
            lines.append("## Available Updates")
            lines.append("")
            for update in analysis.updates_available[:10]:
                breaking = " :warning:" if update.breaking else ""
                lines.append(
                    f"- **{update.package}**: {update.current_version} → "
                    f"{update.new_version} ({update.change_type}){breaking}"
                )
            lines.append("")

        return "\n".join(lines)

    def _format_json(self, analysis: DependencyAnalysis) -> str:
        """Format as JSON."""
        data = {
            "total_packages": analysis.total_packages,
            "direct_packages": analysis.direct_packages,
            "outdated_packages": analysis.outdated_packages,
            "vulnerable_packages": analysis.vulnerable_packages,
            "dependencies": [
                {
                    "name": d.name,
                    "version_spec": d.version_spec,
                    "installed": d.installed_version,
                    "latest": d.latest_version,
                    "type": d.dependency_type.value,
                }
                for d in analysis.dependencies
            ],
            "dev_dependencies": [
                {
                    "name": d.name,
                    "version_spec": d.version_spec,
                    "installed": d.installed_version,
                    "latest": d.latest_version,
                }
                for d in analysis.dev_dependencies
            ],
            "updates": [
                {
                    "package": u.package,
                    "current": u.current_version,
                    "new": u.new_version,
                    "change_type": u.change_type,
                    "breaking": u.breaking,
                }
                for u in analysis.updates_available
            ],
            "conflicts": [
                {
                    "package": c.package,
                    "message": c.message,
                    "severity": c.severity,
                }
                for c in analysis.conflicts
            ],
        }
        return json.dumps(data, indent=2)

    def _find_dependency_files(self, directory: Path) -> list[Path]:
        """Find all dependency files in directory."""
        files = []

        patterns = [
            "requirements*.txt",
            "pyproject.toml",
            "setup.py",
            "package.json",
            "Cargo.toml",
            "go.mod",
        ]

        for pattern in patterns:
            files.extend(directory.glob(pattern))

        # Also check common subdirectories
        for subdir in ["requirements"]:
            subpath = directory / subdir
            if subpath.exists():
                files.extend(subpath.glob("*.txt"))

        return files

    def _dedupe_dependencies(self, deps: list[Dependency]) -> list[Dependency]:
        """Remove duplicate dependencies, keeping most specific version."""
        seen = {}
        for dep in deps:
            if dep.name not in seen or dep.version_spec:
                seen[dep.name] = dep
        return list(seen.values())

    def _get_installed_pip_versions(self, deps: list[Dependency]) -> None:
        """Get installed versions for pip packages."""
        try:
            result = subprocess.run(
                ["pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                installed = {
                    pkg["name"].lower(): pkg["version"] for pkg in json.loads(result.stdout)
                }
                for dep in deps:
                    dep.installed_version = installed.get(dep.name.lower())
        except Exception as e:
            logger.debug(f"Failed to get pip versions: {e}")

    def _get_installed_npm_versions(
        self,
        deps: list[Dependency],
        directory: Path,
    ) -> None:
        """Get installed versions for npm packages."""
        try:
            result = subprocess.run(
                ["npm", "list", "--json", "--depth=0"],
                capture_output=True,
                text=True,
                cwd=directory,
                timeout=30,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                installed = data.get("dependencies", {})
                for dep in deps:
                    if dep.name in installed:
                        dep.installed_version = installed[dep.name].get("version")
        except Exception as e:
            logger.debug(f"Failed to get npm versions: {e}")

    def _check_updates(
        self,
        deps: list[Dependency],
        pkg_manager: PackageManager,
    ) -> list[DependencyUpdate]:
        """Check for available updates."""
        updates = []

        for dep in deps:
            if not dep.installed_version:
                continue

            latest = self._get_latest_version(dep.name, pkg_manager)
            if not latest:
                continue

            dep.latest_version = latest

            if dep.is_outdated:
                installed = Version.parse(dep.installed_version)
                latest_v = Version.parse(latest)

                # Determine change type
                if latest_v.major > installed.major:
                    change_type = "major"
                    breaking = True
                elif latest_v.minor > installed.minor:
                    change_type = "minor"
                    breaking = False
                else:
                    change_type = "patch"
                    breaking = False

                updates.append(
                    DependencyUpdate(
                        package=dep.name,
                        current_version=dep.installed_version,
                        new_version=latest,
                        change_type=change_type,
                        breaking=breaking,
                    )
                )

        return updates

    def _get_latest_version(
        self,
        package: str,
        pkg_manager: PackageManager,
    ) -> Optional[str]:
        """Get latest version of a package."""
        # Check cache first
        cache_key = f"{pkg_manager.value}:{package}"
        if cache_key in self._version_cache:
            return self._version_cache[cache_key]

        version = None

        if pkg_manager in (PackageManager.PIP, PackageManager.POETRY):
            version = self._get_pypi_version(package)
        elif pkg_manager in (
            PackageManager.NPM,
            PackageManager.YARN,
            PackageManager.PNPM,
        ):
            version = self._get_npm_version(package)

        if version:
            self._version_cache[cache_key] = version

        return version

    def _get_pypi_version(self, package: str) -> Optional[str]:
        """Get latest version from PyPI."""
        import urllib.request
        import urllib.error

        try:
            url = f"{self.config.pypi_url}/{package}/json"
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read())
                return data.get("info", {}).get("version")
        except Exception:
            return None

    def _get_npm_version(self, package: str) -> Optional[str]:
        """Get latest version from npm registry."""
        import urllib.request
        import urllib.error

        try:
            url = f"{self.config.npm_registry}/{package}"
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read())
                return data.get("dist-tags", {}).get("latest")
        except Exception:
            return None

    def _detect_conflicts(
        self,
        deps: list[Dependency],
    ) -> list[DependencyConflict]:
        """Detect version conflicts."""
        conflicts = []

        # Group by package name
        by_name: dict[str, list[Dependency]] = {}
        for dep in deps:
            if dep.name not in by_name:
                by_name[dep.name] = []
            by_name[dep.name].append(dep)

        # Check for conflicts
        for name, dep_list in by_name.items():
            if len(dep_list) > 1:
                versions = {d.version_spec for d in dep_list if d.version_spec}
                if len(versions) > 1:
                    conflicts.append(
                        DependencyConflict(
                            package=name,
                            required_by=[(d.source, d.version_spec) for d in dep_list],
                            message=f"Multiple version requirements: {', '.join(versions)}",
                        )
                    )

        return conflicts

    def _build_graph(self, deps: list[Dependency]) -> DependencyGraph:
        """Build dependency graph."""
        graph = DependencyGraph(
            root_packages=[d for d in deps if d.is_direct],
            all_packages={d.name: d for d in deps},
        )

        # Note: Full transitive dependency resolution would require
        # additional API calls or parsing lock files
        for dep in deps:
            graph.edges[dep.name] = []

        return graph


# Global manager singleton
_deps_manager: Optional[DepsManager] = None


def get_deps_manager(
    project_root: Optional[Path] = None,
    config: Optional[DepsConfig] = None,
) -> DepsManager:
    """Get the global dependency manager.

    Args:
        project_root: Project root directory
        config: Configuration

    Returns:
        DepsManager instance
    """
    global _deps_manager
    if _deps_manager is None or (project_root and _deps_manager.project_root != project_root):
        _deps_manager = DepsManager(project_root=project_root, config=config)
    return _deps_manager


def reset_deps_manager() -> None:
    """Reset the global manager."""
    global _deps_manager
    _deps_manager = None
