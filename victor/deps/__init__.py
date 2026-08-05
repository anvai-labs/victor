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

"""Smart dependency management module.

This module provides dependency analysis, conflict detection,
and update management for various package managers.

Example usage:
    from victor.deps import get_deps_manager, DepsConfig
    from pathlib import Path

    # Get manager
    manager = get_deps_manager()

    # Analyze dependencies
    analysis = manager.analyze(Path("my_project/"))
    print(f"Total packages: {analysis.total_packages}")
    print(f"Outdated: {analysis.outdated_packages}")

    # Get outdated dependencies
    outdated = manager.get_outdated()
    for update in outdated:
        print(f"{update.package}: {update.current_version} -> {update.new_version}")

    # Get conflicts
    conflicts = manager.get_conflicts()
    for conflict in conflicts:
        print(f"Conflict: {conflict.package} - {conflict.message}")

    # Get dependency tree
    tree = manager.get_dependency_tree("requests")

    # Format report
    report = manager.format_report(analysis, format="markdown")
    print(report)
"""

from victor.deps.protocol import (
    Dependency,
    DependencyAnalysis,
    DependencyConflict,
    DependencyGraph,
    DependencyType,
    DependencyUpdate,
    DependencyVulnerability,
    DepsConfig,
    LockFile,
    PackageManager,
    Version,
    VersionConstraint,
)
from victor.deps.parsers import (
    BaseDependencyParser,
    CargoTomlParser,
    GoModParser,
    PackageJsonParser,
    PyprojectParser,
    RequirementsTxtParser,
    detect_package_manager,
    get_parser,
)
from victor.deps.manager import (
    DepsManager,
    get_deps_manager,
    reset_deps_manager,
)

__all__ = [
    # Protocol types
    "Dependency",
    "DependencyAnalysis",
    "DependencyConflict",
    "DependencyGraph",
    "DependencyType",
    "DependencyUpdate",
    "DependencyVulnerability",
    "DepsConfig",
    "LockFile",
    "PackageManager",
    "Version",
    "VersionConstraint",
    # Parsers
    "BaseDependencyParser",
    "CargoTomlParser",
    "GoModParser",
    "PackageJsonParser",
    "PyprojectParser",
    "RequirementsTxtParser",
    "detect_package_manager",
    "get_parser",
    # Manager
    "DepsManager",
    "get_deps_manager",
    "reset_deps_manager",
]
