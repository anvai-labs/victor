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

"""Workflow Versioning System with Migration Support.

Provides Temporal.io-like versioning capabilities:
- Semantic versioning for workflow definitions
- Version history tracking
- Migration support when definitions change
- Backward compatibility validation
- Version-aware execution (deterministic replay)

Usage:
    from victor.workflows.versioning import (
        WorkflowVersion,
        VersionedWorkflow,
        WorkflowVersionRegistry,
        get_version_registry,
    )

    # Create a versioned workflow
    workflow = VersionedWorkflow(
        name="data_pipeline",
        version=WorkflowVersion(1, 0, 0),
        definition=workflow_def,
    )

    # Register it
    registry = get_version_registry()
    registry.register(workflow)

    # Get specific version
    v1 = registry.get("data_pipeline", "1.0.0")

    # Get latest version
    latest = registry.get_latest("data_pipeline")

YAML Configuration:
    workflows:
      data_pipeline:
        version: "2.0.0"
        description: "Process data pipeline"
        migrations:
          from_version: "1.0.0"
          changes:
            - type: add_node
              node_id: validation_step
            - type: rename_node
              old_id: process
              new_id: transform
        nodes:
          ...
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Singleton instance
_registry_instance: Optional["WorkflowVersionRegistry"] = None
_registry_lock = threading.Lock()


class VersionCompareResult(Enum):
    """Result of version comparison."""

    LESS_THAN = -1
    EQUAL = 0
    GREATER_THAN = 1


@dataclass(frozen=True, order=True)
class WorkflowVersion:
    """Semantic version for workflows.

    Follows semver conventions:
    - MAJOR: Incompatible changes
    - MINOR: Backward-compatible additions
    - PATCH: Backward-compatible fixes

    Attributes:
        major: Major version number
        minor: Minor version number
        patch: Patch version number
        prerelease: Optional prerelease tag (e.g., "alpha", "beta.1")
        build: Optional build metadata
    """

    major: int
    minor: int = 0
    patch: int = 0
    prerelease: Optional[str] = field(default=None, compare=False)
    build: Optional[str] = field(default=None, compare=False)

    def __str__(self) -> str:
        """Format as semver string."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    @classmethod
    def parse(cls, version_str: str) -> "WorkflowVersion":
        """Parse a semver string.

        Args:
            version_str: Version string (e.g., "1.2.3", "1.0.0-beta")

        Returns:
            WorkflowVersion instance

        Raises:
            ValueError: If version string is invalid
        """
        # Regex for semver with optional prerelease and build
        pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.]+))?(?:\+([a-zA-Z0-9.]+))?$"
        match = re.match(pattern, version_str.strip())

        if not match:
            # Try simple format "1.0" -> "1.0.0"
            simple_pattern = r"^(\d+)\.(\d+)$"
            simple_match = re.match(simple_pattern, version_str.strip())
            if simple_match:
                return cls(
                    major=int(simple_match.group(1)),
                    minor=int(simple_match.group(2)),
                    patch=0,
                )
            raise ValueError(f"Invalid version string: '{version_str}'")

        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4),
            build=match.group(5),
        )

    def is_compatible_with(self, other: "WorkflowVersion") -> bool:
        """Check if this version is backward-compatible with another.

        Same major version with >= minor/patch is considered compatible.

        Args:
            other: Version to check compatibility with

        Returns:
            True if compatible
        """
        return self.major == other.major and (
            self.minor > other.minor or (self.minor == other.minor and self.patch >= other.patch)
        )

    def bump_major(self) -> "WorkflowVersion":
        """Create a new version with bumped major."""
        return WorkflowVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> "WorkflowVersion":
        """Create a new version with bumped minor."""
        return WorkflowVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "WorkflowVersion":
        """Create a new version with bumped patch."""
        return WorkflowVersion(self.major, self.minor, self.patch + 1)


class MigrationType(Enum):
    """Types of workflow migrations."""

    ADD_NODE = "add_node"
    REMOVE_NODE = "remove_node"
    RENAME_NODE = "rename_node"
    MODIFY_NODE = "modify_node"
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    CHANGE_START_NODE = "change_start_node"
    ADD_FIELD = "add_field"
    REMOVE_FIELD = "remove_field"
    RENAME_FIELD = "rename_field"
    CUSTOM = "custom"


@dataclass
class MigrationStep:
    """A single migration step.

    Describes a change between workflow versions.

    Attributes:
        type: Type of migration
        node_id: Node being affected (if applicable)
        old_value: Previous value (for renames/modifications)
        new_value: New value
        description: Human-readable description
        custom_handler: Custom migration function (for CUSTOM type)
    """

    type: MigrationType
    node_id: Optional[str] = None
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    description: str = ""
    custom_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply this migration to workflow state.

        Used for migrating in-progress workflow state when
        workflow definition changes.

        Args:
            state: Current workflow state

        Returns:
            Migrated state
        """
        if self.type == MigrationType.RENAME_FIELD:
            if self.old_value in state:
                state[self.new_value] = state.pop(self.old_value)

        elif self.type == MigrationType.ADD_FIELD:
            if self.node_id not in state:
                state[self.node_id] = self.new_value

        elif self.type == MigrationType.REMOVE_FIELD:
            state.pop(self.node_id, None)

        elif self.type == MigrationType.CUSTOM and self.custom_handler:
            state = self.custom_handler(state)

        return state


@dataclass
class WorkflowMigration:
    """Migration between workflow versions.

    Describes how to migrate from one version to another.

    Attributes:
        from_version: Source version
        to_version: Target version
        steps: Migration steps to apply
        description: Human-readable description
        breaking: Whether this is a breaking change
        created_at: When this migration was created
    """

    from_version: WorkflowVersion
    to_version: WorkflowVersion
    steps: List[MigrationStep] = field(default_factory=list)
    description: str = ""
    breaking: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def migration_id(self) -> str:
        """Generate unique ID for this migration."""
        return f"{self.from_version}->{self.to_version}"

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all migration steps to state.

        Args:
            state: Current workflow state

        Returns:
            Migrated state
        """
        result = dict(state)
        for step in self.steps:
            result = step.apply(result)
        return result

    def validate(self) -> List[str]:
        """Validate the migration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if self.from_version >= self.to_version:
            errors.append(
                f"from_version ({self.from_version}) must be less than "
                f"to_version ({self.to_version})"
            )

        if self.breaking and self.from_version.major == self.to_version.major:
            errors.append("Breaking changes should bump major version")

        return errors


@dataclass
class VersionedWorkflow:
    """A workflow definition with version information.

    Wraps a WorkflowDefinition with versioning metadata.

    Attributes:
        name: Workflow name
        version: Semantic version
        definition: The workflow definition
        deprecated: Whether this version is deprecated
        deprecation_message: Message explaining deprecation
        min_compatible_version: Minimum version this is compatible with
        created_at: When this version was created
        checksum: Hash of the definition for change detection
    """

    name: str
    version: WorkflowVersion
    definition: Any  # WorkflowDefinition
    deprecated: bool = False
    deprecation_message: str = ""
    min_compatible_version: Optional[WorkflowVersion] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checksum: str = ""

    def __post_init__(self):
        """Calculate checksum if not provided."""
        if not self.checksum and self.definition:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate a checksum of the definition."""
        if hasattr(self.definition, "to_dict"):
            data = self.definition.to_dict()
        else:
            data = str(self.definition)

        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    @property
    def version_id(self) -> str:
        """Get unique identifier for this version."""
        return f"{self.name}@{self.version}"

    def is_deprecated(self) -> bool:
        """Check if this version is deprecated."""
        return self.deprecated


class WorkflowVersionRegistry:
    """Registry for versioned workflows.

    Manages multiple versions of workflows with migration support.

    Example:
        registry = WorkflowVersionRegistry()

        # Register versions
        registry.register(VersionedWorkflow(
            name="pipeline",
            version=WorkflowVersion(1, 0, 0),
            definition=v1_def,
        ))
        registry.register(VersionedWorkflow(
            name="pipeline",
            version=WorkflowVersion(2, 0, 0),
            definition=v2_def,
        ))

        # Register migration
        registry.add_migration(WorkflowMigration(
            from_version=WorkflowVersion(1, 0, 0),
            to_version=WorkflowVersion(2, 0, 0),
            steps=[...],
        ))

        # Get specific version
        v1 = registry.get("pipeline", "1.0.0")

        # Get latest
        latest = registry.get_latest("pipeline")

        # Migrate state
        migrated = registry.migrate_state("pipeline", state, "1.0.0", "2.0.0")
    """

    def __init__(self):
        """Initialize the registry."""
        self._versions: Dict[str, Dict[str, VersionedWorkflow]] = {}
        self._migrations: Dict[str, Dict[str, WorkflowMigration]] = {}
        self._lock = threading.RLock()

    def register(self, workflow: VersionedWorkflow) -> None:
        """Register a versioned workflow.

        Args:
            workflow: The versioned workflow to register
        """
        with self._lock:
            if workflow.name not in self._versions:
                self._versions[workflow.name] = {}

            version_key = str(workflow.version)
            if version_key in self._versions[workflow.name]:
                logger.warning(f"Overwriting existing version: {workflow.version_id}")

            self._versions[workflow.name][version_key] = workflow
            logger.info(f"Registered workflow version: {workflow.version_id}")

    def unregister(self, name: str, version: str) -> bool:
        """Unregister a workflow version.

        Args:
            name: Workflow name
            version: Version string

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if name in self._versions and version in self._versions[name]:
                del self._versions[name][version]
                logger.info(f"Unregistered workflow version: {name}@{version}")
                return True
            return False

    def get(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Optional[VersionedWorkflow]:
        """Get a workflow by name and optional version.

        Args:
            name: Workflow name
            version: Version string (defaults to latest)

        Returns:
            VersionedWorkflow or None
        """
        with self._lock:
            if name not in self._versions:
                return None

            if version is None:
                return self.get_latest(name)

            return self._versions[name].get(version)

    def get_latest(self, name: str) -> Optional[VersionedWorkflow]:
        """Get the latest version of a workflow.

        Args:
            name: Workflow name

        Returns:
            Latest VersionedWorkflow or None
        """
        with self._lock:
            if name not in self._versions:
                return None

            versions = self._versions[name]
            if not versions:
                return None

            # Sort by version and get highest
            sorted_versions = sorted(
                versions.values(),
                key=lambda w: w.version,
                reverse=True,
            )
            return sorted_versions[0]

    def get_all_versions(self, name: str) -> List[VersionedWorkflow]:
        """Get all versions of a workflow.

        Args:
            name: Workflow name

        Returns:
            List of versions, sorted newest first
        """
        with self._lock:
            if name not in self._versions:
                return []

            return sorted(
                self._versions[name].values(),
                key=lambda w: w.version,
                reverse=True,
            )

    def list_workflows(self) -> List[str]:
        """List all registered workflow names.

        Returns:
            List of workflow names
        """
        with self._lock:
            return list(self._versions.keys())

    def add_migration(
        self,
        name: str,
        migration: WorkflowMigration,
    ) -> None:
        """Add a migration between versions.

        Args:
            name: Workflow name
            migration: The migration to register
        """
        with self._lock:
            if name not in self._migrations:
                self._migrations[name] = {}

            migration_key = migration.migration_id
            self._migrations[name][migration_key] = migration
            logger.info(f"Registered migration: {name} {migration_key}")

    def get_migration_path(
        self,
        name: str,
        from_version: WorkflowVersion,
        to_version: WorkflowVersion,
    ) -> List[WorkflowMigration]:
        """Find migration path between versions.

        Args:
            name: Workflow name
            from_version: Source version
            to_version: Target version

        Returns:
            List of migrations to apply (in order)
        """
        with self._lock:
            if name not in self._migrations:
                return []

            # Simple case: direct migration exists
            direct_key = f"{from_version}->{to_version}"
            if direct_key in self._migrations[name]:
                return [self._migrations[name][direct_key]]

            # Find path through intermediate versions
            # (BFS for shortest path)
            migrations = self._migrations[name]
            visited = {str(from_version)}
            queue = [(from_version, [])]

            while queue:
                current, path = queue.pop(0)

                for key, migration in migrations.items():
                    if str(migration.from_version) == str(current):
                        new_path = path + [migration]

                        if str(migration.to_version) == str(to_version):
                            return new_path

                        if str(migration.to_version) not in visited:
                            visited.add(str(migration.to_version))
                            queue.append((migration.to_version, new_path))

            return []

    def migrate_state(
        self,
        name: str,
        state: Dict[str, Any],
        from_version: str,
        to_version: str,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Migrate workflow state between versions.

        Args:
            name: Workflow name
            state: Current state
            from_version: Current version
            to_version: Target version

        Returns:
            Tuple of (migrated_state, list of applied migrations)
        """
        from_v = WorkflowVersion.parse(from_version)
        to_v = WorkflowVersion.parse(to_version)

        path = self.get_migration_path(name, from_v, to_v)

        if not path:
            logger.warning(f"No migration path found: {name} {from_version} -> {to_version}")
            return state, []

        result = dict(state)
        applied = []

        for migration in path:
            result = migration.apply(result)
            applied.append(migration.migration_id)

        return result, applied

    def deprecate(
        self,
        name: str,
        version: str,
        message: str = "",
    ) -> bool:
        """Mark a version as deprecated.

        Args:
            name: Workflow name
            version: Version to deprecate
            message: Deprecation message

        Returns:
            True if deprecated, False if not found
        """
        with self._lock:
            workflow = self.get(name, version)
            if workflow:
                workflow.deprecated = True
                workflow.deprecation_message = message
                return True
            return False


def get_version_registry() -> WorkflowVersionRegistry:
    """Get the global workflow version registry.

    Thread-safe singleton access.

    Returns:
        Global WorkflowVersionRegistry instance
    """
    global _registry_instance

    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = WorkflowVersionRegistry()

    return _registry_instance


def parse_version_from_yaml(data: Dict[str, Any]) -> Optional[WorkflowVersion]:
    """Parse version from YAML workflow data.

    Args:
        data: Parsed YAML data

    Returns:
        WorkflowVersion or None
    """
    version_str = data.get("version")
    if not version_str:
        return None

    try:
        return WorkflowVersion.parse(str(version_str))
    except ValueError as e:
        logger.warning(f"Invalid version in YAML: {e}")
        return None


def parse_migrations_from_yaml(
    data: Dict[str, Any],
) -> List[WorkflowMigration]:
    """Parse migrations from YAML workflow data.

    Args:
        data: Parsed YAML data

    Returns:
        List of WorkflowMigration objects
    """
    migrations_data = data.get("migrations")
    if not migrations_data:
        return []

    migrations = []

    # Handle single migration or list
    if isinstance(migrations_data, dict):
        migrations_data = [migrations_data]

    for mig_data in migrations_data:
        from_str = mig_data.get("from_version")
        to_str = data.get("version")

        if not from_str or not to_str:
            continue

        steps = []
        for change in mig_data.get("changes", []):
            step_type = MigrationType(change.get("type", "custom"))
            steps.append(
                MigrationStep(
                    type=step_type,
                    node_id=change.get("node_id") or change.get("old_id"),
                    old_value=change.get("old_value") or change.get("old_id"),
                    new_value=change.get("new_value") or change.get("new_id"),
                    description=change.get("description", ""),
                )
            )

        migrations.append(
            WorkflowMigration(
                from_version=WorkflowVersion.parse(from_str),
                to_version=WorkflowVersion.parse(to_str),
                steps=steps,
                description=mig_data.get("description", ""),
                breaking=mig_data.get("breaking", False),
            )
        )

    return migrations


__all__ = [
    "WorkflowVersion",
    "VersionCompareResult",
    "MigrationType",
    "MigrationStep",
    "WorkflowMigration",
    "VersionedWorkflow",
    "WorkflowVersionRegistry",
    "get_version_registry",
    "parse_version_from_yaml",
    "parse_migrations_from_yaml",
]
