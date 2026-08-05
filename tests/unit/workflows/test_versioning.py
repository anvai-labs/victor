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

"""Tests for workflow versioning system."""

from __future__ import annotations

import pytest


class TestWorkflowVersion:
    """Tests for WorkflowVersion parsing and operations."""

    def test_parse_full_semver(self):
        """Test parsing full semver string."""
        from victor.workflows.versioning import WorkflowVersion

        version = WorkflowVersion.parse("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_parse_simple_version(self):
        """Test parsing simple major.minor version."""
        from victor.workflows.versioning import WorkflowVersion

        version = WorkflowVersion.parse("2.1")
        assert version.major == 2
        assert version.minor == 1
        assert version.patch == 0

    def test_parse_prerelease(self):
        """Test parsing version with prerelease tag."""
        from victor.workflows.versioning import WorkflowVersion

        version = WorkflowVersion.parse("1.0.0-beta.1")
        assert version.major == 1
        assert version.prerelease == "beta.1"

    def test_parse_build_metadata(self):
        """Test parsing version with build metadata."""
        from victor.workflows.versioning import WorkflowVersion

        version = WorkflowVersion.parse("1.0.0+build.123")
        assert version.build == "build.123"

    def test_invalid_version_raises_error(self):
        """Test that invalid version string raises ValueError."""
        from victor.workflows.versioning import WorkflowVersion

        with pytest.raises(ValueError, match="Invalid version string"):
            WorkflowVersion.parse("invalid")

    def test_version_string_representation(self):
        """Test str() formatting."""
        from victor.workflows.versioning import WorkflowVersion

        version = WorkflowVersion(1, 2, 3)
        assert str(version) == "1.2.3"

        version_pre = WorkflowVersion(1, 0, 0, prerelease="alpha")
        assert str(version_pre) == "1.0.0-alpha"

    def test_version_comparison(self):
        """Test version comparison operators."""
        from victor.workflows.versioning import WorkflowVersion

        v1 = WorkflowVersion(1, 0, 0)
        v2 = WorkflowVersion(2, 0, 0)
        v1_1 = WorkflowVersion(1, 1, 0)

        assert v1 < v2
        assert v2 > v1
        assert v1 < v1_1
        assert v1 == WorkflowVersion(1, 0, 0)

    def test_version_compatibility(self):
        """Test backward compatibility check."""
        from victor.workflows.versioning import WorkflowVersion

        v1_0 = WorkflowVersion(1, 0, 0)
        v1_1 = WorkflowVersion(1, 1, 0)
        v2_0 = WorkflowVersion(2, 0, 0)

        assert v1_1.is_compatible_with(v1_0)
        assert not v1_0.is_compatible_with(v1_1)
        assert not v2_0.is_compatible_with(v1_0)

    def test_version_bumping(self):
        """Test version bump methods."""
        from victor.workflows.versioning import WorkflowVersion

        v1 = WorkflowVersion(1, 2, 3)

        assert v1.bump_major() == WorkflowVersion(2, 0, 0)
        assert v1.bump_minor() == WorkflowVersion(1, 3, 0)
        assert v1.bump_patch() == WorkflowVersion(1, 2, 4)


class TestMigrationStep:
    """Tests for MigrationStep."""

    def test_rename_field_migration(self):
        """Test field rename migration."""
        from victor.workflows.versioning import MigrationStep, MigrationType

        step = MigrationStep(
            type=MigrationType.RENAME_FIELD,
            old_value="old_key",
            new_value="new_key",
        )

        state = {"old_key": "value", "other": "data"}
        result = step.apply(state)

        assert "new_key" in result
        assert "old_key" not in result
        assert result["new_key"] == "value"

    def test_add_field_migration(self):
        """Test field add migration."""
        from victor.workflows.versioning import MigrationStep, MigrationType

        step = MigrationStep(
            type=MigrationType.ADD_FIELD,
            node_id="new_field",
            new_value="default",
        )

        state = {"existing": "data"}
        result = step.apply(state)

        assert result["new_field"] == "default"

    def test_remove_field_migration(self):
        """Test field remove migration."""
        from victor.workflows.versioning import MigrationStep, MigrationType

        step = MigrationStep(
            type=MigrationType.REMOVE_FIELD,
            node_id="to_remove",
        )

        state = {"to_remove": "data", "keep": "value"}
        result = step.apply(state)

        assert "to_remove" not in result
        assert result["keep"] == "value"

    def test_custom_migration(self):
        """Test custom migration handler."""
        from victor.workflows.versioning import MigrationStep, MigrationType

        def custom_handler(state):
            state["transformed"] = state.get("value", 0) * 2
            return state

        step = MigrationStep(
            type=MigrationType.CUSTOM,
            custom_handler=custom_handler,
        )

        state = {"value": 5}
        result = step.apply(state)

        assert result["transformed"] == 10


class TestWorkflowMigration:
    """Tests for WorkflowMigration."""

    def test_migration_creation(self):
        """Test creating a migration."""
        from victor.workflows.versioning import (
            WorkflowMigration,
            WorkflowVersion,
            MigrationStep,
            MigrationType,
        )

        migration = WorkflowMigration(
            from_version=WorkflowVersion(1, 0, 0),
            to_version=WorkflowVersion(2, 0, 0),
            steps=[
                MigrationStep(
                    type=MigrationType.RENAME_NODE,
                    old_value="old_node",
                    new_value="new_node",
                ),
            ],
            description="Major version upgrade",
            breaking=True,
        )

        assert migration.migration_id == "1.0.0->2.0.0"
        assert migration.breaking

    def test_migration_apply(self):
        """Test applying a migration."""
        from victor.workflows.versioning import (
            WorkflowMigration,
            WorkflowVersion,
            MigrationStep,
            MigrationType,
        )

        migration = WorkflowMigration(
            from_version=WorkflowVersion(1, 0, 0),
            to_version=WorkflowVersion(1, 1, 0),
            steps=[
                MigrationStep(
                    type=MigrationType.RENAME_FIELD,
                    old_value="data",
                    new_value="payload",
                ),
                MigrationStep(
                    type=MigrationType.ADD_FIELD,
                    node_id="version",
                    new_value="1.1.0",
                ),
            ],
        )

        state = {"data": "test"}
        result = migration.apply(state)

        assert result["payload"] == "test"
        assert result["version"] == "1.1.0"
        assert "data" not in result

    def test_migration_validation(self):
        """Test migration validation."""
        from victor.workflows.versioning import (
            WorkflowMigration,
            WorkflowVersion,
        )

        # Invalid: from >= to
        invalid = WorkflowMigration(
            from_version=WorkflowVersion(2, 0, 0),
            to_version=WorkflowVersion(1, 0, 0),
        )
        errors = invalid.validate()
        assert len(errors) > 0


class TestVersionedWorkflow:
    """Tests for VersionedWorkflow."""

    def test_versioned_workflow_creation(self):
        """Test creating a versioned workflow."""
        from victor.workflows.versioning import VersionedWorkflow, WorkflowVersion

        workflow = VersionedWorkflow(
            name="test_workflow",
            version=WorkflowVersion(1, 0, 0),
            definition={"nodes": []},
        )

        assert workflow.version_id == "test_workflow@1.0.0"
        assert workflow.checksum  # Should be computed

    def test_versioned_workflow_deprecated(self):
        """Test deprecation flag."""
        from victor.workflows.versioning import VersionedWorkflow, WorkflowVersion

        workflow = VersionedWorkflow(
            name="old_workflow",
            version=WorkflowVersion(1, 0, 0),
            definition={},
            deprecated=True,
            deprecation_message="Use v2 instead",
        )

        assert workflow.is_deprecated()


class TestWorkflowVersionRegistry:
    """Tests for WorkflowVersionRegistry."""

    def test_register_workflow(self):
        """Test registering a versioned workflow."""
        from victor.workflows.versioning import (
            WorkflowVersionRegistry,
            VersionedWorkflow,
            WorkflowVersion,
        )

        registry = WorkflowVersionRegistry()
        workflow = VersionedWorkflow(
            name="pipeline",
            version=WorkflowVersion(1, 0, 0),
            definition={},
        )

        registry.register(workflow)

        assert registry.get("pipeline", "1.0.0") is workflow

    def test_get_latest_version(self):
        """Test getting latest version."""
        from victor.workflows.versioning import (
            WorkflowVersionRegistry,
            VersionedWorkflow,
            WorkflowVersion,
        )

        registry = WorkflowVersionRegistry()

        for v in ["1.0.0", "1.1.0", "2.0.0"]:
            workflow = VersionedWorkflow(
                name="pipeline",
                version=WorkflowVersion.parse(v),
                definition={},
            )
            registry.register(workflow)

        latest = registry.get_latest("pipeline")
        assert str(latest.version) == "2.0.0"

    def test_get_all_versions(self):
        """Test getting all versions."""
        from victor.workflows.versioning import (
            WorkflowVersionRegistry,
            VersionedWorkflow,
            WorkflowVersion,
        )

        registry = WorkflowVersionRegistry()

        for v in ["1.0.0", "1.1.0", "2.0.0"]:
            workflow = VersionedWorkflow(
                name="pipeline",
                version=WorkflowVersion.parse(v),
                definition={},
            )
            registry.register(workflow)

        versions = registry.get_all_versions("pipeline")
        assert len(versions) == 3
        # Should be sorted newest first
        assert str(versions[0].version) == "2.0.0"

    def test_migration_path(self):
        """Test finding migration path."""
        from victor.workflows.versioning import (
            WorkflowVersionRegistry,
            VersionedWorkflow,
            WorkflowVersion,
            WorkflowMigration,
        )

        registry = WorkflowVersionRegistry()

        # Register migrations: 1.0 -> 1.1 -> 2.0
        registry.add_migration(
            "pipeline",
            WorkflowMigration(
                from_version=WorkflowVersion(1, 0, 0),
                to_version=WorkflowVersion(1, 1, 0),
            ),
        )
        registry.add_migration(
            "pipeline",
            WorkflowMigration(
                from_version=WorkflowVersion(1, 1, 0),
                to_version=WorkflowVersion(2, 0, 0),
            ),
        )

        # Find path from 1.0 to 2.0
        path = registry.get_migration_path(
            "pipeline",
            WorkflowVersion(1, 0, 0),
            WorkflowVersion(2, 0, 0),
        )

        assert len(path) == 2
        assert path[0].migration_id == "1.0.0->1.1.0"
        assert path[1].migration_id == "1.1.0->2.0.0"

    def test_migrate_state(self):
        """Test migrating state between versions."""
        from victor.workflows.versioning import (
            WorkflowVersionRegistry,
            WorkflowVersion,
            WorkflowMigration,
            MigrationStep,
            MigrationType,
        )

        registry = WorkflowVersionRegistry()

        registry.add_migration(
            "pipeline",
            WorkflowMigration(
                from_version=WorkflowVersion(1, 0, 0),
                to_version=WorkflowVersion(2, 0, 0),
                steps=[
                    MigrationStep(
                        type=MigrationType.RENAME_FIELD,
                        old_value="old_field",
                        new_value="new_field",
                    ),
                ],
            ),
        )

        state = {"old_field": "data"}
        result, applied = registry.migrate_state(
            "pipeline",
            state,
            "1.0.0",
            "2.0.0",
        )

        assert result["new_field"] == "data"
        assert "1.0.0->2.0.0" in applied

    def test_deprecate_version(self):
        """Test deprecating a version."""
        from victor.workflows.versioning import (
            WorkflowVersionRegistry,
            VersionedWorkflow,
            WorkflowVersion,
        )

        registry = WorkflowVersionRegistry()
        workflow = VersionedWorkflow(
            name="old",
            version=WorkflowVersion(1, 0, 0),
            definition={},
        )
        registry.register(workflow)

        result = registry.deprecate("old", "1.0.0", "Use v2")

        assert result
        assert registry.get("old", "1.0.0").deprecated
