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

"""Unit tests for configuration migration utility."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from victor.config.migration import (
    ConfigMigrator,
    MigrationResult,
    check_migration_needed,
    run_migration,
    rollback_migration,
)


@pytest.fixture
def temp_victor_dir():
    """Create a temporary .victor directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        victor_dir = Path(tmpdir) / ".victor"
        victor_dir.mkdir()
        yield victor_dir


@pytest.fixture
def old_profiles_file(temp_victor_dir):
    """Create an old profiles.yaml file."""
    profiles_path = temp_victor_dir / "profiles.yaml"
    profiles_data = {
        "default_profile": "default",
        "profiles": {
            "default": {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "temperature": 0.7,
                "max_tokens": 4096,
                "description": "Default profile",
            },
            "coding": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.5,
                "max_tokens": 8192,
                "description": "Coding profile",
            },
            "local": {
                "provider": "ollama",
                "model": "llama3",
                "temperature": 0.8,
                "description": "Local Ollama",
            },
        },
    }
    with open(profiles_path, "w") as f:
        yaml.dump(profiles_data, f)
    return profiles_path


@pytest.fixture
def old_api_keys_file(temp_victor_dir):
    """Create an old api_keys.yaml file."""
    keys_path = temp_victor_dir / "api_keys.yaml"
    keys_data = {
        "api_keys": {
            "anthropic": "sk-ant-test-key",
            "openai": "sk-openai-test-key",
        }
    }
    with open(keys_path, "w") as f:
        yaml.dump(keys_data, f)
    # Set secure permissions
    keys_path.chmod(0o600)
    return keys_path


@pytest.fixture
def migrator(temp_victor_dir):
    """Create a ConfigMigrator instance."""
    return ConfigMigrator(victor_dir=temp_victor_dir)


class TestMigrationResult:
    """Tests for MigrationResult."""

    def test_migration_result_creation(self):
        """Test creating MigrationResult."""
        result = MigrationResult(success=True, migrated_accounts=2, migrated_keys=2)
        assert result.success
        assert result.migrated_accounts == 2
        assert result.migrated_keys == 2
        assert result.errors == []
        assert result.warnings == []

    def test_migration_result_add_error(self):
        """Test adding error to MigrationResult."""
        result = MigrationResult(success=True, migrated_accounts=0, migrated_keys=0)
        result.add_error("Test error")
        assert len(result.errors) == 1
        assert result.errors[0] == "Test error"

    def test_migration_result_add_warning(self):
        """Test adding warning to MigrationResult."""
        result = MigrationResult(success=True, migrated_accounts=0, migrated_keys=0)
        result.add_warning("Test warning")
        assert len(result.warnings) == 1
        assert result.warnings[0] == "Test warning"


class TestConfigMigrator:
    """Tests for ConfigMigrator."""

    def test_migrator_init(self, temp_victor_dir):
        """Test ConfigMigrator initialization."""
        migrator = ConfigMigrator(victor_dir=temp_victor_dir)
        assert migrator.victor_dir == temp_victor_dir
        assert migrator.old_profiles_file == temp_victor_dir / "profiles.yaml"
        assert migrator.old_api_keys_file == temp_victor_dir / "api_keys.yaml"
        assert migrator.new_config_file == temp_victor_dir / "config.yaml"

    def test_detect_old_config_profiles_only(self, migrator, old_profiles_file):
        """Test detecting old config with only profiles.yaml."""
        assert migrator.detect_old_config()
        assert not migrator.detect_new_config()

    def test_detect_old_config_keys_only(self, migrator, old_api_keys_file):
        """Test detecting old config with only api_keys.yaml."""
        assert migrator.detect_old_config()
        assert not migrator.detect_new_config()

    def test_detect_old_config_both(self, migrator, old_profiles_file, old_api_keys_file):
        """Test detecting old config with both files."""
        assert migrator.detect_old_config()
        assert not migrator.detect_new_config()

    def test_detect_old_config_none(self, migrator):
        """Test detecting no old config."""
        assert not migrator.detect_old_config()

    def test_detect_new_config(self, migrator, temp_victor_dir):
        """Test detecting new config file."""
        # Create new config file
        new_config = temp_victor_dir / "config.yaml"
        new_config.touch()

        assert migrator.detect_new_config()

    def test_load_old_profiles(self, migrator, old_profiles_file):
        """Test loading old profiles.yaml."""
        profiles = migrator.load_old_profiles()
        assert "default_profile" in profiles
        assert "profiles" in profiles
        assert len(profiles["profiles"]) == 3

    def test_load_old_profiles_missing(self, migrator):
        """Test loading missing profiles.yaml."""
        profiles = migrator.load_old_profiles()
        assert profiles == {}

    def test_load_old_api_keys(self, migrator, old_api_keys_file):
        """Test loading old api_keys.yaml."""
        keys = migrator.load_old_api_keys()
        assert "api_keys" in keys
        assert "anthropic" in keys["api_keys"]
        assert "openai" in keys["api_keys"]

    def test_load_old_api_keys_missing(self, migrator):
        """Test loading missing api_keys.yaml."""
        keys = migrator.load_old_api_keys()
        assert keys == {}

    def test_create_backup(self, migrator, old_profiles_file, old_api_keys_file):
        """Test creating backup of old configs."""
        backup_path = migrator.create_backup()

        assert backup_path.exists()
        assert (backup_path / "profiles.yaml").exists()
        assert (backup_path / "api_keys.yaml").exists()

    def test_migrate_success(self, migrator, old_profiles_file, old_api_keys_file):
        """Test successful migration."""
        result = migrator.migrate(prompt=False)

        assert result.success
        assert result.migrated_accounts > 0
        assert migrator.new_config_file.exists()

        # Verify new config format
        with open(migrator.new_config_file, "r") as f:
            new_config = yaml.safe_load(f)

        assert "accounts" in new_config
        assert "defaults" in new_config

    def test_migrate_with_new_config_exists(self, migrator, old_profiles_file, temp_victor_dir):
        """Test migration when new config already exists."""
        # Create existing new config
        new_config = temp_victor_dir / "config.yaml"
        new_config.write_text("existing: config")

        result = migrator.migrate(prompt=False, force=False)

        assert not result.success
        assert len(result.errors) > 0

    def test_migrate_force_overwrite(self, migrator, old_profiles_file, temp_victor_dir):
        """Test migration with force flag."""
        # Create existing new config
        new_config = temp_victor_dir / "config.yaml"
        new_config.write_text("existing: config")

        result = migrator.migrate(prompt=False, force=True)

        assert result.success
        assert migrator.new_config_file.exists()

    def test_migrate_dry_run(self, migrator, old_profiles_file, old_api_keys_file, temp_victor_dir):
        """Test dry run migration."""
        # Create migrator with dry_run=True
        dry_run_migrator = ConfigMigrator(victor_dir=temp_victor_dir, dry_run=True)
        result = dry_run_migrator.migrate(prompt=False)

        assert result.success
        assert not dry_run_migrator.new_config_file.exists()  # Should not create file

    def test_rollback_migration(self, migrator, old_profiles_file, old_api_keys_file):
        """Test rolling back migration."""
        # First migrate
        migrator.migrate(prompt=False)

        # Verify new config exists
        assert migrator.new_config_file.exists()

        # Now rollback
        success = migrator.rollback()

        assert success
        assert not migrator.new_config_file.exists()
        # Old files should be restored
        assert migrator.old_profiles_file.exists()
        assert migrator.old_api_keys_file.exists()

    def test_rollback_without_backup(self, migrator):
        """Test rollback when no backup exists."""
        success = migrator.rollback()
        assert not success


class TestMigrationConversion:
    """Tests for migration conversion logic."""

    def test_convert_simple_profile(self, migrator):
        """Test converting a simple profile."""
        old_profiles = {
            "default_profile": "default",
            "profiles": {
                "default": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                }
            },
        }
        old_api_keys = {"api_keys": {"anthropic": "sk-ant-test"}}

        result = MigrationResult(success=False, migrated_accounts=0, migrated_keys=0)
        new_config = migrator._convert_to_new_format(old_profiles, old_api_keys, result)

        assert "accounts" in new_config
        assert "default" in new_config["accounts"]
        assert new_config["accounts"]["default"]["provider"] == "anthropic"
        assert new_config["accounts"]["default"]["model"] == "claude-sonnet-4-5"

    def test_convert_local_provider(self, migrator):
        """Test converting local provider profile."""
        old_profiles = {
            "default_profile": "local",
            "profiles": {
                "local": {
                    "provider": "ollama",
                    "model": "llama3",
                    "temperature": 0.8,
                }
            },
        }
        old_api_keys = {}

        result = MigrationResult(success=False, migrated_accounts=0, migrated_keys=0)
        new_config = migrator._convert_to_new_format(old_profiles, old_api_keys, result)

        assert "accounts" in new_config
        assert new_config["accounts"]["local"]["auth"]["method"] == "none"

    def test_convert_oauth_provider(self, migrator):
        """Test converting OAuth provider profile."""
        old_profiles = {
            "default_profile": "openai-oauth",
            "profiles": {
                "openai-oauth": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "auth_mode": "oauth",
                }
            },
        }
        old_api_keys = {}

        result = MigrationResult(success=False, migrated_accounts=0, migrated_keys=0)
        new_config = migrator._convert_to_new_format(old_profiles, old_api_keys, result)

        assert "accounts" in new_config
        assert new_config["accounts"]["openai-oauth"]["auth"]["method"] == "oauth"

    def test_convert_api_key_only(self, migrator):
        """Test converting API key without profile."""
        old_profiles = {}
        old_api_keys = {"api_keys": {"anthropic": "sk-ant-test"}}

        result = MigrationResult(success=False, migrated_accounts=0, migrated_keys=0)
        new_config = migrator._convert_to_new_format(old_profiles, old_api_keys, result)

        assert "accounts" in new_config
        assert "anthropic" in new_config["accounts"]
        assert new_config["accounts"]["anthropic"]["auth"]["source"] == "file"


class TestMigrationIntegration:
    """Integration tests for migration functionality."""

    def test_full_migration_workflow(self, migrator, old_profiles_file, old_api_keys_file):
        """Test complete migration workflow."""
        # Initial state
        assert migrator.detect_old_config()
        assert not migrator.detect_new_config()

        # Run migration
        result = migrator.migrate(prompt=False)

        # Verify success
        assert result.success
        assert migrator.new_config_file.exists()

        # Verify new config structure
        with open(migrator.new_config_file, "r") as f:
            new_config = yaml.safe_load(f)

        assert "accounts" in new_config
        assert "defaults" in new_config

        # Verify accounts were migrated
        assert len(new_config["accounts"]) > 0

        # Verify backup was created
        assert result.backup_path is not None
        assert result.backup_path.exists()

    def test_migration_with_invalid_profile(self, migrator, old_profiles_file):
        """Test migration with invalid profile data."""
        # Create profiles with missing provider
        invalid_profiles = {
            "default_profile": "invalid",
            "profiles": {
                "invalid": {
                    "model": "some-model",
                    # Missing provider
                }
            },
        }
        with open(migrator.old_profiles_file, "w") as f:
            yaml.dump(invalid_profiles, f)

        result = migrator.migrate(prompt=False)

        # Should still succeed but skip invalid profile
        assert result.success
        # Should have warnings about missing provider
        assert len(result.warnings) > 0


class TestGlobalMigrationFunctions:
    """Tests for global migration functions."""

    @patch("victor.config.migration.ConfigMigrator")
    def test_check_migration_needed(self, mock_migrator_class):
        """Test check_migration_needed function."""
        mock_migrator = Mock()
        mock_migrator.detect_old_config.return_value = True
        mock_migrator.detect_new_config.return_value = False
        mock_migrator_class.return_value = mock_migrator

        result = check_migration_needed()

        assert result is True
        mock_migrator_class.assert_called_once()

    @patch("victor.config.migration.ConfigMigrator")
    def test_run_migration(self, mock_migrator_class):
        """Test run_migration function."""
        mock_migrator = Mock()
        mock_result = Mock(success=True, migrated_accounts=2, migrated_keys=2)
        mock_migrator.migrate.return_value = mock_result
        mock_migrator_class.return_value = mock_migrator

        result = run_migration(prompt=False)

        assert result.success
        assert result.migrated_accounts == 2
        mock_migrator.migrate.assert_called_once_with(prompt=False, force=False)

    @patch("victor.config.migration.ConfigMigrator")
    def test_rollback_migration_global(self, mock_migrator_class):
        """Test rollback_migration function."""
        mock_migrator = Mock()
        mock_migrator.rollback.return_value = True
        mock_migrator_class.return_value = mock_migrator

        result = rollback_migration()

        assert result is True
        mock_migrator.rollback.assert_called_once_with(None)
