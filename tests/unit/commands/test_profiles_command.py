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

"""Tests for profiles CLI commands - achieving 70%+ coverage."""

import pytest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import yaml

from typer.testing import CliRunner
import typer

from victor.ui.commands.profiles import (
    profiles_app,
)
from victor.config.profiles import ProfileLevel, ProfileManager

runner = CliRunner()


class TestLoadProfilesYaml:
    """Tests for ProfileManager.load_profiles method."""

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading from a nonexistent file returns empty dict."""
        mgr = ProfileManager.for_config_dir(tmp_path)
        result = mgr.load_profiles()
        assert result == {"profiles": {}}

    def test_load_valid_yaml(self, tmp_path):
        """Test loading a valid profiles.yaml file."""
        profiles_file = tmp_path / "profiles.yaml"
        test_data = {
            "profiles": {
                "default": {
                    "provider": "ollama",
                    "model": "llama2",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                }
            }
        }
        with open(profiles_file, "w") as f:
            yaml.safe_dump(test_data, f)

        mgr = ProfileManager.for_config_dir(tmp_path)
        result = mgr.load_profiles()
        assert result == test_data

    def test_load_empty_yaml(self, tmp_path):
        """Test loading an empty yaml file."""
        profiles_file = tmp_path / "profiles.yaml"
        profiles_file.touch()

        mgr = ProfileManager.for_config_dir(tmp_path)
        result = mgr.load_profiles()
        # Function always returns profiles dict for consistency
        assert result == {"profiles": {}}

    def test_load_invalid_yaml(self, tmp_path):
        """Test loading invalid yaml returns empty dict with error."""
        profiles_file = tmp_path / "profiles.yaml"
        with open(profiles_file, "w") as f:
            f.write("invalid: yaml: content: [")

        mgr = ProfileManager.for_config_dir(tmp_path)
        result = mgr.load_profiles()
        assert result == {"profiles": {}}


class TestSaveProfilesYaml:
    """Tests for ProfileManager.save_profiles method."""

    def test_save_profiles_creates_directory(self, tmp_path):
        """Test saving creates parent directories if needed."""
        nested_dir = tmp_path / "nested" / "dir"
        test_data = {"profiles": {"test": {"provider": "test"}}}

        mgr = ProfileManager.for_config_dir(nested_dir)
        mgr.save_profiles(test_data)

        profiles_file = nested_dir / "profiles.yaml"
        assert profiles_file.exists()
        with open(profiles_file) as f:
            loaded = yaml.safe_load(f)
        assert loaded == test_data

    def test_save_profiles_overwrites_existing(self, tmp_path):
        """Test saving overwrites existing file."""
        profiles_file = tmp_path / "profiles.yaml"

        # Write initial data
        initial_data = {"profiles": {"old": {"provider": "old"}}}
        with open(profiles_file, "w") as f:
            yaml.safe_dump(initial_data, f)

        # Save new data
        new_data = {"profiles": {"new": {"provider": "new"}}}
        mgr = ProfileManager.for_config_dir(tmp_path)
        mgr.save_profiles(new_data)

        with open(profiles_file) as f:
            loaded = yaml.safe_load(f)
        assert loaded == new_data

    def test_save_profiles_error_handling(self, tmp_path):
        """Test saving handles permission errors."""
        from click.exceptions import Exit

        # Create read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        profiles_file = readonly_dir / "profiles.yaml"

        # Write initial file
        profiles_file.touch()

        # Make directory read-only - skip on Windows
        import sys

        if sys.platform != "win32":
            readonly_dir.chmod(0o444)
            try:
                mgr = ProfileManager.for_config_dir(readonly_dir)
                with pytest.raises(IOError):
                    mgr.save_profiles({"test": "data"})
            finally:
                readonly_dir.chmod(0o755)


class TestListProfiles:
    """Tests for list_profiles command."""

    def test_list_no_configured_profiles(self, tmp_path):
        """Test listing when no profiles configured."""
        result = runner.invoke(profiles_app, ["list", "--config-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "No configured profiles found" in result.stdout
        assert "profile templates" in result.stdout

    def test_list_with_configured_profiles(self, tmp_path):
        """Test listing profiles.yaml entries displays runtime settings."""
        profiles_file = tmp_path / "profiles.yaml"
        profiles_file.write_text(
            yaml.safe_dump(
                {
                    "default_profile": "cheap",
                    "profiles": {
                        "cheap": {
                            "provider": "openai",
                            "model": "gpt-5-nano",
                            "temperature": 0.2,
                            "max_tokens": 2048,
                            "auth": {"method": "oauth", "source": "keyring"},
                            "description": "Low-cost OpenAI smoke tests",
                        }
                    },
                }
            )
        )

        result = runner.invoke(profiles_app, ["list", "--config-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "Configured Profiles" in result.stdout
        assert "cheap" in result.stdout
        assert "openai" in result.stdout
        assert "gpt-5-nano" in result.stdout
        assert "oauth/keyring" in result.stdout

    def test_templates_lists_built_in_profiles(self):
        """Test built-in profile templates are available separately from configured profiles."""
        mock_profile = MagicMock()
        mock_profile.name = "default"
        mock_profile.display_name = "Default"
        mock_profile.level = ProfileLevel.BASIC
        mock_profile.description = "Test profile"
        mock_profile.settings = {
            "default_provider": "ollama",
            "default_model": "llama2",
        }

        with patch("victor.ui.commands.profiles.list_profiles", return_value=[mock_profile]):
            result = runner.invoke(profiles_app, ["templates"])

        assert result.exit_code == 0
        assert "Built-in Profile Templates" in result.stdout
        assert "Default" in result.stdout
        assert "ollama" in result.stdout


class TestCreateProfile:
    """Tests for create_profile command."""

    def test_create_new_profile(self, tmp_path):
        """Test creating a new profile successfully."""
        result = runner.invoke(
            profiles_app,
            [
                "create",
                "test_profile",
                "--provider",
                "ollama",
                "--model",
                "llama2",
                "--temperature",
                "0.5",
                "--max-tokens",
                "8192",
                "--description",
                "Test description",
                "--config-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0
        assert "Created profile" in result.stdout

    def test_create_profile_with_account_auth_and_default(self, tmp_path):
        """Test creating a profile linked to an account and OAuth source."""
        result = runner.invoke(
            profiles_app,
            [
                "create",
                "openai-cheap",
                "--provider",
                "openai",
                "--model",
                "gpt-5-nano",
                "--account",
                "openai-cheap",
                "--auth-method",
                "oauth",
                "--auth-source",
                "codex",
                "--temperature",
                "0.2",
                "--max-tokens",
                "2048",
                "--default",
                "--config-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0
        with open(tmp_path / "profiles.yaml") as f:
            data = yaml.safe_load(f)
        profile = data["profiles"]["openai-cheap"]
        assert data["default_profile"] == "openai-cheap"
        assert profile["account"] == "openai-cheap"
        assert profile["auth"] == {"method": "oauth", "source": "codex"}
        assert profile["temperature"] == 0.2
        assert profile["max_tokens"] == 2048

    def test_create_profile_already_exists(self, tmp_path):
        """Test creating a profile that already exists shows error."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {"profiles": {"existing": {"provider": "ollama", "model": "llama2"}}}
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        result = runner.invoke(
            profiles_app,
            [
                "create",
                "existing",
                "--provider",
                "anthropic",
                "--model",
                "claude",
                "--config-dir",
                str(tmp_path),
            ],
        )

        assert "already exists" in result.stdout

    def test_create_profile_without_description(self, tmp_path):
        """Test creating a profile without description."""
        result = runner.invoke(
            profiles_app,
            [
                "create",
                "simple",
                "--provider",
                "ollama",
                "--model",
                "llama2",
                "--config-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0

    def test_create_profile_uses_global_victor_dir_by_default(self, tmp_path):
        """Default CLI config dir should come from centralized Victor paths."""
        global_dir = tmp_path / ".victor"

        with patch(
            "victor.ui.commands.profiles.get_project_paths",
            return_value=SimpleNamespace(global_victor_dir=global_dir),
        ):
            result = runner.invoke(
                profiles_app,
                [
                    "create",
                    "default_path_profile",
                    "--provider",
                    "ollama",
                    "--model",
                    "llama2",
                ],
            )

        assert result.exit_code == 0
        assert (global_dir / "profiles.yaml").exists()


class TestEditProfile:
    """Tests for edit_profile command."""

    def test_edit_existing_profile(self, tmp_path):
        """Test editing an existing profile."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {
            "profiles": {
                "edit_me": {
                    "provider": "ollama",
                    "model": "llama2",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                }
            }
        }
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        result = runner.invoke(
            profiles_app,
            ["edit", "edit_me", "--temperature", "0.3", "--config-dir", str(tmp_path)],
        )

        assert result.exit_code == 0
        assert "Updated profile" in result.stdout

    def test_edit_nonexistent_profile(self, tmp_path):
        """Test editing a profile that doesn't exist."""
        profiles_file = tmp_path / "profiles.yaml"
        profiles_file.touch()

        result = runner.invoke(
            profiles_app,
            [
                "edit",
                "nonexistent",
                "--temperature",
                "0.5",
                "--config-dir",
                str(tmp_path),
            ],
        )

        assert "not found" in result.stdout

    def test_edit_no_changes(self, tmp_path):
        """Test editing with no changes specified."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {"profiles": {"no_change": {"provider": "ollama", "model": "llama2"}}}
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        result = runner.invoke(profiles_app, ["edit", "no_change", "--config-dir", str(tmp_path)])

        assert "No changes specified" in result.stdout

    def test_edit_all_fields(self, tmp_path):
        """Test editing all profile fields."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {
            "profiles": {
                "full_edit": {
                    "provider": "ollama",
                    "model": "llama2",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                }
            }
        }
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        result = runner.invoke(
            profiles_app,
            [
                "edit",
                "full_edit",
                "--provider",
                "anthropic",
                "--model",
                "claude",
                "--temperature",
                "0.5",
                "--max-tokens",
                "8192",
                "--description",
                "Updated",
                "--config-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0
        assert "Updated profile" in result.stdout


class TestDeleteProfile:
    """Tests for delete_profile command."""

    def test_delete_with_force(self, tmp_path):
        """Test deleting a profile with --force flag."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {"profiles": {"delete_me": {"provider": "ollama", "model": "llama2"}}}
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        result = runner.invoke(
            profiles_app,
            ["delete", "delete_me", "--force", "--config-dir", str(tmp_path)],
        )

        assert result.exit_code == 0
        assert "Deleted profile" in result.stdout

    def test_delete_nonexistent_profile(self, tmp_path):
        """Test deleting a profile that doesn't exist."""
        profiles_file = tmp_path / "profiles.yaml"
        profiles_file.touch()

        result = runner.invoke(
            profiles_app,
            ["delete", "nonexistent", "--force", "--config-dir", str(tmp_path)],
        )

        assert "not found" in result.stdout

    def test_delete_with_confirmation_yes(self, tmp_path):
        """Test deleting a profile with confirmation."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {"profiles": {"confirm_delete": {"provider": "ollama", "model": "llama2"}}}
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        result = runner.invoke(
            profiles_app,
            ["delete", "confirm_delete", "--config-dir", str(tmp_path)],
            input="y\n",
        )

        assert "Deleted profile" in result.stdout

    def test_delete_with_confirmation_no(self, tmp_path):
        """Test cancelling profile deletion."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {"profiles": {"keep_me": {"provider": "ollama", "model": "llama2"}}}
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        result = runner.invoke(
            profiles_app,
            ["delete", "keep_me", "--config-dir", str(tmp_path)],
            input="n\n",
        )

        assert "Cancelled" in result.stdout


class TestShowProfile:
    """Tests for show_profile command."""

    def test_show_existing_profile(self):
        """Test showing an existing profile."""
        mock_profile = MagicMock()
        mock_profile.name = "default"
        mock_profile.display_name = "Default"
        mock_profile.level = ProfileLevel.BASIC
        mock_profile.description = "Test profile"
        mock_profile.settings = {"provider": "ollama", "model": "llama2"}

        with patch("victor.ui.commands.profiles.get_profile", return_value=mock_profile):
            result = runner.invoke(profiles_app, ["show", "default"])

        assert result.exit_code == 0
        assert "Default" in result.stdout

    def test_show_nonexistent_profile(self):
        """Test showing a profile that doesn't exist."""
        with patch("victor.ui.commands.profiles.get_profile", return_value=None):
            result = runner.invoke(profiles_app, ["show", "nonexistent"])

        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_show_profile_with_tool_selection(self):
        """Test showing a profile with tool_selection configured."""
        mock_profile = MagicMock()
        mock_profile.name = "with_tools"
        mock_profile.display_name = "With Tools"
        mock_profile.level = ProfileLevel.EXPERT
        mock_profile.description = "Test profile with tools"
        mock_profile.settings = {
            "provider": "anthropic",
            "model": "claude",
            "tool_selection": "semantic",
        }

        with patch("victor.ui.commands.profiles.get_profile", return_value=mock_profile):
            result = runner.invoke(profiles_app, ["show", "with_tools"])

        assert result.exit_code == 0


class TestSetDefaultProfile:
    """Tests for set_default_profile command."""

    def test_set_default_success(self, tmp_path):
        """Test setting a profile as default."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {
            "profiles": {
                "default": {"provider": "ollama", "model": "llama2"},
                "anthropic": {"provider": "anthropic", "model": "claude"},
            }
        }
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        result = runner.invoke(
            profiles_app, ["set-default", "anthropic", "--config-dir", str(tmp_path)]
        )

        assert result.exit_code == 0
        assert "Set" in result.stdout or "default" in result.stdout

    def test_set_default_nonexistent(self, tmp_path):
        """Test setting nonexistent profile as default."""
        profiles_file = tmp_path / "profiles.yaml"
        profiles_file.touch()

        result = runner.invoke(
            profiles_app, ["set-default", "nonexistent", "--config-dir", str(tmp_path)]
        )

        assert "not found" in result.stdout

    def test_set_default_already_default(self, tmp_path):
        """Test setting default profile as default (no-op)."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {
            "default_profile": "default",
            "profiles": {
                "default": {"provider": "ollama", "model": "llama2"},
            },
        }
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        result = runner.invoke(
            profiles_app, ["set-default", "default", "--config-dir", str(tmp_path)]
        )

        assert "already the default" in result.stdout

    def test_set_default_without_existing_default(self, tmp_path):
        """Test setting default when no previous default exists."""
        profiles_file = tmp_path / "profiles.yaml"
        existing_data = {
            "profiles": {
                "new_default": {"provider": "anthropic", "model": "claude"},
            }
        }
        with open(profiles_file, "w") as f:
            yaml.safe_dump(existing_data, f)

        result = runner.invoke(
            profiles_app, ["set-default", "new_default", "--config-dir", str(tmp_path)]
        )

        assert result.exit_code == 0


class TestProfilesAppIntegration:
    """Integration tests for profiles CLI app."""

    def test_create_edit_delete_flow(self, tmp_path):
        """Test full create-edit-delete workflow."""
        cd = str(tmp_path)

        # Create
        result = runner.invoke(
            profiles_app,
            [
                "create",
                "workflow_test",
                "--provider",
                "ollama",
                "--model",
                "test",
                "--config-dir",
                cd,
            ],
        )
        assert "Created" in result.stdout

        # Edit
        result = runner.invoke(
            profiles_app,
            ["edit", "workflow_test", "--temperature", "0.8", "--config-dir", cd],
        )
        assert "Updated" in result.stdout

        # Delete
        result = runner.invoke(
            profiles_app, ["delete", "workflow_test", "--force", "--config-dir", cd]
        )
        assert "Deleted" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
