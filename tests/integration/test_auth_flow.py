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

"""Integration tests for authentication flow.

These tests verify the complete authentication workflow including:
- Interactive setup wizard
- API key configuration
- Connection testing
- Account management
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import pytest

from victor.config.accounts import (
    AccountManager,
    AuthConfig,
    ProviderAccount,
    get_account_manager,
    reset_account_manager,
)
from victor.config.connection_validation import (
    ConnectionValidator,
    ConnectionValidationResult,
    ConnectionTestResult,
    ValidationStatus,
)
from victor.config.migration import (
    ConfigMigrator,
    MigrationResult,
)


@pytest.fixture
def temp_victor_dir():
    """Create a temporary .victor directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        victor_dir = Path(tmpdir) / ".victor"
        victor_dir.mkdir()
        yield victor_dir


@pytest.fixture
def clean_account_manager(temp_victor_dir):
    """Create an AccountManager with clean state."""
    reset_account_manager()
    manager = AccountManager(config_path=temp_victor_dir / "config.yaml")
    return manager


class TestAccountManagementIntegration:
    """Integration tests for account management workflows."""

    def test_complete_account_lifecycle(self, clean_account_manager):
        """Test complete account lifecycle: create, read, update, delete."""
        # Create account
        account = ProviderAccount(
            name="test-account",
            provider="anthropic",
            model="claude-sonnet-4-5",
            auth=AuthConfig(method="api_key", source="keyring"),
            tags=["test", "integration"],
        )

        # Save
        clean_account_manager.save_account(account)

        # Read
        loaded = clean_account_manager.get_account("test-account")
        assert loaded is not None
        assert loaded.provider == "anthropic"
        assert loaded.model == "claude-sonnet-4-5"
        assert loaded.tags == ["test", "integration"]

        # Update
        loaded.tags.append("updated")
        clean_account_manager.save_account(loaded)

        # Verify update
        reloaded = clean_account_manager.get_account("test-account")
        assert "updated" in reloaded.tags

        # Delete
        assert clean_account_manager.remove_account("test-account") is True
        assert clean_account_manager.get_account("test-account") is None

    def test_multiple_accounts_management(self, clean_account_manager):
        """Test managing multiple accounts."""
        accounts = [
            ProviderAccount(
                name="anthropic-default",
                provider="anthropic",
                model="claude-sonnet-4-5",
                auth=AuthConfig(method="api_key"),
                tags=["chat"],
            ),
            ProviderAccount(
                name="openai-coding",
                provider="openai",
                model="gpt-4",
                auth=AuthConfig(method="api_key"),
                tags=["coding"],
            ),
            ProviderAccount(
                name="glm-coding",
                provider="zai",
                model="glm-4.6:coding",
                auth=AuthConfig(method="api_key"),
                tags=["coding", "premium"],
            ),
        ]

        # Save all accounts
        for account in accounts:
            clean_account_manager.save_account(account)

        # List all accounts
        all_accounts = clean_account_manager.list_accounts()
        assert len(all_accounts) == 3

        # List unique providers
        providers = clean_account_manager.list_providers()
        assert len(providers) == 3
        assert "anthropic" in providers
        assert "openai" in providers
        assert "zai" in providers

        # Find by provider
        anthropic_accounts = [a for a in all_accounts if a.provider == "anthropic"]
        assert len(anthropic_accounts) == 1

    def test_default_account_resolution(self, clean_account_manager):
        """Test default account resolution."""
        # Create accounts
        default = ProviderAccount(
            name="default",
            provider="anthropic",
            model="claude-sonnet-4-5",
            auth=AuthConfig(method="api_key"),
        )
        custom = ProviderAccount(
            name="custom",
            provider="openai",
            model="gpt-4",
            auth=AuthConfig(method="api_key"),
        )

        clean_account_manager.save_account(default)
        clean_account_manager.save_account(custom)

        # Set default
        config = clean_account_manager.load_config()
        config.defaults.account = "default"
        clean_account_manager.save_config(config)

        # Get default account
        account = clean_account_manager.get_account()
        assert account is not None
        assert account.name == "default"

        # Get specific account
        account2 = clean_account_manager.get_account(name="custom")
        assert account2 is not None
        assert account2.name == "custom"

    def test_provider_model_resolution(self, clean_account_manager):
        """Test account resolution by provider and model."""
        account1 = ProviderAccount(
            name="claude-default",
            provider="anthropic",
            model="claude-sonnet-4-5",
            auth=AuthConfig(method="api_key"),
        )
        account2 = ProviderAccount(
            name="claude-opus",
            provider="anthropic",
            model="claude-opus-4-6",
            auth=AuthConfig(method="api_key"),
        )

        clean_account_manager.save_account(account1)
        clean_account_manager.save_account(account2)

        # Resolve by provider + model
        account = clean_account_manager.get_account(provider="anthropic", model="claude-sonnet-4-5")
        assert account is not None
        assert account.name == "claude-default"


class TestConnectionValidationIntegration:
    """Integration tests for connection validation."""

    @pytest.mark.asyncio
    async def test_local_provider_validation(self):
        """Test validation for local providers."""
        account = ProviderAccount(
            name="ollama-test",
            provider="ollama",
            model="llama3",
            auth=AuthConfig(method="none"),
        )

        validator = ConnectionValidator()

        # Mock local provider detection
        with patch.object(
            validator,
            "_test_local_provider",
            return_value=ConnectionTestResult(
                success=True,
                account_name="ollama-test",
                provider="ollama",
                model="llama3",
            ),
        ):
            result = await validator.test_account(account)

            # Should succeed (Ollama is running)
            assert result.success
            assert result.provider == "ollama"

    @pytest.mark.asyncio
    async def test_api_key_provider_validation(self):
        """Test validation for API key providers."""
        account = ProviderAccount(
            name="anthropic-test",
            provider="anthropic",
            model="claude-sonnet-4-5",
            auth=AuthConfig(method="api_key", source="keyring"),
        )

        validator = ConnectionValidator()

        # Mock auth validation
        with patch.object(
            validator,
            "_validate_auth",
            return_value=ConnectionValidationResult(
                status=ValidationStatus.SUCCESS, message="API key is valid"
            ),
        ):
            # Mock endpoint test
            with patch.object(
                validator,
                "_test_endpoint",
                return_value=ConnectionValidationResult(
                    status=ValidationStatus.SUCCESS, message="Endpoint reachable"
                ),
            ):
                result = await validator.test_account(account)

                # Should have successful auth validation
                auth_validations = [
                    v
                    for v in result.validations
                    if "auth" in str(v.message).lower() or "api key" in str(v.message).lower()
                ]
                assert len(auth_validations) > 0

    @pytest.mark.asyncio
    async def test_oauth_provider_validation(self):
        """Test validation for OAuth providers."""
        account = ProviderAccount(
            name="openai-oauth",
            provider="openai",
            model="gpt-4",
            auth=AuthConfig(method="oauth"),
        )

        validator = ConnectionValidator()

        # Mock OAuth client_id retrieval
        with patch.object(validator, "_get_oauth_client_id", return_value="test-client-id"):
            # Mock auth validation which internally calls _get_oauth_client_id
            with patch.object(
                validator,
                "_validate_auth",
                return_value=ConnectionValidationResult(
                    status=ValidationStatus.SUCCESS,
                    message="OAuth client_id configured",
                ),
            ):
                result = await validator.test_account(account)

                # Should have OAuth validation
                oauth_validations = [
                    v
                    for v in result.validations
                    if "oauth" in str(v.message).lower() or "client_id" in str(v.message).lower()
                ]
                assert len(oauth_validations) > 0

    def test_model_suffix_validation(self):
        """Test model suffix parsing for GLM."""
        account = ProviderAccount(
            name="glm-coding",
            provider="zai",
            model="glm-4.6:coding",
            auth=AuthConfig(method="api_key"),
        )

        # Test model parsing (this is done at the ProviderAccount level)
        assert account.get_base_model() == "glm-4.6"
        assert account.get_endpoint_variant() == "coding"

        # Note: ConnectionValidator doesn't validate model suffixes
        # The model suffix is handled by the provider adapter
        # This test verifies that the account model parsing works correctly


class TestMigrationIntegration:
    """Integration tests for configuration migration."""

    def test_full_migration_workflow(self, temp_victor_dir):
        """Test complete migration from old to new format."""
        # Create old config files
        old_profiles = temp_victor_dir / "profiles.yaml"
        old_keys = temp_victor_dir / "api_keys.yaml"

        import yaml

        # Write old profiles
        profiles_data = {
            "default_profile": "default",
            "profiles": {
                "default": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "temperature": 0.7,
                }
            },
        }
        with open(old_profiles, "w") as f:
            yaml.dump(profiles_data, f)

        # Write old API keys
        keys_data = {"api_keys": {"anthropic": "sk-ant-test"}}
        with open(old_keys, "w") as f:
            yaml.dump(keys_data, f)
        old_keys.chmod(0o600)

        # Run migration
        migrator = ConfigMigrator(victor_dir=temp_victor_dir)
        result = migrator.migrate(prompt=False)

        # Verify success
        assert result.success
        assert result.migrated_accounts > 0

        # Verify new config exists
        new_config = temp_victor_dir / "config.yaml"
        assert new_config.exists()

        # Load and verify new config
        with open(new_config, "r") as f:
            new_data = yaml.safe_load(f)

        assert "accounts" in new_data
        assert "defaults" in new_data
        assert "default" in new_data["accounts"]

    def test_migration_with_local_provider(self, temp_victor_dir):
        """Test migration with local provider (no API key)."""
        old_profiles = temp_victor_dir / "profiles.yaml"

        import yaml

        profiles_data = {
            "default_profile": "local",
            "profiles": {
                "local": {
                    "provider": "ollama",
                    "model": "llama3",
                    "temperature": 0.8,
                }
            },
        }
        with open(old_profiles, "w") as f:
            yaml.dump(profiles_data, f)

        # Run migration
        migrator = ConfigMigrator(victor_dir=temp_victor_dir)
        result = migrator.migrate(prompt=False)

        assert result.success

        # Verify migrated account has no auth
        new_config = temp_victor_dir / "config.yaml"
        with open(new_config, "r") as f:
            new_data = yaml.safe_load(f)

        local_account = new_data["accounts"]["local"]
        assert local_account["auth"]["method"] == "none"

    def test_migration_rollback(self, temp_victor_dir):
        """Test migration rollback functionality."""
        # Create old config
        old_profiles = temp_victor_dir / "profiles.yaml"

        import yaml

        profiles_data = {
            "default_profile": "default",
            "profiles": {
                "default": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                }
            },
        }
        with open(old_profiles, "w") as f:
            yaml.dump(profiles_data, f)

        # Run migration
        migrator = ConfigMigrator(victor_dir=temp_victor_dir)
        result = migrator.migrate(prompt=False)

        assert result.success

        # Verify new config exists
        new_config = temp_victor_dir / "config.yaml"
        assert new_config.exists()

        # Rollback
        success = migrator.rollback()

        assert success
        assert not new_config.exists()
        # Old files should be restored
        assert old_profiles.exists()


class TestOAuthFlowIntegration:
    """Integration tests for OAuth authentication flow."""

    def test_oauth_account_creation(self, clean_account_manager):
        """Test creating OAuth-enabled account."""
        account = ProviderAccount(
            name="openai-oauth",
            provider="openai",
            model="gpt-4",
            auth=AuthConfig(method="oauth"),
            tags=["oauth"],
        )

        clean_account_manager.save_account(account)

        # Load and verify
        loaded = clean_account_manager.get_account("openai-oauth")
        assert loaded is not None
        assert loaded.is_oauth_enabled()
        assert loaded.auth.method == "oauth"

    def test_oauth_provider_resolution(self, clean_account_manager):
        """Test provider config resolution for OAuth."""
        account = ProviderAccount(
            name="qwen-oauth",
            provider="qwen",
            model="qwen-max",
            auth=AuthConfig(method="oauth"),
        )

        clean_account_manager.save_account(account)

        # Mock OAuth client_id retrieval
        with patch.object(
            clean_account_manager, "_get_oauth_client_id", return_value="test-client-id"
        ):
            config = clean_account_manager.resolve_provider_config(account)

            assert config["auth_mode"] == "oauth"
            assert "oauth_client_id" in config

    def test_oauth_provider_resolution_preserves_external_source(self, clean_account_manager):
        """OAuth accounts should pass external token source to provider adapters."""
        account = ProviderAccount(
            name="openai-codex",
            provider="openai",
            model="gpt-5-nano",
            auth=AuthConfig(method="oauth", source="codex"),
        )
        clean_account_manager.save_account(account)

        with patch.object(
            clean_account_manager, "_get_oauth_client_id", return_value="test-client-id"
        ):
            config = clean_account_manager.resolve_provider_config(account=account)

        assert config["auth_mode"] == "oauth"
        assert config["oauth_source"] == "codex"


class TestAccountResolutionIntegration:
    """Integration tests for account resolution logic."""

    def test_cli_flags_override_config(self, clean_account_manager):
        """Test that CLI flags override config settings."""
        account = ProviderAccount(
            name="default",
            provider="anthropic",
            model="claude-sonnet-4-5",
            auth=AuthConfig(method="api_key"),
        )

        clean_account_manager.save_account(account)

        # Resolve with overrides
        config = clean_account_manager.resolve_provider_config(
            account=account,
            temperature=0.5,
            max_tokens=2048,
        )

        assert config["temperature"] == 0.5
        assert config["max_tokens"] == 2048

    def test_env_variable_resolution(self, clean_account_manager):
        """Test API key resolution from environment variables."""
        account = ProviderAccount(
            name="anthropic-env",
            provider="anthropic",
            model="claude-sonnet-4-5",
            auth=AuthConfig(method="api_key", source="env"),
        )

        clean_account_manager.save_account(account)

        # Mock environment variable
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-env-key"}):
            config = clean_account_manager.resolve_provider_config(account)

            assert config["api_key"] == "sk-ant-env-key"

    def test_keyring_resolution(self, clean_account_manager):
        """Test API key resolution from keyring."""
        account = ProviderAccount(
            name="anthropic-keyring",
            provider="anthropic",
            model="claude-sonnet-4-5",
            auth=AuthConfig(method="api_key", source="keyring"),
        )

        clean_account_manager.save_account(account)

        # Mock keyring retrieval
        with patch.object(
            clean_account_manager,
            "_get_api_key_from_keyring",
            return_value="sk-ant-keyring-key",
        ):
            config = clean_account_manager.resolve_provider_config(account)

            assert config["api_key"] == "sk-ant-keyring-key"


class TestModelSuffixIntegration:
    """Integration tests for model suffix functionality."""

    def test_zai_coding_plan_suffix(self, clean_account_manager):
        """Test ZAI coding plan endpoint selection via model suffix."""
        account = ProviderAccount(
            name="glm-coding",
            provider="zai",
            model="glm-4.6:coding",
            auth=AuthConfig(method="api_key"),
        )

        clean_account_manager.save_account(account)

        # Resolve provider config
        with patch.object(
            clean_account_manager, "_get_api_key_from_keyring", return_value="test-key"
        ):
            config = clean_account_manager.resolve_provider_config(account)

            # Verify endpoint is set to coding plan
            # The ZAIProvider will parse the model suffix internally
            assert config["model"] == "glm-4.6:coding"

    def test_zai_standard_endpoint(self, clean_account_manager):
        """Test ZAI standard endpoint (no suffix)."""
        account = ProviderAccount(
            name="glm-standard",
            provider="zai",
            model="glm-4.6",
            auth=AuthConfig(method="api_key"),
        )

        clean_account_manager.save_account(account)

        # Resolve provider config
        with patch.object(
            clean_account_manager, "_get_api_key_from_keyring", return_value="test-key"
        ):
            config = clean_account_manager.resolve_provider_config(account)

            # Verify model has no suffix
            assert config["model"] == "glm-4.6"
            assert config["base_url"] == "https://api.z.ai/api/paas/v4/"

    def test_zai_china_endpoint(self, clean_account_manager):
        """Test ZAI China endpoint via suffix."""
        account = ProviderAccount(
            name="glm-china",
            provider="zai",
            model="glm-4.6:china",
            auth=AuthConfig(method="api_key"),
        )

        clean_account_manager.save_account(account)

        # Verify parsing
        assert account.get_base_model() == "glm-4.6"
        assert account.get_endpoint_variant() == "china"


@pytest.mark.integration
class TestRealConnectionValidation:
    """Integration tests that require real network connections (marked as integration)."""

    @pytest.mark.skip(reason="Requires real Ollama installation")
    def test_real_ollama_connection(self):
        """Test real Ollama connection (requires Ollama to be running)."""
        account = ProviderAccount(
            name="ollama-real",
            provider="ollama",
            model="llama3",
            auth=AuthConfig(method="none"),
        )

        validator = ConnectionValidator()
        result = validator.test_account_sync(account)

        if result.success:
            assert result.provider == "ollama"
        else:
            pytest.skip("Ollama not available")

    @pytest.mark.skip(reason="Requires real API key")
    def test_real_provider_connection(self):
        """Test real provider connection (requires valid API key)."""
        pytest.skip("Skipping - requires real API key")
