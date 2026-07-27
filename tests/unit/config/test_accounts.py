# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Unit tests for the unified accounts configuration system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from victor.config.accounts import (
    AccountManager,
    AuthConfig,
    ProviderAccount,
    VictorConfig,
    ConfigDefaults,
    get_account_manager,
    reset_account_manager,
)


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_account_manager(temp_config_dir):
    """Create an AccountManager with a temporary config directory."""
    manager = AccountManager(config_path=temp_config_dir / "config.yaml")
    return manager


@pytest.fixture
def sample_account():
    """Create a sample provider account."""
    return ProviderAccount(
        name="test-account",
        provider="anthropic",
        model="claude-sonnet-4-5",
        auth=AuthConfig(method="api_key", source="keyring"),
        tags=["test", "sample"],
        temperature=0.7,
        max_tokens=4096,
    )


class TestAuthConfig:
    """Tests for AuthConfig."""

    def test_default_auth_config(self):
        """Test default AuthConfig creation."""
        auth = AuthConfig()
        assert auth.method == "api_key"
        assert auth.source == "keyring"
        assert auth.value is None

    def test_auth_config_is_secure(self):
        """Test AuthConfig.is_secure()."""
        # Keyring is secure
        assert AuthConfig(method="api_key", source="keyring").is_secure()
        # No auth is secure
        assert AuthConfig(method="none", source="keyring").is_secure()
        # File is not secure
        assert not AuthConfig(method="api_key", source="file").is_secure()

    def test_auth_config_requires_api_key(self):
        """Test AuthConfig.requires_api_key()."""
        assert AuthConfig(method="api_key").requires_api_key()
        assert not AuthConfig(method="oauth").requires_api_key()
        assert not AuthConfig(method="none").requires_api_key()

    def test_auth_config_requires_oauth(self):
        """Test AuthConfig.requires_oauth()."""
        assert AuthConfig(method="oauth").requires_oauth()
        assert not AuthConfig(method="api_key").requires_oauth()
        assert not AuthConfig(method="none").requires_oauth()


class TestProviderAccount:
    """Tests for ProviderAccount."""

    def test_provider_account_creation(self):
        """Test creating a ProviderAccount."""
        account = ProviderAccount(
            name="test",
            provider="anthropic",
            model="claude-sonnet-4-5",
            auth=AuthConfig(method="api_key"),
        )
        assert account.name == "test"
        assert account.provider == "anthropic"
        assert account.model == "claude-sonnet-4-5"

    def test_provider_account_is_local(self):
        """Test ProviderAccount.is_local()."""
        # Local providers
        assert ProviderAccount(
            name="ollama",
            provider="ollama",
            model="llama3",
            auth=AuthConfig(method="none"),
        ).is_local()
        assert ProviderAccount(
            name="lmstudio",
            provider="lmstudio",
            model="local",
            auth=AuthConfig(method="none"),
        ).is_local()
        assert ProviderAccount(
            name="vllm", provider="vllm", model="local", auth=AuthConfig(method="none")
        ).is_local()

        # Cloud provider
        assert not ProviderAccount(
            name="anthropic",
            provider="anthropic",
            model="claude",
            auth=AuthConfig(method="api_key"),
        ).is_local()

    def test_provider_account_oauth_enabled(self):
        """Test ProviderAccount.is_oauth_enabled()."""
        assert ProviderAccount(
            name="openai",
            provider="openai",
            model="gpt-4",
            auth=AuthConfig(method="oauth"),
        ).is_oauth_enabled()
        assert not ProviderAccount(
            name="anthropic",
            provider="anthropic",
            model="claude",
            auth=AuthConfig(method="api_key"),
        ).is_oauth_enabled()

    def test_provider_account_get_endpoint_variant(self):
        """Test ProviderAccount.get_endpoint_variant()."""
        account = ProviderAccount(
            name="glm",
            provider="zai",
            model="glm-4.6:coding",
            auth=AuthConfig(method="api_key"),
        )
        assert account.get_endpoint_variant() == "coding"

        # No variant
        account2 = ProviderAccount(
            name="anthropic",
            provider="anthropic",
            model="claude-sonnet-4-5",
            auth=AuthConfig(method="api_key"),
        )
        assert account2.get_endpoint_variant() is None

    def test_provider_account_get_base_model(self):
        """Test ProviderAccount.get_base_model()."""
        account = ProviderAccount(
            name="glm",
            provider="zai",
            model="glm-4.6:coding",
            auth=AuthConfig(method="api_key"),
        )
        assert account.get_base_model() == "glm-4.6"

        # No suffix
        account2 = ProviderAccount(
            name="anthropic",
            provider="anthropic",
            model="claude-sonnet-4-5",
            auth=AuthConfig(method="api_key"),
        )
        assert account2.get_base_model() == "claude-sonnet-4-5"

    def test_provider_account_to_dict(self):
        """Test ProviderAccount.to_dict()."""
        account = ProviderAccount(
            name="test",
            provider="anthropic",
            model="claude-sonnet-4-5",
            auth=AuthConfig(method="api_key", source="keyring"),
            tags=["test"],
            temperature=0.7,
        )
        result = account.to_dict()

        assert result["provider"] == "anthropic"
        assert result["model"] == "claude-sonnet-4-5"
        assert result["name"] == "test"
        assert result["tags"] == ["test"]
        assert result["temperature"] == 0.7
        # Default auth (api_key + keyring) is not serialized
        assert "auth" not in result

    def test_provider_account_from_dict(self):
        """Test ProviderAccount.from_dict()."""
        data = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "auth": {"method": "api_key", "source": "keyring"},
            "tags": ["test"],
            "temperature": 0.7,
        }
        account = ProviderAccount.from_dict("test", data)

        assert account.provider == "anthropic"
        assert account.model == "claude-sonnet-4-5"
        assert account.auth.method == "api_key"
        assert account.auth.source == "keyring"
        assert account.tags == ["test"]
        assert account.temperature == 0.7

    def test_provider_account_from_dict_simple_auth(self):
        """Test ProviderAccount.from_dict() with simple auth string."""
        data = {
            "provider": "ollama",
            "model": "llama3",
            "auth": "none",
        }
        account = ProviderAccount.from_dict("ollama-default", data)

        assert account.provider == "ollama"
        assert account.model == "llama3"
        assert account.auth.method == "none"


class TestVictorConfig:
    """Tests for VictorConfig."""

    def test_victor_config_empty(self):
        """Test creating empty VictorConfig."""
        config = VictorConfig()
        assert config.accounts == {}
        assert config.defaults.account == "default"

    def test_victor_config_add_account(self):
        """Test VictorConfig.add_account()."""
        config = VictorConfig()
        account = ProviderAccount(
            name="test",
            provider="anthropic",
            model="claude",
            auth=AuthConfig(method="api_key"),
        )

        config.add_account(account)
        assert "test" in config.accounts
        assert config.accounts["test"] == account

    def test_victor_config_remove_account(self):
        """Test VictorConfig.remove_account()."""
        config = VictorConfig()
        account = ProviderAccount(
            name="test",
            provider="anthropic",
            model="claude",
            auth=AuthConfig(method="api_key"),
        )
        config.add_account(account)

        assert config.remove_account("test") is True
        assert "test" not in config.accounts

        # Remove non-existent
        assert config.remove_account("nonexistent") is False

    def test_victor_config_list_accounts(self):
        """Test VictorConfig.list_accounts()."""
        config = VictorConfig()
        account1 = ProviderAccount(
            name="test1",
            provider="anthropic",
            model="claude",
            auth=AuthConfig(method="api_key"),
        )
        account2 = ProviderAccount(
            name="test2",
            provider="openai",
            model="gpt-4",
            auth=AuthConfig(method="api_key"),
        )

        config.add_account(account1)
        config.add_account(account2)

        accounts = config.list_accounts()
        assert len(accounts) == 2
        assert account1 in accounts
        assert account2 in accounts

    def test_victor_config_to_dict(self):
        """Test VictorConfig.to_dict()."""
        config = VictorConfig()
        account = ProviderAccount(
            name="test",
            provider="anthropic",
            model="claude",
            auth=AuthConfig(method="api_key"),
        )
        config.add_account(account)
        config.defaults.account = "test"

        result = config.to_dict()
        assert "accounts" in result
        assert "test" in result["accounts"]
        assert result["defaults"]["account"] == "test"

    def test_victor_config_from_dict(self):
        """Test VictorConfig.from_dict()."""
        data = {
            "accounts": {
                "test": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "auth": {"method": "api_key", "source": "keyring"},
                }
            },
            "defaults": {"account": "test"},
        }
        config = VictorConfig.from_dict(data)

        assert "test" in config.accounts
        assert config.accounts["test"].provider == "anthropic"
        assert config.defaults.account == "test"


class TestAccountManager:
    """Tests for AccountManager."""

    def test_account_manager_init(self, mock_account_manager):
        """Test AccountManager initialization."""
        assert mock_account_manager.config_path.name == "config.yaml"

    def test_account_manager_load_config_empty(self, mock_account_manager):
        """Test loading config when file doesn't exist."""
        config = mock_account_manager.load_config()
        assert isinstance(config, VictorConfig)
        assert config.accounts == {}

    def test_account_manager_save_and_load(self, mock_account_manager, sample_account):
        """Test saving and loading config."""
        # Save account
        mock_account_manager.save_account(sample_account)

        # Create new manager instance and load
        manager2 = AccountManager(config_path=mock_account_manager.config_path)
        loaded_account = manager2.get_account("test-account")

        assert loaded_account is not None
        assert loaded_account.provider == sample_account.provider
        assert loaded_account.model == sample_account.model
        assert loaded_account.tags == sample_account.tags

    def test_account_manager_get_account_by_name(self, mock_account_manager, sample_account):
        """Test getting account by name."""
        mock_account_manager.save_account(sample_account)

        account = mock_account_manager.get_account(name="test-account")
        assert account is not None
        assert account.name == "test-account"

    def test_account_manager_get_account_by_provider_model(
        self, mock_account_manager, sample_account
    ):
        """Test getting account by provider and model."""
        mock_account_manager.save_account(sample_account)

        account = mock_account_manager.get_account(provider="anthropic", model="claude-sonnet-4-5")
        assert account is not None
        assert account.provider == "anthropic"

    def test_account_manager_remove_account(self, mock_account_manager, sample_account):
        """Test removing an account."""
        mock_account_manager.save_account(sample_account)

        assert mock_account_manager.remove_account("test-account") is True
        assert mock_account_manager.get_account("test-account") is None

    def test_account_manager_list_accounts(self, mock_account_manager):
        """Test listing all accounts."""
        account1 = ProviderAccount(
            name="test1",
            provider="anthropic",
            model="claude",
            auth=AuthConfig(method="api_key"),
        )
        account2 = ProviderAccount(
            name="test2",
            provider="openai",
            model="gpt-4",
            auth=AuthConfig(method="api_key"),
        )

        mock_account_manager.save_account(account1)
        mock_account_manager.save_account(account2)

        accounts = mock_account_manager.list_accounts()
        assert len(accounts) == 2

    def test_account_manager_list_providers(self, mock_account_manager):
        """Test listing unique providers."""
        account1 = ProviderAccount(
            name="test1",
            provider="anthropic",
            model="claude",
            auth=AuthConfig(method="api_key"),
        )
        account2 = ProviderAccount(
            name="test2",
            provider="anthropic",
            model="claude-opus",
            auth=AuthConfig(method="api_key"),
        )
        account3 = ProviderAccount(
            name="test3",
            provider="openai",
            model="gpt-4",
            auth=AuthConfig(method="api_key"),
        )

        mock_account_manager.save_account(account1)
        mock_account_manager.save_account(account2)
        mock_account_manager.save_account(account3)

        providers = mock_account_manager.list_providers()
        assert len(providers) == 2
        assert "anthropic" in providers
        assert "openai" in providers

    def test_account_manager_resolve_provider_config(self, mock_account_manager, sample_account):
        """Test resolving provider configuration."""
        mock_account_manager.save_account(sample_account)

        # Mock API key retrieval
        with patch.object(
            mock_account_manager,
            "_get_api_key_from_keyring",
            return_value="sk-test-key",
        ):
            config = mock_account_manager.resolve_provider_config(sample_account)

            assert config["provider"] == "anthropic"
            assert config["model"] == "claude-sonnet-4-5"
            assert config["api_key"] == "sk-test-key"
            assert config["base_url"] == "https://api.anthropic.com"

    def test_account_manager_resolve_with_overrides(self, mock_account_manager, sample_account):
        """Test resolving with parameter overrides."""
        mock_account_manager.save_account(sample_account)

        config = mock_account_manager.resolve_provider_config(
            account=sample_account, temperature=0.5, max_tokens=2048
        )

        assert config["temperature"] == 0.5
        assert config["max_tokens"] == 2048


class TestGlobalFunctions:
    """Tests for global module functions."""

    def test_get_account_manager_singleton(self):
        """Test get_account_manager returns singleton."""
        manager1 = get_account_manager()
        manager2 = get_account_manager()
        assert manager1 is manager2

    def test_reset_account_manager(self):
        """Test reset_account_manager."""
        manager1 = get_account_manager()
        reset_account_manager()
        manager2 = get_account_manager()
        assert manager1 is not manager2


class TestModelSuffixParsing:
    """Tests for model suffix parsing in ZAI provider."""

    def test_zai_model_suffix_parsing(self):
        """Test parsing ZAI model suffix for endpoint selection."""
        # Test coding suffix
        account = ProviderAccount(
            name="glm-coding",
            provider="zai",
            model="glm-4.6:coding",
            auth=AuthConfig(method="api_key"),
        )
        assert account.get_endpoint_variant() == "coding"
        assert account.get_base_model() == "glm-4.6"

        # Test standard suffix
        account2 = ProviderAccount(
            name="glm-standard",
            provider="zai",
            model="glm-4.6:standard",
            auth=AuthConfig(method="api_key"),
        )
        assert account2.get_endpoint_variant() == "standard"

        # Test china suffix
        account3 = ProviderAccount(
            name="glm-china",
            provider="zai",
            model="glm-4.6:china",
            auth=AuthConfig(method="api_key"),
        )
        assert account3.get_endpoint_variant() == "china"

        # Test anthropic suffix
        account4 = ProviderAccount(
            name="glm-anthropic",
            provider="zai",
            model="glm-4.6:anthropic",
            auth=AuthConfig(method="api_key"),
        )
        assert account4.get_endpoint_variant() == "anthropic"

        # Test no suffix
        account5 = ProviderAccount(
            name="glm-no-suffix",
            provider="zai",
            model="glm-4.6",
            auth=AuthConfig(method="api_key"),
        )
        assert account5.get_endpoint_variant() is None
        assert account5.get_base_model() == "glm-4.6"


class TestAccountScopedCredential:
    """An explicitly-configured per-account key (auth.value) is authoritative for
    that account, winning over the shared provider-scoped env var / keyring.

    Regression: multiple accounts under one provider (anthropic-dialect upstreams
    kimi/deepseek/zai via /anthropic endpoints) collided on the single provider
    credential slot — the shared ANTHROPIC_API_KEY was sent to moonshot's endpoint,
    and `victor auth add --source keyring` clobbered the real anthropic key.
    """

    def _account(self):
        return ProviderAccount(
            name="kimi-k3-anthropic",
            provider="anthropic",
            model="kimi-k3",
            auth=AuthConfig(method="api_key", source="keyring", value="sk-moonshot-account-key"),
            endpoint="https://api.moonshot.ai/anthropic",
        )

    def test_account_value_beats_provider_env(self, temp_config_dir, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-shared-real-key")
        mgr = AccountManager(config_path=temp_config_dir / "config.yaml")
        cfg = mgr.resolve_provider_config(account=self._account())
        assert cfg["api_key"] == "sk-moonshot-account-key"

    def test_account_without_value_falls_back_to_provider_env(self, temp_config_dir, monkeypatch):
        # A normal account (no per-account value) keeps the old provider-scoped path.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-shared-real-key")
        mgr = AccountManager(config_path=temp_config_dir / "config.yaml")
        acct = ProviderAccount(
            name="claude",
            provider="anthropic",
            model="claude-sonnet-4-5",
            auth=AuthConfig(method="api_key", source="keyring", value=None),
        )
        cfg = mgr.resolve_provider_config(account=acct)
        assert cfg["api_key"] == "sk-ant-shared-real-key"

    def test_explicit_kwarg_still_wins_over_account_value(self, temp_config_dir, monkeypatch):
        mgr = AccountManager(config_path=temp_config_dir / "config.yaml")
        cfg = mgr.resolve_provider_config(account=self._account(), api_key="sk-explicit-override")
        assert cfg["api_key"] == "sk-explicit-override"
