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

"""Account-scoped credential resolution.

Regression coverage for the credential-identity/transport-dialect decoupling:
a profile's ``account`` selects the credential, so multiple accounts under one
provider (e.g. several anthropic-dialect upstreams: kimi/deepseek/zai via
``/anthropic`` endpoints) resolve their OWN key instead of colliding on a single
provider-scoped ``ANTHROPIC_API_KEY`` / shared keyring slot.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from victor.config.accounts import AccountManager, AuthConfig, ProviderAccount
from victor.config.settings import ProfileConfig, Settings


@pytest.fixture
def temp_config_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def manager(temp_config_dir):
    return AccountManager(config_path=temp_config_dir / "config.yaml")


def _account(name, provider, key, source="config"):
    return ProviderAccount(
        name=name,
        provider=provider,
        model=f"{name}-model",
        auth=AuthConfig(method="api_key", source=source, value=key),
        endpoint=f"https://api.example/{name}/anthropic",
    )


class TestResolvePrecedence:
    """resolve_provider_config prefers the account's own key over ambient env."""

    def test_auth_value_beats_env(self, manager):
        account = _account("kimi-anthropic", "anthropic", "sk-moonshot-key")
        # Simulate ANTHROPIC_API_KEY exported in the environment.
        with patch.object(manager, "_get_api_key_from_env", return_value="sk-env-anthropic"):
            resolved = manager.resolve_provider_config(account)
        assert resolved["api_key"] == "sk-moonshot-key"

    def test_env_used_when_no_auth_value(self, manager):
        # Regression guard: env/keyring path is unchanged for accounts without a
        # per-account key.
        account = ProviderAccount(
            name="plain",
            provider="anthropic",
            model="claude",
            auth=AuthConfig(method="api_key", source="keyring", value=None),
        )
        with patch.object(manager, "_get_api_key_from_env", return_value="sk-env-anthropic"):
            resolved = manager.resolve_provider_config(account)
        assert resolved["api_key"] == "sk-env-anthropic"

    def test_sentinelpass_value_not_treated_as_key(self, manager):
        # sentinelpass stores a lookup DOMAIN in auth.value, not a key.
        account = ProviderAccount(
            name="sp",
            provider="anthropic",
            model="claude",
            auth=AuthConfig(method="api_key", source="sentinelpass", value="corp.example.com"),
        )
        with patch.object(manager, "_get_api_key_from_env", return_value="sk-env-anthropic"):
            resolved = manager.resolve_provider_config(account)
        assert resolved["api_key"] == "sk-env-anthropic"
        assert resolved["api_key"] != "corp.example.com"


class TestGetAccountByName:
    """get_account(name=...) ignores the provider default."""

    def test_named_account_wins_over_default(self, manager):
        claude = _account("claude", "anthropic", "sk-real-anthropic")
        kimi = _account("kimi-anthropic", "anthropic", "sk-moonshot-key")
        manager.save_account(claude)
        manager.save_account(kimi)
        config = manager.load_config()
        config.defaults.account = "claude"
        manager.save_config(config)

        assert manager.get_account(name="kimi-anthropic").auth.value == "sk-moonshot-key"
        # provider-only selection still yields the default (legacy behavior)
        assert manager.get_account(provider="anthropic").name == "claude"


class TestGetProviderSettingsThreading:
    """Settings.get_provider_settings threads account_name end-to-end."""

    def _seed(self, manager):
        manager.save_account(_account("claude", "anthropic", "sk-real-anthropic"))
        manager.save_account(_account("kimi-anthropic", "anthropic", "sk-moonshot-key"))
        config = manager.load_config()
        config.defaults.account = "claude"
        manager.save_config(config)

    def test_account_name_selects_own_key_no_shadow(self, manager):
        self._seed(manager)
        settings = Settings()
        with (
            patch("victor.config.accounts.get_account_manager", return_value=manager),
            patch.object(manager, "_get_api_key_from_env", return_value="sk-env-anthropic"),
        ):
            result = settings.get_provider_settings("anthropic", {}, account_name="kimi-anthropic")
        # The kimi account's own key must win over both the default account and
        # the ambient ANTHROPIC_API_KEY.
        assert result.get("api_key") == "sk-moonshot-key"

    def test_no_account_name_uses_provider_default(self, manager):
        self._seed(manager)
        settings = Settings()
        with patch("victor.config.accounts.get_account_manager", return_value=manager):
            result = settings.get_provider_settings("anthropic", {})
        assert result.get("api_key") == "sk-real-anthropic"

    def test_unknown_account_name_falls_back_to_default(self, manager):
        self._seed(manager)
        settings = Settings()
        with patch("victor.config.accounts.get_account_manager", return_value=manager):
            result = settings.get_provider_settings("anthropic", {}, account_name="does-not-exist")
        assert result.get("api_key") == "sk-real-anthropic"


class TestProfileConfigAccountField:
    """ProfileConfig.account is a first-class typed field (no kwarg leak)."""

    def test_account_is_typed_not_extra(self):
        profile = ProfileConfig(provider="anthropic", model="x", account="kimi-anthropic")
        assert profile.account == "kimi-anthropic"
        assert "account" not in (profile.__pydantic_extra__ or {})

    def test_account_defaults_none(self):
        profile = ProfileConfig(provider="anthropic", model="x")
        assert profile.account is None


class TestAuthAddStorageHygiene:
    """`auth add` writes the shared keyring slot only for the provider default."""

    def _invoke(self, manager, keyring_mock, name):
        from typer.testing import CliRunner

        import victor.ui.commands.auth as auth

        with (
            patch.object(auth, "get_account_manager", return_value=manager),
            patch.object(auth, "_sync_profile_from_account", return_value=False),
            patch("victor.config.api_keys._set_key_in_keyring", keyring_mock),
        ):
            return CliRunner().invoke(
                auth.auth_app,
                [
                    "add",
                    "--provider",
                    "anthropic",
                    "--model",
                    "claude-x",
                    "--name",
                    name,
                    "--source",
                    "keyring",
                    "--api-key",
                    "sk-account-key",
                ],
            )

    def test_named_account_does_not_write_shared_slot(self, manager):
        from unittest.mock import Mock

        keyring_mock = Mock()
        result = self._invoke(manager, keyring_mock, name="kimi-anthropic")
        assert result.exit_code == 0, result.output
        keyring_mock.assert_not_called()
        # ...but the key is still persisted per-account in auth.value.
        assert manager.get_account(name="kimi-anthropic").auth.value == "sk-account-key"

    def test_default_account_writes_shared_slot(self, manager):
        from unittest.mock import Mock

        keyring_mock = Mock()
        result = self._invoke(manager, keyring_mock, name="default")
        assert result.exit_code == 0, result.output
        keyring_mock.assert_called_once_with("anthropic", "sk-account-key")
