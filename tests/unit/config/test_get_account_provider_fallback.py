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

"""Provider-constrained account fallback.

Regression coverage for ``AccountManager.get_account``'s provider invariant:
a provider-scoped lookup must never return an account of a *different* provider.

Previously, ``get_account(provider="anthropic")`` (no model) skipped the
provider+model step and returned the GLOBAL default account
(``config.defaults.account``), ignoring the provider filter entirely. When the
global default belonged to another provider (e.g. the ollama ``default``
account), every ``account: null`` profile on a keyring/OAuth-only provider
silently resolved the wrong credential — or no credential at all — because the
runtime then had no key for the requested provider. This is the latent defect
uncovered while wiring the anthropic accounts.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from victor.config.accounts import AccountManager, AuthConfig, ProviderAccount
from victor.config.settings import Settings


@pytest.fixture
def temp_config_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def manager(temp_config_dir):
    return AccountManager(config_path=temp_config_dir / "config.yaml")


def _acct(name, provider, model=None, key="sk-x"):
    return ProviderAccount(
        name=name,
        provider=provider,
        model=model or f"{name}-model",
        auth=AuthConfig(method="api_key", source="config", value=key),
    )


def _set_default(manager, name):
    cfg = manager.load_config()
    cfg.defaults.account = name
    manager.save_config(cfg)


class TestProviderConstrainedFallback:
    """get_account(provider=...) never crosses providers."""

    def test_provider_only_returns_same_provider_not_global_default(self, manager):
        # Global default is an OLLAMA account; an anthropic request must NOT
        # bleed across to it.
        manager.save_account(_acct("default", "ollama", key="sk-ollama-WRONG"))
        manager.save_account(_acct("claude", "anthropic", key="sk-anthropic-REAL"))
        _set_default(manager, "default")

        result = manager.get_account(provider="anthropic")
        assert result is not None
        assert result.provider == "anthropic"
        assert result.name == "claude"

    def test_provider_only_returns_first_when_multiple(self, manager):
        # No model to disambiguate → first account of that provider (insertion
        # order). Both share the provider, so the credential is valid either way.
        manager.save_account(_acct("claude", "anthropic", key="sk-1"))
        manager.save_account(_acct("haiku", "anthropic", key="sk-2"))

        result = manager.get_account(provider="anthropic")
        assert result.name == "claude"

    def test_provider_plus_model_pins_exact_account(self, manager):
        manager.save_account(_acct("claude", "anthropic", model="claude-sonnet", key="sk-sonnet"))
        manager.save_account(_acct("haiku", "anthropic", model="claude-haiku", key="sk-haiku"))

        result = manager.get_account(provider="anthropic", model="claude-haiku")
        assert result.name == "haiku"

    def test_provider_unconfigured_returns_none_not_cross_provider(self, manager):
        # Only ollama accounts exist; anthropic is requested but has no account.
        # Must return None (→ caller degrades to provider-scoped env/keyring),
        # NOT the ollama global default.
        manager.save_account(_acct("default", "ollama", key="sk-ollama"))
        _set_default(manager, "default")

        assert manager.get_account(provider="anthropic") is None

    def test_no_provider_returns_global_default(self, manager):
        # Regression guard: a no-provider lookup (e.g. `auth test` with no
        # flags) still returns the configured global default account.
        manager.save_account(_acct("default", "ollama", key="sk-ollama"))
        manager.save_account(_acct("claude", "anthropic", key="sk-anthropic"))
        _set_default(manager, "default")

        result = manager.get_account()
        assert result is not None
        assert result.name == "default"


class TestRuntimeCrossProviderResolution:
    """Settings.get_provider_settings no longer harvests a cross-provider key."""

    def test_unbound_profile_resolves_same_provider_key(self, manager):
        # Mirrors the original failure: global default is ollama, yet an
        # ``account: null`` anthropic profile must still resolve the anthropic
        # account's own key — not the ollama default's (non-existent) key.
        manager.save_account(_acct("default", "ollama", key="sk-ollama-WRONG"))
        manager.save_account(_acct("claude", "anthropic", key="sk-anthropic-REAL"))
        _set_default(manager, "default")

        settings = Settings()
        with (
            patch("victor.config.accounts.get_account_manager", return_value=manager),
            # Determinism: ensure the strategy layer's env/keyring rescue does
            # not inject a stray anthropic key in the test environment.
            patch("victor.config.api_keys.get_api_key", return_value=None),
        ):
            result = settings.get_provider_settings("anthropic", {})

        assert result.get("api_key") == "sk-anthropic-REAL"
