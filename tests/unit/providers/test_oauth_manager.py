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

"""TDD tests for OAuthTokenManager — provider OAuth authentication."""

import os
import json
import stat
import pytest
import yaml
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from victor.providers.oauth_manager import (
    OAuthProviderConfig,
    OAuthTokenManager,
    OAUTH_PROVIDERS,
)
from victor.workflows.services.credentials import SSOConfig, SSOProvider, SSOTokens

# ---------------------------------------------------------------------------
# OAuthProviderConfig registry tests
# ---------------------------------------------------------------------------


class TestOAuthProviderRegistry:
    """Test the static registry of OAuth-capable LLM providers."""

    def test_openai_registered(self):
        assert "openai" in OAUTH_PROVIDERS

    def test_openai_config_values(self):
        cfg = OAUTH_PROVIDERS["openai"]
        assert cfg.provider_name == "openai"
        assert cfg.sso_provider == SSOProvider.OPENAI_CODEX
        assert cfg.issuer_url == "https://auth.openai.com"
        assert cfg.get_client_id() == "app_EMoamEEZ73f0CkXaXp7hrann"
        assert "offline_access" in cfg.scopes
        assert "api.connectors.read" in cfg.scopes
        assert "api.connectors.invoke" in cfg.scopes
        assert cfg.token_endpoint == "/oauth/token"
        assert cfg.redirect_port == 1455

    def test_qwen_registered(self):
        assert "qwen" in OAUTH_PROVIDERS

    def test_qwen_config_values(self):
        cfg = OAUTH_PROVIDERS["qwen"]
        assert cfg.provider_name == "qwen"
        assert cfg.sso_provider == SSOProvider.QWEN
        assert cfg.issuer_url == "https://chat.qwen.ai"

    def test_to_sso_config(self):
        """OAuthProviderConfig must convert to SSOConfig for SSOAuthenticator."""
        cfg = OAUTH_PROVIDERS["openai"]
        sso = cfg.to_sso_config()
        assert isinstance(sso, SSOConfig)
        assert sso.provider == SSOProvider.OPENAI_CODEX
        assert sso.issuer_url == "https://auth.openai.com"
        assert sso.client_id == "app_EMoamEEZ73f0CkXaXp7hrann"
        assert sso.use_pkce is True

    def test_unsupported_provider_raises(self):
        with pytest.raises(KeyError):
            _ = OAUTH_PROVIDERS["anthropic"]


# ---------------------------------------------------------------------------
# SSOProvider / SSOConfig extension tests
# ---------------------------------------------------------------------------


class TestSSOProviderExtensions:
    """Test that new SSO provider enum values and factory methods exist."""

    def test_openai_codex_enum_exists(self):
        assert SSOProvider.OPENAI_CODEX.value == "openai_codex"

    def test_qwen_enum_exists(self):
        assert SSOProvider.QWEN.value == "qwen"

    def test_for_openai_codex_factory(self):
        cfg = SSOConfig.for_openai_codex()
        assert cfg.provider == SSOProvider.OPENAI_CODEX
        assert cfg.issuer_url == "https://auth.openai.com"
        assert cfg.client_id == "app_EMoamEEZ73f0CkXaXp7hrann"
        assert cfg.use_pkce is True
        assert "offline_access" in cfg.scopes
        assert "api.connectors.read" in cfg.scopes
        assert "api.connectors.invoke" in cfg.scopes
        assert cfg.redirect_uri == "http://localhost:1455/auth/callback"

    def test_for_qwen_factory(self):
        cfg = SSOConfig.for_qwen()
        assert cfg.provider == SSOProvider.QWEN
        assert cfg.issuer_url == "https://chat.qwen.ai"
        assert cfg.use_pkce is True


# ---------------------------------------------------------------------------
# OAuthTokenManager — token persistence
# ---------------------------------------------------------------------------


class TestOAuthTokenManagerPersistence:
    """Test token save / load / clear on disk."""

    @pytest.fixture
    def tmp_storage(self, tmp_path):
        return tmp_path / ".victor"

    @pytest.fixture
    def manager(self, tmp_storage):
        return OAuthTokenManager("openai", storage_dir=tmp_storage)

    def _make_tokens(self, expired=False):
        if expired:
            expires = datetime.now(timezone.utc) - timedelta(hours=1)
        else:
            expires = datetime.now(timezone.utc) + timedelta(hours=1)
        return SSOTokens(
            access_token="acc_test_123",
            refresh_token="ref_test_456",
            expires_at=expires,
            scopes=["openid", "profile"],
        )

    def test_save_creates_file(self, manager, tmp_storage):
        tokens = self._make_tokens()
        manager._save(tokens)
        path = tmp_storage / "oauth_tokens.yaml"
        assert path.exists()

    def test_save_file_permissions(self, manager, tmp_storage):
        tokens = self._make_tokens()
        manager._save(tokens)
        path = tmp_storage / "oauth_tokens.yaml"
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600

    def test_load_cached_returns_tokens(self, manager):
        tokens = self._make_tokens()
        manager._save(tokens)
        with patch("victor.providers.oauth_manager.platform.system", return_value="Linux"):
            loaded = manager._load_cached()
        assert loaded is not None
        assert loaded.access_token == "acc_test_123"
        assert loaded.refresh_token == "ref_test_456"

    def test_load_cached_returns_none_when_no_file(self, manager):
        assert manager._load_cached() is None

    def test_load_cached_reads_codex_source_without_victor_token_file(self, tmp_path):
        codex_auth = tmp_path / ".codex" / "auth.json"
        codex_auth.parent.mkdir(parents=True)
        codex_auth.write_text(
            json.dumps(
                {
                    "auth_mode": "chatgpt",
                    "tokens": {
                        "access_token": "codex_access",
                        "refresh_token": "codex_refresh",
                        "account_id": "workspace_123",
                        "scope": "openid profile email offline_access",
                    },
                }
            )
        )
        manager = OAuthTokenManager(
            "openai",
            storage_dir=tmp_path / ".victor",
            token_source="codex",
            codex_auth_path=codex_auth,
        )

        with patch("victor.providers.oauth_manager.platform.system", return_value="Linux"):
            loaded = manager._load_cached()

        assert loaded is not None
        assert loaded.access_token == "codex_access"
        assert loaded.refresh_token == "codex_refresh"
        assert loaded.scopes == ["openid", "profile", "email", "offline_access"]
        assert manager.get_chatgpt_account_id() == "workspace_123"
        assert not (tmp_path / ".victor" / "oauth_tokens.yaml").exists()

    def test_load_cached_reads_claude_code_env_token(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "claude_env_token")
        manager = OAuthTokenManager(
            "anthropic",
            storage_dir=tmp_path / ".victor",
            token_source="claude-code",
        )

        loaded = manager._load_cached()

        assert loaded is not None
        assert loaded.access_token == "claude_env_token"
        assert loaded.token_type == "Bearer"
        assert not (tmp_path / ".victor" / "oauth_tokens.yaml").exists()

    def test_load_cached_reads_claude_code_macos_keychain(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        manager = OAuthTokenManager(
            "anthropic",
            storage_dir=tmp_path / ".victor",
            token_source="claude-code",
            claude_credentials_path=tmp_path / ".claude" / ".credentials.json",
        )
        completed = MagicMock(
            returncode=0,
            stdout=json.dumps({"claudeAiOauth": {"accessToken": "claude_keychain_token"}}),
        )

        with patch("victor.providers.oauth_manager.platform.system", return_value="Darwin"):
            with patch("victor.providers.oauth_manager.getpass.getuser", return_value="vijay"):
                with patch(
                    "victor.providers.oauth_manager.subprocess.run",
                    return_value=completed,
                ) as run:
                    loaded = manager._load_cached()

        assert loaded is not None
        assert loaded.access_token == "claude_keychain_token"
        assert "claude_keychain_token" not in str(run.call_args)
        assert run.call_args.args[0] == [
            "security",
            "find-generic-password",
            "-a",
            "vijay",
            "-w",
            "-s",
            "Claude Code-credentials",
        ]

    def test_load_cached_reads_claude_code_credentials_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        credentials_path = tmp_path / ".claude" / ".credentials.json"
        credentials_path.parent.mkdir(parents=True)
        credentials_path.write_text(
            json.dumps(
                {
                    "claudeAiOauth": {
                        "accessToken": "claude_file_token",
                        "refreshToken": "claude_refresh",
                        "expiresAt": "2027-01-01T00:00:00Z",
                    }
                }
            )
        )
        manager = OAuthTokenManager(
            "anthropic",
            storage_dir=tmp_path / ".victor",
            token_source="claude-code",
            claude_credentials_path=credentials_path,
        )

        with patch("victor.providers.oauth_manager.platform.system", return_value="Linux"):
            loaded = manager._load_cached()

        assert loaded is not None
        assert loaded.access_token == "claude_file_token"
        assert loaded.refresh_token == "claude_refresh"
        assert loaded.expires_at is not None

    def test_load_cached_returns_none_for_expired(self, manager):
        tokens = self._make_tokens(expired=True)
        manager._save(tokens)
        loaded = manager._load_cached()
        # Should still return the tokens (refresh handles expiry)
        assert loaded is not None
        assert loaded.is_expired is True

    def test_clear_removes_provider_tokens(self, manager, tmp_storage):
        tokens = self._make_tokens()
        manager._save(tokens)
        manager.clear()
        assert manager._load_cached() is None

    def test_multiple_providers_isolated(self, tmp_storage):
        m1 = OAuthTokenManager("openai", storage_dir=tmp_storage)
        m2 = OAuthTokenManager("qwen", storage_dir=tmp_storage)

        t1 = SSOTokens(access_token="openai_tok", refresh_token="r1")
        t2 = SSOTokens(access_token="qwen_tok", refresh_token="r2")
        m1._save(t1)
        m2._save(t2)

        assert m1._load_cached().access_token == "openai_tok"
        assert m2._load_cached().access_token == "qwen_tok"

    def test_clear_only_affects_own_provider(self, tmp_storage):
        m1 = OAuthTokenManager("openai", storage_dir=tmp_storage)
        m2 = OAuthTokenManager("qwen", storage_dir=tmp_storage)
        m1._save(SSOTokens(access_token="a", refresh_token="r"))
        m2._save(SSOTokens(access_token="b", refresh_token="r"))
        m1.clear()
        assert m1._load_cached() is None
        assert m2._load_cached().access_token == "b"


# ---------------------------------------------------------------------------
# OAuthTokenManager — login flow
# ---------------------------------------------------------------------------


class TestOAuthTokenManagerLogin:
    """Test browser-based OAuth login delegation."""

    @pytest.fixture
    def manager(self, tmp_path):
        return OAuthTokenManager("openai", storage_dir=tmp_path / ".victor")

    @pytest.mark.asyncio
    async def test_login_delegates_to_sso_authenticator(self, manager):
        mock_tokens = SSOTokens(
            access_token="new_access",
            refresh_token="new_refresh",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        with patch("victor.providers.oauth_manager.SSOAuthenticator") as MockAuth:
            MockAuth.return_value.login = AsyncMock(return_value=mock_tokens)
            tokens = await manager.login()

        assert tokens.access_token == "new_access"
        MockAuth.assert_called_once()

    @pytest.mark.asyncio
    async def test_login_persists_tokens(self, manager):
        mock_tokens = SSOTokens(
            access_token="persisted",
            refresh_token="ref",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        with patch("victor.providers.oauth_manager.SSOAuthenticator") as MockAuth:
            MockAuth.return_value.login = AsyncMock(return_value=mock_tokens)
            await manager.login()

        cached = manager._load_cached()
        assert cached is not None
        assert cached.access_token == "persisted"


# ---------------------------------------------------------------------------
# OAuthTokenManager — refresh flow
# ---------------------------------------------------------------------------


class TestOAuthTokenManagerRefresh:
    """Test automatic token refresh."""

    @pytest.fixture
    def manager(self, tmp_path):
        return OAuthTokenManager("openai", storage_dir=tmp_path / ".victor")

    @pytest.mark.asyncio
    async def test_refresh_returns_new_tokens(self, manager):
        old = SSOTokens(
            access_token="old",
            refresh_token="ref_tok",
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
        manager._save(old)

        new_tokens = SSOTokens(
            access_token="refreshed",
            refresh_token="ref_tok",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        with patch("victor.providers.oauth_manager.SSOAuthenticator") as MockAuth:
            MockAuth.return_value.refresh = AsyncMock(return_value=new_tokens)
            result = await manager.refresh()

        assert result.access_token == "refreshed"

    @pytest.mark.asyncio
    async def test_refresh_persists_new_tokens(self, manager):
        old = SSOTokens(access_token="old", refresh_token="ref_tok")
        manager._save(old)

        new_tokens = SSOTokens(
            access_token="refreshed_saved",
            refresh_token="ref_tok",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        with patch("victor.providers.oauth_manager.SSOAuthenticator") as MockAuth:
            MockAuth.return_value.refresh = AsyncMock(return_value=new_tokens)
            await manager.refresh()

        cached = manager._load_cached()
        assert cached.access_token == "refreshed_saved"


# ---------------------------------------------------------------------------
# OAuthTokenManager — get_valid_token (main entry point)
# ---------------------------------------------------------------------------


class TestOAuthTokenManagerGetValidToken:
    """Test the high-level get_valid_token() orchestration."""

    @pytest.fixture
    def manager(self, tmp_path):
        return OAuthTokenManager("openai", storage_dir=tmp_path / ".victor")

    @pytest.mark.asyncio
    async def test_returns_cached_token_when_valid(self, manager):
        tokens = SSOTokens(
            access_token="valid_tok",
            refresh_token="ref",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        manager._save(tokens)
        result = await manager.get_valid_token()
        assert result == "valid_tok"

    @pytest.mark.asyncio
    async def test_refreshes_when_expiring_soon(self, manager):
        """Token expiring in <5 minutes should trigger refresh."""
        tokens = SSOTokens(
            access_token="expiring",
            refresh_token="ref",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=2),
        )
        manager._save(tokens)

        refreshed = SSOTokens(
            access_token="fresh",
            refresh_token="ref",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        with patch("victor.providers.oauth_manager.SSOAuthenticator") as MockAuth:
            MockAuth.return_value.refresh = AsyncMock(return_value=refreshed)
            result = await manager.get_valid_token()

        assert result == "fresh"

    @pytest.mark.asyncio
    async def test_triggers_login_when_no_cached(self, manager):
        new_tokens = SSOTokens(
            access_token="brand_new",
            refresh_token="ref",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        with patch("victor.providers.oauth_manager.SSOAuthenticator") as MockAuth:
            MockAuth.return_value.login = AsyncMock(return_value=new_tokens)
            result = await manager.get_valid_token()

        assert result == "brand_new"

    @pytest.mark.asyncio
    async def test_falls_back_to_login_on_refresh_failure(self, manager):
        """If refresh fails, fall back to full login."""
        expired = SSOTokens(
            access_token="expired",
            refresh_token="bad_ref",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        manager._save(expired)

        new_tokens = SSOTokens(
            access_token="relogged",
            refresh_token="new_ref",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        with patch("victor.providers.oauth_manager.SSOAuthenticator") as MockAuth:
            MockAuth.return_value.refresh = AsyncMock(side_effect=ValueError("refresh failed"))
            MockAuth.return_value.login = AsyncMock(return_value=new_tokens)
            result = await manager.get_valid_token()

        assert result == "relogged"


# ---------------------------------------------------------------------------
# OpenAIProvider OAuth integration
# ---------------------------------------------------------------------------


class TestOpenAIProviderOAuth:
    """Test OpenAIProvider with auth_mode='oauth'."""

    @pytest.mark.asyncio
    async def test_oauth_mode_uses_token_as_api_key(self):
        tokens = SSOTokens(
            access_token="oauth_access_token",
            refresh_token="ref",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        with patch("victor.providers.openai_provider.OAuthTokenManager") as MockMgr:
            mock_instance = MagicMock()
            mock_instance.get_valid_token = AsyncMock(return_value="oauth_access_token")
            mock_instance._load_cached = MagicMock(return_value=tokens)
            MockMgr.return_value = mock_instance

            from victor.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider(auth_mode="oauth")

        assert provider.client.api_key == "oauth_access_token"

    def test_api_key_mode_unchanged(self):
        """Default auth_mode='api_key' works exactly as before."""
        from victor.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="sk-test-key")
        assert provider.client.api_key == "sk-test-key"

    @pytest.mark.asyncio
    async def test_ensure_valid_token_refreshes_before_call(self):
        """_ensure_valid_token should refresh expired OAuth tokens."""
        tokens = SSOTokens(
            access_token="initial",
            refresh_token="ref",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        with patch("victor.providers.openai_provider.OAuthTokenManager") as MockMgr:
            mock_instance = MagicMock()
            mock_instance.get_valid_token = AsyncMock(return_value="refreshed_tok")
            mock_instance._load_cached = MagicMock(return_value=tokens)
            MockMgr.return_value = mock_instance

            from victor.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider(auth_mode="oauth")

            # Simulate token refresh on next call
            mock_instance.get_valid_token = AsyncMock(return_value="refreshed_tok")
            await provider._ensure_valid_token()

        assert provider.client.api_key == "refreshed_tok"
