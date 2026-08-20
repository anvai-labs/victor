# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Tests for the auth UX commands: unified list/show key cell, clean oauth status, env bridge."""

from types import SimpleNamespace

from typer.testing import CliRunner

import victor.ui.commands.auth as auth
from victor.ui.commands.auth import (
    AuthStatus,
    _oauth_status_value,
    _provider_key_cell,
    auth_app,
)


def test_provider_key_cell_local_configured_missing():
    status = {
        "zai": {"configured": True, "source": "keyring"},
        "openai": {"configured": False},
    }
    assert _provider_key_cell("ollama", status, plain=True) == "local"
    assert _provider_key_cell("zai", status, plain=True) == "key:keyring"
    assert _provider_key_cell("openai", status, plain=True) == "missing"
    # provider entirely absent from status -> missing
    assert _provider_key_cell("anthropic", status, plain=True) == "missing"


def test_oauth_status_value_does_not_leak_enum_repr(monkeypatch):
    """Regression: f-string of a str-Enum yields 'AuthStatus.EXPIRED', not 'expired'."""
    monkeypatch.setattr(auth, "_get_oauth_status", lambda provider, source: AuthStatus.EXPIRED)
    account = SimpleNamespace(provider="openai", auth=SimpleNamespace(source="oauth"))
    assert _oauth_status_value(account) == "expired"
    assert "AuthStatus" not in _oauth_status_value(account)


def test_auth_env_emits_export_when_captured(monkeypatch):
    """When stdout is captured (CliRunner / eval / pipe), emit a shell export line."""
    monkeypatch.setattr(
        auth.APIKeyManager,
        "get_status",
        lambda self: {"zai": {"configured": True, "source": "keyring", "env_var": "ZAI_API_KEY"}},
    )
    monkeypatch.setattr(auth.APIKeyManager, "get_key", lambda self, p: "sk-secret-value")

    result = CliRunner().invoke(auth_app, ["env", "-p", "zai"])
    assert result.exit_code == 0
    assert "export ZAI_API_KEY=" in result.stdout
    assert "sk-secret-value" in result.stdout  # the eval target carries the real value


def test_auth_env_refuses_on_a_terminal(monkeypatch):
    """Safety: never print keys to an interactive terminal — only when captured."""
    monkeypatch.setattr(auth, "_stdout_is_tty", lambda: True)
    monkeypatch.setattr(
        auth.APIKeyManager,
        "get_status",
        lambda self: {"zai": {"configured": True, "source": "keyring", "env_var": "ZAI_API_KEY"}},
    )
    monkeypatch.setattr(auth.APIKeyManager, "get_key", lambda self, p: "sk-secret-value")

    result = CliRunner().invoke(auth_app, ["env", "-p", "zai"])
    assert result.exit_code == 1
    assert "export" not in result.stdout
    assert "sk-secret-value" not in result.stdout


def test_auth_env_exit_1_when_no_key(monkeypatch):
    monkeypatch.setattr(auth.APIKeyManager, "get_status", lambda self: {})
    monkeypatch.setattr(auth.APIKeyManager, "get_key", lambda self, p: None)
    result = CliRunner().invoke(auth_app, ["env", "-p", "nope"])
    assert result.exit_code == 1
    assert "export" not in result.stdout


def test_auth_show_unknown_name_exits_1(monkeypatch):
    monkeypatch.setattr(
        auth, "get_account_manager", lambda: SimpleNamespace(list_accounts=lambda: [])
    )
    monkeypatch.setattr(auth.APIKeyManager, "get_status", lambda self: {})
    result = CliRunner().invoke(auth_app, ["show", "does-not-exist"])
    assert result.exit_code == 1
