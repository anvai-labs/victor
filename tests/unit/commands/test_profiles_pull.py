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

"""Tests for `victor profiles pull` (Phase 5b — sandhi GET /auth/profile).

The pull downloads the caller's per-user gateway profile with a ``skp_``
bootstrap token and lands it in ``~/.victor/profiles.d/sandhi.yaml`` (0600).
Contract points pinned here: success translation + permissions, the
shown-once virtual key never reaching stdout or logs, the operator-facing
error paths (401 / 404 / 503 / transport), and the token prompt path.
"""

import getpass
import stat
import sys
import yaml
import httpx
import pytest
from typer.testing import CliRunner

from victor.ui.commands.profiles import profiles_app

runner = CliRunner()

GATEWAY_PAYLOAD = {
    "mode": "gateway",
    "endpoint": "https://dataserver3.singh.local:18788",
    "group": "t_abc123",
    "defaults": {"timeout_seconds": 300, "max_tokens": 8192},
    "models": ["claude-sonnet-5-5", "claude-opus-4-8"],
    "credentials": {"virtual_key": "vk_secret0123456789"},
}

LOCAL_PAYLOAD = {
    "mode": "local",
    "group": "t_abc123",
    "defaults": {"max_tokens": 4096},
    "models": ["claude-sonnet-5-5"],
}

BASE_FLAGS = ["--gateway-url", "https://gw.example/v1", "--token", "skp_tok"]


def _invoke(config_dir, extra_args=()):
    return runner.invoke(
        profiles_app,
        ["pull", "--config-dir", str(config_dir), *extra_args],
    )


def _mock_response(status_code=200, payload=None):
    response = httpx.Response(
        status_code, json=payload, request=httpx.Request("GET", "https://gw.example/auth/profile")
    )
    return response


@pytest.fixture
def fetched():
    """Capture the (url, token) the client was called with; return the patcher."""
    calls = {}

    def fake_get(url, headers=None, timeout=None):
        calls["url"] = url
        calls["token"] = (headers or {}).get("Authorization", "")
        return _mock_response(200, GATEWAY_PAYLOAD)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("victor.observability.sandhi_profile.httpx.get", fake_get)
        yield calls


class TestPullSuccess:
    def test_success_writes_translated_profile(self, tmp_path, fetched):
        result = _invoke(tmp_path, BASE_FLAGS)
        assert result.exit_code == 0, result.output
        central = tmp_path / "profiles.d" / "sandhi.yaml"
        assert central.exists()
        doc = yaml.safe_load(central.read_text())
        entry = doc["profiles"]["sandhi"]
        # claude-* → the anthropic dialect; defaults map to native keys.
        assert entry["provider"] == "anthropic"
        assert entry["model"] == "claude-sonnet-5-5"
        assert entry["max_tokens"] == 8192
        assert entry["timeout"] == 300
        # The shown-once vk lands in the gateway block — and only there.
        assert (
            doc["providers"]["anthropic"]["gateway"]["url"]
            == "https://dataserver3.singh.local:18788"
        )
        assert doc["providers"]["anthropic"]["gateway"]["virtual_key"] == "vk_secret0123456789"

    def test_fetch_uses_root_url_and_bearer_token(self, tmp_path, fetched):
        # A trailing /v1 on --gateway-url is stripped; the skp_ token rides
        # the Authorization header.
        result = _invoke(tmp_path, BASE_FLAGS)
        assert result.exit_code == 0, result.output
        assert fetched["url"] == "https://gw.example/auth/profile"
        assert fetched["token"] == "Bearer skp_tok"

    def test_output_masks_virtual_key(self, tmp_path, fetched, capfd):
        result = _invoke(tmp_path, BASE_FLAGS)
        assert result.exit_code == 0, result.output
        assert "vk_secret0123456789" not in result.output
        assert "vk_sec…" in result.output  # first 6 chars + ellipsis
        # Nothing secret on stderr either.
        assert "vk_secret0123456789" not in capfd.readouterr().err

    def test_repull_overwrites_previous_file(self, tmp_path, fetched):
        central_dir = tmp_path / "profiles.d"
        central_dir.mkdir()
        (central_dir / "sandhi.yaml").write_text("profiles:\n  sandhi:\n    provider: old\n")
        result = _invoke(tmp_path, BASE_FLAGS)
        assert result.exit_code == 0, result.output
        doc = yaml.safe_load((central_dir / "sandhi.yaml").read_text())
        assert doc["profiles"]["sandhi"]["provider"] == "anthropic"

    def test_local_mode_writes_no_secrets(self, tmp_path, fetched):
        def fake_get(url, headers=None, timeout=None):
            return _mock_response(200, LOCAL_PAYLOAD)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("victor.observability.sandhi_profile.httpx.get", fake_get)
            result = _invoke(tmp_path, BASE_FLAGS)
        assert result.exit_code == 0, result.output
        text = (tmp_path / "profiles.d" / "sandhi.yaml").read_text()
        assert "providers:" not in text  # no gateway block in local mode
        assert "vk" not in text
        # FFI note: local-mode profiles carry env-var NAME guidance only —
        # the standing env > keyring > api_keys.yaml resolution is untouched.
        assert "ANTHROPIC_API_KEY" in text
        assert "SandhiMeter" in text

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
    def test_file_0600_dir_0700(self, tmp_path, fetched):
        result = _invoke(tmp_path, BASE_FLAGS)
        assert result.exit_code == 0, result.output
        central = tmp_path / "profiles.d" / "sandhi.yaml"
        assert stat.S_IMODE(central.stat().st_mode) == 0o600
        assert stat.S_IMODE((tmp_path / "profiles.d").stat().st_mode) == 0o700


class TestPullTokenSources:
    def test_env_token_used(self, tmp_path, fetched, monkeypatch):
        monkeypatch.setenv("SANDHI_PROFILE_TOKEN", "skp_envtok")
        result = _invoke(tmp_path, ["--gateway-url", "https://gw.example"])
        assert result.exit_code == 0, result.output
        assert fetched["token"] == "Bearer skp_envtok"

    def test_prompted_token_hidden(self, tmp_path, fetched, monkeypatch):
        monkeypatch.setattr("victor.ui.commands.profiles._stdin_isatty", lambda: True)
        monkeypatch.setattr(getpass, "getpass", lambda _prompt: "skp_prompttok")
        result = _invoke(tmp_path, ["--gateway-url", "https://gw.example"])
        assert result.exit_code == 0, result.output
        assert fetched["token"] == "Bearer skp_prompttok"
        # The prompted token itself must never be echoed.
        assert "skp_prompttok" not in result.output

    def test_no_token_and_no_tty_errors(self, tmp_path, fetched):
        # CliRunner stdin is not a TTY — no prompt possible, no env fallback.
        monkeypatch_free_result = _invoke(tmp_path, ["--gateway-url", "https://gw.example"])
        assert monkeypatch_free_result.exit_code == 1
        assert "no profile token" in monkeypatch_free_result.output


class TestPullGatewayUrlSources:
    def test_previous_pull_endpoint_is_default(self, tmp_path, fetched):
        # First pull records the endpoint; a second pull without --gateway-url
        # reuses it (with the /v1 spelling normalized away).
        _invoke(tmp_path, BASE_FLAGS)
        fetched.clear()

        def fake_get(url, headers=None, timeout=None):
            fetched["url"] = url
            return _mock_response(200, GATEWAY_PAYLOAD)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("victor.observability.sandhi_profile.httpx.get", fake_get)
            result = _invoke(tmp_path, ["--token", "skp_tok"])
        assert result.exit_code == 0, result.output
        # The re-pull default is the profile's OWN endpoint (from the payload),
        # not the URL the first pull was pointed at.
        assert fetched["url"] == "https://dataserver3.singh.local:18788/auth/profile"

    def test_env_gateway_url_used_when_no_previous_pull(self, tmp_path, fetched, monkeypatch):
        monkeypatch.setenv("SANDHI_GATEWAY_URL", "https://env-gw.example")
        result = _invoke(tmp_path, ["--token", "skp_tok"])
        assert result.exit_code == 0, result.output
        assert fetched["url"] == "https://env-gw.example/auth/profile"

    def test_no_url_and_no_tty_errors(self, tmp_path, fetched, monkeypatch):
        monkeypatch.delenv("SANDHI_GATEWAY_URL", raising=False)
        result = _invoke(tmp_path, ["--token", "skp_tok"])
        assert result.exit_code == 1
        assert "no gateway URL" in result.output


class TestPullErrorPaths:
    def _invoke_status(self, tmp_path, status_code, payload=None):
        def fake_get(url, headers=None, timeout=None):
            return _mock_response(status_code, payload)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("victor.observability.sandhi_profile.httpx.get", fake_get)
            return _invoke(tmp_path, BASE_FLAGS)

    def test_401_token_rejected(self, tmp_path):
        result = self._invoke_status(tmp_path, 401)
        assert result.exit_code == 1
        assert "rejected or revoked" in result.output
        assert not (tmp_path / "profiles.d" / "sandhi.yaml").exists()

    def test_404_no_profile_for_group(self, tmp_path):
        result = self._invoke_status(tmp_path, 404)
        assert result.exit_code == 1
        assert "no profile configured for your group" in result.output

    def test_503_store_unavailable(self, tmp_path):
        result = self._invoke_status(tmp_path, 503)
        assert result.exit_code == 1
        assert "store is unavailable" in result.output

    def test_unexpected_status(self, tmp_path):
        result = self._invoke_status(tmp_path, 500)
        assert result.exit_code == 1
        assert "HTTP 500" in result.output

    def test_network_error_never_raises(self, tmp_path):
        def fake_get(url, headers=None, timeout=None):
            raise httpx.ConnectError("boom")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("victor.observability.sandhi_profile.httpx.get", fake_get)
            result = _invoke(tmp_path, BASE_FLAGS)
        assert result.exit_code == 1
        assert "could not reach the gateway" in result.output

    def test_contract_violation_reported_not_raised(self, tmp_path):
        result = self._invoke_status(tmp_path, 200, {"mode": "quantum"})
        assert result.exit_code == 1
        assert "unknown profile mode" in result.output
