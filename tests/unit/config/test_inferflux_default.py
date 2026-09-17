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

"""Pins for InferFlux as the default provider (Ollama demoted to optional).

Covers the three seams the defaults flip touches: the ProviderSettings
allowlist (without an entry the flip fails validation), bundled profile
seed shape, and ``--endpoint`` plumbing for the two llama.cpp-family
providers.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from victor.config.groups.provider_config import ProviderSettings
from victor.config.settings import Settings


def test_default_provider_settings_are_inferflux():
    ps = ProviderSettings()
    assert ps.default_provider == "inferflux"
    assert ps.default_model == "qwen3-coder-30b"


def test_allowlist_accepts_inferflux_and_rejects_unknown():
    assert ProviderSettings(default_provider="inferflux") is not None
    with pytest.raises(ValidationError, match="Unknown provider"):
        ProviderSettings(default_provider="nope")


def test_bundled_default_profile_seeds_inferflux(tmp_path):
    """First run seeds `default: inferflux` without touching user files."""
    from unittest.mock import patch

    with patch.object(Settings, "get_config_dir", classmethod(lambda cls: tmp_path)):
        profiles = Settings.load_profiles()
    assert profiles["default"].provider == "inferflux"
    assert profiles["default"].model == "qwen3-coder-30b"
    # ollama remains available as the air-gapped profile
    assert profiles["local"].provider == "ollama"
    assert "local-llamacpp" in profiles


def test_endpoint_override_writes_generic_base_url(tmp_path, monkeypatch):
    """--endpoint for llamacpp/inferflux writes providers[<name>].base_url."""
    from victor.framework.session_config import SessionConfig

    monkeypatch.setenv("VICTOR_CONFIG_DIR", str(tmp_path))
    for provider, model, endpoint in (
        # llamacpp has no policy default by design (llama-server serves whatever
        # model it was launched with) — pass it explicitly.
        ("llamacpp", "local-model", "http://localhost:8081/v1"),
        ("inferflux", "qwen3-coder-30b", "http://127.0.0.1:8080/v1"),
    ):
        config = SessionConfig.from_cli_flags(
            provider=provider,
            model=model,
            endpoint=endpoint,
        )
        settings = Settings()
        config.apply_to_settings(settings)
        pcfg = settings.provider.providers.get(provider)
        assert pcfg is not None, f"{provider} provider config missing"
        assert pcfg.base_url == endpoint


def test_first_time_user_false_when_inferflux_reachable(monkeypatch):
    """A reachable InferFlux endpoint means the user is set up (no wizard).

    Runs inside the suite's sandboxed HOME (tests/unit/conftest.py): the
    pre-created ``.victor`` dir is used as-is. The "no profiles.yaml =
    first-time" short-circuit predates the probe and stays authoritative, so a
    profiles file is what unlocks the endpoint-probe path.
    """
    import pathlib

    import victor.config.settings as settings_mod

    class _Response:
        status_code = 200

    calls = {}

    def _fake_get(url, headers=None, timeout=None):
        calls["url"] = url
        calls["headers"] = headers or {}
        return _Response()

    monkeypatch.setattr("httpx.get", _fake_get)
    monkeypatch.setenv("INFERFLUX_API_KEY", "test-key-123")
    victor_dir = pathlib.Path.home() / ".victor"
    profiles_file = victor_dir / "profiles.yaml"
    profiles_file.write_text("profiles: {}\n")
    try:
        assert settings_mod.is_first_time_user() is False
        assert calls["url"].startswith("http://127.0.0.1:8080/")
        assert calls["headers"].get("Authorization") == "Bearer test-key-123"
    finally:
        profiles_file.unlink(missing_ok=True)


class _StubProviderConfig:
    base_url = None
    api_key_value = None


class _StubProviderSettings:
    default_provider = "inferflux"
    providers = {"inferflux": _StubProviderConfig()}


class _StubSettings:
    provider = _StubProviderSettings()


def test_doctor_inferflux_reachable(monkeypatch):
    """Doctor's InferFlux check reports SUCCESS on a healthy healthz."""
    import httpx
    from victor.ui.commands import doctor as doctor_mod

    class _Response:
        status_code = 200

    monkeypatch.setattr(httpx, "get", lambda url, headers=None, timeout=None: _Response())
    monkeypatch.setattr("victor.config.settings.load_settings", lambda: _StubSettings())
    checks = doctor_mod.DoctorChecks()
    checks.check_inferflux()
    check = checks.checks[-1]
    assert "InferFlux" in check.name
    assert check.severity == doctor_mod.Severity.SUCCESS


def test_doctor_inferflux_unreachable_warns_when_default(monkeypatch):
    import httpx
    from victor.ui.commands import doctor as doctor_mod

    def _raise(url, headers=None, timeout=None):
        raise httpx.ConnectError("tunnel down")

    monkeypatch.setattr(httpx, "get", _raise)
    monkeypatch.setattr("victor.config.settings.load_settings", lambda: _StubSettings())
    checks = doctor_mod.DoctorChecks()
    checks.check_inferflux()
    check = checks.checks[-1]
    assert check.severity == doctor_mod.Severity.WARNING
    assert "ssh -N -L 8080" in (check.suggestion or "")
