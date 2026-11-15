"""Shared fixtures for security-boundary policy owners."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from victor.config.sandbox_settings import SandboxSettings
from victor.runtime.policy_snapshot import RunPolicySnapshot
from victor.tools.sandbox import backends


@pytest.fixture
def install_policy(monkeypatch: pytest.MonkeyPatch):
    """Install a deterministic Linux sandbox-policy environment."""

    def _install(
        *,
        policy: str,
        backend_available: bool,
    ) -> RunPolicySnapshot:
        settings = SimpleNamespace(
            sandbox=SandboxSettings(sandbox_enabled=True, sandbox_policy=policy)
        )
        monkeypatch.setattr("victor.config.settings.get_settings", lambda: settings)
        monkeypatch.setattr(backends.platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            backends.shutil,
            "which",
            lambda name: "/usr/bin/bwrap" if backend_available and name == "bwrap" else None,
        )
        return RunPolicySnapshot.from_settings(settings)

    return _install
