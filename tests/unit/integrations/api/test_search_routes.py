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

"""Search-route launch-policy boundary tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from victor.integrations.api.fastapi_server import CodeSearchRequest
from victor.integrations.api.routes import search_routes


def _endpoint(server: SimpleNamespace) -> Any:
    return next(
        route.endpoint
        for route in search_routes.create_router(server).routes
        if route.path == "/search/code"
    )


def _install_policy(monkeypatch, policy: str, backend_available: bool) -> None:
    from victor.config.sandbox_settings import SandboxSettings

    settings = SimpleNamespace(sandbox=SandboxSettings(sandbox_enabled=True, sandbox_policy=policy))
    monkeypatch.setattr("victor.config.settings.get_settings", lambda: settings)

    from victor.tools.sandbox import backends

    monkeypatch.setattr(backends.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        backends.shutil,
        "which",
        lambda name: "/usr/bin/bwrap" if backend_available and name == "bwrap" else None,
    )


def _trap_launch(monkeypatch, launched: list[dict[str, Any]]) -> None:
    def _run(argv: list[str], **kwargs: Any):
        launched.append({"argv": [str(value) for value in argv], "kwargs": kwargs})
        return SimpleNamespace(stdout="", returncode=1)

    import subprocess as subprocess_mod

    monkeypatch.setattr(subprocess_mod, "run", _run)


async def test_required_policy_wraps_search_route_launch(monkeypatch, tmp_path):
    """Required + enforceable wraps the actual rg argv at the OS boundary."""
    server = SimpleNamespace(workspace_root=str(tmp_path))
    endpoint = _endpoint(server)
    _install_policy(monkeypatch, "required", backend_available=True)
    launched: list[dict[str, Any]] = []
    _trap_launch(monkeypatch, launched)

    response = await endpoint(CodeSearchRequest(query="needle"))

    assert response.error is None
    assert launched, "expected a launch to inspect"
    assert launched[0]["argv"][0] == "bwrap", launched[0]["argv"]
    assert "rg" in launched[0]["argv"]
    assert "--" in launched[0]["argv"]


async def test_required_unavailable_refuses_search_route_before_launch(monkeypatch, tmp_path):
    """Required policy with no enforcing backend refuses with zero spawns."""
    server = SimpleNamespace(workspace_root=str(tmp_path))
    endpoint = _endpoint(server)
    _install_policy(monkeypatch, "required", backend_available=False)
    launched: list[dict[str, Any]] = []
    _trap_launch(monkeypatch, launched)

    response = await endpoint(CodeSearchRequest(query="needle"))

    assert launched == []
    assert response.results == []
    assert "required" in (response.error or "").lower()


@pytest.mark.parametrize("policy", ["disabled", "preferred"])
async def test_non_required_policy_keeps_search_route_raw_launch(monkeypatch, tmp_path, policy):
    """Explicit disabled/preferred behavior retains historical raw rg exec."""
    server = SimpleNamespace(workspace_root=str(tmp_path))
    endpoint = _endpoint(server)
    _install_policy(monkeypatch, policy, backend_available=True)
    launched: list[dict[str, Any]] = []
    _trap_launch(monkeypatch, launched)

    response = await endpoint(CodeSearchRequest(query="needle"))

    assert response.error is None
    assert launched[0]["argv"][0] == "rg"


async def test_unreadable_policy_refuses_search_route_before_launch(monkeypatch, tmp_path):
    """Unknowable policy refuses the search launch with zero spawns."""
    server = SimpleNamespace(workspace_root=str(tmp_path))
    endpoint = _endpoint(server)
    monkeypatch.setattr(
        "victor.config.settings.get_settings",
        lambda: (_ for _ in ()).throw(RuntimeError("settings unavailable")),
    )
    launched: list[dict[str, Any]] = []
    _trap_launch(monkeypatch, launched)

    response = await endpoint(CodeSearchRequest(query="needle"))

    assert launched == []
    assert response.results == []
    assert "policy" in (response.error or "").lower()


async def test_admitted_snapshot_overrides_reloaded_required_settings(monkeypatch, tmp_path):
    """A route admitted before reload keeps its disabled policy snapshot."""
    from victor.runtime.policy_snapshot import RunPolicySnapshot, run_policy_context

    server = SimpleNamespace(workspace_root=str(tmp_path))
    endpoint = _endpoint(server)
    _install_policy(monkeypatch, "disabled", backend_available=True)
    from victor.config.settings import get_settings

    admitted = RunPolicySnapshot.from_settings(get_settings())
    _install_policy(monkeypatch, "required", backend_available=False)
    launched: list[dict[str, Any]] = []
    _trap_launch(monkeypatch, launched)

    with run_policy_context(admitted):
        response = await endpoint(CodeSearchRequest(query="needle"))

    assert response.error is None
    assert launched and launched[0]["argv"][0] == "rg"


async def test_search_required_launch_carries_and_closes_descriptors(monkeypatch, tmp_path):
    """Finding 8: the direct search launcher forwards the backend's owned
    descriptors with the wrapped argv and closes them after the spawn."""
    import os

    server = SimpleNamespace(workspace_root=str(tmp_path))
    _install_policy(monkeypatch, "required", backend_available=True)
    captured: dict = {}

    def _run(argv: list[str], **kwargs: Any):
        captured["pass_fds"] = kwargs.get("pass_fds")
        return SimpleNamespace(stdout="", returncode=0)

    import subprocess as subprocess_mod

    monkeypatch.setattr(subprocess_mod, "run", _run)

    endpoint = _endpoint(server)
    await endpoint(CodeSearchRequest(query="probe", max_results=5))

    fds = captured.get("pass_fds")
    assert fds, "search launch dropped the backend's seccomp descriptor"
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)
