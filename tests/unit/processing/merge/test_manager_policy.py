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

"""Merge-manager git launches follow the admitted sandbox policy."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import victor.processing.merge.manager as manager_module
from victor.processing.merge.manager import MergeManager
from victor.processing.merge.protocol import FileResolution, ResolutionStrategy


def _settings(policy: str) -> SimpleNamespace:
    from victor.config.sandbox_settings import SandboxSettings

    return SimpleNamespace(sandbox=SandboxSettings(sandbox_enabled=True, sandbox_policy=policy))


def _install_policy(monkeypatch, policy: str, backend_available: bool) -> None:
    from victor.tools.sandbox import backends

    monkeypatch.setattr(manager_module, "load_settings", lambda: _settings(policy))
    monkeypatch.setattr(backends.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        backends.shutil,
        "which",
        lambda name: "/usr/bin/bwrap" if backend_available and name == "bwrap" else None,
    )


def _manager(monkeypatch, tmp_path: Path, **kwargs) -> MergeManager:
    monkeypatch.setattr(
        manager_module,
        "get_project_paths",
        lambda _root: SimpleNamespace(project_victor_dir=tmp_path / ".victor"),
    )
    return MergeManager(tmp_path, **kwargs)


def _trap_git(monkeypatch, launched: list[list[str]]) -> None:
    def _run(argv: list[str], **_kwargs):
        launched.append([str(value) for value in argv])
        return MagicMock(returncode=0, stdout="")

    monkeypatch.setattr(manager_module.subprocess, "run", _run)


async def test_required_available_wraps_conflict_discovery_and_branch_reads(monkeypatch, tmp_path):
    _install_policy(monkeypatch, "required", backend_available=True)
    manager = _manager(monkeypatch, tmp_path)
    launched: list[list[str]] = []
    _trap_git(monkeypatch, launched)

    await manager.detect_conflicts()

    assert launched
    assert all(argv[0] == "bwrap" for argv in launched)
    assert all("git" in argv for argv in launched)


async def test_required_unavailable_refuses_conflict_discovery_before_spawn(monkeypatch, tmp_path):
    _install_policy(monkeypatch, "required", backend_available=False)
    manager = _manager(monkeypatch, tmp_path)
    launched: list[list[str]] = []
    _trap_git(monkeypatch, launched)

    with pytest.raises(Exception, match="(?i)required|policy"):
        await manager.detect_conflicts()

    assert launched == []


@pytest.mark.parametrize("policy", ["disabled", "preferred"])
async def test_non_required_policy_keeps_conflict_discovery_raw(monkeypatch, tmp_path, policy):
    _install_policy(monkeypatch, policy, backend_available=True)
    manager = _manager(monkeypatch, tmp_path)
    launched: list[list[str]] = []
    _trap_git(monkeypatch, launched)

    await manager.detect_conflicts()

    assert launched
    assert all(argv[0] == "git" for argv in launched)


async def test_unreadable_policy_refuses_conflict_discovery_before_spawn(monkeypatch, tmp_path):
    _install_policy(monkeypatch, "disabled", backend_available=True)
    manager = _manager(monkeypatch, tmp_path)

    def _unavailable():
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(manager_module, "load_settings", _unavailable)
    launched: list[list[str]] = []
    _trap_git(monkeypatch, launched)

    with pytest.raises(Exception, match="(?i)policy"):
        await manager.detect_conflicts()

    assert launched == []


async def test_required_unavailable_apply_resolution_neither_writes_nor_spawns(
    monkeypatch, tmp_path
):
    _install_policy(monkeypatch, "required", backend_available=False)
    target = tmp_path / "conflicted.py"
    target.write_text("original\n", encoding="utf-8")
    manager = _manager(monkeypatch, tmp_path)
    launched: list[list[str]] = []
    _trap_git(monkeypatch, launched)
    resolution = FileResolution(
        file_path=target,
        resolutions=[],
        final_content="resolved\n",
        fully_resolved=True,
    )

    with pytest.raises(Exception, match="(?i)required|policy"):
        await manager._apply_resolution(resolution)

    assert launched == []
    assert target.read_text(encoding="utf-8") == "original\n"
    assert resolution.applied is False


async def test_required_available_wraps_checkout_and_stage(monkeypatch, tmp_path):
    _install_policy(monkeypatch, "required", backend_available=True)
    manager = _manager(monkeypatch, tmp_path)
    launched: list[list[str]] = []
    _trap_git(monkeypatch, launched)

    applied = await manager.apply_resolution("conflicted.py", ResolutionStrategy.OURS)

    assert applied is True
    assert [argv[0] for argv in launched] == ["bwrap", "bwrap"]
    assert any("checkout" in argv for argv in launched)
    assert any("add" in argv for argv in launched)


async def test_required_available_keeps_one_policy_for_merge_to_rebase_abort_fallback(
    monkeypatch, tmp_path
):
    _install_policy(monkeypatch, "required", backend_available=True)
    manager = _manager(monkeypatch, tmp_path)
    launched: list[list[str]] = []

    def _run(argv: list[str], **_kwargs):
        launched.append([str(value) for value in argv])
        if "merge" in argv:
            raise manager_module.subprocess.CalledProcessError(1, argv)
        return MagicMock(returncode=0)

    monkeypatch.setattr(manager_module.subprocess, "run", _run)

    aborted = await manager.abort_merge()

    assert aborted is True
    assert [argv[0] for argv in launched] == ["bwrap", "bwrap"]
    assert any("merge" in argv for argv in launched)
    assert any("rebase" in argv for argv in launched)


async def test_admitted_snapshot_overrides_reloaded_required_settings(monkeypatch, tmp_path):
    from victor.runtime.policy_snapshot import RunPolicySnapshot

    _install_policy(monkeypatch, "disabled", backend_available=True)
    admitted = RunPolicySnapshot.from_settings(manager_module.load_settings())
    _install_policy(monkeypatch, "required", backend_available=False)
    manager = _manager(monkeypatch, tmp_path, run_policy_snapshot=admitted)
    launched: list[list[str]] = []
    _trap_git(monkeypatch, launched)

    await manager.detect_conflicts()

    assert launched
    assert all(argv[0] == "git" for argv in launched)


async def test_admitted_snapshot_survives_unreadable_settings_after_admission(
    monkeypatch, tmp_path
):
    from victor.runtime.policy_snapshot import RunPolicySnapshot

    _install_policy(monkeypatch, "disabled", backend_available=True)
    admitted = RunPolicySnapshot.from_settings(manager_module.load_settings())

    def _unavailable():
        raise RuntimeError("settings unavailable after admission")

    monkeypatch.setattr(manager_module, "load_settings", _unavailable)
    manager = _manager(monkeypatch, tmp_path, run_policy_snapshot=admitted)
    launched: list[list[str]] = []
    _trap_git(monkeypatch, launched)

    await manager.detect_conflicts()

    assert launched
    assert all(argv[0] == "git" for argv in launched)
