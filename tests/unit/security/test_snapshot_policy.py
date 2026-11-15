"""Snapshot-store git discovery obeys the canonical required-policy gate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import victor.agent.snapshot_store as snapshot_module
from victor.agent.snapshot_store import FileSnapshotStore
from victor.runtime.policy_snapshot import run_policy_context
from victor.tools.sandbox import backends


@pytest.fixture
def launches(monkeypatch):
    commands: list[list[str]] = []

    def _run(command, **_kwargs):
        commands.append(list(command))
        if "status" in command:
            return SimpleNamespace(returncode=0, stdout=" M example.py\n", stderr="")
        if "rev-parse" in command:
            return SimpleNamespace(returncode=0, stdout="abcdef123456\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(snapshot_module.subprocess, "run", _run)
    return commands


def _count_resolver(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    original = backends.resolve_required_launch

    def _resolver(settings):
        calls.append(1)
        return original(settings)

    import victor.tools.sandbox as sandbox_module

    monkeypatch.setattr(sandbox_module, "resolve_required_launch", _resolver)
    return calls


def test_required_available_wraps_status_and_head_with_one_decision(
    monkeypatch, install_policy, launches, tmp_path
):
    install_policy(policy="required", backend_available=True)
    (tmp_path / "example.py").write_text("content\n", encoding="utf-8")
    resolver_calls = _count_resolver(monkeypatch)
    store = FileSnapshotStore(workspace_root=tmp_path)

    snapshot_id = store.create_snapshot(description="modified files")

    assert len(resolver_calls) == 1
    assert len(launches) == 2
    assert {command[0] for command in launches} == {"bwrap"}
    assert all("git" in command for command in launches)
    snapshot = store.get_snapshot(snapshot_id)
    assert snapshot is not None
    assert snapshot.git_ref == "abcdef123456"
    assert [file_snapshot.path for file_snapshot in snapshot.files] == ["example.py"]


def test_required_unavailable_refuses_before_spawn_or_publication(
    monkeypatch, install_policy, launches, tmp_path
):
    install_policy(policy="required", backend_available=False)
    (tmp_path / "example.py").write_text("content\n", encoding="utf-8")
    persist_dir = tmp_path / ".snapshots"
    store = FileSnapshotStore(workspace_root=tmp_path, persist_dir=persist_dir)

    with pytest.raises(snapshot_module.SnapshotPolicyRefusal) as refusal:
        store.create_snapshot(files=["example.py"], description="refused")

    assert refusal.value.reason == "sandbox_required"
    assert launches == []
    assert store._snapshots == {}
    assert store._snapshot_order == []
    assert store._counter == 0
    assert list(persist_dir.glob("snap_*.json")) == []


def test_unreadable_policy_refuses_before_spawn_or_publication(
    monkeypatch, install_policy, launches, tmp_path
):
    def _unavailable():
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr("victor.config.settings.get_settings", _unavailable)
    store = FileSnapshotStore(workspace_root=tmp_path)

    with pytest.raises(snapshot_module.SnapshotPolicyRefusal) as refusal:
        store.create_snapshot(description="refused")

    assert refusal.value.reason == "policy_unavailable"
    assert launches == []
    assert store._snapshots == {}
    assert store._counter == 0


@pytest.mark.parametrize("policy", ["disabled", "preferred"])
def test_non_required_policy_preserves_raw_git(
    monkeypatch, install_policy, launches, tmp_path, policy
):
    install_policy(policy=policy, backend_available=True)
    (tmp_path / "example.py").write_text("content\n", encoding="utf-8")
    store = FileSnapshotStore(workspace_root=tmp_path)

    store.create_snapshot(files=["example.py"], description="raw")

    assert len(launches) == 1
    assert launches[0][:2] == ["git", "rev-parse"]


def test_admitted_snapshot_precedence(monkeypatch, install_policy, launches, tmp_path):
    admitted = install_policy(policy="disabled", backend_available=False)
    install_policy(policy="required", backend_available=False)
    (tmp_path / "example.py").write_text("content\n", encoding="utf-8")
    store = FileSnapshotStore(workspace_root=tmp_path, run_policy_snapshot=admitted)

    store.create_snapshot(files=["example.py"], description="admitted")

    assert len(launches) == 1
    assert launches[0][:2] == ["git", "rev-parse"]


def test_current_admitted_required_overrides_reloaded_disabled(
    monkeypatch, install_policy, launches, tmp_path
):
    admitted = install_policy(policy="required", backend_available=True)
    install_policy(policy="disabled", backend_available=True)
    (tmp_path / "example.py").write_text("content\n", encoding="utf-8")
    store = FileSnapshotStore(workspace_root=tmp_path)

    with run_policy_context(admitted):
        store.create_snapshot(files=["example.py"], description="admitted")

    assert len(launches) == 1
    assert launches[0][0] == "bwrap"
