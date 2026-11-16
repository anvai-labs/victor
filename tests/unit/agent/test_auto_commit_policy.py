"""Auto-commit git launches obey the canonical required-policy gate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import victor.agent.auto_commit as auto_commit_module
from victor.agent.auto_commit import AutoCommitter
from victor.config.sandbox_settings import SandboxSettings
from victor.runtime.policy_snapshot import RunPolicySnapshot, run_policy_context
from victor.tools.sandbox import backends


@pytest.fixture
def launches(monkeypatch):
    commands: list[list[str]] = []

    def _run(command, **_kwargs):
        commands.append(list(command))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(auto_commit_module.subprocess, "run", _run)
    return commands


def _install_policy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    policy: str,
    backend_available: bool,
) -> RunPolicySnapshot:
    settings = SimpleNamespace(sandbox=SandboxSettings(sandbox_enabled=True, sandbox_policy=policy))
    monkeypatch.setattr("victor.config.settings.get_settings", lambda: settings)
    monkeypatch.setattr(backends.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        backends.shutil,
        "which",
        lambda name: "/usr/bin/bwrap" if backend_available and name == "bwrap" else None,
    )
    return RunPolicySnapshot.from_settings(settings)


def test_required_available_wraps_fixed_git_argv(monkeypatch, launches):
    _install_policy(monkeypatch, policy="required", backend_available=True)

    AutoCommitter()._run_git("status", "--porcelain")

    assert launches[0][0] == "bwrap"
    assert launches[0][-4:] == ["--", "git", "status", "--porcelain"]


def test_required_unavailable_refuses_before_git_spawn(monkeypatch, launches):
    _install_policy(monkeypatch, policy="required", backend_available=False)

    with pytest.raises(auto_commit_module.AutoCommitPolicyRefusal) as refusal:
        AutoCommitter()._run_git("status")

    assert refusal.value.reason == "sandbox_required"
    assert launches == []


def test_unreadable_policy_refuses_before_git_spawn(monkeypatch, launches):
    def _unavailable():
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr("victor.config.settings.get_settings", _unavailable)

    with pytest.raises(auto_commit_module.AutoCommitPolicyRefusal) as refusal:
        AutoCommitter()._run_git("status")

    assert refusal.value.reason == "policy_unavailable"
    assert launches == []


@pytest.mark.parametrize("policy", ["disabled", "preferred"])
def test_non_required_policy_preserves_raw_git_argv(monkeypatch, launches, policy):
    _install_policy(monkeypatch, policy=policy, backend_available=True)

    AutoCommitter()._run_git("status")

    assert launches == [["git", "status"]]


def test_admitted_snapshot_takes_precedence_over_reload(monkeypatch, launches):
    admitted = _install_policy(monkeypatch, policy="disabled", backend_available=False)
    _install_policy(monkeypatch, policy="required", backend_available=False)

    AutoCommitter(run_policy_snapshot=admitted)._run_git("status")

    assert launches == [["git", "status"]]


def test_current_run_snapshot_precedes_mutable_settings(monkeypatch, launches):
    admitted = _install_policy(monkeypatch, policy="required", backend_available=True)
    _install_policy(monkeypatch, policy="disabled", backend_available=True)

    with run_policy_context(admitted):
        AutoCommitter()._run_git("status")

    assert launches[0][0] == "bwrap"


def test_commit_resolves_one_decision_and_wraps_every_git_child(monkeypatch, launches):
    _install_policy(monkeypatch, policy="required", backend_available=True)
    resolver_calls = 0

    def _command(argv, **_kwargs):
        launches.append(list(argv))
        suffix = argv[argv.index("--") + 1 :] if "--" in argv else argv
        if suffix[1:] == ["rev-parse", "--git-dir"]:
            return SimpleNamespace(returncode=0, stdout=".git", stderr="")
        if "commit" in suffix and "-m" in suffix:
            return SimpleNamespace(returncode=0, stdout="[main abc123] committed", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(auto_commit_module.subprocess, "run", _command)
    original_resolver = backends.resolve_required_launch

    def _counting_resolver(settings):
        nonlocal resolver_calls
        resolver_calls += 1
        return original_resolver(settings)

    import victor.tools.sandbox as sandbox_module

    monkeypatch.setattr(sandbox_module, "resolve_required_launch", _counting_resolver)

    result = AutoCommitter().commit_changes(
        files=["example.py"],
        description="AI-assisted changes",
        auto_stage=True,
    )

    assert result.success is True
    assert result.commit_hash == "abc123"
    assert resolver_calls == 1
    assert len(launches) == 3
    assert {command[0] for command in launches} == {"bwrap"}
    suffixes = [command[command.index("--") + 1 :] for command in launches]
    assert [command[1] for command in suffixes] == [
        "rev-parse",
        "add",
        "commit",
    ]
    assert suffixes[0][-1] == "--git-dir"
    assert suffixes[1][-1] == "example.py"
    assert suffixes[2][-1].startswith("chore: AI-assisted changes")


def test_required_unavailable_commit_refuses_before_repository_discovery(monkeypatch, launches):
    _install_policy(monkeypatch, policy="required", backend_available=False)

    with pytest.raises(auto_commit_module.AutoCommitPolicyRefusal):
        AutoCommitter().commit_changes(files=["example.py"], description="change")

    assert launches == []


def test_undo_resolves_one_decision_and_wraps_reset(monkeypatch, launches):
    _install_policy(monkeypatch, policy="required", backend_available=True)
    resolver_calls = 0
    original_resolver = backends.resolve_required_launch

    def _counting_resolver(settings):
        nonlocal resolver_calls
        resolver_calls += 1
        return original_resolver(settings)

    import victor.tools.sandbox as sandbox_module

    monkeypatch.setattr(sandbox_module, "resolve_required_launch", _counting_resolver)

    assert AutoCommitter().undo_last_commit(keep_changes=False)

    assert resolver_calls == 1
    assert len(launches) == 1
    assert launches[0][0] == "bwrap"
    suffix = launches[0][launches[0].index("--") + 1 :]
    assert suffix[:2] == ["git", "reset"]


def test_commit_info_resolves_one_decision_for_both_reads(monkeypatch, launches):
    _install_policy(monkeypatch, policy="required", backend_available=True)
    resolver_calls = 0
    original_resolver = backends.resolve_required_launch

    def _counting_resolver(settings):
        nonlocal resolver_calls
        resolver_calls += 1
        return original_resolver(settings)

    import victor.tools.sandbox as sandbox_module

    monkeypatch.setattr(sandbox_module, "resolve_required_launch", _counting_resolver)

    def _command(argv, **_kwargs):
        launches.append(list(argv))
        if len(launches) == 1:
            return SimpleNamespace(
                returncode=0,
                stdout="hash|subject|author|email|2026-11-15",
                stderr="",
            )
        return SimpleNamespace(
            returncode=0,
            stdout=AutoCommitter.COMMIT_SIGNATURE,
            stderr="",
        )

    monkeypatch.setattr(auto_commit_module.subprocess, "run", _command)

    info = AutoCommitter().get_last_commit_info()
    assert launches

    assert info is not None
    assert info["is_victor"] is True
    assert resolver_calls == 1
    assert len(launches) == 2
    assert {command[0] for command in launches} == {"bwrap"}
