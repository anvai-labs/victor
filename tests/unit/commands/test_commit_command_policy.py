"""The UI /commit command obeys the canonical required-policy gate."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import subprocess as subprocess_module

from victor.config.sandbox_settings import SandboxSettings
from victor.tools.sandbox import backends
from victor.ui.slash.commands.navigation import CommitCommand
from victor.ui.slash.protocol import CommandContext


@pytest.fixture
def launches(monkeypatch):
    commands: list[list[str]] = []

    def _run(command, **_kwargs):
        commands.append(list(command))
        suffix = command[command.index("--") + 1 :] if "--" in command else command
        if "status" in suffix:
            return SimpleNamespace(returncode=0, stdout=" M example.py\n", stderr="")
        if "diff" in suffix:
            return SimpleNamespace(returncode=0, stdout="1 file changed\n", stderr="")
        if "commit" in suffix:
            return SimpleNamespace(returncode=0, stdout="done\n", stderr="")
        if "rev-parse" in suffix:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess_module, "run", _run)
    return commands


@pytest.fixture
def install_policy(monkeypatch):
    def _install(*, policy: str, backend_available: bool) -> SimpleNamespace:
        settings = SimpleNamespace(
            sandbox=SandboxSettings(sandbox_enabled=True, sandbox_policy=policy)
        )
        monkeypatch.setattr(backends.platform, "system", lambda: "Linux")
        monkeypatch.setattr(
            backends.shutil,
            "which",
            lambda name: "/usr/bin/bwrap" if backend_available and name == "bwrap" else None,
        )
        return settings

    return _install


def _context(settings, *, args=None, agent=None):
    chat = AsyncMock(return_value=SimpleNamespace(content="feat: generated message"))
    command_agent = agent or SimpleNamespace(chat=chat)
    return (
        CommandContext(
            console=MagicMock(),
            settings=settings,
            agent=command_agent,
            args=list(args or []),
        ),
        chat,
    )


def _git_suffix(command: list[str]) -> list[str]:
    return command[command.index("--") + 1 :] if "--" in command else command


async def test_required_available_wraps_full_command_workflow_once(
    monkeypatch, launches, install_policy
):
    settings = install_policy(policy="required", backend_available=True)
    resolver_calls = 0
    original = backends.resolve_required_launch

    def _counting(settings_view):
        nonlocal resolver_calls
        resolver_calls += 1
        return original(settings_view)

    import victor.tools.sandbox as sandbox_module

    monkeypatch.setattr(sandbox_module, "resolve_required_launch", _counting)
    ctx, chat = _context(settings)

    await CommitCommand().execute(ctx)

    assert chat.await_count == 1
    assert resolver_calls == 1
    assert len(launches) == 5
    assert {command[0] for command in launches} == {"bwrap"}
    suffixes = [command[command.index("--") + 1 :] for command in launches]
    assert [command[1] for command in suffixes] == [
        "status",
        "diff",
        "add",
        "commit",
        "rev-parse",
    ]


async def test_required_unavailable_refuses_before_git_or_model_work(launches, install_policy):
    settings = install_policy(policy="required", backend_available=False)
    ctx, chat = _context(settings)

    await CommitCommand().execute(ctx)

    assert launches == []
    chat.assert_not_awaited()
    refusal_messages = [
        call.args[0]
        for call in ctx.console.print.call_args_list
        if "Commit policy unavailable" in call.args[0]
    ]
    assert refusal_messages


async def test_unreadable_settings_refuse_before_git_or_model_work(monkeypatch, launches):
    class BrokenSettings:
        @property
        def sandbox(self):
            raise RuntimeError("settings unreadable")

    ctx, chat = _context(BrokenSettings())

    await CommitCommand().execute(ctx)

    assert launches == []
    chat.assert_not_awaited()
    assert any(
        "Commit policy unavailable" in call.args[0] for call in ctx.console.print.call_args_list
    )


@pytest.mark.parametrize("policy", ["disabled", "preferred"])
async def test_non_required_policy_preserves_raw_git(launches, install_policy, policy):
    settings = install_policy(policy=policy, backend_available=True)
    ctx, _chat = _context(settings, args=["fix: provided"])

    await CommitCommand().execute(ctx)

    assert len(launches) == 4
    assert {command[0] for command in launches} == {"git"}
    assert [command[1] for command in launches] == ["status", "add", "commit", "rev-parse"]


async def test_failed_stage_does_not_attempt_commit_or_hash(monkeypatch, launches, install_policy):
    settings = install_policy(policy="required", backend_available=True)
    ctx, _chat = _context(settings, args=["fix: provided"])

    def _run(command, **_kwargs):
        launches.append(list(command))
        if "add" in command:
            return SimpleNamespace(returncode=128, stdout="", stderr="stage failed")
        return SimpleNamespace(returncode=0, stdout=" M example.py\n", stderr="")

    monkeypatch.setattr(subprocess_module, "run", _run)
    await CommitCommand().execute(ctx)

    assert [command[1] for command in map(_git_suffix, launches)] == ["status", "add"]
    assert any("Stage failed" in call.args[0] for call in ctx.console.print.call_args_list)
