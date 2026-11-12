"""CLI and slash exports share bounded Codex handoff admission and diagnostics."""

from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from victor.agent.conversation.session_model import ConversationSession
from victor.agent.conversation.types import ConversationMessage
from victor.ui.commands import sessions
from victor.ui.slash.commands import import_export
from victor.ui.slash.protocol import CommandContext


@pytest.fixture
def export_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".codex" / "sessions").mkdir(parents=True)
    source = ConversationSession(
        session_id="active_session",
        messages=[ConversationMessage(role="user", content="x" * 90_000) for _ in range(3)],
    )
    backend = SimpleNamespace(
        list_session_summaries=lambda **kw: [SimpleNamespace(session_id=source.session_id)],
        load_session=lambda sid: {
            "metadata": {"session_id": sid},
            "conversation": {
                "messages": [{"role": "user", "content": m.content} for m in source.messages]
            },
        },
    )
    monkeypatch.setattr(sessions, "_get_session_backend", lambda: backend)
    loaded_ids = []

    def load(ctx, session_id):
        loaded_ids.append(session_id)
        return source

    monkeypatch.setattr(import_export, "_load_session_ir", load)
    context = tmp_path / "handoff with spaces.md"
    context.write_text("Historical notes: af_heart, full agenda and dynamic diagrams.")
    return tmp_path, source, context, loaded_ids


def test_cli_handoff_uses_explicit_context_and_truthful_count(export_env):
    root, _, context, _ = export_env
    result = CliRunner().invoke(
        sessions.sessions_app, ["handoff", "codex", "--context-file", str(context)]
    )
    assert result.exit_code == 0, result.output
    assert "Installed 1 message(s)" in result.output
    assert "3 source messages" in result.output and "SHA256" in result.output
    assert len(list((root / ".codex").rglob("*.jsonl"))) == 1


def test_cli_oversized_install_is_not_success(export_env):
    root, _, _, _ = export_env
    result = CliRunner().invoke(sessions.sessions_app, ["export", "--format", "codex", "--install"])
    assert result.exit_code == 1
    assert "context-file" in result.output
    assert "Resume there" not in result.output and "Installed" not in result.output
    assert not list((root / ".codex").rglob("*.jsonl"))


def test_cli_archive_warns_but_keeps_export(export_env):
    root, _, _, _ = export_env
    out = root / "archive.jsonl"
    result = CliRunner().invoke(
        sessions.sessions_app, ["export", "--format", "codex", "-o", str(out)]
    )
    assert result.exit_code == 0 and out.is_file()
    assert "archive" in result.output and "Warning:" in result.output
    assert "context-file" in result.output


@pytest.mark.parametrize("fmt", ["json", "victor"])
def test_cli_install_cannot_reinterpret_json_as_codex(export_env, fmt):
    root, _, _, _ = export_env
    result = CliRunner().invoke(sessions.sessions_app, ["export", "--format", fmt, "--install"])
    assert result.exit_code == 1 and "--install requires" in result.output
    assert not list((root / ".codex").rglob("*.jsonl"))


@pytest.mark.parametrize("handoff", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("flags_first", [False, True])
def test_slash_context_options_are_not_session_ids(export_env, handoff, explicit, flags_first):
    _, _, context, loaded = export_env
    positional = (["codex"] if handoff else []) + (["explicit_session"] if explicit else [])
    flags = ["--context-file", str(context)] + (
        [] if handoff else ["--format", "codex", "--install"]
    )
    stream = StringIO()
    ctx = CommandContext(
        console=Console(file=stream, width=180),
        settings=SimpleNamespace(),
        agent=SimpleNamespace(active_session_id="active_session"),
        args=flags + positional if flags_first else positional + flags,
    )
    command = import_export.HandoffCommand() if handoff else import_export.ExportCommand()
    command.execute(ctx)
    assert loaded == ["explicit_session" if explicit else "active_session"]
    assert "failed" not in stream.getvalue().lower()
    assert "3 source messages" in stream.getvalue()


@pytest.mark.parametrize("flag", ["--format", "--context-file", "--out"])
def test_slash_missing_option_value_does_not_export(export_env, flag):
    root, _, _, loaded = export_env
    stream = StringIO()
    ctx = CommandContext(console=Console(file=stream), settings=SimpleNamespace(), args=[flag])
    import_export.ExportCommand().execute(ctx)
    assert "requires a value" in stream.getvalue()
    assert not loaded and not list((root / ".codex").rglob("*.jsonl"))


def test_slash_inline_context_value_keeps_active_session(export_env):
    _, _, context, loaded = export_env
    stream = StringIO()
    ctx = CommandContext(
        console=Console(file=stream, width=180),
        settings=SimpleNamespace(),
        agent=SimpleNamespace(active_session_id="active_session"),
        args=["--format=codex", f"--context-file={context}", "--install"],
    )
    import_export.ExportCommand().execute(ctx)
    assert loaded == ["active_session"] and "Installed into codex store" in stream.getvalue()


def test_slash_archive_suppresses_resume_hint(export_env):
    root, _, _, _ = export_env
    stream = StringIO()
    ctx = CommandContext(
        console=Console(file=stream, width=180),
        settings=SimpleNamespace(),
        agent=SimpleNamespace(active_session_id="active_session"),
        args=["--format", "codex", "--out", str(root / "archive")],
    )
    import_export.ExportCommand().execute(ctx)
    assert "Warning:" in stream.getvalue() and "archive" in stream.getvalue()
    assert "codex resume" not in stream.getvalue()


def test_slash_handoff_keeps_existing_conversion_loss_report(export_env):
    root, source, _, _ = export_env
    (root / ".claude" / "projects").mkdir(parents=True)
    source.messages = [ConversationMessage(role="user", content="x" * 200_001)]
    stream = StringIO()
    ctx = CommandContext(
        console=Console(file=stream, width=180),
        settings=SimpleNamespace(),
        agent=SimpleNamespace(active_session_id=source.session_id),
        args=["claude"],
    )
    import_export.HandoffCommand().execute(ctx)
    assert "1 truncations" in stream.getvalue()
