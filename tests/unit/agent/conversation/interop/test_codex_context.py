"""Codex install admission is bounded; archives and explicit handoffs are distinct."""

import json
import os

import pytest

from victor.agent.conversation.interop import (
    ConversionReport,
    TranscriptFormat,
    export_session,
    import_transcript,
    install_session,
)
from victor.agent.conversation.interop import writers
from victor.agent.conversation.session_model import ConversationSession
from victor.agent.conversation.types import ConversationMessage


def session(text="original history"):
    return ConversationSession(
        session_id="context_test", messages=[ConversationMessage(role="user", content=text)]
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    base = tmp_path / ".codex" / "sessions"
    base.mkdir(parents=True)
    return base


def test_oversized_install_leaves_no_rollout(store):
    source = session()
    source.messages = [ConversationMessage(role="user", content="x" * 90_000) for _ in range(3)]
    with pytest.raises(ValueError, match="context-file"):
        install_session(source, "codex")
    assert not list(store.rglob("*.jsonl"))
    assert len(source.messages) == 3


def test_archive_keeps_items_and_reports_context_warning(tmp_path):
    source = session()
    source.messages = [ConversationMessage(role="user", content="x" * 90_000) for _ in range(3)]
    report = ConversionReport()
    out = export_session(source, TranscriptFormat.CODEX_ROLLOUT, tmp_path / "archive", report)
    restored = import_transcript(out)
    assert [m.content for m in restored.session.messages] == [m.content for m in source.messages]
    assert any("context-file" in w for w in report.warnings)
    assert "Warning:" in report.render_summary()


def test_explicit_handoff_preserves_source_and_records_losses(store, tmp_path):
    source = session("original private tool history")
    context = tmp_path / "handoff with spaces.md"
    context.write_text("Historical task: use af_heart; fix agenda and dynamic diagrams.")
    report = ConversionReport()
    out = install_session(source, "codex", report, context_file=context)
    assert report.target_format == "codex"
    restored = import_transcript(out)
    assert len(restored.session.messages) == 1
    text = restored.session.messages[0].content
    assert "af_heart" in text and "historical" in text.lower()
    assert "authorization" in text and "context_test" in text
    assert "original private tool history" not in text
    assert source.messages[0].content == "original private tool history"
    assert any("1 source messages" in w and "SHA256" in w for w in report.warnings)


@pytest.mark.parametrize("kind", ["empty", "oversized", "invalid", "directory"])
def test_invalid_context_is_rejected_without_install(store, tmp_path, kind):
    context = tmp_path / "context"
    if kind == "directory":
        context.mkdir()
    else:
        context.write_bytes({"empty": b"  ", "oversized": b"x" * 65_537, "invalid": b"\xff"}[kind])
    with pytest.raises((ValueError, OSError, UnicodeError)):
        install_session(session(), "codex", context_file=context)
    assert not list(store.rglob("*.jsonl"))


def test_context_file_is_codex_only(tmp_path):
    context = tmp_path / "context"
    context.write_text("summary")
    with pytest.raises(ValueError, match="codex"):
        export_session(
            session(), TranscriptFormat.CLAUDE_CODE, tmp_path / "out", context_file=context
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX named pipes only")
def test_context_fifo_is_rejected_without_blocking(store, tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular"):
        install_session(session(), "codex", context_file=fifo)
    assert not list(store.rglob("*.jsonl"))


@pytest.mark.parametrize("text", ["x" * 65_536, "界" * 21_845], ids=["ascii", "multibyte"])
def test_context_file_allows_exact_utf8_boundary(store, tmp_path, text):
    context = tmp_path / "context"
    context.write_text(text, encoding="utf-8")
    out = install_session(session(), "codex", context_file=context)
    assert text in import_transcript(out).session.messages[0].content


@pytest.mark.parametrize("content", ['"' * 10, "界" * 10])
def test_budget_counts_serialized_utf8_and_exact_boundary(tmp_path, content):
    report = ConversionReport()
    full = export_session(session(content), TranscriptFormat.CODEX_ROLLOUT, tmp_path / "full")
    payloads = [
        json.loads(line)["payload"]
        for line in full.read_text().splitlines()
        if json.loads(line)["type"] == "response_item"
    ]
    size = sum(len(json.dumps(p, ensure_ascii=False).encode("utf-8")) for p in payloads)
    writers.write_codex_rollout(
        session(content), tmp_path / "equal", report, max_context_bytes=size
    )
    with pytest.raises(ValueError, match="context-file"):
        writers.write_codex_rollout(
            session(content), tmp_path / "below", report, max_context_bytes=size - 1
        )
    assert not (tmp_path / "below").exists()


@pytest.mark.parametrize("role", ["call", "output"])
def test_tools_are_counted_in_install_budget(store, role):
    source = session()
    source.messages = (
        [
            ConversationMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call1",
                        "name": "write",
                        "arguments": '{"content":"' + "x" * 140_000 + '"}',
                    }
                ],
            )
        ]
        if role == "call"
        else [ConversationMessage(role="tool", content="x" * 140_000, tool_call_id="call1")]
    )
    with pytest.raises(ValueError, match="context-file"):
        install_session(source, "codex")
    assert not list(store.rglob("*.jsonl"))


def test_ui_events_do_not_duplicate_inference_history(tmp_path):
    source = session("hello")
    source.messages.append(ConversationMessage(role="assistant", content="world"))
    out = export_session(source, TranscriptFormat.CODEX_ROLLOUT, tmp_path / "out")
    entries = [json.loads(line) for line in out.read_text().splitlines()]
    events = [e["payload"] for e in entries if e["type"] == "event_msg"]
    assert [e["message"] for e in events if e["type"] in ("user_message", "agent_message")] == [
        "hello",
        "world",
    ]
    assert [m.content for m in import_transcript(out).session.messages] == ["hello", "world"]
