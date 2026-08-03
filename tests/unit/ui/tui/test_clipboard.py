"""Unit tests for the OS-clipboard reader (pure, injected runner/candidates)."""

from __future__ import annotations

from typing import List, Sequence

from victor.ui.tui.clipboard import read_clipboard


def _runner_returning(mapping: dict) -> object:
    """A runner that returns mapping[argv[0]] or raises if the key is absent."""

    def run(argv: Sequence[str]) -> str:
        key = argv[0]
        if key not in mapping:
            raise RuntimeError(f"no such tool: {key}")
        return mapping[key]

    return run


def test_returns_first_successful_backend() -> None:
    cands = [(["pbpaste"], False), (["xclip", "-o"], False)]
    runner = _runner_returning({"pbpaste": "hello"})
    assert read_clipboard(runner=runner, candidates=cands) == "hello"


def test_falls_through_to_next_backend_on_failure() -> None:
    cands = [(["pbpaste"], False), (["xclip", "-o"], False)]
    runner = _runner_returning({"xclip": "from-xclip"})  # pbpaste missing → raises
    assert read_clipboard(runner=runner, candidates=cands) == "from-xclip"


def test_normalizes_crlf_line_endings() -> None:
    cands = [(["powershell.exe"], False)]
    runner = _runner_returning({"powershell.exe": "line1\r\nline2\r\n"})
    assert read_clipboard(runner=runner, candidates=cands) == "line1\nline2\n"


def test_strips_one_trailing_newline_when_flagged() -> None:
    cands = [(["powershell.exe"], True)]
    runner = _runner_returning({"powershell.exe": "message\r\n"})
    assert read_clipboard(runner=runner, candidates=cands) == "message"


def test_preserves_interior_and_multiple_trailing_newlines_without_flag() -> None:
    cands = [(["pbpaste"], False)]
    runner = _runner_returning({"pbpaste": "a\nb\n\n"})
    assert read_clipboard(runner=runner, candidates=cands) == "a\nb\n\n"


def test_empty_output_skips_backend() -> None:
    cands = [(["pbpaste"], False), (["xclip", "-o"], False)]
    runner = _runner_returning({"pbpaste": "", "xclip": "real"})
    assert read_clipboard(runner=runner, candidates=cands) == "real"


def test_returns_none_when_no_backend_succeeds(monkeypatch) -> None:
    # No CLI candidate works and pyperclip is forced unavailable.
    monkeypatch.setattr("victor.ui.tui.clipboard._read_pyperclip", lambda: None)
    called: List[Sequence[str]] = []

    def run(argv: Sequence[str]) -> str:
        called.append(argv)
        raise RuntimeError("nope")

    assert read_clipboard(runner=run, candidates=[(["pbpaste"], False)]) is None
    assert called == [["pbpaste"]]
