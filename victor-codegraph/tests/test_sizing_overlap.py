"""Overlap correctness and no-data-loss guarantees for body-splitting.

Pins the fix for the minified-line hard-cut: a single line over the char budget
used to be truncated (``ln[:max_chars]``), silently dropping the remainder.
"""

from __future__ import annotations

from victor_codegraph.config import ChunkConfig
from victor_codegraph.model import CodeSymbol, CodeSymbolType, SourceLocation
from victor_codegraph.sizing import chunks_for_symbol


def _cfg(max_tokens: int = 40, overlap_tokens: int = 8) -> ChunkConfig:
    # Small budgets so tests exercise splitting with tiny fixtures.
    return ChunkConfig(max_chunk_tokens=max_tokens, chunk_overlap_tokens=overlap_tokens)


def _symbol(source: str, start_line: int = 10) -> CodeSymbol:
    line_count = source.count("\n") + (0 if source.endswith("\n") else 1)
    return CodeSymbol(
        id="sym1",
        symbol_type=CodeSymbolType.FUNCTION,
        fully_qualified_name="f.py::big",
        simple_name="big",
        location=SourceLocation(
            file_path="f.py",
            start_line=start_line,
            end_line=start_line + line_count - 1,
        ),
        source_code=source,
        language="python",
    )


def _multiline_source(n_lines: int = 120) -> str:
    return "".join(f"    value_{i} = compute_{i}(x)\n" for i in range(n_lines))


def test_no_chunk_exceeds_budget():
    cfg = _cfg()
    chunks = chunks_for_symbol(_symbol(_multiline_source()), cfg)
    assert len(chunks) > 1
    assert all(len(c.text) <= cfg.max_chunk_chars for c in chunks)


def test_every_source_line_appears_in_some_chunk():
    cfg = _cfg()
    source = _multiline_source()
    chunks = chunks_for_symbol(_symbol(source), cfg)
    joined = [c.text for c in chunks]
    for line in source.splitlines():
        assert any(line in text for text in joined), f"line lost: {line!r}"


def test_consecutive_windows_overlap_by_whole_lines():
    # Overlap is whole-line granular: give it budget for at least one line.
    cfg = _cfg(max_tokens=40, overlap_tokens=16)
    chunks = chunks_for_symbol(_symbol(_multiline_source()), cfg)
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:]):
        # The next window begins with whole lines that terminate the previous one:
        # some non-empty line-suffix of `a` is a prefix of `b`, within budget.
        a_lines = a.text.splitlines(keepends=True)
        overlap = 0
        for k in range(len(a_lines)):
            suffix = "".join(a_lines[k:])
            if b.text.startswith(suffix):
                overlap = len(suffix)
                break
        assert overlap > 0
        assert overlap <= cfg.chunk_overlap_chars


def test_body_split_window_line_metadata():
    cfg = _cfg()
    sym = _symbol(_multiline_source(), start_line=10)
    chunks = chunks_for_symbol(sym, cfg)
    for c in chunks:
        start = c.metadata["start_line"]
        end = c.metadata["end_line"]
        assert start >= 10
        assert end >= start
        # end_line reflects THIS window, not the symbol's overall end.
        spanned = c.text.count("\n") + (0 if c.text.endswith("\n") else 1)
        assert end == start + spanned - 1
    assert chunks[0].metadata["start_line"] == 10


def test_minified_line_is_not_truncated():
    """A single line 10x over budget must be fully preserved across chunks."""
    cfg = _cfg()
    long_line = "x=1;" * (cfg.max_chunk_chars * 10 // 4)
    source = long_line + "\n"
    chunks = chunks_for_symbol(_symbol(source), cfg)

    assert len(chunks) > 1
    assert all(len(c.text) <= cfg.max_chunk_chars for c in chunks)
    assert all(c.metadata.get("is_line_split") for c in chunks)

    # Consecutive pieces advance by (max - overlap); every char offset is covered
    # and the full line (incl. trailing newline) is reconstructible.
    full = long_line + "\n"
    step = cfg.max_chunk_chars - cfg.chunk_overlap_chars
    buf = [""] * len(full)
    for idx, c in enumerate(chunks):
        start = idx * step
        for j, ch in enumerate(c.text):
            buf[start + j] = ch
    assert "".join(buf) == full


def test_minified_line_mixed_with_normal_lines():
    cfg = _cfg()
    long_line = "y=2;" * (cfg.max_chunk_chars // 2)
    source = "def big():\n" + f"    a = {long_line}\n" + "    return a\n"
    sym = _symbol(source)
    chunks = chunks_for_symbol(sym, cfg)
    text_all = "".join(c.text for c in chunks)
    assert "def big():" in text_all
    assert "return a" in text_all
    # The long middle line's tail must not be lost.
    assert text_all.count("y=2;") >= source.count("y=2;")


def test_small_symbol_single_chunk_has_no_line_split_flag():
    cfg = _cfg(max_tokens=512, overlap_tokens=64)
    sym = _symbol("def small():\n    return 1\n")
    chunks = chunks_for_symbol(sym, cfg)
    assert len(chunks) == 1
    assert "is_line_split" not in chunks[0].metadata
    assert "is_body_split" not in chunks[0].metadata
