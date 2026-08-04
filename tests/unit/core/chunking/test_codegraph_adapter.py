from __future__ import annotations

import pytest

from victor.core.chunking.codegraph_adapter import chunk_with_codegraph


def test_projection_translates_utf8_byte_spans_to_character_offsets() -> None:
    pytest.importorskip("victor_codegraph")
    content = "π = 3\n\ndef compute():\n    return π\n"

    chunks = chunk_with_codegraph(
        content,
        file_path="unicode.py",
        language="python",
        max_chunk_size=80,
        chunk_overlap=0,
    )

    assert chunks
    assert all(content[chunk.start_char : chunk.end_char] == chunk.content for chunk in chunks)
    compute = next(chunk for chunk in chunks if chunk.symbol_name == "compute")
    assert compute.metadata["strategy"] == "victor_codegraph_v2"


def test_dependency_failure_is_bounded_by_the_same_adapter(monkeypatch) -> None:
    codegraph = pytest.importorskip("victor_codegraph")
    monkeypatch.setattr(
        codegraph,
        "chunk",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    content = "first line\nsecond line\nthird line\n"

    chunks = chunk_with_codegraph(
        content,
        file_path="fallback.py",
        language="python",
        max_chunk_size=12,
        chunk_overlap=2,
    )

    assert chunks
    assert all(len(chunk.content) <= 12 for chunk in chunks)
    assert all(chunk.metadata["strategy"] == "plain_code_fallback" for chunk in chunks)
