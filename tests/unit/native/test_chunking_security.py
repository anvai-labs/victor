"""Chunking must terminate, cover input, and retain character offset semantics."""

import json
import os
import subprocess
import sys

import pytest

from victor.native.python.chunker import PythonTextChunker
from victor.processing.native import chunking


@pytest.mark.parametrize("text", ["", "abc"])
@pytest.mark.parametrize("size,overlap", [(0, 0), (1, 1), (2, 3), (2, -1)])
@pytest.mark.parametrize("native_available", [False, True])
def test_invalid_parameters_rejected_before_backend(
    text, size, overlap, native_available, monkeypatch
):
    monkeypatch.setattr(chunking, "_NATIVE_AVAILABLE", native_available)
    monkeypatch.setattr(chunking, "_native", None)
    for function in (
        chunking.chunk_by_chars,
        chunking.chunk_by_sentences,
        chunking.chunk_by_paragraphs,
        PythonTextChunker().chunk_with_overlap,
    ):
        with pytest.raises(ValueError):
            function(text, size, overlap)


@pytest.mark.parametrize("text", ["a\nbbbbbbbb", "ééé", "é漢\n🙂xy\n尾巴", "ab\ncdefghijkl"])
def test_line_chunks_cover_every_character_and_advance(text):
    chunker = PythonTextChunker()
    for size in range(1, 9):
        for overlap in range(size):
            chunks = chunker.chunk_with_overlap(text, size, overlap)
            assert len(chunks) <= len(text)
            previous_start = -1
            covered = 0
            for chunk in chunks:
                assert previous_start < chunk.start_offset <= covered
                assert chunk.start_offset < chunk.end_offset
                assert chunk.text == text[chunk.start_offset : chunk.end_offset]
                assert chunk.overlap_prev == max(0, covered - chunk.start_offset)
                assert chunk.start_line == text[: chunk.start_offset].count("\n") + 1
                assert chunk.end_line == text[: chunk.end_offset - 1].count("\n") + 1
                previous_start = chunk.start_offset
                covered = chunk.end_offset
            assert covered == len(text)


@pytest.mark.parametrize("text", ["", "abc"])
@pytest.mark.parametrize(
    "size,overlap,error",
    [
        (0, 0, ValueError),
        (1, 1, ValueError),
        (2, -1, ValueError),
        (True, 0, TypeError),
        (2, False, TypeError),
        (2.5, 0, TypeError),
        (2, "0", TypeError),
    ],
)
def test_rust_and_python_wrappers_validate_before_empty_shortcut(
    text, size, overlap, error, monkeypatch
):
    native = pytest.importorskip("victor_native")
    from victor.native.rust.chunker import RustTextChunker

    def unexpected_dispatch(*args):
        pytest.fail("invalid arguments reached the native implementation")

    monkeypatch.setattr(native, "chunk_with_overlap", unexpected_dispatch)
    for chunker in (PythonTextChunker(), RustTextChunker()):
        with pytest.raises(error):
            chunker.chunk_with_overlap(text, size, overlap)


def test_unicode_line_helpers_and_chunk_offsets_agree_between_backends():
    pytest.importorskip("victor_native")
    from victor.native.rust.chunker import RustTextChunker

    python_chunker, rust_chunker = PythonTextChunker(), RustTextChunker()
    for text in ("é\na\nb", "é\n", "é漢\n🙂xy\n尾巴", ""):
        assert rust_chunker.find_line_boundaries(text) == python_chunker.find_line_boundaries(text)
        for offset in range(-2, len(text) + 3):
            assert rust_chunker.line_at_offset(text, offset) == python_chunker.line_at_offset(
                text, offset
            )
        for chunk in rust_chunker.chunk_with_overlap(text, 2, 0):
            assert rust_chunker.line_at_offset(text, chunk.start_offset) == chunk.start_line
            assert rust_chunker.line_at_offset(text, chunk.end_offset - 1) == chunk.end_line


def test_native_chunking_ffi_rejects_invalid_inputs_and_preserves_unicode():
    native = pytest.importorskip("victor_native")
    # A subprocess deadline makes a reintroduced native GIL-holding loop fail
    # this test instead of hanging the entire test worker.
    script = """
import json
import victor_native as native
for name in ('chunk_by_chars', 'chunk_by_sentences', 'chunk_by_paragraphs', 'chunk_with_overlap'):
    for size, overlap in ((0, 0), (1, 1), (2, 3)):
        try:
            getattr(native, name)('abc', size, overlap)
        except ValueError:
            pass
        else:
            raise AssertionError((name, size, overlap))
for text in ('a\\nbbbbbbbb', 'ééé', 'é漢\\n🙂xy\\n尾巴'):
    for size in range(1, 9):
        for overlap in range(size):
            chunks = native.chunk_with_overlap(text, size, overlap)
            assert len(chunks) <= len(text)
            previous_start, covered = -1, 0
            for chunk in chunks:
                assert previous_start < chunk.start_offset <= covered
                assert chunk.text == text[chunk.start_offset:chunk.end_offset]
                assert chunk.overlap_prev == max(0, covered - chunk.start_offset)
                previous_start, covered = chunk.start_offset, chunk.end_offset
            assert covered == len(text)
assert native.chunk_by_paragraphs('éé\\n\\nx', 4, 1) == ['éé', 'é\\n\\nx']
assert native.chunk_by_sentences('ééééééa.aaaaa.', 4, 1) == ['ééééééa.aaaaa.']
large = 'Sentence. ' * 200_000
assert native.chunk_by_sentences(large, len(large) + 1, 0) == [large.strip()]
print(json.dumps({'version': native.__version__}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=15, env=env
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["version"] == native.__version__
