from __future__ import annotations

import pytest

from victor_codegraph import ChunkConfig, chunk


def test_fallback_caps_a_single_oversized_line():
    config = ChunkConfig(max_chunk_tokens=10, chunk_overlap_tokens=2, chars_per_token=1)
    chunks = chunk("x" * 100, language="text", file_path="notes.txt", config=config)
    assert len(chunks) > 1
    assert all(len(c.text) <= config.max_chunk_chars for c in chunks)


def test_python_chunks_report_exact_nonzero_byte_spans():
    source = "prefix = 1\n\ndef f():\n    return 1\n"
    function = next(
        c for c in chunk(source, file_path="m.py") if c.metadata.get("simple_name") == "f"
    )
    assert function.start_pos == source.index("def f")
    assert function.end_pos == function.start_pos + len(function.text.encode())


def test_chunking_preserves_module_level_semantic_content_without_container_duplication():
    source = """import os
GLOBAL = configure()

class Service:
    def run(self):
        return work()

main()
"""
    chunks = chunk(source, file_path="m.py", config=ChunkConfig(max_chunk_tokens=1000))
    text = "\n".join(c.text for c in chunks)
    assert "import os" in text
    assert "GLOBAL = configure()" in text
    assert "main()" in text
    assert sum(c.text.count("return work()") for c in chunks) == 1


def test_include_tests_and_language_filters_are_effective():
    assert (
        chunk(
            "def test_it():\n    pass\n",
            file_path="test_demo.py",
            config=ChunkConfig(include_tests=False),
        )
        == []
    )
    assert (
        chunk(
            "def f():\n    pass\n",
            file_path="m.py",
            config=ChunkConfig(languages=["rust"]),
        )
        == []
    )


def test_custom_token_counter_is_a_hard_limit():
    source = "def f():\n" + "\n".join(f"    value_{i} = {i}" for i in range(30))
    config = ChunkConfig(
        max_chunk_tokens=8,
        chunk_overlap_tokens=0,
        token_counter=lambda text: len(text.split()),
    )
    chunks = chunk(source, file_path="m.py", config=config)
    assert all(config.estimate_tokens(c.text) <= config.max_chunk_tokens for c in chunks)


def test_impossible_token_counter_fails_instead_of_emitting_over_budget_chunk():
    config = ChunkConfig(max_chunk_tokens=1, token_counter=lambda text: 2 if text else 0)
    with pytest.raises(ValueError, match="cannot fit"):
        chunk("x", language="text", file_path="notes.txt", config=config)
