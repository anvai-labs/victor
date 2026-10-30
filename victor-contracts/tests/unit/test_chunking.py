"""Golden parity tests for ``victor_contracts.chunking``.

The expected spans were produced by running the pre-0.12.0 copies of this
logic, ``victor.framework.ingestion.chunker.BaseChunker`` (and, identically,
``victor_rag.chunker.DocumentChunker``), on the inputs built below. Each span is
pinned as ``(start_char, end_char, sha256(text)[:16])`` so the SDK core is
proven byte-for-byte equal to both callers it replaces.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

import pytest

from victor_contracts.chunking import ChunkingConfig, ChunkSpan, TextChunker

_SMALL: dict[str, Any] = {
    "chunk_size": 120,
    "chunk_overlap": 20,
    "min_chunk_size": 30,
    "max_chunk_size": 200,
}


def _prose(sentences: int) -> str:
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    out = []
    for i in range(sentences):
        n = 4 + (i * 3) % 7
        body = " ".join(words[(i + k) % len(words)] for k in range(n))
        out.append(body.capitalize() + (".", "!", "?")[i % 3])
    return " ".join(out)


def _markdown() -> str:
    return (
        "# Title\n\nIntro paragraph that is long enough to survive the minimum size. "
        + _prose(2)
        + "\n\n## Tiny\n\nshort\n\n## Large section\n\n"
        + _prose(30)
        + "\n\n### Closing\n\n"
        + _prose(5)
        + "\n"
    )


def _code() -> str:
    preamble = (
        "import os\nimport sys\n\n# module preamble comment "
        + "x" * 60
        + "\n"
        + ("CONSTANT_VALUE = 42  # explanatory trailing comment for padding\n" * 3)
    )
    small_fn = "def tiny():\n    return 1\n\n"
    medium_fn = (
        "def medium(a, b):\n"
        + "".join(f"    value_{i} = a + b * {i}  # step {i}\n" for i in range(4))
        + "    return a\n\n"
    )
    long_fn = (
        "async def long_running(task):\n"
        + "".join(
            f"    await task.step({i})  # long body line number {i}. Done.\n" for i in range(12)
        )
        + "    return task\n\n"
    )
    cls = (
        "class Widget:\n    def __init__(self):\n        self.parts = []\n"
        + "".join(f"        self.parts.append('part-{i}')\n" for i in range(4))
        + "\n"
    )
    rust = "fn helper(x: i32) -> i32 {\n" + "    x + 1 // add one to the input value\n" * 4 + "}\n"
    return preamble + small_fn + medium_fn + long_fn + cls + rust


CASES: dict[str, tuple[dict[str, Any], str, str]] = {
    "text_small": (_SMALL, "text", _prose(25)),
    "text_no_sentence_bounds": (
        dict(_SMALL, respect_sentence_boundaries=False),
        "text",
        _prose(25),
    ),
    "text_default_config_long": ({}, "text", _prose(400)),
    "text_non_ascii": (_SMALL, "text", ("Grüße aus München. Ünïcödé façade naïve café! " * 12)),
    "text_below_min": (_SMALL, "text", "Too short."),
    "markdown_headers": (_SMALL, "markdown", _markdown()),
    "markdown_no_headers": (_SMALL, "markdown", _prose(12)),
    "code_defs_and_classes": (_SMALL, "code", _code()),
    "code_no_defs": (_SMALL, "code", "x = 1\n" * 60),
    "text_sentence_search_window": (
        {"chunk_size": 120, "chunk_overlap": 0, "min_chunk_size": 10, "max_chunk_size": 200},
        "text",
        "A" * 20 + ". " + "b" * 200 + ". " + "c" * 40,
    ),
    "code_preamble_at_min_size": (
        {"chunk_size": 120, "chunk_overlap": 20, "min_chunk_size": 30, "max_chunk_size": 200},
        "code",
        "#" * 30 + "\ndef f():\n    return " + "1 + " * 10 + "1\n",
    ),
    "markdown_section_at_min_size": (
        _SMALL,
        "markdown",
        "# " + "x" * 28 + "\n\n# " + "y" * 40 + "\n",
    ),
    "code_definition_at_start": (
        _SMALL,
        "code",
        "def first():\n    return "
        + "2 * " * 12
        + "2\n\nclass Second:\n    value = "
        + "3 + " * 12
        + "3\n",
    ),
    "code_blank_preamble": (
        _SMALL,
        "code",
        " " * 40 + "\n\ndef body():\n    return " + "4 - " * 12 + "4\n",
    ),
    "empty": (_SMALL, "text", ""),
    "whitespace_only": (_SMALL, "text", " \n\t " * 50),
    "markdown_empty": (_SMALL, "markdown", ""),
    "code_empty": (_SMALL, "code", ""),
}


# Generated from victor.framework.ingestion.chunker.BaseChunker (develop 6dd1b0e8d).
GOLDEN: dict[str, tuple[int, list[tuple[int, int, str]]]] = {
    "text_small": (
        10,
        [
            (0, 160, "9505adb3b0940854"),
            (140, 289, "035ae45c7895c424"),
            (269, 412, "fcc3418e6d99f29c"),
            (392, 531, "1157020621a79e8b"),
            (511, 643, "a07ac03b41dcea9a"),
            (623, 789, "436bd60cc5b9a63a"),
            (769, 930, "d411bff45af3af56"),
            (910, 1025, "239b4fe85e7c0d0b"),
        ],
    ),
    "text_no_sentence_bounds": (
        10,
        [
            (0, 120, "da0cf71216b2380e"),
            (100, 220, "41f784e1b9cec8da"),
            (200, 320, "3fadfd9772e2c65e"),
            (300, 420, "03b2f2119e3392b1"),
            (400, 520, "ffb9d3ad26a61076"),
            (500, 620, "06ae57d01512535b"),
            (600, 720, "6696ebd54d33adb8"),
            (700, 820, "e8561cacedc4a10d"),
            (800, 920, "86c372ea090be910"),
            (900, 1020, "c6e9ff23b8625885"),
        ],
    ),
    "text_default_config_long": (
        13,
        [
            (0, 1366, "bfb0d85c47b7fab0"),
            (1232, 2623, "06905e2dbf67beab"),
            (2489, 3875, "c84145c42a2a03dd"),
            (3741, 5121, "70bc110c301fac61"),
            (4987, 6377, "82271a507f9b196d"),
            (6243, 7631, "bba5cebd9a9ffa77"),
            (7497, 8872, "6ee8a86bc5c002af"),
            (8738, 10128, "4a31cf0ab86194fb"),
            (9994, 11388, "3b96c0299991ab05"),
            (11254, 12628, "555065b45a6d0a8d"),
            (12494, 13884, "fe489aa33eec6408"),
            (13750, 15135, "5954a7f425a09e36"),
            (15001, 16385, "1fda33d664171099"),
            (16251, 16482, "b558440d776838fa"),
        ],
    ),
    "text_non_ascii": (
        5,
        [
            (0, 157, "fdda70ee1a38dfb4"),
            (137, 295, "fdda70ee1a38dfb4"),
            (275, 433, "fdda70ee1a38dfb4"),
            (413, 552, "5936205899847ef8"),
        ],
    ),
    "text_below_min": (1, []),
    "markdown_headers": (
        16,
        [
            (0, 140, "396b3e0d869a1921"),
            (156, 298, "9fed28e902b5e0c5"),
            (278, 416, "8b61e2c7093fd579"),
            (396, 528, "2eb6164da45f7015"),
            (508, 675, "a2c644b75e2e223f"),
            (655, 817, "29afb7d03a904769"),
            (797, 963, "436bd60cc5b9a63a"),
            (943, 1104, "d411bff45af3af56"),
            (1084, 1252, "da7e503d96d03762"),
            (1232, 1353, "32b1637e0e3e165e"),
            (1333, 1391, "b883949663df95ad"),
            (1393, 1530, "ee5c01fba55450d7"),
            (1510, 1620, "ef58c9600260af33"),
        ],
    ),
    "markdown_no_headers": (
        5,
        [
            (0, 160, "9505adb3b0940854"),
            (140, 289, "035ae45c7895c424"),
            (269, 412, "fcc3418e6d99f29c"),
            (392, 500, "fc1af72de7e93046"),
        ],
    ),
    "code_defs_and_classes": (
        16,
        [
            (0, 301, "4e588ca52ecc2d75"),
            (327, 495, "86e370d2c79e9523"),
            (495, 643, "b7ad47d04093c8f3"),
            (623, 757, "45b2f894ad6864e6"),
            (737, 871, "fa83dfd53dd7b574"),
            (851, 985, "59cbd151cbcaa265"),
            (965, 1099, "5ccbf77b3a339645"),
            (1079, 1217, "a8afb90c53128827"),
            (1197, 1228, "649b2ec6f02ed348"),
            (1230, 1350, "f150efbd6cbdb172"),
            (1330, 1435, "eef6fbdcd021fd3d"),
            (1437, 1626, "eeca457cf69eaf08"),
        ],
    ),
    "code_no_defs": (
        3,
        [
            (0, 120, "ebf3402c3fcc6e59"),
            (100, 220, "aa4cad3ce8b6a1d2"),
            (200, 320, "fe90567693741500"),
            (300, 360, "da8c6d061b33fa58"),
        ],
    ),
    "text_sentence_search_window": (
        2,
        [
            (0, 22, "11474b73c86f952c"),
            (22, 142, "392add47fbaa8197"),
            (142, 224, "a510eb9f9a3b5685"),
            (224, 264, "83ba021acb570beb"),
        ],
    ),
    "code_preamble_at_min_size": (1, [(0, 31, "b8aacb33de80a0af"), (31, 93, "0f8dd0590c96b5af")]),
    "markdown_section_at_min_size": (
        1,
        [(0, 32, "8f71448681467b9b"), (32, 75, "560b87ef9208d8c0")],
    ),
    "code_definition_at_start": (1, [(0, 75, "e2d8a87ea8f85f2c"), (75, 151, "13a38fb391118a7c")]),
    "code_blank_preamble": (1, [(42, 115, "9609b2d1a9e8469c")]),
    "empty": (0, []),
    "whitespace_only": (2, []),
    "markdown_empty": (0, []),
    "code_empty": (0, []),
}


def _digest(spans: list[ChunkSpan]) -> list[tuple[int, int, str]]:
    return [
        (s.start_char, s.end_char, hashlib.sha256(s.text.encode()).hexdigest()[:16]) for s in spans
    ]


def _run(name: str) -> tuple[TextChunker, str, list[ChunkSpan]]:
    cfg, kind, content = CASES[name]
    chunker = TextChunker(ChunkingConfig(**cfg))
    strategy = {
        "text": chunker.chunk_text,
        "markdown": chunker.chunk_markdown,
        "code": chunker.chunk_code,
    }[kind]
    return chunker, content, strategy(content)


def test_goldens_cover_every_case() -> None:
    assert set(GOLDEN) == set(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_spans_match_framework_chunker_goldens(name: str) -> None:
    chunker, content, spans = _run(name)
    expected_estimate, expected_spans = GOLDEN[name]
    assert _digest(spans) == expected_spans
    assert chunker.estimate_chunks(content) == expected_estimate


@pytest.mark.parametrize("name", sorted(CASES))
def test_spans_are_stripped_slices_of_the_source(name: str) -> None:
    _, content, spans = _run(name)
    for span in spans:
        assert isinstance(span, ChunkSpan)
        assert span.text == content[span.start_char : span.end_char].strip()


def test_chunking_config_defaults_match_both_former_copies() -> None:
    defaults = {f.name: f.default for f in dataclasses.fields(ChunkingConfig)}
    assert defaults == {
        "chunk_size": 1344,
        "chunk_overlap": 134,
        "min_chunk_size": 200,
        "max_chunk_size": 2000,
        "respect_sentence_boundaries": True,
        "respect_paragraph_boundaries": True,
        "code_aware": True,
    }
    assert TextChunker().config == ChunkingConfig()


def test_chunk_span_is_an_immutable_triple() -> None:
    span = ChunkSpan("abc", 1, 4)
    text, start, end = span
    assert (text, start, end) == ("abc", 1, 4)
    with pytest.raises(AttributeError):
        span.text = "x"  # type: ignore[misc]


def test_strategies_dispatch_through_overridable_chunk_text() -> None:
    """Oversized sections are split via ``self.chunk_text`` so overrides apply."""
    calls: list[str] = []

    class Recording(TextChunker):
        def chunk_text(self, content: str) -> list[ChunkSpan]:
            calls.append(content)
            return super().chunk_text(content)

    cfg, _, content = CASES["markdown_headers"]
    Recording(ChunkingConfig(**cfg)).chunk_markdown(content)
    assert calls, "large markdown section must be re-chunked through chunk_text"
    calls.clear()
    Recording(ChunkingConfig(**cfg)).chunk_code(CASES["code_no_defs"][2])
    assert calls == [CASES["code_no_defs"][2]]


def test_chunking_module_is_stdlib_only() -> None:
    import ast
    import sys
    from pathlib import Path

    import victor_contracts.chunking as module

    tree = ast.parse(Path(module.__file__).read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert roots <= set(sys.stdlib_module_names) | {"__future__"}, roots


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"chunk_size": 0}, "chunk_size must be positive"),
        ({"chunk_size": -5, "chunk_overlap": 0}, "chunk_size must be positive"),
        ({"min_chunk_size": -1, "chunk_overlap": 0}, "min_chunk_size must be >= 0"),
        ({"chunk_size": 100, "chunk_overlap": 100}, "chunk_overlap must satisfy"),
        ({"chunk_size": 100, "chunk_overlap": 150}, "chunk_overlap must satisfy"),
        ({"chunk_overlap": -1}, "chunk_overlap must satisfy"),
        ({"chunk_overlap": 201}, "must not exceed min_chunk_size"),
    ],
)
def test_config_rejects_settings_that_never_terminate(
    overrides: dict[str, int], message: str
) -> None:
    """These settings made the pre-0.12.0 ``chunk_text`` loop forever."""
    with pytest.raises(ValueError, match=message):
        ChunkingConfig(**overrides)


@pytest.mark.parametrize("sentences", [True, False])
def test_tightest_valid_overlap_terminates(sentences: bool) -> None:
    config = ChunkingConfig(
        chunk_size=100,
        chunk_overlap=40,
        min_chunk_size=40,
        respect_sentence_boundaries=sentences,
    )
    spans = TextChunker(config).chunk_text("Word one. Two! " * 200)
    assert spans
    starts = [span.start_char for span in spans]
    assert starts == sorted(set(starts))
