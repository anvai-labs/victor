"""Characterization goldens for ``victor.framework.ingestion.chunker``.

The goldens in ``chunker_goldens.json`` were recorded from ``BaseChunker``
before its text/markdown/code strategies moved to the SDK chunking core
(``victor_contracts.chunking``). Any output change fails here; regenerate
only for an intended behavior change::

    UPDATE_CHUNKER_GOLDENS=1 pytest tests/unit/framework/ingestion/test_chunker.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from victor.framework.ingestion.chunker import BaseChunker, detect_document_type
from victor.framework.ingestion.models import ChunkingConfig

GOLDENS = Path(__file__).with_name("chunker_goldens.json")
_UPDATE = os.environ.get("UPDATE_CHUNKER_GOLDENS") == "1"

_PROSE = " ".join(
    f"Sentence {i} talks about topic {i % 7}{'!' if i % 5 == 0 else '.'}" for i in range(120)
)
_UNICODE = " ".join(
    f"Ünïcödé séntence {i} — naïve café 日本語 {'🙂' * (i % 3)}." for i in range(80)
)
_MARKDOWN = "\n\n".join(
    [
        "Preamble paragraph before any header. " * 3,
        "# Title\n\n" + "Intro text. " * 30,
        "## Short\n\nTiny.",
        "## Long section\n\n" + "Long body sentence here. " * 120,
        "### Code fence\n\n```python\nprint('x')\n```\n" + "Trailing words. " * 20,
    ]
)
_CODE = "\n".join(
    [
        '"""Module docstring."""',
        "import os",
        "",
        "x = 1\n" * 60,
        "def small():\n    return 1\n",
        "class Big:\n" + "".join(f"    attr_{i} = {i}\n" for i in range(150)),
        "async def coro():\n" + "    await thing()\n" * 40,
        "def huge():\n" + "".join(f"    value_{i} = compute({i}).\n" for i in range(140)),
    ]
)
_HTML = (
    "<!DOCTYPE html><html><head><title>T</title><style>p{}</style></head><body>"
    "<header>Site header</header><nav>Nav links</nav>"
    "<h1>Heading One</h1>"
    + "".join(f"<p>Paragraph {i} has some text in it. " * 4 + "</p>" for i in range(25))
    + "<h2>Second</h2><div>"
    + "Div content sentence. " * 150
    + "</div><ul>"
    + "".join(f"<li>Item {i} — ünïcode</li>" for i in range(30))
    + "</ul><script>var x = 1;</script><footer>Footer</footer></body></html>"
)
_JSON_DICT = json.dumps(
    {
        "title": "Doc",
        "small": {"a": 1},
        "big": [{"id": i, "text": "lorem ipsum " * 10} for i in range(40)],
        "medium": {f"k{i}": f"välue {i} " * 5 for i in range(12)},
        "名前": "日本語の値 " * 50,
    },
    ensure_ascii=False,
)
_JSON_LIST = json.dumps(
    [{"id": i, "name": f"ïtem {i}", "body": "word " * (i * 7 % 90)} for i in range(70)],
    ensure_ascii=False,
)

# Sections sized exactly at the "small" config's limits (max 240, min 30),
# one either side, so off-by-one changes in the size checks show up.
_BOUNDARY_MD = "".join(
    f"# H{n}\n" + "x" * (size - 5) + "\n" for n, size in enumerate((29, 30, 31, 239, 240, 241))
)
_POLYGLOT = "\n".join(
    [
        "struct Point {\n" + "  x: i32,\n" * 30 + "}",
        "impl Point {\n" + "  fn norm() {}\n" * 30 + "}",
        "interface Shape {\n" + "  area(): number;\n" * 30 + "}",
        "fn main() {\n" + "  run();\n" * 30 + "}",
        "func Go() {\n" + "  do()\n" * 30 + "}",
        "function js() {\n" + "  call();\n" * 30 + "}",
    ]
)
_FENCED = (
    "# Real header\n"
    + "Intro words. " * 20
    + "\n```bash\n# not a header\necho hi\n## also not\n```\n"
    + "## Next\n"
    + "More words here. " * 25
)

CONTENTS: dict[str, str] = {
    "empty": "",
    "blank": "   \n\n  ",
    "tiny": "Just a few words.",
    "prose": _PROSE,
    "unicode": _UNICODE,
    "markdown": _MARKDOWN,
    "code": _CODE,
    "html": _HTML,
    "json_dict": _JSON_DICT,
    "json_list": _JSON_LIST,
    "json_invalid": '{"broken": [1, 2,',
    "boundary_md": _BOUNDARY_MD,
    "polyglot": _POLYGLOT,
    "fenced": _FENCED,
}

CONFIGS: dict[str, dict[str, Any]] = {
    "default": {},
    "small": {"chunk_size": 160, "chunk_overlap": 20, "min_chunk_size": 30, "max_chunk_size": 240},
    "no_sentences": {
        "chunk_size": 300,
        "chunk_overlap": 0,
        "min_chunk_size": 0,
        "max_chunk_size": 500,
        "respect_sentence_boundaries": False,
    },
}

DOC_TYPES = ["text", "markdown", "code", "html", "json"]

DETECTION_CASES: list[tuple[str, str]] = [
    ("notes.md", "plain"),
    ("page.HTML", "plain"),
    ("data.yaml", "a: 1"),
    ("data.csv", "a,b"),
    ("lib.rs", "fn main() {}"),
    ("https://example.com/api", '{"a": 1}'),
    ("", "<html><body>x</body></html>"),
    ("", "[1, 2, 3]"),
    ("", "{not json"),
    ("", '<?xml version="1.0"?><root/>'),
    ("", "<root><a/></root>"),
    ("", "# Heading\nbody"),
    ("", "text\n```\ncode\n```"),
    ("", "def f():\n    pass"),
    ("", "package main"),
    ("", "Just prose."),
]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _actual() -> dict[str, Any]:
    out: dict[str, Any] = {"chunks": {}, "estimates": {}, "detect": []}
    for cfg_name, cfg in CONFIGS.items():
        chunker = BaseChunker(ChunkingConfig(**cfg))
        for content_name, content in CONTENTS.items():
            out["estimates"][f"{cfg_name}/{content_name}"] = chunker.estimate_chunks(content)
            for doc_type in DOC_TYPES:
                chunks = chunker.chunk(content, doc_type=doc_type, metadata={"k": "v"})
                assert all(c.doc_type == doc_type and c.metadata == {"k": "v"} for c in chunks)
                out["chunks"][f"{cfg_name}/{content_name}/{doc_type}"] = [
                    [c.chunk_index, c.start_char, c.end_char, _digest(c.content)] for c in chunks
                ]
    out["detect"] = [
        [source, content, str(detect_document_type(source, content).value)]
        for source, content in DETECTION_CASES
    ]
    return out


@pytest.fixture(scope="module")
def actual() -> dict[str, Any]:
    pytest.importorskip("bs4")
    pytest.importorskip("lxml")
    result = _actual()
    if _UPDATE:
        GOLDENS.write_text(json.dumps(result, ensure_ascii=False) + "\n", "utf-8")
    return result


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    return json.loads(GOLDENS.read_text("utf-8"))


@pytest.mark.parametrize(
    "key",
    [f"{c}/{n}/{t}" for c in CONFIGS for n in CONTENTS for t in DOC_TYPES],
)
def test_chunk_output_matches_golden(actual: dict[str, Any], golden: dict[str, Any], key: str):
    assert actual["chunks"][key] == golden["chunks"][key]


def test_estimates_match_golden(actual: dict[str, Any], golden: dict[str, Any]):
    assert actual["estimates"] == golden["estimates"]


def test_document_type_detection_matches_golden(actual: dict[str, Any], golden: dict[str, Any]):
    assert actual["detect"] == golden["detect"]


def test_goldens_exercise_every_strategy(golden: dict[str, Any]):
    """Guard against degenerate goldens: each strategy yields several chunks somewhere."""
    for doc_type in DOC_TYPES:
        counts = [
            len(v)
            for k, v in golden["chunks"].items()
            if k.startswith("small/") and k.endswith(f"/{doc_type}")
        ]
        assert max(counts) >= 3, doc_type


def test_overridden_strategy_is_used_by_markdown_and_code():
    """Subclass overrides of ``_chunk_text`` reach the strategies that split long sections."""
    calls: list[str] = []

    class Spy(BaseChunker):
        def _chunk_text(self, content: str):
            calls.append(content[:10])
            return [("spy", 0, 3)]

    cfg = ChunkingConfig(chunk_size=160, chunk_overlap=20, min_chunk_size=30, max_chunk_size=240)
    spy = Spy(cfg)
    md = spy.chunk(_MARKDOWN, doc_type="markdown")
    code = spy.chunk(_CODE, doc_type="code")
    assert calls
    assert any(c.content == "spy" for c in md)
    assert any(c.content == "spy" for c in code)


def test_cooperative_mro_reaches_sdk_mixin():
    """Private hooks call ``super()``, so a TextChunker mixin later in the MRO is used."""
    from victor_contracts.chunking import TextChunker

    class Mixin(TextChunker):
        def chunk_text(self, content: str):
            return [("mixin", 0, 5)]

    class Combined(BaseChunker, Mixin):
        pass

    assert [c.content for c in Combined().chunk("Some text. " * 50)] == ["mixin"]
