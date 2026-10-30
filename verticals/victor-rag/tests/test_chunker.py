"""Characterization goldens for ``victor_rag.chunker.DocumentChunker``.

Recorded before the text/markdown/code strategies moved to the SDK chunking
core (``victor_contracts.chunking``); any output change fails here.
Regenerate only for an intended behavior change::

    UPDATE_CHUNKER_GOLDENS=1 pytest verticals/victor-rag/tests/test_chunker.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from victor_rag.chunker import ChunkingConfig, DocumentChunker, detect_document_type
from victor_rag.document_models import Document

GOLDENS = Path(__file__).with_name("chunker_goldens.json")

SMALL = {"chunk_size": 150, "chunk_overlap": 15, "min_chunk_size": 25, "max_chunk_size": 220}

SAMPLES: dict[str, tuple[str, str]] = {
    # name -> (source, content); the source drives auto-detection
    "prose.txt": ("notes.txt", " ".join(f"Line {n} ends here{'?' * (n % 2)}." for n in range(90))),
    "unicode": ("", " ".join(f"Grüße {n} — 東京 {'✓' * (n % 4)}!" for n in range(70))),
    "guide.md": (
        "guide.md",
        "Lead-in text. " * 12
        + "\n# One\n"
        + "Alpha beta gamma. " * 40
        + "\n## Two\nshort\n## Three\n"
        + "Delta epsilon. " * 90,
    ),
    "mod.py": (
        "mod.py",
        "import sys\n" * 20
        + "def a():\n    pass\n\n"
        + "class K:\n"
        + "    field = 0\n" * 120
        + "func b() {\n"
        + "  call();\n" * 70
        + "}\n",
    ),
    "page.html": (
        "",
        "<html><body><nav>menu</nav><h1>Top</h1>"
        + "".join(f"<p>Para {n} with ümlauts and words. </p>" * 3 for n in range(20))
        + "<h3>Tail</h3><section>"
        + "Tail text. " * 200
        + "</section><style>x{}</style></body></html>",
    ),
    "obj.json": (
        "obj.json",
        json.dumps(
            {"meta": {"v": 1}, "rows": [{"n": n, "s": "ab " * 20} for n in range(30)]},
            ensure_ascii=False,
        ),
    ),
    "arr.json": ("", json.dumps([{"n": n, "w": "ü" * (n * 5 % 60)} for n in range(50)])),
    "yaml": ("conf.yaml", "key: value\nother: 2\n"),
    "bad.json": ("", "[{"),
    # sections at exactly the SMALL limits (max 220, min 25) and one either side
    "edges.md": (
        "edges.md",
        "".join(
            f"## S{n}\n" + "z" * (k - 6) + "\n" for n, k in enumerate((24, 25, 26, 219, 220, 221))
        ),
    ),
    "defs.rs": (
        "defs.rs",
        "".join(
            f"{kw} Thing{n} {{\n" + "  body();\n" * 25 + "}\n"
            for n, kw in enumerate(("struct", "impl", "interface", "fn"))
        ),
    ),
    "fence.md": (
        "fence.md",
        "# Top\n" + "Words. " * 30 + "\n```\n# comment\n```\n## Tail\n" + "End. " * 40,
    ),
}


async def _embed(text: str) -> list[float]:
    return [float(len(text))]


async def _record() -> dict[str, Any]:
    result: dict[str, Any] = {"documents": {}, "strategies": {}, "detect": {}}
    flat = {"chunk_size": 200, "chunk_overlap": 0, "min_chunk_size": 0, "max_chunk_size": 300}
    configs = (
        ("default", {}),
        ("small", SMALL),
        ("flat", {**flat, "respect_sentence_boundaries": False}),
    )
    for label, cfg in configs:
        chunker = DocumentChunker(ChunkingConfig(**cfg))
        for name, (source, content) in SAMPLES.items():
            doc = Document(id=name, content=content, source=source, metadata={"m": 1})
            chunks = await chunker.chunk_document(doc, _embed)
            rows = []
            for chunk in chunks:
                assert chunk.id.startswith(f"{name}_{chunk.chunk_index}_")
                assert chunk.embedding == [float(len(chunk.content))]
                assert chunk.metadata == {"doc_type": "text", "source": source, "m": 1}
                rows.append(
                    [chunk.chunk_index, chunk.start_char, chunk.end_char, _sha(chunk.content)]
                )
            result["documents"][f"{label}/{name}"] = rows
            for strategy in (
                "_chunk_text",
                "_chunk_markdown",
                "_chunk_code",
                "_chunk_html",
                "_chunk_json",
            ):
                spans = getattr(chunker, strategy)(content)
                result["strategies"][f"{label}/{name}/{strategy}"] = [
                    [start, end, _sha(text)] for text, start, end in spans
                ]
            result["strategies"][f"{label}/{name}/estimate"] = chunker.estimate_chunks(content)
    result["detect"] = {name: detect_document_type(*sample) for name, sample in SAMPLES.items()}
    return result


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@pytest.fixture(scope="module")
def recorded() -> dict[str, Any]:
    import asyncio

    pytest.importorskip("bs4")
    data = asyncio.run(_record())
    if os.environ.get("UPDATE_CHUNKER_GOLDENS") == "1":
        GOLDENS.write_text(json.dumps(data, ensure_ascii=False) + "\n", "utf-8")
    return data


@pytest.mark.parametrize("section", ["documents", "strategies", "detect"])
def test_matches_golden(recorded: dict[str, Any], section: str):
    expected = json.loads(GOLDENS.read_text("utf-8"))[section]
    mismatched = [k for k in expected if recorded[section].get(k) != expected[k]]
    assert recorded[section].keys() == expected.keys()
    assert not mismatched, mismatched


def test_goldens_are_not_degenerate():
    expected = json.loads(GOLDENS.read_text("utf-8"))
    for strategy in ("_chunk_text", "_chunk_markdown", "_chunk_code", "_chunk_html", "_chunk_json"):
        lengths = [len(v) for k, v in expected["strategies"].items() if k.endswith(strategy)]
        assert max(lengths) >= 3, strategy
    assert set(expected["detect"].values()) >= {"text", "markdown", "code", "html", "json", "yaml"}


async def test_subclass_text_override_reaches_markdown_and_code():
    class Spy(DocumentChunker):
        def _chunk_text(self, content: str):
            return [("spy", 0, 3)]

    spy = Spy(ChunkingConfig(**SMALL))
    assert "spy" in [text for text, _, _ in spy._chunk_markdown(SAMPLES["guide.md"][1])]
    assert "spy" in [text for text, _, _ in spy._chunk_code(SAMPLES["mod.py"][1])]


def test_private_hook_uses_cooperative_super():
    from victor_contracts.chunking import TextChunker

    class Mixin(TextChunker):
        def chunk_code(self, content: str):
            return [("mixin", 0, 5)]

    class Combined(DocumentChunker, Mixin):
        pass

    assert Combined()._chunk_code("def f():\n    pass\n") == [("mixin", 0, 5)]
