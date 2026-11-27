"""Victor's indexer classifies a ``.h`` header by the same rule on every path.

The CodeGraph path resolves C or C++ from the header's text and the project
inventory (victor-codegraph). The fallback path without a bound CodeGraph
repository, the single-file path and the edge builders must not fall back to
the extension's ``c``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("victor_codegraph")

from victor.core.graph_rag.config import GraphIndexConfig  # noqa: E402
from victor.core.graph_rag.indexing import GraphIndexingPipeline  # noqa: E402
from victor.storage.graph.sqlite_store import SqliteGraphStore  # noqa: E402

NEUTRAL_HEADER = "struct point { int x; int y; };\n"


def test_header_default_follows_the_run_inventory():
    default = GraphIndexingPipeline._project_header_default
    assert default([Path("a.c"), Path("b.h"), Path("c.py")]) == "c"
    assert default([Path("a.c"), Path("b.cc"), Path("c.h")]) == "cpp"
    assert default([Path("b.h")]) == "cpp"
    assert default([]) == "cpp"


def test_header_resolution_reads_text_first_then_the_project_default():
    resolve = GraphIndexingPipeline._resolve_header_language
    assert resolve(Path("p.h"), NEUTRAL_HEADER, "c", "c") == "c"
    assert resolve(Path("p.h"), NEUTRAL_HEADER, "c") == "cpp"
    assert resolve(Path("p.h"), NEUTRAL_HEADER, "c", "cpp") == "cpp"
    assert resolve(Path("p.h"), "class X {};\n", "c", "c") == "cpp"
    assert resolve(Path("p.py"), NEUTRAL_HEADER, "python", "c") == "python"


def _bare_pipeline(resolved: dict[str, str]) -> GraphIndexingPipeline:
    pipeline = GraphIndexingPipeline.__new__(GraphIndexingPipeline)
    pipeline._resolved_header_languages = resolved
    pipeline._canonical_file_str = staticmethod(lambda file_path: file_path.as_posix())  # type: ignore[method-assign]
    pipeline._detect_language = staticmethod(  # type: ignore[method-assign]
        lambda file_path: {".py": "python", ".h": "c"}.get(file_path.suffix, "unknown")
    )
    return pipeline


def test_edge_builders_see_the_language_a_header_was_parsed_with():
    pipeline = _bare_pipeline({"pkg/a.h": "cpp"})
    assert pipeline._file_language(Path("pkg/a.h")) == "cpp"
    assert pipeline._file_language(Path("pkg/b.h")) == "c"  # not re-classified: extension rule
    assert pipeline._file_language(Path("pkg/m.py")) == "python"


@pytest.mark.asyncio
async def test_run_inventory_sets_the_fallback_header_default(tmp_path):
    root = tmp_path / "pure-c"
    root.mkdir()
    (root / "point.c").write_text("int point_x(void) { struct point p; return p.x; }\n")
    (root / "point.h").write_text(NEUTRAL_HEADER)
    pipeline = GraphIndexingPipeline(
        SqliteGraphStore(project_path=root),
        GraphIndexConfig(
            root_path=root,
            enable_ccg=False,
            enable_embeddings=False,
            enable_subgraph_cache=False,
        ),
    )
    files = [root / "point.c", root / "point.h"]

    await pipeline._prepare_incremental_work(files, root)
    assert pipeline._header_default == "c"

    (root / "util.cc").write_text("int util() { return 0; }\n")
    await pipeline._prepare_incremental_work([*files, root / "util.cc"], root)
    assert pipeline._header_default == "cpp"
