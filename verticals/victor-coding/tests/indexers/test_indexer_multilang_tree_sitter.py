# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest

pytest.importorskip("victor_coding.codebase.graph.registry")
pytest.importorskip("victor_coding.codebase.indexer")

from victor_coding.codebase.graph.registry import create_graph_store
from victor_coding.codebase.indexer import CodebaseIndex


@pytest.mark.asyncio
async def test_indexer_tree_sitter_javascript(tmp_path: Path):
    try:
        from victor_coding.codebase.tree_sitter_manager import get_parser
    except Exception:  # pragma: no cover - tree-sitter unavailable
        pytest.skip("tree-sitter not available")

    try:
        parser = get_parser("javascript")
    except Exception:  # pragma: no cover - language grammar missing
        pytest.skip("javascript parser not available")

    if parser is None:  # pragma: no cover
        pytest.skip("javascript parser not available")

    src = tmp_path / "app.js"
    src.write_text(
        """
        class Greeter {
          greet() { helper(); }
        }
        function helper() { return 1; }
        const arrowHelper = () => helper();
        """,
        encoding="utf-8",
    )

    index = CodebaseIndex(
        tmp_path,
        graph_store=create_graph_store("memory", Path(":memory:")),
        enable_watcher=False,
    )

    await index.index_codebase()

    rel = str(src.relative_to(tmp_path))
    assert rel in index.files
    metadata = index.files[rel]
    assert metadata.symbols  # class/function captured
    names = {s.name for s in metadata.symbols}
    assert {"Greeter", "helper", "arrowHelper"} & names
    # References should include helper from call expression and arrow call
    assert "helper" in metadata.references

    stats = await index.graph_store.stats()  # type: ignore[union-attr]
    assert stats["nodes"] > 0
    assert stats["edges"] > 0
