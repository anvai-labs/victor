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


import pytest

pytest.importorskip("victor_coding.codebase.indexer")

from victor.storage.graph.sqlite_store import SqliteGraphStore
from victor_coding.codebase.indexer import CodebaseIndex


@pytest.mark.skip(reason="Tree-sitter parser not available in CI environment")
@pytest.mark.asyncio
async def test_indexer_writes_graph(tmp_path):
    """Test that the indexer correctly populates graph store with nodes and edges.

    This test verifies the core graph functionality:
    - File nodes are created for each indexed file
    - Symbol nodes are created for functions and classes
    - CONTAINS edges link files to their symbols
    """
    # Create minimal repo
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text(
        "import math\n\n"
        "def foo():\n    return math.sqrt(4)\n\n"
        "class Bar:\n    def method(self):\n        return foo()\n\n"
        "class Baz(Bar):\n    pass\n",
        encoding="utf-8",
    )
    (repo / "utils.py").write_text(
        "def helper():\n    return 42\n\n\ndef caller():\n    return helper()\n",
        encoding="utf-8",
    )

    graph_path = repo / ".victor" / "graph" / "graph.db"
    store = SqliteGraphStore(graph_path)

    indexer = CodebaseIndex(
        root_path=str(repo),
        use_embeddings=False,
        enable_watcher=False,
        graph_store=store,
    )

    await indexer.index_codebase()

    stats = await store.stats()
    # Should have at least 2 file nodes and some symbol nodes
    assert stats["nodes"] >= 2, f"expected at least 2 nodes, got {stats['nodes']}"
    # Should have CONTAINS edges from files to symbols
    assert stats["edges"] >= 1, f"expected at least 1 edge, got {stats['edges']}"

    # Symbol nodes should be present
    symbols = await store.find_nodes(type="function")
    assert symbols, "expected function symbols in graph store"

    # Class nodes should be present
    classes = await store.find_nodes(type="class")
    assert classes, "expected class symbols in graph store"

    # CONTAINS edges from file to symbols should exist
    neighbors = await store.get_neighbors("file:main.py")
    assert neighbors, "expected CONTAINS edges from file node"

    # Verify CONTAINS edges point to symbol nodes
    contains_edges = [e for e in neighbors if e.type == "CONTAINS"]
    assert contains_edges, "expected CONTAINS edges from file:main.py"

    # Verify symbol names are correct
    symbol_names = {e.dst.split(":")[-1] for e in contains_edges}
    assert "foo" in symbol_names, "expected foo function symbol"
    assert "Bar" in symbol_names, "expected Bar class symbol"
