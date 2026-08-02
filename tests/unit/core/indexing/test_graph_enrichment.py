# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path

from victor.core.database import get_project_database
from victor.core.indexing.graph_enrichment import ensure_project_graph_enriched


def _write_tool_sample(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "from victor.tools.decorators import tool",
                "from victor.storage.graph.protocol import GraphStoreProtocol",
                "",
                "@tool(name='sample_tool')",
                "def sample_tool() -> None:",
                "    pass",
                "",
                "class SampleStore(GraphStoreProtocol):",
                "    pass",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _seed_graph(root: Path) -> None:
    db = get_project_database(root)
    with db.transaction() as conn:
        conn.executemany(
            """
            INSERT INTO graph_node (
                node_id, type, name, file, line, end_line, lang, signature,
                docstring, parent_id, embedding_ref, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "symbol:victor/tools/decorators.py:tool",
                    "function",
                    "tool",
                    "victor/tools/decorators.py",
                    224,
                    500,
                    "python",
                    None,
                    None,
                    None,
                    None,
                    "{}",
                ),
                (
                    "symbol:victor/tools/metadata_registry.py:ToolMetadataRegistry",
                    "class",
                    "ToolMetadataRegistry",
                    "victor/tools/metadata_registry.py",
                    431,
                    900,
                    "python",
                    None,
                    None,
                    None,
                    None,
                    "{}",
                ),
                (
                    "symbol:victor/storage/graph/protocol.py:GraphStoreProtocol",
                    "class",
                    "GraphStoreProtocol",
                    "victor/storage/graph/protocol.py",
                    39,
                    105,
                    "python",
                    None,
                    None,
                    None,
                    None,
                    "{}",
                ),
                (
                    "symbol:tools_sample.py:sample_tool",
                    "function",
                    "sample_tool",
                    "tools_sample.py",
                    5,
                    6,
                    "python",
                    None,
                    None,
                    None,
                    None,
                    "{}",
                ),
                (
                    "symbol:tools_sample.py:SampleStore",
                    "class",
                    "SampleStore",
                    "tools_sample.py",
                    8,
                    9,
                    "python",
                    None,
                    None,
                    None,
                    None,
                    "{}",
                ),
            ],
        )
        conn.execute(
            """
            INSERT INTO graph_edge (src, dst, type, weight, metadata)
            VALUES (?, ?, 'INHERITS', NULL, '{}')
            """,
            (
                "symbol:tools_sample.py:SampleStore",
                "symbol:victor/storage/graph/protocol.py:GraphStoreProtocol",
            ),
        )


def test_graph_enrichment_adds_synthetic_protocol_and_tool_edges(
    tmp_path: Path,
) -> None:
    _write_tool_sample(tmp_path / "tools_sample.py")
    _seed_graph(tmp_path)

    stats = ensure_project_graph_enriched(tmp_path, latest_mtime=1.0)

    assert stats.implements_edges == 1
    assert stats.decorates_edges == 1
    assert stats.registers_edges == 1

    db = get_project_database(tmp_path)
    edge_rows = db.query("""
        SELECT src, dst, type
        FROM graph_edge
        WHERE type IN ('IMPLEMENTS', 'DECORATES', 'REGISTERS')
        ORDER BY type, src, dst
        """)
    edges = {(row["src"], row["dst"], row["type"]) for row in edge_rows}
    assert (
        "symbol:tools_sample.py:SampleStore",
        "symbol:victor/storage/graph/protocol.py:GraphStoreProtocol",
        "IMPLEMENTS",
    ) in edges
    assert (
        "symbol:victor/tools/decorators.py:tool",
        "symbol:tools_sample.py:sample_tool",
        "DECORATES",
    ) in edges
    assert (
        "symbol:tools_sample.py:sample_tool",
        "symbol:victor/tools/metadata_registry.py:ToolMetadataRegistry",
        "REGISTERS",
    ) in edges


def test_graph_enrichment_is_idempotent_for_same_repo_state(tmp_path: Path) -> None:
    _write_tool_sample(tmp_path / "tools_sample.py")
    _seed_graph(tmp_path)

    first = ensure_project_graph_enriched(tmp_path, latest_mtime=10.0)
    second = ensure_project_graph_enriched(tmp_path, latest_mtime=10.0)

    assert first.total_edges == 3
    assert second.skipped is True

    db = get_project_database(tmp_path)
    row = db.query_one("""
        SELECT COUNT(*)
        FROM graph_edge
        WHERE type IN ('IMPLEMENTS', 'DECORATES', 'REGISTERS')
        """)
    assert row is not None
    assert int(row[0]) == 3
