"""Shared one-hop frontier-edge SELECT for the SQL graph stores (private).

`SqliteGraphStore._select_frontier_edges` and
`DuckDBGraphStore._select_frontier_edges` were byte-identical 41-line copies
(jscpd + supervised dedup, round 7) differing in exactly three tokens: the
connection type, the edge-table name, and the metadata decoder (stdlib json
vs orjson). The one implementation lives here, parameterized by those three
differences; each store keeps a one-line delegate so signatures, connection
typing, and the stores' differing caller-wrapping (worker offload vs
direct-under-lock) are unchanged.
"""

from __future__ import annotations

from typing import Any, Callable

from victor.storage.graph.protocol import GraphEdge


def select_frontier_edges(
    conn: Any,
    *,
    table: str,
    frontier: set[str],
    edge_types: list[str],
    node_column: str,
    neighbor_column: str,
    decode_metadata: Callable[[Any], dict],
) -> list[tuple[GraphEdge, str]]:
    """One-hop edges FROM the frontier nodes, as (edge, neighbor_id) pairs.

    ``table`` is the edge table name; ``decode_metadata`` decodes the
    metadata column (orjson for duckdb, stdlib json for sqlite — both
    backends must keep their existing decoder, not silently share one).
    """

    if not frontier:
        return []

    frontier_params = list(frontier)
    frontier_placeholders = ",".join("?" for _ in frontier_params)
    params: list[Any] = frontier_params
    type_clause = ""

    if edge_types:
        type_placeholders = ",".join("?" for _ in edge_types)
        type_clause = f" AND type IN ({type_placeholders})"
        params.extend(edge_types)

    query = f"""
        SELECT src, dst, type, weight, metadata, {neighbor_column}
        FROM {table}
        WHERE {node_column} IN ({frontier_placeholders}){type_clause}
    """
    cur = conn.execute(query, params)
    return [
        (
            GraphEdge(
                src=row[0],
                dst=row[1],
                type=row[2],
                weight=row[3],
                metadata=decode_metadata(row[4]) if row[4] else {},
            ),
            row[5],
        )
        for row in cur.fetchall()
    ]
