# Simple in-memory GraphStoreProtocol implementation (non-persistent).
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from victor.storage.graph.protocol import (
    GraphEdge,
    GraphNode,
    GraphStoreProtocol,
    GraphTraversalDirection,
)


class MemoryGraphStore(GraphStoreProtocol):
    """In-memory graph store useful for testing or ephemeral runs."""

    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[tuple[str, str, str], GraphEdge] = {}
        self._file_mtimes: Dict[str, float] = {}
        self._file_hashes: Dict[str, str] = {}

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def _canonical_file_path(self, file: str | Path) -> str:
        """Normalize file paths so equivalent aliases share one key."""
        path = Path(file).expanduser()
        try:
            return str(path.resolve(strict=False))
        except OSError:
            return str(path.absolute())

    def _file_path_variants(self, file: str | Path) -> set[str]:
        """Return raw plus canonical path forms for compatibility lookups."""
        raw_path = str(file)
        return {raw_path, self._canonical_file_path(raw_path)}

    async def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None:
        for n in nodes:
            self._nodes[n.node_id] = n

    async def upsert_edges(self, edges: Iterable[GraphEdge]) -> None:
        for e in edges:
            self._edges[(e.src, e.dst, e.type)] = e

    async def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[Iterable[str]] = None,
        *,
        direction: GraphTraversalDirection = "both",
        max_depth: int = 1,
    ) -> List[GraphEdge]:
        if direction not in {"out", "in", "both"}:
            raise ValueError(f"Unsupported graph traversal direction: {direction}")
        if max_depth < 1:
            return []

        allowed_types = set(edge_types) if edge_types else None
        frontier = {node_id}
        visited_nodes = {node_id}
        seen_edges: Dict[tuple[str, str, str], GraphEdge] = {}

        for _depth in range(max_depth):
            next_frontier: set[str] = set()
            for edge in self._edges.values():
                if allowed_types is not None and edge.type not in allowed_types:
                    continue

                traversed = False
                if direction in {"out", "both"} and edge.src in frontier:
                    traversed = True
                    next_frontier.add(edge.dst)
                if direction in {"in", "both"} and edge.dst in frontier:
                    traversed = True
                    next_frontier.add(edge.src)

                if traversed:
                    seen_edges[(edge.src, edge.dst, edge.type)] = edge

            next_frontier -= visited_nodes
            if not next_frontier:
                break
            visited_nodes.update(next_frontier)
            frontier = next_frontier

        return sorted(seen_edges.values(), key=lambda edge: (edge.src, edge.dst, edge.type))

    async def find_nodes(
        self,
        *,
        name: str | None = None,
        type: str | None = None,
        file: str | None = None,
    ) -> List[GraphNode]:
        file_variants = self._file_path_variants(file) if file is not None else None
        results: List[GraphNode] = []
        for node in self._nodes.values():
            if name is not None and node.name != name:
                continue
            if type is not None and node.type != type:
                continue
            if file_variants is not None and node.file not in file_variants:
                continue
            results.append(node)
        return results

    async def search_symbols(
        self,
        query: str,
        *,
        limit: int = 20,
        symbol_types: Optional[Iterable[str]] = None,
    ) -> List[GraphNode]:
        query_lower = query.lower()
        allowed_types = set(symbol_types) if symbol_types else None
        matches: List[GraphNode] = []
        for node in self._nodes.values():
            if allowed_types is not None and node.type not in allowed_types:
                continue
            haystacks = [
                node.name,
                node.signature or "",
                node.docstring or "",
            ]
            if any(query_lower in value.lower() for value in haystacks if value):
                matches.append(node)
        matches.sort(key=lambda node: (node.file, node.line or 0, node.name))
        return matches[:limit]

    async def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    async def get_all_nodes(self) -> List[GraphNode]:
        return sorted(
            self._nodes.values(),
            key=lambda node: (node.file, node.line or 0, node.name),
        )

    async def get_nodes_by_file(self, file: str) -> List[GraphNode]:
        file_variants = self._file_path_variants(file)
        return sorted(
            [node for node in self._nodes.values() if node.file in file_variants],
            key=lambda node: (node.line or 0, node.name),
        )

    async def update_file_mtime(
        self, file: str, mtime: float, content_hash: str | None = None
    ) -> None:
        key = self._canonical_file_path(file)
        self._file_mtimes[key] = mtime
        if content_hash is None:
            self._file_hashes.pop(key, None)
        else:
            self._file_hashes[key] = content_hash

    async def get_file_hashes(self, files: list[str]) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for file in files:
            for variant in self._file_path_variants(file):
                if variant in self._file_hashes:
                    hashes[file] = self._file_hashes[variant]
                    break
        return hashes

    async def get_stale_files(self, file_mtimes: Dict[str, float]) -> List[str]:
        return [
            file
            for file, current_mtime in file_mtimes.items()
            if max(
                (
                    self._file_mtimes.get(variant, float("-inf"))
                    for variant in self._file_path_variants(file)
                ),
                default=float("-inf"),
            )
            < current_mtime
        ]

    async def get_indexed_files(self) -> List[str]:
        return sorted(self._file_mtimes.keys())

    async def delete_by_file(self, file: str) -> None:
        file_variants = self._file_path_variants(file)
        node_ids = {node.node_id for node in self._nodes.values() if node.file in file_variants}
        self._nodes = {
            node_id: node for node_id, node in self._nodes.items() if node_id not in node_ids
        }
        self._edges = {
            key: edge
            for key, edge in self._edges.items()
            if edge.src not in node_ids and edge.dst not in node_ids
        }
        for file_variant in file_variants:
            self._file_mtimes.pop(file_variant, None)
            self._file_hashes.pop(file_variant, None)

    async def delete_by_repo(self, clear_embeddings: bool = False) -> None:
        del clear_embeddings  # This backend has no external embedding collection.
        self._nodes.clear()
        self._edges.clear()
        self._file_mtimes.clear()
        self._file_hashes.clear()

    async def stats(self) -> Dict[str, Any]:
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "path": ":memory:",
        }

    async def get_all_edges(self) -> List[GraphEdge]:
        return sorted(self._edges.values(), key=lambda edge: (edge.src, edge.dst, edge.type))
