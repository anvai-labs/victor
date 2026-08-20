"""Soft-boundary helpers for projecting victor-codegraph repository relations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Any, Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepositoryRelationProjection:
    """Resolved relation rows and per-category counts for Graph-RAG."""

    edges: tuple[tuple[str, str, str], ...] = ()
    calls: int = 0
    relationships: int = 0
    imports: int = 0
    dropped_unmaterialized: int = 0


def prepare_repository_snapshot(
    root_path: Path,
    allowed_files: Iterable[str],
) -> Any | None:
    """Parse and filter one repository snapshot, or return ``None`` softly."""
    try:
        import victor_codegraph as codegraph

        repository = codegraph.parse_repo(root_path)
    except Exception as exc:
        logger.debug("victor-codegraph repository snapshot unavailable: %s", exc)
        return None

    allowed = set(allowed_files)
    repository.files = {
        path: parsed for path, parsed in repository.files.items() if path in allowed
    }
    repository.manifest = {
        path: digest for path, digest in repository.manifest.items() if path in allowed
    }
    repository.symbols = [
        symbol for symbol in repository.symbols if symbol.location.file_path in allowed
    ]
    symbol_ids = {symbol.id for symbol in repository.symbols}
    repository.relations = [
        relation
        for relation in repository.relations
        if relation.from_symbol_id in symbol_ids
        and (relation.target_ref is not None or relation.to_symbol_id in symbol_ids)
    ]
    return repository


def project_resolved_relations(
    repository: Any,
    *,
    materialized_symbol_ids: AbstractSet[str] | None = None,
) -> RepositoryRelationProjection:
    """Map resolved relations onto the materialized Graph-RAG graph.

    The repository snapshot describes parser output, not committed graph state.
    When ``materialized_symbol_ids`` is supplied, both endpoints must exist in
    that committed set. This keeps permissive and referentially strict graph
    stores behaviorally identical instead of letting SQLite accept edges that
    ORION rejects.
    """
    edge_types = {
        "CALLS": "CALLS",
        "EXTENDS": "INHERITS",
        "IMPLEMENTS": "IMPLEMENTS",
        "IMPORTS": "IMPORTS",
    }
    edges: list[tuple[str, str, str]] = []
    calls = relationships = imports = dropped_unmaterialized = 0
    seen: set[tuple[str, str, str]] = set()
    for relation in repository.relations:
        if relation.target_ref is not None or relation.provenance not in {
            "repository_resolver",
            "repository_import_resolver",
        }:
            continue
        edge_type = edge_types.get(relation.relation_type.name)
        if edge_type is None:
            continue
        edge = (relation.from_symbol_id, relation.to_symbol_id, edge_type)
        if materialized_symbol_ids is not None and (
            edge[0] not in materialized_symbol_ids or edge[1] not in materialized_symbol_ids
        ):
            dropped_unmaterialized += 1
            continue
        if edge in seen or edge[0] == edge[1]:
            continue
        seen.add(edge)
        edges.append(edge)
        if edge_type == "CALLS":
            calls += 1
        elif edge_type == "IMPORTS":
            imports += 1
        else:
            relationships += 1
    return RepositoryRelationProjection(
        tuple(edges),
        calls,
        relationships,
        imports,
        dropped_unmaterialized,
    )
