"""Soft-boundary helpers for projecting victor-codegraph repository relations."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from importlib import metadata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, AbstractSet, Any, Iterable, Mapping

if TYPE_CHECKING:
    from victor_codegraph import ParsedRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepositorySnapshotQualification:
    """Prepared-input evidence, never proof of a committed graph generation.

    Missing optional parser/version information remains explicit. Partial parses
    retain their diagnostics in the repository; admission does not erase them.
    """

    canonical_available: bool
    selected_count: int
    missing_files: tuple[str, ...] = ()
    failed_files: tuple[str, ...] = ()
    partial_files: tuple[str, ...] = ()
    parser_versions: tuple[tuple[str, str | None], ...] = ()
    prepared_id: str | None = None

    @property
    def status(self) -> str:
        if not self.canonical_available:
            return "legacy-unqualified"
        if self.missing_files or self.failed_files:
            return "failed"
        return "partial" if self.partial_files else "complete"


def snapshot_parser_versions(repository: ParsedRepository | None) -> dict[str, str | None]:
    """Record installed parser versions without loading grammars or models.

    Versions describe the installed environment, not an editable source/build
    fingerprint. Missing distribution metadata cannot establish an identity.
    """
    packages = ["victor-codegraph"]
    if repository is not None and any(p.language != "python" for p in repository.files.values()):
        packages.extend(("tree-sitter", "tree-sitter-language-pack"))
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def qualify_repository_snapshot(
    repository: ParsedRepository | None,
    selected_files: Iterable[str],
    *,
    settings: Mapping[str, object],
    parser_versions: Mapping[str, str | None],
) -> RepositorySnapshotQualification:
    """Qualify exactly the selected manifest without changing source/symbol IDs.

    This does not qualify CCG/vector output, later source reads, or publication.
    Configured extraction intent is part of this prepared identity; operational
    batching/concurrency settings are deliberately supplied separately by callers.
    """
    selected = set(selected_files)
    versions = tuple(sorted(parser_versions.items()))
    if repository is None:
        return RepositorySnapshotQualification(False, len(selected), parser_versions=versions)
    missing = tuple(sorted(selected - (repository.files.keys() & repository.manifest.keys())))
    failed = tuple(
        sorted(
            path
            for path in selected - set(missing)
            if repository.files[path].status.value not in {"success", "partial"}
            or not repository.manifest[path]
            or repository.manifest[path] != repository.files[path].content_hash
        )
    )
    partial = tuple(
        sorted(
            path
            for path in selected & repository.files.keys()
            if repository.files[path].status.value == "partial"
        )
    )
    prepared_id = None
    if not missing and not failed and versions and all(value for _, value in versions):
        evidence = {
            "contract": "graph-preflight-v1",
            "manifest": [
                (p, repository.manifest[p], repository.files[p].status.value)
                for p in sorted(selected)
            ],
            "parser_versions": versions,
            "settings": dict(settings),
        }
        raw = json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False)
        prepared_id = "prepared-v1:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return RepositorySnapshotQualification(
        True, len(selected), missing, failed, partial, versions, prepared_id
    )


@dataclass(frozen=True)
class ProjectedRelationEdge:
    """One resolved relation ready to persist, with its binding provenance.

    ``provenance``/``confidence`` ride along from victor-codegraph's resolver
    (``scope_resolver`` | ``repository_resolver`` | ...) instead of being
    discarded at the bare-3-tuple boundary — the ADR-032 provenance model. The
    ``use`` payload (use-kind for REFERENCES) is the tier-3.5 enrichment
    schema: SCIP harvesting can later upgrade precision without re-harvesting.
    """

    src: str
    dst: str
    type: str
    provenance: str | None = None
    confidence: float | None = None
    use: str | None = None


@dataclass(frozen=True)
class RepositoryRelationProjection:
    """Resolved relation rows and per-category counts for Graph-RAG."""

    edges: tuple[ProjectedRelationEdge, ...] = ()
    calls: int = 0
    relationships: int = 0
    imports: int = 0
    references: int = 0
    dropped_unmaterialized: int = 0


def prepare_repository_snapshot(
    root_path: Path,
    allowed_files: Iterable[str],
    *,
    emit_references: bool = False,
) -> ParsedRepository | None:
    """Parse the caller's exact inventory using the canonical shared resolver.

    Only an absent optional package permits legacy extraction. Parser failures
    propagate; empty input stays explicitly empty, never default discovery.
    ``emit_references`` (ADR-032 tier 2.5) is passed through when the installed
    victor-codegraph supports it; an older package raises TypeError, which the
    caller's capability probe prevents from ever reaching here.
    """
    try:
        import victor_codegraph as codegraph
    except ModuleNotFoundError as exc:
        if exc.name != "victor_codegraph":
            raise
        logger.warning("victor-codegraph repository snapshot unavailable: %s", exc)
        return None
    if emit_references:
        return codegraph.parse_repo(root_path, selected_paths=allowed_files, emit_references=True)
    return codegraph.parse_repo(root_path, selected_paths=allowed_files)


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
        "REFERENCES": "REFERENCES",
    }
    edges: list[ProjectedRelationEdge] = []
    calls = relationships = imports = references = dropped_unmaterialized = 0
    seen: set[tuple[str, str, str]] = set()
    for relation in repository.relations:
        # scope_resolver relations are consumed by _build_calls_edges' in-file
        # path; only repository-resolved relations project here (unchanged
        # contract — REFERENCES reach this pass as repository_resolver).
        if relation.target_ref is not None or relation.provenance not in {
            "repository_resolver",
            "repository_import_resolver",
        }:
            continue
        edge_type = edge_types.get(relation.relation_type.name)
        if edge_type is None:
            continue
        key = (relation.from_symbol_id, relation.to_symbol_id, edge_type)
        if materialized_symbol_ids is not None and (
            key[0] not in materialized_symbol_ids or key[1] not in materialized_symbol_ids
        ):
            dropped_unmaterialized += 1
            continue
        if key in seen or key[0] == key[1]:
            continue
        seen.add(key)
        edges.append(
            ProjectedRelationEdge(
                src=key[0],
                dst=key[1],
                type=edge_type,
                provenance=getattr(relation, "provenance", None),
                confidence=getattr(relation, "confidence", None),
                use=getattr(relation, "context", None),
            )
        )
        if edge_type == "CALLS":
            calls += 1
        elif edge_type == "IMPORTS":
            imports += 1
        elif edge_type == "REFERENCES":
            references += 1
        else:
            relationships += 1
    return RepositoryRelationProjection(
        tuple(edges),
        calls,
        relationships,
        imports,
        references,
        dropped_unmaterialized,
    )


def project_file_symbols(parsed: Any) -> list[Any]:
    """Project canonical symbols without recovering identities from names/lines."""
    from dataclasses import asdict
    import hashlib
    from victor.storage.graph.protocol import GraphNode

    parents = {
        r.to_symbol_id: r.from_symbol_id
        for r in parsed.relations
        if r.relation_type.name == "CONTAINS" and r.target_ref is None
    }
    return [
        GraphNode(
            node_id=s.id,
            type=s.symbol_type.name.lower(),
            name=s.simple_name,
            file=s.location.file_path,
            line=s.location.start_line,
            end_line=s.location.end_line,
            lang=s.language,
            signature=s.signature,
            docstring=s.documentation,
            parent_id=parents.get(s.id),
            metadata={
                **s.metadata,
                "identity_version": s.identity_version,
                "source_span": asdict(s.location),
                "source_hash": hashlib.sha256(s.source_code.encode("utf-8")).hexdigest(),
                "parse_status": parsed.status.value,
                "diagnostics": [asdict(d) for d in parsed.diagnostics],
                "unresolved_relations": [
                    asdict(r)
                    for r in parsed.relations
                    if r.from_symbol_id == s.id and r.target_ref is not None
                    # Tier-2.5 retained REFERENCES stay relation-level evidence
                    # (capped per owner in victor-codegraph); mirroring them
                    # into every node's metadata would multiply the payload by
                    # the symbol count.
                    and r.relation_type.name != "REFERENCES"
                ],
            },
        )
        for s in parsed.symbols
    ]
