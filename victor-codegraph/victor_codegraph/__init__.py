"""victor-codegraph — shared repository-to-Code-Semantic-Graph compiler.

One tree-sitter symbol+relation chunker, three consumers (Victor, ProximaDB SDK,
AnvaiOps). See ProximaDB ADR-029 / Victor ADR-014.

    from victor_codegraph import chunk, parse, to_proxima_records, ChunkConfig

    chunks = chunk(source, file_path="foo.py")          # size-capped, embeddable
    parsed = parse(source, file_path="foo.py")           # symbols + relations
    records = to_proxima_records(parsed, repo_graph_id="myrepo")
"""

from __future__ import annotations

from .adapter import relation_to_record, symbol_to_record, to_proxima_records
from .config import ChunkConfig
from .languages import detect_language
from .model import (
    CapabilityTier,
    LINE_BASE,
    CodeChunk,
    CodeRelation,
    CodeRelationType,
    CodeSymbol,
    CodeSymbolType,
    ParsedCode,
    ParseDiagnostic,
    ParseStatus,
    SourceLocation,
    SymbolReference,
    deterministic_symbol_id,
    stable_symbol_oid,
    stable_symbol_key,
)
from .manifest import ManifestDiff, build_manifest, diff_manifest, iter_changed_files
from .parser import chunk, parse
from .repo import chunk_path, chunk_repo, iter_source_files, parse_path, read_source_text
from .resolution import resolve_relations
from .repository_index import (
    IndexDelta,
    ParsedRepository,
    apply_index_delta,
    diff_repository,
    parse_repo,
    relation_key,
)

__version__ = "0.9.0"

__all__ = [
    "__version__",
    "chunk",
    "parse",
    "chunk_repo",
    "chunk_path",
    "parse_path",
    "iter_source_files",
    "read_source_text",
    "build_manifest",
    "diff_manifest",
    "iter_changed_files",
    "ManifestDiff",
    "ChunkConfig",
    "LINE_BASE",
    "detect_language",
    "to_proxima_records",
    "symbol_to_record",
    "relation_to_record",
    "CodeChunk",
    "CodeSymbol",
    "CodeRelation",
    "CodeSymbolType",
    "CodeRelationType",
    "ParsedCode",
    "ParsedRepository",
    "IndexDelta",
    "ParseStatus",
    "CapabilityTier",
    "ParseDiagnostic",
    "SymbolReference",
    "SourceLocation",
    "stable_symbol_oid",
    "stable_symbol_key",
    "deterministic_symbol_id",
    "resolve_relations",
    "parse_repo",
    "diff_repository",
    "apply_index_delta",
    "relation_key",
]
