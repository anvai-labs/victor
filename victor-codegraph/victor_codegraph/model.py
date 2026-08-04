"""Canonical, neutral data model for code symbols, relations, and chunks.

This is the *union* of the two donor implementations (ProximaDB SDK ``code.py`` and
Victor ``victor-coding``): ProximaDB contributed the richer symbol/relation taxonomy;
Victor contributed the size-aware ``CodeChunk`` with hierarchical IDs. The model carries
no SaaS/DB/framework concept, so every consumer can depend on it freely.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL LINE-NUMBER CONTRACT
#
# Every line number this package emits (``SourceLocation.start_line`` / ``end_line``,
# ``CodeChunk.metadata['start_line']``/``['end_line']``, the ``call_site`` line on a
# ``CodeRelation``) is **1-based** — line 1 is the first line of the file. This matches
# editors (Monaco/VS Code), pgwire, and how humans reference code.
#
# victor-codegraph is the SHARED parser across Victor, the ProximaDB SDK, and AnvaiOps,
# so this convention is the cross-surface contract: a symbol's ``(file, name, line)``
# coordinate must be identical in every consumer for the correlated code-graph to
# reconcile. Consumers MUST pass these line numbers through unchanged — do NOT convert
# to 0-based (a `- 1` shift) for an internal/tree-sitter-compat convention; that breaks
# cross-surface reconciliation. Byte offsets (``byte_offset``/``start_pos``) are 0-based.
LINE_BASE = 1
# ─────────────────────────────────────────────────────────────────────────────


class CodeSymbolType(IntEnum):
    """Kinds of code symbol that can be extracted (superset across languages)."""

    FILE = 1
    MODULE = 2
    PACKAGE = 3
    CLASS = 4
    INTERFACE = 5
    TRAIT = 6
    STRUCT = 7
    ENUM = 8
    FUNCTION = 9
    METHOD = 10
    CONSTRUCTOR = 11
    PROPERTY = 12
    FIELD = 13
    CONSTANT = 14
    VARIABLE = 15
    PARAMETER = 16
    TYPE_ALIAS = 17
    MACRO = 18


class CodeRelationType(IntEnum):
    """Directed relationships between code symbols."""

    CALLS = 1
    CALLED_BY = 2
    EXTENDS = 3
    IMPLEMENTS = 4
    USES_TYPE = 5
    RETURNS_TYPE = 6
    IMPORTS = 7
    IMPORTED_BY = 8
    DEPENDS_ON = 9
    CONTAINS = 10
    CONTAINED_BY = 11
    DEFINES = 12
    REFERENCES = 13
    REFERENCED_BY = 14
    OVERRIDES = 15
    OVERRIDDEN_BY = 16
    TESTS = 17
    TESTED_BY = 18


class ParseStatus(str, Enum):
    """Observable outcome of a language frontend."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FALLBACK = "fallback"
    ERROR = "error"


class CapabilityTier(str, Enum):
    """Honest extraction fidelity advertised for a parse result."""

    FULL = "full"
    SYMBOLS = "symbols"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class ParseDiagnostic:
    """Machine-readable explanation of degraded or partial extraction."""

    code: str
    message: str
    severity: str = "warning"
    line: int | None = None
    column: int | None = None


@dataclass(frozen=True)
class SymbolReference:
    """A structured target that has not yet resolved to a local symbol."""

    name: str
    qualifier: str | None = None
    arity: int | None = None
    text: str | None = None


@dataclass
class SourceLocation:
    """Where a symbol lives in source.

    Lines (``start_line``/``end_line``) are **1-based** (see ``LINE_BASE`` and the
    canonical line-number contract at the top of this module); byte offsets
    (``byte_offset``/``byte_length``) are 0-based. Pass line numbers through
    unchanged — do not convert to 0-based.
    """

    file_path: str
    start_line: int = 0
    start_column: int = 0
    end_line: int = 0
    end_column: int = 0
    byte_offset: int = 0
    byte_length: int = 0


@dataclass
class CodeSymbol:
    """A semantic code unit (function/class/method/struct/...)."""

    id: str
    symbol_type: CodeSymbolType
    fully_qualified_name: str
    simple_name: str
    location: SourceLocation
    source_code: str
    language: str
    documentation: str | None = None
    signature: str | None = None
    modifiers: list[str] = field(default_factory=list)
    scope_chain: list[str] = field(default_factory=list)
    parameters: list[dict[str, Any]] = field(default_factory=list)
    return_type: str | None = None
    complexity: dict[str, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # v2 identity is structural and line-independent. ``legacy_id`` retains the
    # pre-v2 (path, name, line, column) alias for mixed reads.
    legacy_id: str = ""
    identity_version: str = "v2"


@dataclass
class CodeRelation:
    """A directed edge between two symbols (by id)."""

    from_symbol_id: str
    to_symbol_id: str
    relation_type: CodeRelationType
    call_site: SourceLocation | None = None
    context: str | None = None
    confidence: float = 1.0
    target_ref: SymbolReference | None = None
    provenance: str = "parser"


@dataclass
class ParsedCode:
    """Result of parsing one source file."""

    file_path: str
    language: str
    symbols: list[CodeSymbol] = field(default_factory=list)
    relations: list[CodeRelation] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    content_hash: str = ""
    status: ParseStatus = ParseStatus.SUCCESS
    capability_tier: CapabilityTier = CapabilityTier.SYMBOLS
    diagnostics: list[ParseDiagnostic] = field(default_factory=list)
    source_code: str = ""


@dataclass
class CodeChunk:
    """An embeddable, size-capped chunk projected from a symbol.

    A symbol within the size budget yields exactly one chunk; an oversized symbol is
    body-split into several chunks sharing ``symbol_id`` (see ``sizing``). ``chunk_id``
    is hierarchical and deterministic so incremental re-index is an idempotent upsert.
    """

    chunk_id: str
    text: str
    symbol_id: str
    start_pos: int
    end_pos: int
    metadata: dict[str, Any] = field(default_factory=dict)


def deterministic_symbol_id(file_path: str, name: str, line: int, column: int = 0) -> str:
    """Stable 16-hex id keyed on (file, name, line, col) — same input, same id.

    LINE-COUPLED: editing code above a symbol moves its line and changes this id.
    Retained as the *legacy* alias during the ADR-044 mixed-read bake; new identity
    is :func:`stable_symbol_oid`.
    """

    key = f"{file_path}:{name}:{line}:{column}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def stable_symbol_oid(
    repo_id: str,
    language: str,
    fully_qualified_name: str,
    signature: str | None,
) -> str:
    """Line-independent stable symbol id (ADR-044) — the correlated-CPG join key.

    Identity = the structural coordinates that survive edits which don't change *what
    the symbol is*: ``repo ‖ language ‖ fully_qualified_name ‖ overload_discriminator``.
    The FQN already encodes ``path::scope::name`` (no line), and ``signature`` (the
    parameter list / arity) disambiguates same-name overloads. **Line, column, byte
    offset and body text are NOT in identity** — they are metadata / a separate
    ``content_version``. So a symbol that merely moves lines (or whose body is edited)
    keeps this id; it changes only on rename / move / signature change.

    Uses ``blake2b`` (stdlib, dependency-free) and the unit-separator ``\\x1f`` as the
    field delimiter so a coordinate containing the delimiter can't forge a collision.
    """

    disc = signature or ""
    key = "\x1f".join((repo_id, language, fully_qualified_name, disc))
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()


def normalize_signature(signature: str | None) -> str:
    """Normalize an overload discriminator without changing quoted text.

    Tree-sitter frontends expose source text rather than a typed signature IR. Removing
    insignificant whitespace outside string literals provides stable identity across
    formatting edits while preserving defaults and literal types that may be semantic.
    """

    if not signature:
        return ""
    out: list[str] = []
    quote: str | None = None
    escaped = False
    for char in signature:
        if quote is not None:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
            out.append(char)
        elif not char.isspace():
            out.append(char)
    return "".join(out)


def stable_symbol_key(
    language: str,
    symbol_type: CodeSymbolType,
    fully_qualified_name: str,
    signature: str | None,
    file_path: str = "",
) -> str:
    """v2 repository-local structural key shared by symbols, chunks, and adapters."""

    key = "\x1f".join(
        (
            "v2",
            language,
            symbol_type.name,
            file_path,
            fully_qualified_name,
            normalize_signature(signature),
        )
    )
    return hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()


def content_hash(content: str) -> str:
    """SHA-256 of file content, for change detection on the re-index hot path."""

    return hashlib.sha256(content.encode("utf-8")).hexdigest()
