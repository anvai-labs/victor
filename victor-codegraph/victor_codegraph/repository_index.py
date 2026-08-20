"""Repository-wide resolution and deletion-aware semantic index deltas."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from .model import (
    CodeRelation,
    CodeRelationType,
    CodeSymbol,
    CodeSymbolType,
    ParsedCode,
    SourceLocation,
    SymbolReference,
    deterministic_symbol_id,
    stable_symbol_key,
)
from .repo import iter_source_files, read_source_text


def relation_key(relation: CodeRelation) -> str:
    """Stable semantic edge key; source locations remain evidence, not identity."""

    raw = "\x1f".join(
        (
            relation.from_symbol_id,
            relation.to_symbol_id,
            relation.relation_type.name,
            relation.target_ref.text if relation.target_ref and relation.target_ref.text else "",
        )
    )
    return hashlib.blake2b(raw.encode(), digest_size=16).hexdigest()


@dataclass
class ParsedRepository:
    repo_id: str
    root: str
    files: dict[str, ParsedCode] = field(default_factory=dict)
    symbols: list[CodeSymbol] = field(default_factory=list)
    relations: list[CodeRelation] = field(default_factory=list)
    manifest: dict[str, str] = field(default_factory=dict)
    revision: str | None = None
    generation_id: str = ""


@dataclass
class IndexDelta:
    repo_id: str
    upsert_symbols: list[CodeSymbol] = field(default_factory=list)
    delete_symbol_ids: list[str] = field(default_factory=list)
    upsert_relations: list[CodeRelation] = field(default_factory=list)
    delete_relation_keys: list[str] = field(default_factory=list)
    files: dict[str, ParsedCode] = field(default_factory=dict)
    manifest: dict[str, str] = field(default_factory=dict)
    base_generation_id: str = ""
    target_generation_id: str = ""
    revision: str | None = None


def _resolve_across_files(
    symbols: list[CodeSymbol],
    relations: list[CodeRelation],
    files: dict[str, ParsedCode],
) -> list[CodeRelation]:
    """Resolve only unambiguous repository targets; ambiguity remains explicit."""

    # The scope resolver handles local relationships first. On the repository pass,
    # unique name + optional arity is a safe floor until import-aware language plugins
    # provide richer module bindings.
    by_name: dict[str, list[CodeSymbol]] = {}
    for symbol in symbols:
        by_name.setdefault(symbol.simple_name, []).append(symbol)
    source_symbols = {symbol.id: symbol for symbol in symbols}
    imported_paths: dict[str, set[str]] = {}
    for relative, parsed in files.items():
        paths: set[str] = set()
        if parsed.language == "python":
            for raw_import in parsed.imports:
                module = _python_import_module(raw_import)
                if module:
                    paths.update(
                        candidate
                        for candidate in _module_candidates(module, relative)
                        if candidate in files
                    )
        imported_paths[relative] = paths

    out: list[CodeRelation] = []
    for relation in relations:
        ref = relation.target_ref
        if ref is None or relation.relation_type not in {
            CodeRelationType.CALLS,
            CodeRelationType.EXTENDS,
            CodeRelationType.IMPLEMENTS,
        }:
            out.append(relation)
            continue
        candidates = list(by_name.get(ref.name, ()))
        source = source_symbols.get(relation.from_symbol_id)
        if source is None:
            out.append(relation)
            continue
        allowed = imported_paths.get(source.location.file_path, set())
        candidates = [
            candidate for candidate in candidates if candidate.location.file_path in allowed
        ]
        if ref.arity is not None:
            arity = [s for s in candidates if len(s.parameters) == ref.arity]
            if arity:
                candidates = arity
        if len(candidates) == 1:
            out.append(
                replace(
                    relation,
                    to_symbol_id=candidates[0].id,
                    target_ref=None,
                    confidence=1.0,
                    provenance="repository_resolver",
                )
            )
        else:
            out.append(relation)
    return out


def _python_import_module(raw: str) -> str | None:
    text = raw.strip()
    if text.startswith("from "):
        module, _, imported = text[5:].partition(" import ")
        module = module.strip()
        if module and not module.lstrip(".") and imported:
            module += imported.split(",", 1)[0].split(" as ", 1)[0].strip()
        return module
    if text.startswith("import "):
        return text[7:].split(",", 1)[0].split(" as ", 1)[0].strip()
    return None


def _module_candidates(module: str, importer: str) -> tuple[str, str]:
    level = len(module) - len(module.lstrip("."))
    tail = module.lstrip(".").replace(".", "/")
    if level:
        base = Path(importer).parent
        for _ in range(max(0, level - 1)):
            base = base.parent
        stem = (base / tail).as_posix()
    else:
        stem = tail
    return f"{stem}.py", f"{stem}/__init__.py"


def parse_repo(
    root: str | os.PathLike[str],
    *,
    repo_id: str | None = None,
    languages: list[str] | None = None,
    revision: str | None = None,
) -> ParsedRepository:
    """Parse one consistent read of every recognized source file and resolve the graph."""

    from .parser import parse

    root_path = Path(root).resolve()
    files: dict[str, ParsedCode] = {}
    manifest: dict[str, str] = {}
    for path in iter_source_files(root_path, languages=languages):
        content = read_source_text(path)
        if content is None:
            continue
        relative = path.resolve().relative_to(root_path).as_posix()
        parsed = parse(content, file_path=relative)
        files[relative] = parsed
        manifest[relative] = parsed.content_hash

    file_symbols: dict[str, CodeSymbol] = {}
    for relative, parsed in files.items():
        line_count = parsed.source_code.count("\n") + (
            0 if parsed.source_code.endswith("\n") else 1
        )
        file_symbol = CodeSymbol(
            id=stable_symbol_key("repository", CodeSymbolType.FILE, relative, None, relative),
            legacy_id=deterministic_symbol_id(relative, relative, 1),
            symbol_type=CodeSymbolType.FILE,
            fully_qualified_name=relative,
            simple_name=Path(relative).name,
            location=SourceLocation(
                file_path=relative,
                start_line=1,
                end_line=max(1, line_count),
                byte_offset=0,
                byte_length=len(parsed.source_code.encode("utf-8")),
            ),
            source_code="",
            language=parsed.language,
            metadata={"embedding_excluded": True, "content_version": parsed.content_hash},
        )
        file_symbols[relative] = file_symbol

    symbols = [*file_symbols.values(), *(s for parsed in files.values() for s in parsed.symbols)]
    relations = [relation for parsed in files.values() for relation in parsed.relations]
    for relative, parsed in files.items():
        owner = file_symbols[relative]
        for symbol in parsed.symbols:
            if not symbol.scope_chain:
                relations.append(
                    CodeRelation(
                        from_symbol_id=owner.id,
                        to_symbol_id=symbol.id,
                        relation_type=CodeRelationType.CONTAINS,
                        provenance="repository_structure",
                    )
                )
        for raw_import in parsed.imports:
            module = _python_import_module(raw_import) if parsed.language == "python" else None
            target = None
            if module:
                for candidate in _module_candidates(module, relative):
                    if candidate in file_symbols:
                        target = file_symbols[candidate]
                        break
            ref = SymbolReference(name=module or raw_import, text=raw_import)
            relations.append(
                CodeRelation(
                    from_symbol_id=owner.id,
                    to_symbol_id=target.id if target is not None else ref.name,
                    relation_type=CodeRelationType.IMPORTS,
                    target_ref=None if target is not None else ref,
                    confidence=1.0 if target is not None else 0.5,
                    provenance="repository_import_resolver",
                )
            )
    # Re-run the scope resolver over the full stable table only for already textual
    # targets, then apply conservative unique-name cross-file linking.
    relations = _resolve_across_files(symbols, relations, files)
    generation_raw = "\n".join(f"{path}\x1f{digest}" for path, digest in sorted(manifest.items()))
    generation_id = hashlib.blake2b(generation_raw.encode(), digest_size=16).hexdigest()
    return ParsedRepository(
        repo_id=repo_id or root_path.name,
        root=str(root_path),
        files=files,
        symbols=symbols,
        relations=relations,
        manifest=manifest,
        revision=revision,
        generation_id=generation_id,
    )


def diff_repository(current: ParsedRepository, previous: ParsedRepository) -> IndexDelta:
    """Create a deletion-complete semantic delta between repository snapshots."""

    if current.repo_id != previous.repo_id:
        raise ValueError("repository ids must match")
    old_symbols = {s.id: s for s in previous.symbols}
    new_symbols = {s.id: s for s in current.symbols}
    old_relations = {relation_key(r): r for r in previous.relations}
    new_relations = {relation_key(r): r for r in current.relations}

    changed_files = {
        path
        for path in set(current.manifest) | set(previous.manifest)
        if current.manifest.get(path) != previous.manifest.get(path)
    }
    source_files = {symbol.id: symbol.location.file_path for symbol in current.symbols}
    return IndexDelta(
        repo_id=current.repo_id,
        upsert_symbols=[
            s
            for key, s in new_symbols.items()
            if key not in old_symbols or s.location.file_path in changed_files
        ],
        delete_symbol_ids=sorted(set(old_symbols) - set(new_symbols)),
        upsert_relations=[
            r
            for key, r in new_relations.items()
            if key not in old_relations or source_files.get(r.from_symbol_id) in changed_files
        ],
        delete_relation_keys=sorted(set(old_relations) - set(new_relations)),
        files=dict(current.files),
        manifest=dict(current.manifest),
        base_generation_id=previous.generation_id,
        target_generation_id=current.generation_id,
        revision=current.revision,
    )


def apply_index_delta(previous: ParsedRepository, delta: IndexDelta) -> ParsedRepository:
    """Pure reference implementation used to verify incremental/full equivalence."""

    if previous.repo_id != delta.repo_id:
        raise ValueError("repository ids must match")
    if delta.base_generation_id and previous.generation_id != delta.base_generation_id:
        raise ValueError("index delta base generation does not match the current snapshot")
    symbols = {s.id: s for s in previous.symbols}
    for symbol_id in delta.delete_symbol_ids:
        symbols.pop(symbol_id, None)
    symbols.update((s.id, s) for s in delta.upsert_symbols)

    relations = {relation_key(r): r for r in previous.relations}
    for key in delta.delete_relation_keys:
        relations.pop(key, None)
    relations.update((relation_key(r), r) for r in delta.upsert_relations)
    return ParsedRepository(
        repo_id=previous.repo_id,
        root=previous.root,
        files=dict(delta.files),
        symbols=list(symbols.values()),
        relations=list(relations.values()),
        manifest=dict(delta.manifest),
        revision=delta.revision,
        generation_id=delta.target_generation_id,
    )
