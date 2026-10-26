"""Repository-wide resolution and deletion-aware semantic index deltas."""

from __future__ import annotations

import ast
import hashlib
import os
import stat
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .rust_repository import RustBinding

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
from .languages import detect_language
from .javascript_repository import JavaScriptRepository


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
    rust_bindings: dict[str, list[RustBinding]] | None = None,
    javascript: JavaScriptRepository | None = None,
) -> list[CodeRelation]:
    """Resolve only unambiguous repository targets; ambiguity remains explicit."""

    # File-local resolution runs first; repository edges require exact import bindings.
    by_name: dict[str, list[CodeSymbol]] = {}
    for symbol in symbols:
        by_name.setdefault(symbol.simple_name, []).append(symbol)
    source_symbols = {symbol.id: symbol for symbol in symbols}
    by_path: dict[tuple[str, tuple[str, ...]], list[CodeSymbol]] = {}
    for symbol in symbols:
        by_path.setdefault(
            (symbol.location.file_path, (*symbol.scope_chain, symbol.simple_name)), []
        ).append(symbol)
    python_bindings = {
        path: _python_bindings(parsed.imports, path, files)
        for path, parsed in files.items()
        if parsed.language == "python"
    }

    out: list[CodeRelation] = []
    for relation in relations:
        ref = relation.target_ref
        if ref is None or relation.relation_type not in {
            CodeRelationType.CALLS,
            CodeRelationType.EXTENDS,
            CodeRelationType.IMPLEMENTS,
            CodeRelationType.REFERENCES,
        }:
            out.append(relation)
            continue
        candidates = list(by_name.get(ref.name, ()))
        source = source_symbols.get(relation.from_symbol_id)
        if source is None:
            out.append(relation)
            continue
        if source.language == "rust":
            candidates = []
            # Imports currently represent file-level declarations. Rust modules
            # do not inherit their parent's use bindings.
            module_scoped = any(
                symbol.symbol_type == CodeSymbolType.MODULE
                and source.scope_chain[: len(symbol.scope_chain) + 1]
                == [*symbol.scope_chain, symbol.simple_name]
                for symbol in files[source.location.file_path].symbols
            )
            # Only explicit imports, with exact path + alias matching. Unknown
            # receivers, globs and re-exports retain their unresolved references.
            raw_path = (ref.text or ref.name).split("::")
            for binding in (rust_bindings or {}).get(source.location.file_path, []):
                if module_scoped or source.metadata.get("rust_conditional"):
                    break
                if binding.name == "*" or raw_path[0] != binding.name:
                    continue
                target_path = (*binding.path, *raw_path[1:])
                # by_path already indexes exactly (file, path); the linear
                # scan over ALL repository symbols per relation x binding
                # made the repository pass quadratic on rust-heavy repos.
                for candidate in by_path.get((binding.file, target_path), ()):
                    if not candidate.metadata.get("rust_conditional"):
                        candidates.append(candidate)
            candidates = list({candidate.id: candidate for candidate in candidates}.values())
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
            continue
        candidates = []
        if source.language == "python":
            text = ref.text or ref.name
            root, *tail = text.split(".")
            if (
                not source.metadata.get("python_wildcard_import")
                and root not in source.metadata.get("python_bound_names", [])
                and root not in source.metadata.get("python_definition_scopes", {})
            ):
                for import_name, targets in python_bindings.get(
                    source.location.file_path, {}
                ).items():
                    prefix = import_name.split(".")
                    parts = text.split(".")
                    if parts[: len(prefix)] != prefix:
                        continue
                    for file, path in targets:
                        target_path = (*path, *parts[len(prefix) :])
                        candidates.extend(
                            s
                            for s in by_path.get((file, target_path), [])
                            if not set(target_path).intersection(
                                s.metadata.get("python_rebound_names", [])
                            )
                        )
        if source.language in {"javascript", "typescript", "tsx"} and javascript is not None:
            text = ref.text or ref.name
            if text.split(".")[0] not in source.metadata.get("javascript_bound_names", []):
                candidates = javascript.target(source.location.file_path, text)
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


def _python_bindings(
    imports: list[str], importer: str, files: dict[str, ParsedCode]
) -> dict[str, list[tuple[str, tuple[str, ...]]]]:
    """Bind exact imported names; no glob, receiver inference or bare-name lookup."""
    bindings: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for raw in imports:
        node = ast.parse(raw).body[0]
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            if isinstance(node, ast.ImportFrom):
                module = "." * node.level + (node.module or "")
                name = alias.asname or alias.name
                target: tuple[str, ...] = (alias.name,)
                options = [
                    (module, target),
                    (module + ("." if node.module else "") + alias.name, ()),
                ]
            else:
                name = alias.asname or alias.name.split(".")[0]
                prefix = () if alias.asname else tuple(alias.name.split(".")[1:])
                options = [(alias.name, ())]
                # Unaliased import a.b binds a, but only a.b's members are proven.
                if prefix:
                    name = alias.name
            for module, target in options:
                matches = [p for p in _module_candidates(module, importer) if p in files]
                for file in matches:
                    bindings.setdefault(name, []).append((file, target))
    return bindings


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


def _module_candidates(module: str, importer: str) -> tuple[str, ...]:
    level = len(module) - len(module.lstrip("."))
    tail = module.lstrip(".").replace(".", "/")
    if level:
        base = Path(importer).parent
        if level > len(base.parts):
            return ()
        for _ in range(max(0, level - 1)):
            base = base.parent
        stem = (base / tail).as_posix()
    else:
        stem = tail
    return f"{stem}.py", f"{stem}/__init__.py"


def _file_projection_metadata(parsed: ParsedCode) -> dict[str, object]:
    """Keep diagnostic evidence while quarantining ambiguous symbol identities."""
    quarantined = any(item.code == "identity_collision" for item in parsed.diagnostics)
    groups: dict[str, list[CodeSymbol]] = {}
    if quarantined:
        for symbol in parsed.symbols:
            groups.setdefault(symbol.id, []).append(symbol)
    return {
        "parse_status": parsed.status.value,
        "capability_tier": parsed.capability_tier.value,
        "diagnostics": [asdict(item) for item in parsed.diagnostics],
        "projection_status": "file_marker_only" if quarantined else "symbols",
        "omitted_symbol_count": len(parsed.symbols) if quarantined else 0,
        "omitted_relation_count": len(parsed.relations) if quarantined else 0,
        "identity_collisions": [
            {
                "symbol_id": symbol_id,
                "resolution": "ambiguous",
                "declarations": [
                    {
                        "name": symbol.simple_name,
                        "qualified_name": symbol.fully_qualified_name,
                        "symbol_type": symbol.symbol_type.name.lower(),
                        "source_span": asdict(symbol.location),
                    }
                    for symbol in declarations
                ],
            }
            for symbol_id, declarations in groups.items()
            if len(declarations) > 1
        ],
    }


def _selected_source_paths(
    root: Path,
    selected_paths: Iterable[str | os.PathLike[str]],
    languages: list[str] | None,
) -> list[Path]:
    """Validate the whole explicit inventory before reading any source content."""
    if isinstance(selected_paths, (str, bytes, os.PathLike)):
        raise ValueError("selected inventory must be an iterable of file paths, not one path")
    if not root.is_dir():
        raise ValueError("selected inventory requires an existing repository directory")
    paths: set[Path] = set()
    for selected in selected_paths:
        path = (root / selected).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"selected inventory path is outside repository: {selected}")
        try:
            mode = path.stat().st_mode
        except FileNotFoundError:
            pass  # read_source_text preserves missing-input evidence by omission.
        else:
            if not stat.S_ISREG(mode):
                raise ValueError(f"selected inventory path is not a regular file: {selected}")
        paths.add(path)
    allowed = set(languages) if languages else None
    return [
        path
        for path in sorted(paths, key=lambda p: (p.relative_to(root).parent.parts, p.name))
        if allowed is None or detect_language(str(path)) in allowed
    ]


def parse_repo(
    root: str | os.PathLike[str],
    *,
    repo_id: str | None = None,
    languages: list[str] | None = None,
    revision: str | None = None,
    selected_paths: Iterable[str | os.PathLike[str]] | None = None,
    emit_references: bool = False,
) -> ParsedRepository:
    """Read each selected source once and resolve the repository graph.

    With ``selected_paths=None``, retain the default language-aware walker and
    its exclusions. An explicit iterable (including an empty one) replaces that
    discovery policy; hidden/build paths are permitted. Paths may be relative to
    root or absolute within root. Resolve/deduplicate aliases and validate the
    entire inventory before source reads; non-regular files and paths escaping root
    are rejected. ``languages`` still filters the resulting inventory.

    ``emit_references=True`` turns on tier-2.5 scope binding (REFERENCES
    relations; ADR-032) for every parsed file; the default output is unchanged.

    Missing files and source-content read failures remain absent from ``files``
    and ``manifest``, as in default discovery. Path/type validation errors propagate
    before source reads. Consumers requiring complete preparation must compare the
    selected inventory with the returned manifest before publishing. This is not
    a filesystem snapshot or protection against concurrent path replacement.
    """

    from .parser import parse

    root_path = Path(root).resolve()
    files: dict[str, ParsedCode] = {}
    manifest: dict[str, str] = {}
    paths = (
        iter_source_files(root_path, languages=languages)
        if selected_paths is None
        else _selected_source_paths(root_path, selected_paths, languages)
    )
    for path in paths:
        content = read_source_text(path)
        if content is None:
            continue
        relative = path.resolve().relative_to(root_path).as_posix()
        parsed = parse(content, file_path=relative, emit_references=emit_references)
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
            metadata={
                "embedding_excluded": True,
                "content_version": parsed.content_hash,
                **_file_projection_metadata(parsed),
            },
        )
        file_symbols[relative] = file_symbol

    # Retain raw ParsedCode in `files` for diagnostics, but never feed ambiguous
    # declarations into symbol lookup or cross-file resolution. The canonical
    # FILE node stays visible, including to healthy files importing this module.
    resolution_files = {
        path: parsed
        for path, parsed in files.items()
        if file_symbols[path].metadata["projection_status"] != "file_marker_only"
    }
    symbols = [
        *file_symbols.values(),
        *(symbol for parsed in resolution_files.values() for symbol in parsed.symbols),
    ]
    relations = [relation for parsed in resolution_files.values() for relation in parsed.relations]
    from .rust_repository import RustRepository, use_paths

    rust = RustRepository(root_path, resolution_files)
    javascript = JavaScriptRepository(resolution_files)
    rust_bindings: dict[str, list[RustBinding]] = {}
    for relative, parsed in resolution_files.items():
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
            if parsed.language in {"javascript", "typescript", "tsx"}:
                matches = [
                    target
                    for raw, target in javascript.dependencies.get(relative, [])
                    if raw == raw_import
                ]
                js_target = matches[0] if len(matches) == 1 else None
                relations.append(
                    CodeRelation(
                        from_symbol_id=owner.id,
                        to_symbol_id=file_symbols[js_target].id if js_target else raw_import,
                        relation_type=CodeRelationType.IMPORTS,
                        target_ref=(
                            None if js_target else SymbolReference(name=raw_import, text=raw_import)
                        ),
                        confidence=1.0 if js_target else 0.5,
                        provenance="repository_import_resolver",
                    )
                )
                continue
            if parsed.language == "rust":
                for imported in use_paths(raw_import):
                    parts, name = imported.path, imported.name
                    binding = None if imported.conditional else rust.resolve(relative, parts, name)
                    if binding is None:
                        ref = SymbolReference(name="::".join(parts), text=raw_import)
                        relations.append(
                            CodeRelation(
                                from_symbol_id=owner.id,
                                to_symbol_id=ref.name,
                                relation_type=CodeRelationType.IMPORTS,
                                target_ref=ref,
                                confidence=0.5,
                                provenance="repository_import_resolver",
                            )
                        )
                        continue
                    rust_bindings.setdefault(relative, []).append(binding)
                    relations.append(
                        CodeRelation(
                            from_symbol_id=owner.id,
                            to_symbol_id=file_symbols[binding.file].id,
                            relation_type=CodeRelationType.IMPORTS,
                            provenance="repository_import_resolver",
                        )
                    )
                continue
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
    relations = _resolve_across_files(
        symbols, relations, resolution_files, rust_bindings, javascript
    )
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
