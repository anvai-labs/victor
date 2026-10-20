"""Shared in-file relation resolution for both parser paths.

Both the Python (stdlib ``ast``) and tree-sitter extractors first record relation
targets as *textual names* (the callee/base as written in source). This module
resolves those names to real in-file symbol ids where possible, with the same
semantics on every path:

- in-file match       -> ``to_symbol_id`` = the symbol's id, ``confidence`` = 1.0
- unresolved name     -> ``to_symbol_id`` = the bare name (RETAINED), ``confidence`` = 0.5
- self-reference      -> retained (recursion is graph evidence)

Unresolved targets are kept on purpose: cross-file/external callees and bases are
what a CPG's blast-radius analysis needs; consumers build their own cross-file link
layer on top of the retained names.

Relations whose endpoints are already symbol ids (e.g. CONTAINS, built from the
symbol table directly) pass through untouched.
"""

from __future__ import annotations

from .model import CodeRelation, CodeRelationType, CodeSymbol, CodeSymbolType, SymbolReference

# Relation types whose ``to_symbol_id`` starts life as a textual name.
# REFERENCES joins the set for tier-2.5 scope binding (ADR-032): occurrence
# extraction emits textual names and this resolver binds them, exactly like a
# pre-resolution CALLS target.
_NAME_TARGETED = frozenset(
    {
        CodeRelationType.CALLS,
        CodeRelationType.EXTENDS,
        CodeRelationType.IMPLEMENTS,
        CodeRelationType.REFERENCES,
    }
)


def resolve_relations(
    symbols: list[CodeSymbol],
    relations: list[CodeRelation],
) -> list[CodeRelation]:
    """Resolve textual relation targets to in-file symbol ids.

    Args:
        symbols: All symbols extracted from the file.
        relations: Raw relations; CALLS/EXTENDS/IMPLEMENTS carry a textual
            target name in ``to_symbol_id``, other types carry real ids.

    Returns:
        Relations with name targets resolved (confidence 1.0) or retained as a
        structured unresolved reference (confidence 0.5). Recursive calls remain.
    """
    by_name: dict[str, list[CodeSymbol]] = {}
    by_id = {s.id: s for s in symbols}
    for symbol in symbols:
        by_name.setdefault(symbol.simple_name, []).append(symbol)
    resolved: list[CodeRelation] = []
    for r in relations:
        if r.relation_type not in _NAME_TARGETED:
            resolved.append(r)
            continue
        if r.to_symbol_id in by_id:
            resolved.append(r)
            continue
        ref = r.target_ref or SymbolReference(name=r.to_symbol_id, text=r.context)
        candidates = list(by_name.get(ref.name, ()))
        source = by_id.get(r.from_symbol_id)

        if ref.arity is not None and (
            source is None or source.language not in {"python", "javascript", "typescript", "tsx"}
        ):
            arity_matches = [c for c in candidates if len(c.parameters) == ref.arity]
            if arity_matches:
                candidates = arity_matches

        if source is not None and source.language == "rust":
            candidates = _rust_candidates(source, ref, candidates, symbols)
        elif source is not None and source.language == "python":
            definition_scope = source.metadata.get("python_definition_scopes", {}).get(ref.name)
            if source.metadata.get("python_wildcard_import"):
                candidates = []
            elif ref.qualifier in {"self", "cls"}:
                candidates = [
                    c
                    for c in candidates
                    if source.scope_chain
                    and ref.qualifier == source.metadata.get("python_receiver")
                    and ref.qualifier not in source.metadata.get("python_rebound_names", [])
                    and c.scope_chain == source.scope_chain
                    and c.symbol_type == CodeSymbolType.METHOD
                ]
            elif (
                ref.qualifier
                or ref.name in source.metadata.get("python_bound_names", [])
                or (
                    ref.name in source.metadata.get("python_import_names", [])
                    and definition_scope is None
                )
            ):
                candidates = []
            else:
                # Python class namespaces are not enclosing function namespaces.
                candidates = [
                    c
                    for c in candidates
                    if c.symbol_type != CodeSymbolType.METHOD
                    and not any(
                        parent.symbol_type == CodeSymbolType.CLASS
                        and [*parent.scope_chain, parent.simple_name] == c.scope_chain
                        for parent in symbols
                    )
                    and [*source.scope_chain, source.simple_name][: len(c.scope_chain)]
                    == c.scope_chain
                ]
                # A lexical definition shadows imports and outer definitions even
                # when its conditional body is not represented by this parser.
                if definition_scope is not None:
                    candidates = [c for c in candidates if c.scope_chain == definition_scope]
                nearest = max((len(c.scope_chain) for c in candidates), default=-1)
                candidates = [c for c in candidates if len(c.scope_chain) == nearest]
        elif source is not None and source.language in {"javascript", "typescript", "tsx"}:
            candidates = [c for c in candidates if not c.metadata.get("javascript_conditional")]
            if ref.qualifier == "this" and source.symbol_type == CodeSymbolType.METHOD:
                candidates = [c for c in candidates if c.scope_chain == source.scope_chain]
            elif ref.qualifier or ref.name in source.metadata.get("javascript_bound_names", []):
                candidates = []
            else:
                candidates = [
                    c
                    for c in candidates
                    if c.symbol_type != CodeSymbolType.METHOD
                    and [*source.scope_chain, source.simple_name][: len(c.scope_chain)]
                    == c.scope_chain
                ]
                nearest = max((len(c.scope_chain) for c in candidates), default=-1)
                candidates = [c for c in candidates if len(c.scope_chain) == nearest]
        elif source is not None and candidates:
            qualifier = (ref.qualifier or "").split(".")[-1]
            receiver_var = source.metadata.get("go_receiver_var")
            if qualifier in {"self", "cls", "this", "super"} or (
                receiver_var is not None and qualifier == receiver_var
            ):
                same_scope = [c for c in candidates if c.scope_chain == source.scope_chain]
                if same_scope:
                    candidates = same_scope
            else:
                # Prefer the nearest lexical scope, but never choose arbitrarily among
                # equally good overloads/collisions.
                def score(candidate: CodeSymbol) -> int:
                    return sum(
                        1 for a, b in zip(candidate.scope_chain, source.scope_chain) if a == b
                    )

                best = max((score(c) for c in candidates), default=-1)
                candidates = [c for c in candidates if score(c) == best]

        target = candidates[0] if len(candidates) == 1 else None
        resolved.append(
            CodeRelation(
                from_symbol_id=r.from_symbol_id,
                to_symbol_id=target.id if target is not None else ref.name,
                relation_type=r.relation_type,
                context=r.context,
                call_site=r.call_site,
                confidence=1.0 if target is not None else 0.5,
                target_ref=None if target is not None else ref,
                provenance="scope_resolver" if target is not None else r.provenance,
            )
        )
    return resolved


def _rust_candidates(
    source: CodeSymbol,
    ref: SymbolReference,
    candidates: list[CodeSymbol],
    symbols: list[CodeSymbol],
) -> list[CodeSymbol]:
    """Respect explicit Rust paths; unknown receivers never imply a local type."""
    qualifier = ref.qualifier
    if source.metadata.get("rust_conditional"):
        return []
    candidates = [c for c in candidates if not c.metadata.get("rust_conditional")]
    scope = source.scope_chain
    module_scope: list[str] = []
    for symbol in symbols:
        path = [*symbol.scope_chain, symbol.simple_name]
        if (
            symbol.symbol_type == CodeSymbolType.MODULE
            and scope[: len(path)] == path
            and len(path) > len(module_scope)
        ):
            module_scope = path
    if qualifier in {"self", "Self"} and ref.text and "." in ref.text:
        return [c for c in candidates if c.scope_chain == scope]
    if qualifier and ref.text and "." in ref.text:
        return []
    if qualifier:
        parts = qualifier.split("::")
        if parts[0] == "crate":
            if source.location.file_path.rsplit("/", 1)[-1] not in {"lib.rs", "main.rs"}:
                return []
            paths = [parts[1:]]
        elif parts[0] == "self":
            paths = [module_scope + parts[1:]]
        elif parts[0] == "Self":
            paths = [scope + parts[1:]]
        elif parts[0] == "super":
            base = list(module_scope)
            while parts and parts[0] == "super":
                if not base:
                    return []
                base.pop()
                parts.pop(0)
            paths = [base + parts]
        else:
            paths = [module_scope + parts]
    else:
        paths = [scope, module_scope]
    for path in paths:
        matches = [c for c in candidates if c.scope_chain == path]
        if matches:
            return matches
    return []
