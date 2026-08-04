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

from .model import CodeRelation, CodeRelationType, CodeSymbol, SymbolReference

# Relation types whose ``to_symbol_id`` starts life as a textual name.
_NAME_TARGETED = frozenset(
    {
        CodeRelationType.CALLS,
        CodeRelationType.EXTENDS,
        CodeRelationType.IMPLEMENTS,
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

        if ref.arity is not None:
            arity_matches = [c for c in candidates if len(c.parameters) == ref.arity]
            if arity_matches:
                candidates = arity_matches

        if source is not None and candidates:
            qualifier = (ref.qualifier or "").split(".")[-1]
            if qualifier in {"self", "cls", "this", "super"}:
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
