"""Shared in-file relation resolution for both parser paths.

Both the Python (stdlib ``ast``) and tree-sitter extractors first record relation
targets as *textual names* (the callee/base as written in source). This module
resolves those names to real in-file symbol ids where possible, with the same
semantics on every path:

- in-file match       -> ``to_symbol_id`` = the symbol's id, ``confidence`` = 1.0
- unresolved name     -> ``to_symbol_id`` = the bare name (RETAINED), ``confidence`` = 0.5
- self-reference      -> dropped (no self-edges, e.g. recursive calls)

Unresolved targets are kept on purpose: cross-file/external callees and bases are
what a CPG's blast-radius analysis needs; consumers build their own cross-file link
layer on top of the retained names.

Relations whose endpoints are already symbol ids (e.g. CONTAINS, built from the
symbol table directly) pass through untouched.
"""

from __future__ import annotations

from .model import CodeRelation, CodeRelationType, CodeSymbol

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
        Relations with name targets resolved (confidence 1.0), retained as bare
        names when external (confidence 0.5), and self-references dropped.
    """
    by_name: dict[str, str] = {s.simple_name: s.id for s in symbols}
    resolved: list[CodeRelation] = []
    for r in relations:
        if r.relation_type not in _NAME_TARGETED:
            resolved.append(r)
            continue
        target_id = by_name.get(r.to_symbol_id)
        if target_id == r.from_symbol_id:
            continue  # self-reference (recursive call) — emit no self-edge
        resolved.append(
            CodeRelation(
                from_symbol_id=r.from_symbol_id,
                to_symbol_id=target_id if target_id is not None else r.to_symbol_id,
                relation_type=r.relation_type,
                context=r.context,
                call_site=r.call_site,
                confidence=1.0 if target_id is not None else 0.5,
            )
        )
    return resolved
