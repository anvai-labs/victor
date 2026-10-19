"""Python (stdlib ``ast``) adapter for tier-2.5 scope binding.

Extracts identifier-load occurrences owned by one symbol, mirroring the CALLS
collector's ownership discipline (``_calls_owned_by``): nested defs and classes
own their own occurrences. Shadow suppression reuses the per-symbol lexical
metadata the visitor already stamps (``python_bound_names``) plus the shared
noise roots — the in-file resolver's Python branch consults the same metadata,
so extraction-time skipping and resolution-time candidate filtering agree.

Callee positions and class bases are excluded here: they are CALLS/EXTENDS/
IMPLEMENTS territory and would duplicate those edges as REFERENCES.

Kept separate from :mod:`victor_codegraph.scope_binding` so the pure-Python
reference implementation has no tree-sitter imports, matching the package's
zero-dependency base install.
"""

from __future__ import annotations

import ast

from .model import CodeRelation, CodeSymbol
from .scope_binding import NOISE_ROOTS, ScopeOccurrence, reference_relation


def _name_loads_owned_by(node: ast.AST, excluded: set[int]):
    """Yield ``ast.Name`` loads owned by ``node`` without entering nested defs.

    ``excluded`` carries ``id()``s of callee expressions and class bases —
    positions owned by the CALLS/EXTENDS/IMPLEMENTS collectors.
    """

    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            if id(child) not in excluded:
                yield child
        stack.extend(ast.iter_child_nodes(child))


def collect_python_references(
    node: ast.AST,
    owner: CodeSymbol,
    file_path: str,
    relations: list[CodeRelation],
    *,
    excluded: set[int] | None = None,
) -> None:
    """Append REFERENCES relations for name loads owned by ``owner``.

    ``skip`` semantics: a name in ``python_bound_names`` (parameters, writes,
    enclosing bindings) is a local with no CodeSymbol to resolve to — emitting
    it would only flood the retained-unresolved set. Import aliases and in-file
    definition names are KEPT: those are exactly what in-file and cross-file
    resolution can bind. One relation per (owner, name) at the FIRST
    (lowest-line) use, emitted in source order.
    """

    skip = set(owner.metadata.get("python_bound_names", ())) | NOISE_ROOTS.get(
        "python", frozenset()
    )
    occurrences: dict[str, ScopeOccurrence] = {}
    for name_node in _name_loads_owned_by(node, excluded or set()):
        name = name_node.id
        if name in skip:
            continue
        line = getattr(name_node, "lineno", 0) or 0
        existing = occurrences.get(name)
        if existing is not None and existing.line <= line:
            continue
        occurrences[name] = ScopeOccurrence(owner_id=owner.id, name=name, line=line)
    for occurrence in sorted(occurrences.values(), key=lambda o: (o.line, o.name)):
        relations.append(reference_relation(occurrence, file_path))
