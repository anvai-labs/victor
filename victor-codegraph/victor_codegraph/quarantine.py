"""Symbol-level quarantine for identity collisions.

A collision (two declarations sharing one structural key) makes those
declarations ambiguous, not their whole file. The ambiguous declarations, and
whatever they contain, stay out of the graph and out of resolution; the file's
other declarations are projected as usual.

The view is derived from the parse and never re-runs resolution: the in-file
resolver already left a reference to an ambiguous name unresolved, and
resolving again over fewer candidates would bind it to a namesake elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .model import CodeRelationType, CodeSymbol, ParsedCode


@dataclass(frozen=True)
class QuarantineView:
    """What stays out of a file's projection, and the parse without it."""

    parsed: ParsedCode
    symbol_ids: frozenset[str]
    names: frozenset[str]
    dropped_relations: int
    declared: int
    # Scope paths (``(*scope_chain, name)``) at which a left-out declaration
    # lived. A surviving namesake at such a path is not evidence for a
    # cross-file binding: the file declared more than one thing there.
    paths: frozenset[tuple[str, ...]] = frozenset()

    @property
    def leaves_nothing(self) -> bool:
        """True when the file declared symbols and none of them survives."""
        return self.declared > 0 and not self.parsed.symbols

    @property
    def omitted_symbols(self) -> int:
        """Declarations left out; colliding declarations share one key."""
        return self.declared - len(self.parsed.symbols)


def collision_groups(symbols: list[CodeSymbol]) -> dict[str, list[CodeSymbol]]:
    """Declarations that share a structural key, keyed by that key."""
    groups: dict[str, list[CodeSymbol]] = {}
    for symbol in symbols:
        groups.setdefault(symbol.id, []).append(symbol)
    return {symbol_id: group for symbol_id, group in groups.items() if len(group) > 1}


def _span(symbol: CodeSymbol) -> tuple[tuple[int, int], tuple[int, int]]:
    location = symbol.location
    return (location.start_line, location.start_column), (location.end_line, location.end_column)


def _inside(inner: CodeSymbol, outer: CodeSymbol) -> bool:
    (start, end), (outer_start, outer_end) = _span(inner), _span(outer)
    return (
        inner is not outer and outer_start <= start and end <= outer_end and outer_end > outer_start
    )


def quarantine_view(parsed: ParsedCode) -> QuarantineView:
    """The ambiguous declarations of one parse, and the parse without them.

    Also left out is whatever an ambiguous declaration contains: each copy of
    a duplicated container shadows the others, so its members have no
    unambiguous parent. Containment is read two ways, and either suffices:
    the member lies inside the declaration's source span (the parser's own
    CONTAINS edge attaches it to the first container at its scope path, which
    may be a different overload), or a CONTAINS chain leads to it (members
    declared outside the body, such as Rust ``impl`` items and Go receiver
    methods). A relation is dropped with either endpoint. A healthy parse is
    returned as the same object.
    """

    groups = collision_groups(parsed.symbols)
    if not groups:
        return QuarantineView(parsed, frozenset(), frozenset(), 0, len(parsed.symbols))
    quarantined = set(groups)
    names = {symbol.simple_name for group in groups.values() for symbol in group}
    colliding = [symbol for group in groups.values() for symbol in group]
    for symbol in parsed.symbols:
        if symbol.id not in quarantined and any(_inside(symbol, outer) for outer in colliding):
            quarantined.add(symbol.id)
    parents = {
        relation.to_symbol_id: relation.from_symbol_id
        for relation in parsed.relations
        if relation.relation_type == CodeRelationType.CONTAINS
    }
    for symbol in parsed.symbols:
        if symbol.id in quarantined:
            continue
        ancestor, seen = symbol.id, {symbol.id}
        while ancestor in parents:
            ancestor = parents[ancestor]
            if ancestor in seen:  # a cyclic CONTAINS chain is itself ambiguous
                break
            seen.add(ancestor)
            if ancestor in quarantined:
                quarantined.add(symbol.id)
                break
    relations: list = []
    dropped = 0
    for relation in parsed.relations:
        # An unresolved relation names its target: that text is not an id.
        if relation.from_symbol_id in quarantined or (
            relation.target_ref is None and relation.to_symbol_id in quarantined
        ):
            dropped += 1
        else:
            relations.append(relation)
    kept = [symbol for symbol in parsed.symbols if symbol.id not in quarantined]
    paths = {
        (*symbol.scope_chain, symbol.simple_name)
        for symbol in parsed.symbols
        if symbol.id in quarantined
    }
    return QuarantineView(
        replace(parsed, symbols=kept, relations=relations),
        frozenset(quarantined),
        frozenset(names),
        dropped,
        len(parsed.symbols),
        frozenset(paths),
    )
