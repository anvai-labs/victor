"""Generic tree-sitter parser for non-Python languages.

Includes a *real* JavaScript/TypeScript extractor — the donor ProximaDB ``code.py``
shipped a JS/TS stub that returned no symbols (CLAUDE-mandate "plausible-but-wrong"
failure). Here JS/TS, plus a language-agnostic node-walk for common grammars, produce
functions, classes, methods, imports, and CALLS / EXTENDS / IMPLEMENTS / CONTAINS
relations (mirroring the Python path's extraction; see ``resolution``).

Requires the ``treesitter`` extra (``tree-sitter`` + ``tree-sitter-language-pack``).
If the grammar is unavailable, :func:`parse_treesitter` raises :class:`GrammarUnavailable`
so the orchestrator can fall back.
"""

from __future__ import annotations

from .languages import TREE_SITTER_GRAMMAR
from .model import (
    CodeRelation,
    CodeRelationType,
    CodeSymbol,
    CodeSymbolType,
    CapabilityTier,
    ParseDiagnostic,
    ParseStatus,
    ParsedCode,
    SourceLocation,
    SymbolReference,
    content_hash,
    deterministic_symbol_id,
)
from .resolution import resolve_relations
from .treesitter_tables import (
    CALL_NODES,
    CALLEE_FIELDS,
    HERITAGE_CLAUSES,
    HERITAGE_SKIP_NODES,
    IDENTIFIER_NODES,
    NAME_FIELDS,
)


class GrammarUnavailable(RuntimeError):
    """Raised when a tree-sitter grammar can't be loaded for a language."""


# Node types that denote a callable, per common tree-sitter grammars.
_FUNC_NODES = {
    "function_declaration",
    "function_definition",
    "function_item",  # rust
    "method_definition",  # js/ts
    "method_declaration",  # java/go
    "function",
    "arrow_function",
}
_CLASS_NODES = {
    "class_declaration",
    "class_definition",
    "class_specifier",
    "struct_item",  # rust
    "struct_specifier",
    "interface_declaration",
    "impl_item",  # rust (special-cased: named after the TYPE, not the trait)
    "trait_item",  # rust
    "enum_item",  # rust
    "enum_declaration",  # java / ts / c#
    "enum_specifier",  # c / cpp
}
_IMPORT_NODES = {
    "import_statement",
    "import_declaration",
    "import_from_statement",
    "use_declaration",  # rust
    "preproc_include",  # c/cpp
}

# c/cpp *_specifier nodes also appear in type positions (``struct Foo x;``).
# Only ones WITH a body are definitions worth a symbol.
_BODY_REQUIRED_NODES = {"struct_specifier", "class_specifier", "enum_specifier"}

# Symbol types that can contain other symbols (for CONTAINS resolution).
_CONTAINER_SYMBOL_TYPES = frozenset(
    {
        CodeSymbolType.CLASS,
        CodeSymbolType.STRUCT,
        CodeSymbolType.INTERFACE,
        CodeSymbolType.TRAIT,
        CodeSymbolType.ENUM,
        CodeSymbolType.FUNCTION,
        CodeSymbolType.METHOD,
        CodeSymbolType.CONSTRUCTOR,
    }
)


def _get_parser(grammar: str):
    # Build an OFFICIAL `tree_sitter.Parser` from the pack's Language, rather than
    # `tree_sitter_language_pack.get_parser()` — the latter can return a vendored
    # binding whose nodes are a minimal `builtins.Node` lacking `.type`/`.children`.
    # The official Parser yields standard nodes (type/children/start_byte properties).
    try:
        from tree_sitter import Parser
        from tree_sitter_language_pack import get_language
    except Exception as e:  # ImportError or native load failure
        raise GrammarUnavailable(f"tree-sitter unavailable: {e}") from e
    try:
        language = get_language(grammar)
    except Exception as e:
        raise GrammarUnavailable(f"grammar '{grammar}' unavailable: {e}") from e
    try:
        return Parser(language)  # tree_sitter >= 0.23
    except TypeError:
        parser = Parser()  # older API: set the language attribute
        parser.language = language
        return parser


def _text(node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _attr(obj, name):
    """Access a tree-sitter attribute that may be a property OR a zero-arg method.

    tree-sitter-language-pack's bundled binding exposes `root_node`/`children` as
    *methods* (callables), whereas the canonical `tree_sitter` exposes them as
    properties. A list/Node is never callable, so this is safe for both shapes.
    """

    v = getattr(obj, name)
    return v() if callable(v) else v


def _children(node):
    """Return a node's children across tree-sitter binding flavors.

    `children` may be a property, a zero-arg method, or absent entirely (the
    bundled binding exposes only `child_count` + `child(i)`, the universal C API).
    """

    children = getattr(node, "children", None)
    if children is not None:
        return children() if callable(children) else children
    count = _attr(node, "child_count")
    return [node.child(i) for i in range(count)]


def _name_of(node, src: bytes) -> str | None:
    """Find the declared name of a function/class node (grammar-agnostic)."""

    if node.type == "arrow_function":
        return None  # an arrow never carries its own name (the binding does);
        # the child scan below would steal an unparenthesized parameter (x => ...)
    field = node.child_by_field_name("name")
    if field is not None:
        return _text(field, src)
    declarator = node.child_by_field_name("declarator")
    if declarator is not None:
        # c/c++: function_definition -> function_declarator -> ... -> identifier
        name = _rightmost_name(declarator, src)
        if name:
            return name
    for child in _children(node):
        if child.type in (
            "identifier",
            "type_identifier",
            "field_identifier",
            "property_identifier",
        ):
            return _text(child, src)
    return None


def _rightmost_name(node, src: bytes, _depth: int = 0) -> str | None:
    """Reduce a (possibly composite) reference node to its rightmost simple name.

    ``a.b.c()`` -> ``c``; ``Foo::new`` -> ``new``; ``pkg.Fn`` -> ``Fn``. Best-effort
    and grammar-agnostic: identifier leaves win, then well-known fields, then the
    last identifier-ish child.
    """

    if _depth > 8 or node is None:
        return None
    if node.type in IDENTIFIER_NODES:
        return _text(node, src)
    for f in NAME_FIELDS:
        ch = node.child_by_field_name(f)
        if ch is not None:
            return _rightmost_name(ch, src, _depth + 1)
    for ch in reversed(_children(node)):
        if ch.type in IDENTIFIER_NODES:
            return _text(ch, src)
    return None


def _collect_calls(node, owner: CodeSymbol, src: bytes, file_path: str, relations) -> None:
    """Record CALLS relations for every call expression in a symbol's body.

    Stops at nested *named* definitions (they own their calls) but descends into
    anonymous functions — a callback's calls belong to the enclosing named symbol,
    matching the Python path's ``ast.walk`` semantics.
    """

    stack = list(_children(node))
    while stack:
        n = stack.pop()
        t = n.type
        if t in _CLASS_NODES:
            continue  # nested type owns its contents
        if t in _FUNC_NODES and _name_of(n, src) is not None:
            continue  # nested named function owns its calls
        if t in CALL_NODES:
            callee_node = None
            for f in CALLEE_FIELDS:
                callee_node = n.child_by_field_name(f)
                if callee_node is not None:
                    break
            callee = _rightmost_name(callee_node, src) if callee_node is not None else None
            if callee:
                raw = _text(callee_node, src) if callee_node is not None else callee
                qualifier = raw.rsplit(".", 1)[0] if "." in raw else None
                args = n.child_by_field_name("arguments")
                arity = None
                if args is not None:
                    arity = len([c for c in _children(args) if getattr(c, "is_named", True)])
                relations.append(
                    CodeRelation(
                        from_symbol_id=owner.id,
                        to_symbol_id=callee,
                        relation_type=CodeRelationType.CALLS,
                        context=callee,
                        target_ref=SymbolReference(
                            name=callee, qualifier=qualifier, arity=arity, text=raw
                        ),
                        call_site=SourceLocation(
                            file_path=file_path,
                            start_line=n.start_point[0] + 1,  # 1-based line contract
                            start_column=n.start_point[1],
                        ),
                    )
                )
        stack.extend(_children(n))


def _heritage_names(clause, src: bytes) -> list[str]:
    """Collect base-type names from a heritage clause, skipping generic args."""

    names: list[str] = []
    stack = list(_children(clause))
    while stack:
        n = stack.pop(0)
        if n.type in HERITAGE_SKIP_NODES:
            continue
        if n.type in IDENTIFIER_NODES:
            text = _text(n, src)
            if text not in names:
                names.append(text)
            continue
        stack = list(_children(n)) + stack
    return names


def _extract_heritage(class_node, sym: CodeSymbol, src: bytes, relations) -> None:
    """EXTENDS / IMPLEMENTS edges from a class-like node's heritage clauses.

    Clause nodes are found by walking children only — java's ``superclass`` /
    ``super_interfaces`` are ALSO exposed as fields, so probing both would emit
    each edge twice. Deduped per (relation, base) regardless.
    """

    seen: set[tuple[CodeRelationType, str]] = set()

    def _emit(names: list[str], kind: str) -> None:
        rel = CodeRelationType.EXTENDS if kind == "extends" else CodeRelationType.IMPLEMENTS
        for name in names:
            if name and name != sym.simple_name and (rel, name) not in seen:
                seen.add((rel, name))
                relations.append(
                    CodeRelation(
                        from_symbol_id=sym.id,
                        to_symbol_id=name,
                        relation_type=rel,
                        context=name,
                        target_ref=SymbolReference(name=name, text=name),
                    )
                )

    for ch in _children(class_node):
        kind = HERITAGE_CLAUSES.get(ch.type)
        if kind is None:
            continue
        # TS wraps extends/implements clauses inside class_heritage; JS puts the
        # raw expression there. Prefer the inner clauses when present.
        inner = [c for c in _children(ch) if c.type in HERITAGE_CLAUSES]
        if inner:
            for c in inner:
                _emit(_heritage_names(c, src), HERITAGE_CLAUSES[c.type])
        else:
            _emit(_heritage_names(ch, src), kind)


def _class_symbol_type(node_type: str) -> CodeSymbolType:
    if node_type.startswith("struct"):
        return CodeSymbolType.STRUCT
    if "interface" in node_type:
        return CodeSymbolType.INTERFACE
    if node_type.startswith("trait"):
        return CodeSymbolType.TRAIT
    if node_type.startswith("enum"):
        return CodeSymbolType.ENUM
    return CodeSymbolType.CLASS


def _scope_names(frames: list[tuple[str, str]]) -> list[str]:
    return [name for name, _kind in frames]


def _handle_impl_item(child, src, file_path, language, frames, symbols, relations, impl_traits):
    """Rust ``impl Type`` / ``impl Trait for Type``: attribute members to the TYPE.

    No symbol is emitted for the impl block itself — it would share the type's
    FQN (OID collision with the struct/enum, and with sibling impl blocks). The
    old generic probe additionally picked the first ``type_identifier``, which
    for ``impl Trait for Type`` is the *trait* — misnaming everything under it.
    Instead: members get the TYPE on their scope chain (METHOD + CONTAINS via
    scope), and ``impl Trait for Type`` is recorded in ``impl_traits`` so an
    IMPLEMENTS edge can be attached to the type's symbol after the walk.
    """

    type_node = child.child_by_field_name("type")
    name = _rightmost_name(type_node, src) if type_node is not None else None
    name = name or _name_of(child, src) or "<anonymous>"

    trait_node = child.child_by_field_name("trait")
    trait_name = _rightmost_name(trait_node, src) if trait_node is not None else None
    if trait_name:
        impl_traits.append((name, trait_name))

    body = child.child_by_field_name("body")
    _walk_collect(
        body if body is not None else child,
        src,
        file_path,
        language,
        [*frames, (name, "class")],
        symbols,
        relations,
        impl_traits,
    )


def _handle_go_type_spec(child, src, file_path, language, frames, symbols, relations):
    """Go ``type Foo struct/interface {...}``: STRUCT/INTERFACE symbols + embedding."""

    name_node = child.child_by_field_name("name")
    type_node = child.child_by_field_name("type")
    if name_node is None or type_node is None:
        return
    tt = type_node.type
    if tt not in ("struct_type", "interface_type"):
        return  # plain type aliases stay out (matches prior behavior)
    name = _text(name_node, src)
    stype = CodeSymbolType.STRUCT if tt == "struct_type" else CodeSymbolType.INTERFACE
    sym = _mk(child, src, file_path, language, name, stype, _scope_names(frames))
    symbols.append(sym)

    # Embedding => EXTENDS (best-effort): a struct field_declaration with a type
    # but no name, or a bare type in an interface body.
    embedded: list[str] = []
    if tt == "struct_type":
        for lst in _children(type_node):
            if lst.type != "field_declaration_list":
                continue
            for fd in _children(lst):
                if fd.type != "field_declaration":
                    continue
                if fd.child_by_field_name("name") is not None:
                    continue
                base = _rightmost_name(fd.child_by_field_name("type"), src)
                if base:
                    embedded.append(base)
    else:
        for ch in _children(type_node):
            if ch.type in ("type_identifier", "qualified_type", "type_elem"):
                base = _rightmost_name(ch, src)
                if base:
                    embedded.append(base)
    for base in embedded:
        if base != name:
            relations.append(
                CodeRelation(
                    from_symbol_id=sym.id,
                    to_symbol_id=base,
                    relation_type=CodeRelationType.EXTENDS,
                    context=base,
                    target_ref=SymbolReference(name=base, text=base),
                )
            )


def _go_receiver_type(node, src: bytes) -> str | None:
    """The receiver type of a Go method_declaration (``func (s *Svc) Run()`` -> Svc)."""

    receiver = node.child_by_field_name("receiver")
    if receiver is None:
        return None
    for pd in _children(receiver):
        if pd.type == "parameter_declaration":
            t = pd.child_by_field_name("type")
            if t is not None:
                return _rightmost_name(t, src)
    return None


def _walk_collect(node, src, file_path, language, frames, symbols, relations, impl_traits):
    for child in _children(node):
        t = child.type
        if t == "impl_item":
            _handle_impl_item(
                child, src, file_path, language, frames, symbols, relations, impl_traits
            )
        elif t == "type_spec" and language == "go":
            _handle_go_type_spec(child, src, file_path, language, frames, symbols, relations)
        elif t in _CLASS_NODES:
            if t in _BODY_REQUIRED_NODES and child.child_by_field_name("body") is None:
                # c/cpp ``struct Foo x;`` — a type *reference*, not a definition.
                continue
            name = _name_of(child, src) or "<anonymous>"
            sym = _mk(
                child, src, file_path, language, name, _class_symbol_type(t), _scope_names(frames)
            )
            symbols.append(sym)
            _extract_heritage(child, sym, src, relations)
            # Recurse into the body with the type name on the scope chain.
            body = child.child_by_field_name("body")
            _walk_collect(
                body if body is not None else child,
                src,
                file_path,
                language,
                [*frames, (name, "class")],
                symbols,
                relations,
                impl_traits,
            )
        elif t in _FUNC_NODES:
            name = _name_of(child, src)
            if name is None:
                # Anonymous functions have no durable structural discriminator. Do
                # not promote them to symbols; recurse to collect named children.
                _walk_collect(
                    child, src, file_path, language, frames, symbols, relations, impl_traits
                )
                continue
            in_class = bool(frames) and frames[-1][1] == "class"
            stype = CodeSymbolType.METHOD if in_class else CodeSymbolType.FUNCTION
            if name in ("constructor", "__init__", "new"):
                stype = CodeSymbolType.CONSTRUCTOR
            receiver = None
            if language == "go" and t == "method_declaration":
                receiver = _go_receiver_type(child, src)
                if receiver and stype == CodeSymbolType.FUNCTION:
                    stype = CodeSymbolType.METHOD  # a receiver makes it a method
            sym = _mk(child, src, file_path, language, name, stype, _scope_names(frames))
            if receiver:
                sym.metadata["receiver"] = receiver
            symbols.append(sym)
            _collect_calls(child, sym, src, file_path, relations)
            # Recurse into the body so nested named functions/classes are collected.
            body = child.child_by_field_name("body")
            _walk_collect(
                body if body is not None else child,
                src,
                file_path,
                language,
                [*frames, (name, "function")],
                symbols,
                relations,
                impl_traits,
            )
        else:
            _walk_collect(child, src, file_path, language, frames, symbols, relations, impl_traits)


def _handle_const_arrow(node, src, file_path, language, symbols, relations):
    """JS/TS: ``const foo = (...) => {...}`` / ``export const foo = () => {}``."""

    for decl in _children(node):
        if decl.type != "variable_declarator":
            continue
        name_node = decl.child_by_field_name("name")
        value = decl.child_by_field_name("value")
        if name_node is not None and value is not None and value.type == "arrow_function":
            name = _text(name_node, src)
            sym = _mk(decl, src, file_path, language, name, CodeSymbolType.FUNCTION, [])
            symbols.append(sym)
            _collect_calls(value, sym, src, file_path, relations)


# Parameter-list node types across tree-sitter grammars (Python/JS-TS/Java/Go/Rust/C…).
_PARAM_NODE_TYPES = frozenset(
    {
        "parameters",
        "formal_parameters",
        "parameter_list",
        "argument_list",
        "function_value_parameters",  # Kotlin/Rust-style
    }
)


def _param_signature(node, src) -> str | None:
    """The symbol's parameter-list text — the ADR-044 overload discriminator.

    Stable under line moves (it's the params, not the position). ``None`` for symbols
    with no parameter node (classes/structs), giving an empty discriminator there.
    Python's AST parser supplies a richer ``signature``; this is the tree-sitter path.
    """

    stack = [(node, 0)]
    while stack:
        current, depth = stack.pop(0)
        if current is not node and _attr(current, "type") in _PARAM_NODE_TYPES:
            return _text(current, src)
        if depth < 5:
            stack.extend((ch, depth + 1) for ch in _children(current))
    return None


def _mk(node, src, file_path, language, name, stype, scope) -> CodeSymbol:
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    fqn = "::".join([file_path.replace("/", "."), *scope, name])
    return CodeSymbol(
        id=deterministic_symbol_id(file_path, name, start_line, node.start_point[1]),
        symbol_type=stype,
        fully_qualified_name=fqn,
        simple_name=name,
        signature=_param_signature(node, src),
        location=SourceLocation(
            file_path=file_path,
            start_line=start_line,
            start_column=node.start_point[1],
            end_line=end_line,
            end_column=node.end_point[1],
            byte_offset=node.start_byte,
            byte_length=node.end_byte - node.start_byte,
        ),
        source_code=_text(node, src),
        language=language,
        scope_chain=list(scope),
        complexity={"lines": end_line - start_line + 1},
    )


def _build_contains(symbols: list[CodeSymbol]) -> list[CodeRelation]:
    """CONTAINS edges from full scope paths (parent scope -> symbol).

    Keyed on the complete scope path, not the collision-prone simple name, and
    covering every container type (class/struct/interface/trait/enum/function).
    Go receiver methods live at top level; their CONTAINS edge comes from the
    recorded receiver type instead.
    """

    by_path: dict[tuple[str, ...], str] = {}
    for s in symbols:
        if s.symbol_type in _CONTAINER_SYMBOL_TYPES:
            by_path.setdefault((*s.scope_chain, s.simple_name), s.id)

    contains: list[CodeRelation] = []
    seen: set[tuple[str, str]] = set()

    def _emit(parent_id: str, child_id: str) -> None:
        if parent_id != child_id and (parent_id, child_id) not in seen:
            seen.add((parent_id, child_id))
            contains.append(
                CodeRelation(
                    from_symbol_id=parent_id,
                    to_symbol_id=child_id,
                    relation_type=CodeRelationType.CONTAINS,
                )
            )

    for s in symbols:
        if s.scope_chain:
            parent = by_path.get(tuple(s.scope_chain))
            if parent is not None:
                _emit(parent, s.id)
        receiver = s.metadata.get("receiver")
        if receiver:
            parent = by_path.get((*s.scope_chain, receiver))
            if parent is not None:
                _emit(parent, s.id)
    return contains


def parse_treesitter(content: str, file_path: str, language: str) -> ParsedCode:
    """Parse non-Python source via tree-sitter. Raises GrammarUnavailable on fallback."""

    grammar = TREE_SITTER_GRAMMAR.get(language)
    if grammar is None:
        raise GrammarUnavailable(f"no grammar mapping for language '{language}'")
    parser = _get_parser(grammar)

    src = content.encode("utf-8")
    # tree-sitter's Parser.parse() takes bytes on most builds, but some
    # (notably via tree-sitter-language-pack) require a str and raise
    # TypeError on bytes. Node byte offsets are UTF-8 positions either way,
    # so `src` stays valid for slicing in _text() regardless of which we pass.
    try:
        tree = parser.parse(src)
    except TypeError:
        tree = parser.parse(content)
    root = _attr(tree, "root_node")

    symbols: list[CodeSymbol] = []
    relations: list[CodeRelation] = []
    imports: list[str] = []
    impl_traits: list[tuple[str, str]] = []  # rust: (type name, trait name)

    for child in _children(root):
        if child.type in _IMPORT_NODES:
            imports.append(_text(child, src))
        elif child.type == "export_statement":
            exported = _text(child, src)
            # Re-exports create a module dependency; plain declarations do not.
            if " from " in exported:
                imports.append(exported)

    _walk_collect(root, src, file_path, language, [], symbols, relations, impl_traits)

    # JS/TS arrow-function-as-const: a real surface the stub missed.
    if language in ("javascript", "typescript", "tsx"):
        for child in _children(root):
            target = child
            # unwrap `export const ...`
            if child.type in ("export_statement",) and child.child_count:
                for c in _children(child):
                    if c.type in ("lexical_declaration", "variable_declaration"):
                        target = c
                        break
            if target.type in ("lexical_declaration", "variable_declaration"):
                _handle_const_arrow(target, src, file_path, language, symbols, relations)

    # rust: attach `impl Trait for Type` to the TYPE's symbol (struct/enum),
    # now that the whole file is walked (the type may be declared after the impl).
    if impl_traits:
        by_top_name = {
            s.simple_name: s.id
            for s in symbols
            if not s.scope_chain and s.symbol_type in _CONTAINER_SYMBOL_TYPES
        }
        seen_impl: set[tuple[str, str]] = set()
        for type_name, trait_name in impl_traits:
            from_id = by_top_name.get(type_name)
            if from_id is None or (from_id, trait_name) in seen_impl:
                continue  # impl for an out-of-file type — best-effort skip
            seen_impl.add((from_id, trait_name))
            relations.append(
                CodeRelation(
                    from_symbol_id=from_id,
                    to_symbol_id=trait_name,
                    relation_type=CodeRelationType.IMPLEMENTS,
                    context=trait_name,
                    target_ref=SymbolReference(name=trait_name, text=trait_name),
                )
            )

    relations.extend(_build_contains(symbols))

    has_error = bool(getattr(root, "has_error", False))
    return ParsedCode(
        file_path=file_path,
        language=language,
        symbols=symbols,
        relations=resolve_relations(symbols, relations),
        imports=imports,
        content_hash=content_hash(content),
        status=ParseStatus.PARTIAL if has_error else ParseStatus.SUCCESS,
        capability_tier=CapabilityTier.SYMBOLS,
        diagnostics=(
            [ParseDiagnostic(code="syntax_error", message="tree-sitter produced an error node")]
            if has_error
            else []
        ),
        source_code=content,
    )
