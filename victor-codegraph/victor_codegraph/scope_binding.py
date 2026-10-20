"""Tier 2.5 scope binding: def/use occurrence extraction for REFERENCES relations.

ADR-032's first milestone. CALLS/EXTENDS/IMPLEMENTS already record *call targets
and heritage bases* as textual references and resolve them through
:mod:`victor_codegraph.resolution`. This module adds the missing half of name
binding: plain identifier USES (type annotations, value loads, decorated names)
attributed to their innermost owning symbol and resolved through the SAME
machinery — ``REFERENCES`` joins the name-targeted set, so in-file resolution
(``scope_resolver``) and cross-file resolution (``repository_resolver``) apply
unchanged, and ambiguity is retained as a structured ``SymbolReference`` at
confidence 0.5, never guessed.

Adopt-vs-build is settled by evidence: GitHub's stack-graphs is archived
(RUSTSEC-2026-0303, no published successor), so this is the hand-rolled subset
the ADR anticipated. It is pure Python, per-file (only a file's own subtree is
walked), and incremental by construction — relations flow through the existing
``relation_key``/delta machinery.

Precision/noise contract (v1):

- Only uses that can resolve to EXTRACTED symbols are interesting. Locals and
  parameters have no CodeSymbol, so occurrences shadowed by a local binding are
  skipped at extraction time (the generic resolver branch consults no
  bound-names metadata; without this, a local ``helper`` would falsely bind to a
  function ``helper``). Shadow collection over-collects on purpose
  (initializer expressions inside declarations shadow their own names too):
  the error direction is suppressed edges, never false bindings.
- Call callees and heritage bases are excluded here — they are CALLS/EXTENDS/
  IMPLEMENTS territory and would duplicate those edges.
- Member/property positions of member accesses (``a.b``'s ``b``) are excluded:
  member names are the noisiest identifier class and rarely resolve to a
  file-local symbol.
- Module-level uses are skipped (v1 documented limitation: no module CodeSymbol
  exists at ``parse`` time, so they have no owner in the id-map rewrite).
- One occurrence per (owner, name): ``call_site`` records the FIRST use; dedup
  here is a correctness requirement, not an optimization — ``relation_key``
  collapses duplicates keeping the last silently, which would break the delta
  equivalence and identity ratchets.
- Unresolved references are capped per owner (deterministic first-N) so a large
  body cannot flood a file's retained-reference set.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import CodeRelation, CodeRelationType, SourceLocation, SymbolReference

# Languages where reference extraction runs, VERIFIED per grammar against the
# three-fact contract (one resolved in-file edge, one retained-unresolved edge,
# shadowed locals producing no edge). Membership is honest capability, not
# grammar availability: a language whose declaration/member node kinds the
# generic tables do not yet map correctly (leaking locals as retained refs, or
# binding member names to same-named symbols) is excluded until its tables are
# verified — an absent REFERENCES edge is recoverable, a wrong one is not.
#
# Verified: python, javascript, typescript, tsx, rust, go, java, c, bash.
# Pending grammar-table verification (mismatched declaration/parameter/member
# node kinds or extractor gaps): cpp, csharp, php, swift, scala, lua, kotlin
# (extractor mints no function symbols), ruby (``method`` nodes are not in
# ``_FUNC_NODES``). Vacuous: sql (tree-sitter-sql yields no symbols).
# Adding a language = verify the three facts, then list it here.
REFERENCE_LANGUAGES = frozenset(
    {
        "python",
        "javascript",
        "typescript",
        "tsx",
        "rust",
        "go",
        "java",
        "c",
        "bash",
    }
)

# Names that never resolve to a file-local symbol and would only flood the
# retained-unresolved set: language preludes, builtins, and ubiquitous globals.
# Static lists on purpose — importing builtins dynamically would make emission
# environment-dependent.
NOISE_ROOTS: dict[str, frozenset[str]] = {
    "python": frozenset(
        {
            "abs",
            "all",
            "any",
            "bool",
            "bytes",
            "callable",
            "chr",
            "classmethod",
            "dict",
            "dir",
            "divmod",
            "enumerate",
            "Exception",
            "False",
            "filter",
            "float",
            "frozenset",
            "getattr",
            "hasattr",
            "hash",
            "id",
            "input",
            "int",
            "isinstance",
            "issubclass",
            "iter",
            "len",
            "list",
            "map",
            "max",
            "min",
            "next",
            "None",
            "object",
            "open",
            "pow",
            "print",
            "property",
            "range",
            "repr",
            "reversed",
            "round",
            "set",
            "setattr",
            "slice",
            "sorted",
            "staticmethod",
            "str",
            "sum",
            "super",
            "True",
            "tuple",
            "type",
            "ValueError",
            "TypeError",
            "KeyError",
            "IndexError",
            "AttributeError",
            "StopIteration",
            "zip",
            "__name__",
            "__file__",
        }
    ),
    "javascript": frozenset(
        {
            "console",
            "Math",
            "JSON",
            "Object",
            "Array",
            "String",
            "Number",
            "Boolean",
            "Promise",
            "Symbol",
            "Map",
            "Set",
            "WeakMap",
            "WeakSet",
            "Date",
            "Error",
            "TypeError",
            "RangeError",
            "RegExp",
            "globalThis",
            "window",
            "document",
            "setTimeout",
            "setInterval",
            "fetch",
            "require",
            "module",
            "exports",
            "undefined",
            "this",
        }
    ),
    "typescript": frozenset(
        {
            "console",
            "Math",
            "JSON",
            "Object",
            "Array",
            "String",
            "Number",
            "Boolean",
            "Promise",
            "Symbol",
            "Map",
            "Set",
            "Date",
            "Error",
            "RegExp",
            "globalThis",
            "window",
            "document",
            "fetch",
            "require",
            "undefined",
            "Record",
            "Partial",
            "Readonly",
            "Pick",
            "Omit",
        }
    ),
    "tsx": frozenset(
        {
            "console",
            "Math",
            "JSON",
            "Object",
            "Array",
            "String",
            "Number",
            "Promise",
            "Map",
            "Set",
            "Date",
            "Error",
            "React",
            "window",
            "document",
            "undefined",
            "Record",
            "Partial",
        }
    ),
    "rust": frozenset(
        {
            "Some",
            "None",
            "Ok",
            "Err",
            "Option",
            "Result",
            "Vec",
            "String",
            "Box",
            "Rc",
            "Arc",
            "RefCell",
            "Cell",
            "HashMap",
            "HashSet",
            "BTreeMap",
            "BTreeSet",
            "Cow",
            "Default",
            "Clone",
            "Copy",
            "Debug",
            "PartialEq",
            "Eq",
            "PartialOrd",
            "Ord",
            "Hash",
            "Iterator",
            "IntoIterator",
            "From",
            "Into",
            "ToString",
            "ToOwned",
            "Send",
            "Sync",
            "Sized",
            "Unpin",
            "vec",
            "format",
            "println",
            "print",
            "eprintln",
            "write",
            "writeln",
            "panic",
            "assert",
            "assert_eq",
            "assert_ne",
            "unreachable",
            "todo",
            "unimplemented",
            "self",
            "Self",
        }
    ),
    "go": frozenset(
        {
            "append",
            "cap",
            "close",
            "complex",
            "copy",
            "delete",
            "imag",
            "len",
            "make",
            "new",
            "panic",
            "print",
            "println",
            "real",
            "recover",
            "error",
            "string",
            "int",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "uintptr",
            "float32",
            "float64",
            "complex64",
            "complex128",
            "byte",
            "rune",
            "bool",
            "any",
            "nil",
            "true",
            "false",
            "iota",
        }
    ),
    "java": frozenset(
        {
            "String",
            "Integer",
            "Long",
            "Double",
            "Float",
            "Boolean",
            "Character",
            "Byte",
            "Short",
            "Object",
            "System",
            "Math",
            "List",
            "ArrayList",
            "Map",
            "HashMap",
            "Set",
            "HashSet",
            "Optional",
            "Exception",
            "RuntimeException",
            "IllegalArgumentException",
            "IllegalStateException",
            "NullPointerException",
            "Override",
            "Deprecated",
            "SuppressWarnings",
            "FunctionalInterface",
            "this",
            "super",
        }
    ),
    "c": frozenset(
        {
            "int",
            "char",
            "long",
            "short",
            "float",
            "double",
            "void",
            "unsigned",
            "signed",
            "size_t",
            "ssize_t",
            "NULL",
            "malloc",
            "free",
            "calloc",
            "realloc",
            "memcpy",
            "memset",
            "strlen",
            "printf",
            "fprintf",
            "sprintf",
            "snprintf",
            "true",
            "false",
        }
    ),
    "cpp": frozenset(
        {
            "int",
            "char",
            "long",
            "short",
            "float",
            "double",
            "void",
            "auto",
            "bool",
            "unsigned",
            "signed",
            "size_t",
            "NULL",
            "nullptr",
            "true",
            "false",
            "std",
            "string",
            "vector",
            "map",
            "set",
            "make_unique",
            "make_shared",
            "malloc",
            "free",
            "memcpy",
            "printf",
        }
    ),
    "csharp": frozenset(
        {
            "string",
            "int",
            "long",
            "double",
            "float",
            "bool",
            "object",
            "var",
            "new",
            "null",
            "true",
            "false",
            "Task",
            "List",
            "Dictionary",
            "Console",
            "Math",
            "Exception",
            "Override",
            "this",
            "base",
        }
    ),
    "ruby": frozenset(
        {
            "puts",
            "print",
            "p",
            "require",
            "require_relative",
            "lambda",
            "proc",
            "raise",
            "new",
            "self",
            "nil",
            "true",
            "false",
            "yield",
            "freeze",
            "length",
            "size",
            "each",
            "map",
            "select",
        }
    ),
    "php": frozenset(
        {
            "strlen",
            "count",
            "array",
            "implode",
            "explode",
            "print",
            "echo",
            "printf",
            "sprintf",
            "in_array",
            "array_map",
            "array_filter",
            "array_keys",
            "array_values",
            "null",
            "true",
            "false",
            "this",
        }
    ),
    "swift": frozenset(
        {
            "String",
            "Int",
            "Double",
            "Float",
            "Bool",
            "Array",
            "Dictionary",
            "Set",
            "Optional",
            "Result",
            "Error",
            "print",
            "debugPrint",
            "self",
            "nil",
            "true",
            "false",
        }
    ),
    "kotlin": frozenset(
        {
            "String",
            "Int",
            "Long",
            "Double",
            "Float",
            "Boolean",
            "List",
            "Map",
            "Set",
            "Pair",
            "Triple",
            "println",
            "print",
            "error",
            "require",
            "check",
            "lazy",
            "lateinit",
            "it",
            "this",
        }
    ),
    "scala": frozenset(
        {
            "String",
            "Int",
            "Long",
            "Double",
            "Float",
            "Boolean",
            "List",
            "Map",
            "Set",
            "Option",
            "Some",
            "None",
            "Success",
            "Failure",
            "Try",
            "Either",
            "println",
            "print",
            "require",
            "Nil",
            "Unit",
        }
    ),
    "bash": frozenset(
        {
            "echo",
            "cd",
            "pwd",
            "exit",
            "export",
            "local",
            "readonly",
            "return",
            "source",
            "set",
            "shift",
            "unset",
            "eval",
            "printf",
            "read",
            "true",
            "false",
        }
    ),
    "lua": frozenset(
        {
            "print",
            "pairs",
            "ipairs",
            "type",
            "tostring",
            "tonumber",
            "require",
            "pcall",
            "error",
            "assert",
            "table",
            "string",
            "math",
            "io",
            "os",
            "self",
            "nil",
            "true",
            "false",
        }
    ),
}

# Identifier leaf node types that constitute a simple-name USE when not in a
# declaring or member position. Mirrors treesitter_tables.IDENTIFIER_NODES.
_USE_LEAF_NODES = frozenset(
    {
        "identifier",
        "type_identifier",
        "field_identifier",
        "property_identifier",
        "constant",
        "simple_identifier",
        "word",
    }
)

# Node types whose subtree DECLARES (shadows) names. Suffix-typed on purpose —
# grammar spellings (``variable_declarator`` js/ts/java/c#,
# ``short_variable_declaration`` go, ``let_declaration`` rust, ``declaration``
# c/cpp, ``formal_parameter`` java) all end in these suffixes. The whole
# subtree's identifier leaves are collected: initializer expressions inside a
# declaration shadow their own names too, which only suppresses edges.
_DECLARATION_SUFFIXES = (
    "declarator",
    "declaration",
    "parameter",
    "parameter_list",
    "formal_parameters",
    "parameters",
    "param_list",
)

# Fields that mark member-access positions (``a.b``'s ``b``) — excluded from uses.
_MEMBER_FIELDS = ("property", "field", "method", "attribute")

# Maximum retained-unresolved references per owning symbol.
UNRESOLVED_CAP_PER_OWNER = 64


@dataclass(frozen=True)
class ScopeOccurrence:
    """One identifier use attributed to its innermost owning symbol."""

    owner_id: str
    name: str
    line: int  # 1-based, canonical line contract
    use: str = "load"


def reference_relation(occurrence: ScopeOccurrence, file_path: str) -> CodeRelation:
    """Project one occurrence as an unresolved REFERENCES relation.

    ``to_symbol_id`` carries the textual name, exactly like a pre-resolution
    CALLS target: ``resolve_relations`` (in-file) and ``_resolve_across_files``
    (repository) turn it into a real symbol id with ``scope_resolver`` /
    ``repository_resolver`` provenance, or retain the structured reference.
    """

    return CodeRelation(
        from_symbol_id=occurrence.owner_id,
        to_symbol_id=occurrence.name,
        relation_type=CodeRelationType.REFERENCES,
        context=occurrence.use,
        target_ref=SymbolReference(name=occurrence.name, text=occurrence.name),
        call_site=SourceLocation(
            file_path=file_path,
            start_line=occurrence.line,
        ),
    )


def cap_unresolved_references(
    relations: list[CodeRelation],
    per_owner: int = UNRESOLVED_CAP_PER_OWNER,
) -> list[CodeRelation]:
    """Drop unresolved REFERENCES beyond ``per_owner`` per source symbol.

    Deterministic in emission order (first occurrences kept). Resolved
    references are never capped; other relation types pass through untouched.
    """

    kept_unresolved: dict[str, int] = {}
    out: list[CodeRelation] = []
    for relation in relations:
        if relation.relation_type != CodeRelationType.REFERENCES or relation.target_ref is None:
            out.append(relation)
            continue
        count = kept_unresolved.get(relation.from_symbol_id, 0)
        if count >= per_owner:
            continue
        kept_unresolved[relation.from_symbol_id] = count + 1
        out.append(relation)
    return out


def _noise_roots(language: str) -> frozenset[str]:
    return NOISE_ROOTS.get(language, frozenset())


def _identifier_leaves(node):
    """Yield identifier-ish leaf nodes under ``node`` (subtree scan)."""

    from .treesitter_parser import _children

    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in _USE_LEAF_NODES:
            yield current
        stack.extend(_children(current))


def _node_key(n) -> tuple[int, int, str]:
    """Stable identity for a tree-sitter node across wrapper objects.

    PyTreeSitter ``Node`` is a value wrapper: the same underlying node yields a
    DIFFERENT Python object per access, so ``id()`` cannot key exclusions across
    passes. The (start, end, type) span is unique per node within one tree.
    """

    return (n.start_byte, n.end_byte, n.type)


# ── Tree-sitter path ─────────────────────────────────────────────────────────


def collect_all_treesitter_references(
    root,
    src: bytes,
    file_path: str,
    language: str,
    symbols,
    relations: list[CodeRelation],
) -> None:
    """Dispatch per-symbol occurrence extraction over a whole file.

    A post-pass over the parsed tree — ``_walk_collect`` stays untouched. Symbols
    are indexed by their defining node's ``start_byte`` (``_mk`` writes
    ``location.byte_offset = node.start_byte``), so a node at that byte IS the
    symbol's definition node. Walking the tree, each indexed node becomes an
    owner boundary: its subtree is extracted for that symbol and skipped for
    everyone else, which reproduces the CALLS collector's innermost-owner
    discipline. Module-level code (no owner) yields nothing, per the v1 limit.
    """

    from .treesitter_parser import _children

    by_start_byte = {symbol.location.byte_offset: symbol for symbol in symbols}

    stack = [root]
    while stack:
        n = stack.pop()
        if n is not root:
            owner = by_start_byte.get(n.start_byte)
            if owner is not None:
                # Collect for this owner, then KEEP DESCENDING: nested owners
                # (methods inside a class) dispatch on their own defining nodes.
                collect_treesitter_references(n, owner, src, file_path, language, relations)
        stack.extend(_children(n))


def collect_treesitter_references(
    node,
    owner,
    src: bytes,
    file_path: str,
    language: str,
    relations: list[CodeRelation],
) -> None:
    """Collect REFERENCES occurrences owned by ``owner`` from its defining node.

    Same ownership discipline as the CALLS collector: nested named functions and
    classes own their own occurrences (their subtrees are skipped), anonymous
    functions and callbacks belong to the enclosing named symbol. Imports
    (``treesitter_parser`` privates) are resolved lazily to keep this module
    importable without tree-sitter and cycle-free.
    """

    from .treesitter_parser import (
        CALL_NODES,
        _CLASS_NODES,
        _FUNC_NODES,
        _children,
        _name_of,
    )
    from .treesitter_tables import NAME_FIELDS

    def is_owner_boundary(n) -> bool:
        return n.type in _CLASS_NODES or (n.type in _FUNC_NODES and _name_of(n, src) is not None)

    def rightmost_name_node(n, depth: int = 0):
        if depth > 8 or n is None:
            return None
        if n.type in _USE_LEAF_NODES:
            return n
        for field in NAME_FIELDS:
            child = n.child_by_field_name(field)
            if child is not None:
                return rightmost_name_node(child, depth + 1)
        for child in reversed(_children(n)):
            if child.type in _USE_LEAF_NODES:
                return child
        return None

    noise = _noise_roots(language)

    # The owner's own declared name is a defining position, not a use.
    excluded: set[tuple[int, int, str]] = set()
    for field in ("name", "declarator", "pattern"):
        own = node.child_by_field_name(field)
        if own is not None:
            excluded.update(_node_key(leaf) for leaf in _identifier_leaves(own))
            break

    # Pass 1 — mark excluded node identities and collect shadow names.
    shadowed: set[str] = set()

    def collect_subtree_names(n) -> None:
        for leaf in _identifier_leaves(n):
            shadowed.add(leaf.text.decode("utf-8", "replace"))

    def shadow_declaration(n) -> None:
        """Shadow declared names without eating initializer USES.

        Prefer a grammar's name-bearing field (``name``/``declarator``/
        ``pattern``/``left``); parameter containers shadow their parameter
        names. Otherwise walk direct children IN ORDER: recurse into nested
        declarator/declaration children, shadow the FIRST identifier leaf (the
        declared binding in every grammar we map), and STOP at the first other
        child — that position is the initializer/value, whose identifiers are
        uses. The error direction stays "suppressed edge", never false binding.
        """
        for field in ("name", "declarator", "pattern", "left"):
            named = n.child_by_field_name(field)
            if named is not None:
                collect_subtree_names(named)
                return
        if n.type.endswith(("parameters", "parameter_list", "formal_parameters")):
            for child in _children(n):
                named = child.child_by_field_name("name")
                if named is not None:
                    collect_subtree_names(named)
                elif child.type in _USE_LEAF_NODES:
                    shadowed.add(child.text.decode("utf-8", "replace"))
            return
        for child in _children(n):
            if not getattr(child, "is_named", True):
                continue  # keywords (`let`, `fn`, `var`) precede the binding
            if child.type.endswith(_DECLARATION_SUFFIXES):
                shadow_declaration(child)
                continue
            if child.type in _USE_LEAF_NODES:
                shadowed.add(child.text.decode("utf-8", "replace"))
            return

    stack = list(_children(node))
    while stack:
        n = stack.pop()
        t = n.type
        if is_owner_boundary(n):
            # A nested def/class owns its contents, but its NAME binds in this
            # scope (a def is an assignment to the name) — conservative shadow.
            name = _name_of(n, src)
            if name:
                shadowed.add(name)
            continue
        if t in CALL_NODES:
            callee = None
            for field in ("function", "constructor", "name", "type", "method"):
                callee = n.child_by_field_name(field)
                if callee is not None:
                    break
            if callee is not None:
                # Exclude only the callee's rightmost name (the CALLED name);
                # base identifiers of qualified callees (``a.b()``'s ``a``)
                # remain uses.
                name_node = rightmost_name_node(callee)
                if name_node is not None:
                    excluded.add(_node_key(name_node))
        if t.endswith(_DECLARATION_SUFFIXES):
            shadow_declaration(n)
            continue
        for field in _MEMBER_FIELDS:
            member = n.child_by_field_name(field)
            if member is not None and member.type in _USE_LEAF_NODES:
                excluded.add(_node_key(member))
        stack.extend(_children(n))

    # Pass 2 — emit one occurrence per (owner, name) at its lowest line.
    occurrences: dict[str, ScopeOccurrence] = {}
    stack = list(_children(node))
    while stack:
        n = stack.pop()
        if is_owner_boundary(n):
            continue
        t = n.type
        if t in _USE_LEAF_NODES:
            if _node_key(n) not in excluded:
                name = n.text.decode("utf-8", "replace")
                if (
                    name
                    and name not in shadowed
                    and name not in noise
                    and not name.startswith("<")
                    and name.isidentifier()
                ):
                    line = n.start_point[0] + 1  # 1-based line contract
                    existing = occurrences.get(name)
                    if existing is None or line < existing.line:
                        occurrences[name] = ScopeOccurrence(owner_id=owner.id, name=name, line=line)
            continue
        stack.extend(_children(n))

    for occurrence in sorted(occurrences.values(), key=lambda o: (o.line, o.name)):
        relations.append(reference_relation(occurrence, file_path))
