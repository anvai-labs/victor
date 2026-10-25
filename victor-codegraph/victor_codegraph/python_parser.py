"""Python parser using the stdlib ``ast`` module.

This is the primary Python path (Victor's approach): it needs no native grammar, is
deterministic, and works fully offline. Extracts modules/classes/functions/methods with
signatures, docstrings, decorators, parameters, cyclomatic complexity, and CALLS edges.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Sequence
from typing import NotRequired, TypedDict

from .model import (
    CodeRelation,
    CodeRelationType,
    CodeSymbol,
    CodeSymbolType,
    ParsedCode,
    SourceLocation,
    SymbolReference,
    content_hash,
    deterministic_symbol_id,
)
from .resolution import resolve_relations

_BRANCH_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.ExceptHandler,
    ast.BoolOp,
    ast.IfExp,
    ast.comprehension,
)


def _branch_subtree_counts(root: ast.AST) -> dict[int, int]:
    """Branch-node count of every subtree, one post-order pass.

    ``id(node) -> branch nodes in node's subtree`` (all descendants, exactly
    the set ``ast.walk`` yielded). The per-symbol ``_cyclomatic`` used to
    re-walk each symbol's full subtree — nested definitions included, once
    per enclosing symbol — which made symbol extraction O(n * nesting depth)
    on deeply nested files; this precompute makes it an O(1) lookup.
    """
    counts: dict[int, int] = {}
    stack: list[tuple[ast.AST, bool]] = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            total = 1 if isinstance(node, _BRANCH_NODES) else 0
            for child in ast.iter_child_nodes(node):
                total += counts[id(child)]
            counts[id(node)] = total
        else:
            stack.append((node, True))
            for child in ast.iter_child_nodes(node):
                stack.append((child, False))
    return counts


class _Parameter(TypedDict):
    name: str
    type: NotRequired[str]
    is_variadic: NotRequired[bool]
    is_kwonly: NotRequired[bool]


def _params(args: ast.arguments) -> list[_Parameter]:
    out: list[_Parameter] = []
    posonly = getattr(args, "posonlyargs", [])
    for a in [*posonly, *args.args]:
        if a.arg in ("self", "cls"):
            continue
        p: _Parameter = {"name": a.arg}
        if a.annotation is not None:
            p["type"] = ast.unparse(a.annotation)
        out.append(p)
    if args.vararg is not None:
        out.append({"name": f"*{args.vararg.arg}", "is_variadic": True})
    for a in args.kwonlyargs:
        p = {"name": a.arg, "is_kwonly": True}
        if a.annotation is not None:
            p["type"] = ast.unparse(a.annotation)
        out.append(p)
    if args.kwarg is not None:
        out.append({"name": f"**{args.kwarg.arg}", "is_variadic": True})
    return out


def _signature(name: str, args: ast.arguments, returns: ast.AST | None) -> str:
    return _signature_from_params(name, _params(args), returns)


def _signature_from_params(name: str, params: list[_Parameter], returns: ast.AST | None) -> str:
    parts = []
    for p in params:
        s = p["name"]
        if p.get("type"):
            s += f": {p['type']}"
        parts.append(s)
    sig = f"{name}({', '.join(parts)})"
    if returns is not None:
        sig += f" -> {ast.unparse(returns)}"
    return sig


def _modifiers(name: str, decorators: list[ast.expr], is_async: bool) -> list[str]:
    mods = [f"@{ast.unparse(d)}" for d in decorators]
    if is_async:
        mods.append("async")
    if name.startswith("__") and name.endswith("__"):
        mods.append("dunder")
    elif name.startswith("_"):
        mods.append("private")
    return mods


def _callee_ref(call: ast.Call) -> SymbolReference | None:
    f = call.func
    if isinstance(f, ast.Name):
        return SymbolReference(name=f.id, arity=len(call.args) + len(call.keywords), text=f.id)
    if isinstance(f, ast.Attribute):
        text = ast.unparse(f)
        qualifier = text.rsplit(".", 1)[0] if "." in text else None
        return SymbolReference(
            name=f.attr,
            qualifier=qualifier,
            arity=len(call.args) + len(call.keywords),
            text=text,
        )
    return None


def _calls_owned_by(node: ast.AST) -> Iterator[ast.Call]:
    """Yield calls owned by ``node`` without stealing calls from nested definitions."""

    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(child, ast.Call):
            yield child
        stack.extend(ast.iter_child_nodes(child))


def _lexical_bindings(nodes: Sequence[ast.AST]) -> tuple[set[str], set[str]]:
    """Collect writes and definition bindings without entering nested namespaces."""
    names: set[str] = set()
    definitions: set[str] = set()
    stack = list(nodes)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.add(node.name)
            continue
        if isinstance(node, ast.arg):
            names.add(node.arg)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        if isinstance(node, (ast.MatchAs, ast.MatchStar, ast.ExceptHandler)) and node.name:
            names.add(node.name)
        if isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        stack.extend(ast.iter_child_nodes(node))
    return names, definitions


class _Visitor:
    def __init__(self, file_path: str, source: str, *, emit_references: bool = False) -> None:
        # One-pass branch counts (see _branch_subtree_counts); filled by run().
        self._branch_counts: dict[int, int] = {}
        self.file_path = file_path
        self.source = source
        self.emit_references = emit_references
        self.symbols: list[CodeSymbol] = []
        self.relations: list[CodeRelation] = []
        self.imports: list[str] = []
        self._enclosing_bindings: list[set[str]] = []
        self._definition_scopes: list[dict[str, list[str]]] = [{}]
        self._import_names: set[str] = set()
        self._wildcard_import = False
        self._fqn_prefix = file_path.replace("/", ".").replace("\\", ".")
        self._source_bytes = source.encode("utf-8")
        self._line_byte_starts: list[int] = [0]
        for line in source.splitlines(keepends=True):
            self._line_byte_starts.append(self._line_byte_starts[-1] + len(line.encode("utf-8")))

    def _span(self, node: ast.AST) -> tuple[int, int, int, int, int, int]:
        first = node
        decorators = getattr(node, "decorator_list", None) or []
        if decorators:
            first = min(decorators, key=lambda d: (d.lineno, d.col_offset))
        start_line = getattr(first, "lineno", 1)
        start_col = getattr(first, "col_offset", 0)
        if decorators:
            # AST decorator nodes begin after the '@'; the semantic symbol span does not.
            start_col = max(0, start_col - 1)
        end_line = getattr(node, "end_lineno", start_line) or start_line
        end_col = getattr(node, "end_col_offset", 0) or 0
        start = self._line_byte_starts[start_line - 1] + start_col
        end = self._line_byte_starts[end_line - 1] + end_col
        return start_line, start_col, end_line, end_col, start, max(0, end - start)

    def _src(self, node: ast.AST) -> str:
        try:
            *_coords, start, length = self._span(node)
            return self._source_bytes[start : start + length].decode("utf-8")
        except Exception:
            return ""

    def _complexity(self, node: ast.AST) -> dict[str, int]:
        lineno = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", lineno) or lineno
        return {
            "cyclomatic": 1 + self._branch_counts.get(id(node), 0),
            "lines": end - lineno + 1,
        }

    def _make_symbol(
        self,
        node: ast.AST,
        name: str,
        symbol_type: CodeSymbolType,
        scope: list[str],
        signature: str | None = None,
        params: list[_Parameter] | None = None,
        return_type: str | None = None,
        modifiers: list[str] | None = None,
    ) -> CodeSymbol:
        lineno, col, end, end_col, byte_offset, byte_length = self._span(node)
        fqn = "::".join([self._fqn_prefix, *scope, name])
        return CodeSymbol(
            id=deterministic_symbol_id(self.file_path, name, lineno, col),
            symbol_type=symbol_type,
            fully_qualified_name=fqn,
            simple_name=name,
            location=SourceLocation(
                file_path=self.file_path,
                start_line=lineno,
                start_column=col,
                end_line=end,
                end_column=end_col,
                byte_offset=byte_offset,
                byte_length=byte_length,
            ),
            source_code=self._src(node),
            language="python",
            documentation=(
                ast.get_docstring(node)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                else None
            ),
            signature=signature,
            modifiers=modifiers or [],
            scope_chain=list(scope),
            parameters=[dict(parameter) for parameter in params or []],
            return_type=return_type,
            complexity=self._complexity(node),
        )

    def visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        scope: list[str],
        *,
        in_class: bool = False,
    ) -> CodeSymbol:
        name = node.name
        if in_class:
            stype = CodeSymbolType.CONSTRUCTOR if name == "__init__" else CodeSymbolType.METHOD
        else:
            stype = CodeSymbolType.FUNCTION
        params = _params(node.args)
        sym = self._make_symbol(
            node,
            name,
            stype,
            scope,
            signature=_signature_from_params(name, params, node.returns),
            params=params,
            return_type=ast.unparse(node.returns) if node.returns is not None else None,
            modifiers=_modifiers(name, node.decorator_list, isinstance(node, ast.AsyncFunctionDef)),
        )
        # Bindings in a function apply throughout its lexical scope, even before
        # the assignment. Record shadowing so neither resolver guesses a global.
        bound = {a.arg for a in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]}
        if node.args.vararg:
            bound.add(node.args.vararg.arg)
        if node.args.kwarg:
            bound.add(node.args.kwarg.arg)
        writes, definitions = _lexical_bindings(node.body)
        inner = [*scope, name]
        definition_scopes = {
            **self._definition_scopes[-1],
            **{definition: inner for definition in definitions},
        }
        enclosing = set().union(*self._enclosing_bindings)
        sym.metadata["python_bound_names"] = sorted(bound | writes | enclosing)
        sym.metadata["python_rebound_names"] = sorted(writes | enclosing)
        sym.metadata["python_definition_scopes"] = definition_scopes
        sym.metadata["python_import_names"] = sorted(self._import_names)
        sym.metadata["python_wildcard_import"] = self._wildcard_import
        sym.metadata["python_receiver"] = (
            node.args.args[0].arg
            if in_class
            and node.args.args
            and not any(
                ast.unparse(d).split(".")[-1] == "staticmethod" for d in node.decorator_list
            )
            else None
        )
        self.symbols.append(sym)
        # CALLS edges. ``to_symbol_id`` is the textual callee here; ``parse_python``
        # resolves it to a real in-file symbol id when the callee is defined locally
        # and otherwise keeps it as a bare name (so cross-file/external calls — e.g.
        # a CPG's blast radius — are not silently dropped). ``call_site`` records the
        # call line for consumers that need it.
        owned_calls = list(_calls_owned_by(node))
        for call in owned_calls:
            ref = _callee_ref(call)
            if ref:
                line, col, end_line, end_col, offset, length = self._span(call)
                self.relations.append(
                    CodeRelation(
                        from_symbol_id=sym.id,
                        to_symbol_id=ref.name,
                        relation_type=CodeRelationType.CALLS,
                        context=ref.text,
                        target_ref=ref,
                        call_site=SourceLocation(
                            file_path=self.file_path,
                            start_line=line,
                            start_column=col,
                            end_line=end_line,
                            end_column=end_col,
                            byte_offset=offset,
                            byte_length=length,
                        ),
                    )
                )
        self._enclosing_bindings.append(bound | writes)
        self._definition_scopes.append(definition_scopes)
        if self.emit_references:
            from .python_references import collect_python_references

            # Callee positions belong to the CALLS collector; the func node of
            # each owned call is excluded so REFERENCES never duplicates a call.
            collect_python_references(
                node,
                sym,
                self.file_path,
                self.relations,
                excluded={id(call.func) for call in owned_calls},
            )
        for child in node.body:
            nested: CodeSymbol | None = None
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nested = self.visit_function(child, inner, in_class=False)
            elif isinstance(child, ast.ClassDef):
                nested = self.visit_class(child, inner)
            if nested is not None:
                self.relations.append(
                    CodeRelation(
                        from_symbol_id=sym.id,
                        to_symbol_id=nested.id,
                        relation_type=CodeRelationType.CONTAINS,
                    )
                )
        self._enclosing_bindings.pop()
        self._definition_scopes.pop()
        return sym

    def visit_class(self, node: ast.ClassDef, scope: list[str]) -> CodeSymbol:
        bases = [ast.unparse(b) for b in node.bases]
        mods = [f"@{ast.unparse(d)}" for d in node.decorator_list]
        if bases:
            mods.append(f"extends({','.join(bases)})")
        cls = self._make_symbol(node, node.name, CodeSymbolType.CLASS, scope, modifiers=mods)
        class_writes, _ = _lexical_bindings(node.body)
        enclosing = set().union(*self._enclosing_bindings)
        cls.metadata["python_definition_scopes"] = self._definition_scopes[-1]
        cls.metadata["python_rebound_names"] = sorted(enclosing)
        # Tier-2.5 scope binding reads these on every owner; without the stamp a
        # class-owned use would falsely in-file resolve (the resolver consults
        # exactly these keys on the Python branch).
        cls.metadata["python_bound_names"] = sorted(enclosing | class_writes)
        cls.metadata["python_import_names"] = sorted(self._import_names)
        self.symbols.append(cls)
        if self.emit_references:
            from .python_references import collect_python_references

            # Simple-name bases belong to the EXTENDS collector.
            collect_python_references(
                node,
                cls,
                self.file_path,
                self.relations,
                excluded={id(b) for b in node.bases if isinstance(b, ast.Name)},
            )
        for base in bases:
            self.relations.append(
                CodeRelation(
                    from_symbol_id=cls.id,
                    to_symbol_id=base,
                    relation_type=CodeRelationType.EXTENDS,
                    context=base,
                    target_ref=SymbolReference(
                        name=base.split(".")[-1],
                        qualifier=base.rsplit(".", 1)[0] if "." in base else None,
                        text=base,
                    ),
                )
            )
        inner = [*scope, node.name]
        for child in node.body:
            nested: CodeSymbol | None = None
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nested = self.visit_function(child, inner, in_class=True)
            elif isinstance(child, ast.ClassDef):
                nested = self.visit_class(child, inner)
            if nested is not None:
                self.relations.append(
                    CodeRelation(
                        from_symbol_id=cls.id,
                        to_symbol_id=nested.id,
                        relation_type=CodeRelationType.CONTAINS,
                    )
                )
        return cls

    def run(self, tree: ast.Module) -> None:
        module_writes, _ = _lexical_bindings(
            [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        )
        self._enclosing_bindings = [module_writes]
        self._definition_scopes = [{}]
        self._import_names = set()
        for declaration in tree.body:
            if isinstance(declaration, (ast.Import, ast.ImportFrom)):
                self._import_names.update(
                    a.asname or a.name.split(".")[0] for a in declaration.names
                )
            elif isinstance(declaration, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self._import_names.discard(declaration.name)
        self._wildcard_import = "*" in self._import_names
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.imports.append(ast.unparse(node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit_function(node, [])
            elif isinstance(node, ast.ClassDef):
                self.visit_class(node, [])


def parse_python(content: str, file_path: str, *, emit_references: bool = False) -> ParsedCode:
    """Parse Python source into symbols + relations using the stdlib ``ast``.

    With ``emit_references=True``, tier-2.5 scope binding also records
    REFERENCES relations for identifier uses owned by each symbol (see
    ``scope_binding`` / ``python_references``); the default preserves the
    pre-tier-2.5 output byte-for-byte.
    """

    tree = ast.parse(content)
    v = _Visitor(file_path, content, emit_references=emit_references)
    v._branch_counts = _branch_subtree_counts(tree)
    v.run(tree)
    # Resolve CALLS/EXTENDS/REFERENCES targets to real in-file symbol ids when
    # possible (shared with the tree-sitter path — see ``resolution`` for the
    # semantics: unresolved external targets are RETAINED at confidence 0.5).
    relations = resolve_relations(v.symbols, v.relations)
    if emit_references:
        from .scope_binding import cap_unresolved_references

        relations = cap_unresolved_references(relations)
    return ParsedCode(
        file_path=file_path,
        language="python",
        symbols=v.symbols,
        relations=relations,
        imports=v.imports,
        content_hash=content_hash(content),
    )
