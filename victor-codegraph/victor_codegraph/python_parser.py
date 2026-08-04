"""Python parser using the stdlib ``ast`` module.

This is the primary Python path (Victor's approach): it needs no native grammar, is
deterministic, and works fully offline. Extracts modules/classes/functions/methods with
signatures, docstrings, decorators, parameters, cyclomatic complexity, and CALLS edges.
"""

from __future__ import annotations

import ast

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


def _cyclomatic(node: ast.AST) -> dict[str, int]:
    count = 1
    for child in ast.walk(node):
        if isinstance(child, _BRANCH_NODES):
            count += 1
    lineno = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", lineno) or lineno
    return {"cyclomatic": count, "lines": end - lineno + 1}


def _params(args: ast.arguments) -> list[dict]:
    out: list[dict] = []
    posonly = getattr(args, "posonlyargs", [])
    for a in [*posonly, *args.args]:
        if a.arg in ("self", "cls"):
            continue
        p: dict = {"name": a.arg}
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
    parts = []
    for p in _params(args):
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


def _calls_owned_by(node: ast.AST):
    """Yield calls owned by ``node`` without stealing calls from nested definitions."""

    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(child, ast.Call):
            yield child
        stack.extend(ast.iter_child_nodes(child))


class _Visitor:
    def __init__(self, file_path: str, source: str) -> None:
        self.file_path = file_path
        self.source = source
        self.symbols: list[CodeSymbol] = []
        self.relations: list[CodeRelation] = []
        self.imports: list[str] = []
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

    def _make_symbol(
        self,
        node: ast.AST,
        name: str,
        symbol_type: CodeSymbolType,
        scope: list[str],
        signature: str | None = None,
        params: list[dict] | None = None,
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
            parameters=params or [],
            return_type=return_type,
            complexity=_cyclomatic(node),
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
        sym = self._make_symbol(
            node,
            name,
            stype,
            scope,
            signature=_signature(name, node.args, node.returns),
            params=_params(node.args),
            return_type=ast.unparse(node.returns) if node.returns is not None else None,
            modifiers=_modifiers(name, node.decorator_list, isinstance(node, ast.AsyncFunctionDef)),
        )
        self.symbols.append(sym)
        # CALLS edges. ``to_symbol_id`` is the textual callee here; ``parse_python``
        # resolves it to a real in-file symbol id when the callee is defined locally
        # and otherwise keeps it as a bare name (so cross-file/external calls — e.g.
        # a CPG's blast radius — are not silently dropped). ``call_site`` records the
        # call line for consumers that need it.
        for child in _calls_owned_by(node):
            ref = _callee_ref(child)
            if ref:
                self.relations.append(
                    CodeRelation(
                        from_symbol_id=sym.id,
                        to_symbol_id=ref.name,
                        relation_type=CodeRelationType.CALLS,
                        context=ref.text,
                        target_ref=ref,
                        call_site=SourceLocation(
                            file_path=self.file_path,
                            start_line=getattr(child, "lineno", 0),
                            start_column=getattr(child, "col_offset", 0),
                        ),
                    )
                )
        inner = [*scope, name]
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
        return sym

    def visit_class(self, node: ast.ClassDef, scope: list[str]) -> CodeSymbol:
        bases = [ast.unparse(b) for b in node.bases]
        mods = [f"@{ast.unparse(d)}" for d in node.decorator_list]
        if bases:
            mods.append(f"extends({','.join(bases)})")
        cls = self._make_symbol(node, node.name, CodeSymbolType.CLASS, scope, modifiers=mods)
        self.symbols.append(cls)
        for base in bases:
            self.relations.append(
                CodeRelation(
                    from_symbol_id=cls.id,
                    to_symbol_id=base,
                    relation_type=CodeRelationType.EXTENDS,
                    context=base,
                    target_ref=SymbolReference(name=base.split(".")[-1], text=base),
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
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.imports.append(ast.unparse(node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit_function(node, [])
            elif isinstance(node, ast.ClassDef):
                self.visit_class(node, [])


def parse_python(content: str, file_path: str) -> ParsedCode:
    """Parse Python source into symbols + relations using the stdlib ``ast``."""

    tree = ast.parse(content)
    v = _Visitor(file_path, content)
    v.run(tree)
    # Resolve CALLS/EXTENDS targets to real in-file symbol ids when possible
    # (shared with the tree-sitter path — see ``resolution`` for the semantics:
    # unresolved external targets are RETAINED at confidence 0.5, self-edges drop).
    return ParsedCode(
        file_path=file_path,
        language="python",
        symbols=v.symbols,
        relations=resolve_relations(v.symbols, v.relations),
        imports=v.imports,
        content_hash=content_hash(content),
    )
