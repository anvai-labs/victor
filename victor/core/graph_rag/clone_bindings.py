"""Local binding identities for syntax comparison, independent of SCIP globals.

These identities describe lexical declarations within one analyzed callable.
They are never exported as compiler symbols or cross-language identities.
"""

from __future__ import annotations

import ast
import symtable
from typing import Any

from victor.core.graph_rag.language_parsers import IDENTIFIER_NODES

NodeKey = tuple[int, int, str]


def node_key(node: Any) -> NodeKey:
    return node.start_byte, node.end_byte, node.type


def parameter_nodes(function: Any) -> list[Any]:
    """Find declaration leaves, excluding types and callback parameter scopes."""
    parameters = function.child_by_field_name("parameters")
    declarator = function.child_by_field_name("declarator")
    if declarator is not None:
        parameters = declarator.child_by_field_name("parameters")
    if parameters is None:
        parameters = next(
            (
                n
                for n in function.named_children
                if n.type in {"function_value_parameters", "closure_parameters"}
            ),
            None,
        )
    candidates = (
        parameters.named_children
        if parameters is not None
        else [n for n in function.named_children if n.type == "parameter"]
    )
    result = []
    for candidate in candidates:
        if candidate.type in IDENTIFIER_NODES or candidate.type == "self":
            result.append(candidate)
            continue
        selected = [
            child
            for i, child in enumerate(candidate.children)
            if candidate.field_name_for_child(i) in {"pattern", "name", "declarator"}
        ]
        if not selected and candidate.type in {"parameter", "simple_parameter", "self_parameter"}:
            selected = candidate.named_children[:1]
        for pattern in selected:
            result.extend(pattern_names(pattern))
    return result


def pattern_names(pattern: Any) -> list[Any]:
    if pattern.type in {
        "type_identifier",
        "user_type",
        "type_annotation",
        "scoped_identifier",
        "member_expression",
        "field_expression",
    }:
        return []
    if pattern.type in IDENTIFIER_NODES or pattern.type in {"self", "name"}:
        return [pattern]
    if pattern.type == "function_declarator":
        child = pattern.child_by_field_name("declarator")
        return pattern_names(child) if child is not None else []
    if pattern.type in {"field_pattern", "pair_pattern"}:
        value = pattern.child_by_field_name("pattern") or pattern.child_by_field_name("value")
        return pattern_names(value) if value is not None else []
    return [
        leaf
        for i, child in enumerate(pattern.children)
        if child.is_named and pattern.field_name_for_child(i) not in {"condition", "type", "key"}
        for leaf in pattern_names(child)
    ]


def tree_bindings(
    function: Any, data: bytes, language: str
) -> tuple[dict[NodeKey, int], list[Any]]:
    """Resolve parameters, captures and lexical locals for a callable's AST.

    Rust declarations become visible after their initializer. JavaScript var
    declarations are hoisted to the closest function; let/const stay in blocks.
    Other frontends use declaration and block scopes, without claiming complete
    compiler-level handling of every language's binding constructs.
    """
    labels: dict[NodeKey, int] = {}
    next_id = 0
    root_scope: dict[str, int] = {}

    def spelling(node: Any) -> str:
        return data[node.start_byte : node.end_byte].decode("utf-8")

    def declare(node: Any, scope: dict[str, int], reuse: bool = False) -> None:
        nonlocal next_id
        name = spelling(node)
        if not reuse or name not in scope:
            scope[name] = next_id
            next_id += 1
        labels[node_key(node)] = scope[name]

    parameters = parameter_nodes(function)
    for parameter in parameters:
        declare(parameter, root_scope)

    function_types = {
        "closure_expression",
        "lambda_expression",
        "arrow_function",
        "function_expression",
        "function_declaration",
        "function_definition",
        "function_item",
        "method_definition",
        "method_declaration",
    }
    blocks = {"block", "statement_block", "compound_statement"}
    declarations = {
        "let_declaration",
        "variable_declarator",
        "init_declarator",
        "short_var_declaration",
        "assignment",
        "val_definition",
        "var_definition",
        "property_declaration",
    }

    def local_pattern(node: Any) -> Any:
        return (
            node.child_by_field_name("pattern")
            or node.child_by_field_name("name")
            or node.child_by_field_name("left")
            or node.child_by_field_name("declarator")
            or next(
                (
                    c
                    for c in node.named_children
                    if c.type in {"variable_declaration", "variable_list"}
                ),
                None,
            )
        )

    def hoist(node: Any, scope: dict[str, int]) -> None:
        if node.type in function_types:
            name = node.child_by_field_name("name")
            if node.type == "function_declaration" and name is not None:
                declare(name, scope, True)
            return
        if (
            node.type == "variable_declarator"
            and node.parent is not None
            and node.parent.type == "variable_declaration"
        ):
            pattern = local_pattern(node)
            if pattern is not None:
                for leaf in pattern_names(pattern):
                    declare(leaf, scope, True)
        for child in node.named_children:
            hoist(child, scope)

    def walk(
        node: Any,
        scopes: list[dict[str, int]],
        function_scope: dict[str, int],
        function_body: bool = False,
    ) -> None:
        if node_key(node) in labels:
            return
        if language == "rust" and node.type in {"if_expression", "while_expression"}:
            condition = node.child_by_field_name("condition")
            if condition is not None and condition.type == "let_condition":
                value = condition.child_by_field_name("value")
                pattern = condition.child_by_field_name("pattern")
                if value is not None and pattern is not None:
                    walk(value, scopes, function_scope)
                    branch: dict[str, int] = {}
                    for leaf in pattern_names(pattern):
                        if not spelling(leaf)[:1].isupper():
                            declare(leaf, branch, True)
                    consequence = node.child_by_field_name(
                        "consequence"
                    ) or node.child_by_field_name("body")
                    if consequence is not None:
                        walk(consequence, [*scopes, branch], function_scope)
                    alternative = node.child_by_field_name("alternative")
                    if alternative is not None:
                        walk(alternative, scopes, function_scope)
                    return
        if node.type in function_types:
            nested: dict[str, int] = {}
            name = node.child_by_field_name("name")
            if node.type == "function_expression" and name is not None:
                declare(name, nested)
            for parameter in parameter_nodes(node):
                declare(parameter, nested)
            body = node.child_by_field_name("body")
            if body is not None:
                if language in {"javascript", "typescript", "tsx"}:
                    hoist(body, nested)
                walk(body, [*scopes, nested], nested, True)
            return
        if (node.type in blocks and not function_body) or node.type in {
            "match_arm",
            "catch_clause",
        }:
            scopes = [*scopes, {}]
        if node.type == "catch_clause":
            parameter = node.child_by_field_name("parameter")
            if parameter is not None:
                for leaf in pattern_names(parameter):
                    declare(leaf, scopes[-1])
        if node.type == "match_arm":
            pattern = node.child_by_field_name("pattern")
            if pattern is not None:
                for leaf in pattern_names(pattern):
                    # Bare constants/variants require compiler evidence to
                    # distinguish completely. Preserve conventional capitals.
                    if not spelling(leaf)[:1].isupper():
                        declare(leaf, scopes[-1], True)
        if node.type in {"for_expression", "for_in_statement"}:
            pattern = node.child_by_field_name("pattern") or node.child_by_field_name("left")
            value = node.child_by_field_name("value") or node.child_by_field_name("right")
            if pattern is not None:
                if value is not None:
                    walk(value, scopes, function_scope)
                scopes = [*scopes, {}]
                for leaf in pattern_names(pattern):
                    declare(leaf, scopes[-1])
                for child in node.named_children:
                    if child != value:
                        walk(child, scopes, function_scope)
                return
        if node.type in declarations or (
            language == "lua"
            and node.type == "assignment_statement"
            and node.parent is not None
            and node.parent.type == "variable_declaration"
        ):
            pattern = local_pattern(node)
            if pattern is not None:
                # Rust allows `let x = x + 1` to use an outer x on the RHS.
                if language == "rust":
                    for child in node.named_children:
                        if child != pattern:
                            walk(child, scopes, function_scope)
                target = scopes[-1]
                reuse = node.type == "short_var_declaration"
                if language in {"ruby", "php"}:
                    target, reuse = function_scope, True
                if (
                    language in {"javascript", "typescript", "tsx"}
                    and node.parent is not None
                    and node.parent.type == "variable_declaration"
                ):
                    target, reuse = function_scope, True
                for leaf in pattern_names(pattern):
                    declare(leaf, target, reuse)
                if language == "rust":
                    return
        if language == "php" and node.type == "variable_name":
            # PHP function variables have no declaration keyword.
            leaves = pattern_names(node)
            for leaf in leaves:
                declare(leaf, function_scope, True)
        if node.type in IDENTIFIER_NODES or node.type in {"name", "self"}:
            for scope in reversed(scopes):
                if spelling(node) in scope:
                    labels.setdefault(node_key(node), scope[spelling(node)])
                    break
        for child in node.named_children:
            walk(child, scopes, function_scope)

    body = function.child_by_field_name("body") or next(
        (n for n in function.named_children if n.type == "function_body"), None
    )
    if body is not None:
        if language in {"javascript", "typescript", "tsx"}:
            hoist(body, root_scope)
        walk(body, [root_scope], root_scope, True)
    return labels, parameters


def python_bindings(tree: ast.AST, source: str) -> dict[ast.AST, int]:
    """Use Python's compiler symbol tables for locals, free and nonlocal names."""
    table = symtable.symtable(source, "<clone-source>", "exec")
    labels: dict[ast.AST, int] = {}
    ids: dict[tuple[int, str], int] = {}
    used: set[int] = set()
    inline_scopes: list[tuple[int, dict[str, int]]] = []

    def identity(t: Any, name: str) -> int:
        key = (t.get_id(), name)
        if key not in ids:
            ids[key] = len(ids)
        return ids[key]

    def resolve(node: ast.AST, name: str, scopes: list[Any]) -> None:
        try:
            symbol = scopes[-1].lookup(name)
        except KeyError:
            return
        for depth, variables in reversed(inline_scopes):
            if name in variables and (depth == len(scopes) or symbol.is_free()):
                labels[node] = variables[name]
                return
        if symbol.is_local() and scopes[-1].get_type() != "module":
            labels[node] = identity(scopes[-1], name)
        elif symbol.is_free() or symbol.is_nonlocal():
            for outer in reversed(scopes[:-1]):
                if outer.get_type() == "class":
                    continue
                try:
                    candidate = outer.lookup(name)
                except KeyError:
                    continue
                if candidate.is_local() and outer.get_type() != "module":
                    labels[node] = identity(outer, name)
                    break

    def child_table(node: ast.AST, parent: Any, name: str) -> Any:
        candidates = [
            candidate
            for candidate in parent.get_children()
            if (
                candidate.get_id() not in used
                and candidate.get_name() == name
                and candidate.get_lineno() == node.lineno
            )
        ]
        if len(candidates) == 1:
            used.add(candidates[0].get_id())
            return candidates[0]
        # Symbol tables do not expose columns. Two lambdas on one line can
        # have different compiler/evaluation traversal order; never guess.
        raise ValueError(f"Unresolved compiler scope: {name} at {node.lineno}")

    def walk(node: ast.AST, scopes: list[Any]) -> None:
        if isinstance(node, ast.Name):
            resolve(node, node.id, scopes)
        elif isinstance(node, ast.arg):
            resolve(node, node.arg, scopes)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            resolve(node, node.name, scopes)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            name = "lambda" if isinstance(node, ast.Lambda) else node.name
            if not isinstance(node, ast.Lambda):
                resolve(node, name, scopes)
            child = child_table(node, scopes[-1], name)
            if isinstance(node, ast.ClassDef):
                for item in [*node.decorator_list, *node.bases, *node.keywords]:
                    walk(item, scopes)
            else:
                for item in [
                    *node.args.defaults,
                    *[d for d in node.args.kw_defaults if d is not None],
                    *getattr(node, "decorator_list", []),
                ]:
                    walk(item, scopes)
                for arg in [
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                    *([node.args.vararg] if node.args.vararg else []),
                    *([node.args.kwarg] if node.args.kwarg else []),
                ]:
                    resolve(arg, arg.arg, [*scopes, child])
                    if arg.annotation is not None:
                        walk(arg.annotation, scopes)
            body = [node.body] if isinstance(node, ast.Lambda) else node.body
            for item in body:
                walk(item, [*scopes, child])
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            name = {
                ast.ListComp: "listcomp",
                ast.SetComp: "setcomp",
                ast.DictComp: "dictcomp",
                ast.GeneratorExp: "genexpr",
            }[type(node)]
            candidates = [
                t
                for t in scopes[-1].get_children()
                if t.get_name() == name and t.get_lineno() == node.lineno
            ]
            if not candidates:
                # PEP 709: Python 3.12+ inlines list/set/dict comprehensions.
                # Their iteration variables still have an isolated lexical scope.
                variables = {}
                for generator in node.generators:
                    for target in ast.walk(generator.target):
                        if isinstance(target, ast.Name):
                            key = (-id(node), target.id)
                            ids.setdefault(key, len(ids))
                            variables[target.id] = ids[key]
                walk(node.generators[0].iter, scopes)
                inline_scopes.append((len(scopes), variables))
                for i, generator in enumerate(node.generators):
                    walk(generator.target, scopes)
                    if i:
                        walk(generator.iter, scopes)
                    for test in generator.ifs:
                        walk(test, scopes)
                for field in ("elt", "key", "value"):
                    value = getattr(node, field, None)
                    if value is not None:
                        walk(value, scopes)
                inline_scopes.pop()
                return
            child = child_table(node, scopes[-1], name)
            walk(node.generators[0].iter, scopes)
            for i, generator in enumerate(node.generators):
                walk(generator.target, [*scopes, child])
                if i:
                    walk(generator.iter, [*scopes, child])
                for test in generator.ifs:
                    walk(test, [*scopes, child])
            for field in ("elt", "key", "value"):
                value = getattr(node, field, None)
                if value is not None:
                    walk(value, [*scopes, child])
            return
        for item in ast.iter_child_nodes(node):
            walk(item, scopes)

    walk(tree, [table])
    return labels
