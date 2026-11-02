"""Portable operation shapes for exploratory cross-language body matching.

The frontends deliberately reject unsupported constructs instead of deleting
them. Numeric widths, overflow, ownership, dispatch and effects are not proven
equivalent by this projection. SCIP identities remain separate compiler evidence;
names from different producers are never treated as shared symbol identities.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
import math
import re
from typing import Any

from victor.core.graph_rag.ast_fingerprints import NormalizedNode as Node, subtree_fingerprints

MIN_PORTABLE_NODES = 8


class UnsupportedOperation(ValueError):
    """A body cannot be represented completely by the current operation schema."""


@dataclass(frozen=True)
class PortableBody:
    fingerprints: tuple[str, ...]
    size: int


def _finish(body: Node, arity: int) -> PortableBody:
    # Parameter declarations anchor their order in the repetition fingerprints.
    # Without them, first-use alpha-renaming would equate x-y with y-x.
    parameters = Node("parameters", tuple(Node(f"v{i}") for i in range(arity)))
    root = Node(f"function:{arity}", (parameters, body))
    pending, size = [root], 0
    while pending:
        current = pending.pop()
        size += 1
        pending.extend(current.children)
    return PortableBody(subtree_fingerprints(root, "portable-operations-v2"), size)


class _Names:
    def __init__(self) -> None:
        self.locals: dict[str, str] = {}

    def bind(self, value: str) -> Node:
        return Node(self.locals.setdefault(value, f"v{len(self.locals)}"))

    def read(self, value: str) -> Node:
        return Node(self.locals.get(value, "global:" + value))


_PY_OPERATORS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.And: "and",
    ast.Or: "or",
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Not: "not",
    ast.USub: "-",
    ast.UAdd: "+",
    ast.Invert: "~",
    ast.In: "in",
    ast.NotIn: "not-in",
    ast.Is: "is",
    ast.IsNot: "is-not",
}


def python_body(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, *, bindings: dict[ast.AST, int] | None = None
) -> PortableBody:
    """Lower a complete supported Python body, preserving parameter positions."""
    names = _Names()
    if (
        isinstance(fn, ast.AsyncFunctionDef)
        or fn.args.vararg
        or fn.args.kwarg
        or fn.args.kwonlyargs
    ):
        raise UnsupportedOperation("async-or-variadic-parameters")
    args = fn.args.posonlyargs + fn.args.args
    if bindings is None:
        from victor.core.graph_rag.clone_bindings import python_bindings

        rendered = ast.unparse(fn)
        parsed = ast.parse(rendered)
        resolved = python_bindings(parsed, rendered)
        bindings = {
            old: resolved[new]
            for old, new in zip(ast.walk(fn), ast.walk(parsed.body[0]))
            if new in resolved
        }
    declared = {
        bindings[n]
        for n in ast.walk(fn)
        if n in bindings
        and (
            isinstance(n, (ast.arg, ast.ExceptHandler))
            or isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Store)
        )
    }
    labels: dict[int, str] = {}
    for arg in args:
        names.bind(arg.arg)
        if arg in bindings:
            labels[bindings[arg]] = f"v{len(labels)}"

    def variable(node: ast.AST, spelling: str) -> Node:
        binding = bindings.get(node)
        if binding is not None and binding in declared:
            return Node(labels.setdefault(binding, f"v{len(labels)}"))
        return Node("global:" + spelling)

    def block(statements: list[ast.stmt]) -> Node:
        return Node("block", tuple(lower(stmt) for stmt in statements))

    def lower(node: ast.AST) -> Node:
        if isinstance(node, ast.Name):
            return variable(node, node.id)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return Node(type(node).__name__.lower(), tuple(lower(item) for item in node.elts))
        if isinstance(node, ast.Dict):
            return Node(
                "dict",
                tuple(
                    Node(
                        "entry" if key is not None else "spread",
                        (lower(key), lower(value)) if key is not None else (lower(value),),
                    )
                    for key, value in zip(node.keys, node.values)
                ),
            )
        if isinstance(node, ast.Slice):
            return Node(
                "slice",
                tuple(
                    lower(value) if value is not None else Node("omitted")
                    for value in (node.lower, node.upper, node.step)
                ),
            )
        if isinstance(node, ast.Starred):
            return Node("spread", (lower(node.value),))
        if isinstance(node, ast.JoinedStr):
            return Node("interpolate", tuple(lower(value) for value in node.values))
        if isinstance(node, ast.FormattedValue):
            return Node(
                "format:" + str(node.conversion),
                (
                    lower(node.value),
                    lower(node.format_spec) if node.format_spec is not None else Node("no-format"),
                ),
            )
        if isinstance(node, ast.IfExp):
            return Node("select", (lower(node.test), lower(node.body), lower(node.orelse)))
        if isinstance(node, ast.Lambda):
            if node.args.vararg or node.args.kwarg or node.args.kwonlyargs or node.args.defaults:
                raise UnsupportedOperation("lambda-parameter-contract")
            parameters = [*node.args.posonlyargs, *node.args.args]
            return Node(
                "lambda",
                (
                    Node("parameters", tuple(variable(arg, arg.arg) for arg in parameters)),
                    Node("return", (lower(node.body),)),
                ),
            )
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            generators = []
            for generator in node.generators:
                iterator = lower(generator.iter)
                generators.append(
                    Node(
                        "async-generator" if generator.is_async else "generator",
                        (
                            lower(generator.target),
                            iterator,
                            *(lower(test) for test in generator.ifs),
                        ),
                    )
                )
            result = (
                (lower(node.key), lower(node.value))
                if isinstance(node, ast.DictComp)
                else (lower(node.elt),)
            )
            return Node(type(node).__name__.lower(), (*generators, *result))
        if isinstance(node, ast.Constant):
            value = node.value
            if type(value) in (int, float):
                if isinstance(value, float) and not math.isfinite(value):
                    raise UnsupportedOperation("nonfinite-literal")
                return Node("number:" + repr(value))
            if value is None or type(value) in (str, bool):
                return Node(f"{type(value).__name__}:{value!r}")
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp)):
            if isinstance(node, ast.Compare):
                if len(node.ops) != 1:
                    raise UnsupportedOperation("chained-comparison")
                operator, operands = node.ops[0], [node.left, node.comparators[0]]
            elif isinstance(node, ast.BoolOp):
                operator, operands = node.op, node.values
            elif isinstance(node, ast.UnaryOp):
                operator, operands = node.op, [node.operand]
            else:
                operator, operands = node.op, [node.left, node.right]
            label = _PY_OPERATORS.get(type(operator))
            if label is not None:
                return Node("operator:" + label, tuple(lower(value) for value in operands))
        if isinstance(node, ast.Attribute):
            return Node("field:" + node.attr, (lower(node.value),))
        if isinstance(node, ast.Subscript):
            return Node("index", (lower(node.value), lower(node.slice)))
        if isinstance(node, ast.Call):
            return Node(
                "call",
                (
                    lower(node.func),
                    *(lower(arg) for arg in node.args),
                    *(
                        Node("keyword:" + (kw.arg or "**"), (lower(kw.value),))
                        for kw in node.keywords
                    ),
                ),
            )
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    raise UnsupportedOperation("multiple-assignment")
                target, value = node.targets[0], node.value
            else:
                target, value = node.target, node.value
            if value is not None:
                # Evaluate the RHS before introducing the declaration's name.
                right = lower(value)
                return Node("assign", (lower(target), right))
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            operator = _PY_OPERATORS.get(type(node.op))
            if operator:
                target = lower(node.target)
                return Node(
                    "assign", (target, Node("operator:" + operator, (target, lower(node.value))))
                )
        if isinstance(node, ast.AugAssign) and type(node.op) in _PY_OPERATORS:
            return Node(
                "augment:" + _PY_OPERATORS[type(node.op)], (lower(node.target), lower(node.value))
            )
        if isinstance(node, ast.Return):
            return Node("return", (lower(node.value),) if node.value else ())
        if isinstance(node, ast.If):
            return Node("if", (lower(node.test), block(node.body), block(node.orelse)))
        if isinstance(node, ast.While) and not node.orelse:
            return Node("while", (lower(node.test), block(node.body)))
        if isinstance(node, ast.For) and not node.orelse:
            iterator = lower(node.iter)
            return Node("foreach", (lower(node.target), iterator, block(node.body)))
        if isinstance(node, ast.Expr):
            return Node("evaluate", (lower(node.value),))
        if isinstance(node, ast.With):
            return Node(
                "with",
                (
                    Node(
                        "contexts",
                        tuple(
                            Node(
                                "context",
                                (
                                    lower(item.context_expr),
                                    (
                                        lower(item.optional_vars)
                                        if item.optional_vars is not None
                                        else Node("no-binding")
                                    ),
                                ),
                            )
                            for item in node.items
                        ),
                    ),
                    block(node.body),
                ),
            )
        if isinstance(node, ast.Try):
            return Node(
                "try",
                (
                    block(node.body),
                    Node(
                        "handlers",
                        tuple(
                            Node(
                                "handler",
                                (
                                    lower(h.type) if h.type is not None else Node("any-exception"),
                                    variable(h, h.name) if h.name else Node("no-binding"),
                                    block(h.body),
                                ),
                            )
                            for h in node.handlers
                        ),
                    ),
                    Node("else", (block(node.orelse),)),
                    Node("finally", (block(node.finalbody),)),
                ),
            )
        if isinstance(node, ast.Raise):
            return Node(
                "raise",
                (
                    lower(node.exc) if node.exc is not None else Node("reraise"),
                    lower(node.cause) if node.cause is not None else Node("implicit-cause"),
                ),
            )
        if isinstance(node, ast.Assert):
            return Node(
                "assert",
                (lower(node.test), lower(node.msg) if node.msg is not None else Node("no-message")),
            )
        if isinstance(node, (ast.Break, ast.Continue, ast.Pass)):
            return Node(type(node).__name__.lower())
        raise UnsupportedOperation(type(node).__name__)

    statements = list(fn.body)
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and isinstance(statements[0].value.value, str)
    ):
        statements = statements[1:]
    return _finish(block(statements), len(args))


def tree_body(fn: Any, source: bytes, language: str) -> PortableBody:
    """Lower supported official grammar nodes to the shared operation schema."""
    names = _Names()
    from victor.core.graph_rag.clone_bindings import node_key, tree_bindings

    scoped, _ = tree_bindings(fn, source, language)

    def read(node: Any) -> Node:
        identity = scoped.get(node_key(node))
        return Node(f"v{identity}") if identity is not None else names.read(text(node))

    def text(node: Any) -> str:
        return source[node.start_byte : node.end_byte].decode("utf-8")

    def children(node: Any) -> list[Any]:
        return [child for child in node.named_children if "comment" not in child.type]

    def field(node: Any, *keys: str) -> Any:
        for key in keys:
            found = node.child_by_field_name(key)
            if found is not None:
                return found
        raise UnsupportedOperation(node.type)

    parameters = fn.child_by_field_name("parameters")
    if parameters is None:
        declarator = fn.child_by_field_name("declarator")
        if declarator is not None:
            parameters = declarator.child_by_field_name("parameters")
    arity = 0
    if parameters is not None:
        for parameter in children(parameters):
            if parameter.type == "identifier":
                bindings = [parameter]
            elif parameter.type == "self_parameter":
                bindings = [child for child in children(parameter) if child.type == "self"]
            else:
                bindings = [
                    child
                    for i, child in enumerate(parameter.children)
                    if parameter.field_name_for_child(i) in {"name", "pattern", "declarator"}
                ]
                for i, binding in enumerate(bindings):
                    while binding.type in {"pointer_declarator", "reference_declarator"}:
                        binding = field(binding, "declarator")
                    bindings[i] = binding
            if not bindings or any(item.type not in {"identifier", "self"} for item in bindings):
                raise UnsupportedOperation("parameter-pattern")
            for binding in bindings:
                names.bind(text(binding))
                arity += 1

    blocks = {"block", "statement_block", "compound_statement", "body_statement", "then", "else"}
    wrappers = {
        "parenthesized_expression",
        "expression_list",
        "argument_list",
        "condition_clause",
        "else_clause",
    }
    declarations = {
        "let_declaration",
        "variable_declarator",
        "init_declarator",
        "short_var_declaration",
        "assignment",
        "assignment_expression",
    }
    containers = {
        "lexical_declaration",
        "variable_declaration",
        "local_variable_declaration",
        "declaration",
    }
    expressions = {
        "identifier",
        "self",
        "binary_expression",
        "binary",
        "unary_expression",
        "unary",
        "call_expression",
        "call",
        "method_invocation",
        "field_expression",
        "member_expression",
        "selector_expression",
        "subscript_expression",
        "index_expression",
        "reference_expression",
        "type_cast_expression",
        "try_expression",
        "closure_expression",
        "match_expression",
        "array_expression",
        "tuple_expression",
        "struct_expression",
        "range_expression",
        "unit_expression",
        "if_expression",
        "block",
    }

    def block(node: Any, implicit_return: bool = False) -> Node:
        statements = children(node)
        if len(statements) == 1 and statements[0].type == "statement_list":
            statements = children(statements[0])
        lowered = []
        for i, statement in enumerate(statements):
            value = lower(statement)
            raw = statement
            if statement.type == "expression_statement" and len(children(statement)) == 1:
                raw = children(statement)[0]
            is_expression = (
                raw.type in expressions
                or raw.type.endswith("_literal")
                or raw.type in {"integer", "float", "number"}
            )
            if (
                implicit_return
                and i == len(statements) - 1
                and is_expression
                and (language == "ruby" or not text(statement).rstrip().endswith(";"))
            ):
                value = Node("return", (lower(raw),))
            elif is_expression and statement.type != "expression_statement":
                value = Node("evaluate", (value,))
            lowered.append(value)
        return Node("block", tuple(lowered))

    def lower(node: Any) -> Node:
        kind = node.type
        parts = children(node)
        if any(child.type == "optional_chain" for child in node.children):
            raise UnsupportedOperation("optional-chain")
        if kind in blocks:
            if (
                language == "rust"
                and node.parent is not None
                and node.parent.type in {"let_declaration", "block", "expression_statement"}
            ):
                return value_block(node)
            return block(node)
        if kind in wrappers and len(parts) == 1:
            return lower(parts[0])
        if kind in {"identifier", "self"}:
            return read(node)
        if kind == "scoped_identifier":
            return Node("global:" + re.sub(r"\s+", "", text(node)))
        if kind == "generic_function":
            return Node(
                "generic",
                (
                    lower(field(node, "function")),
                    Node(
                        "type-arguments:" + re.sub(r"\s+", "", text(field(node, "type_arguments")))
                    ),
                ),
            )
        if kind in {"string_literal", "string", "raw_string_literal"}:
            raw = text(node)
            if kind == "raw_string_literal":
                match = re.fullmatch(r'r(#+)?"([\s\S]*)"\1', raw)
                if match:
                    return Node("str:" + repr(match[2]))
                if raw.startswith('r"') and raw.endswith('"'):
                    return Node("str:" + repr(raw[2:-1]))
            try:
                value = json.loads(raw)
            except (ValueError, TypeError) as exc:
                raise UnsupportedOperation("string-escape-or-interpolation") from exc
            if isinstance(value, str):
                return Node("str:" + repr(value))
            raise UnsupportedOperation(kind)
        if kind in {"char_literal", "character_literal"}:
            return Node("char:" + text(node))
        if kind in {"index_expression", "subscript_expression"}:
            if len(parts) == 2:
                return Node("index", (lower(parts[0]), lower(parts[1])))
        if kind in {"array_expression", "tuple_expression"}:
            if any(child.type in {";", ".."} for child in node.children):
                raise UnsupportedOperation("array-repeat-or-spread")
            return Node(
                "list" if kind == "array_expression" else "tuple", tuple(lower(p) for p in parts)
            )
        if kind == "try_expression":
            return Node("propagate-error", tuple(lower(p) for p in parts))
        if kind == "unit_expression":
            return Node("unit")
        if kind == "empty_statement":
            return Node("pass")
        if kind == "use_declaration":
            return Node("import:" + re.sub(r"\s+", "", text(node)))
        if kind == "unsafe_block":
            return Node(
                "unsafe", tuple(value_block(p) if p.type == "block" else lower(p) for p in parts)
            )
        if kind == "let_condition":
            return Node(
                "pattern-test", (lower_pattern(field(node, "pattern")), lower(field(node, "value")))
            )
        if kind == "range_expression":
            operator = next((child for child in node.children if child.type in {"..", "..="}), None)
            if operator is None:
                raise UnsupportedOperation("range-operator")
            start = next((child for child in parts if child.end_byte <= operator.start_byte), None)
            end = next((child for child in parts if child.start_byte >= operator.end_byte), None)
            return Node(
                (
                    "range:inclusive"
                    if any(c.type == "..=" for c in node.children)
                    else "range:exclusive"
                ),
                (
                    lower(start) if start is not None else Node("omitted"),
                    lower(end) if end is not None else Node("omitted"),
                ),
            )
        if kind == "struct_expression":
            constructor = field(node, "name")
            fields = []
            for entry in children(field(node, "body")):
                if entry.type == "field_initializer":
                    fields.append(
                        Node(
                            "field:" + text(field(entry, "field")), (lower(field(entry, "value")),)
                        )
                    )
                elif entry.type == "shorthand_field_initializer":
                    value = entry.child_by_field_name("name") or next(iter(children(entry)), None)
                    if value is None:
                        raise UnsupportedOperation("shorthand-field")
                    fields.append(Node("field:" + text(value), (lower(value),)))
                elif entry.type == "base_field_initializer":
                    fields.append(Node("spread", tuple(lower(p) for p in children(entry))))
                else:
                    raise UnsupportedOperation(entry.type)
            return Node("construct:" + re.sub(r"\s+", "", text(constructor)), tuple(fields))
        if kind == "reference_expression":
            return Node(
                (
                    "borrow:mut"
                    if any(p.type == "mutable_specifier" for p in parts)
                    else "borrow:shared"
                ),
                (lower(field(node, "value")),),
            )
        if kind == "type_cast_expression":
            return Node(
                "cast:" + re.sub(r"\s+", "", text(field(node, "type"))),
                (lower(field(node, "value")),),
            )
        if kind == "closure_expression":
            from victor.core.graph_rag.clone_bindings import parameter_nodes

            parameters = parameter_nodes(node)
            body = field(node, "body")
            value = (
                block(body, implicit_return=True)
                if body.type == "block"
                else Node("return", (lower(body),))
            )
            closure = Node(
                "lambda", (Node("parameters", tuple(read(p) for p in parameters)), value)
            )
            if any(c.type == "move" for c in node.children):
                closure = Node("capture:move", (closure,))
            if any(c.type == "async" for c in node.children):
                closure = Node("async", (closure,))
            return closure
        if kind == "match_expression":
            arms = []
            for arm in children(field(node, "body")):
                pattern = field(arm, "pattern")
                guard = pattern.child_by_field_name("condition")
                parts_pattern = [p for p in children(pattern) if p != guard]
                if not parts_pattern and text(pattern).strip() == "_":
                    normalized_pattern = Node("pattern:any")
                elif len(parts_pattern) == 1:
                    normalized_pattern = lower_pattern(parts_pattern[0])
                else:
                    raise UnsupportedOperation("match-pattern")
                arms.append(
                    Node(
                        "arm",
                        (
                            normalized_pattern,
                            lower(guard) if guard is not None else Node("no-guard"),
                            lower(field(arm, "value")),
                        ),
                    )
                )
            return Node("match", (lower(field(node, "value")), *arms))
        if kind in {
            "integer_literal",
            "int_literal",
            "decimal_integer_literal",
            "hex_integer_literal",
            "number_literal",
            "number",
            "integer",
            "float_literal",
            "float",
            "decimal_floating_point_literal",
        }:
            value = text(node).replace("_", "")
            value = re.sub(r"(?:[iu](?:8|16|32|64|128|size)|[uUlL]+)$", "", value)
            try:
                number: int | float = (
                    int(value, 0) if value.lower().startswith(("0x", "0b", "0o")) else int(value)
                )
            except ValueError:
                try:
                    number = float(value)
                except ValueError as exc:
                    raise UnsupportedOperation("numeric-literal") from exc
                if not math.isfinite(number):
                    raise UnsupportedOperation("nonfinite-literal")
            return Node("number:" + repr(number))
        if kind in {"true", "false", "boolean_literal"}:
            return Node("bool:" + ("True" if text(node) == "true" else "False"))
        if kind in {"binary_expression", "binary", "unary_expression", "unary"}:
            operator_node = node.child_by_field_name("operator")
            if operator_node is None and kind in {"unary_expression", "unary"}:
                operator_node = next((c for c in node.children if not c.is_named), None)
            if operator_node is None:
                raise UnsupportedOperation("operator")
            operator = text(operator_node)
            operator = {"&&": "and", "||": "or", "!": "not"}.get(operator, operator)
            if kind in {"unary_expression", "unary"}:
                operand = (
                    node.child_by_field_name("argument")
                    or node.child_by_field_name("operand")
                    or (parts[-1] if parts else None)
                )
                if operand is None:
                    raise UnsupportedOperation("unary-operand")
                return Node("operator:" + operator, (lower(operand),))
            return Node(
                "operator:" + operator, (lower(field(node, "left")), lower(field(node, "right")))
            )
        if kind in declarations:
            target = field(node, "pattern", "name", "left", "declarator")
            right = lower(field(node, "value", "right"))
            operator_node = node.child_by_field_name("operator")
            operator = text(operator_node) if operator_node is not None else "="
            if operator != "=":
                if not operator.endswith("=") or operator in {"==", "!=", "<=", ">="}:
                    raise UnsupportedOperation("assignment-operator:" + operator)
                operation = operator[:-1]
                if target.type == "identifier":
                    variable = read(target)
                    return Node(
                        "assign", (variable, Node("operator:" + operation, (variable, right)))
                    )
                # Do not duplicate side-effectful target evaluation.
                return Node("augment:" + operation, (lower(target), right))
            if target.type == "expression_list" and len(children(target)) == 1:
                target = children(target)[0]
            return Node("assign", (lower_target(target), right))
        if kind == "compound_assignment_expr":
            target = field(node, "left")
            value = lower(field(node, "right"))
            operator = text(field(node, "operator")).removesuffix("=")
            if target.type == "identifier":
                variable = read(target)
                return Node("assign", (variable, Node("operator:" + operator, (variable, value))))
            return Node("augment:" + operator, (lower(target), value))
        if kind in containers:
            values = [child for child in parts if child.type in declarations]
            if len(values) == 1:
                return lower(values[0])
        if kind in {"return_statement", "return_expression", "return"}:
            return Node("return", tuple(lower(part) for part in parts))
        if kind in {"if_statement", "if_expression", "if"}:
            alternative = node.child_by_field_name("alternative")
            if language == "rust" and kind == "if_expression" and alternative is not None:
                if alternative.type == "else_clause" and len(children(alternative)) == 1:
                    alternative = children(alternative)[0]
                return Node(
                    "select",
                    (
                        lower(field(node, "condition")),
                        value_block(field(node, "consequence")),
                        (
                            value_block(alternative)
                            if alternative.type == "block"
                            else lower(alternative)
                        ),
                    ),
                )
            return Node(
                "if",
                (
                    lower(field(node, "condition")),
                    lower(field(node, "consequence")),
                    lower(alternative) if alternative is not None else Node("block"),
                ),
            )
        if kind in {"while_statement", "while_expression", "while"}:
            return Node("while", (lower(field(node, "condition")), lower(field(node, "body"))))
        if kind in {"for_expression", "for_in_statement", "enhanced_for_statement"}:
            if kind == "for_in_statement":
                operator = node.child_by_field_name("operator")
                if operator is None or text(operator) != "of":
                    raise UnsupportedOperation("property-key-iteration")
            target = field(node, "pattern", "left", "name")
            iterator = lower(field(node, "value", "right"))
            return Node("foreach", (lower_target(target), iterator, lower(field(node, "body"))))
        if kind in {"field_expression", "member_expression", "selector_expression"}:
            return Node(
                "field:" + text(field(node, "field", "property")),
                (lower(field(node, "value", "object", "operand", "argument")),),
            )
        if kind in {"call_expression", "call", "method_invocation"}:
            if node.child_by_field_name("block") is not None:
                raise UnsupportedOperation("call-block")
            arguments = node.child_by_field_name("arguments")
            if kind == "call_expression":
                function = lower(field(node, "function"))
            else:
                callee = field(node, "method", "name")
                receiver = node.child_by_field_name("receiver") or node.child_by_field_name(
                    "object"
                )
                function = (
                    Node("field:" + text(callee), (lower(receiver),))
                    if receiver is not None
                    else names.read(text(callee))
                )
            return (
                Node("call", (function, *(lower(arg) for arg in children(arguments))))
                if arguments is not None
                else Node("call", (function,))
            )
        if kind == "expression_statement" and len(parts) == 1:
            inner = lower(parts[0])
            if inner.label in {"return", "if", "assign", "while", "foreach"}:
                return inner
            return Node("evaluate", (inner,))
        if kind in {
            "break_statement",
            "continue_statement",
            "break_expression",
            "continue_expression",
        }:
            if parts:
                raise UnsupportedOperation("labeled-control-flow")
            return Node(kind.removesuffix("_statement").removesuffix("_expression"))
        raise UnsupportedOperation(kind)

    def lower_pattern(node: Any) -> Node:
        if node.type == "_":
            return Node("pattern:any")
        if node.type == "identifier":
            return (
                Node("pattern:bind", (read(node),))
                if node_key(node) in scoped
                else Node("pattern:value", (read(node),))
            )
        if node.type == "scoped_identifier":
            return Node("pattern:value", (lower(node),))
        if node.type == "tuple_struct_pattern":
            constructor = field(node, "type")
            return Node(
                "pattern:constructor:" + re.sub(r"\s+", "", text(constructor)),
                tuple(lower_pattern(c) for c in children(node) if c != constructor),
            )
        if node.type == "or_pattern":
            return Node("pattern:or", tuple(lower_pattern(c) for c in children(node)))
        if node.type == "tuple_pattern":
            return Node(
                "pattern:tuple",
                tuple(lower_pattern(c) for c in node.children if c.is_named or c.type == "_"),
            )
        if node.type.endswith("_literal"):
            return Node("pattern:value", (lower(node),))
        raise UnsupportedOperation("pattern:" + node.type)

    def value_block(node: Any) -> Node:
        statements = children(node)
        lowered = []
        for i, statement in enumerate(statements):
            raw = (
                children(statement)[0]
                if statement.type == "expression_statement" and len(children(statement)) == 1
                else statement
            )
            if (
                i == len(statements) - 1
                and not text(statement).rstrip().endswith(";")
                and (raw.type in expressions or raw.type.endswith("_literal"))
            ):
                lowered.append(Node("yield-value", (lower(raw),)))
            else:
                lowered.append(lower(statement))
        return Node("block-value", tuple(lowered))

    def lower_target(node: Any) -> Node:
        if node.type == "_":
            return Node("discard")
        if node.type == "identifier":
            return read(node)
        if node.type in {"tuple_pattern", "tuple_expression"}:
            return Node(
                "tuple",
                tuple(lower_target(c) for c in node.children if c.is_named or c.type == "_"),
            )
        if node.type in {"reference_pattern", "mut_pattern"}:
            return Node(
                node.type
                + (
                    ":mut"
                    if any(c.type == "mutable_specifier" for c in children(node))
                    else ":shared"
                ),
                tuple(lower_target(c) for c in children(node) if c.type != "mutable_specifier"),
            )
        raise UnsupportedOperation("assignment-pattern")

    body = field(fn, "body")
    if body.type not in blocks:
        return _finish(Node("block", (Node("return", (lower(body),)),)), arity)
    return _finish(block(body, implicit_return=language in {"rust", "ruby"}), arity)
