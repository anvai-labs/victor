"""Restricted boolean conditions shared by workflow handlers and debugging.

Expressions select plain state values using names, literals, comparisons and
boolean/unary operators. Calls, attribute access and subscriptions are rejected.
"""

import ast
import operator
from typing import Any, Callable, Dict


def _validate_data(value: Any) -> None:
    # Reject object hooks and aliased/cyclic containers before comparison can
    # recursively traverse them. A per-operand budget also bounds plain trees.
    remaining = 10000
    seen: set[int] = set()

    def visit(item: Any, depth: int) -> None:
        nonlocal remaining
        remaining -= 1
        if remaining < 0:
            raise ValueError("Condition value is too large")
        if type(item) in (str, int, float, bool, type(None), bytes):
            return
        if depth >= 20 or type(item) not in (list, tuple, dict, set, frozenset):
            raise ValueError("Condition values must be plain data")
        if id(item) in seen or len(item) > 10000:
            raise ValueError("Condition containers must be bounded trees")
        seen.add(id(item))
        for child in item:
            visit(child, depth + 1)
            if type(item) is dict:
                visit(item[child], depth + 1)

    visit(value, 0)


def compile_condition(expression: str) -> Callable[[Dict[str, Any]], bool]:
    """Validate syntax once and return a predicate; invalid state yields False."""
    if len(expression) > 4096:
        raise ValueError("Condition is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("Invalid condition syntax") from exc
    allowed = (
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Compare,
        *_SAFE_COMPARE_OPS,
        *_SAFE_UNARY_OPS,
    )
    nodes = list(ast.walk(tree))
    if len(nodes) > 256 or any(not isinstance(node, allowed) for node in nodes):
        raise ValueError("Conditions support only names, literals and boolean comparisons")

    def condition(state: Dict[str, Any]) -> bool:
        try:
            if type(state) is not dict or len(state) > 10000:
                return False
            if any(type(key) is not str for key in state):
                return False
            return bool(_safe_eval_node(tree.body, state))
        except Exception:
            return False

    return condition


# Restricted operators permitted in condition expressions. No call,
# attribute, subscript, or name binding is allowed by _safe_eval_node, so these
# syntax cannot request calls or attribute access.
_SAFE_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}
_SAFE_UNARY_OPS = {
    ast.Not: operator.not_,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_node(node: ast.AST, names: Dict[str, Any]) -> Any:
    """Evaluate a restricted AST node for condition expressions.

    Only literals, names resolved from ``names``, boolean operators,
    comparisons and unary operators are supported. Any other node type raises
    ``ValueError`` so untrusted expressions cannot reach dangerous constructs.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            value = names[node.id]
            _validate_data(value)
            return value
        raise ValueError(f"Unknown name in condition: {node.id}")
    if isinstance(node, ast.BoolOp):
        values = (_safe_eval_node(v, names) for v in node.values)
        if isinstance(node.op, ast.And):
            return all(values)
        return any(values)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPS:
        return _SAFE_UNARY_OPS[type(node.op)](_safe_eval_node(node.operand, names))
    if isinstance(node, ast.Compare):
        left = _safe_eval_node(node.left, names)
        for op, comparator in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in _SAFE_COMPARE_OPS:
                raise ValueError(f"Unsupported comparison: {op_type.__name__}")
            right = _safe_eval_node(comparator, names)
            if not _SAFE_COMPARE_OPS[op_type](left, right):
                return False
            left = right
        return True
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")
