"""Pure-data node-type tables for tree-sitter relation extraction.

Tree-sitter node types are grammar-specific, so a *union* set is safe to apply to
every language: a Java tree simply contains no ``call_expression`` nodes, and a Go
tree no ``method_invocation``. This keeps the walk logic generic while the tables
below document which grammar contributes which node type.

No imports beyond the stdlib; this module is data only.
"""

from __future__ import annotations

# ── Call expressions ─────────────────────────────────────────────────────────
# js/ts/go/rust/c/cpp: call_expression        js/ts/cpp: new_expression
# java: method_invocation, object_creation_expression
# c#: invocation_expression                    ruby: call
CALL_NODES = frozenset(
    {
        "call_expression",
        "new_expression",
        "method_invocation",
        "object_creation_expression",
        "invocation_expression",
        "call",
    }
)

# Fields probed (in order) on a call node to reach its callee node.
# call_expression -> function; new_expression -> constructor;
# method_invocation -> name; object_creation_expression -> type; ruby call -> method
CALLEE_FIELDS = ("function", "constructor", "name", "type", "method")

# Fields probed (in order) on a composite callee node to reach its rightmost
# simple name: member_expression -> property (js/ts); selector_expression /
# field_expression -> field (go/rust/c); scoped_identifier / qualified_identifier
# -> name (rust/java/cpp); generic_function -> function (rust turbofish);
# function_declarator -> declarator (c/cpp definition names).
NAME_FIELDS = ("property", "field", "name", "member", "function", "declarator")

# Leaf node types that ARE a simple name.
IDENTIFIER_NODES = frozenset(
    {
        "identifier",
        "field_identifier",
        "property_identifier",
        "type_identifier",
        "shorthand_property_identifier",
        "constant",  # ruby constants (class names)
        "simple_identifier",  # kotlin
        "word",  # bash
    }
)

# ── Heritage clauses (inheritance / interface implementation) ────────────────
# Clause node type -> relation kind ("extends" | "implements"). Node types are
# grammar-specific: class_heritage (js), extends_clause/implements_clause (ts),
# superclass (java/ruby), extends_interfaces (java interfaces),
# super_interfaces (java classes), base_class_clause (cpp), base_list (c#).
HERITAGE_CLAUSES: dict[str, str] = {
    "class_heritage": "extends",  # js: raw expression; ts: wraps the clauses below
    "extends_clause": "extends",
    "implements_clause": "implements",
    "superclass": "extends",
    "super_interfaces": "implements",
    "extends_interfaces": "extends",
    "base_class_clause": "extends",
    "base_list": "extends",
}

# Subtrees to skip when collecting base names from a heritage clause (generic
# type arguments are not bases: ``class A extends B<C>`` extends B, not C).
HERITAGE_SKIP_NODES = frozenset({"type_arguments", "type_parameters"})
