"""Multi-language body normalization for structural candidates.

This is syntax evidence, refined by compiler identities when available.
No compilation, model inference or new native toolchain is involved.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from typing import Any

from victor.core.graph_rag.structural_clones import FunctionUnit, _digest, _shape
from victor.core.graph_rag.ast_fingerprints import NormalizedNode, subtree_fingerprints
from victor.core.graph_rag.clone_bindings import node_key, tree_bindings
from victor.core.graph_rag.language_parsers import IDENTIFIER_NODES

FUNCTION_NODES = frozenset(
    {
        "function_item",
        "function_declaration",
        "function_definition",
        "function_expression",
        "generator_function_declaration",
        "arrow_function",
        "method_definition",
        "method_declaration",
        "method",
        "singleton_method",
    }
)


def extract_tree_units(
    path: str,
    source: str,
    parser: Any,
    min_nodes: int,
    include_tests: bool,
    language: str,
    occurrences: Any = None,
    with_subtrees: bool = False,
    with_portable: bool = False,
    binding_facts: list[Any] | None = None,
    diagnostics: dict[str, int] | None = None,
    retain_source: bool = False,
) -> list[FunctionUnit]:
    """Normalize callable bodies; SCIP identities refine lexical names when available."""
    data = source.encode("utf-8")
    line_starts = [0, *(i + 1 for i, value in enumerate(data) if value == 10)]

    def position(node: Any) -> tuple[int, int]:
        # tree-sitter 0.26 Point access corrupts non-small row integers on
        # affected wheels. Byte offsets avoid that native accessor entirely.
        offset = node.start_byte
        row = bisect_right(line_starts, offset) - 1
        return row, offset - line_starts[row]

    # Retain the owning Tree while native Node wrappers are visited.
    from victor.core.graph_rag.language_parsers import parser_input

    tree = parser.parse(parser_input(data, language))
    root = tree.root_node
    partial = root.has_error
    if partial:
        if diagnostics is None:
            raise ValueError(f"{language} parse errors: {path}")
        diagnostics["parse_error_files"] = 1
        occurrences = None
    if binding_facts is not None and not partial:
        from victor.core.graph_rag.ffi_bindings import collect_binding_facts

        binding_facts.extend(collect_binding_facts(path, data, root, language, occurrences))

    def text(node: Any) -> str:
        return data[node.start_byte : node.end_byte].decode("utf-8")

    def interpolated(node: Any) -> bool:
        return any(
            "interpolation" in child.type
            or child.type in {"simple_expansion", "expansion", "template_substitution"}
            or interpolated(child)
            for child in node.named_children
        )

    def is_test(node: Any) -> bool:
        previous = node.prev_named_sibling
        while previous is not None and previous.type == "attribute_item":
            if re.search(r"#\[\s*(?:[\w:]*test\b|cfg\s*\(\s*test\b)", text(previous)):
                return True
            previous = previous.prev_named_sibling
        return False

    units = []

    def visit(node: Any, in_test: bool = False, in_function: bool = False) -> None:
        name_node = node.child_by_field_name("name")
        in_test = in_test or (
            language == "rust"
            and (
                is_test(node)
                or (
                    node.type == "mod_item" and name_node is not None and text(name_node) == "tests"
                )
            )
        )
        is_function = node.type in FUNCTION_NODES
        if (
            is_function
            and not in_function
            and not node.has_error
            and (include_tests or not in_test)
        ):
            body = node.child_by_field_name("body") or next(
                (c for c in node.named_children if c.type == "function_body"), None
            )
            name = node.child_by_field_name("name")
            declarator = node.child_by_field_name("declarator")
            if declarator is not None:
                name = declarator.child_by_field_name("declarator")
                while name is not None and name.type not in {
                    "identifier",
                    "field_identifier",
                    "destructor_name",
                    "operator_name",
                    "operator_cast",
                }:
                    name = name.child_by_field_name("declarator") or name.child_by_field_name(
                        "name"
                    )
            if (
                name is None
                and node.parent is not None
                and node.parent.type == "variable_declarator"
            ):
                name = node.parent.child_by_field_name("name")
            if body is not None and name is not None:
                bindings, parameter_bindings = tree_bindings(node, data, language)
                tokens = [f"parameters:{len(parameter_bindings)}"]
                resolved = 0

                def resolve(item: Any) -> str | None:
                    nonlocal resolved
                    if occurrences is None:
                        return None
                    row, column = position(item)
                    hit = occurrences.get((row + 1, column))
                    if hit is not None and hit[0] == item.end_byte - item.start_byte:
                        resolved += 1
                        return hit[1]
                    return None

                def normalize(item: Any) -> NormalizedNode | None:
                    if "comment" in item.type:
                        return None
                    if not item.is_named and text(item) in {
                        "(",
                        ")",
                        "{",
                        "}",
                        "[",
                        "]",
                        ";",
                        ",",
                        ".",
                    }:
                        return None
                    children = []
                    if (
                        item.type.endswith("_literal")
                        or item.type
                        in {
                            "string",
                            "string_content",
                            "string_fragment",
                            "number",
                            "integer",
                            "float",
                            "true",
                            "false",
                            "nil",
                        }
                    ) and not interpolated(item):
                        label = "C:" + item.type
                    elif item.type in (
                        IDENTIFIER_NODES
                        - {"field_identifier", "property_identifier", "type_identifier"}
                    ) or item.type in {"name", "self"}:
                        value = text(item)
                        symbol = resolve(item)
                        binding = bindings.get(node_key(item))
                        label = f"v{binding}" if binding is not None else "G:" + (symbol or value)
                    elif item.type in {"field_identifier", "property_identifier"}:
                        label = "A:" + (resolve(item) or text(item))
                    else:
                        label = item.type if item.child_count else item.type + ":" + text(item)
                        tokens.append(label)
                        for child in item.children:
                            digest = normalize(child)
                            if digest is not None:
                                children.append(digest)
                        return NormalizedNode(label, tuple(children)) if with_subtrees else None
                    tokens.append(label)
                    return NormalizedNode(label) if with_subtrees else None

                normalized = normalize(body)
                if normalized is not None:
                    normalized = NormalizedNode(
                        tokens[0],
                        (
                            NormalizedNode(
                                "parameters",
                                tuple(
                                    NormalizedNode(f"v{i}") for i in range(len(parameter_bindings))
                                ),
                            ),
                            normalized,
                        ),
                    )
                fingerprints = subtree_fingerprints(normalized, language) if normalized else ()
                if len(tokens) >= min_nodes:
                    portable_hashes: tuple[str, ...] = ()
                    portable_size = 0
                    portable_error: str | None = None
                    if with_portable:
                        from victor.core.graph_rag.portable_clones import (
                            tree_body,
                            UnsupportedOperation,
                        )

                        try:
                            portable = tree_body(node, data, language)
                            portable_hashes, portable_size = portable.fingerprints, portable.size
                        except (UnsupportedOperation, RecursionError) as exc:
                            portable_error = str(exc)
                    stub = re.fullmatch(
                        r"\{\s*(?:(?:unimplemented|todo)!\s*\([^;]*\)\s*;?)?\s*\}", text(body), re.S
                    )
                    units.append(
                        FunctionUnit(
                            file_path=path,
                            name=text(name),
                            line=position(name)[0] + 1,
                            size=len(tokens),
                            tokens=tuple(tokens),
                            exact_hash=_digest([language, *tokens]),
                            shape_hash=_digest([language, *_shape(tokens)]),
                            sanctioned="stub" if stub else None,
                            subtree_hashes=fingerprints,
                            language=language,
                            resolved_symbols=resolved,
                            end_line=bisect_right(
                                line_starts, max(node.start_byte, node.end_byte - 1)
                            ),
                            portable_hashes=portable_hashes,
                            portable_size=portable_size,
                            portable_error=portable_error,
                            recovered_from_partial_file=partial,
                            source_code=text(node) if retain_source else "",
                            source_start=node.start_byte,
                        )
                    )
        for child in node.named_children:
            visit(child, in_test, in_function or is_function)

    visit(root)
    return units
